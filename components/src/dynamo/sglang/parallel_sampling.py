# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel sampling (``n > 1``) fan-out for SGLang disaggregated serving.

SGLang cannot run parallel sampling across a prefill/decode handoff. Its
scheduler expands an ``n > 1`` request by cloning the first sub-request for
the prefix primer and for every sample, so all decode receivers end up on the
same bootstrap room while prefill registered a single sender, and the request
hangs after prefill (ai-dynamo/dynamo#14098, sgl-project/sglang#30723).

Dynamo therefore keeps SGLang blind to ``n`` in PD mode: the prefill and decode
handlers turn an ``n > 1`` request into ``n`` independent ``n=1`` sub-requests,
one bootstrap room each, and the decode handler merges the sub-streams back
into one multi-choice response keyed by choice index.

Rooms must satisfy ``room % dp_size == dp_rank`` (the decode receiver asserts
it), so the frontend's prefill router draws them and carries them as
``bootstrap_info.bootstrap_rooms`` next to the single ``bootstrap_room`` that
older peers understand. A frontend that predates the field only sends one room;
such ``n > 1`` requests are rejected with HTTP 400 instead of hanging.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any, TypeVar

from dynamo.llm import HttpError

logger = logging.getLogger(__name__)

BOOTSTRAP_ROOMS_KEY = "bootstrap_rooms"

T = TypeVar("T")


def requested_parallel_samples(sampling_params: Mapping[str, Any]) -> int:
    """Return the number of samples an SGLang ``sampling_params`` dict asks for.

    Anything that is not a positive integer is left for SGLang to validate and
    counts as a single sample here.
    """
    n = sampling_params.get("n")
    if isinstance(n, bool) or not isinstance(n, int):
        return 1
    return max(n, 1)


def single_sample_params(sampling_params: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of ``sampling_params`` for one fanned-out sub-request."""
    return {**sampling_params, "n": 1}


def choice_request_id(request_id: str | None, choice_index: int) -> str | None:
    """Per-choice SGLang ``rid`` for a fanned-out sub-request.

    SGLang rejects duplicate request ids, so sibling sub-requests cannot share
    the parent's id. ``None`` stays ``None`` and lets SGLang assign one.
    """
    if request_id is None:
        return None
    return f"{request_id}-choice-{choice_index}"


def _unsupported_parallel_sampling_error(num_choices: int, reason: str) -> HttpError:
    return HttpError(
        400,
        f"SGLang disaggregated serving runs n={num_choices} as {num_choices} "
        f"single-sample sub-requests and needs one bootstrap room per choice, "
        f"but {reason}. Upgrade the Dynamo frontend so its prefill router "
        "draws per-choice bootstrap rooms, or send n=1.",
    )


def _validated_rooms(rooms: Any, num_choices: int) -> list[int] | None:
    if not isinstance(rooms, list) or len(rooms) != num_choices:
        return None
    if not all(isinstance(room, int) and not isinstance(room, bool) for room in rooms):
        return None
    if len(set(rooms)) != num_choices:
        return None
    return list(rooms)


def resolve_decode_bootstrap_rooms(
    bootstrap_info: Mapping[str, Any], num_choices: int
) -> list[int]:
    """Rooms the decode worker pairs on, one per choice.

    ``bootstrap_info`` is the frontend-supplied handoff for this request.
    Raises HTTP 400 for ``n > 1`` when it carries no usable per-choice rooms,
    which is what an older frontend produces.
    """
    if num_choices == 1:
        return [bootstrap_info["bootstrap_room"]]
    rooms = _validated_rooms(bootstrap_info.get(BOOTSTRAP_ROOMS_KEY), num_choices)
    if rooms is None:
        raise _unsupported_parallel_sampling_error(
            num_choices, "the request carried no usable per-choice room list"
        )
    return rooms


def resolve_prefill_bootstrap_rooms(
    bootstrap_info: Mapping[str, Any] | None,
    num_choices: int,
    generate_room: Callable[[], int],
) -> list[int]:
    """Rooms the prefill worker registers, one per choice.

    Router-drawn rooms win. Without any router room the worker owns room
    generation (the same fallback the single-sample path uses), so it draws
    one distinct room per choice. A single router room for ``n > 1`` means an
    older frontend and is rejected with HTTP 400.
    """
    router_room = (
        bootstrap_info.get("bootstrap_room")
        if isinstance(bootstrap_info, Mapping)
        else None
    )
    if num_choices == 1:
        return [router_room if router_room is not None else generate_room()]

    if isinstance(bootstrap_info, Mapping):
        rooms = _validated_rooms(bootstrap_info.get(BOOTSTRAP_ROOMS_KEY), num_choices)
        if rooms is not None:
            return rooms
    if router_room is not None:
        raise _unsupported_parallel_sampling_error(
            num_choices, "the frontend supplied a single bootstrap room"
        )

    rooms: list[int] = []
    while len(rooms) < num_choices:
        room = generate_room()
        if room not in rooms:
            rooms.append(room)
    return rooms


def reject_disagg_parallel_sampling(sampling_params: Mapping[str, Any]) -> None:
    """Raise HTTP 400 for ``n > 1`` on a disaggregated path without fan-out.

    The dedicated multimodal prefill/decode workers still hand SGLang a single
    room for the whole request, so parallel sampling would hang there exactly
    as described in ai-dynamo/dynamo#14098.
    """
    num_choices = requested_parallel_samples(sampling_params)
    if num_choices > 1:
        raise HttpError(
            400,
            f"n={num_choices} is not supported by SGLang multimodal "
            "disaggregated serving; send n=1.",
        )


_STREAM_DONE = object()


async def merge_choice_streams(
    streams: Sequence[AsyncIterator[T]],
) -> AsyncIterator[tuple[int, T]]:
    """Interleave per-choice streams as they produce output.

    Every stream is driven concurrently: SGLang submits a request when its
    generator is first iterated, and each fanned-out decode sub-request must be
    in flight for its prefill counterpart to pair with it. Yields
    ``(choice_index, item)`` pairs. A failure in one stream is re-raised after
    the siblings are cancelled, and closing the merged stream closes them all.
    """
    queue: asyncio.Queue[tuple[int, Any, BaseException | None]] = asyncio.Queue()

    async def pump(choice_index: int, stream: AsyncIterator[T]) -> None:
        try:
            async for item in stream:
                queue.put_nowait((choice_index, item, None))
        except Exception as error:  # noqa: BLE001 - forwarded to the consumer
            queue.put_nowait((choice_index, _STREAM_DONE, error))
            return
        queue.put_nowait((choice_index, _STREAM_DONE, None))

    tasks = [
        asyncio.create_task(pump(choice_index, stream))
        for choice_index, stream in enumerate(streams)
    ]
    open_streams = len(tasks)
    try:
        while open_streams:
            choice_index, item, error = await queue.get()
            if item is _STREAM_DONE:
                if error is not None:
                    raise error
                open_streams -= 1
                continue
            yield choice_index, item
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for stream in streams:
            aclose = getattr(stream, "aclose", None)
            if aclose is None:
                continue
            try:
                await aclose()
            except Exception:  # noqa: BLE001 - best-effort sibling cleanup
                logger.debug(
                    "Failed to close a fanned-out choice stream", exc_info=True
                )
