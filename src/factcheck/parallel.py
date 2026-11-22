"""Run independent calls together without nesting event loops."""

from __future__ import annotations

import asyncio
import threading

from factcheck.request_ctx import bind_context

# One cap for every Desearch query in this process, including queries from different claims.
SEARCH_SLOTS = threading.Semaphore(8)


def parallel_map(functions: list, limit: int = 8) -> list:
    """asyncio.gather over thread calls. `limit` is the fan-out; SEARCH_SLOTS caps Desearch itself."""
    if not functions:
        return []

    async def _run():
        sem = asyncio.Semaphore(max(1, limit))

        async def _one(fn):
            async with sem:
                return await asyncio.to_thread(bind_context(fn))

        return await asyncio.gather(*(_one(fn) for fn in functions))

    return list(asyncio.run(_run()))
