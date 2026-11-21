"""Per-request usage and timers. Concurrent checks must not share these."""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass, field

from factcheck.types import UsageMeter

_CURRENT: contextvars.ContextVar["RequestState | None"] = contextvars.ContextVar("factcheck_request", default=None)


@dataclass
class RequestState:
    meter: UsageMeter
    phases: dict[str, float] = field(default_factory=lambda: {"extract": 0.0, "retrieve": 0.0, "judge": 0.0, "review": 0.0})
    deadline: float | None = None
    disconnect: object | None = None
    draft_text: str = ""
    on_progress: object | None = None
    retrieval_calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def begin_request(meter: UsageMeter) -> tuple[RequestState, contextvars.Token]:
    state = RequestState(meter=meter)
    token = _CURRENT.set(state)
    return state, token


def end_request(token: contextvars.Token) -> None:
    _CURRENT.reset(token)


def current_request() -> RequestState | None:
    return _CURRENT.get()


def active_meter(fallback: UsageMeter | None) -> UsageMeter | None:
    state = _CURRENT.get()
    if state is not None:
        return state.meter
    return fallback


def charge_tokens(fallback: UsageMeter | None, **kwargs) -> None:
    meter = active_meter(fallback)
    if meter is not None:
        meter.add_tokens(**kwargs)


def note_retrieval(count: int = 1) -> None:
    state = _CURRENT.get()
    if state is None or count <= 0:
        return
    with state.lock:
        state.retrieval_calls += count


def retrieval_calls() -> int:
    state = _CURRENT.get()
    if state is None:
        return 0
    with state.lock:
        return state.retrieval_calls


def add_phase(name: str, seconds: float) -> None:
    state = _CURRENT.get()
    if state is None:
        return
    with state.lock:
        state.phases[name] = state.phases.get(name, 0.0) + seconds


def set_phase(name: str, seconds: float) -> None:
    state = _CURRENT.get()
    if state is None:
        return
    with state.lock:
        state.phases[name] = seconds


def bind_context(func, /, *args, **kwargs):
    """Capture the caller's context now. The worker thread on this runtime does not."""
    context = contextvars.copy_context()

    def wrapper():
        return context.run(func, *args, **kwargs)

    return wrapper


def phase_seconds(name: str) -> float:
    state = _CURRENT.get()
    if state is None:
        return 0.0
    with state.lock:
        return float(state.phases.get(name, 0.0))
