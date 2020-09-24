from threading import RLock
import time


class Counter:
    def __init__(self, start=0, step=1):
        self._count = start
        self._start = start
        self._step = step
        self._lock = RLock()

    def count(self):
        """
        Move counter forward by ``step``
        """
        with self._lock:
            self._count += self._step

    def get(self):
        """
        Get the internal number of counter.
        """
        return self._count

    def reset(self):
        """
        Reset the counter.
        """
        with self._lock:
            self._count = self._start

    def __lt__(self, other):
        return self._count < other

    def __gt__(self, other):
        return self._count > other

    def __le__(self, other):
        return self._count <= other

    def __ge__(self, other):
        return self._count >= other

    def __eq__(self, other):
        return self._count == other

    def __repr__(self):
        return "%d" % self._count


class Switch:
    def __init__(self, state: bool = False):
        """
        Args:
            state: Internal state, ``True`` for on, ``False`` for off.
        """
        self._on = state
        self._lock = RLock()

    def flip(self):
        """
        Inverse the internal state.
        """
        with self._lock:
            self._on = not self._on

    def get(self) -> bool:
        """
        Returns:
            state of switch.
        """
        return self._on

    def on(self):
        """
        Set to on.
        """
        with self._lock:
            self._on = True

    def off(self):
        """
        Set to off.
        """
        with self._lock:
            self._on = False


class Trigger(Switch):
    def get(self):
        """
        Get the state of trigger, will also set trigger to off.

        Returns:
            state of trigger.
        """
        on = self._on
        if self._on:
            self._on = False
        return on


class Timer:
    def __init__(self):
        self._last = time.time()

    def begin(self):
        """
        Begin timing.
        """
        self._last = time.time()

    def end(self):
        """
        Returns:
            Curent time difference since last ``begin()``
        """
        return time.time() - self._last
