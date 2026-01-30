import threading
import time
from typing import Dict, List, Callable
from core.event_bus import EventBus


class ScheduledTask:
    """Repräsentiert eine geplante Aufgabe"""

    def __init__(self, name: str, callback: Callable, interval_ms: int, enabled: bool = True):
        self.name = name
        self.callback = callback
        self.interval_ms = interval_ms
        self.enabled = enabled
        self.last_execution = 0


class ThreadManager:
    """Verwaltet alle Threads und geplante Aufgaben"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._tasks: Dict[int, List[ScheduledTask]] = {}  # interval -> tasks
        self._threads: Dict[int, threading.Thread] = {}
        self._running = False
        self._lock = threading.Lock()

    def add_task(self, task: ScheduledTask):
        """Fügt eine neue geplante Aufgabe hinzu"""
        with self._lock:
            interval = task.interval_ms
            if interval not in self._tasks:
                self._tasks[interval] = []
            self._tasks[interval].append(task)

    def start(self):
        """Startet alle Thread-Zyklen"""
        self._running = True
        for interval in self._tasks.keys():
            thread = threading.Thread(target=self._run_cycle, args=(interval,))
            thread.daemon = True
            self._threads[interval] = thread
            thread.start()

    def stop(self):
        """Stoppt alle Threads"""
        self._running = False
        for thread in self._threads.values():
            thread.join(timeout=1.0)

    def _run_cycle(self, interval_ms: int):
        """Führt einen Thread-Zyklus aus"""
        while self._running:
            start_time = time.time()

            for task in self._tasks.get(interval_ms, []):
                if task.enabled:
                    #try:
                        task.callback()
                        task.last_execution = start_time
                    #except Exception as e:
                        #print(f"Error in task {task.name}: {e}")

            # Warte bis zum nächsten Zyklus
            elapsed = (time.time() - start_time) * 1000
            sleep_time = max(0.0, (interval_ms - elapsed) / 1000.0)
            time.sleep(sleep_time)
