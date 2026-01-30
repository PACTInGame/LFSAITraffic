from typing import Dict, List, Callable, Any
from threading import Lock


class EventBus:
    """Zentrale Event-Verteilung zwischen allen Komponenten"""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = Lock()

    def subscribe(self, event_type: str, callback: Callable):
        """Registriert einen Event-Handler"""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        """Entfernt einen Event-Handler"""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type].remove(callback)

    def emit(self, event_type: str, data: Any = None):
        """Sendet ein Event an alle registrierten Handler"""
        subscribers = []
        with self._lock:
            if event_type in self._subscribers:
                subscribers = self._subscribers[event_type].copy()

        for callback in subscribers:
            #try:
                callback(data)
                #print(f"Event '{event_type}' emitted with data: {data}")
                #print(f"Handler {callback.__name__} executed successfully.")
            #except Exception as e:
                #print(f"Error in event handler: {e}")
