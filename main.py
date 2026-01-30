import sys
import time

import helpers
import pyinsim
from assistance.manager import AssistanceManager
from core.connection_test import LfsConnectionTest
from core.event_bus import EventBus
from core.settings_manager import SettingsManager
from core.thread_manager import ThreadManager, ScheduledTask
from lfs.connector import LFSConnector
from lfs.message_sender import MessageSender
from ui.menu_system import MenuSystem
from ui.ui_manager import UIManager
from vehicles.vehicle_manager import VehicleManager


class LFSAssistantApp:
    """Hauptanwendung - orchestriert alle Komponenten"""

    def __init__(self):
        # Core-Komponenten
        self.event_bus = EventBus()
        self.settings = SettingsManager()
        self.thread_manager = ThreadManager(self.event_bus)
        backoff = 1
        while not helpers.is_lfs_running():
            if backoff > 60:
                sys.exit("LFS is not running")
            print(f"LFS is not running, checking again in {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2


        is_connected = False
        test = LfsConnectionTest()
        backoff = 2
        while not is_connected:
            print("Trying to connect to LFS...")
            is_connected = test.run_test()
            if not is_connected:
                print(f"Connection failed, retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
                if backoff > 60:
                    sys.exit("Connection failed")

        # LFS-Kommunikation
        self.lfs_connector = LFSConnector(self.event_bus, self.settings)
        self.message_sender = MessageSender(self.lfs_connector)

        # Fahrzeug-Management
        self.vehicle_manager = VehicleManager(self.event_bus)

        # Assistenzsysteme
        self.assistance_manager = AssistanceManager(self.event_bus, self.settings)

        # UI
        self.ui_manager = UIManager(self.event_bus, self.message_sender, self.settings)
        self.menu_system = MenuSystem(self.ui_manager, self.settings)

        # Event-Handler registrieren
        self._setup_event_handlers()

        # Scheduled Tasks hinzufügen
        self._setup_scheduled_tasks()

    def _setup_event_handlers(self):
        """Registriert globale Event-Handler"""
        self.event_bus.subscribe('lfs_connected', self._on_lfs_connected)

    def _setup_scheduled_tasks(self):
        """Richtet geplante Aufgaben ein"""
        # Assistenzsysteme alle 100ms
        assistance_task = ScheduledTask(
            "assistance_processing",
            self.assistance_manager.process_all_systems,
            self.settings.get('assistance_refresh_rate')
        )
        self.thread_manager.add_task(assistance_task)

        # UI-Updates alle 200ms
        ui_task = ScheduledTask(
            "ui_updates",
            self.ui_manager.update_hud,
            self.settings.get('ui_refresh_rate')
        )
        self.thread_manager.add_task(ui_task)

    def _on_lfs_connected(self, data=None):
        """Wird aufgerufen wenn LFS-Verbindung hergestellt wurde"""
        print("Connected to LFS")


    def start(self):
        """Startet die Anwendung"""
        print("Starting LFS Assistant...")

        try:
            # LFS-Verbindung herstellen
            self.lfs_connector.connect()

            # Thread-Manager starten
            self.thread_manager.start()

            # Hauptschleife
            self._run_main_loop()

        except KeyboardInterrupt:
            print("Shutting down...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.shutdown()

    def _run_main_loop(self):
        """Hauptschleife der Anwendung"""
        try:
            # pyinsim run() ist blocking
            pyinsim.run()
        except Exception as e:
            print(f"Error in main loop: {e}")

    def shutdown(self):
        """Fährt die Anwendung sauber herunter"""
        print("Shutting down LFS Assistant...")
        self.thread_manager.stop()
        if self.lfs_connector.insim:
            # Cleanup LFS connection
            pass


if __name__ == '__main__':
    app = LFSAssistantApp()
    app.start()