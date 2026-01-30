from typing import List

import pyinsim
from core.settings_manager import SettingsManager
from ui.ui_manager import UIManager


class MenuSystem:
    """Verwaltet das Menüsystem"""

    def __init__(self, ui_manager: UIManager, settings: SettingsManager):
        self.ui_manager = ui_manager
        self.settings = settings
        self.current_menu = 'none'
        self.menu_stack: List[str] = []
        self.on_track = False
        self.set_language = settings.get('language')
        self.ui_manager.event_bus.subscribe('state_data', self._state_change)
        self.ui_manager.event_bus.subscribe('button_clicked', self._handle_ui_action)
        self.ui_manager.event_bus.subscribe('player_name_changed', self._handle_player_change)
        self.ui_manager.event_bus.subscribe('new_keybinding', self._rebind_key)

    def _rebind_key(self, data):
        setting = data['setting']
        new_key = data['button']
        self.settings.set(setting, new_key)
        print(f"Rebound {setting} to {new_key}")
        if self.current_menu == 'await_key':
            self.open_system_settings()

    def _handle_player_change(self, data):
        control_mode = data.get('control_mode', 0)
        self.settings.set('own_control_mode', control_mode)

    def _state_change(self, data):
        new_on_track = data['on_track']
        if new_on_track != self.on_track:
            if new_on_track and self.current_menu == 'none':
                self.create_open_menu_button()
            else:
                self.close_menu()

    def open_main_menu(self):
        """Öffnet das Hauptmenü"""
        self.current_menu = 'main'
        self._clear_menu_buttons()

        # Hauptmenü-Buttons
        buttons = [
            (21, 0, 80, 20, 5, self.translator.get("Main Menu", self.set_language),
             pyinsim.ISB_LIGHT),
            (22, 0, 85, 20, 5, self.translator.get("Driving", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (23, 0, 90, 20, 5, self.translator.get("Parking", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (24, 0, 95, 20, 5, (self.translator.get("Language", self.set_language) + f": {self.set_language}"),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (25, 0, 100, 20, 5, self.translator.get("System", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (40, 0, 105, 20, 5, "^1" + self.translator.get("Close", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
        ]

        for button_id, x, y, w, h, text, style in buttons:
            self.ui_manager.message_sender.create_button(button_id, x, y, w, h, text, style)

    def open_system_settings(self):
        """Öffnet das System-Einstellungsmenü"""
        self.current_menu = 'system'
        self._clear_menu_buttons()
        handbrake_key = self.settings.get('user_handbrake_key').upper()
        shift_up_key = self.settings.get('user_shift_up_key').upper()
        shift_down_key = self.settings.get('user_shift_down_key').upper()
        clutch_key = self.settings.get('user_clutch_key').upper()
        ignition_key = self.settings.get('user_ignition_key').upper()
        brake_key = self.settings.get('user_brake_key').upper()
        brake_axis = self.settings.get('user_axis_brake')
        vjoy_axis = self.settings.get('vjoy_axis_1')
        # System-Einstellungsmenü-Buttons
        buttons = [
            (21, 0, 80, 25, 5, self.translator.get("System Settings", self.set_language),
             pyinsim.ISB_LIGHT),
            (24, 0, 90, 20, 5, self.translator.get("Handbrake Key", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (25, 0, 95, 20, 5, self.translator.get("Shift Up Key", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (35, 0, 100, 20, 5, self.translator.get("Shift Down Key", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (26, 0, 105, 20, 5, self.translator.get("Clutch Key", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (27, 0, 110, 20, 5, self.translator.get("Ignition Key", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (28, 20, 90, 5, 5, f"{handbrake_key}", pyinsim.ISB_LIGHT),
            (29, 20, 95, 5, 5, f"{shift_up_key}", pyinsim.ISB_LIGHT),
            (30, 20, 105, 5, 5, f"{clutch_key}", pyinsim.ISB_LIGHT),
            (31, 20, 110, 5, 5, f"{ignition_key}", pyinsim.ISB_LIGHT),
            (36, 20, 100, 5, 5, f"{shift_down_key}", pyinsim.ISB_LIGHT),
            (40, 0, 115, 25, 5, "^1" + self.translator.get("Close", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK)

        ]
        if self.settings.get('own_control_mode') == 0 or self.settings.get('own_control_mode') == 1:
            buttons.append((22, 0, 85, 20, 5, self.translator.get("Brake Key", self.set_language),
                            pyinsim.ISB_DARK | pyinsim.ISB_CLICK))
            buttons.append((32, 20, 85, 5, 5, f"{brake_key}", pyinsim.ISB_LIGHT))
        elif self.settings.get('own_control_mode') == 2:
            buttons.append((22, 0, 85, 20, 5, self.translator.get("Brake Axis", self.set_language),
                            pyinsim.ISB_DARK | pyinsim.ISB_CLICK))
            buttons.append((23, 25, 85, 20, 5, self.translator.get("Vjox Axis", self.set_language),
                            pyinsim.ISB_DARK | pyinsim.ISB_CLICK))
            buttons.append((33, 20, 85, 5, 5, f"{brake_axis}", pyinsim.ISB_LIGHT))
            buttons.append((34, 45, 85, 5, 5, f"{vjoy_axis}", pyinsim.ISB_LIGHT))

        for button_id, x, y, w, h, text, style in buttons:
            self.ui_manager.message_sender.create_button(button_id, x, y, w, h, text, style)

    def open_awaiting_key(self, setting):
        """Show user prompt to press a key for binding"""
        self.current_menu = 'await_key'
        self._clear_menu_buttons()
        text = f"^7Key {setting}, currently bound to '{self.settings.get(setting)}'."
        # System-Einstellungsmenü-Buttons
        buttons = [
            (21, 0, 80, 25, 5, self.translator.get("Rebind Key", self.set_language),
             pyinsim.ISB_LIGHT),
            (22, 0, 85, 25, 5, self.translator.get("Press a key to bind...", self.set_language), pyinsim.ISB_LIGHT),
            (23, 0, 90, 50, 5, text,
             pyinsim.ISB_LIGHT),

            (40, 0, 95, 25, 5, "^1" + self.translator.get("Cancel", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK)
        ]

        for button_id, x, y, w, h, text, style in buttons:
            self.ui_manager.message_sender.create_button(button_id, x, y, w, h, text, style)

    def open_driving_menu(self):
        """Öffnet das Fahrer-Menü"""
        self.current_menu = 'driving'
        self._clear_menu_buttons()

        # Fahrer-Menü mit aktuellen Einstellungen
        fcw = "^2" if self.settings.get('forward_collision_warning') else "^1"
        bsw = "^2" if self.settings.get('blind_spot_warning') else "^1"
        ctw = "^2" if self.settings.get('cross_traffic_warning') else "^1"
        distance = self.settings.get('collision_warning_distance')
        distance = "^2Early" if distance == 0 else "^3Medium" if distance == 1 else "^1Late"
        aeb = self.settings.get('automatic_emergency_brake')  # 0 = Off, 1 = Warn, 2 = Warn & Brake
        aeb_text = "^3Warn only" if aeb == 1 else "^2Warn & Brake" if aeb == 2 else "^1Off"
        buttons = [
            (21, 0, 80, 25, 5, self.translator.get("Driving Menu", self.set_language),
             pyinsim.ISB_LIGHT),
            (22, 0, 85, 25, 5, fcw + self.translator.get("Collision Warning", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (23, 0, 90, 25, 5, bsw + self.translator.get("Blind Spot Warn.", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (24, 0, 95, 25, 5, ctw + (self.translator.get("Cross Traffic Warn.", self.set_language)),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (25, 0, 100, 25, 5, (self.translator.get("Autom. Braking", self.set_language)),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (26, 25, 85, 15, 5, distance,
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (27, 25, 100, 15, 5, aeb_text,
             pyinsim.ISB_LIGHT | pyinsim.ISB_CLICK),
            (40, 0, 105, 25, 5, "^1" + self.translator.get("Close", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
        ]

        for button_id, x, y, w, h, text, style in buttons:
            self.ui_manager.message_sender.create_button(button_id, x, y, w, h, text, style)

    def open_parking_menu(self):
        """Öffnet das Parken-Menü"""
        self.current_menu = 'parking'
        self._clear_menu_buttons()
        pdc_mode = self.settings.get('park_distance_control_mode')
        pdc_text = self.translator.get("Visual") if pdc_mode == 1 else self.translator.get(
            "Visual & Audio") if pdc_mode == 2 else "^1Off"
        pdc = "^2" if self.settings.get('park_distance_control') else "^1"
        peb = "^2" if self.settings.get('parking_emergency_brake') else "^1"
        pdc_on = self.settings.get('park_distance_control')

        # Parken-Menü-Buttons
        buttons = [
            (21, 0, 80, 25, 5, self.translator.get("Parking Menu", self.set_language),
             pyinsim.ISB_LIGHT),
            (22, 0, 85, 25, 5, pdc + self.translator.get("Park Distance Control", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (23, 0, 90, 25, 5, peb + self.translator.get("Parking Emer. Brake", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
            (24, 25, 85, 20, 5, pdc_text,
             (pyinsim.ISB_DARK | pyinsim.ISB_CLICK) if pdc_on else pyinsim.ISB_LIGHT),
            (40, 0, 95, 25, 5, "^1" + self.translator.get("Close", self.set_language),
             pyinsim.ISB_DARK | pyinsim.ISB_CLICK),
        ]
        for button_id, x, y, w, h, text, style in buttons:
            self.ui_manager.message_sender.create_button(button_id, x, y, w, h, text, style)

    def _clear_menu_buttons(self):
        """Löscht alle Menü-Buttons"""
        for button_id in range(20, 41):
            self.ui_manager.message_sender.remove_button(button_id)

    def create_open_menu_button(self):
        self.ui_manager.message_sender.create_button(20, 0, 100, 20, 10,
                                                     "Main Menu",
                                                     pyinsim.ISB_DARK | pyinsim.ISB_CLICK),

    def _handle_ui_action(self, data):
        """Verarbeitet UI-Aktionen"""
        btc = data
        button_id = btc.ClickID

        # Menü-Buttons
        if 20 <= button_id <= 40:
            self._handle_menu_click(button_id)

    def _handle_menu_click(self, button_id: int):
        """Verarbeitet Menü-Klicks"""
        if self.current_menu == 'none':
            if button_id == 20:
                self.open_main_menu()

        elif self.current_menu == 'main':
            if button_id == 22:
                self.open_driving_menu()
            elif button_id == 23:
                self.open_parking_menu()
            elif button_id == 24:
                self.change_language()
            elif button_id == 25:
                self.open_system_settings()
            elif button_id == 40:
                self.close_menu()

        elif self.current_menu == 'driving':
            if button_id == 22:  # Toggle FCW
                current = self.settings.get('forward_collision_warning')
                self.settings.set('forward_collision_warning', not current)
                self.open_driving_menu()  # Refresh menu
            elif button_id == 23:  # Toggle BSW
                current = self.settings.get('blind_spot_warning')
                self.settings.set('blind_spot_warning', not current)
                self.open_driving_menu()  # Refresh menu
            elif button_id == 24:  # Toggle CTW
                current = self.settings.get('cross_traffic_warning')
                self.settings.set('cross_traffic_warning', not current)
                self.open_driving_menu()
            elif button_id == 25:  # Toggle AEB
                current = self.settings.get('automatic_emergency_brake')
                new_value = (current + 1) % 3  # 0 = Off, 1 = Warn, 2 = Warn & Brake
                self.settings.set('automatic_emergency_brake', new_value)
                self.open_driving_menu()
            elif button_id == 26:  # Change Collision Warning Distance
                current = self.settings.get('collision_warning_distance')
                new_value = (current + 1) % 3  # 0 = Early, 1 = Normal, 2 = Late
                self.settings.set('collision_warning_distance', new_value)
                self.open_driving_menu()

        elif self.current_menu == 'parking':
            if button_id == 22:
                current = self.settings.get('park_distance_control')
                self.settings.set('park_distance_control', not current)
                if not current:
                    self.settings.set('park_distance_control_mode', 2)
                else:
                    self.settings.set('park_distance_control_mode', 0)
                    self.ui_manager.remove_pdc_display()
                self.open_parking_menu()
            elif button_id == 23:
                current = self.settings.get('parking_emergency_brake')
                self.settings.set('parking_emergency_brake', not current)
                self.open_parking_menu()
            elif button_id == 24:  # Change Park Distance Control Mode
                current = self.settings.get('park_distance_control_mode')
                new_value = (current + 1)
                if new_value > 2:
                    new_value = 1

                self.settings.set('park_distance_control_mode', new_value)
                self.open_parking_menu()
                if new_value == 0:
                    self.ui_manager.remove_pdc_display()

        elif self.current_menu == 'system':
            setting = ''
            if button_id == 22:
                if self.settings.get('own_control_mode') in [0, 1]:
                    setting = 'user_brake_key'
                else:
                    setting = 'user_axis_brake'
                    current_axis = self.settings.get('user_axis_brake')
                    new_axis = (current_axis + 1) % 25
                    self.settings.set('user_axis_brake', new_axis)
                    self.open_system_settings()
            elif button_id == 23:
                setting = 'vjoy_axis_1'
                current_axis = self.settings.get('vjoy_axis_1')
                new_axis = (current_axis + 1) % 25
                self.settings.set('vjoy_axis_1', new_axis)
                self.open_system_settings()
            elif button_id == 24:
                setting = 'user_handbrake_key'
            elif button_id == 25:
                setting = 'user_shift_up_key'
            elif button_id == 26:
                setting = 'user_clutch_key'
            elif button_id == 27:
                setting = 'user_ignition_key'
            elif button_id == 35:
                setting = 'user_shift_down_key'

            if button_id in [24, 25, 26, 27]:
                self.ui_manager.event_bus.emit('await_keybinding', {'setting': setting})
                self.open_awaiting_key(setting)
            if button_id == 22 and self.settings.get('own_control_mode') in [0, 1]:
                self.open_awaiting_key(setting)

        elif self.current_menu == 'await_key':
            if button_id == 40:
                self.keybinder.stop_listening()
                self.ui_manager.event_bus.emit("notification", {'notification': '^1Keybinding cancelled.'})
                self.open_system_settings()
        if self.current_menu not in ['none', 'main']:
            if button_id == 40:
                self.close_menu()
                self.open_main_menu()



    def close_menu(self):
        """Schließt alle Menüs"""
        self.current_menu = 'none'
        self._clear_menu_buttons()
        self.create_open_menu_button()

    def change_language(self):
        """Wechselt die Sprache"""
        available_langs = self.translator.supported_languages
        current_index = available_langs.index(self.set_language)
        new_index = (current_index + 1) % len(available_langs)
        new_language = available_langs[new_index]
        self.settings.set('language', new_language)
        self.set_language = new_language
        self.open_main_menu()
