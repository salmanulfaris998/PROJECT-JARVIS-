#!/usr/bin/env python3
"""
JARVIS Advanced Phone Controller v2.0
Advanced Android device control system with comprehensive features
Supports all Android devices with robust error handling and extensive functionality
"""

import subprocess
import time
import json
import os
import re
import threading
import queue
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum


class DeviceStatus(Enum):
    """Device connection status"""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    UNAUTHORIZED = "unauthorized"
    OFFLINE = "offline"


class CommandResult(Enum):
    """Command execution results"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NO_DEVICE = "no_device"


@dataclass
class DeviceInfo:
    """Device information structure"""
    model: str = "Unknown"
    brand: str = "Unknown"
    android_version: str = "Unknown"
    api_level: str = "Unknown"
    serial: str = "Unknown"
    screen_width: int = 0
    screen_height: int = 0
    screen_density: int = 0
    battery_level: int = 0
    wifi_enabled: bool = False


@dataclass
class CommandLog:
    """Command execution log entry"""
    command: str
    timestamp: str
    result: CommandResult
    execution_time: float
    output: Optional[str] = None
    error: Optional[str] = None


class JARVISPhoneController:
    """Advanced phone controller with comprehensive Android device management"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize JARVIS Phone Controller with advanced features"""
        self.version = "2.0"
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        print(f"🤖 JARVIS Advanced Phone Controller v{self.version} initializing...")
        
        # Core attributes
        self.device_id: Optional[str] = None
        self.device_info: DeviceInfo = DeviceInfo()
        self.device_status: DeviceStatus = DeviceStatus.DISCONNECTED
        self.command_history: List[CommandLog] = []
        self.screenshot_counter: int = 0
        
        # Performance tracking
        self.total_commands: int = 0
        self.successful_commands: int = 0
        self.connection_attempts: int = 0
        
        # Thread safety
        self._command_lock = threading.Lock()
        self._command_queue = queue.Queue()
        
        # Auto-connect on initialization
        self._initialize_connection()
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "timeouts": {
                "command": 15,
                "screenshot": 30,
                "app_launch": 10
            },
            "paths": {
                "screenshots": "./screenshots/",
                "logs": "./logs/",
                "temp": "/sdcard/jarvis_temp/"
            },
            "retry": {
                "max_attempts": 3,
                "delay": 1.0
            },
            "logging": {
                "level": "INFO",
                "file": "jarvis_controller.log"
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                print(f"⚠️ Config load failed, using defaults: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system"""
        log_dir = Path(self.config["paths"]["logs"])
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("JARVIS_Controller")
        logger.setLevel(getattr(logging, self.config["logging"]["level"]))
        
        # File handler
        file_handler = logging.FileHandler(
            log_dir / self.config["logging"]["file"]
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_connection(self) -> None:
        """Initialize device connection with retry mechanism"""
        print("🔌 Initializing device connection...")
        
        for attempt in range(self.config["retry"]["max_attempts"]):
            self.connection_attempts += 1
            if self._connect_device():
                break
            elif attempt < self.config["retry"]["max_attempts"] - 1:
                wait_time = self.config["retry"]["delay"] * (attempt + 1)
                print(f"⏳ Retrying connection in {wait_time}s... (Attempt {attempt + 2})")
                time.sleep(wait_time)
        
        if self.device_status != DeviceStatus.CONNECTED:
            print("❌ Failed to establish connection after all attempts")
            print("💡 Troubleshooting tips:")
            print("   • Enable USB Debugging in Developer Options")
            print("   • Accept computer authorization on phone")
            print("   • Try different USB cable/port")
            print("   • Restart ADB server: adb kill-server && adb start-server")
    
    def _connect_device(self) -> bool:
        """Connect to Android device with comprehensive validation"""
        try:
            # Check ADB availability
            result = subprocess.run(['adb', 'version'], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print("❌ ADB not accessible")
                return False
            
            # Get device list
            result = subprocess.run(['adb', 'devices', '-l'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("❌ Failed to get device list")
                return False
            
            # Parse devices
            lines = result.stdout.strip().split('\n')[1:]
            connected_devices = []
            
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        device_id = parts[0]
                        status = parts[1]
                        
                        if status == "device":
                            connected_devices.append(device_id)
                        elif status == "unauthorized":
                            print(f"⚠️ Device {device_id} unauthorized - please accept on phone")
                            self.device_status = DeviceStatus.UNAUTHORIZED
                        elif status == "offline":
                            print(f"⚠️ Device {device_id} offline")
                            self.device_status = DeviceStatus.OFFLINE
            
            if not connected_devices:
                self.device_status = DeviceStatus.DISCONNECTED
                return False
            
            # Use first available device
            self.device_id = connected_devices[0]
            self.device_status = DeviceStatus.CONNECTED
            
            print(f"✅ Connected to device: {self.device_id}")
            
            # Get comprehensive device information
            self._gather_device_info()
            self._setup_device_environment()
            
            return True
            
        except subprocess.TimeoutExpired:
            print("⏰ ADB connection timeout")
            return False
        except FileNotFoundError:
            print("❌ ADB not found - install Android SDK Platform Tools")
            print("📥 Download: https://developer.android.com/studio/releases/platform-tools")
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            self.logger.error(f"Connection failed: {e}")
            return False
    
    def _gather_device_info(self) -> None:
        """Gather comprehensive device information"""
        if not self._is_connected():
            return
        
        print("📱 Gathering device information...")
        
        # Basic device properties
        properties = {
            'model': 'ro.product.model',
            'brand': 'ro.product.brand',
            'android_version': 'ro.build.version.release',
            'api_level': 'ro.build.version.sdk',
            'serial': 'ro.serialno'
        }
        
        for key, prop in properties.items():
            value = self._get_device_property(prop)
            if value:
                setattr(self.device_info, key, value)
        
        # Screen information
        screen_info = self._get_screen_info()
        if screen_info:
            self.device_info.screen_width = screen_info[0]
            self.device_info.screen_height = screen_info[1]
            self.device_info.screen_density = screen_info[2] if len(screen_info) > 2 else 0
        
        # System status
        self.device_info.battery_level = self._get_battery_level()
        self.device_info.wifi_enabled = self._get_wifi_status()
        
        # Display info
        print(f"📱 Device: {self.device_info.brand} {self.device_info.model}")
        print(f"🤖 Android: {self.device_info.android_version} (API {self.device_info.api_level})")
        print(f"📺 Screen: {self.device_info.screen_width}×{self.device_info.screen_height} ({self.device_info.screen_density} DPI)")
        print(f"🔋 Battery: {self.device_info.battery_level}%")
        print(f"📶 WiFi: {'Enabled' if self.device_info.wifi_enabled else 'Disabled'}")
    
    def _setup_device_environment(self) -> None:
        """Setup device environment and temporary directories"""
        temp_path = self.config["paths"]["temp"]
        self._execute_adb_command(['shell', 'mkdir', '-p', temp_path])
        
        # Ensure screenshots directory exists locally
        screenshot_dir = Path(self.config["paths"]["screenshots"])
        screenshot_dir.mkdir(exist_ok=True)
    
    def _is_connected(self) -> bool:
        """Check if device is properly connected"""
        return (self.device_status == DeviceStatus.CONNECTED and 
                self.device_id is not None)
    
    def _get_device_property(self, property_name: str) -> Optional[str]:
        """Get device system property"""
        result = self._execute_adb_command(['shell', 'getprop', property_name])
        return result.strip() if result else None
    
    def _get_screen_info(self) -> Optional[Tuple[int, int, int]]:
        """Get detailed screen information"""
        size_result = self._execute_adb_command(['shell', 'wm', 'size'])
        density_result = self._execute_adb_command(['shell', 'wm', 'density'])
        
        width = height = density = 0
        
        if size_result and 'Physical size:' in size_result:
            try:
                size_str = size_result.split('Physical size:')[1].strip()
                width, height = map(int, size_str.split('x'))
            except ValueError:
                pass
        
        if density_result and 'Physical density:' in density_result:
            try:
                density_str = density_result.split('Physical density:')[1].strip()
                density = int(density_str)
            except ValueError:
                pass
        
        return (width, height, density) if width and height else None
    
    def _get_battery_level(self) -> int:
        """Get current battery level"""
        result = self._execute_adb_command(['shell', 'dumpsys', 'battery'])
        if result:
            for line in result.split('\n'):
                if 'level:' in line:
                    try:
                        return int(line.split(':')[1].strip())
                    except (IndexError, ValueError):
                        pass
        return 0
    
    def _get_wifi_status(self) -> bool:
        """Check WiFi status"""
        result = self._execute_adb_command(['shell', 'dumpsys', 'wifi'])
        if result:
            return any(phrase in result for phrase in [
                'Wi-Fi is enabled', 'mWifiEnabled: true', 'WiFi is enabled'
            ])
        return False
    
    def _execute_adb_command(self, command: List[str], timeout: Optional[int] = None) -> Optional[str]:
        """Execute ADB command with comprehensive error handling and logging"""
        if not self._is_connected():
            self.logger.warning("Command attempted without device connection")
            return None
        
        if timeout is None:
            timeout = self.config["timeouts"]["command"]
        
        full_command = ['adb', '-s', self.device_id] + command
        start_time = time.time()
        
        with self._command_lock:
            self.total_commands += 1
            
            try:
                result = subprocess.run(
                    full_command, 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout
                )
                
                execution_time = time.time() - start_time
                
                if result.returncode == 0:
                    self.successful_commands += 1
                    self._log_command(command, CommandResult.SUCCESS, execution_time, result.stdout)
                    return result.stdout.strip()
                else:
                    self._log_command(command, CommandResult.FAILED, execution_time, 
                                    result.stdout, result.stderr)
                    self.logger.warning(f"Command failed: {' '.join(command)} - {result.stderr}")
                    return None
                    
            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                self._log_command(command, CommandResult.TIMEOUT, execution_time)
                self.logger.error(f"Command timeout: {' '.join(command)}")
                return None
            except Exception as e:
                execution_time = time.time() - start_time
                self._log_command(command, CommandResult.FAILED, execution_time, error=str(e))
                self.logger.error(f"Command exception: {e}")
                return None
    
    def _log_command(self, command: List[str], result: CommandResult, 
                    execution_time: float, output: Optional[str] = None, 
                    error: Optional[str] = None) -> None:
        """Log command execution details"""
        log_entry = CommandLog(
            command=' '.join(command),
            timestamp=datetime.now().isoformat(),
            result=result,
            execution_time=execution_time,
            output=output,
            error=error
        )
        self.command_history.append(log_entry)
        
        # Keep history manageable
        if len(self.command_history) > 1000:
            self.command_history = self.command_history[-500:]
    
    # === ENHANCED BASIC CONTROLS ===
    
    def press_key(self, keycode: Union[str, int], long_press: bool = False) -> bool:
        """Press any key with optional long press"""
        action = "long press" if long_press else "press"
        print(f"⌨️ JARVIS {action} key: {keycode}")
        
        cmd = ['shell', 'input', 'keyevent']
        if long_press:
            cmd.append('--longpress')
        cmd.append(str(keycode))
        
        success = self._execute_adb_command(cmd) is not None
        if success:
            print(f"✅ Key {action} successful: {keycode}")
        return success
    
    def press_home(self, long_press: bool = False) -> bool:
        """Press home button with optional long press for Google Assistant"""
        return self.press_key('KEYCODE_HOME', long_press)
    
    def press_back(self) -> bool:
        """Press back button"""
        return self.press_key('KEYCODE_BACK')
    
    def press_menu(self) -> bool:
        """Press menu button"""
        return self.press_key('KEYCODE_MENU')
    
    def press_power(self) -> bool:
        """Press power button"""
        return self.press_key('KEYCODE_POWER')
    
    def press_volume_up(self) -> bool:
        """Press volume up"""
        return self.press_key('KEYCODE_VOLUME_UP')
    
    def press_volume_down(self) -> bool:
        """Press volume down"""
        return self.press_key('KEYCODE_VOLUME_DOWN')
    
    def press_recent_apps(self) -> bool:
        """Show recent apps (task switcher)"""
        return self.press_key('KEYCODE_APP_SWITCH')
    
    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, 
             duration: int = 300, steps: Optional[int] = None) -> bool:
        """Enhanced swipe with configurable steps"""
        print(f"👆 JARVIS swiping from ({start_x},{start_y}) to ({end_x},{end_y})")
        
        cmd = ['shell', 'input', 'swipe', str(start_x), str(start_y), 
               str(end_x), str(end_y), str(duration)]
        
        success = self._execute_adb_command(cmd) is not None
        if success:
            print("✅ Swipe completed")
        return success
    
    def swipe_up(self, duration: int = 300) -> bool:
        """Swipe up from bottom"""
        if not self.device_info.screen_height:
            return False
        x = self.device_info.screen_width // 2
        start_y = int(self.device_info.screen_height * 0.8)
        end_y = int(self.device_info.screen_height * 0.2)
        return self.swipe(x, start_y, x, end_y, duration)
    
    def swipe_down(self, duration: int = 300) -> bool:
        """Swipe down from top"""
        if not self.device_info.screen_height:
            return False
        x = self.device_info.screen_width // 2
        start_y = int(self.device_info.screen_height * 0.2)
        end_y = int(self.device_info.screen_height * 0.8)
        return self.swipe(x, start_y, x, end_y, duration)
    
    def swipe_left(self, duration: int = 300) -> bool:
        """Swipe left"""
        if not self.device_info.screen_width:
            return False
        y = self.device_info.screen_height // 2
        start_x = int(self.device_info.screen_width * 0.8)
        end_x = int(self.device_info.screen_width * 0.2)
        return self.swipe(start_x, y, end_x, y, duration)
    
    def swipe_right(self, duration: int = 300) -> bool:
        """Swipe right"""
        if not self.device_info.screen_width:
            return False
        y = self.device_info.screen_height // 2
        start_x = int(self.device_info.screen_width * 0.2)
        end_x = int(self.device_info.screen_width * 0.8)
        return self.swipe(start_x, y, end_x, y, duration)
    
    def tap(self, x: int, y: int, duration: Optional[int] = None) -> bool:
        """Tap screen with optional duration"""
        print(f"👆 JARVIS tapping at ({x}, {y})")
        
        cmd = ['shell', 'input', 'tap', str(x), str(y)]
        if duration:
            cmd.extend(['--duration', str(duration)])
        
        success = self._execute_adb_command(cmd) is not None
        if success:
            print("✅ Tap completed")
        return success
    
    def tap_center(self) -> bool:
        """Tap center of screen"""
        if not (self.device_info.screen_width and self.device_info.screen_height):
            return False
        x = self.device_info.screen_width // 2
        y = self.device_info.screen_height // 2
        return self.tap(x, y)
    
    def long_press(self, x: int, y: int, duration: int = 1000) -> bool:
        """Long press at coordinates"""
        print(f"👆 JARVIS long pressing at ({x}, {y})")
        
        # Use swipe with same start/end coordinates for long press
        success = self.swipe(x, y, x, y, duration)
        if success:
            print("✅ Long press completed")
        return success
    
    def multi_tap(self, coordinates: List[Tuple[int, int]], delay: float = 0.5) -> bool:
        """Tap multiple coordinates in sequence"""
        print(f"👆 JARVIS multi-tap: {len(coordinates)} points")
        
        for i, (x, y) in enumerate(coordinates):
            if not self.tap(x, y):
                print(f"❌ Multi-tap failed at point {i + 1}")
                return False
            if delay > 0 and i < len(coordinates) - 1:
                time.sleep(delay)
        
        print("✅ Multi-tap completed")
        return True
    
    def type_text(self, text: str, clear_field: bool = False) -> bool:
        """Type text with enhanced character support"""
        if clear_field:
            # Clear field first
            self.press_key('KEYCODE_CTRL_A')  # Select all
            self.press_key('KEYCODE_DEL')     # Delete
        
        print(f"⌨️ JARVIS typing: '{text}'")
        
        # Handle special characters
        escaped_text = text.replace(' ', '%s').replace('&', '\\&').replace("'", "\\'")
        
        success = self._execute_adb_command(['shell', 'input', 'text', escaped_text]) is not None
        if success:
            print("✅ Text typed successfully")
        return success
    
    def paste_text(self, text: str) -> bool:
        """Paste text using clipboard"""
        print(f"📋 JARVIS pasting text: '{text}'")
        
        # Set clipboard content
        if self._execute_adb_command(['shell', 'am', 'broadcast', '-a', 'clipper.set', '-e', 'text', text]):
            # Paste from clipboard
            return self.press_key('KEYCODE_CTRL_V')
        return False
    
    # === ADVANCED SCREENSHOT AND SCREEN RECORDING ===
    
    def take_screenshot(self, filename: Optional[str] = None, 
                       quality: int = 100) -> Optional[str]:
        """Take high-quality screenshot with automatic naming"""
        self.screenshot_counter += 1
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"jarvis_screenshot_{timestamp}_{self.screenshot_counter:03d}.png"
        
        # Ensure absolute path
        screenshot_dir = Path(self.config["paths"]["screenshots"])
        screenshot_path = screenshot_dir / filename
        
        phone_path = f"{self.config['paths']['temp']}screenshot_temp.png"
        
        print(f"📸 JARVIS taking screenshot: {filename}")
        
        # Take screenshot with quality setting
        cmd = ['shell', 'screencap', '-p']
        if quality < 100:
            cmd.extend(['-q', str(quality)])
        cmd.append(phone_path)
        
        if self._execute_adb_command(cmd, timeout=self.config["timeouts"]["screenshot"]):
            # Pull to computer
            if self._execute_adb_command(['pull', phone_path, str(screenshot_path)]):
                # Clean up phone
                self._execute_adb_command(['shell', 'rm', phone_path])
                
                # Verify file exists and has content
                if screenshot_path.exists() and screenshot_path.stat().st_size > 1000:
                    print(f"✅ Screenshot saved: {screenshot_path}")
                    self.logger.info(f"Screenshot captured: {screenshot_path}")
                    return str(screenshot_path)
                else:
                    print("❌ Screenshot file invalid")
            else:
                print("❌ Failed to pull screenshot from device")
        else:
            print("❌ Failed to capture screenshot on device")
        
        return None
    
    def start_screen_recording(self, filename: Optional[str] = None, 
                             duration: int = 60, bitrate: str = "8M") -> bool:
        """Start screen recording (Android 4.4+)"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"jarvis_recording_{timestamp}.mp4"
        
        phone_path = f"{self.config['paths']['temp']}{filename}"
        
        print(f"🎥 JARVIS starting screen recording: {filename} ({duration}s)")
        
        cmd = ['shell', 'screenrecord', '--bit-rate', bitrate, 
               '--time-limit', str(duration), phone_path]
        
        # Start recording in background
        success = self._execute_adb_command(cmd) is not None
        if success:
            print(f"✅ Screen recording started (max {duration}s)")
        return success
    
    # === ENHANCED APP MANAGEMENT ===
    
    def launch_app(self, package_name: str, activity: Optional[str] = None, 
                   wait_for_launch: bool = True) -> bool:
        """Launch app with optional activity and launch verification"""
        print(f"🚀 JARVIS launching app: {package_name}")
        
        if activity:
            cmd = ['shell', 'am', 'start', '-n', f"{package_name}/{activity}"]
        else:
            cmd = ['shell', 'monkey', '-p', package_name, '-c', 
                   'android.intent.category.LAUNCHER', '1']
        
        success = self._execute_adb_command(
            cmd, timeout=self.config["timeouts"]["app_launch"]
        ) is not None
        
        if success and wait_for_launch:
            # Verify app launched
            time.sleep(2)
            current_app = self.get_current_app()
            if current_app and package_name in current_app:
                print(f"✅ App launched successfully: {package_name}")
                return True
            else:
                print(f"⚠️ App may not have launched properly: {package_name}")
                return False
        
        return success
    
    def force_stop_app(self, package_name: str) -> bool:
        """Force stop application"""
        print(f"❌ JARVIS force stopping: {package_name}")
        
        success = self._execute_adb_command(['shell', 'am', 'force-stop', package_name]) is not None
        if success:
            print(f"✅ App force stopped: {package_name}")
        return success
    
    def clear_app_data(self, package_name: str) -> bool:
        """Clear app data and cache"""
        print(f"🧹 JARVIS clearing data for: {package_name}")
        
        success = self._execute_adb_command(['shell', 'pm', 'clear', package_name]) is not None
        if success:
            print(f"✅ App data cleared: {package_name}")
        return success
    
    def get_current_app(self) -> Optional[str]:
        """Get currently focused app package name"""
        result = self._execute_adb_command([
            'shell', 'dumpsys', 'window', 'windows'
        ])
        
        if result:
            # Look for current focus
            for line in result.split('\n'):
                if 'mCurrentFocus' in line or 'mFocusedApp' in line:
                    # Extract package name using regex
                    match = re.search(r'([a-zA-Z0-9_.]+)/([a-zA-Z0-9_.]+)', line)
                    if match:
                        return match.group(1)
        
        return None
    
    def get_installed_apps(self, system_apps: bool = False) -> List[Dict[str, str]]:
        """Get comprehensive list of installed apps"""
        print("📱 Getting installed apps...")
        
        cmd = ['shell', 'pm', 'list', 'packages']
        if not system_apps:
            cmd.append('-3')  # Third-party apps only
        cmd.append('-f')  # Include APK path
        
        result = self._execute_adb_command(cmd)
        apps = []
        
        if result:
            for line in result.split('\n'):
                if line.startswith('package:'):
                    parts = line.replace('package:', '').split('=')
                    if len(parts) == 2:
                        apk_path, package_name = parts
                        apps.append({
                            'package': package_name,
                            'apk_path': apk_path
                        })
        
        print(f"📱 Found {len(apps)} installed apps")
        return apps
    
    def get_app_info(self, package_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed app information"""
        result = self._execute_adb_command(['shell', 'dumpsys', 'package', package_name])
        
        if not result:
            return None
        
        app_info = {'package': package_name}
        
        for line in result.split('\n'):
            line = line.strip()
            if 'versionName=' in line:
                app_info['version'] = line.split('versionName=')[1].split()[0]
            elif 'versionCode=' in line:
                app_info['version_code'] = line.split('versionCode=')[1].split()[0]
            elif 'firstInstallTime=' in line:
                app_info['install_time'] = line.split('firstInstallTime=')[1]
            elif 'lastUpdateTime=' in line:
                app_info['update_time'] = line.split('lastUpdateTime=')[1]
        
        return app_info
    
    # === SYSTEM INTENTS AND SHORTCUTS ===
    
    def open_settings(self, setting_action: Optional[str] = None) -> bool:
        """Open settings with optional specific setting"""
        if setting_action:
            print(f"⚙️ JARVIS opening settings: {setting_action}")
            action = f"android.settings.{setting_action}"
        else:
            print("⚙️ JARVIS opening settings")
            action = "android.settings.SETTINGS"
        
        cmd = ['shell', 'am', 'start', '-a', action]
        success = self._execute_adb_command(cmd) is not None
        
        if success:
            print("✅ Settings opened")
        return success
    
    def open_wifi_settings(self) -> bool:
        """Open WiFi settings"""
        return self.open_settings("WIFI_SETTINGS")
    
    def open_bluetooth_settings(self) -> bool:
        """Open Bluetooth settings"""
        return self.open_settings("BLUETOOTH_SETTINGS")
    
    def open_app_settings(self, package_name: str) -> bool:
        """Open specific app settings"""
        print(f"⚙️ JARVIS opening app settings: {package_name}")
        
        cmd = ['shell', 'am', 'start', '-a', 'android.settings.APPLICATION_DETAILS_SETTINGS',
               '-d', f'package:{package_name}']
        
        success = self._execute_adb_command(cmd) is not None
        if success:
            print(f"✅ App settings opened: {package_name}")
        return success
    
    def open_dialer(self, number: Optional[str] = None) -> bool:
        """Open dialer with optional number"""
        if number:
            print(f"📞 JARVIS opening dialer with: {number}")
            cmd = ['shell', 'am', 'start', '-a', 'android.intent.action.DIAL', '-d', f'tel:{number}']
        else:
            print("📞 JARVIS opening dialer")
            cmd = ['shell', 'am', 'start', '-a', 'android.intent.action.DIAL']
        
        success = self._execute_adb_command(cmd) is not None
        if success:
            print("✅ Dialer opened")
        return success
    
    def make_call(self, number: str) -> bool:
        """Initiate phone call (requires CALL_PHONE permission)"""
        print(f"📞 JARVIS making call to: {number}")
        
        cmd = ['shell', 'am', 'start', '-a', 'android.intent.action.CALL', '-d', f'tel:{number}']
        success = self._execute_adb_command(cmd) is not None
        
        if success:
            print("✅ Call initiated")
        else:
            print("⚠️ Call failed - may need CALL_PHONE permission")
        return success
    
    def open_sms(self, number: Optional[str] = None, message: Optional[str] = None) -> bool:
        """Open SMS app with optional recipient and message"""
        print("💬 JARVIS opening SMS")
        
        cmd = ['shell', 'am', 'start', '-a', 'android.intent.action.SENDTO']
        
        if number:
            cmd.extend(['-d', f'sms:{number}'])
        if message:
            cmd.extend(['--es', 'sms_body', message])
        
        success = self._execute_adb_command(cmd) is not None
        if success:
            print("✅ SMS opened")
        return success
    
    def send_broadcast(self, action: str, extras: Optional[Dict[str, str]] = None) -> bool:
        """Send custom broadcast intent"""
        print(f"📡 JARVIS sending broadcast: {action}")
        
        cmd = ['shell', 'am', 'broadcast', '-a', action]
        
        if extras:
            for key, value in extras.items():
                cmd.extend(['--es', key, value])
        
        success = self._execute_adb_command(cmd) is not None
        if success:
            print("✅ Broadcast sent")
        return success
    
    # === FILE MANAGEMENT ===
    
    def push_file(self, local_path: str, remote_path: str) -> bool:
        """Push file from computer to device"""
        print(f"📤 JARVIS pushing file: {local_path} → {remote_path}")
        
        if not os.path.exists(local_path):
            print(f"❌ Local file not found: {local_path}")
            return False
        
        success = self._execute_adb_command(['push', local_path, remote_path]) is not None
        if success:
            print("✅ File pushed successfully")
        return success
    
    def pull_file(self, remote_path: str, local_path: str) -> bool:
        """Pull file from device to computer"""
        print(f"📥 JARVIS pulling file: {remote_path} → {local_path}")
        
        success = self._execute_adb_command(['pull', remote_path, local_path]) is not None
        if success:
            print("✅ File pulled successfully")
        return success
    
    def list_directory(self, path: str = "/sdcard/") -> List[str]:
        """List directory contents on device"""
        result = self._execute_adb_command(['shell', 'ls', '-la', path])
        
        if result:
            files = []
            for line in result.split('\n')[1:]:  # Skip total line
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 8:
                        filename = ' '.join(parts[8:])  # Handle filenames with spaces
                        files.append(filename)
            return files
        
        return []
    
    def create_directory(self, path: str) -> bool:
        """Create directory on device"""
        print(f"📁 JARVIS creating directory: {path}")
        
        success = self._execute_adb_command(['shell', 'mkdir', '-p', path]) is not None
        if success:
            print("✅ Directory created")
        return success
    
    def delete_file(self, path: str) -> bool:
        """Delete file on device"""
        print(f"🗑️ JARVIS deleting file: {path}")
        
        success = self._execute_adb_command(['shell', 'rm', '-f', path]) is not None
        if success:
            print("✅ File deleted")
        return success
    
    # === SYSTEM MONITORING ===
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        print("📊 JARVIS gathering system info...")
        
        info = {
            'device': asdict(self.device_info),
            'memory': self._get_memory_info(),
            'cpu': self._get_cpu_info(),
            'storage': self._get_storage_info(),
            'network': self._get_network_info(),
            'processes': self._get_top_processes()
        }
        
        return info
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information"""
        result = self._execute_adb_command(['shell', 'cat', '/proc/meminfo'])
        
        memory_info = {}
        if result:
            for line in result.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    memory_info[key.strip()] = value.strip()
        
        return memory_info
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information"""
        result = self._execute_adb_command(['shell', 'cat', '/proc/cpuinfo'])
        
        cpu_info = {}
        if result:
            for line in result.split('\n'):
                if ':' in line and line.count(':') == 1:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    if key not in cpu_info:  # Take first occurrence
                        cpu_info[key] = value.strip()
        
        return cpu_info
    
    def _get_storage_info(self) -> Dict[str, Any]:
        """Get storage information"""
        result = self._execute_adb_command(['shell', 'df', '/sdcard'])
        
        storage_info = {}
        if result:
            lines = result.strip().split('\n')
            if len(lines) >= 2:
                headers = lines[0].split()
                values = lines[1].split()
                
                if len(headers) == len(values):
                    storage_info = dict(zip(headers, values))
        
        return storage_info
    
    def _get_network_info(self) -> Dict[str, Any]:
        """Get network information"""
        wifi_result = self._execute_adb_command(['shell', 'dumpsys', 'wifi'])
        
        network_info = {
            'wifi_enabled': self.device_info.wifi_enabled,
            'connected_network': None,
            'ip_address': None
        }
        
        if wifi_result:
            # Extract connected network name
            for line in wifi_result.split('\n'):
                if 'mNetworkInfo' in line and 'CONNECTED' in line:
                    # Extract SSID if available
                    ssid_match = re.search(r'"([^"]*)"', line)
                    if ssid_match:
                        network_info['connected_network'] = ssid_match.group(1)
                    break
        
        # Get IP address
        ip_result = self._execute_adb_command(['shell', 'ip', 'addr', 'show', 'wlan0'])
        if ip_result:
            ip_match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', ip_result)
            if ip_match:
                network_info['ip_address'] = ip_match.group(1)
        
        return network_info
    
    def _get_top_processes(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get top running processes"""
        result = self._execute_adb_command(['shell', 'ps', '-o', 'PID,NAME,CPU'])
        
        processes = []
        if result:
            lines = result.strip().split('\n')
            if len(lines) > 1:
                for line in lines[1:limit+1]:  # Skip header, limit results
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        processes.append({
                            'pid': parts[0],
                            'name': parts[1],
                            'cpu': parts[2] if len(parts) > 2 else 'N/A'
                        })
        
        return processes
    
    def get_battery_info(self) -> Dict[str, Any]:
        """Get detailed battery information"""
        result = self._execute_adb_command(['shell', 'dumpsys', 'battery'])
        
        battery_info = {}
        if result:
            for line in result.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    # Convert numeric values
                    if key in ['level', 'scale', 'voltage', 'temperature']:
                        try:
                            battery_info[key] = int(value)
                        except ValueError:
                            battery_info[key] = value
                    else:
                        battery_info[key] = value
        
        # Calculate percentage if not present
        if 'level' in battery_info and 'scale' in battery_info:
            battery_info['percentage'] = (battery_info['level'] / battery_info['scale']) * 100
        
        return battery_info
    
    # === ADVANCED TESTING FRAMEWORK ===
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive system tests"""
        print("🧪 JARVIS running comprehensive tests...")
        print("=" * 60)
        
        test_results = {
            'connection_tests': self._test_connection(),
            'input_tests': self._test_input_methods(),
            'app_tests': self._test_app_management(),
            'system_tests': self._test_system_functions(),
            'file_tests': self._test_file_operations(),
            'performance_tests': self._test_performance()
        }
        
        # Calculate overall results
        total_tests = sum(len(category) for category in test_results.values())
        passed_tests = sum(
            sum(1 for result in category.values() if result) 
            for category in test_results.values()
        )
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 COMPREHENSIVE TEST RESULTS")
        print("=" * 60)
        print(f"📈 Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        print(f"🔌 Connection Success Rate: {(self.successful_commands / max(1, self.total_commands)) * 100:.1f}%")
        
        for category, tests in test_results.items():
            category_passed = sum(1 for result in tests.values() if result)
            category_total = len(tests)
            category_rate = (category_passed / category_total * 100) if category_total > 0 else 0
            
            print(f"\n{category.replace('_', ' ').title()}:")
            for test_name, result in tests.items():
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  {test_name}: {status}")
            print(f"  Category Success: {category_rate:.1f}% ({category_passed}/{category_total})")
        
        return {
            'results': test_results,
            'overall_success_rate': success_rate,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'performance_metrics': {
                'total_commands': self.total_commands,
                'successful_commands': self.successful_commands,
                'connection_attempts': self.connection_attempts,
                'command_success_rate': (self.successful_commands / max(1, self.total_commands)) * 100
            }
        }
    
    def _test_connection(self) -> Dict[str, bool]:
        """Test connection functionality"""
        return {
            'device_connected': self._is_connected(),
            'device_info_gathered': bool(self.device_info.model != "Unknown"),
            'adb_responsive': self._execute_adb_command(['shell', 'echo', 'test']) == 'test',
            'screen_info_available': bool(self.device_info.screen_width > 0)
        }
    
    def _test_input_methods(self) -> Dict[str, bool]:
        """Test input functionality"""
        tests = {}
        
        # Test basic keys
        tests['home_button'] = self.press_home()
        time.sleep(1)
        
        tests['back_button'] = self.press_back()
        time.sleep(1)
        
        # Test tap
        if self.device_info.screen_width > 0:
            x, y = self.device_info.screen_width // 2, self.device_info.screen_height // 2
            tests['screen_tap'] = self.tap(x, y)
        else:
            tests['screen_tap'] = False
        
        # Test swipe
        if self.device_info.screen_width > 0:
            tests['screen_swipe'] = self.swipe_up()
        else:
            tests['screen_swipe'] = False
        
        return tests
    
    def _test_app_management(self) -> Dict[str, bool]:
        """Test app management functionality"""
        tests = {}
        
        # Test settings app
        tests['open_settings'] = self.open_settings()
        time.sleep(2)
        
        tests['detect_current_app'] = self.get_current_app() is not None
        
        tests['close_app'] = self.press_home()  # Close settings
        
        tests['get_installed_apps'] = len(self.get_installed_apps()) > 0
        
        return tests
    
    def _test_system_functions(self) -> Dict[str, bool]:
        """Test system information functions"""
        tests = {}
        
        battery_info = self.get_battery_info()
        tests['battery_info'] = len(battery_info) > 0 and 'level' in battery_info
        
        system_info = self.get_system_info()
        tests['system_info'] = len(system_info) > 0
        
        tests['screenshot'] = self.take_screenshot() is not None
        
        return tests
    
    def _test_file_operations(self) -> Dict[str, bool]:
        """Test file operations"""
        tests = {}
        
        # Test directory listing
        files = self.list_directory('/sdcard/')
        tests['list_directory'] = len(files) > 0
        
        # Test directory creation
        test_dir = f"{self.config['paths']['temp']}jarvis_test"
        tests['create_directory'] = self.create_directory(test_dir)
        
        # Test file deletion (cleanup)
        if tests['create_directory']:
            tests['delete_directory'] = self.delete_file(test_dir)
        else:
            tests['delete_directory'] = False
        
        return tests
    
    def _test_performance(self) -> Dict[str, bool]:
        """Test performance metrics"""
        start_time = time.time()
        
        # Quick response test
        quick_response = self._execute_adb_command(['shell', 'echo', 'performance_test'])
        response_time = time.time() - start_time
        
        return {
            'quick_response': quick_response == 'performance_test',
            'response_under_1s': response_time < 1.0,
            'command_success_rate_good': (self.successful_commands / max(1, self.total_commands)) > 0.8
        }
    
    # === UTILITY AND MANAGEMENT FUNCTIONS ===
    
    def save_logs(self, filename: Optional[str] = None) -> str:
        """Save command history and logs"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"jarvis_logs_{timestamp}.json"
        
        log_data = {
            'controller_info': {
                'version': self.version,
                'timestamp': datetime.now().isoformat(),
                'device_info': asdict(self.device_info),
                'total_commands': self.total_commands,
                'successful_commands': self.successful_commands,
                'success_rate': (self.successful_commands / max(1, self.total_commands)) * 100
            },
            'command_history': [asdict(cmd) for cmd in self.command_history[-100:]],  # Last 100 commands
            'configuration': self.config
        }
        
        log_path = Path(self.config["paths"]["logs"]) / filename
        
        try:
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            print(f"📄 Logs saved: {log_path}")
            return str(log_path)
        except Exception as e:
            print(f"❌ Failed to save logs: {e}")
            return ""
    
    def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        return {
            'controller': {
                'version': self.version,
                'uptime': datetime.now().isoformat(),
                'connected': self._is_connected(),
                'device_id': self.device_id,
                'device_status': self.device_status.value
            },
            'device': asdict(self.device_info),
            'performance': {
                'total_commands': self.total_commands,
                'successful_commands': self.successful_commands,
                'success_rate': (self.successful_commands / max(1, self.total_commands)) * 100,
                'connection_attempts': self.connection_attempts,
                'recent_commands': len([cmd for cmd in self.command_history 
                                      if (datetime.now() - datetime.fromisoformat(cmd.timestamp)).seconds < 300])
            },
            'capabilities': {
                'screen_control': bool(self.device_info.screen_width > 0),
                'app_management': self._is_connected(),
                'file_operations': self._is_connected(),
                'system_monitoring': self._is_connected(),
                'screenshot': self._is_connected()
            }
        }
    
    def cleanup(self) -> None:
        """Cleanup temporary files and connections"""
        print("🧹 JARVIS cleaning up...")
        
        if self._is_connected():
            # Clean temporary files on device
            temp_path = self.config["paths"]["temp"]
            self._execute_adb_command(['shell', 'rm', '-rf', f"{temp_path}*"])
        
        # Save final logs
        self.save_logs("jarvis_final_session.json")
        
        print("✅ Cleanup completed")
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during destruction


def main():
    """Main function for testing and demonstration"""
    print("🤖 JARVIS Advanced Phone Controller v2.0")
    print("=" * 70)
    
    # Initialize controller
    controller = JARVISPhoneController()
    
    if controller._is_connected():
        print("\n🎯 Running comprehensive test suite...")
        test_results = controller.run_comprehensive_tests()
        
        print("\n📊 Generating status report...")
        status = controller.get_status_report()
        
        print(f"\n📱 Device Summary:")
        print(f"   Model: {status['device']['brand']} {status['device']['model']}")
        print(f"   Android: {status['device']['android_version']} (API {status['device']['api_level']})")
        print(f"   Screen: {status['device']['screen_width']}×{status['device']['screen_height']}")
        print(f"   Battery: {status['device']['battery_level']}%")
        
        print(f"\n⚡ Performance Metrics:")
        print(f"   Commands Executed: {status['performance']['total_commands']}")
        print(f"   Success Rate: {status['performance']['success_rate']:.1f}%")
        print(f"   Connection Attempts: {status['performance']['connection_attempts']}")
        
        # Save logs
        controller.save_logs()
        
        if test_results['results']:
            overall_success = test_results['overall_success_rate']
            if overall_success >= 90:
                print("🎉 EXCELLENT! All systems operational - 10/10 rating achieved!")
            elif overall_success >= 80:
                print("✅ VERY GOOD! Most systems working well")
            elif overall_success >= 60:
                print("⚠️ FAIR! Some issues detected")
            else:
                print("❌ POOR! Multiple system failures detected")
        
        print("\n🏁 JARVIS Advanced Controller test completed!")
        
    else:
        print("❌ Unable to connect to device")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Enable Developer Options on your Android device")
        print("   2. Enable USB Debugging in Developer Options")
        print("   3. Connect device via USB cable")
        print("   4. Accept 'Allow USB Debugging' prompt on device")
        print("   5. Verify ADB is installed: adb version")
        print("   6. Try: adb kill-server && adb start-server")
    
    # Cleanup
    controller.cleanup()


if __name__ == "__main__":
    main()