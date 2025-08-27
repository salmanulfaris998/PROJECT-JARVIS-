#!/usr/bin/env python3
"""
JARVIS Device Control Mastery v3.0 - REAL Nothing Phone 2a Control
ACTUAL hardware control - NO SIMULATION
"""

import asyncio
import subprocess
import json
import sqlite3
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque

class DeviceFeature(Enum):
    GLYPH_LEDS = "glyph_leds"
    CAMERA = "camera"
    PERFORMANCE = "performance"
    SYSTEM = "system"

class GlyphZone(Enum):
    CAMERA = "camera"
    DIAGONAL = "diagonal" 
    DOT = "dot"
    BOTTOM = "bottom"
    STRIP = "strip"
    ALL = "all"

class SystemMode(Enum):
    BATTERY_SAVER = "battery_saver"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    GAMING = "gaming"

class RealRootExecutor:
    """REAL root command execution - NO SIMULATION"""
    
    def __init__(self):
        self.root_verified = False
        self.device_connected = False
    
    async def verify_device_and_root(self) -> Dict[str, Any]:
        """Verify REAL device connection and root access"""
        try:
            # Check ADB connection
            result = await self._execute_adb(['devices'])
            if not result['success'] or 'device' not in result['output']:
                return {
                    'success': False,
                    'device_connected': False,
                    'root_verified': False,
                    'error': 'No device connected via ADB. Please connect your Nothing Phone 2a.'
                }
            
            self.device_connected = True
            
            # Check root access
            root_result = await self._execute_adb(['shell', 'su', '-c', 'id'])
            if root_result['success'] and 'uid=0' in root_result['output']:
                self.root_verified = True
                
                # Get device info
                model_result = await self._execute_adb(['shell', 'getprop', 'ro.product.model'])
                brand_result = await self._execute_adb(['shell', 'getprop', 'ro.product.brand'])
                
                return {
                    'success': True,
                    'device_connected': True,
                    'root_verified': True,
                    'device_model': model_result['output'] if model_result['success'] else 'Unknown',
                    'device_brand': brand_result['output'] if brand_result['success'] else 'Unknown',
                    'is_nothing_phone': 'A142' in model_result.get('output', '') or 'Nothing' in brand_result.get('output', '')
                }
            else:
                return {
                    'success': False,
                    'device_connected': True,
                    'root_verified': False,
                    'error': 'Device connected but root access denied. Please enable root access.'
                }
                
        except Exception as e:
            return {
                'success': False,
                'device_connected': False,
                'root_verified': False,
                'error': str(e)
            }
    
    async def _execute_adb(self, command: List[str], timeout: float = 10.0) -> Dict[str, Any]:
        """Execute REAL ADB command"""
        try:
            process = await asyncio.create_subprocess_exec(
                'adb', *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                return {'success': False, 'error': f'Command timeout after {timeout}s'}
            
            output = stdout.decode('utf-8', errors='ignore').strip()
            error = stderr.decode('utf-8', errors='ignore').strip()
            success = process.returncode == 0
            
            return {
                'success': success,
                'output': output,
                'error': error if not success else '',
                'returncode': process.returncode
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'output': ''}
    
    async def execute_root_command(self, command: str) -> Dict[str, Any]:
        """Execute REAL root command on device"""
        if not self.root_verified:
            return {'success': False, 'error': 'Root access not verified'}
        
        if not self.device_connected:
            return {'success': False, 'error': 'Device not connected'}
        
        return await self._execute_adb(['shell', 'su', '-c', command])

class RealGlyphController:
    """REAL Glyph LED control - NO SIMULATION"""
    
    def __init__(self, root_executor: RealRootExecutor):
        self.root_executor = root_executor
        self.glyph_leds = {}
        self.led_paths = {}
        self.discovered = False
    
    async def discover_real_glyph_hardware(self) -> Dict[str, Any]:
        """Discover REAL Glyph LED hardware"""
        try:
            print("🔍 Scanning for REAL Glyph LED hardware...")
            
            if not self.root_executor.root_verified:
                return {'success': False, 'error': 'Root access required for Glyph discovery'}
            
            # Real LED discovery paths for Nothing Phone 2a
            potential_led_paths = [
                '/sys/class/leds/aw20036_led',
                '/sys/class/leds/glyph_led',
                '/sys/class/leds/indicator',
                '/sys/class/leds/led:',
                '/sys/class/leds/white:',
                '/sys/class/leds/rgb:'
            ]
            
            discovered_leds = {}
            
            # Scan /sys/class/leds/ directory
            scan_result = await self.root_executor.execute_root_command('ls -la /sys/class/leds/')
            
            if scan_result['success']:
                print(f"📋 Available LEDs on device:")
                led_entries = scan_result['output'].split('\n')
                
                for entry in led_entries:
                    if '->' in entry:  # LED symlinks
                        led_name = entry.split()[-3] if len(entry.split()) >= 3 else ''
                        if any(pattern in led_name.lower() for pattern in ['glyph', 'aw20036', 'led', 'indicator']):
                            led_path = f'/sys/class/leds/{led_name}'
                            
                            # Test if we can read brightness
                            test_result = await self.root_executor.execute_root_command(f'cat {led_path}/brightness')
                            if test_result['success']:
                                discovered_leds[led_name] = led_path
                                print(f"   ✅ Found controllable LED: {led_name}")
                            else:
                                print(f"   ❌ LED not controllable: {led_name}")
                
                if discovered_leds:
                    self.glyph_leds = discovered_leds
                    self.discovered = True
                    
                    # Map LEDs to zones (Nothing Phone 2a specific)
                    self._map_real_leds_to_zones()
                    
                    return {
                        'success': True,
                        'leds_found': len(discovered_leds),
                        'led_names': list(discovered_leds.keys()),
                        'zones_mapped': len(self.led_paths)
                    }
                else:
                    return {
                        'success': False,
                        'error': 'No controllable Glyph LEDs found on device'
                    }
            else:
                return {
                    'success': False,
                    'error': f'Failed to scan LED directory: {scan_result["error"]}'
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _map_real_leds_to_zones(self):
        """Map discovered LEDs to Glyph zones"""
        self.led_paths = {zone: [] for zone in GlyphZone}
        
        for led_name, led_path in self.glyph_leds.items():
            # Nothing Phone 2a LED mapping based on common patterns
            if any(x in led_name.lower() for x in ['cam', 'camera', 'top']):
                self.led_paths[GlyphZone.CAMERA].append(led_path)
            elif any(x in led_name.lower() for x in ['diag', 'slash', 'diagonal']):
                self.led_paths[GlyphZone.DIAGONAL].append(led_path)
            elif any(x in led_name.lower() for x in ['dot', 'point', 'center']):
                self.led_paths[GlyphZone.DOT].append(led_path)
            elif any(x in led_name.lower() for x in ['bottom', 'lower', 'base']):
                self.led_paths[GlyphZone.BOTTOM].append(led_path)
            else:
                self.led_paths[GlyphZone.STRIP].append(led_path)
        
        # Add all LEDs to ALL zone
        self.led_paths[GlyphZone.ALL] = list(self.glyph_leds.values())
        
        print(f"🗺️ LED Zone Mapping:")
        for zone, paths in self.led_paths.items():
            if paths:
                print(f"   {zone.value}: {len(paths)} LEDs")
    
    async def control_real_glyph_zone(self, zone: GlyphZone, brightness: int = 255, 
                                     pattern: str = "solid", duration: float = 0) -> Dict[str, Any]:
        """Control REAL Glyph LEDs - NO SIMULATION"""
        try:
            if not self.discovered:
                return {'success': False, 'error': 'Glyph hardware not discovered. Run discovery first.'}
            
            if not self.root_executor.root_verified:
                return {'success': False, 'error': 'Root access required for Glyph control'}
            
            target_leds = self.led_paths.get(zone, [])
            
            if not target_leds:
                return {'success': False, 'error': f'No LEDs mapped to zone: {zone.value}'}
            
            print(f"🌟 REAL Glyph Control: {zone.value} zone, {len(target_leds)} LEDs, brightness={brightness}")
            
            success_count = 0
            failed_leds = []
            
            # Control each LED in the zone
            for led_path in target_leds:
                # Set brightness
                brightness_cmd = f'echo {brightness} > {led_path}/brightness'
                brightness_result = await self.root_executor.execute_root_command(brightness_cmd)
                
                if brightness_result['success']:
                    success_count += 1
                    print(f"   ✅ LED controlled: {led_path}")
                    
                    # Set trigger pattern if not solid
                    if pattern != "solid" and pattern != "none":
                        trigger_cmd = f'echo {pattern} > {led_path}/trigger'
                        trigger_result = await self.root_executor.execute_root_command(trigger_cmd)
                        
                        if not trigger_result['success']:
                            print(f"   ⚠️ Pattern '{pattern}' not supported on {led_path}")
                else:
                    failed_leds.append(led_path)
                    print(f"   ❌ Failed to control LED: {led_path} - {brightness_result['error']}")
            
            # Handle duration
            if duration > 0 and success_count > 0:
                print(f"⏰ Waiting {duration}s before turning off LEDs...")
                await asyncio.sleep(duration)
                
                # Turn off LEDs
                off_count = 0
                for led_path in target_leds:
                    if led_path not in failed_leds:
                        off_result = await self.root_executor.execute_root_command(f'echo 0 > {led_path}/brightness')
                        if off_result['success']:
                            off_count += 1
                
                print(f"🌑 Turned off {off_count} LEDs")
            
            return {
                'success': success_count > 0,
                'zone': zone.value,
                'brightness': brightness,
                'pattern': pattern,
                'leds_controlled': success_count,
                'total_leds': len(target_leds),
                'failed_leds': len(failed_leds),
                'success_rate': (success_count / len(target_leds)) * 100 if target_leds else 0
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

class RealPerformanceController:
    """REAL performance control - NO SIMULATION"""
    
    def __init__(self, root_executor: RealRootExecutor):
        self.root_executor = root_executor
        self.cpu_cores = 8  # Nothing Phone 2a typical core count
        self.current_mode = SystemMode.BALANCED
    
    async def discover_cpu_info(self) -> Dict[str, Any]:
        """Discover REAL CPU information"""
        try:
            if not self.root_executor.root_verified:
                return {'success': False, 'error': 'Root access required'}
            
            # Get CPU core count
            cpu_info = await self.root_executor.execute_root_command('cat /proc/cpuinfo | grep processor | wc -l')
            if cpu_info['success'] and cpu_info['output'].isdigit():
                self.cpu_cores = int(cpu_info['output'])
            
            # Get current governor
            gov_result = await self.root_executor.execute_root_command('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')
            current_governor = gov_result['output'] if gov_result['success'] else 'unknown'
            
            # Get available governors
            avail_result = await self.root_executor.execute_root_command('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors')
            available_governors = avail_result['output'].split() if avail_result['success'] else []
            
            # Get frequency info
            freq_result = await self.root_executor.execute_root_command('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')
            current_freq = freq_result['output'] if freq_result['success'] else 'unknown'
            
            return {
                'success': True,
                'cpu_cores': self.cpu_cores,
                'current_governor': current_governor,
                'available_governors': available_governors,
                'current_frequency': current_freq
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def set_real_performance_mode(self, mode: SystemMode) -> Dict[str, Any]:
        """Set REAL performance mode - NO SIMULATION"""
        try:
            if not self.root_executor.root_verified:
                return {'success': False, 'error': 'Root access required'}
            
            print(f"⚡ Setting REAL performance mode: {mode.value}")
            
            # Performance mode configurations
            mode_configs = {
                SystemMode.BATTERY_SAVER: {
                    'governor': 'powersave',
                    'description': 'Maximum battery life'
                },
                SystemMode.BALANCED: {
                    'governor': 'ondemand', 
                    'description': 'Balanced performance and battery'
                },
                SystemMode.PERFORMANCE: {
                    'governor': 'performance',
                    'description': 'Maximum performance'
                },
                SystemMode.GAMING: {
                    'governor': 'performance',
                    'description': 'Gaming optimized performance'
                }
            }
            
            config = mode_configs[mode]
            success_cores = 0
            failed_cores = []
            
            # Apply governor to all CPU cores
            for core in range(self.cpu_cores):
                gov_cmd = f'echo {config["governor"]} > /sys/devices/system/cpu/cpu{core}/cpufreq/scaling_governor'
                result = await self.root_executor.execute_root_command(gov_cmd)
                
                if result['success']:
                    success_cores += 1
                    print(f"   ✅ Core {core}: {config['governor']}")
                else:
                    failed_cores.append(core)
                    print(f"   ❌ Core {core}: Failed - {result['error']}")
            
            # Apply additional optimizations
            if mode == SystemMode.GAMING:
                await self._apply_real_gaming_optimizations()
            elif mode == SystemMode.BATTERY_SAVER:
                await self._apply_real_battery_optimizations()
            
            if success_cores > 0:
                self.current_mode = mode
            
            return {
                'success': success_cores > 0,
                'mode': mode.value,
                'description': config['description'],
                'governor': config['governor'],
                'cores_configured': success_cores,
                'total_cores': self.cpu_cores,
                'failed_cores': failed_cores,
                'success_rate': (success_cores / self.cpu_cores) * 100
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _apply_real_gaming_optimizations(self):
        """Apply REAL gaming optimizations"""
        optimizations = [
            ('debug.sf.disable_backpressure', '1'),
            ('debug.sf.latch_unsignaled', '1'),
            ('ro.surface_flinger.max_frame_buffer_acquired_buffers', '3'),
            ('debug.egl.hw', '1')
        ]
        
        for prop, value in optimizations:
            cmd = f'setprop {prop} {value}'
            result = await self.root_executor.execute_root_command(cmd)
            status = "✅" if result['success'] else "❌"
            print(f"   {status} Gaming opt: {prop} = {value}")
    
    async def _apply_real_battery_optimizations(self):
        """Apply REAL battery optimizations"""
        optimizations = [
            ('ro.config.low_ram', 'true'),
            ('ro.vendor.qti.sys.fw.bg_apps_limit', '24'),
            ('persist.vendor.radio.enableadvancedscan', '0')
        ]
        
        for prop, value in optimizations:
            cmd = f'setprop {prop} {value}'
            result = await self.root_executor.execute_root_command(cmd)
            status = "✅" if result['success'] else "❌"
            print(f"   {status} Battery opt: {prop} = {value}")

class RealCameraController:
    """REAL camera control - NO SIMULATION"""
    
    def __init__(self, root_executor: RealRootExecutor):
        self.root_executor = root_executor
        self.camera_packages = []
        self.active_camera = None
    
    async def discover_real_camera_apps(self) -> Dict[str, Any]:
        """Discover REAL camera applications"""
        try:
            potential_cameras = [
                'com.nothing.camera',
                'com.android.camera2',
                'com.google.android.GoogleCamera',
                'org.codeaurora.snapcam',
                'com.oneplus.camera'
            ]
            
            available_cameras = []
            
            for package in potential_cameras:
                result = await self.root_executor._execute_adb(['shell', 'pm', 'list', 'packages', package])
                
                if result['success'] and package in result['output']:
                    available_cameras.append(package)
                    print(f"   📸 Found camera app: {package}")
            
            self.camera_packages = available_cameras
            
            return {
                'success': True,
                'available_cameras': available_cameras,
                'primary_camera': available_cameras[0] if available_cameras else None,
                'nothing_camera_available': 'com.nothing.camera' in available_cameras
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def open_real_camera(self, camera_type: str = "back") -> Dict[str, Any]:
        """Open REAL camera application"""
        try:
            if not self.camera_packages:
                return {'success': False, 'error': 'No camera apps discovered'}
            
            print(f"📸 Opening REAL camera: {camera_type}")
            
            # Use Nothing camera if available, otherwise first available
            camera_package = 'com.nothing.camera' if 'com.nothing.camera' in self.camera_packages else self.camera_packages[0]
            
            # Launch camera app
            result = await self.root_executor._execute_adb([
                'shell', 'am', 'start', '-n', f'{camera_package}/.CameraActivity'
            ])
            
            if not result['success']:
                # Try generic camera intent
                result = await self.root_executor._execute_adb([
                    'shell', 'am', 'start', '-a', 'android.media.action.IMAGE_CAPTURE'
                ])
            
            if result['success']:
                self.active_camera = camera_type
                return {
                    'success': True,
                    'camera_type': camera_type,
                    'camera_package': camera_package,
                    'message': f'{camera_type} camera opened using {camera_package}'
                }
            else:
                return {
                    'success': False,
                    'error': f'Failed to open camera: {result["error"]}'
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def take_real_photo_with_glyph(self, glyph_controller: RealGlyphController) -> Dict[str, Any]:
        """Take REAL photo with REAL Glyph flash"""
        try:
            print("📸 Taking REAL photo with REAL Glyph flash")
            
            # Pre-flash - illuminate all glyphs
            await glyph_controller.control_real_glyph_zone(GlyphZone.ALL, 255, "solid", 0.5)
            
            # Take photo using camera intent
            photo_result = await self.root_executor._execute_adb([
                'shell', 'input', 'keyevent', 'KEYCODE_CAMERA'
            ])
            
            # Flash effect sequence
            flash_sequence = [
                (GlyphZone.ALL, 255, 0.1),
                (GlyphZone.ALL, 0, 0.1),
                (GlyphZone.ALL, 255, 0.1),
                (GlyphZone.ALL, 0, 0.1)
            ]
            
            for zone, brightness, duration in flash_sequence:
                await glyph_controller.control_real_glyph_zone(zone, brightness, "solid", duration)
            
            return {
                'success': photo_result['success'],
                'message': 'Real photo taken with real Glyph flash',
                'flash_used': True,
                'timestamp': datetime.now().isoformat(),
                'camera_triggered': photo_result['success']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

class AdvancedNothingController:
    """REAL Nothing Phone 2a Controller - NO SIMULATION"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.root_executor = RealRootExecutor()
        self.glyph_controller = RealGlyphController(self.root_executor)
        self.performance_controller = RealPerformanceController(self.root_executor)
        self.camera_controller = RealCameraController(self.root_executor)
        
        self.device_info = {}
        self.features_available = {}
        
        self.db_path = Path("logs/real_device_control.db")
        self._init_database()
        
        self.logger.info("🎯 REAL Nothing Controller v3.0 initialized - NO SIMULATION")

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('real_device_control')
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / f'real_device_{datetime.now().strftime("%Y%m%d")}.log')
        file_formatter = logging.Formatter('%(asctime)s | REAL_DEVICE | %(levelname)s | %(message)s')
        file_handler.setFormatter(file_formatter)
        
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('🎯 %(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def _init_database(self):
        try:
            self.db_path.parent.mkdir(exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS real_device_commands (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        feature TEXT NOT NULL,
                        action TEXT NOT NULL,
                        parameters TEXT NOT NULL,
                        execution_time REAL,
                        success BOOLEAN,
                        result TEXT,
                        device_info TEXT
                    )
                ''')
                
                conn.commit()
            
            self.logger.info("✅ Real device database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")

    async def initialize_real_system(self) -> bool:
        """Initialize REAL device control system - NO SIMULATION"""
        try:
            print("\n🔥 INITIALIZING REAL NOTHING PHONE CONTROL")
            print("=" * 60)
            print("⚠️  THIS IS REAL HARDWARE CONTROL - NOT SIMULATION")
            print("=" * 60)
            
            # Step 1: Verify device and root
            print("📱 Verifying REAL device connection and root access...")
            device_status = await self.root_executor.verify_device_and_root()
            
            if not device_status['success']:
                print(f"❌ Device verification failed: {device_status['error']}")
                return False
            
            self.device_info = device_status
            print(f"✅ Device verified: {device_status['device_model']} ({device_status['device_brand']})")
            print(f"✅ Root access: {'Verified' if device_status['root_verified'] else 'Failed'}")
            print(f"✅ Nothing Phone: {'Yes' if device_status['is_nothing_phone'] else 'No'}")
            
            # Step 2: Discover REAL Glyph hardware
            print("\n🌟 Discovering REAL Glyph LED hardware...")
            glyph_result = await self.glyph_controller.discover_real_glyph_hardware()
            
            if glyph_result['success']:
                print(f"✅ Found {glyph_result['leds_found']} controllable Glyph LEDs")
                print(f"   LEDs: {', '.join(glyph_result['led_names'])}")
                self.features_available[DeviceFeature.GLYPH_LEDS] = True
            else:
                print(f"❌ Glyph discovery failed: {glyph_result['error']}")
                self.features_available[DeviceFeature.GLYPH_LEDS] = False
            
            # Step 3: Discover REAL CPU info
            print("\n⚡ Discovering REAL CPU performance controls...")
            cpu_result = await self.performance_controller.discover_cpu_info()
            
            if cpu_result['success']:
                print(f"✅ CPU: {cpu_result['cpu_cores']} cores")
                print(f"   Current governor: {cpu_result['current_governor']}")
                print(f"   Available governors: {', '.join(cpu_result['available_governors'])}")
                self.features_available[DeviceFeature.PERFORMANCE] = True
            else:
                print(f"❌ CPU discovery failed: {cpu_result['error']}")
                self.features_available[DeviceFeature.PERFORMANCE] = False
            
            # Step 4: Discover REAL camera apps
            print("\n📸 Discovering REAL camera applications...")
            camera_result = await self.camera_controller.discover_real_camera_apps()
            
            if camera_result['success'] and camera_result['available_cameras']:
                print(f"✅ Found {len(camera_result['available_cameras'])} camera apps")
                if camera_result['nothing_camera_available']:
                    print("   🌟 Nothing Camera app available!")
                self.features_available[DeviceFeature.CAMERA] = True
            else:
                print("❌ No camera apps found")
                self.features_available[DeviceFeature.CAMERA] = False
            
            print("\n" + "="*60)
            print("🔥 REAL NOTHING PHONE CONTROL INITIALIZED!")
            print("="*60)
            print(f"📱 Device: {self.device_info.get('device_model', 'Unknown')}")
            print(f"🔓 Root: {'✅' if self.device_info.get('root_verified') else '❌'}")
            print(f"🌟 Glyph LEDs: {'✅' if self.features_available.get(DeviceFeature.GLYPH_LEDS) else '❌'}")
            print(f"⚡ Performance: {'✅' if self.features_available.get(DeviceFeature.PERFORMANCE) else '❌'}")
            print(f"📸 Camera: {'✅' if self.features_available.get(DeviceFeature.CAMERA) else '❌'}")
            print("="*60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Real system initialization failed: {e}")
            return False

    # High-level REAL control methods
    
    async def control_glyph_leds_advanced(self, pattern: str = "solid", brightness: int = 255, 
                                        duration: int = 3, zone: str = "all", 
                                        animation_speed: float = 1.0) -> Dict[str, Any]:
        """REAL Glyph LED control"""
        try:
            glyph_zone = GlyphZone.ALL if zone == "all" else GlyphZone(zone)
            result = await self.glyph_controller.control_real_glyph_zone(
                glyph_zone, brightness, pattern, duration
            )
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def set_performance_mode_advanced(self, mode) -> Dict[str, Any]:
        """Set REAL performance mode"""
        try:
            if hasattr(mode, 'value'):
                mode_name = mode.value
            else:
                mode_name = str(mode).lower()
            
            mode_mapping = {
                'gaming': SystemMode.GAMING,
                'performance': SystemMode.PERFORMANCE,
                'balanced': SystemMode.BALANCED,
                'battery': SystemMode.BATTERY_SAVER,
                'battery_saver': SystemMode.BATTERY_SAVER
            }
            
            system_mode = mode_mapping.get(mode_name, SystemMode.BALANCED)
            result = await self.performance_controller.set_real_performance_mode(system_mode)
            
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def camera_control_advanced(self, action: str, camera_type: str = "back", 
                                    settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """REAL camera control"""
        try:
            settings = settings or {}
            
            if action == "open":
                result = await self.camera_controller.open_real_camera(camera_type)
            elif action == "photo":
                if settings.get('glyph_flash', False):
                    result = await self.camera_controller.take_real_photo_with_glyph(
                        self.glyph_controller
                    )
                else:
                    # Take photo without Glyph flash
                    photo_result = await self.root_executor._execute_adb([
                        'shell', 'input', 'keyevent', 'KEYCODE_CAMERA'
                    ])
                    result = {
                        'success': photo_result['success'],
                        'message': 'Real photo taken',
                        'flash_used': False
                    }
            else:
                result = {'success': False, 'error': f'Unknown camera action: {action}'}
            
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def monitor_system_stats_advanced(self, duration: int = 5) -> Dict[str, Any]:
        """Monitor REAL system statistics"""
        try:
            stats_samples = []
            
            for i in range(duration):
                sample = {'timestamp': time.time()}
                
                # Real battery info
                battery_result = await self.root_executor.execute_root_command('dumpsys battery | grep level')
                if battery_result['success']:
                    for line in battery_result['output'].split('\n'):
                        if 'level:' in line:
                            try:
                                sample['battery_level'] = int(line.split(':')[1].strip())
                            except:
                                pass
                            break
                
                # Real temperature
                temp_result = await self.root_executor.execute_root_command('cat /sys/class/thermal/thermal_zone0/temp')
                if temp_result['success'] and temp_result['output'].isdigit():
                    sample['temperature'] = float(temp_result['output']) / 1000
                
                # Real CPU frequency
                freq_result = await self.root_executor.execute_root_command('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')
                if freq_result['success'] and freq_result['output'].isdigit():
                    sample['cpu_frequency'] = int(freq_result['output']) // 1000  # Convert to MHz
                
                stats_samples.append(sample)
                
                if i < duration - 1:
                    await asyncio.sleep(1)
            
            # Calculate real averages
            averages = {}
            for key in ['battery_level', 'temperature', 'cpu_frequency']:
                values = [s[key] for s in stats_samples if key in s]
                if values:
                    averages[key] = sum(values) / len(values)
            
            return {
                'success': True,
                'stats': {
                    'samples': stats_samples,
                    'averages': averages
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Main execution
async def main():
    """Main REAL device control test"""
    try:
        print("🔥 REAL Nothing Phone 2a Control Test")
        print("=" * 60)
        print("⚠️  CONNECTING TO REAL HARDWARE")
        
        controller = AdvancedNothingController()
        
        if await controller.initialize_real_system():
            print("\n🧪 Testing REAL device control features...")
            
            # Test REAL Glyph control
            print("\n🌟 Testing REAL Glyph LED control...")
            glyph_result = await controller.control_glyph_leds_advanced("solid", 200, 3, "all")
            print(f"Glyph Test: {'✅' if glyph_result['success'] else '❌'} - {glyph_result.get('message', glyph_result.get('error'))}")
            
            # Test REAL performance mode
            print("\n⚡ Testing REAL performance control...")
            perf_result = await controller.set_performance_mode_advanced("gaming")
            print(f"Performance Test: {'✅' if perf_result['success'] else '❌'} - {perf_result.get('message', perf_result.get('error'))}")
            
            # Test REAL camera
            print("\n📸 Testing REAL camera control...")
            camera_result = await controller.camera_control_advanced("photo", "back", {"glyph_flash": True})
            print(f"Camera Test: {'✅' if camera_result['success'] else '❌'} - {camera_result.get('message', camera_result.get('error'))}")
            
            # Test REAL system stats
            print("\n📊 Testing REAL system monitoring...")
            stats_result = await controller.monitor_system_stats_advanced(3)
            print(f"Stats Test: {'✅' if stats_result['success'] else '❌'}")
            if stats_result['success'] and stats_result['stats']['averages']:
                for key, value in stats_result['stats']['averages'].items():
                    print(f"   {key}: {value}")
            
            print(f"\n✅ REAL device control test completed!")
            
        else:
            print("❌ REAL device controller initialization failed")
            print("💡 Make sure your Nothing Phone 2a is:")
            print("   1. Connected via USB cable")
            print("   2. ADB debugging enabled")
            print("   3. Root access granted")
            print("   4. USB debugging authorized on device")
        
    except Exception as e:
        print(f"❌ REAL device control test error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
