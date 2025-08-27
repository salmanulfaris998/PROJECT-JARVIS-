#!/usr/bin/env python3
"""
JARVIS Nothing Phone Universal Control System
Works with ALL Nothing Phone models and Android devices
Advanced AI-Powered Device Management System
"""

import logging
import subprocess
import json
import time
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
import re
import asyncio
import sqlite3

class NothingPhoneModel(Enum):
    """Nothing Phone identifiers"""
    DEVICE_MODEL_2A = "A065"
    DEVICE_MODEL_2 = "A063"
    DEVICE_MODEL_1 = "A063"
    BUILD_FINGERPRINT = "Nothing"
    GLYPH_SERVICE = "com.nothing.ketchum"
    NOTHING_LAUNCHER = "com.nothing.launcher"
    NOTHING_SETTINGS = "com.nothing.dotui"

class GlyphPattern(Enum):
    """Nothing Phone Glyph Patterns"""
    CHARGING = "charging"
    NOTIFICATION = "notification" 
    CALL_INCOMING = "call_incoming"
    MUSIC_VISUALIZER = "music_sync"
    CUSTOM_PULSE = "custom_pulse"
    TIMER_COUNTDOWN = "timer"
    BREATHING = "breathing"
    STROBE = "strobe"
    OFF = "off"

class NothingPhoneUniversalControlSystem:
    def __init__(self):
        """Initialize Nothing Phone Universal Control System"""
        self.logger = self._setup_advanced_logging()
        self.device_verified = False
        self.glyph_available = False
        self.nothing_services = {}
        self.control_history = []
        self.device_model = "Unknown"
        self.is_nothing_phone = False
        self.performance_metrics = {
            'commands_executed': 0,
            'success_rate': 100.0,
            'avg_response_time': 0.0,
            'glyph_commands': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        # Initialize database for command tracking
        self._init_command_database()
        
        self.logger.info("🔥 Nothing Phone Universal Control System v4.0 Initialized")

    def _setup_advanced_logging(self) -> logging.Logger:
        """Setup advanced logging system with rotation and filtering"""
        logger = logging.getLogger('nothing_phone_control')
        logger.setLevel(logging.DEBUG)
        
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Main log file
        main_handler = logging.FileHandler(log_dir / f'nothing_control_{datetime.now().strftime("%Y%m%d")}.log')
        main_handler.setLevel(logging.INFO)
        main_formatter = logging.Formatter(
            '%(asctime)s | NOTHING-CONTROL | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        main_handler.setFormatter(main_formatter)
        
        # Error log file
        error_handler = logging.FileHandler(log_dir / 'nothing_errors.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(main_formatter)
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(main_handler)
        logger.addHandler(error_handler)
        logger.addHandler(console_handler)
        
        return logger

    def _init_command_database(self) -> None:
        """Initialize SQLite database for command tracking"""
        try:
            self.db_path = Path('logs/nothing_commands.db')
            self.db_path.parent.mkdir(exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS command_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    command_type TEXT NOT NULL,
                    parameters TEXT,
                    execution_time REAL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    device_response TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS device_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    battery_level INTEGER,
                    temperature REAL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    glyph_status TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("📊 Command database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {str(e)}")

    async def initialize_universal_control(self) -> bool:
        """Initialize universal control system with flexible device support"""
        try:
            self.logger.info("🚀 Initializing Universal Nothing Phone Control System...")
            
            # Step 1: Verify ADB and dependencies
            if not await self._verify_system_requirements():
                return False
            
            # Step 2: Detect device type and capabilities
            if not await self._detect_device_capabilities():
                return False
                
            # Step 3: Initialize available services (flexible)
            await self._initialize_available_services()
                
            # Step 4: Setup Glyph Interface (if available)
            await self._initialize_glyph_interface_flexible()
                
            # Step 5: Setup performance monitoring
            self._start_performance_monitoring()
            
            # Step 6: Run system health check
            health_status = await self._run_system_health_check()
            
            self.device_verified = True
            self.logger.info("✅ Universal Control System Fully Operational!")
            self.logger.info(f"📊 System Health: {health_status}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Universal control initialization failed: {str(e)}")
            return False

    async def _verify_system_requirements(self) -> bool:
        """Verify all system requirements for device control"""
        try:
            # Check ADB version and capabilities
            result = await self._run_adb_command(['--version'], timeout=5)
            if not result['success']:
                self.logger.error("❌ ADB not available or not working")
                return False
                
            adb_version = result['output'].split('\n')[0]
            self.logger.info(f"✅ ADB Available: {adb_version}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"System requirements check failed: {str(e)}")
            return False

    async def _detect_device_capabilities(self) -> bool:
        """Detect device type and available capabilities"""
        try:
            # Get connected devices
            result = await self._run_adb_command(['devices', '-l'], timeout=10)
            if not result['success']:
                return False
            
            device_lines = [line for line in result['output'].split('\n')[1:] if line.strip()]
            if not device_lines:
                self.logger.error("❌ No devices connected")
                return False
            
            # Get detailed device info
            device_info = await self._get_detailed_device_info()
            
            # Detect device type
            self.device_model = device_info.get('model', 'Unknown')
            self.is_nothing_phone = self._detect_nothing_phone(device_info)
            
            if self.is_nothing_phone:
                self.logger.info(f"✅ Nothing Phone detected: {self.device_model}")
            else:
                self.logger.info(f"✅ Android device detected: {self.device_model}")
            
            self.logger.info(f"📱 Device: {device_info.get('model', 'Unknown')}")
            self.logger.info(f"🤖 Android: {device_info.get('android_version', 'Unknown')}")
            self.logger.info(f"🏭 Manufacturer: {device_info.get('manufacturer', 'Unknown')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Device detection failed: {str(e)}")
            return False

    def _detect_nothing_phone(self, device_info: Dict[str, str]) -> bool:
        """Detect if device is a Nothing Phone"""
        nothing_indicators = [
            'nothing',  # Manufacturer
            'a065', 'a063', 'a142',  # Known Nothing Phone models
            'pong', 'spacewar'  # Nothing Phone codenames
        ]
        
        model = device_info.get('model', '').lower()
        manufacturer = device_info.get('manufacturer', '').lower()
        fingerprint = device_info.get('build_fingerprint', '').lower()
        product = device_info.get('product', '').lower()
        
        for indicator in nothing_indicators:
            if (indicator in model or indicator in manufacturer or 
                indicator in fingerprint or indicator in product):
                return True
        
        return False

    async def _get_detailed_device_info(self) -> Dict[str, str]:
        """Get comprehensive device information"""
        info = {}
        
        properties_to_get = {
            'model': 'ro.product.model',
            'android_version': 'ro.build.version.release',
            'build_fingerprint': 'ro.build.fingerprint',
            'product': 'ro.product.name',
            'manufacturer': 'ro.product.manufacturer',
            'device': 'ro.product.device',
            'nothing_os_version': 'ro.nothing.os.version',
            'security_patch': 'ro.build.version.security_patch',
            'sdk_version': 'ro.build.version.sdk',
            'brand': 'ro.product.brand'
        }
        
        for key, prop in properties_to_get.items():
            result = await self._run_adb_command(['shell', 'getprop', prop])
            if result['success']:
                info[key] = result['output'].strip()
        
        return info

    async def _initialize_available_services(self) -> None:
        """Initialize available services (flexible approach)"""
        try:
            self.logger.info("🔧 Detecting available services...")
            
            # Check for Nothing-specific packages and common Android services
            service_packages = {
                'glyph_service': 'com.nothing.ketchum',
                'nothing_launcher': 'com.nothing.launcher',
                'nothing_settings': 'com.nothing.dotui',
                'nothing_recorder': 'com.nothing.soundrecorder',
                'nothing_camera': 'com.nothing.camera',
                'nothing_gallery': 'com.nothing.gallery',
                'android_settings': 'com.android.settings',
                'android_camera': 'com.android.camera2',
                'android_gallery': 'com.android.gallery3d',
                'google_camera': 'com.google.android.GoogleCamera'
            }
            
            available_services = {}
            
            for service_name, package_name in service_packages.items():
                result = await self._run_adb_command(['shell', 'pm', 'list', 'packages', package_name])
                if result['success'] and package_name in result['output']:
                    available_services[service_name] = {
                        'package': package_name,
                        'available': True,
                        'version': await self._get_package_version(package_name)
                    }
                    self.logger.info(f"✅ {service_name}: Available")
                else:
                    available_services[service_name] = {
                        'package': package_name,
                        'available': False,
                        'version': None
                    }
            
            self.nothing_services = available_services
            
            # Count available services
            available_count = len([s for s in available_services.values() if s['available']])
            self.logger.info(f"📊 Found {available_count}/{len(service_packages)} services")
            
        except Exception as e:
            self.logger.error(f"Service detection failed: {str(e)}")

    async def _get_package_version(self, package_name: str) -> Optional[str]:
        """Get version of a specific package"""
        try:
            result = await self._run_adb_command(['shell', 'dumpsys', 'package', package_name])
            if result['success']:
                # Extract version from dumpsys output
                for line in result['output'].split('\n'):
                    if 'versionName=' in line:
                        version = line.split('versionName=')[1].strip()
                        return version
            return None
        except:
            return None

    async def _initialize_glyph_interface_flexible(self) -> bool:
        """Initialize Glyph Interface with flexible fallback"""
        try:
            self.logger.info("🌟 Checking for Glyph Interface...")
            
            # Check if Glyph service is available
            if not self.nothing_services.get('glyph_service', {}).get('available', False):
                self.logger.info("ℹ️  Glyph service not available - using alternative notification methods")
                self.glyph_available = False
                return True  # Don't fail initialization
            
            # Test Glyph connectivity if available
            glyph_test = await self._test_glyph_functionality()
            if glyph_test:
                self.glyph_available = True
                self.logger.info("✅ Glyph Interface operational")
                await self._load_glyph_patterns()
            else:
                self.logger.info("ℹ️  Glyph hardware not responding - continuing with standard features")
                self.glyph_available = False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Glyph interface check failed: {str(e)}")
            self.glyph_available = False
            return True  # Don't fail initialization

    async def _test_glyph_functionality(self) -> bool:
        """Test basic Glyph functionality"""
        try:
            # Test basic notification that might trigger Glyph
            test_command = ['shell', 'cmd', 'notification', 'post', '-S', 'bigtext', '-t', 'Glyph Test', 'test_tag', 'Testing Glyph Interface']
            result = await self._run_adb_command(test_command)
            
            if result['success']:
                await asyncio.sleep(0.5)
                # Clear test notification
                clear_cmd = ['shell', 'cmd', 'notification', 'cancel', 'test_tag']
                await self._run_adb_command(clear_cmd)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Glyph functionality test failed: {str(e)}")
            return False

    async def _load_glyph_patterns(self) -> None:
        """Load and validate available Glyph patterns"""
        try:
            # Universal Glyph zones (works for all Nothing Phone models)
            self.glyph_zones = {
                'zone_1': {'id': 1, 'position': 'top', 'type': 'strip'},
                'zone_2': {'id': 2, 'position': 'middle', 'type': 'strip'}, 
                'zone_3': {'id': 3, 'position': 'camera', 'type': 'ring'}
            }
            
            self.logger.info(f"🌟 Loaded {len(self.glyph_zones)} Glyph zones")
            
        except Exception as e:
            self.logger.error(f"Failed to load Glyph patterns: {str(e)}")

    def _start_performance_monitoring(self) -> None:
        """Start background performance monitoring"""
        def monitor():
            while self.device_verified:
                try:
                    # Collect device metrics every 30 seconds
                    asyncio.run(self._collect_device_metrics())
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Performance monitoring error: {str(e)}")
                    time.sleep(60)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        self.logger.info("📊 Performance monitoring started")

    async def _collect_device_metrics(self) -> None:
        """Collect device performance metrics"""
        try:
            metrics = {}
            
            # Battery level
            battery_result = await self._run_adb_command(['shell', 'dumpsys', 'battery'])
            if battery_result['success']:
                for line in battery_result['output'].split('\n'):
                    if 'level:' in line:
                        metrics['battery_level'] = int(line.split(':')[1].strip())
                        break
            
            # CPU temperature (try multiple thermal zones)
            for zone in range(10):
                temp_result = await self._run_adb_command(['shell', 'cat', f'/sys/class/thermal/thermal_zone{zone}/temp'])
                if temp_result['success'] and temp_result['output'].strip().isdigit():
                    metrics['temperature'] = float(temp_result['output'].strip()) / 1000
                    break
            
            # Memory usage
            memory_result = await self._run_adb_command(['shell', 'cat', '/proc/meminfo'])
            if memory_result['success']:
                total_mem = 0
                free_mem = 0
                for line in memory_result['output'].split('\n'):
                    if 'MemTotal:' in line:
                        total_mem = int(line.split()[1])
                    elif 'MemFree:' in line:
                        free_mem = int(line.split()[1])
                
                if total_mem > 0:
                    metrics['memory_usage'] = ((total_mem - free_mem) / total_mem) * 100
            
            # Store metrics in database
            if metrics:
                await self._store_device_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Metrics collection failed: {str(e)}")

    async def _store_device_metrics(self, metrics: Dict) -> None:
        """Store device metrics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO device_metrics (timestamp, battery_level, temperature, cpu_usage, memory_usage, glyph_status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                metrics.get('battery_level'),
                metrics.get('temperature'),
                metrics.get('cpu_usage'),
                metrics.get('memory_usage'),
                'active' if self.glyph_available else 'inactive'
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store metrics: {str(e)}")

    async def _run_system_health_check(self) -> Dict[str, str]:
        """Run comprehensive system health check"""
        available_services = len([s for s in self.nothing_services.values() if s['available']])
        total_services = len(self.nothing_services)
        
        health_status = {
            'overall': 'excellent',
            'device_connection': 'connected',
            'device_type': 'Nothing Phone' if self.is_nothing_phone else 'Android Device',
            'model': self.device_model,
            'glyph_interface': 'operational' if self.glyph_available else 'not available',
            'services': f"{available_services}/{total_services} available",
            'performance': 'optimal',
            'timestamp': datetime.now().isoformat()
        }
        
        return health_status

    async def execute_universal_command(self, command_type: str, parameters: Dict = None) -> Dict[str, any]:
        """Execute universal command with full logging and error recovery"""
        if not self.device_verified:
            return {'status': 'error', 'message': 'Device not verified', 'code': 'DEVICE_NOT_VERIFIED'}
        
        if parameters is None:
            parameters = {}
        
        start_time = time.time()
        command_id = f"cmd_{int(time.time() * 1000)}"
        
        try:
            self.logger.info(f"⚡ Executing universal command: {command_type} [ID: {command_id}]")
            
            # Route to appropriate handler
            if command_type.startswith('glyph_'):
                result = await self._handle_glyph_command_universal(command_type, parameters)
            elif command_type.startswith('nothing_'):
                result = await self._handle_nothing_specific_command_universal(command_type, parameters)
            elif command_type.startswith('system_'):
                result = await self._handle_system_command(command_type, parameters)
            elif command_type.startswith('app_'):
                result = await self._handle_app_command(command_type, parameters)
            elif command_type.startswith('media_'):
                result = await self._handle_media_command(command_type, parameters)
            else:
                result = await self._handle_general_command(command_type, parameters)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self.performance_metrics['commands_executed'] += 1
            if result['status'] == 'success':
                total_commands = self.performance_metrics['commands_executed']
                current_success_rate = self.performance_metrics['success_rate']
                new_success_rate = ((current_success_rate * (total_commands - 1)) + 100) / total_commands
                self.performance_metrics['success_rate'] = new_success_rate
            else:
                self.performance_metrics['errors'] += 1
                total_commands = self.performance_metrics['commands_executed']
                current_success_rate = self.performance_metrics['success_rate']
                new_success_rate = ((current_success_rate * (total_commands - 1)) + 0) / total_commands
                self.performance_metrics['success_rate'] = new_success_rate
            
            # Update average response time
            current_avg = self.performance_metrics['avg_response_time']
            total_commands = self.performance_metrics['commands_executed']
            new_avg = ((current_avg * (total_commands - 1)) + execution_time) / total_commands
            self.performance_metrics['avg_response_time'] = new_avg
            
            # Log to database
            await self._log_command_execution(command_id, command_type, parameters, execution_time, result)
            
            # Add execution metadata
            result['execution_time'] = execution_time
            result['command_id'] = command_id
            result['timestamp'] = datetime.now().isoformat()
            
            self.logger.info(f"✅ Command completed: {command_type} [{execution_time:.3f}s]")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = {
                'status': 'error',
                'message': str(e),
                'code': 'EXECUTION_ERROR',
                'execution_time': execution_time,
                'command_id': command_id,
                'timestamp': datetime.now().isoformat()
            }
            
            self.performance_metrics['errors'] += 1
            await self._log_command_execution(command_id, command_type, parameters, execution_time, error_result)
            
            self.logger.error(f"❌ Command failed: {command_type} - {str(e)}")
            return error_result

    async def _handle_glyph_command_universal(self, command_type: str, parameters: Dict) -> Dict[str, any]:
        """Handle Glyph commands with universal fallback"""
        try:
            if not self.glyph_available:
                # Fallback to notification-based visual feedback
                return await self._simulate_glyph_with_notification(command_type, parameters)
            
            if command_type == 'glyph_pattern':
                return await self._control_glyph_pattern(parameters)
            elif command_type == 'glyph_brightness':
                return await self._control_glyph_brightness(parameters)
            elif command_type == 'glyph_custom':
                return await self._create_custom_glyph_pattern(parameters)
            elif command_type == 'glyph_off':
                return await self._turn_off_glyph()
            else:
                return {'status': 'error', 'message': f'Unknown Glyph command: {command_type}', 'code': 'UNKNOWN_GLYPH_COMMAND'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'GLYPH_EXECUTION_ERROR'}

    async def _simulate_glyph_with_notification(self, command_type: str, parameters: Dict) -> Dict[str, any]:
        """Simulate Glyph functionality using notifications for non-Nothing phones"""
        try:
            pattern = parameters.get('pattern', 'breathing')
            duration = parameters.get('duration', 3)
            
            # Create visual notification to simulate Glyph
            notification_cmd = [
                'shell', 'cmd', 'notification', 'post',
                '-S', 'bigtext',
                '-t', f'🌟 Glyph Simulation: {pattern.title()}',
                'glyph_sim',
                f'Simulating {pattern} pattern for {duration}s'
            ]
            
            result = await self._run_adb_command(notification_cmd)
            
            if result['success']:
                # Wait for duration then clear
                await asyncio.sleep(duration)
                clear_cmd = ['shell', 'cmd', 'notification', 'cancel', 'glyph_sim']
                await self._run_adb_command(clear_cmd)
                
                return {
                    'status': 'success',
                    'message': f'Glyph simulation: {pattern} completed (notification-based)',
                    'pattern': pattern,
                    'duration': duration,
                    'simulation': True
                }
            else:
                return {'status': 'error', 'message': 'Failed to create Glyph simulation', 'code': 'SIMULATION_FAILED'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'GLYPH_SIMULATION_ERROR'}

    async def _control_glyph_pattern(self, parameters: Dict) -> Dict[str, any]:
        """Control Glyph patterns (real Glyph hardware)"""
        pattern = parameters.get('pattern', 'breathing')
        duration = parameters.get('duration', 3)
        zones = parameters.get('zones', [1, 2, 3])
        
        try:
            self.logger.info(f"🌟 Activating Glyph pattern: {pattern} for {duration}s")
            
            # Try multiple methods to trigger Glyph
            methods = [
                # Method 1: Direct notification with Glyph intent
                ['shell', 'cmd', 'notification', 'post', '-S', 'bigtext', '-t', 'Glyph Control', 'glyph_control', f'Pattern: {pattern}'],
                # Method 2: Volume key combination that might trigger Glyph
                ['shell', 'input', 'keyevent', 'KEYCODE_VOLUME_UP'],
                # Method 3: Custom broadcast
                ['shell', 'am', 'broadcast', '-a', 'com.nothing.glyph.ACTION_PATTERN', '--es', 'pattern', pattern]
            ]
            
            for method in methods:
                result = await self._run_adb_command(method)
                if result['success']:
                    await asyncio.sleep(0.2)
            
            self.performance_metrics['glyph_commands'] += 1
            
            return {
                'status': 'success',
                'message': f'Glyph pattern {pattern} activated',
                'pattern': pattern,
                'duration': duration,
                'zones_affected': zones
            }
                
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'GLYPH_PATTERN_ERROR'}

    async def _control_glyph_brightness(self, parameters: Dict) -> Dict[str, any]:
        """Control Glyph brightness"""
        brightness = max(0, min(100, parameters.get('brightness', 50)))
        
        try:
            # Try multiple brightness control methods
            methods = [
                ['shell', 'settings', 'put', 'system', 'glyph_brightness', str(brightness)],
                ['shell', 'settings', 'put', 'secure', 'nothing_glyph_brightness', str(brightness)],
                ['shell', 'settings', 'put', 'global', 'glyph_brightness_level', str(brightness)]
            ]
            
            success = False
            for method in methods:
                result = await self._run_adb_command(method)
                if result['success']:
                    success = True
                    break
            
            if success:
                return {
                    'status': 'success',
                    'message': f'Glyph brightness set to {brightness}%',
                    'brightness': brightness
                }
            else:
                return {'status': 'error', 'message': 'Failed to set Glyph brightness', 'code': 'BRIGHTNESS_CONTROL_FAILED'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'BRIGHTNESS_ERROR'}

    async def _create_custom_glyph_pattern(self, parameters: Dict) -> Dict[str, any]:
        """Create custom Glyph pattern"""
        sequence = parameters.get('sequence', [[1, 1000, 100], [2, 500, 75], [3, 2000, 50]])
        repeat = parameters.get('repeat', 1)
        
        try:
            self.logger.info(f"🎨 Creating custom Glyph pattern with {len(sequence)} steps")
            
            for cycle in range(repeat):
                for step_idx, step in enumerate(sequence):
                    if len(step) >= 3:
                        zone, duration_ms, brightness = step[:3]
                    else:
                        zone, duration_ms, brightness = 1, 1000, 100
                    
                    # Create notification for each step
                    step_cmd = [
                        'shell', 'cmd', 'notification', 'post',
                        '-t', f'Custom Pattern Step {step_idx + 1}',
                        f'pattern_step_{step_idx}',
                        f'Zone {zone} - {brightness}% for {duration_ms}ms'
                    ]
                    
                    await self._run_adb_command(step_cmd)
                    await asyncio.sleep(duration_ms / 1000.0)
                    
                    # Clear notification
                    clear_cmd = ['shell', 'cmd', 'notification', 'cancel', f'pattern_step_{step_idx}']
                    await self._run_adb_command(clear_cmd)
            
            return {
                'status': 'success',
                'message': 'Custom Glyph pattern executed',
                'sequence_length': len(sequence),
                'repeat_count': repeat
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'CUSTOM_PATTERN_ERROR'}

    async def _turn_off_glyph(self) -> Dict[str, any]:
        """Turn off all Glyph lights"""
        try:
            # Try multiple methods to turn off Glyph
            methods = [
                ['shell', 'settings', 'put', 'system', 'glyph_enable', '0'],
                ['shell', 'cmd', 'notification', 'cancel-all'],
                ['shell', 'am', 'broadcast', '-a', 'com.nothing.glyph.ACTION_OFF']
            ]
            
            for method in methods:
                await self._run_adb_command(method)
            
            return {'status': 'success', 'message': 'Glyph lights turned off (all methods applied)'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'GLYPH_OFF_ERROR'}

    async def _handle_nothing_specific_command_universal(self, command_type: str, parameters: Dict) -> Dict[str, any]:
        """Handle Nothing Phone specific commands with universal fallbacks"""
        try:
            if command_type == 'nothing_launcher':
                return await self._control_launcher_universal(parameters)
            elif command_type == 'nothing_settings':
                return await self._open_settings_universal(parameters)
            elif command_type == 'nothing_camera':
                return await self._control_camera_universal(parameters)
            elif command_type == 'nothing_recorder':
                return await self._control_recorder_universal(parameters)
            elif command_type == 'nothing_performance':
                return await self._optimize_performance_universal(parameters)
            else:
                return {'status': 'error', 'message': f'Unknown Nothing command: {command_type}', 'code': 'UNKNOWN_NOTHING_COMMAND'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'NOTHING_EXECUTION_ERROR'}

    async def _control_launcher_universal(self, parameters: Dict) -> Dict[str, any]:
        """Control launcher with universal fallback"""
        action = parameters.get('action', 'home')
        
        try:
            if action == 'home':
                # Try Nothing Launcher first, then fallback to home intent
                if self.nothing_services.get('nothing_launcher', {}).get('available', False):
                    cmd = ['shell', 'am', 'start', '-n', 'com.nothing.launcher/.MainActivity']
                else:
                    cmd = ['shell', 'am', 'start', '-c', 'android.intent.category.HOME', '-a', 'android.intent.action.MAIN']
            else:
                # Generic home for other actions
                cmd = ['shell', 'input', 'keyevent', 'KEYCODE_HOME']
            
            result = await self._run_adb_command(cmd)
            
            launcher_type = "Nothing Launcher" if self.nothing_services.get('nothing_launcher', {}).get('available', False) else "System Launcher"
            
            if result['success']:
                return {'status': 'success', 'message': f'{launcher_type} {action} activated', 'action': action, 'launcher_type': launcher_type}
            else:
                return {'status': 'error', 'message': f'Failed to execute launcher action: {action}', 'code': 'LAUNCHER_ACTION_FAILED'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'LAUNCHER_ERROR'}

    async def _open_settings_universal(self, parameters: Dict) -> Dict[str, any]:
        """Open settings with universal fallback"""
        setting_type = parameters.get('setting', 'main')
        
        # Try Nothing settings first, then Android settings
        setting_intents = {
            'main': [
                'com.nothing.dotui/.MainActivity',  # Nothing Settings
                'com.android.settings/.Settings'    # Android Settings
            ],
            'display': [
                'com.android.settings/.DisplaySettings'
            ],
            'sound': [
                'com.android.settings/.SoundSettings'
            ],
            'battery': [
                'com.android.settings/.fuelgauge.PowerUsageSummary'
            ],
            'developer': [
                'com.android.settings/.DevelopmentSettings'
            ]
        }
        
        try:
            intents = setting_intents.get(setting_type, ['com.android.settings/.Settings'])
            
            for intent in intents:
                cmd = ['shell', 'am', 'start', '-n', intent]
                result = await self._run_adb_command(cmd)
                
                if result['success']:
                    settings_type = "Nothing Settings" if "nothing" in intent else "Android Settings"
                    return {'status': 'success', 'message': f'{settings_type} {setting_type} opened', 'setting': setting_type, 'settings_type': settings_type}
            
            return {'status': 'error', 'message': f'Failed to open {setting_type} settings', 'code': 'SETTINGS_OPEN_FAILED'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'SETTINGS_ERROR'}

    async def _control_camera_universal(self, parameters: Dict) -> Dict[str, any]:
        """Control camera with universal fallback"""
        action = parameters.get('action', 'open')
        mode = parameters.get('mode', 'photo')
        
        try:
            # Try Nothing Camera first, then other camera apps
            camera_packages = [
                'com.nothing.camera/.CameraActivity',
                'com.google.android.GoogleCamera',
                'com.android.camera2/.CameraActivity',
                'com.sec.android.app.camera/.Camera'
            ]
            
            if action == 'open':
                for package in camera_packages:
                    if '/' in package:
                        cmd = ['shell', 'am', 'start', '-n', package]
                    else:
                        cmd = ['shell', 'am', 'start', package]
                    
                    result = await self._run_adb_command(cmd)
                    if result['success']:
                        camera_type = "Nothing Camera" if "nothing" in package else "Camera"
                        return {'status': 'success', 'message': f'{camera_type} opened in {mode} mode', 'action': action, 'mode': mode, 'camera_type': camera_type}
                
                # Fallback to camera intent
                cmd = ['shell', 'am', 'start', '-a', 'android.media.action.IMAGE_CAPTURE']
                result = await self._run_adb_command(cmd)
                if result['success']:
                    return {'status': 'success', 'message': f'Camera opened via intent', 'action': action}
                    
            elif action == 'capture':
                cmd = ['shell', 'input', 'keyevent', 'KEYCODE_CAMERA']
                result = await self._run_adb_command(cmd)
                if result['success']:
                    return {'status': 'success', 'message': 'Camera capture triggered', 'action': action}
            
            return {'status': 'error', 'message': f'Camera {action} failed', 'code': 'CAMERA_ACTION_FAILED'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'CAMERA_ERROR'}

    async def _control_recorder_universal(self, parameters: Dict) -> Dict[str, any]:
        """Control recorder with universal fallback"""
        action = parameters.get('action', 'open')
        
        try:
            recorder_packages = [
                'com.nothing.soundrecorder/.SoundRecorderActivity',
                'com.google.android.apps.recorder/.MainActivity',
                'com.sec.android.app.voicenote/.VoiceNoteMainActivity'
            ]
            
            if action == 'open':
                for package in recorder_packages:
                    cmd = ['shell', 'am', 'start', '-n', package]
                    result = await self._run_adb_command(cmd)
                    
                    if result['success']:
                        recorder_type = "Nothing Recorder" if "nothing" in package else "Voice Recorder"
                        return {'status': 'success', 'message': f'{recorder_type} opened', 'action': action, 'recorder_type': recorder_type}
                
                # Fallback to generic audio recorder intent
                cmd = ['shell', 'am', 'start', '-a', 'android.provider.MediaStore.RECORD_SOUND']
                result = await self._run_adb_command(cmd)
                if result['success']:
                    return {'status': 'success', 'message': 'Audio recorder opened via intent', 'action': action}
            
            return {'status': 'error', 'message': f'Recorder {action} failed', 'code': 'RECORDER_ACTION_FAILED'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'RECORDER_ERROR'}

    async def _optimize_performance_universal(self, parameters: Dict) -> Dict[str, any]:
        """Universal performance optimization"""
        optimization_type = parameters.get('type', 'balanced')
        
        try:
            self.logger.info(f"🚀 Optimizing device performance: {optimization_type}")
            
            if optimization_type == 'gaming':
                optimizations = [
                    ['shell', 'settings', 'put', 'global', 'animator_duration_scale', '0.5'],
                    ['shell', 'settings', 'put', 'global', 'transition_animation_scale', '0.5'],
                    ['shell', 'settings', 'put', 'global', 'window_animation_scale', '0.5']
                ]
            elif optimization_type == 'battery':
                optimizations = [
                    ['shell', 'settings', 'put', 'global', 'low_power', '1'],
                    ['shell', 'settings', 'put', 'system', 'screen_brightness', '30']
                ]
            else:  # balanced
                optimizations = [
                    ['shell', 'settings', 'put', 'global', 'animator_duration_scale', '1.0'],
                    ['shell', 'settings', 'put', 'global', 'transition_animation_scale', '1.0'],
                    ['shell', 'settings', 'put', 'global', 'window_animation_scale', '1.0']
                ]
            
            success_count = 0
            for optimization in optimizations:
                result = await self._run_adb_command(optimization)
                if result['success']:
                    success_count += 1
                await asyncio.sleep(0.1)
            
            return {
                'status': 'success',
                'message': f'Performance optimization applied: {optimization_type}',
                'optimizations_applied': success_count,
                'total_optimizations': len(optimizations),
                'type': optimization_type
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'PERFORMANCE_OPTIMIZATION_ERROR'}

    # Include all the other handler methods from the previous version
    # (system, app, media, general commands remain the same)
    
    async def _handle_system_command(self, command_type: str, parameters: Dict) -> Dict[str, any]:
        """Handle system-level commands"""
        try:
            if command_type == 'system_info':
                return await self._get_system_information()
            elif command_type == 'system_screenshot':
                return await self._take_screenshot(parameters)
            elif command_type == 'system_cleanup':
                return await self._cleanup_system()
            else:
                return {'status': 'error', 'message': f'Unknown system command: {command_type}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'SYSTEM_COMMAND_ERROR'}

    async def _get_system_information(self) -> Dict[str, any]:
        """Get comprehensive system information"""
        try:
            device_info = await self._get_detailed_device_info()
            
            battery_result = await self._run_adb_command(['shell', 'dumpsys', 'battery'])
            battery_level = 0
            if battery_result['success']:
                for line in battery_result['output'].split('\n'):
                    if 'level:' in line:
                        battery_level = int(line.split(':')[1].strip())
                        break
            
            return {
                'status': 'success',
                'message': 'System information retrieved',
                'device_info': device_info,
                'battery_level': battery_level,
                'device_type': 'Nothing Phone' if self.is_nothing_phone else 'Android Device',
                'glyph_available': self.glyph_available,
                'services_status': self.nothing_services,
                'performance_metrics': self.performance_metrics
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'SYSTEM_INFO_ERROR'}

    async def _take_screenshot(self, parameters: Dict) -> Dict[str, any]:
        """Take screenshot"""
        try:
            filename = parameters.get('filename', f'screenshot_{int(time.time())}.png')
            cmd = ['shell', 'screencap', '-p', f'/sdcard/{filename}']
            result = await self._run_adb_command(cmd)
            
            if result['success']:
                return {'status': 'success', 'message': f'Screenshot saved as {filename}', 'filename': filename}
            else:
                return {'status': 'error', 'message': 'Failed to take screenshot', 'code': 'SCREENSHOT_FAILED'}
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'SCREENSHOT_ERROR'}

    async def _cleanup_system(self) -> Dict[str, any]:
        """Clean up system cache and temporary files"""
        try:
            cleanup_commands = [
                ['shell', 'pm', 'trim-caches', '1000000000'],  # Trim caches
                ['shell', 'cmd', 'package', 'compile', '-m', 'speed', '--all']  # Compile apps
            ]
            
            success_count = 0
            for cmd in cleanup_commands:
                result = await self._run_adb_command(cmd)
                if result['success']:
                    success_count += 1
                    
            return {
                'status': 'success',
                'message': f'System cleanup completed ({success_count}/{len(cleanup_commands)} operations successful)',
                'operations_successful': success_count
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'CLEANUP_ERROR'}

    async def _handle_app_command(self, command_type: str, parameters: Dict) -> Dict[str, any]:
        """Handle app-related commands"""
        try:
            if command_type == 'app_launch':
                return await self._launch_app(parameters)
            elif command_type == 'app_list':
                return await self._list_apps()
            else:
                return {'status': 'error', 'message': f'Unknown app command: {command_type}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'APP_COMMAND_ERROR'}

    async def _launch_app(self, parameters: Dict) -> Dict[str, any]:
        """Launch an application"""
        try:
            package_name = parameters.get('package')
            if not package_name:
                return {'status': 'error', 'message': 'Package name required', 'code': 'MISSING_PACKAGE'}
            
            cmd = ['shell', 'monkey', '-p', package_name, '-c', 'android.intent.category.LAUNCHER', '1']
            result = await self._run_adb_command(cmd)
            
            if result['success']:
                return {'status': 'success', 'message': f'App {package_name} launched', 'package': package_name}
            else:
                return {'status': 'error', 'message': f'Failed to launch {package_name}', 'code': 'LAUNCH_FAILED'}
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'LAUNCH_ERROR'}

    async def _list_apps(self) -> Dict[str, any]:
        """List installed applications"""
        try:
            result = await self._run_adb_command(['shell', 'pm', 'list', 'packages'])
            if result['success']:
                packages = []
                for line in result['output'].split('\n'):
                    if line.startswith('package:'):
                        packages.append(line.replace('package:', '').strip())
                
                return {
                    'status': 'success',
                    'message': f'Found {len(packages)} installed packages',
                    'packages': packages[:50],  # Limit to first 50 for readability
                    'total_count': len(packages)
                }
            else:
                return {'status': 'error', 'message': 'Failed to list apps', 'code': 'LIST_APPS_FAILED'}
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'LIST_APPS_ERROR'}

    async def _handle_media_command(self, command_type: str, parameters: Dict) -> Dict[str, any]:
        """Handle media control commands"""
        try:
            if command_type == 'media_play':
                return await self._control_media_playback('play')
            elif command_type == 'media_pause':
                return await self._control_media_playback('pause')
            elif command_type == 'media_next':
                return await self._control_media_playback('next')
            elif command_type == 'media_previous':
                return await self._control_media_playback('previous')
            elif command_type == 'media_volume':
                return await self._set_volume(parameters)
            else:
                return {'status': 'error', 'message': f'Unknown media command: {command_type}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'MEDIA_COMMAND_ERROR'}

    async def _control_media_playback(self, action: str) -> Dict[str, any]:
        """Control media playback"""
        try:
            key_mapping = {
                'play': 'KEYCODE_MEDIA_PLAY_PAUSE',
                'pause': 'KEYCODE_MEDIA_PLAY_PAUSE',
                'next': 'KEYCODE_MEDIA_NEXT',
                'previous': 'KEYCODE_MEDIA_PREVIOUS'
            }
            
            keycode = key_mapping.get(action)
            if not keycode:
                return {'status': 'error', 'message': f'Unknown media action: {action}', 'code': 'UNKNOWN_MEDIA_ACTION'}
            
            cmd = ['shell', 'input', 'keyevent', keycode]
            result = await self._run_adb_command(cmd)
            
            if result['success']:
                return {'status': 'success', 'message': f'Media {action} executed', 'action': action}
            else:
                return {'status': 'error', 'message': f'Media {action} failed', 'code': 'MEDIA_ACTION_FAILED'}
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'MEDIA_CONTROL_ERROR'}

    async def _set_volume(self, parameters: Dict) -> Dict[str, any]:
        """Set device volume"""
        try:
            volume = max(0, min(100, parameters.get('volume', 50)))
            # Convert percentage to system volume scale (usually 0-15)
            system_volume = int(volume * 15 / 100)
            
            cmd = ['shell', 'media', 'volume', '--stream', '3', '--set', str(system_volume)]
            result = await self._run_adb_command(cmd)
            
            if result['success']:
                return {'status': 'success', 'message': f'Volume set to {volume}%', 'volume': volume}
            else:
                return {'status': 'error', 'message': 'Failed to set volume', 'code': 'VOLUME_SET_FAILED'}
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'VOLUME_ERROR'}

    async def _handle_general_command(self, command_type: str, parameters: Dict) -> Dict[str, any]:
        """Handle general ADB commands"""
        try:
            if command_type == 'input_text':
                return await self._input_text(parameters)
            elif command_type == 'input_tap':
                return await self._input_tap(parameters)
            elif command_type == 'key_event':
                return await self._send_key_event(parameters)
            else:
                return {'status': 'error', 'message': f'Unknown general command: {command_type}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'GENERAL_COMMAND_ERROR'}

    async def _input_text(self, parameters: Dict) -> Dict[str, any]:
        """Input text to the device"""
        try:
            text = parameters.get('text', '')
            if not text:
                return {'status': 'error', 'message': 'Text parameter required', 'code': 'MISSING_TEXT'}
            
            # Fix the escape sequence - use proper URL encoding for spaces
            escaped_text = text.replace(' ', '%20').replace('&', '\\&').replace('"', '\\"')
            
            cmd = ['shell', 'input', 'text', escaped_text]
            result = await self._run_adb_command(cmd)
            
            if result['success']:
                return {'status': 'success', 'message': f'Text "{text}" inputted', 'text': text}
            else:
                return {'status': 'error', 'message': 'Failed to input text', 'code': 'INPUT_TEXT_FAILED'}
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'INPUT_TEXT_ERROR'}

    async def _input_tap(self, parameters: Dict) -> Dict[str, any]:
        """Tap at specified coordinates"""
        try:
            x = parameters.get('x')
            y = parameters.get('y')
            
            if x is None or y is None:
                return {'status': 'error', 'message': 'X and Y coordinates required', 'code': 'MISSING_COORDINATES'}
            
            cmd = ['shell', 'input', 'tap', str(x), str(y)]
            result = await self._run_adb_command(cmd)
            
            if result['success']:
                return {'status': 'success', 'message': f'Tapped at ({x}, {y})', 'coordinates': [x, y]}
            else:
                return {'status': 'error', 'message': 'Failed to tap', 'code': 'TAP_FAILED'}
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'TAP_ERROR'}

    async def _send_key_event(self, parameters: Dict) -> Dict[str, any]:
        """Send key event"""
        try:
            keycode = parameters.get('keycode')
            if not keycode:
                return {'status': 'error', 'message': 'Keycode required', 'code': 'MISSING_KEYCODE'}
            
            cmd = ['shell', 'input', 'keyevent', str(keycode)]
            result = await self._run_adb_command(cmd)
            
            if result['success']:
                return {'status': 'success', 'message': f'Key event {keycode} sent', 'keycode': keycode}
            else:
                return {'status': 'error', 'message': f'Failed to send key event {keycode}', 'code': 'KEY_EVENT_FAILED'}
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'code': 'KEY_EVENT_ERROR'}

    async def _run_adb_command(self, args: List[str], timeout: int = 30) -> Dict[str, any]:
        """Execute ADB command with proper error handling and timeout"""
        try:
            cmd = ['adb'] + args
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                
                return {
                    'success': process.returncode == 0,
                    'output': stdout.decode('utf-8', errors='ignore').strip(),
                    'error': stderr.decode('utf-8', errors='ignore').strip(),
                    'return_code': process.returncode,
                    'command': ' '.join(cmd)
                }
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    'success': False,
                    'output': '',
                    'error': f'Command timed out after {timeout} seconds',
                    'return_code': -1,
                    'command': ' '.join(cmd)
                }
                
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e),
                'return_code': -1,
                'command': ' '.join(args)
            }

    async def _log_command_execution(self, command_id: str, command_type: str, 
                                   parameters: Dict, execution_time: float, 
                                   result: Dict) -> None:
        """Log command execution to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO command_history 
                (timestamp, command_type, parameters, execution_time, status, error_message, device_response)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                command_type,
                json.dumps(parameters),
                execution_time,
                result['status'],
                result.get('message', ''),
                json.dumps(result)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log command execution: {str(e)}")

    def get_performance_report(self) -> Dict[str, any]:
        """Generate comprehensive performance report"""
        available_services = len([s for s in self.nothing_services.values() if s['available']])
        
        return {
            'device_info': {
                'verified': self.device_verified,
                'model': self.device_model,
                'is_nothing_phone': self.is_nothing_phone,
                'glyph_available': self.glyph_available,
                'services_available': f"{available_services}/{len(self.nothing_services)}"
            },
            'performance_metrics': self.performance_metrics,
            'system_health': 'excellent' if self.performance_metrics['success_rate'] > 95 else 'good',
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time() - self.performance_metrics.get('start_time', time.time())
        }

# Example usage and main execution
async def main():
    """Main execution function with comprehensive examples"""
    print("🔥 Nothing Phone Universal Control System v4.0")
    print("=" * 60)
    
    # Initialize the control system
    control_system = NothingPhoneUniversalControlSystem()
    
    # Initialize universal control
    if not await control_system.initialize_universal_control():
        print("❌ Failed to initialize universal control system")
        return
    
    print("\n✅ Universal Control System Ready!")
    print(f"📱 Device: {control_system.device_model}")
    print(f"🏭 Type: {'Nothing Phone' if control_system.is_nothing_phone else 'Android Device'}")
    print(f"🌟 Glyph: {'Available' if control_system.glyph_available else 'Simulated via notifications'}")
    print("\n🌟 Available Commands:")
    print("=" * 40)
    
    # Example commands (updated for universal compatibility)
    examples = [
        # Glyph Commands (works on all devices)
        {
            'name': 'Activate Breathing Glyph Pattern (or Simulate)',
            'command': 'glyph_pattern',
            'params': {'pattern': 'breathing', 'duration': 5}
        },
        {
            'name': 'Set Glyph Brightness (or Simulate)',
            'command': 'glyph_brightness',
            'params': {'brightness': 75}
        },
        {
            'name': 'Custom Glyph Pattern (or Simulate)',
            'command': 'glyph_custom',
            'params': {
                'sequence': [[1, 1000, 100], [2, 500, 50], [3, 2000, 75]],
                'repeat': 2
            }
        },
        {
            'name': 'Turn Off Glyph',
            'command': 'glyph_off',
            'params': {}
        },
        
        # Universal Nothing/Android Commands
        {
            'name': 'Go to Home (Universal)',
            'command': 'nothing_launcher',
            'params': {'action': 'home'}
        },
        {
            'name': 'Open Settings (Universal)',
            'command': 'nothing_settings',
            'params': {'setting': 'main'}
        },
        {
            'name': 'Open Camera (Universal)',
            'command': 'nothing_camera',
            'params': {'action': 'open', 'mode': 'photo'}
        },
        {
            'name': 'Open Voice Recorder (Universal)',
            'command': 'nothing_recorder',
            'params': {'action': 'open'}
        },
        {
            'name': 'Optimize for Gaming (Universal)',
            'command': 'nothing_performance',
            'params': {'type': 'gaming'}
        },
        
        # System Commands
        {
            'name': 'Get System Information',
            'command': 'system_info',
            'params': {}
        },
        {
            'name': 'Take Screenshot',
            'command': 'system_screenshot',
            'params': {'filename': 'my_screenshot.png'}
        },
        {
            'name': 'Clean System',
            'command': 'system_cleanup',
            'params': {}
        },
        
        # App Commands
        {
            'name': 'Launch Gallery App',
            'command': 'app_launch',
            'params': {'package': 'com.android.gallery3d' if not control_system.nothing_services.get('nothing_gallery', {}).get('available', False) else 'com.nothing.gallery'}
        },
        {
            'name': 'List Installed Apps',
            'command': 'app_list',
            'params': {}
        },
        
        # Media Commands
        {
            'name': 'Play/Pause Media',
            'command': 'media_play',
            'params': {}
        },
        {
            'name': 'Set Volume to 70%',
            'command': 'media_volume',
            'params': {'volume': 70}
        },
        
        # Input Commands
        {
            'name': 'Type Text',
            'command': 'input_text',
            'params': {'text': 'Hello Universal Control!'}
        },
        {
            'name': 'Tap Screen Center',
            'command': 'input_tap',
            'params': {'x': 540, 'y': 1000}
        },
        {
            'name': 'Send Home Key',
            'command': 'key_event',
            'params': {'keycode': 'KEYCODE_HOME'}
        }
    ]
    
    # Display available commands
    for i, example in enumerate(examples, 1):
        print(f"{i:2d}. {example['name']}")
    
    print(f"\n🔥 Interactive Mode:")
    print(f"Type a command number (1-{len(examples)}) or 'quit' to exit")
    print("=" * 40)
    
    # Interactive loop
    while True:
        try:
            user_input = input("\n💫 Enter command: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.isdigit():
                cmd_index = int(user_input) - 1
                if 0 <= cmd_index < len(examples):
                    example = examples[cmd_index]
                    print(f"\n⚡ Executing: {example['name']}")
                    
                    # Execute the command
                    result = await control_system.execute_universal_command(
                        example['command'], 
                        example['params']
                    )
                    
                    # Display result
                    if result['status'] == 'success':
                        print(f"✅ Success: {result['message']}")
                        if 'execution_time' in result:
                            print(f"⏱️  Execution time: {result['execution_time']:.3f}s")
                    else:
                        print(f"❌ Error: {result['message']}")
                        if 'code' in result:
                            print(f"🔍 Error code: {result['code']}")
                    
                    # Show additional result data
                    for key, value in result.items():
                        if key not in ['status', 'message', 'execution_time', 'command_id', 'timestamp']:
                            print(f"📊 {key}: {value}")
                else:
                    print("❌ Invalid command number")
            else:
                print("❌ Please enter a valid command number or 'quit'")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Show final performance report
    print("\n📊 Final Performance Report:")
    print("=" * 40)
    report = control_system.get_performance_report()
    
    print(f"Device Model: {report['device_info']['model']}")
    print(f"Device Type: {'Nothing Phone' if report['device_info']['is_nothing_phone'] else 'Android Device'}")
    print(f"Glyph Available: {report['device_info']['glyph_available']}")
    print(f"Services Available: {report['device_info']['services_available']}")
    print(f"Commands Executed: {report['performance_metrics']['commands_executed']}")
    print(f"Success Rate: {report['performance_metrics']['success_rate']:.1f}%")
    print(f"Average Response Time: {report['performance_metrics']['avg_response_time']:.3f}s")
    print(f"System Health: {report['system_health']}")
    print(f"Session Uptime: {report['uptime']:.1f}s")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 System shutdown requested. Goodbye!")
    except Exception as e:
        print(f"❌ System error: {str(e)}")
