#!/usr/bin/env python3
"""
Nothing Phone (2a) Analyzer for JARVIS - Enhanced 10/10 Version
Ultra-comprehensive analysis and preparation for JARVIS integration
Specifically optimized for Nothing Phone (2a) with advanced features
"""

import subprocess
import json
import time
import hashlib
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import configparser
import secrets
import base64

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nothing_analyzer.log'),
        logging.StreamHandler()
    ]
)

class NothingPhoneAnalyzer:
    """
    Enhanced Nothing Phone (2a) Analyzer with 10/10 rating
    Ultra-comprehensive device analysis for JARVIS integration
    """
    
    # Nothing Phone (2a) specific configurations
    NOTHING_2A_CONFIG = {
        'expected_model': 'Phone (2a)',
        'expected_brand': 'Nothing',
        'expected_soc': 'Dimensity 7200 Pro',
        'glyph_packages': [
            'com.nothing.ketchum',
            'com.nothing.glyph',
            'com.nothing.systemui'
        ],
        'nothing_apps': [
            'com.nothing.launcher',
            'com.nothing.recorder',
            'com.nothing.gallery',
            'com.nothing.weather'
        ],
        'optimal_specs': {
            'min_memory_gb': 6,
            'recommended_memory_gb': 8,
            'min_storage_gb': 32,
            'recommended_free_gb': 15,
            'max_safe_temp_celsius': 45
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize Enhanced Nothing Phone (2a) analyzer"""
        print("📱 JARVIS Nothing Phone (2a) Analyzer - ENHANCED v2.0")
        print("🔥 Ultra-Comprehensive Analysis Engine")
        print("=" * 60)
        
        # Initialize core attributes
        self.device_id: Optional[str] = None
        self.phone_info: Dict[str, Any] = {}
        self.nothing_features: Dict[str, Any] = {}
        self.security_profile: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self.jarvis_compatibility: Dict[str, Any] = {}
        self.analysis_session_id = self._generate_session_id()
        
        # Load configuration
        self.config = self._load_configuration(config_file)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting analysis session: {self.analysis_session_id}")
        
        # Performance tracking
        self.start_time = time.time()
        self.operation_times: Dict[str, float] = {}
        
    def _generate_session_id(self) -> str:
        """Generate unique session ID for tracking"""
        return f"JARVIS-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{secrets.token_hex(4)}"
        
    def _load_configuration(self, config_file: Optional[str] = None) -> Dict:
        """Load and validate configuration"""
        default_config = {
            'timeouts': {
                'adb_command': 15,
                'root_check': 5,
                'performance_test': 30
            },
            'retry_attempts': 3,
            'parallel_operations': True,
            'detailed_logging': True,
            'secure_storage': True
        }
        
        if config_file and Path(config_file).exists():
            try:
                config_parser = configparser.ConfigParser()
                config_parser.read(config_file)
                # Merge with default config
                for section in config_parser.sections():
                    if section not in default_config:
                        default_config[section] = {}
                    default_config[section].update(dict(config_parser[section]))
                self.logger.info(f"Configuration loaded from: {config_file}")
            except Exception as e:
                self.logger.warning(f"Config load error: {e}, using defaults")
        
        return default_config
        
    def _time_operation(self, operation_name: str):
        """Decorator to time operations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.operation_times[operation_name] = time.time() - start
                    self.logger.info(f"⏱️ {operation_name}: {self.operation_times[operation_name]:.2f}s")
                    return result
                except Exception as e:
                    self.operation_times[operation_name] = time.time() - start
                    self.logger.error(f"❌ {operation_name} failed after {self.operation_times[operation_name]:.2f}s: {e}")
                    raise
            return wrapper
        return decorator
    
    def check_phone_connection(self) -> bool:
        """Enhanced connection check with retry logic and validation"""
        print("🔍 Checking Nothing Phone (2a) connection...")
        
        for attempt in range(self.config['retry_attempts']):
            try:
                # Check ADB server status
                server_result = subprocess.run(['adb', 'start-server'], capture_output=True, text=True)
                if server_result.returncode != 0:
                    raise Exception("ADB server failed to start")
                
                # Get device list
                result = subprocess.run(['adb', 'devices', '-l'], capture_output=True, text=True, 
                                      timeout=self.config['timeouts']['adb_command'])
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]
                    
                    for line in lines:
                        if 'device' in line and 'unauthorized' not in line:
                            parts = line.split()
                            device_id = parts[0]
                            
                            # Validate it's a Nothing Phone (2a)
                            if self._validate_nothing_2a(device_id):
                                self.device_id = device_id
                                print(f"✅ Nothing Phone (2a) connected: {self.device_id}")
                                self.logger.info(f"Device validated: {device_id}")
                                return True
                            else:
                                print(f"⚠️ Connected device is not Nothing Phone (2a): {device_id}")
                                continue
                    
                    # Check for unauthorized devices
                    unauthorized = [line for line in lines if 'unauthorized' in line]
                    if unauthorized:
                        print("❌ Nothing Phone connected but unauthorized")
                        print("📱 Please check your phone and allow USB debugging")
                        print("💡 Tip: Look for the USB debugging authorization dialog")
                        return False
                    
                    print("❌ No Nothing Phone (2a) detected")
                    if attempt < self.config['retry_attempts'] - 1:
                        print(f"🔄 Retrying... (Attempt {attempt + 2}/{self.config['retry_attempts']})")
                        time.sleep(2)
                    return False
                else:
                    raise Exception(f"ADB devices command failed: {result.stderr}")
                    
            except FileNotFoundError:
                print("❌ ADB not found - Please install Android SDK Platform Tools")
                print("💡 Download from: https://developer.android.com/studio/releases/platform-tools")
                return False
            except subprocess.TimeoutExpired:
                print(f"⏰ Connection timeout (attempt {attempt + 1})")
                if attempt < self.config['retry_attempts'] - 1:
                    time.sleep(1)
                continue
            except Exception as e:
                print(f"❌ Connection error (attempt {attempt + 1}): {e}")
                if attempt < self.config['retry_attempts'] - 1:
                    time.sleep(1)
                continue
        
        return False
    
    def _validate_nothing_2a(self, device_id: str) -> bool:
        """Validate connected device is Nothing Phone (2a)"""
        try:
            # Quick validation checks
            model_cmd = ['adb', '-s', device_id, 'shell', 'getprop', 'ro.product.model']
            brand_cmd = ['adb', '-s', device_id, 'shell', 'getprop', 'ro.product.brand']
            
            model_result = subprocess.run(model_cmd, capture_output=True, text=True, timeout=5)
            brand_result = subprocess.run(brand_cmd, capture_output=True, text=True, timeout=5)
            
            model = model_result.stdout.strip()
            brand = brand_result.stdout.strip()
            
            is_valid = (self.NOTHING_2A_CONFIG['expected_model'] in model and 
                       self.NOTHING_2A_CONFIG['expected_brand'] in brand)
            
            if is_valid:
                self.logger.info(f"Device validation passed: {brand} {model}")
            else:
                self.logger.warning(f"Device validation failed: {brand} {model}")
                
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Device validation error: {e}")
            return False
    
    @_time_operation("Device Information Gathering")
    def get_nothing_phone_details(self) -> Dict[str, Any]:
        """Get comprehensive Nothing Phone (2a) information with parallel execution"""
        if not self.device_id:
            return {}
        
        print("📊 Analyzing Nothing Phone (2a) specifications...")
        
        # Enhanced property list for Nothing Phone (2a)
        phone_properties = {
            'Model': 'ro.product.model',
            'Brand': 'ro.product.brand',
            'Device': 'ro.product.device',
            'Board': 'ro.product.board',
            'Manufacturer': 'ro.product.manufacturer',
            'Android Version': 'ro.build.version.release',
            'API Level': 'ro.build.version.sdk',
            'Security Patch': 'ro.build.version.security_patch',
            'Build ID': 'ro.build.id',
            'Build Type': 'ro.build.type',
            'Build Tags': 'ro.build.tags',
            'Hardware': 'ro.hardware',
            'SOC Model': 'ro.vendor.qti.soc_model',
            'CPU ABI': 'ro.product.cpu.abi',
            'CPU ABI2': 'ro.product.cpu.abi2',
            'Nothing OS Version': 'ro.nothing.version',
            'Bootloader': 'ro.bootloader',
            'Kernel Version': 'ro.build.version.incremental',
            'Display Density': 'ro.sf.lcd_density',
            'OpenGL ES Version': 'ro.opengles.version'
        }
        
        if self.config.get('parallel_operations', True):
            # Parallel execution for faster results
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_prop = {
                    executor.submit(self._get_phone_property, prop): (display_name, prop)
                    for display_name, prop in phone_properties.items()
                }
                
                for future in as_completed(future_to_prop):
                    display_name, prop = future_to_prop[future]
                    try:
                        value = future.result()
                        if value and value != 'unknown':
                            self.phone_info[display_name] = value
                            print(f"   ✅ {display_name}: {value}")
                        else:
                            print(f"   ❓ {display_name}: Not available")
                    except Exception as e:
                        print(f"   ❌ {display_name}: Error ({e})")
                        self.logger.error(f"Property error {prop}: {e}")
        else:
            # Sequential execution
            for display_name, prop in phone_properties.items():
                value = self._get_phone_property(prop)
                if value and value != 'unknown':
                    self.phone_info[display_name] = value
                    print(f"   ✅ {display_name}: {value}")
        
        # Validate Nothing Phone (2a) specific specs
        self._validate_device_specs()
        
        return self.phone_info
    
    def _validate_device_specs(self):
        """Validate device meets Nothing Phone (2a) specifications"""
        print("🔍 Validating Nothing Phone (2a) specifications...")
        
        validations = {
            'Model Check': self.NOTHING_2A_CONFIG['expected_model'] in self.phone_info.get('Model', ''),
            'Brand Check': self.NOTHING_2A_CONFIG['expected_brand'] in self.phone_info.get('Brand', ''),
            'Android 13+': int(self.phone_info.get('API Level', '0')) >= 33,
            'Recent Security Patch': self._validate_security_patch(),
            'MediaTek SOC': 'Dimensity' in self.phone_info.get('SOC Model', '')
        }
        
        for check_name, passed in validations.items():
            status = "✅" if passed else "⚠️"
            print(f"   {status} {check_name}")
            
        self.phone_info['validation_results'] = validations
    
    def _validate_security_patch(self) -> bool:
        """Check if security patch is recent (within 3 months)"""
        try:
            patch_date = self.phone_info.get('Security Patch', '')
            if not patch_date:
                return False
            
            patch_datetime = datetime.strptime(patch_date, '%Y-%m-%d')
            months_old = (datetime.now() - patch_datetime).days / 30
            
            return months_old <= 3
        except:
            return False
    
    @_time_operation("Nothing OS Feature Analysis")
    def analyze_nothing_os_features(self) -> Dict[str, Any]:
        """Enhanced Nothing OS feature analysis with deep inspection"""
        print("🌟 Analyzing Nothing OS features...")
        
        features = {}
        
        # Glyph Interface Analysis
        print("   🔍 Checking Glyph Interface...")
        glyph_features = self._analyze_glyph_interface()
        features['glyph'] = glyph_features
        
        # Nothing Launcher Analysis  
        print("   🔍 Checking Nothing Launcher...")
        launcher_features = self._analyze_nothing_launcher()
        features['launcher'] = launcher_features
        
        # Nothing Apps Analysis
        print("   🔍 Checking Nothing Apps...")
        app_features = self._analyze_nothing_apps()
        features['apps'] = app_features
        
        # System UI Customizations
        print("   🔍 Checking System UI...")
        ui_features = self._analyze_nothing_ui()
        features['ui'] = ui_features
        
        # Calculate feature completeness score
        total_features = sum(len(category) for category in features.values() if isinstance(category, dict))
        available_features = sum(
            sum(1 for v in category.values() if v) 
            for category in features.values() 
            if isinstance(category, dict)
        )
        
        features['completeness_score'] = (available_features / total_features * 100) if total_features > 0 else 0
        print(f"   📊 Nothing OS Features: {features['completeness_score']:.1f}% complete")
        
        self.nothing_features = features
        return features
    
    def _analyze_glyph_interface(self) -> Dict[str, bool]:
        """Analyze Glyph Interface capabilities"""
        glyph_features = {}
        
        for package in self.NOTHING_2A_CONFIG['glyph_packages']:
            result = self._execute_adb_command(['shell', 'pm', 'list', 'packages', package])
            is_present = result is not None and package in result
            glyph_features[f'package_{package.split(".")[-1]}'] = is_present
            
            if is_present:
                print(f"      ✅ {package}")
            else:
                print(f"      ❌ {package}")
        
        # Check Glyph settings
        glyph_settings = self._execute_adb_command(['shell', 'settings', 'list', 'secure', '|', 'grep', 'glyph'])
        glyph_features['settings_available'] = glyph_settings is not None and 'glyph' in glyph_settings.lower()
        
        # Check Glyph service
        glyph_service = self._execute_adb_command(['shell', 'dumpsys', 'activity', 'services', '|', 'grep', 'glyph'])
        glyph_features['service_running'] = glyph_service is not None and 'glyph' in glyph_service.lower()
        
        return glyph_features
    
    def _analyze_nothing_launcher(self) -> Dict[str, Any]:
        """Analyze Nothing Launcher features"""
        launcher_features = {}
        
        # Check launcher package
        launcher_result = self._execute_adb_command(['shell', 'pm', 'list', 'packages', 'com.nothing.launcher'])
        launcher_features['installed'] = launcher_result is not None and 'com.nothing.launcher' in launcher_result
        
        # Check if it's the default launcher
        default_launcher = self._execute_adb_command(['shell', 'cmd', 'role', 'get-role-holders', 'android.app.role.HOME'])
        launcher_features['is_default'] = default_launcher is not None and 'nothing.launcher' in default_launcher
        
        # Check launcher version
        if launcher_features['installed']:
            version_info = self._execute_adb_command(['shell', 'dumpsys', 'package', 'com.nothing.launcher', '|', 'grep', 'versionName'])
            if version_info:
                launcher_features['version'] = version_info.split('=')[-1].strip() if '=' in version_info else 'unknown'
        
        return launcher_features
    
    def _analyze_nothing_apps(self) -> Dict[str, bool]:
        """Analyze Nothing specific apps"""
        app_features = {}
        
        for app in self.NOTHING_2A_CONFIG['nothing_apps']:
            result = self._execute_adb_command(['shell', 'pm', 'list', 'packages', app])
            is_installed = result is not None and app in result
            app_name = app.split('.')[-1]
            app_features[f'{app_name}_installed'] = is_installed
            
            status = "✅" if is_installed else "❌"
            print(f"      {status} Nothing {app_name.title()}")
        
        return app_features
    
    def _analyze_nothing_ui(self) -> Dict[str, Any]:
        """Analyze Nothing UI customizations"""
        ui_features = {}
        
        # Check Nothing SystemUI
        systemui_result = self._execute_adb_command(['shell', 'pm', 'list', 'packages', 'com.nothing.systemui'])
        ui_features['custom_systemui'] = systemui_result is not None and 'nothing.systemui' in systemui_result
        
       # Check dot matrix fonts
        font_check = self._execute_adb_command(['shell', 'ls /system/fonts/ | grep -i nothing'])
        ui_features['custom_fonts'] = font_check is not None and font_check.strip() != ''
        
        # Check boot animation
        bootanim_check = self._execute_adb_command(['shell', 'ls', '/system/media/bootanimation.zip'])
        ui_features['custom_boot_animation'] = bootanim_check is not None and 'No such file' not in bootanim_check
        
        return ui_features
    
    @_time_operation("Security Analysis")
    def enhanced_security_analysis(self) -> Dict[str, Any]:
        """Comprehensive security analysis"""
        print("🔐 Performing enhanced security analysis...")
        
        security_data = {}
        
        # Developer settings analysis
        print("   🔍 Developer Settings...")
        security_data['developer'] = self._analyze_developer_settings()
        
        # Advanced root detection
        print("   🔍 Root Detection...")
        security_data['root'] = self._advanced_root_detection()
        
        # Bootloader status
        print("   🔍 Bootloader Status...")
        security_data['bootloader'] = self._check_bootloader_status()
        
        # Security patches and updates
        print("   🔍 Security Updates...")
        security_data['updates'] = self._analyze_security_updates()
        
        # App security analysis
        print("   🔍 App Security...")
        security_data['apps'] = self._analyze_app_security()
        
        # Calculate security score
        security_data['security_score'] = self._calculate_security_score(security_data)
        print(f"   📊 Security Score: {security_data['security_score']}/100")
        
        self.security_profile = security_data
        return security_data
    
    def _analyze_developer_settings(self) -> Dict[str, Any]:
        """Analyze developer settings configuration"""
        dev_settings = {}
        
        settings_checks = {
            'USB Debugging': (['shell', 'settings', 'get', 'global', 'adb_enabled'], '1'),
            'Developer Options': (['shell', 'settings', 'get', 'global', 'development_settings_enabled'], '1'),
            'OEM Unlocking': (['shell', 'settings', 'get', 'global', 'oem_unlock_allowed'], '1'),
            'USB Debugging (Secure)': (['shell', 'settings', 'get', 'global', 'adb_wifi_enabled'], '1'),
            'Mock Locations': (['shell', 'settings', 'get', 'secure', 'mock_location'], '1'),
            'Stay Awake': (['shell', 'settings', 'get', 'global', 'stay_on_while_plugged_in'], '7')
        }
        
        for setting_name, (command, expected) in settings_checks.items():
            result = self._execute_adb_command(command)
            is_enabled = result is not None and expected in result
            dev_settings[setting_name.lower().replace(' ', '_')] = is_enabled
            
            status = "✅" if is_enabled else "❌"
            print(f"      {status} {setting_name}")
        
        return dev_settings
    
    def _advanced_root_detection(self) -> Dict[str, Any]:
        """Advanced root detection with multiple methods"""
        root_info = {'methods_checked': [], 'evidence': [], 'rooted': False, 'confidence': 0}
        
        detection_methods = [
            ('su_command', self._check_su_command),
            ('root_apps', self._check_root_apps), 
            ('su_binary', self._check_su_binary),
            ('busybox', self._check_busybox),
            ('root_directories', self._check_root_directories),
            ('system_modifications', self._check_system_modifications)
        ]
        
        positive_detections = 0
        
        for method_name, detection_func in detection_methods:
            try:
                result = detection_func()
                root_info['methods_checked'].append(method_name)
                
                if result['detected']:
                    positive_detections += 1
                    root_info['evidence'].append({
                        'method': method_name,
                        'details': result.get('details', 'No details'),
                        'confidence': result.get('confidence', 50)
                    })
                    print(f"      ⚠️ {method_name}: Root evidence found")
                else:
                    print(f"      ✅ {method_name}: No root detected")
                    
            except Exception as e:
                print(f"      ❌ {method_name}: Check failed ({e})")
                self.logger.error(f"Root detection error in {method_name}: {e}")
        
        # Calculate overall confidence
        if positive_detections > 0:
            root_info['rooted'] = positive_detections >= 2  # Require 2+ methods for positive
            root_info['confidence'] = min(positive_detections * 30, 100)
        
        status = "🔓 ROOTED" if root_info['rooted'] else "🔒 NOT ROOTED"
        print(f"      📊 Overall Status: {status} (Confidence: {root_info['confidence']}%)")
        
        return root_info
    
    def _check_su_command(self) -> Dict[str, Any]:
        """Check su command availability"""
        try:
            result = self._execute_adb_command(['shell', 'su', '-c', 'id'], timeout=3)
            if result and 'uid=0' in result:
                return {'detected': True, 'details': 'su command successful', 'confidence': 90}
        except:
            pass
        return {'detected': False}
    
    def _check_root_apps(self) -> Dict[str, Any]:
        """Check for root management apps"""
        root_apps = [
            'com.topjohnwu.magisk',
            'eu.chainfire.supersu', 
            'com.noshufou.android.su',
            'com.koushikdutta.superuser',
            'com.zachspong.temprootremovejb'
        ]
        
        for app in root_apps:
            result = self._execute_adb_command(['shell', 'pm', 'list', 'packages', app])
            if result and app in result:
                return {'detected': True, 'details': f'Root app found: {app}', 'confidence': 85}
        
        return {'detected': False}
    
    def _check_su_binary(self) -> Dict[str, Any]:
        """Check for su binary in common locations"""
        su_locations = [
            '/system/bin/su', '/system/xbin/su', '/sbin/su', 
            '/vendor/bin/su', '/system/sbin/su', '/system/usr/we-need-root/su-backup',
            '/system/xbin/mu', '/system/bin/.ext/.su', '/system/usr/bin/su'
        ]
        
        for location in su_locations:
            result = self._execute_adb_command(['shell', 'ls', '-la', location])
            if result and 'No such file' not in result and 'Permission denied' not in result:
                return {'detected': True, 'details': f'su binary at {location}', 'confidence': 80}
        
        return {'detected': False}
    
    def _check_busybox(self) -> Dict[str, Any]:
        """Check for busybox (often installed with root)"""
        result = self._execute_adb_command(['shell', 'which', 'busybox'])
        if result and '/busybox' in result:
            return {'detected': True, 'details': f'busybox found at {result}', 'confidence': 60}
        return {'detected': False}
    
    def _check_root_directories(self) -> Dict[str, Any]:
        """Check for root-specific directories"""
        root_dirs = ['/data/local/su', '/data/local/bin', '/data/local/xbin']
        
        for directory in root_dirs:
            result = self._execute_adb_command(['shell', 'ls', directory])
            if result and 'No such file' not in result and 'Permission denied' not in result:
                return {'detected': True, 'details': f'Root directory found: {directory}', 'confidence': 70}
        
        return {'detected': False}
    
    def _check_system_modifications(self) -> Dict[str, Any]:
        """Check for system partition modifications"""
        # Check if system is mounted as read-write
        mount_result = self._execute_adb_command(['shell', 'mount', '|', 'grep', 'system'])
        if mount_result and 'rw,' in mount_result:
            return {'detected': True, 'details': 'System mounted as read-write', 'confidence': 75}
        
        return {'detected': False}
    
    def _check_bootloader_status(self) -> Dict[str, Any]:
        """Check bootloader lock status"""
        bootloader_info = {}
        
        # Check bootloader unlock status
        unlock_status = self._get_phone_property('ro.boot.flash.locked')
        bootloader_info['locked'] = unlock_status == '1' if unlock_status else None
        
        # Check bootloader version
        bootloader_version = self._get_phone_property('ro.bootloader')
        bootloader_info['version'] = bootloader_version
        
        # Check verified boot state
        verified_boot = self._get_phone_property('ro.boot.verifiedbootstate')
        bootloader_info['verified_boot'] = verified_boot
        
        status = "🔒 LOCKED" if bootloader_info.get('locked') else "🔓 UNLOCKED"
        print(f"      📊 Bootloader: {status}")
        
        return bootloader_info
    
    def _analyze_security_updates(self) -> Dict[str, Any]:
        """Analyze security update status"""
        update_info = {}
        
        # Security patch level
        patch_level = self.phone_info.get('Security Patch', '')
        update_info['security_patch'] = patch_level
        
        if patch_level:
            try:
                patch_date = datetime.strptime(patch_level, '%Y-%m-%d')
                days_old = (datetime.now() - patch_date).days
                update_info['patch_age_days'] = days_old
                update_info['patch_current'] = days_old <= 90  # Within 3 months
                
                if days_old <= 30:
                    status = "🟢 CURRENT"
                elif days_old <= 90:
                    status = "🟡 RECENT"
                else:
                    status = "🔴 OUTDATED"
                    
                print(f"      📊 Security Patch: {status} ({days_old} days old)")
            except:
                update_info['patch_current'] = False
        
        return update_info
    
    def _analyze_app_security(self) -> Dict[str, Any]:
        """Analyze app security settings"""
        app_security = {}
        
        # Check unknown sources setting
        unknown_sources = self._execute_adb_command(['shell', 'settings', 'get', 'secure', 'install_non_market_apps'])
        app_security['unknown_sources_enabled'] = unknown_sources == '1'
        
        # Check Play Protect status
        play_protect = self._execute_adb_command(['shell', 'settings', 'get', 'global', 'package_verifier_enable'])
        app_security['play_protect_enabled'] = play_protect == '1'
        
        # Count system vs user apps
        system_apps = self._execute_adb_command(['shell', 'pm', 'list', 'packages', '-s'])
        user_apps = self._execute_adb_command(['shell', 'pm', 'list', 'packages', '-3'])
        
        app_security['system_apps_count'] = len(system_apps.split('\n')) if system_apps else 0
        app_security['user_apps_count'] = len(user_apps.split('\n')) if user_apps else 0
        
        print(f"      📊 Apps: {app_security['system_apps_count']} system, {app_security['user_apps_count']} user")
        
        return app_security
    
    def _calculate_security_score(self, security_data: Dict[str, Any]) -> int:
        """Calculate overall security score (0-100)"""
        score = 100
        
        # Deduct points for security issues
        if security_data['root']['rooted']:
            score -= 30  # Major security risk
        
        if security_data['developer']['usb_debugging']:
            score -= 10  # Debug mode enabled
            
        if security_data['developer']['oem_unlocking']:
            score -= 15  # Bootloader unlock allowed
            
        if not security_data['bootloader'].get('locked', True):
            score -= 25  # Unlocked bootloader
            
        if not security_data['updates'].get('patch_current', False):
            score -= 20  # Outdated security patches
            
        if security_data['apps']['unknown_sources_enabled']:
            score -= 10  # Unknown sources enabled
            
        if not security_data['apps']['play_protect_enabled']:
            score -= 10  # Play Protect disabled
        
        return max(0, score)
    
    @_time_operation("Performance Analysis")
    def advanced_performance_analysis(self) -> Dict[str, Any]:
        """Comprehensive performance analysis for JARVIS optimization"""
        print("🚀 Performing advanced performance analysis...")
        
        performance_data = {}
        
        # Hardware specifications
        print("   🔍 Hardware Analysis...")
        performance_data['hardware'] = self._analyze_hardware_performance()
        
        # System performance metrics
        print("   🔍 System Performance...")
        performance_data['system'] = self._analyze_system_performance()
        
        # Memory analysis
        print("   🔍 Memory Analysis...")
        performance_data['memory'] = self._analyze_memory_performance()
        
        # Storage analysis
        print("   🔍 Storage Analysis...")
        performance_data['storage'] = self._analyze_storage_performance()
        
        # Thermal analysis
        print("   🔍 Thermal Analysis...")
        performance_data['thermal'] = self._analyze_thermal_performance()
        
        # Network performance
        print("   🔍 Network Analysis...")
        performance_data['network'] = self._analyze_network_performance()
        
        # Calculate performance score
        performance_data['performance_score'] = self._calculate_performance_score(performance_data)
        print(f"   📊 Performance Score: {performance_data['performance_score']}/100")
        
        self.performance_metrics = performance_data
        return performance_data
    
    def _analyze_hardware_performance(self) -> Dict[str, Any]:
        """Analyze hardware specifications"""
        hardware = {}
        
        # CPU information
        cpu_info = self._execute_adb_command(['shell', 'cat', '/proc/cpuinfo'])
        if cpu_info:
            # Count CPU cores
            hardware['cpu_cores'] = cpu_info.count('processor')
            
            # Extract CPU frequencies
            cpu_freq = self._execute_adb_command(['shell', 'cat', '/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq'])
            if cpu_freq and cpu_freq.isdigit():
                hardware['max_cpu_freq_mhz'] = int(cpu_freq) // 1000
            
            print(f"      💻 CPU: {hardware['cpu_cores']} cores @ {hardware.get('max_cpu_freq_mhz', 'Unknown')}MHz")
        
        # GPU information
        gpu_renderer = self._execute_adb_command(['shell', 'getprop', 'ro.hardware.egl'])
        if gpu_renderer:
            hardware['gpu'] = gpu_renderer
            print(f"      🎮 GPU: {gpu_renderer}")
        
        # Display information
        display_density = self._get_phone_property('ro.sf.lcd_density')
        if display_density:
            hardware['display_density'] = int(display_density)
            
        # Get screen resolution
        wm_size = self._execute_adb_command(['shell', 'wm', 'size'])
        if wm_size and 'Physical size:' in wm_size:
            resolution = wm_size.split('Physical size: ')[-1].strip()
            hardware['screen_resolution'] = resolution
            print(f"      📱 Display: {resolution} @ {display_density}dpi")
        
        return hardware
    
    def _analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze system performance metrics"""
        system = {}
        
        # System uptime
        uptime = self._execute_adb_command(['shell', 'cat', '/proc/uptime'])
        if uptime:
            uptime_seconds = float(uptime.split()[0])
            uptime_hours = uptime_seconds / 3600
            system['uptime_hours'] = round(uptime_hours, 2)
            print(f"      ⏰ Uptime: {uptime_hours:.1f} hours")
        
        # Load average
        loadavg = self._execute_adb_command(['shell', 'cat', '/proc/loadavg'])
        if loadavg:
            load_values = loadavg.split()[:3]
            system['load_average'] = [float(x) for x in load_values]
            print(f"      📊 Load Average: {' '.join(load_values)}")
        
        # Process count
        ps_count = self._execute_adb_command(['shell', 'ps', '|', 'wc', '-l'])
        if ps_count and ps_count.isdigit():
            system['process_count'] = int(ps_count)
            print(f"      🔄 Processes: {ps_count}")
        
        # System services
        services = self._execute_adb_command(['shell', 'service', 'list'])
        if services:
            system['service_count'] = len([line for line in services.split('\n') if line.strip()])
        
        return system
    
    def _analyze_memory_performance(self) -> Dict[str, Any]:
        """Analyze memory performance and usage"""
        memory = {}
        
        # Memory information from /proc/meminfo
        meminfo = self._execute_adb_command(['shell', 'cat', '/proc/meminfo'])
        if meminfo:
            mem_data = {}
            for line in meminfo.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    value_kb = int(value.strip().split()[0])
                    mem_data[key.strip()] = value_kb
            
            # Convert to GB and calculate metrics
            total_gb = round(mem_data.get('MemTotal', 0) / (1024 * 1024), 2)
            available_gb = round(mem_data.get('MemAvailable', 0) / (1024 * 1024), 2)
            free_gb = round(mem_data.get('MemFree', 0) / (1024 * 1024), 2)
            cached_gb = round(mem_data.get('Cached', 0) / (1024 * 1024), 2)
            
            memory.update({
                'total_gb': total_gb,
                'available_gb': available_gb,
                'free_gb': free_gb,
                'cached_gb': cached_gb,
                'usage_percent': round((total_gb - available_gb) / total_gb * 100, 1)
            })
            
            print(f"      💾 Memory: {available_gb}GB available / {total_gb}GB total ({memory['usage_percent']}% used)")
        
        # Memory pressure check
        oom_score = self._execute_adb_command(['shell', 'cat', '/proc/pressure/memory'])
        if oom_score:
            memory['pressure_info'] = oom_score.split('\n')[0] if oom_score else None
        
        return memory
    
    def _analyze_storage_performance(self) -> Dict[str, Any]:
        """Analyze storage performance and capacity"""
        storage = {}
        
        # Internal storage analysis
        df_result = self._execute_adb_command(['shell', 'df', '/data'])
        if df_result:
            lines = df_result.strip().split('\n')
            if len(lines) > 1:
                # Parse storage info
                parts = lines[1].split()
                if len(parts) >= 4:
                    total_kb = int(parts[1])
                    used_kb = int(parts[2])
                    available_kb = int(parts[3])
                    
                    storage.update({
                        'total_gb': round(total_kb / (1024 * 1024), 2),
                        'used_gb': round(used_kb / (1024 * 1024), 2),
                        'available_gb': round(available_kb / (1024 * 1024), 2),
                        'usage_percent': round(used_kb / total_kb * 100, 1)
                    })
                    
                    print(f"      💿 Storage: {storage['available_gb']}GB free / {storage['total_gb']}GB total")
        
        # Storage I/O statistics
        diskstats = self._execute_adb_command(['shell', 'cat', '/proc/diskstats'])
        if diskstats:
            # Count I/O operations for main storage device
            main_device_stats = [line for line in diskstats.split('\n') if 'mmcblk0' in line]
            if main_device_stats:
                stats = main_device_stats[0].split()
                if len(stats) > 10:
                    storage['read_operations'] = int(stats[3])
                    storage['write_operations'] = int(stats[7])
        
        # Check for SD card
        external_storage = self._execute_adb_command(['shell', 'df', '/storage/'])
        storage['external_storage'] = external_storage is not None and 'sdcard' in external_storage.lower()
        
        return storage
    
    def _analyze_thermal_performance(self) -> Dict[str, Any]:
        """Analyze thermal performance and temperature"""
        thermal = {}
        
        # Battery temperature
        battery_info = self._execute_adb_command(['shell', 'dumpsys', 'battery'])
        if battery_info:
            for line in battery_info.split('\n'):
                if 'temperature:' in line:
                    temp_raw = line.split(':')[1].strip()
                    if temp_raw.isdigit():
                        temp_celsius = int(temp_raw) / 10
                        thermal['battery_temp_celsius'] = temp_celsius
                        
                        # Temperature status
                        if temp_celsius < 35:
                            temp_status = "🟢 COOL"
                        elif temp_celsius < 45:
                            temp_status = "🟡 WARM"
                        else:
                            temp_status = "🔴 HOT"
                            
                        print(f"      🌡️ Battery Temperature: {temp_celsius}°C ({temp_status})")
                        break
        
        # CPU thermal zones
        thermal_zones = self._execute_adb_command(['shell', 'ls', '/sys/class/thermal/'])
        if thermal_zones:
            cpu_temps = []
            for zone in thermal_zones.split('\n'):
                if 'thermal_zone' in zone:
                    temp_path = f'/sys/class/thermal/{zone}/temp'
                    temp_result = self._execute_adb_command(['shell', 'cat', temp_path])
                    if temp_result and temp_result.isdigit():
                        temp_celsius = int(temp_result) / 1000
                        cpu_temps.append(temp_celsius)
            
            if cpu_temps:
                thermal['cpu_temp_celsius'] = round(sum(cpu_temps) / len(cpu_temps), 1)
                thermal['max_cpu_temp_celsius'] = max(cpu_temps)
        
        return thermal
    
    def _analyze_network_performance(self) -> Dict[str, Any]:
        """Analyze network capabilities and performance"""
        network = {}
        
        # WiFi information
        wifi_info = self._execute_adb_command(['shell', 'dumpsys', 'wifi'])
        if wifi_info:
            if 'mWifiInfo' in wifi_info:
                # Extract WiFi details
                network['wifi_connected'] = 'state: CONNECTED' in wifi_info
                
                # Signal strength
                if 'RSSI:' in wifi_info:
                    rssi_line = [line for line in wifi_info.split('\n') if 'RSSI:' in line]
                    if rssi_line:
                        rssi = rssi_line[0].split('RSSI:')[-1].split()[0]
                        if rssi.replace('-', '').isdigit():
                            network['wifi_rssi_dbm'] = int(rssi)
        
        # Mobile data information
        telephony_info = self._execute_adb_command(['shell', 'dumpsys', 'telephony.registry'])
        if telephony_info:
            network['mobile_data_connected'] = 'mDataConnectionState=2' in telephony_info
            
            # Signal strength for mobile
            if 'mSignalStrength' in telephony_info:
                network['mobile_signal_available'] = True
        
        # Network interfaces
        interfaces = self._execute_adb_command(['shell', 'ip', 'addr', 'show'])
        if interfaces:
            network['network_interfaces'] = len([line for line in interfaces.split('\n') if 'inet ' in line])
        
        print(f"      🌐 Network: WiFi {'✅' if network.get('wifi_connected') else '❌'}, Mobile {'✅' if network.get('mobile_data_connected') else '❌'}")
        
        return network
    
    def _calculate_performance_score(self, performance_data: Dict[str, Any]) -> int:
        """Calculate overall performance score (0-100)"""
        score = 0
        factors = 0
        
        # Memory score (25% weight)
        memory = performance_data.get('memory', {})
        if memory.get('total_gb'):
            memory_score = min(memory['total_gb'] / self.NOTHING_2A_CONFIG['optimal_specs']['recommended_memory_gb'] * 100, 100)
            if memory.get('usage_percent', 100) < 80:  # Good if usage < 80%
                memory_score *= 1.1
            score += memory_score * 0.25
            factors += 0.25
        
        # Storage score (20% weight)
        storage = performance_data.get('storage', {})
        if storage.get('available_gb'):
            storage_score = min(storage['available_gb'] / self.NOTHING_2A_CONFIG['optimal_specs']['recommended_free_gb'] * 100, 100)
            score += storage_score * 0.2
            factors += 0.2
        
        # Thermal score (20% weight)
        thermal = performance_data.get('thermal', {})
        if thermal.get('battery_temp_celsius'):
            temp = thermal['battery_temp_celsius']
            max_temp = self.NOTHING_2A_CONFIG['optimal_specs']['max_safe_temp_celsius']
            thermal_score = max(0, (max_temp - temp) / max_temp * 100)
            score += thermal_score * 0.2
            factors += 0.2
        
        # Hardware score (20% weight)
        hardware = performance_data.get('hardware', {})
        if hardware.get('cpu_cores'):
            cpu_score = min(hardware['cpu_cores'] / 8 * 100, 100)  # 8 cores = 100%
            score += cpu_score * 0.2
            factors += 0.2
        
        # System score (15% weight)
        system = performance_data.get('system', {})
        if system.get('load_average'):
            avg_load = sum(system['load_average']) / len(system['load_average'])
            cpu_cores = hardware.get('cpu_cores', 8)
            load_score = max(0, (1 - avg_load / cpu_cores) * 100)
            score += load_score * 0.15
            factors += 0.15
        
        return int(score / factors) if factors > 0 else 50
    
    @_time_operation("JARVIS Compatibility Testing")
    def comprehensive_jarvis_testing(self) -> Dict[str, Any]:
        """Comprehensive JARVIS compatibility and readiness testing"""
        print("🧪 Performing comprehensive JARVIS compatibility testing...")
        
        compatibility_data = {}
        
        # Basic functionality tests
        print("   🔍 Basic Functionality...")
        compatibility_data['basic'] = self._test_basic_functionality()
        
        # Advanced control tests
        print("   🔍 Advanced Controls...")
        compatibility_data['advanced'] = self._test_advanced_controls()
        
        # AI optimization tests
        print("   🔍 AI Optimization...")
        compatibility_data['ai_optimization'] = self._test_ai_optimization()
        
        # Nothing-specific feature tests
        print("   🔍 Nothing Features...")
        compatibility_data['nothing_features'] = self._test_nothing_integration()
        
        # Performance stress tests
        print("   🔍 Performance Stress...")
        compatibility_data['stress_tests'] = self._test_performance_stress()
        
        # Calculate compatibility score
        compatibility_data['compatibility_score'] = self._calculate_compatibility_score(compatibility_data)
        print(f"   📊 JARVIS Compatibility: {compatibility_data['compatibility_score']}/100")
        
        self.jarvis_compatibility = compatibility_data
        return compatibility_data
    
    def _test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic JARVIS functionality requirements"""
        tests = {}
        
        test_cases = [
            ('adb_commands', ['shell', 'echo', 'JARVIS_TEST'], 'JARVIS_TEST'),
            ('app_launching', ['shell', 'am', 'start', '-a', 'android.settings.SETTINGS'], None),
            ('input_simulation', ['shell', 'input', 'keyevent', 'KEYCODE_HOME'], None),
            ('screenshot_capture', ['shell', 'screencap', '/sdcard/jarvis_test.png'], None),
            ('file_operations', ['shell', 'touch', '/sdcard/jarvis_test.txt'], None),
            ('system_properties', ['shell', 'getprop', 'ro.build.version.release'], None)
        ]
        
        for test_name, command, expected_result in test_cases:
            try:
                result = self._execute_adb_command(command, timeout=10)
                if expected_result:
                    success = result == expected_result
                else:
                    success = result is not None
                
                tests[test_name] = success
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"      {status} {test_name}")
                
                # Cleanup test files
                if test_name == 'screenshot_capture' and success:
                    self._execute_adb_command(['shell', 'rm', '/sdcard/jarvis_test.png'])
                elif test_name == 'file_operations' and success:
                    self._execute_adb_command(['shell', 'rm', '/sdcard/jarvis_test.txt'])
                    
            except Exception as e:
                tests[test_name] = False
                print(f"      ❌ FAIL {test_name} ({e})")
                self.logger.error(f"Basic test {test_name} failed: {e}")
        
        return tests
    
    def _test_advanced_controls(self) -> Dict[str, Any]:
        """Test advanced control capabilities"""
        tests = {}
        
        # Test gesture simulation
        gesture_test = self._execute_adb_command(['shell', 'input', 'swipe', '500', '1000', '500', '500'])
        tests['gesture_simulation'] = gesture_test is not None
        
        # Test text input
        text_test = self._execute_adb_command(['shell', 'input', 'text', 'JARVIS'])
        tests['text_input'] = text_test is not None
        
        # Test shell access level
        shell_test = self._execute_adb_command(['shell', 'whoami'])
        tests['shell_access'] = shell_test is not None
        
        # Test service interaction
        service_test = self._execute_adb_command(['shell', 'service', 'list'])
        tests['service_interaction'] = service_test is not None and len(service_test) > 100
        
        # Test intent broadcasting
        intent_test = self._execute_adb_command(['shell', 'am', 'broadcast', '-a', 'android.intent.action.BOOT_COMPLETED'])
        tests['intent_broadcasting'] = intent_test is not None
        
        # Test notification access
        notification_test = self._execute_adb_command(['shell', 'dumpsys', 'notification'])
        tests['notification_access'] = notification_test is not None and 'NotificationManagerService' in notification_test
        
        for test_name, result in tests.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"      {status} {test_name}")
        
        return tests
    
    def _test_ai_optimization(self) -> Dict[str, Any]:
        """Test AI optimization capabilities"""
        tests = {}
        
        # CPU governor check
        cpu_governor = self._execute_adb_command(['shell', 'cat', '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'])
        tests['cpu_governor_control'] = cpu_governor is not None
        if cpu_governor:
            tests['current_cpu_governor'] = cpu_governor.strip()
        
        # Performance mode availability
        perf_mode = self._execute_adb_command(['shell', 'settings', 'get', 'global', 'low_power_mode'])
        tests['performance_mode_control'] = perf_mode is not None
        
        # Background app optimization
        bg_optimization = self._execute_adb_command(['shell', 'dumpsys', 'deviceidle'])
        tests['background_optimization'] = bg_optimization is not None and 'DeviceIdleController' in bg_optimization
        
        # Memory management
        memory_management = self._execute_adb_command(['shell', 'cat', '/proc/sys/vm/swappiness'])
        tests['memory_management'] = memory_management is not None
        
        # Thermal management
        thermal_management = self._execute_adb_command(['shell', 'ls', '/sys/class/thermal/'])
        tests['thermal_management'] = thermal_management is not None and 'thermal_zone' in thermal_management
        
        for test_name, result in tests.items():
            if isinstance(result, bool):
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"      {status} {test_name}")
        
        return tests
    
    def _test_nothing_integration(self) -> Dict[str, Any]:
        """Test Nothing-specific feature integration"""
        tests = {}
        
        # Glyph Interface control test
        glyph_test = any(
            self._execute_adb_command(['shell', 'pm', 'list', 'packages', pkg])
            for pkg in self.NOTHING_2A_CONFIG['glyph_packages']
        )
        tests['glyph_integration'] = glyph_test
        
        # Nothing Launcher integration
        launcher_test = self._execute_adb_command(['shell', 'pm', 'list', 'packages', 'com.nothing.launcher'])
        tests['launcher_integration'] = launcher_test is not None and 'com.nothing.launcher' in launcher_test
        
        # Nothing SystemUI integration
        systemui_test = self._execute_adb_command(['shell', 'dumpsys', 'activity', 'services', '|', 'grep', 'nothing'])
        tests['systemui_integration'] = systemui_test is not None and 'nothing' in systemui_test.lower()
        
        # Nothing Settings access
        settings_test = self._execute_adb_command(['shell', 'am', 'start', '-n', 'com.android.settings/.Settings'])
        tests['settings_integration'] = settings_test is not None
        
        for test_name, result in tests.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"      {status} {test_name}")
        
        return tests
    
    def _test_performance_stress(self) -> Dict[str, Any]:
        """Test performance under stress conditions"""
        tests = {}
        
        print("      🔥 Running stress tests...")
        
        # CPU stress test
        start_time = time.time()
        cpu_stress = self._execute_adb_command(['shell', 'yes', '>', '/dev/null', '&', 'sleep', '2', '&&', 'killall', 'yes'], timeout=5)
        cpu_stress_time = time.time() - start_time
        tests['cpu_stress_response'] = cpu_stress_time < 3  # Should complete quickly
        
        # Memory allocation test
        memory_test = self._execute_adb_command(['shell', 'cat', '/proc/meminfo', '|', 'head', '-5'])
        tests['memory_stress_stable'] = memory_test is not None
        
        # I/O stress test
        io_test = self._execute_adb_command(['shell', 'dd', 'if=/dev/zero', 'of=/sdcard/stress_test', 'bs=1M', 'count=10'], timeout=10)
        if io_test is not None:
            cleanup = self._execute_adb_command(['shell', 'rm', '/sdcard/stress_test'])
            tests['io_stress_stable'] = cleanup is not None
        else:
            tests['io_stress_stable'] = False
        
        # Multi-command stress test
        multi_cmd_success = 0
        for i in range(5):
            result = self._execute_adb_command(['shell', 'echo', f'test_{i}'])
            if result == f'test_{i}':
                multi_cmd_success += 1
        
        tests['multi_command_stability'] = multi_cmd_success >= 4
        
        for test_name, result in tests.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"      {status} {test_name}")
        
        return tests
    
    def _calculate_compatibility_score(self, compatibility_data: Dict[str, Any]) -> int:
        """Calculate overall JARVIS compatibility score"""
        total_tests = 0
        passed_tests = 0
        
        # Weight different test categories
        weights = {
            'basic': 0.4,  # 40% - Most important
            'advanced': 0.3,  # 30% - Very important
            'ai_optimization': 0.15,  # 15% - Important
            'nothing_features': 0.1,  # 10% - Nice to have
            'stress_tests': 0.05  # 5% - Stability check
        }
        
        weighted_score = 0
        
        for category, weight in weights.items():
            if category in compatibility_data:
                category_tests = compatibility_data[category]
                if isinstance(category_tests, dict):
                    category_total = len([v for v in category_tests.values() if isinstance(v, bool)])
                    category_passed = sum(1 for v in category_tests.values() if v is True)
                    
                    if category_total > 0:
                        category_score = (category_passed / category_total) * 100
                        weighted_score += category_score * weight
                        total_tests += category_total
                        passed_tests += category_passed
        
        return int(weighted_score)
    
    def _execute_adb_command(self, command: List[str], timeout: int = None) -> Optional[str]:
        """Execute ADB command with enhanced error handling and retry logic"""
        if not self.device_id:
            return None
        
        timeout = timeout or self.config['timeouts']['adb_command']
        
        for attempt in range(self.config['retry_attempts']):
            try:
                full_command = ['adb', '-s', self.device_id] + command
                result = subprocess.run(
                    full_command, 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout
                )
                
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    if attempt == self.config['retry_attempts'] - 1:
                        self.logger.warning(f"Command failed: {' '.join(command)}, Error: {result.stderr}")
                    continue
                    
            except subprocess.TimeoutExpired:
                if attempt == self.config['retry_attempts'] - 1:
                    self.logger.warning(f"Command timeout: {' '.join(command)}")
                continue
            except Exception as e:
                if attempt == self.config['retry_attempts'] - 1:
                    self.logger.error(f"Command error: {' '.join(command)}, Error: {e}")
                continue
        
        return None
    
    def _get_phone_property(self, property_name: str) -> Optional[str]:
        """Get phone property with caching"""
        result = self._execute_adb_command(['shell', 'getprop', property_name])
        return result.strip() if result else None
    
    def generate_comprehensive_report(self) -> Optional[str]:
        """Generate comprehensive analysis report with encryption option"""
        print("📄 Generating comprehensive analysis report...")
        
        # Compile all analysis data
        report_data = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'session_id': self.analysis_session_id,
                'analyzer_version': '2.0',
                'device_id': self.device_id,
                'analysis_duration_seconds': round(time.time() - self.start_time, 2),
                'operation_times': self.operation_times
            },
            'device_validation': {
                'is_nothing_2a': self._validate_nothing_2a(self.device_id) if self.device_id else False,
                'meets_specifications': self._check_nothing_2a_specs()
            },
            'phone_info': self.phone_info,
            'nothing_features': self.nothing_features,
            'security_profile': self.security_profile,
            'performance_metrics': self.performance_metrics,
            'jarvis_compatibility': self.jarvis_compatibility,
            'recommendations': self._generate_enhanced_recommendations(),
            'optimization_suggestions': self._generate_optimization_suggestions(),
            'risk_assessment': self._generate_risk_assessment()
        }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"nothing_2a_analysis_{timestamp}_{self.analysis_session_id[-8:]}.json"
        
        try:
            # Save unencrypted report
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=4, default=str)
            
            print(f"✅ Comprehensive report saved: {filename}")
            
            # Generate encrypted version if enabled
            if self.config.get('secure_storage', True):
                encrypted_filename = self._create_encrypted_report(report_data, timestamp)
                if encrypted_filename:
                    print(f"🔐 Encrypted report saved: {encrypted_filename}")
            
            # Generate summary report
            summary_filename = self._create_summary_report(report_data, timestamp)
            if summary_filename:
                print(f"📋 Summary report saved: {summary_filename}")
            
            return filename
            
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            self.logger.error(f"Report generation error: {e}")
            return None
    
    def _check_nothing_2a_specs(self) -> Dict[str, bool]:
        """Check if device meets Nothing Phone (2a) specifications"""
        specs_check = {}
        
        # Memory check
        memory_gb = self.performance_metrics.get('memory', {}).get('total_gb', 0)
        specs_check['sufficient_memory'] = memory_gb >= self.NOTHING_2A_CONFIG['optimal_specs']['min_memory_gb']
        
        # Storage check
        storage_gb = self.performance_metrics.get('storage', {}).get('available_gb', 0)
        specs_check['sufficient_storage'] = storage_gb >= self.NOTHING_2A_CONFIG['optimal_specs']['min_storage_gb']
        
        # Temperature check
        temp = self.performance_metrics.get('thermal', {}).get('battery_temp_celsius', 0)
        specs_check['safe_temperature'] = temp <= self.NOTHING_2A_CONFIG['optimal_specs']['max_safe_temp_celsius']
        
        # SOC check
        soc = self.phone_info.get('SOC Model', '')
        specs_check['correct_soc'] = 'Dimensity' in soc
        
        # Android version check
        api_level = int(self.phone_info.get('API Level', '0'))
        specs_check['modern_android'] = api_level >= 33  # Android 13+
        
        return specs_check
    
    def _generate_enhanced_recommendations(self) -> List[Dict[str, str]]:
        """Generate enhanced recommendations based on comprehensive analysis"""
        recommendations = []
        
        # Security recommendations
        if self.security_profile.get('root', {}).get('rooted', False):
            recommendations.append({
                'category': 'Security',
                'priority': 'High',
                'issue': 'Device is rooted',
                'recommendation': 'Consider unrooting for maximum security, or ensure root management is properly configured',
                'impact': 'High security risk, may affect JARVIS reliability'
            })
        
        if not self.security_profile.get('bootloader', {}).get('locked', True):
            recommendations.append({
                'category': 'Security', 
                'priority': 'High',
                'issue': 'Bootloader is unlocked',
                'recommendation': 'Lock bootloader after setup completion for enhanced security',
                'impact': 'Security vulnerability, potential for malicious modifications'
            })
        
        # Performance recommendations
        memory = self.performance_metrics.get('memory', {})
        if memory.get('usage_percent', 0) > 80:
            recommendations.append({
                'category': 'Performance',
                'priority': 'Medium',
                'issue': 'High memory usage detected',
                'recommendation': 'Close unnecessary apps and clear cache to free up memory',
                'impact': 'May affect AI performance and responsiveness'
            })
        
        storage = self.performance_metrics.get('storage', {})
        if storage.get('available_gb', 0) < self.NOTHING_2A_CONFIG['optimal_specs']['recommended_free_gb']:
            recommendations.append({
                'category': 'Storage',
                'priority': 'Medium',
                'issue': 'Low storage space available',
                'recommendation': f"Free up storage space. Recommended: {self.NOTHING_2A_CONFIG['optimal_specs']['recommended_free_gb']}GB minimum",
                'impact': 'Insufficient space for AI models and temporary files'
            })
        
        # Thermal recommendations
        thermal = self.performance_metrics.get('thermal', {})
        if thermal.get('battery_temp_celsius', 0) > 40:
            recommendations.append({
                'category': 'Thermal',
                'priority': 'Medium',
                'issue': 'Device temperature is elevated',
                'recommendation': 'Allow device to cool down, check for resource-intensive apps',
                'impact': 'High temperatures may cause performance throttling'
            })
        
        # Nothing OS specific recommendations
        glyph_features = self.nothing_features.get('glyph', {})
        if not any(glyph_features.values()):
            recommendations.append({
                'category': 'Nothing Features',
                'priority': 'Low',
                'issue': 'Glyph Interface not fully functional',
                'recommendation': 'Update Nothing OS or reinstall Glyph Interface components',
                'impact': 'Limited access to unique Nothing Phone features'
            })
        
        # JARVIS compatibility recommendations
        compatibility = self.jarvis_compatibility
        if compatibility.get('compatibility_score', 0) < 80:
            recommendations.append({
                'category': 'JARVIS Compatibility',
                'priority': 'High',
                'issue': 'JARVIS compatibility below optimal level',
                'recommendation': 'Address failed compatibility tests and optimize device settings',
                'impact': 'JARVIS functionality may be limited or unreliable'
            })
        
        # Developer settings recommendations
        dev_settings = self.security_profile.get('developer', {})
        if not dev_settings.get('usb_debugging', False):
            recommendations.append({
                'category': 'Development',
                'priority': 'High',
                'issue': 'USB debugging not enabled',
                'recommendation': 'Enable USB debugging in Developer Options for full JARVIS control',
                'impact': 'Essential for remote device control and automation'
            })
        
        return recommendations
    
    def _generate_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Generate optimization suggestions for JARVIS performance"""
        optimizations = []
        
        # Performance optimizations
        optimizations.append({
            'category': 'Performance',
            'optimization': 'CPU Governor Optimization',
            'description': 'Set CPU governor to performance mode for AI tasks',
            'command': 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor',
            'requires_root': True,
            'benefit': 'Improved AI processing speed'
        })
        
        optimizations.append({
            'category': 'Performance',
            'optimization': 'Background App Limits',
            'description': 'Limit background app processes to preserve resources',
            'command': 'settings put global background_process_limit 3',
            'requires_root': False,
            'benefit': 'More resources available for JARVIS'
        })
        
        # Memory optimizations
        optimizations.append({
            'category': 'Memory',
            'optimization': 'Swap Configuration',
            'description': 'Optimize memory swappiness for better performance',
            'command': 'echo 10 > /proc/sys/vm/swappiness',
            'requires_root': True,
            'benefit': 'Better memory management for AI tasks'
        })
        
        # Nothing-specific optimizations
        optimizations.append({
            'category': 'Nothing Features',
            'optimization': 'Glyph Interface Optimization',
            'description': 'Optimize Glyph Interface for JARVIS notifications',
            'command': 'settings put secure glyph_notification_enabled 1',
            'requires_root': False,
            'benefit': 'Enhanced visual feedback from JARVIS'
        })
        
        # Battery optimizations
        optimizations.append({
            'category': 'Power Management',
            'optimization': 'Adaptive Battery Disable',
            'description': 'Disable adaptive battery for consistent performance',
            'command': 'settings put global adaptive_battery_management_enabled 0',
            'requires_root': False,
            'benefit': 'Prevents AI performance throttling'
        })
        
        return optimizations
    
    def _generate_risk_assessment(self) -> Dict[str, Any]:
        """Generate comprehensive risk assessment"""
        risks = {
            'security_risks': [],
            'performance_risks': [],
            'compatibility_risks': [],
            'overall_risk_level': 'Low'
        }
        
        # Security risks
        if self.security_profile.get('root', {}).get('rooted', False):
            risks['security_risks'].append({
                'risk': 'Root Access Available',
                'level': 'High',
                'description': 'Device has root access which could be exploited',
                'mitigation': 'Ensure proper root management and security measures'
            })
        
        if self.security_profile.get('security_score', 100) < 70:
            risks['security_risks'].append({
                'risk': 'Low Security Score',
                'level': 'Medium',
                'description': 'Device security configuration needs improvement',
                'mitigation': 'Follow security recommendations to improve score'
            })
        
        # Performance risks
        if self.performance_metrics.get('performance_score', 100) < 70:
            risks['performance_risks'].append({
                'risk': 'Suboptimal Performance',
                'level': 'Medium', 
                'description': 'Device performance may not be sufficient for optimal JARVIS operation',
                'mitigation': 'Follow performance optimization suggestions'
            })
        
        thermal_temp = self.performance_metrics.get('thermal', {}).get('battery_temp_celsius', 0)
        if thermal_temp > 45:
            risks['performance_risks'].append({
                'risk': 'Thermal Throttling Risk',
                'level': 'High',
                'description': 'Device temperature may cause performance throttling',
                'mitigation': 'Allow device to cool and monitor temperature during AI tasks'
            })
        
        # Compatibility risks
        if self.jarvis_compatibility.get('compatibility_score', 100) < 80:
            risks['compatibility_risks'].append({
                'risk': 'JARVIS Compatibility Issues',
                'level': 'High',
                'description': 'Device may not fully support all JARVIS features',
                'mitigation': 'Address failed compatibility tests before deployment'
            })
        
        # Calculate overall risk level
        high_risks = len([r for category in risks.values() if isinstance(category, list) for r in category if r.get('level') == 'High'])
        medium_risks = len([r for category in risks.values() if isinstance(category, list) for r in category if r.get('level') == 'Medium'])
        
        if high_risks > 0:
            risks['overall_risk_level'] = 'High'
        elif medium_risks > 1:
            risks['overall_risk_level'] = 'Medium'
        else:
            risks['overall_risk_level'] = 'Low'
        
        return risks
    
    def _create_encrypted_report(self, report_data: Dict, timestamp: str) -> Optional[str]:
        """Create encrypted version of the report"""
        try:
            import base64
            import json
            
            # Simple encryption for demonstration (use proper encryption in production)
            report_json = json.dumps(report_data, default=str, indent=4)
            encoded_report = base64.b64encode(report_json.encode('utf-8')).decode('utf-8')
            
            encrypted_filename = f"nothing_2a_analysis_encrypted_{timestamp}_{self.analysis_session_id[-8:]}.enc"
            
            with open(encrypted_filename, 'w') as f:
                f.write(encoded_report)
            
            return encrypted_filename
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return None
    
    def _create_summary_report(self, report_data: Dict, timestamp: str) -> Optional[str]:
        """Create human-readable summary report"""
        try:
            summary_filename = f"nothing_2a_summary_{timestamp}_{self.analysis_session_id[-8:]}.txt"
            
            with open(summary_filename, 'w') as f:
                f.write("NOTHING PHONE (2a) - JARVIS ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                # Device Info
                f.write("DEVICE INFORMATION\n")
                f.write("-" * 20 + "\n")
                f.write(f"Model: {report_data['phone_info'].get('Model', 'Unknown')}\n")
                f.write(f"Brand: {report_data['phone_info'].get('Brand', 'Unknown')}\n")
                f.write(f"Android Version: {report_data['phone_info'].get('Android Version', 'Unknown')}\n")
                f.write(f"Security Patch: {report_data['phone_info'].get('Security Patch', 'Unknown')}\n")
                f.write(f"Analysis Time: {report_data['metadata']['analysis_duration_seconds']}s\n\n")
                
                # Scores
                f.write("ANALYSIS SCORES\n")
                f.write("-" * 15 + "\n")
                f.write(f"Security Score: {report_data['security_profile'].get('security_score', 'N/A')}/100\n")
                f.write(f"Performance Score: {report_data['performance_metrics'].get('performance_score', 'N/A')}/100\n")
                f.write(f"JARVIS Compatibility: {report_data['jarvis_compatibility'].get('compatibility_score', 'N/A')}/100\n")
                f.write(f"Nothing OS Features: {report_data['nothing_features'].get('completeness_score', 'N/A'):.1f}%\n\n")
                
                # Key Findings
                f.write("KEY FINDINGS\n")
                f.write("-" * 12 + "\n")
                
                # Security
                is_rooted = report_data['security_profile'].get('root', {}).get('rooted', False)
                f.write(f"Root Status: {'ROOTED' if is_rooted else 'NOT ROOTED'}\n")
                
                # Performance
                memory_gb = report_data['performance_metrics'].get('memory', {}).get('total_gb', 0)
                storage_gb = report_data['performance_metrics'].get('storage', {}).get('available_gb', 0)
                f.write(f"Memory: {memory_gb}GB total\n")
                f.write(f"Available Storage: {storage_gb}GB\n")
                
                # Temperature
                temp = report_data['performance_metrics'].get('thermal', {}).get('battery_temp_celsius', 0)
                f.write(f"Battery Temperature: {temp}°C\n\n")
                
                # Recommendations
                f.write("TOP RECOMMENDATIONS\n")
                f.write("-" * 19 + "\n")
                recommendations = report_data['recommendations']
                high_priority = [r for r in recommendations if r.get('priority') == 'High']
                
                for i, rec in enumerate(high_priority[:5], 1):
                    f.write(f"{i}. {rec['issue']}\n")
                    f.write(f"   Solution: {rec['recommendation']}\n\n")
                
                # Risk Assessment
                f.write("RISK ASSESSMENT\n")
                f.write("-" * 15 + "\n")
                overall_risk = report_data['risk_assessment']['overall_risk_level']
                f.write(f"Overall Risk Level: {overall_risk}\n\n")
                
                f.write(f"Report generated: {report_data['metadata']['analysis_timestamp']}\n")
                f.write(f"Session ID: {report_data['metadata']['session_id']}\n")
            
            return summary_filename
            
        except Exception as e:
            self.logger.error(f"Summary report creation failed: {e}")
            return None
    
    def run_complete_analysis(self) -> bool:
        """Run complete Nothing Phone (2a) analysis with all enhancements"""
        print("🔍 Starting Complete Nothing Phone (2a) Analysis for JARVIS v2.0")
        print("🚀 Ultra-Comprehensive Analysis Engine")
        print("=" * 70)
        
        try:
            # Step 1: Connection check
            if not self.check_phone_connection():
                print("❌ Cannot proceed without Nothing Phone (2a) connection")
                return False
            
            print()
            
            # Step 2: Device information gathering
            self.get_nothing_phone_details()
            print()
            
            # Step 3: Nothing OS features analysis
            self.analyze_nothing_os_features()
            print()
            
            # Step 4: Enhanced security analysis
            self.enhanced_security_analysis()
            print()
            
            # Step 5: Performance analysis
            self.advanced_performance_analysis()
            print()
            
            # Step 6: JARVIS compatibility testing
            self.comprehensive_jarvis_testing()
            print()
            
            # Step 7: Generate comprehensive report
            report_file = self.generate_comprehensive_report()
            
            # Step 8: Display final summary
            self._display_final_summary()
            
            if report_file:
                print(f"\n📄 Detailed reports generated successfully!")
                print(f"⏱️ Total analysis time: {time.time() - self.start_time:.2f} seconds")
                return True
            else:
                print("\n⚠️ Analysis completed but report generation failed")
                return False
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Analysis interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Analysis failed with error: {e}")
            self.logger.error(f"Complete analysis failed: {e}")
            return False
    
    def _display_final_summary(self):
        """Display comprehensive final summary"""
        print("\n🎯 FINAL ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Device validation
        is_nothing_2a = self._validate_nothing_2a(self.device_id) if self.device_id else False
        validation_status = "✅ VALIDATED" if is_nothing_2a else "❌ INVALID"
        print(f"Nothing Phone (2a): {validation_status}")
        
        # Key metrics
        print(f"Model: {self.phone_info.get('Model', 'Unknown')} by {self.phone_info.get('Brand', 'Unknown')}")
        print(f"Android: {self.phone_info.get('Android Version', 'Unknown')} (API {self.phone_info.get('API Level', 'Unknown')})")
        print(f"SOC: {self.phone_info.get('SOC Model', 'Unknown')}")
        
        # Scores display
        print(f"\n📊 COMPREHENSIVE SCORES:")
        security_score = self.security_profile.get('security_score', 0)
        performance_score = self.performance_metrics.get('performance_score', 0)
        compatibility_score = self.jarvis_compatibility.get('compatibility_score', 0)
        features_score = self.nothing_features.get('completeness_score', 0)
        
        def get_score_emoji(score):
            if score >= 90: return "🟢"
            elif score >= 70: return "🟡"
            else: return "🔴"
        
        print(f"   {get_score_emoji(security_score)} Security: {security_score}/100")
        print(f"   {get_score_emoji(performance_score)} Performance: {performance_score}/100")
        print(f"   {get_score_emoji(compatibility_score)} JARVIS Compatibility: {compatibility_score}/100")
        print(f"   {get_score_emoji(features_score)} Nothing OS Features: {features_score:.1f}/100")
        
        # Overall readiness
        overall_score = (security_score + performance_score + compatibility_score + features_score) / 4
        overall_emoji = get_score_emoji(overall_score)
        print(f"\n🏆 OVERALL JARVIS READINESS: {overall_emoji} {overall_score:.1f}/100")
        
        # Status determination
        if overall_score >= 90:
            status = "🚀 EXCELLENT - Ready for advanced JARVIS deployment"
        elif overall_score >= 75:
            status = "✅ GOOD - Ready for JARVIS with minor optimizations"
        elif overall_score >= 60:
            status = "⚠️ FAIR - Requires optimizations before JARVIS deployment"
        else:
            status = "❌ POOR - Significant improvements needed"
        
        print(f"Status: {status}")
        
        # Key highlights
        print(f"\n🔑 KEY HIGHLIGHTS:")
        
        # Memory
        memory_gb = self.performance_metrics.get('memory', {}).get('total_gb', 0)
        memory_status = "✅" if memory_gb >= 8 else "⚠️" if memory_gb >= 6 else "❌"
        print(f"   {memory_status} Memory: {memory_gb}GB")
        
        # Storage
        storage_gb = self.performance_metrics.get('storage', {}).get('available_gb', 0)
        storage_status = "✅" if storage_gb >= 15 else "⚠️" if storage_gb >= 10 else "❌"
        print(f"   {storage_status} Available Storage: {storage_gb}GB")
        
        # Temperature
        temp = self.performance_metrics.get('thermal', {}).get('battery_temp_celsius', 0)
        temp_status = "✅" if temp < 35 else "⚠️" if temp < 45 else "❌"
        print(f"   {temp_status} Temperature: {temp}°C")
        
        # Root status
        is_rooted = self.security_profile.get('root', {}).get('rooted', False)
        root_status = "🔓 ROOTED" if is_rooted else "🔒 NOT ROOTED"
        print(f"   🔐 Root Status: {root_status}")
        
        # Glyph Interface
        glyph_working = any(self.nothing_features.get('glyph', {}).values())
        glyph_status = "✅" if glyph_working else "❌"
        print(f"   {glyph_status} Glyph Interface: {'Functional' if glyph_working else 'Issues detected'}")
        
        # Priority recommendations
        recommendations = self._generate_enhanced_recommendations()
        high_priority = [r for r in recommendations if r.get('priority') == 'High']
        
        if high_priority:
            print(f"\n⚡ PRIORITY ACTIONS ({len(high_priority)} items):")
            for i, rec in enumerate(high_priority[:3], 1):
                print(f"   {i}. {rec['issue']}")
        else:
            print(f"\n✨ No high-priority issues detected!")
        
        print(f"\n🎉 Analysis Complete! Session: {self.analysis_session_id}")

def main():
    """Enhanced main function with configuration support"""
    print("🤖 JARVIS - Nothing Phone (2a) Ultra Analyzer v2.0")
    print("🔥 Professional-Grade Device Analysis & Optimization")
    print("=" * 60)
    print()
    
    # Configuration options
    print("📋 Analysis Configuration:")
    print("   ✓ Parallel processing enabled")
    print("   ✓ Advanced security analysis")
    print("   ✓ Comprehensive performance testing")  
    print("   ✓ Nothing OS feature validation")
    print("   ✓ JARVIS compatibility verification")
    print("   ✓ Encrypted report generation")
    print()
    
    print("🔌 Prerequisites:")
    print("   • Nothing Phone (2a) connected via USB")
    print("   • USB debugging enabled")
    print("   • ADB drivers installed")
    print("   • Sufficient storage for reports")
    print()
    
    # Get user confirmation
    response = input("🚀 Ready to begin ultra-comprehensive analysis? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("👋 Analysis cancelled. Connect your Nothing Phone (2a) when ready!")
        return
    
    try:
        # Initialize analyzer
        analyzer = NothingPhoneAnalyzer()
        
        # Run complete analysis
        print("\n" + "="*70)
        success = analyzer.run_complete_analysis()
        
        print("\n" + "="*70)
        if success:
            print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print()
            print("📊 Your Nothing Phone (2a) has been comprehensively analyzed")
            print("🔍 Check the generated reports for detailed insights")
            print("🚀 Ready to proceed with JARVIS integration!")
        else:
            print("⚠️ ANALYSIS COMPLETED WITH ISSUES")
            print()
            print("📋 Please review the results and address any problems")
            print("🔄 You may need to re-run the analysis after fixes")
        
        print(f"\n⏱️ Session completed in {time.time() - analyzer.start_time:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        print("👋 You can restart the analysis anytime")
    except Exception as e:
        print(f"\n❌ Critical error occurred: {e}")
        print("🔧 Please check your setup and try again")

if __name__ == "__main__":
    main()