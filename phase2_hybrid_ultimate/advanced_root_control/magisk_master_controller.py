#!/usr/bin/env python3
"""
JARVIS Magisk Master Controller v1.0
Advanced Magisk Module Management for Nothing Phone A142
Integrates with Intelligent Root Orchestrator
"""

import asyncio
import logging
import json
import time
from pathlib import Path
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Any

class MagiskMasterController:
    """Advanced Magisk module management system"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.db_path = Path('logs/magisk_master.db')
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        
        # State management
        self.magisk_modules = {}
        self.operation_queue = []
        self.root_verified = False
        self.device_verified = False
        self.magisk_verified = False
        
        # Metrics
        self.metrics = {
            'operations_executed': 0,
            'operations_succeeded': 0,
            'operations_failed': 0,
            'modules_managed': 0,
            'start_time': time.time()
        }
        
        # Nothing Phone A142 specific paths
        self.magisk_paths = {
            'magisk_binary': '/system/bin/magisk',
            'magisk_db': '/data/adb/magisk.db',
            'modules_path': '/data/adb/modules',
            'magisk_img': '/data/adb/magisk.img'
        }
        
        self.logger.info("🛡️ Magisk Master Controller v1.0 initialized")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Magisk operations"""
        logger = logging.getLogger('magisk_master')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f'magisk_master_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s | MAGISK | %(levelname)s | %(funcName)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def _init_database(self):
        """Initialize SQLite database for Magisk operations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS magisk_operations (
                    id TEXT PRIMARY KEY,
                    operation_type TEXT NOT NULL,
                    command TEXT NOT NULL,
                    description TEXT NOT NULL,
                    module_name TEXT,
                    success BOOLEAN,
                    execution_time REAL,
                    timestamp TEXT NOT NULL,
                    result_data TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS magisk_modules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT,
                    author TEXT,
                    description TEXT,
                    status TEXT,
                    install_date TEXT,
                    last_updated TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Magisk database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {str(e)}")

    async def initialize_magisk_controller(self) -> bool:
        """Initialize Magisk Master Controller"""
        try:
            self.logger.info("🚀 Initializing Magisk Master Controller...")
            
            # Verify root access
            if not await self._verify_root_access():
                return False
            
            # Verify device
            if not await self._verify_device_compatibility():
                return False
            
            # Verify Magisk installation
            if not await self._verify_magisk_installation():
                return False
            
            # Scan existing modules
            await self._scan_installed_modules()
            
            self.logger.info("✅ Magisk Master Controller operational!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Magisk controller initialization failed: {str(e)}")
            return False

    async def _verify_root_access(self) -> bool:
        """Verify root access"""
        try:
            result = await self._execute_command("id")
            if result['success'] and "uid=0" in result['output']:
                self.root_verified = True
                self.logger.info("✅ Root access verified")
                return True
            else:
                self.logger.error("❌ Root access not available")
                return False
                
        except Exception as e:
            self.logger.error(f"Root verification failed: {str(e)}")
            return False

    async def _verify_device_compatibility(self) -> bool:
        """Verify Nothing Phone A142 compatibility"""
        try:
            result = await self._execute_command("getprop ro.product.model")
            if result['success']:
                device_model = result['output'].strip()
                if "A142" in device_model:
                    self.device_verified = True
                    self.logger.info(f"✅ Device verified: {device_model}")
                    return True
                else:
                    self.logger.warning(f"⚠️ Device not verified as A142: {device_model}")
                    return True  # Continue anyway
            
            return False
            
        except Exception as e:
            self.logger.error(f"Device verification failed: {str(e)}")
            return False

    async def _verify_magisk_installation(self) -> bool:
        """Verify Magisk is properly installed"""
        try:
            # Check for Magisk binary
            result = await self._execute_command("which magisk")
            if result['success'] and result['output']:
                self.magisk_verified = True
                self.logger.info("✅ Magisk binary found")
                
                # Get Magisk version
                version_result = await self._execute_command("magisk -V")
                if version_result['success']:
                    self.logger.info(f"📊 Magisk version: {version_result['output']}")
                
                return True
            else:
                self.logger.warning("⚠️ Magisk binary not found - checking alternative paths")
                
                # Check alternative paths
                for path in ['/system/bin/magisk', '/system/xbin/magisk']:
                    result = await self._execute_command(f"test -f {path}")
                    if result['success']:
                        self.magisk_verified = True
                        self.logger.info(f"✅ Magisk found at: {path}")
                        return True
                
                self.logger.error("❌ Magisk not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Magisk verification failed: {str(e)}")
            return False

    async def _execute_command(self, command: str) -> Dict[str, Any]:
        """Execute ADB command"""
        try:
            process = await asyncio.create_subprocess_exec(
                "adb", "shell", "su", "-c", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'success': process.returncode == 0,
                'output': stdout.decode().strip(),
                'error': stderr.decode().strip(),
                'return_code': process.returncode
            }
            
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e),
                'return_code': -1
            }

    async def _scan_installed_modules(self):
        """Scan for installed Magisk modules"""
        try:
            self.logger.info("🔍 Scanning installed Magisk modules...")
            
            # List modules directory
            result = await self._execute_command("ls -la /data/adb/modules/")
            if result['success']:
                lines = result['output'].split('\n')
                module_count = 0
                
                for line in lines:
                    if line.startswith('d') and not line.endswith('.'):
                        parts = line.split()
                        if len(parts) >= 9:
                            module_name = parts[-1]
                            if module_name not in ['.', '..']:
                                self.magisk_modules[module_name] = {
                                    'name': module_name,
                                    'path': f'/data/adb/modules/{module_name}',
                                    'status': 'installed',
                                    'scan_date': datetime.now().isoformat()
                                }
                                module_count += 1
                
                self.metrics['modules_managed'] = module_count
                self.logger.info(f"📊 Found {module_count} Magisk modules")
                
                # Get detailed info for each module
                for module_name in self.magisk_modules.keys():
                    await self._get_module_details(module_name)
                
            else:
                self.logger.warning("⚠️ Could not scan modules directory")
                
        except Exception as e:
            self.logger.error(f"Module scanning failed: {str(e)}")

    async def _get_module_details(self, module_name: str):
        """Get detailed information about a Magisk module"""
        try:
            module_path = f"/data/adb/modules/{module_name}"
            
            # Read module.prop if it exists
            prop_result = await self._execute_command(f"cat {module_path}/module.prop")
            if prop_result['success']:
                props = {}
                for line in prop_result['output'].split('\n'):
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        props[key.strip()] = value.strip()
                
                # Update module info
                if module_name in self.magisk_modules:
                    self.magisk_modules[module_name].update({
                        'id': props.get('id', module_name),
                        'name': props.get('name', module_name),
                        'version': props.get('version', 'unknown'),
                        'author': props.get('author', 'unknown'),
                        'description': props.get('description', 'No description')
                    })
                
        except Exception as e:
            self.logger.error(f"Failed to get details for module {module_name}: {str(e)}")

    async def list_magisk_modules(self) -> Dict[str, Any]:
        """List all installed Magisk modules"""
        try:
            self.logger.info("📋 Listing Magisk modules...")
            
            if not self.magisk_modules:
                await self._scan_installed_modules()
            
            return {
                'success': True,
                'modules': self.magisk_modules,
                'count': len(self.magisk_modules),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to list modules: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'modules': {},
                'count': 0
            }

    async def install_magisk_module(self, module_path: str, description: str = "") -> Dict[str, Any]:
        """Install a Magisk module"""
        try:
            operation_id = hashlib.md5(f"install_{module_path}_{time.time()}".encode()).hexdigest()[:12]
            
            self.logger.info(f"📦 Installing Magisk module: {module_path}")
            
            start_time = time.time()
            
            # Install module using Magisk
            result = await self._execute_command(f"magisk --install-module {module_path}")
            
            execution_time = time.time() - start_time
            
            if result['success']:
                self.logger.info(f"✅ Module installed successfully: {description}")
                self.metrics['operations_succeeded'] += 1
                
                # Rescan modules
                await self._scan_installed_modules()
                
                # Log operation
                await self._log_operation(
                    operation_id, 'install', f"magisk --install-module {module_path}",
                    description, execution_time, True, result
                )
                
                return {
                    'success': True,
                    'message': f'Module installed: {description}',
                    'operation_id': operation_id,
                    'execution_time': execution_time
                }
            else:
                self.logger.error(f"❌ Module installation failed: {result['error']}")
                self.metrics['operations_failed'] += 1
                
                return {
                    'success': False,
                    'error': result['error'],
                    'operation_id': operation_id
                }
                
        except Exception as e:
            self.logger.error(f"Module installation error: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def uninstall_magisk_module(self, module_name: str) -> Dict[str, Any]:
        """Uninstall a Magisk module"""
        try:
            operation_id = hashlib.md5(f"uninstall_{module_name}_{time.time()}".encode()).hexdigest()[:12]
            
            self.logger.info(f"🗑️ Uninstalling Magisk module: {module_name}")
            
            start_time = time.time()
            
            # Remove module directory
            result = await self._execute_command(f"rm -rf /data/adb/modules/{module_name}")
            
            execution_time = time.time() - start_time
            
            if result['success']:
                self.logger.info(f"✅ Module uninstalled: {module_name}")
                self.metrics['operations_succeeded'] += 1
                
                # Remove from our tracking
                if module_name in self.magisk_modules:
                    del self.magisk_modules[module_name]
                
                # Log operation
                await self._log_operation(
                    operation_id, 'uninstall', f"rm -rf /data/adb/modules/{module_name}",
                    f"Uninstall module {module_name}", execution_time, True, result
                )
                
                return {
                    'success': True,
                    'message': f'Module uninstalled: {module_name}',
                    'operation_id': operation_id
                }
            else:
                self.logger.error(f"❌ Module uninstallation failed: {result['error']}")
                self.metrics['operations_failed'] += 1
                
                return {
                    'success': False,
                    'error': result['error'],
                    'operation_id': operation_id
                }
                
        except Exception as e:
            self.logger.error(f"Module uninstallation error: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def get_magisk_status(self) -> Dict[str, Any]:
        """Get comprehensive Magisk system status"""
        try:
            uptime = time.time() - self.metrics['start_time']
            success_rate = (
                self.metrics['operations_succeeded'] / 
                max(1, self.metrics['operations_executed']) * 100
            )
            
            return {
                'status': 'operational' if self.magisk_verified else 'limited',
                'root_verified': self.root_verified,
                'device_verified': self.device_verified,
                'magisk_verified': self.magisk_verified,
                'uptime_seconds': uptime,
                'modules': {
                    'installed_count': len(self.magisk_modules),
                    'managed_modules': self.metrics['modules_managed']
                },
                'operations': {
                    'total': self.metrics['operations_executed'],
                    'succeeded': self.metrics['operations_succeeded'],
                    'failed': self.metrics['operations_failed'],
                    'success_rate': f"{success_rate:.1f}%"
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    async def _log_operation(self, operation_id: str, operation_type: str, command: str,
                           description: str, execution_time: float, success: bool, result: Dict):
        """Log operation to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO magisk_operations 
                (id, operation_type, command, description, success, execution_time, timestamp, result_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                operation_id, operation_type, command, description, success,
                execution_time, datetime.now().isoformat(), json.dumps(result)
            ))
            
            conn.commit()
            conn.close()
            
            self.metrics['operations_executed'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to log operation: {str(e)}")

# Demo and testing
async def main():
    """Demo the Magisk Master Controller"""
    controller = MagiskMasterController()
    
    print("🛡️ JARVIS Magisk Master Controller v1.0")
    print("=" * 60)
    
    if await controller.initialize_magisk_controller():
        print("✅ Magisk Master Controller operational!")
        
        # List installed modules
        print("\n📋 Listing installed Magisk modules...")
        modules_result = await controller.list_magisk_modules()
        if modules_result['success']:
            print(f"   Found {modules_result['count']} modules:")
            for module_name, module_info in modules_result['modules'].items():
                print(f"     - {module_info.get('name', module_name)}: {module_info.get('description', 'No description')}")
        
        # Get system status
        print("\n📊 Getting Magisk system status...")
        status = await controller.get_magisk_status()
        print("   Status Summary:")
        print(f"     Status: {status['status']}")
        print(f"     Root: {'✅' if status['root_verified'] else '❌'}")
        print(f"     Device: {'✅' if status['device_verified'] else '❌'}")  
        print(f"     Magisk: {'✅' if status['magisk_verified'] else '❌'}")
        print(f"     Modules: {status['modules']['installed_count']} installed")
        
        print("\n✅ Magisk Master Controller demonstration completed!")
        
    else:
        print("❌ Magisk Master Controller initialization failed!")

if __name__ == '__main__':
    asyncio.run(main())
