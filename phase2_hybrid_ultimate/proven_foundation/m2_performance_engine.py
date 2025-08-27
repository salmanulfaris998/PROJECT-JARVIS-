#!/usr/bin/env python3
"""
JARVIS M2 Performance Analytics - Perfect Edition
Optimized exclusively for MacBook Air M2 chip with advanced Apple Silicon monitoring
"""

import logging
import psutil
import json
import time
import subprocess
import asyncio
import threading
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import weakref
import signal
import sys

@dataclass
class M2Metrics:
    """M2-specific performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_pressure: str
    gpu_utilization: float
    neural_engine_usage: float
    efficiency_cores_active: int
    performance_cores_active: int
    cpu_temperature: float
    gpu_temperature: float
    power_consumption: float
    thermal_state: str
    swap_usage: float
    bandwidth_utilization: float

@dataclass
class JARVISMetrics:
    """JARVIS-specific performance metrics"""
    ai_processing_time: float
    ai_success_rate: float
    device_control_success_rate: float
    voice_recognition_accuracy: float
    m2_gflops_current: float
    m2_gflops_peak: float
    neural_network_efficiency: float
    response_latency: float

class M2PerformanceMonitor:
    """Advanced M2 chip performance monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger('m2_monitor')
        self._monitoring = False
        self._metrics_cache = {}
        
    async def get_m2_metrics(self) -> Optional[M2Metrics]:
        """Get comprehensive M2 performance metrics"""
        try:
            # Use powermetrics for accurate M2 data
            cmd = [
                'sudo', 'powermetrics',
                '--samplers', 'cpu_power,gpu_power,thermal',
                '--sample-rate', '1000',
                '--sample-count', '1',
                '--format', 'plist'
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return self._parse_powermetrics(stdout.decode())
            else:
                return self._get_fallback_metrics()
                
        except Exception as e:
            self.logger.warning(f"M2 metrics collection failed: {e}")
            return self._get_fallback_metrics()
    
    def _parse_powermetrics(self, output: str) -> M2Metrics:
        """Parse powermetrics output for M2 data"""
        try:
            import plistlib
            
            # Extract plist data
            plist_data = plistlib.loads(output.encode())
            
            # Extract M2-specific metrics
            cpu_data = plist_data.get('processor', {})
            gpu_data = plist_data.get('gpu', {})
            thermal_data = plist_data.get('thermal_pressure', {})
            
            return M2Metrics(
                timestamp=time.time(),
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=psutil.virtual_memory().percent,
                memory_pressure=self._get_memory_pressure(),
                gpu_utilization=self._extract_gpu_usage(gpu_data),
                neural_engine_usage=self._get_neural_engine_usage(),
                efficiency_cores_active=self._get_efficiency_cores(),
                performance_cores_active=self._get_performance_cores(),
                cpu_temperature=self._extract_temperature(cpu_data, 'cpu'),
                gpu_temperature=self._extract_temperature(gpu_data, 'gpu'),
                power_consumption=self._extract_power_consumption(cpu_data, gpu_data),
                thermal_state=thermal_data.get('state', 'nominal'),
                swap_usage=self._get_swap_usage(),
                bandwidth_utilization=self._get_bandwidth_utilization()
            )
            
        except Exception as e:
            self.logger.error(f"Powermetrics parsing failed: {e}")
            return self._get_fallback_metrics()
    
    def _get_fallback_metrics(self) -> M2Metrics:
        """Fallback metrics when powermetrics fails"""
        vm = psutil.virtual_memory()
        
        return M2Metrics(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=vm.percent,
            memory_pressure=self._get_memory_pressure(),
            gpu_utilization=self._estimate_gpu_usage(),
            neural_engine_usage=0.0,
            efficiency_cores_active=4,  # M2 has 4 efficiency cores
            performance_cores_active=4,  # M2 has 4 performance cores
            cpu_temperature=0.0,
            gpu_temperature=0.0,
            power_consumption=self._estimate_power_usage(),
            thermal_state='unknown',
            swap_usage=self._get_swap_usage(),
            bandwidth_utilization=0.0
        )
    
    def _get_memory_pressure(self) -> str:
        """Get macOS memory pressure state"""
        try:
            result = subprocess.run(['memory_pressure'], 
                                  capture_output=True, text=True, timeout=5)
            if 'System-wide memory free percentage:' in result.stdout:
                percentage = float(result.stdout.split(':')[-1].strip().replace('%', ''))
                if percentage > 20:
                    return 'normal'
                elif percentage > 10:
                    return 'warn'
                else:
                    return 'urgent'
        except:
            pass
        
        # Fallback based on memory usage
        mem_percent = psutil.virtual_memory().percent
        if mem_percent < 70:
            return 'normal'
        elif mem_percent < 85:
            return 'warn'
        else:
            return 'urgent'
    
    def _get_neural_engine_usage(self) -> float:
        """Estimate Neural Engine usage (M2-specific)"""
        try:
            # Check for CoreML processes
            coreml_processes = 0
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                if any(keyword in proc.info['name'].lower() for keyword in 
                      ['coreml', 'neural', 'mlcompute', 'ane']):
                    coreml_processes += 1
            
            return min(coreml_processes * 25.0, 100.0)
        except:
            return 0.0
    
    def _get_efficiency_cores(self) -> int:
        """Get active efficiency cores count"""
        try:
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            return min(max(1, int(load_avg)), 4)  # M2 has 4 efficiency cores
        except:
            return 4
    
    def _get_performance_cores(self) -> int:
        """Get active performance cores count"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 75:
                return 4  # All performance cores active
            elif cpu_percent > 50:
                return 3
            elif cpu_percent > 25:
                return 2
            else:
                return 1
        except:
            return 4
    
    def _estimate_gpu_usage(self) -> float:
        """Estimate GPU usage on M2"""
        try:
            gpu_processes = 0
            for proc in psutil.process_iter(['pid', 'name']):
                if any(keyword in proc.info['name'].lower() for keyword in 
                      ['gpu', 'metal', 'graphics', 'render']):
                    gpu_processes += 1
            
            return min(gpu_processes * 15.0, 100.0)
        except:
            return 0.0
    
    def _estimate_power_usage(self) -> float:
        """Estimate power consumption"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            # M2 typical power range: 5W (idle) to 22W (max)
            base_power = 5.0
            max_additional = 17.0
            return base_power + (max_additional * cpu_percent / 100.0)
        except:
            return 10.0
    
    def _get_swap_usage(self) -> float:
        """Get swap usage percentage"""
        try:
            swap = psutil.swap_memory()
            return swap.percent if swap.total > 0 else 0.0
        except:
            return 0.0
    
    def _get_bandwidth_utilization(self) -> float:
        """Estimate memory bandwidth utilization"""
        try:
            mem_percent = psutil.virtual_memory().percent
            # M2 has 100 GB/s memory bandwidth
            return min(mem_percent * 1.2, 100.0)
        except:
            return 0.0
    
    def _extract_gpu_usage(self, gpu_data: Dict) -> float:
        """Extract GPU usage from powermetrics"""
        try:
            return float(gpu_data.get('utilization', 0.0))
        except:
            return 0.0
    
    def _extract_temperature(self, data: Dict, component: str) -> float:
        """Extract temperature from powermetrics data"""
        try:
            temp_key = f'{component}_die_temperature'
            return float(data.get(temp_key, 0.0))
        except:
            return 0.0
    
    def _extract_power_consumption(self, cpu_data: Dict, gpu_data: Dict) -> float:
        """Extract power consumption from powermetrics"""
        try:
            cpu_power = float(cpu_data.get('package_watts', 0.0))
            gpu_power = float(gpu_data.get('watts', 0.0))
            return cpu_power + gpu_power
        except:
            return 0.0

class PerformanceDatabase:
    """SQLite database for performance metrics storage"""
    
    def __init__(self, db_path: str = "jarvis_performance.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger('perf_db')
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with proper schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS m2_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        cpu_percent REAL,
                        memory_percent REAL,
                        memory_pressure TEXT,
                        gpu_utilization REAL,
                        neural_engine_usage REAL,
                        efficiency_cores_active INTEGER,
                        performance_cores_active INTEGER,
                        cpu_temperature REAL,
                        gpu_temperature REAL,
                        power_consumption REAL,
                        thermal_state TEXT,
                        swap_usage REAL,
                        bandwidth_utilization REAL
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS jarvis_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        ai_processing_time REAL,
                        ai_success_rate REAL,
                        device_control_success_rate REAL,
                        voice_recognition_accuracy REAL,
                        m2_gflops_current REAL,
                        m2_gflops_peak REAL,
                        neural_network_efficiency REAL,
                        response_latency REAL
                    )
                ''')
                
                # Create indexes for better query performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_m2_timestamp ON m2_metrics(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_jarvis_timestamp ON jarvis_metrics(timestamp)')
                
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def store_m2_metrics(self, metrics: M2Metrics):
        """Store M2 metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO m2_metrics VALUES (
                        NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                ''', (
                    metrics.timestamp, metrics.cpu_percent, metrics.memory_percent,
                    metrics.memory_pressure, metrics.gpu_utilization, metrics.neural_engine_usage,
                    metrics.efficiency_cores_active, metrics.performance_cores_active,
                    metrics.cpu_temperature, metrics.gpu_temperature, metrics.power_consumption,
                    metrics.thermal_state, metrics.swap_usage, metrics.bandwidth_utilization
                ))
        except Exception as e:
            self.logger.error(f"Failed to store M2 metrics: {e}")
    
    def store_jarvis_metrics(self, metrics: JARVISMetrics):
        """Store JARVIS metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO jarvis_metrics VALUES (
                        NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                ''', (
                    time.time(), metrics.ai_processing_time, metrics.ai_success_rate,
                    metrics.device_control_success_rate, metrics.voice_recognition_accuracy,
                    metrics.m2_gflops_current, metrics.m2_gflops_peak,
                    metrics.neural_network_efficiency, metrics.response_latency
                ))
        except Exception as e:
            self.logger.error(f"Failed to store JARVIS metrics: {e}")
    
    def get_recent_metrics(self, hours: int = 1) -> Tuple[List[M2Metrics], List[JARVISMetrics]]:
        """Get recent metrics from database"""
        cutoff_time = time.time() - (hours * 3600)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get M2 metrics
                m2_cursor = conn.execute('''
                    SELECT * FROM m2_metrics 
                    WHERE timestamp > ? 
                    ORDER BY timestamp DESC
                ''', (cutoff_time,))
                
                m2_metrics = []
                for row in m2_cursor.fetchall():
                    m2_metrics.append(M2Metrics(
                        timestamp=row[1], cpu_percent=row[2], memory_percent=row[3],
                        memory_pressure=row[4], gpu_utilization=row[5], neural_engine_usage=row[6],
                        efficiency_cores_active=row[7], performance_cores_active=row[8],
                        cpu_temperature=row[9], gpu_temperature=row[10], power_consumption=row[11],
                        thermal_state=row[12], swap_usage=row[13], bandwidth_utilization=row[14]
                    ))
                
                # Get JARVIS metrics
                jarvis_cursor = conn.execute('''
                    SELECT * FROM jarvis_metrics 
                    WHERE timestamp > ? 
                    ORDER BY timestamp DESC
                ''', (cutoff_time,))
                
                jarvis_metrics = []
                for row in jarvis_cursor.fetchall():
                    jarvis_metrics.append(JARVISMetrics(
                        ai_processing_time=row[2], ai_success_rate=row[3],
                        device_control_success_rate=row[4], voice_recognition_accuracy=row[5],
                        m2_gflops_current=row[6], m2_gflops_peak=row[7],
                        neural_network_efficiency=row[8], response_latency=row[9]
                    ))
                
                return m2_metrics, jarvis_metrics
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve metrics: {e}")
            return [], []
    
    def cleanup_old_data(self, days_to_keep: int = 7):
        """Clean up old performance data"""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM m2_metrics WHERE timestamp < ?', (cutoff_time,))
                conn.execute('DELETE FROM jarvis_metrics WHERE timestamp < ?', (cutoff_time,))
                conn.execute('VACUUM')
                
                self.logger.info(f"Cleaned up data older than {days_to_keep} days")
                
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")

class M2PerformanceAnalytics:
    """Perfect M2 Performance Analytics System"""
    
    def __init__(self, log_dir: str = "../comprehensive_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.m2_monitor = M2PerformanceMonitor()
        self.database = PerformanceDatabase()
        
        self._monitoring = False
        self._monitor_task = None
        self._metrics_lock = threading.RLock()
        self._current_m2_metrics = None
        self._current_jarvis_metrics = None
        
        # M2-specific constants
        self.M2_SPECS = {
            'max_gflops': 628,
            'max_power_watts': 22.0,
            'efficiency_cores': 4,
            'performance_cores': 4,
            'max_memory_bandwidth_gbps': 100,
            'neural_engine_tops': 15.8
        }
        
        # JARVIS performance trackers
        self._jarvis_trackers = {
            'ai_requests': [],
            'device_commands': [],
            'voice_commands': [],
            'response_times': []
        }
        
        self.logger.info("🎯 M2 Performance Analytics - Perfect Edition Initialized")
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system"""
        logger = logging.getLogger('m2_analytics')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.log_dir / 'jarvis_m2_analytics.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s | M2-ANALYTICS | %(levelname)s | %(message)s')
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s')
        )
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_monitoring()
        sys.exit(0)
    
    async def start_monitoring(self, interval: float = 10.0) -> bool:
        """Start advanced M2 performance monitoring"""
        try:
            if self._monitoring:
                self.logger.warning("Monitoring already active")
                return True
            
            self.logger.info("🚀 Starting M2 performance monitoring...")
            
            self._monitoring = True
            self._monitor_task = asyncio.create_task(self._monitoring_loop(interval))
            
            # Start database cleanup task
            asyncio.create_task(self._periodic_cleanup())
            
            self.logger.info("✅ M2 monitoring started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start monitoring: {e}")
            self._monitoring = False
            return False
    
    async def _monitoring_loop(self, interval: float):
        """Main monitoring loop with async/await"""
        consecutive_errors = 0
        max_errors = 5
        
        while self._monitoring:
            try:
                start_time = time.time()
                
                # Get M2 metrics
                m2_metrics = await self.m2_monitor.get_m2_metrics()
                
                if m2_metrics:
                    with self._metrics_lock:
                        self._current_m2_metrics = m2_metrics
                    
                    # Store in database
                    self.database.store_m2_metrics(m2_metrics)
                    
                    # Log significant events
                    self._check_performance_alerts(m2_metrics)
                    
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        self.logger.error(f"Too many consecutive monitoring errors ({max_errors})")
                        break
                
                # Maintain precise interval timing
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                consecutive_errors += 1
                await asyncio.sleep(interval)
    
    async def _periodic_cleanup(self):
        """Periodic database cleanup"""
        while self._monitoring:
            try:
                await asyncio.sleep(3600)  # Run every hour
                self.database.cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    def _check_performance_alerts(self, metrics: M2Metrics):
        """Check for performance alerts and log them"""
        alerts = []
        
        # CPU temperature alert
        if metrics.cpu_temperature > 85:
            alerts.append(f"🔥 High CPU temperature: {metrics.cpu_temperature:.1f}°C")
        
        # GPU temperature alert
        if metrics.gpu_temperature > 85:
            alerts.append(f"🔥 High GPU temperature: {metrics.gpu_temperature:.1f}°C")
        
        # Memory pressure alert
        if metrics.memory_pressure in ['warn', 'urgent']:
            alerts.append(f"⚠️ Memory pressure: {metrics.memory_pressure}")
        
        # Power consumption alert
        if metrics.power_consumption > 20:
            alerts.append(f"⚡ High power consumption: {metrics.power_consumption:.1f}W")
        
        # Thermal throttling alert
        if metrics.thermal_state != 'nominal':
            alerts.append(f"🌡️ Thermal state: {metrics.thermal_state}")
        
        # Log all alerts
        for alert in alerts:
            self.logger.warning(alert)
    
    def track_ai_processing(self, processing_time: float, success: bool, 
                          model_type: str = "default") -> None:
        """Track AI processing with enhanced metrics"""
        try:
            timestamp = time.time()
            
            # Calculate current GFLOPS based on processing complexity
            estimated_gflops = self._estimate_gflops(processing_time, model_type)
            
            with self._metrics_lock:
                self._jarvis_trackers['ai_requests'].append({
                    'timestamp': timestamp,
                    'processing_time': processing_time,
                    'success': success,
                    'model_type': model_type,
                    'estimated_gflops': estimated_gflops
                })
                
                # Keep only recent entries
                self._cleanup_tracker('ai_requests', 1000)
            
            # Update JARVIS metrics
            self._update_jarvis_metrics()
            
            self.logger.info(f"🧠 AI Processing: {processing_time:.3f}s, Success: {success}, "
                           f"Model: {model_type}, Est. GFLOPS: {estimated_gflops:.1f}")
            
        except Exception as e:
            self.logger.error(f"AI processing tracking error: {e}")
    
    def track_device_command(self, command_type: str, success: bool, 
                           execution_time: float = 0.0) -> None:
        """Track device control commands"""
        try:
            timestamp = time.time()
            
            with self._metrics_lock:
                self._jarvis_trackers['device_commands'].append({
                    'timestamp': timestamp,
                    'command_type': command_type,
                    'success': success,
                    'execution_time': execution_time
                })
                
                self._cleanup_tracker('device_commands', 1000)
            
            self._update_jarvis_metrics()
            
            self.logger.info(f"🎛️ Device Command: {command_type}, Success: {success}, "
                           f"Time: {execution_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Device command tracking error: {e}")
    
    def track_voice_command(self, recognized_correctly: bool, confidence: float = 0.0,
                          processing_time: float = 0.0) -> None:
        """Track voice recognition with confidence scoring"""
        try:
            timestamp = time.time()
            
            with self._metrics_lock:
                self._jarvis_trackers['voice_commands'].append({
                    'timestamp': timestamp,
                    'recognized': recognized_correctly,
                    'confidence': confidence,
                    'processing_time': processing_time
                })
                
                self._cleanup_tracker('voice_commands', 1000)
            
            self._update_jarvis_metrics()
            
            self.logger.info(f"🎤 Voice Command: Recognized: {recognized_correctly}, "
                           f"Confidence: {confidence:.2f}, Time: {processing_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Voice command tracking error: {e}")
    
    def track_response_time(self, response_time: float, request_type: str = "general") -> None:
        """Track JARVIS response times"""
        try:
            timestamp = time.time()
            
            with self._metrics_lock:
                self._jarvis_trackers['response_times'].append({
                    'timestamp': timestamp,
                    'response_time': response_time,
                    'request_type': request_type
                })
                
                self._cleanup_tracker('response_times', 1000)
            
            self._update_jarvis_metrics()
            
            self.logger.info(f"⏱️ Response Time: {response_time:.3f}s, Type: {request_type}")
            
        except Exception as e:
            self.logger.error(f"Response time tracking error: {e}")
    
    def _estimate_gflops(self, processing_time: float, model_type: str) -> float:
        """Estimate GFLOPS usage based on processing time and model type"""
        base_multipliers = {
            'lightweight': 50,
            'default': 100,
            'complex': 200,
            'neural_network': 300,
            'large_language_model': 400
        }
        
        multiplier = base_multipliers.get(model_type, 100)
        
        # Estimate based on processing time (inverse relationship)
        if processing_time > 0:
            estimated = min(multiplier / processing_time, self.M2_SPECS['max_gflops'])
            return max(estimated, 1.0)
        
        return 1.0
    
    def _cleanup_tracker(self, tracker_name: str, max_entries: int):
        """Clean up tracker to maintain reasonable memory usage"""
        if len(self._jarvis_trackers[tracker_name]) > max_entries:
            self._jarvis_trackers[tracker_name] = \
                self._jarvis_trackers[tracker_name][-max_entries:]
    
    def _update_jarvis_metrics(self):
        """Update current JARVIS metrics"""
        try:
            current_time = time.time()
            hour_ago = current_time - 3600
            
            # Calculate AI processing metrics
            recent_ai = [req for req in self._jarvis_trackers['ai_requests'] 
                        if req['timestamp'] > hour_ago]
            
            if recent_ai:
                ai_success_rate = sum(1 for req in recent_ai if req['success']) / len(recent_ai) * 100
                avg_processing_time = np.mean([req['processing_time'] for req in recent_ai])
                current_gflops = np.mean([req['estimated_gflops'] for req in recent_ai])
            else:
                ai_success_rate = 100.0
                avg_processing_time = 0.0
                current_gflops = 0.0
            
            # Calculate device control metrics
            recent_devices = [cmd for cmd in self._jarvis_trackers['device_commands'] 
                            if cmd['timestamp'] > hour_ago]
            
            device_success_rate = (sum(1 for cmd in recent_devices if cmd['success']) / 
                                 len(recent_devices) * 100) if recent_devices else 100.0
            
            # Calculate voice recognition metrics
            recent_voice = [cmd for cmd in self._jarvis_trackers['voice_commands'] 
                          if cmd['timestamp'] > hour_ago]
            
            voice_accuracy = (sum(1 for cmd in recent_voice if cmd['recognized']) / 
                            len(recent_voice) * 100) if recent_voice else 100.0
            
            # Calculate response times
            recent_responses = [resp for resp in self._jarvis_trackers['response_times'] 
                              if resp['timestamp'] > hour_ago]
            
            avg_response_time = (np.mean([resp['response_time'] for resp in recent_responses]) 
                               if recent_responses else 0.0)
            
            # Calculate neural network efficiency
            neural_efficiency = min((current_gflops / self.M2_SPECS['max_gflops']) * 100, 100.0)
            
            # Update current JARVIS metrics
            with self._metrics_lock:
                self._current_jarvis_metrics = JARVISMetrics(
                    ai_processing_time=avg_processing_time,
                    ai_success_rate=ai_success_rate,
                    device_control_success_rate=device_success_rate,
                    voice_recognition_accuracy=voice_accuracy,
                    m2_gflops_current=current_gflops,
                    m2_gflops_peak=self.M2_SPECS['max_gflops'],
                    neural_network_efficiency=neural_efficiency,
                    response_latency=avg_response_time
                )
            
            # Store in database
            if self._current_jarvis_metrics:
                self.database.store_jarvis_metrics(self._current_jarvis_metrics)
            
        except Exception as e:
            self.logger.error(f"JARVIS metrics update error: {e}")
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            with self._metrics_lock:
                current_time = datetime.now()
                
                # Get recent data from database
                m2_history, jarvis_history = self.database.get_recent_metrics(hours=24)
                
                report = {
                    'report_timestamp': current_time.isoformat(),
                    'system_info': self._get_system_info(),
                    'current_status': {
                        'm2_metrics': asdict(self._current_m2_metrics) if self._current_m2_metrics else {},
                        'jarvis_metrics': asdict(self._current_jarvis_metrics) if self._current_jarvis_metrics else {},
                        'monitoring_active': self._monitoring
                    },
                    'performance_analysis': self._analyze_performance_trends(m2_history, jarvis_history),
                    'm2_optimization': self._get_m2_optimization_insights(),
                    'jarvis_insights': self._get_jarvis_insights(),
                    'health_score': self._calculate_health_score(),
                    'recommendations': self._generate_smart_recommendations(),
                    'alerts': self._get_active_alerts(),
                    'statistics': self._generate_statistics(m2_history, jarvis_history)
                }
                
                return report
                
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive M2 system information"""
        try:
            # Get macOS version
            macos_version = subprocess.run(['sw_vers', '-productVersion'], 
                                         capture_output=True, text=True).stdout.strip()
            
            # Get system uptime
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            
            return {
                'hardware': {
                    'model': 'MacBook Air',
                    'chip': 'Apple M2',
                    'architecture': 'ARM64',
                    'efficiency_cores': self.M2_SPECS['efficiency_cores'],
                    'performance_cores': self.M2_SPECS['performance_cores'],
                    'max_gflops': self.M2_SPECS['max_gflops'],
                    'neural_engine_tops': self.M2_SPECS['neural_engine_tops'],
                    'max_memory_bandwidth': f"{self.M2_SPECS['max_memory_bandwidth_gbps']} GB/s"
                },
                'software': {
                    'macos_version': macos_version,
                    'python_version': sys.version.split()[0],
                    'uptime_hours': uptime / 3600,
                    'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                    'available_memory_gb': psutil.virtual_memory().available / (1024**3)
                },
                'jarvis_version': '2.0 Perfect Edition',
                'analytics_version': '1.0'
            }
            
        except Exception as e:
            self.logger.error(f"System info error: {e}")
            return {}
    
    def _analyze_performance_trends(self, m2_history: List[M2Metrics], 
                                  jarvis_history: List[JARVISMetrics]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        try:
            if len(m2_history) < 10:
                return {'status': 'Insufficient data for trend analysis', 'data_points': len(m2_history)}
            
            # Convert to numpy arrays for analysis
            timestamps = np.array([m.timestamp for m in m2_history])
            cpu_usage = np.array([m.cpu_percent for m in m2_history])
            memory_usage = np.array([m.memory_percent for m in m2_history])
            gpu_usage = np.array([m.gpu_utilization for m in m2_history])
            power_usage = np.array([m.power_consumption for m in m2_history])
            
            # Calculate trends using linear regression
            def calculate_trend(x, y):
                if len(x) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    return 'improving' if slope < -0.1 else 'degrading' if slope > 0.1 else 'stable'
                return 'unknown'
            
            trends = {
                'cpu_trend': calculate_trend(timestamps, cpu_usage),
                'memory_trend': calculate_trend(timestamps, memory_usage),
                'gpu_trend': calculate_trend(timestamps, gpu_usage),
                'power_trend': calculate_trend(timestamps, power_usage),
                'data_points_analyzed': len(m2_history),
                'analysis_period_hours': (timestamps[-1] - timestamps[0]) / 3600 if len(timestamps) > 1 else 0
            }
            
            # Add statistical analysis
            trends.update({
                'cpu_stats': {
                    'mean': float(np.mean(cpu_usage)),
                    'max': float(np.max(cpu_usage)),
                    'min': float(np.min(cpu_usage)),
                    'std': float(np.std(cpu_usage))
                },
                'memory_stats': {
                    'mean': float(np.mean(memory_usage)),
                    'max': float(np.max(memory_usage)),
                    'min': float(np.min(memory_usage)),
                    'std': float(np.std(memory_usage))
                },
                'power_stats': {
                    'mean': float(np.mean(power_usage)),
                    'max': float(np.max(power_usage)),
                    'min': float(np.min(power_usage)),
                    'efficiency_rating': self._calculate_power_efficiency(cpu_usage, power_usage)
                }
            })
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Trend analysis error: {e}")
            return {'error': str(e)}
    
    def _calculate_power_efficiency(self, cpu_usage: np.ndarray, power_usage: np.ndarray) -> str:
        """Calculate M2 power efficiency rating"""
        try:
            if len(cpu_usage) == 0 or len(power_usage) == 0:
                return 'unknown'
            
            # Calculate performance per watt
            avg_perf_per_watt = np.mean(cpu_usage) / np.mean(power_usage) if np.mean(power_usage) > 0 else 0
            
            if avg_perf_per_watt > 8:
                return 'excellent'
            elif avg_perf_per_watt > 6:
                return 'good'
            elif avg_perf_per_watt > 4:
                return 'average'
            else:
                return 'poor'
                
        except:
            return 'unknown'
    
    def _get_m2_optimization_insights(self) -> Dict[str, Any]:
        """Get M2-specific optimization insights"""
        try:
            if not self._current_m2_metrics:
                return {'status': 'No current metrics available'}
            
            metrics = self._current_m2_metrics
            insights = {}
            
            # Core utilization analysis
            total_cores = metrics.efficiency_cores_active + metrics.performance_cores_active
            core_utilization = total_cores / 8.0 * 100  # M2 has 8 cores total
            
            insights['core_utilization'] = {
                'efficiency_cores_active': metrics.efficiency_cores_active,
                'performance_cores_active': metrics.performance_cores_active,
                'total_utilization_percent': core_utilization,
                'optimization_status': 'optimal' if core_utilization < 80 else 'high_usage'
            }
            
            # Neural Engine analysis
            insights['neural_engine'] = {
                'current_usage_percent': metrics.neural_engine_usage,
                'estimated_tops': metrics.neural_engine_usage / 100 * self.M2_SPECS['neural_engine_tops'],
                'optimization_potential': 'high' if metrics.neural_engine_usage < 50 else 'moderate'
            }
            
            # Memory bandwidth analysis
            insights['memory_bandwidth'] = {
                'utilization_percent': metrics.bandwidth_utilization,
                'estimated_gbps': metrics.bandwidth_utilization / 100 * self.M2_SPECS['max_memory_bandwidth_gbps'],
                'efficiency_rating': self._rate_bandwidth_efficiency(metrics.bandwidth_utilization)
            }
            
            # Thermal analysis
            insights['thermal_performance'] = {
                'cpu_temp_status': self._rate_temperature(metrics.cpu_temperature),
                'gpu_temp_status': self._rate_temperature(metrics.gpu_temperature),
                'thermal_state': metrics.thermal_state,
                'throttling_risk': 'high' if max(metrics.cpu_temperature, metrics.gpu_temperature) > 80 else 'low'
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"M2 optimization insights error: {e}")
            return {'error': str(e)}
    
    def _rate_temperature(self, temp: float) -> str:
        """Rate temperature status"""
        if temp == 0:
            return 'unknown'
        elif temp < 60:
            return 'cool'
        elif temp < 75:
            return 'normal'
        elif temp < 85:
            return 'warm'
        else:
            return 'hot'
    
    def _rate_bandwidth_efficiency(self, utilization: float) -> str:
        """Rate memory bandwidth efficiency"""
        if utilization < 30:
            return 'underutilized'
        elif utilization < 70:
            return 'optimal'
        elif utilization < 90:
            return 'high'
        else:
            return 'saturated'
    
    def _get_jarvis_insights(self) -> Dict[str, Any]:
        """Get JARVIS-specific performance insights"""
        try:
            if not self._current_jarvis_metrics:
                return {'status': 'No current metrics available'}
            
            metrics = self._current_jarvis_metrics
            
            insights = {
                'ai_performance': {
                    'processing_speed_rating': self._rate_processing_speed(metrics.ai_processing_time),
                    'success_rate_status': self._rate_success_rate(metrics.ai_success_rate),
                    'current_gflops_utilization': f"{metrics.m2_gflops_current:.1f} / {metrics.m2_gflops_peak} GFLOPS",
                    'neural_efficiency_percent': metrics.neural_network_efficiency
                },
                'interaction_quality': {
                    'voice_accuracy_rating': self._rate_accuracy(metrics.voice_recognition_accuracy),
                    'device_control_rating': self._rate_success_rate(metrics.device_control_success_rate),
                    'response_time_rating': self._rate_response_time(metrics.response_latency)
                },
                'optimization_opportunities': self._identify_optimization_opportunities(metrics)
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"JARVIS insights error: {e}")
            return {'error': str(e)}
    
    def _rate_processing_speed(self, time: float) -> str:
        """Rate AI processing speed"""
        if time < 0.5:
            return 'excellent'
        elif time < 1.0:
            return 'good'
        elif time < 2.0:
            return 'average'
        else:
            return 'slow'
    
    def _rate_success_rate(self, rate: float) -> str:
        """Rate success rate"""
        if rate >= 98:
            return 'excellent'
        elif rate >= 95:
            return 'good'
        elif rate >= 90:
            return 'average'
        else:
            return 'needs_improvement'
    
    def _rate_accuracy(self, accuracy: float) -> str:
        """Rate accuracy"""
        if accuracy >= 98:
            return 'excellent'
        elif accuracy >= 95:
            return 'good'
        elif accuracy >= 90:
            return 'average'
        else:
            return 'needs_improvement'
    
    def _rate_response_time(self, time: float) -> str:
        """Rate response time"""
        if time < 0.1:
            return 'instant'
        elif time < 0.5:
            return 'excellent'
        elif time < 1.0:
            return 'good'
        elif time < 2.0:
            return 'average'
        else:
            return 'slow'
    
    def _identify_optimization_opportunities(self, metrics: JARVISMetrics) -> List[str]:
        """Identify optimization opportunities"""
        opportunities = []
        
        if metrics.ai_processing_time > 1.0:
            opportunities.append("AI processing time can be optimized with model quantization")
        
        if metrics.neural_network_efficiency < 70:
            opportunities.append("Neural Engine utilization can be increased")
        
        if metrics.response_latency > 0.5:
            opportunities.append("Response latency can be reduced with caching")
        
        if metrics.voice_recognition_accuracy < 95:
            opportunities.append("Voice recognition can be improved with better acoustic models")
        
        if metrics.device_control_success_rate < 95:
            opportunities.append("Device control reliability can be enhanced")
        
        if not opportunities:
            opportunities.append("System is running optimally - no immediate improvements needed")
        
        return opportunities
    
    def _calculate_health_score(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        try:
            if not self._current_m2_metrics or not self._current_jarvis_metrics:
                return {'score': 0, 'status': 'insufficient_data'}
            
            m2 = self._current_m2_metrics
            jarvis = self._current_jarvis_metrics
            
            # M2 health components (0-100 each)
            cpu_health = max(0, 100 - m2.cpu_percent)
            memory_health = max(0, 100 - m2.memory_percent)
            temp_health = max(0, 100 - max(m2.cpu_temperature, m2.gpu_temperature))
            power_health = max(0, 100 - (m2.power_consumption / self.M2_SPECS['max_power_watts'] * 100))
            
            # JARVIS health components
            ai_health = jarvis.ai_success_rate
            device_health = jarvis.device_control_success_rate
            voice_health = jarvis.voice_recognition_accuracy
            response_health = max(0, 100 - (jarvis.response_latency * 50))  # Penalize slow responses
            
            # Weighted average
            m2_score = (cpu_health * 0.3 + memory_health * 0.3 + temp_health * 0.2 + power_health * 0.2)
            jarvis_score = (ai_health * 0.3 + device_health * 0.25 + voice_health * 0.25 + response_health * 0.2)
            
            overall_score = (m2_score * 0.6 + jarvis_score * 0.4)  # M2 hardware is more critical
            
            # Determine status
            if overall_score >= 90:
                status = 'excellent'
            elif overall_score >= 80:
                status = 'good'
            elif overall_score >= 70:
                status = 'fair'
            elif overall_score >= 60:
                status = 'poor'
            else:
                status = 'critical'
            
            return {
                'overall_score': round(overall_score, 1),
                'status': status,
                'component_scores': {
                    'm2_hardware': round(m2_score, 1),
                    'jarvis_software': round(jarvis_score, 1)
                },
                'detailed_scores': {
                    'cpu_health': round(cpu_health, 1),
                    'memory_health': round(memory_health, 1),
                    'thermal_health': round(temp_health, 1),
                    'power_efficiency': round(power_health, 1),
                    'ai_performance': round(ai_health, 1),
                    'device_reliability': round(device_health, 1),
                    'voice_accuracy': round(voice_health, 1),
                    'response_speed': round(response_health, 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Health score calculation error: {e}")
            return {'score': 0, 'status': 'error', 'error': str(e)}
    
    def _generate_smart_recommendations(self) -> List[Dict[str, str]]:
        """Generate intelligent recommendations based on current state"""
        recommendations = []
        
        try:
            if not self._current_m2_metrics or not self._current_jarvis_metrics:
                return [{'priority': 'info', 'category': 'system', 
                        'message': 'Insufficient data for recommendations'}]
            
            m2 = self._current_m2_metrics
            jarvis = self._current_jarvis_metrics
            
            # High priority recommendations
            if max(m2.cpu_temperature, m2.gpu_temperature) > 85:
                recommendations.append({
                    'priority': 'high',
                    'category': 'thermal',
                    'message': 'System running hot - ensure proper ventilation and consider reducing workload'
                })
            
            if m2.memory_pressure in ['warn', 'urgent']:
                recommendations.append({
                    'priority': 'high',
                    'category': 'memory',
                    'message': f'Memory pressure is {m2.memory_pressure} - close unnecessary applications'
                })
            
            if jarvis.ai_success_rate < 90:
                recommendations.append({
                    'priority': 'high',
                    'category': 'ai',
                    'message': 'AI success rate is low - check model integrity and system resources'
                })
            
            # Medium priority recommendations
            if m2.power_consumption > 18:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'power',
                    'message': 'High power consumption detected - optimize background processes'
                })
            
            if jarvis.response_latency > 1.0:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'performance',
                    'message': 'Response times are slow - consider optimizing AI models or caching'
                })
            
            if m2.neural_engine_usage < 20 and jarvis.ai_processing_time > 0.5:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'optimization',
                    'message': 'Neural Engine is underutilized - migrate AI workloads for better performance'
                })
            
            # Low priority recommendations
            if m2.cpu_percent < 30 and m2.memory_percent < 50:
                recommendations.append({
                    'priority': 'low',
                    'category': 'optimization',
                    'message': 'System has spare capacity - consider enabling additional JARVIS features'
                })
            
            if jarvis.voice_recognition_accuracy < 98:
                recommendations.append({
                    'priority': 'low',
                    'category': 'voice',
                    'message': 'Voice recognition can be improved with environment optimization'
                })
            
            # Positive feedback
            if (jarvis.ai_success_rate > 95 and jarvis.voice_recognition_accuracy > 95 and 
                jarvis.device_control_success_rate > 95):
                recommendations.append({
                    'priority': 'info',
                    'category': 'status',
                    'message': 'JARVIS is performing excellently across all metrics!'
                })
            
            return recommendations if recommendations else [{
                'priority': 'info',
                'category': 'status',
                'message': 'All systems optimal - no recommendations at this time'
            }]
            
        except Exception as e:
            self.logger.error(f"Smart recommendations error: {e}")
            return [{'priority': 'error', 'category': 'system', 'message': f'Error: {str(e)}'}]
    
    def _get_active_alerts(self) -> List[Dict[str, str]]:
        """Get currently active system alerts"""
        alerts = []
        
        try:
            if not self._current_m2_metrics:
                return alerts
            
            m2 = self._current_m2_metrics
            current_time = datetime.now()
            
            # Critical alerts
            if max(m2.cpu_temperature, m2.gpu_temperature) > 90:
                alerts.append({
                    'level': 'critical',
                    'type': 'thermal',
                    'message': f'Critical temperature: CPU {m2.cpu_temperature}°C, GPU {m2.gpu_temperature}°C',
                    'timestamp': current_time.isoformat()
                })
            
            if m2.memory_pressure == 'urgent':
                alerts.append({
                    'level': 'critical',
                    'type': 'memory',
                    'message': 'Urgent memory pressure - system may become unresponsive',
                    'timestamp': current_time.isoformat()
                })
            
            # Warning alerts
            if m2.power_consumption > 20:
                alerts.append({
                    'level': 'warning',
                    'type': 'power',
                    'message': f'High power consumption: {m2.power_consumption:.1f}W',
                    'timestamp': current_time.isoformat()
                })
            
            if m2.swap_usage > 50:
                alerts.append({
                    'level': 'warning',
                    'type': 'memory',
                    'message': f'High swap usage: {m2.swap_usage:.1f}%',
                    'timestamp': current_time.isoformat()
                })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Active alerts error: {e}")
            return [{'level': 'error', 'type': 'system', 'message': str(e)}]
    
    def _generate_statistics(self, m2_history: List[M2Metrics], 
                           jarvis_history: List[JARVISMetrics]) -> Dict[str, Any]:
        """Generate comprehensive statistics"""
        try:
            stats = {
                'monitoring': {
                    'total_m2_samples': len(m2_history),
                    'total_jarvis_samples': len(jarvis_history),
                    'data_coverage_hours': (m2_history[-1].timestamp - m2_history[0].timestamp) / 3600 if m2_history else 0,
                    'monitoring_uptime_percent': 100.0 if self._monitoring else 0.0
                },
                'performance_records': {},
                'efficiency_metrics': {}
            }
            
            if m2_history:
                cpu_data = [m.cpu_percent for m in m2_history]
                power_data = [m.power_consumption for m in m2_history]
                temp_data = [max(m.cpu_temperature, m.gpu_temperature) for m in m2_history if m.cpu_temperature > 0]
                
                stats['performance_records'] = {
                    'peak_cpu_usage': max(cpu_data),
                    'peak_power_consumption': max(power_data),
                    'peak_temperature': max(temp_data) if temp_data else 0,
                    'average_cpu_usage': np.mean(cpu_data),
                    'average_power_consumption': np.mean(power_data)
                }
                
                stats['efficiency_metrics'] = {
                    'power_efficiency_score': self._calculate_power_efficiency(np.array(cpu_data), np.array(power_data)),
                    'thermal_management_rating': 'excellent' if max(temp_data) < 80 else 'good' if max(temp_data) < 85 else 'needs_attention' if temp_data else 'unknown',
                    'resource_utilization_balance': self._calculate_resource_balance(m2_history)
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Statistics generation error: {e}")
            return {'error': str(e)}
    
    def _calculate_resource_balance(self, m2_history: List[M2Metrics]) -> str:
        """Calculate resource utilization balance"""
        try:
            if not m2_history:
                return 'unknown'
            
            cpu_usage = np.mean([m.cpu_percent for m in m2_history])
            memory_usage = np.mean([m.memory_percent for m in m2_history])
            gpu_usage = np.mean([m.gpu_utilization for m in m2_history])
            
            # Calculate balance score (lower variance = better balance)
            balance_variance = np.var([cpu_usage, memory_usage, gpu_usage])
            
            if balance_variance < 100:
                return 'excellent'
            elif balance_variance < 300:
                return 'good'
            elif balance_variance < 600:
                return 'fair'
            else:
                return 'unbalanced'
                
        except:
            return 'unknown'
    
    def stop_monitoring(self):
        """Stop monitoring gracefully"""
        try:
            self.logger.info("🛑 Stopping M2 performance monitoring...")
            self._monitoring = False
            
            if self._monitor_task:
                self._monitor_task.cancel()
            
            # Final database cleanup
            self.database.cleanup_old_data()
            
            self.logger.info("✅ M2 monitoring stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping monitoring: {e}")
    
    def export_data(self, format: str = 'json', filepath: Optional[str] = None) -> str:
        """Export performance data"""
        try:
            report = self.get_comprehensive_report()
            
            if filepath is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = f"jarvis_m2_report_{timestamp}.{format}"
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"📁 Data exported to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            raise

async def main():
    """Main function for testing M2 Performance Analytics"""
    print("🎯 JARVIS M2 Performance Analytics - Perfect Edition")
    print("=" * 60)
    
    analytics = M2PerformanceAnalytics()
    
    try:
        # Start monitoring
        if await analytics.start_monitoring(interval=5.0):
            print("✅ M2 monitoring started successfully!")
            
            # Simulate some JARVIS operations
            print("\n🧪 Simulating JARVIS operations...")
            
            analytics.track_ai_processing(0.8, True, "neural_network")
            analytics.track_device_command("smart_light_control", True, 0.2)
            analytics.track_voice_command(True, 0.95, 0.3)
            analytics.track_response_time(0.4, "complex_query")
            
            # Wait for some monitoring data
            print("⏳ Collecting performance data...")
            await asyncio.sleep(10)
            
            # Generate comprehensive report
            print("\n📊 Generating comprehensive report...")
            report = analytics.get_comprehensive_report()
            
            # Display key metrics
            if report.get('current_status', {}).get('m2_metrics'):
                m2 = report['current_status']['m2_metrics']
                print(f"\n🔧 M2 Metrics:")
                print(f"   CPU Usage: {m2.get('cpu_percent', 0):.1f}%")
                print(f"   Memory Usage: {m2.get('memory_percent', 0):.1f}%")
                print(f"   GPU Usage: {m2.get('gpu_utilization', 0):.1f}%")
                print(f"   Power: {m2.get('power_consumption', 0):.1f}W")
                print(f"   Neural Engine: {m2.get('neural_engine_usage', 0):.1f}%")
            
            if report.get('current_status', {}).get('jarvis_metrics'):
                jarvis = report['current_status']['jarvis_metrics']
                print(f"\n🤖 JARVIS Metrics:")
                print(f"   AI Success Rate: {jarvis.get('ai_success_rate', 0):.1f}%")
                print(f"   Voice Accuracy: {jarvis.get('voice_recognition_accuracy', 0):.1f}%")
                print(f"   Device Control: {jarvis.get('device_control_success_rate', 0):.1f}%")
                print(f"   Current GFLOPS: {jarvis.get('m2_gflops_current', 0):.1f}")
            
            # Display health score
            health = report.get('health_score', {})
            print(f"\n💚 System Health: {health.get('overall_score', 0)}/100 ({health.get('status', 'unknown').upper()})")
            
            # Display recommendations
            recommendations = report.get('recommendations', [])[:3]  # Top 3
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                priority = rec.get('priority', 'info').upper()
                category = rec.get('category', 'system').upper()
                message = rec.get('message', 'No message')
                print(f"   {i}. [{priority}] {category}: {message}")
            
            # Display alerts if any
            alerts = report.get('alerts', [])
            if alerts:
                print(f"\n🚨 Active Alerts:")
                for alert in alerts:
                    level = alert.get('level', 'info').upper()
                    alert_type = alert.get('type', 'system').upper()
                    message = alert.get('message', 'No message')
                    print(f"   [{level}] {alert_type}: {message}")
            
            # Export report
            print(f"\n📁 Exporting performance report...")
            export_path = analytics.export_data('json')
            print(f"   Report saved to: {export_path}")
            
            print(f"\n🎉 M2 Performance Analytics completed successfully!")
            print(f"   Monitoring will continue in background...")
            
        else:
            print("❌ Failed to start M2 monitoring!")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        analytics.stop_monitoring()

if __name__ == '__main__':
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)