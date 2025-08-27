#!/usr/bin/env python3
"""
JARVIS Performance Analytics System v3.0 - Phase 2 Ultimate (STANDALONE)
Advanced real-time performance monitoring - Ready to run immediately
Comprehensive system metrics, analysis, and optimization recommendations
"""

import asyncio
import psutil
import sqlite3
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import numpy as np
from collections import deque, defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class MetricCategory(Enum):
    """Performance metric categories"""
    SYSTEM = "system"
    AI = "ai"
    DEVICE = "device"
    VOICE = "voice"
    NETWORK = "network"
    BATTERY = "battery"

class AlertLevel(Enum):
    """Performance alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    category: MetricCategory
    name: str
    value: float
    unit: str
    timestamp: datetime
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

@dataclass
class SystemSnapshot:
    """Complete system performance snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    battery_level: Optional[float]
    temperature: Optional[float]
    network_sent: int
    network_recv: int
    active_processes: int
    ai_response_time: Optional[float]
    device_commands_pending: int
    glyph_status: bool
    voice_processing_active: bool

@dataclass
class PerformanceAlert:
    """Performance alert data"""
    level: AlertLevel
    category: MetricCategory
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    suggestions: List[str]
    auto_fix_available: bool

class SystemMonitor:
    """Advanced system performance monitoring"""
    
    def __init__(self):
        self.monitoring_active = False
        self.sample_interval = 1.0  # seconds
        self.metric_history = defaultdict(lambda: deque(maxlen=3600))  # 1 hour of data
        self.network_baseline = None
        self.setup_network_baseline()
    
    def setup_network_baseline(self):
        """Setup network monitoring baseline"""
        try:
            net_io = psutil.net_io_counters()
            self.network_baseline = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'timestamp': time.time()
            }
        except:
            self.network_baseline = {'bytes_sent': 0, 'bytes_recv': 0, 'timestamp': time.time()}
    
    async def get_system_snapshot(self) -> SystemSnapshot:
        """Get comprehensive system performance snapshot"""
        try:
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network metrics
            net_io = psutil.net_io_counters()
            if self.network_baseline:
                time_diff = time.time() - self.network_baseline['timestamp']
                if time_diff > 0:
                    network_sent = int((net_io.bytes_sent - self.network_baseline['bytes_sent']) / time_diff)
                    network_recv = int((net_io.bytes_recv - self.network_baseline['bytes_recv']) / time_diff)
                else:
                    network_sent = network_recv = 0
            else:
                network_sent = network_recv = 0
            
            # Process count
            active_processes = len(psutil.pids())
            
            # Battery (if available)
            battery_level = None
            try:
                battery = psutil.sensors_battery()
                if battery:
                    battery_level = battery.percent
            except:
                pass
            
            # Temperature (if available)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get first available temperature sensor
                    for sensor_name, sensor_list in temps.items():
                        if sensor_list:
                            temperature = sensor_list[0].current
                            break
            except:
                pass
            
            return SystemSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                battery_level=battery_level,
                temperature=temperature,
                network_sent=network_sent,
                network_recv=network_recv,
                active_processes=active_processes,
                ai_response_time=None,  # Simulated
                device_commands_pending=0,  # Simulated
                glyph_status=False,  # Simulated
                voice_processing_active=False  # Simulated
            )
            
        except Exception as e:
            # Return default snapshot on error
            return SystemSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_usage_percent=0.0,
                battery_level=None,
                temperature=None,
                network_sent=0,
                network_recv=0,
                active_processes=0,
                ai_response_time=None,
                device_commands_pending=0,
                glyph_status=False,
                voice_processing_active=False
            )
    
    def add_metric_sample(self, metric_name: str, value: float):
        """Add metric sample to history"""
        self.metric_history[metric_name].append({
            'value': value,
            'timestamp': time.time()
        })
    
    def get_metric_trend(self, metric_name: str, duration_minutes: int = 10) -> str:
        """Analyze metric trend over specified duration"""
        if metric_name not in self.metric_history:
            return "unknown"
        
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_samples = [
            sample for sample in self.metric_history[metric_name]
            if sample['timestamp'] >= cutoff_time
        ]
        
        if len(recent_samples) < 3:
            return "insufficient_data"
        
        values = [sample['value'] for sample in recent_samples]
        
        # Linear trend analysis
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        slope = z[0]
        
        if abs(slope) < 0.001:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

class PerformanceAnalytics:
    """Main performance analytics system - STANDALONE VERSION"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.db_path = Path("logs/performance_analytics.db")
        self.monitoring_active = False
        
        # Monitors
        self.system_monitor = SystemMonitor()
        
        # Analytics data
        self.performance_history = []
        self.active_alerts = []
        self.optimization_suggestions = []
        
        # Thresholds
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'temperature_warning': 45.0,
            'temperature_critical': 55.0,
            'battery_warning': 20.0,
            'battery_critical': 10.0,
            'ai_response_warning': 2.0,
            'ai_response_critical': 5.0
        }
        
        self._init_database()
        self.logger.info("📊 Performance Analytics System v3.0 initialized (STANDALONE)")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup performance analytics logging"""
        logger = logging.getLogger('performance_analytics')
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / f'performance_analytics_{datetime.now().strftime("%Y%m%d")}.log')
        file_formatter = logging.Formatter('%(asctime)s | ANALYTICS | %(levelname)s | %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('📊 %(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _init_database(self):
        """Initialize performance analytics database"""
        try:
            self.db_path.parent.mkdir(exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # System snapshots table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        cpu_percent REAL,
                        memory_percent REAL,
                        disk_usage_percent REAL,
                        battery_level REAL,
                        temperature REAL,
                        network_sent INTEGER,
                        network_recv INTEGER,
                        active_processes INTEGER,
                        ai_response_time REAL,
                        device_commands_pending INTEGER,
                        glyph_status BOOLEAN,
                        voice_processing_active BOOLEAN
                    )
                ''')
                
                # Performance metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        category TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        value REAL NOT NULL,
                        unit TEXT,
                        context TEXT
                    )
                ''')
                
                # Alerts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        category TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        current_value REAL,
                        threshold_value REAL,
                        message TEXT,
                        suggestions TEXT,
                        auto_fix_applied BOOLEAN DEFAULT FALSE,
                        resolved BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                # Optimization history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS optimization_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        optimization_type TEXT NOT NULL,
                        description TEXT,
                        before_metrics TEXT,
                        after_metrics TEXT,
                        improvement_percent REAL,
                        success BOOLEAN
                    )
                ''')
                
                conn.commit()
            
            self.logger.info("✅ Performance analytics database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def collect_comprehensive_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system performance metrics"""
        try:
            start_time = time.time()
            
            # Collect system snapshot
            system_snapshot = await self.system_monitor.get_system_snapshot()
            
            # Simulate AI and device metrics for standalone version
            ai_metrics = {
                'avg_response_time': np.random.uniform(0.5, 2.0),
                'queue_size': 0,
                'model_status': 'simulated',
                'inference_rate': np.random.uniform(10, 30),
                'error_rate': np.random.uniform(0, 5),
                'memory_usage_mb': np.random.uniform(100, 500)
            }
            
            device_metrics = {
                'command_queue_size': 0,
                'glyph_active': False,
                'camera_active': False,
                'performance_mode': 'balanced',
                'command_success_rate': np.random.uniform(85, 98),
                'thermal_status': 'normal',
                'battery_optimization': False,
                'root_access_active': True
            }
            
            # Update system snapshot with simulated data
            system_snapshot.ai_response_time = ai_metrics['avg_response_time']
            system_snapshot.device_commands_pending = device_metrics['command_queue_size']
            system_snapshot.glyph_status = device_metrics['glyph_active']
            
            # Store in history
            self.performance_history.append(system_snapshot)
            
            # Keep only recent history (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.performance_history = [
                snapshot for snapshot in self.performance_history
                if snapshot.timestamp >= cutoff_time
            ]
            
            # Save to database
            await self._save_snapshot_to_db(system_snapshot)
            
            collection_time = time.time() - start_time
            
            return {
                'success': True,
                'system_snapshot': asdict(system_snapshot),
                'ai_metrics': ai_metrics,
                'device_metrics': device_metrics,
                'collection_time': collection_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def analyze_performance_trends(self, hours_back: int = 1) -> Dict[str, Any]:
        """Analyze performance trends over specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_snapshots = [
                snapshot for snapshot in self.performance_history
                if snapshot.timestamp >= cutoff_time
            ]
            
            if len(recent_snapshots) < 3:
                return {
                    'success': False,
                    'error': 'Insufficient data for trend analysis',
                    'data_points': len(recent_snapshots)
                }
            
            # Calculate trends for key metrics
            trends = {}
            
            # CPU trend
            cpu_values = [s.cpu_percent for s in recent_snapshots]
            trends['cpu'] = {
                'current': cpu_values[-1],
                'average': statistics.mean(cpu_values),
                'trend': self._calculate_trend(cpu_values),
                'volatility': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            }
            
            # Memory trend  
            memory_values = [s.memory_percent for s in recent_snapshots]
            trends['memory'] = {
                'current': memory_values[-1],
                'average': statistics.mean(memory_values),
                'trend': self._calculate_trend(memory_values),
                'volatility': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            }
            
            # AI response time trend (if available)
            ai_times = [s.ai_response_time for s in recent_snapshots if s.ai_response_time is not None]
            if ai_times:
                trends['ai_response'] = {
                    'current': ai_times[-1],
                    'average': statistics.mean(ai_times),
                    'trend': self._calculate_trend(ai_times),
                    'volatility': statistics.stdev(ai_times) if len(ai_times) > 1 else 0
                }
            
            # Temperature trend (if available)
            temp_values = [s.temperature for s in recent_snapshots if s.temperature is not None]
            if temp_values:
                trends['temperature'] = {
                    'current': temp_values[-1],
                    'average': statistics.mean(temp_values),
                    'trend': self._calculate_trend(temp_values),
                    'volatility': statistics.stdev(temp_values) if len(temp_values) > 1 else 0
                }
            
            return {
                'success': True,
                'trends': trends,
                'analysis_period_hours': hours_back,
                'data_points': len(recent_snapshots),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "stable"
        
        # Linear regression
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        slope = z[0]
        
        if abs(slope) < 0.1:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    async def check_performance_alerts(self) -> List[PerformanceAlert]:
        """Check for performance alerts and thresholds"""
        new_alerts = []
        
        try:
            if not self.performance_history:
                return new_alerts
            
            latest_snapshot = self.performance_history[-1]
            
            # CPU alerts
            if latest_snapshot.cpu_percent >= self.thresholds['cpu_critical']:
                alert = PerformanceAlert(
                    level=AlertLevel.CRITICAL,
                    category=MetricCategory.SYSTEM,
                    metric_name="cpu_usage",
                    current_value=latest_snapshot.cpu_percent,
                    threshold_value=self.thresholds['cpu_critical'],
                    message=f"Critical CPU usage: {latest_snapshot.cpu_percent:.1f}%",
                    timestamp=datetime.now(),
                    suggestions=[
                        "Close unnecessary applications",
                        "Restart high-CPU processes",
                        "Enable battery saver mode"
                    ],
                    auto_fix_available=True
                )
                new_alerts.append(alert)
            elif latest_snapshot.cpu_percent >= self.thresholds['cpu_warning']:
                alert = PerformanceAlert(
                    level=AlertLevel.WARNING,
                    category=MetricCategory.SYSTEM,
                    metric_name="cpu_usage",
                    current_value=latest_snapshot.cpu_percent,
                    threshold_value=self.thresholds['cpu_warning'],
                    message=f"High CPU usage: {latest_snapshot.cpu_percent:.1f}%",
                    timestamp=datetime.now(),
                    suggestions=[
                        "Monitor running processes",
                        "Consider reducing AI inference frequency"
                    ],
                    auto_fix_available=False
                )
                new_alerts.append(alert)
            
            # Memory alerts
            if latest_snapshot.memory_percent >= self.thresholds['memory_critical']:
                alert = PerformanceAlert(
                    level=AlertLevel.CRITICAL,
                    category=MetricCategory.SYSTEM,
                    metric_name="memory_usage",
                    current_value=latest_snapshot.memory_percent,
                    threshold_value=self.thresholds['memory_critical'],
                    message=f"Critical memory usage: {latest_snapshot.memory_percent:.1f}%",
                    timestamp=datetime.now(),
                    suggestions=[
                        "Clear system cache",
                        "Reduce AI model memory usage",
                        "Close background applications"
                    ],
                    auto_fix_available=True
                )
                new_alerts.append(alert)
            
            # Temperature alerts (if available)
            if (latest_snapshot.temperature is not None and 
                latest_snapshot.temperature >= self.thresholds['temperature_critical']):
                alert = PerformanceAlert(
                    level=AlertLevel.CRITICAL,
                    category=MetricCategory.SYSTEM,
                    metric_name="temperature",
                    current_value=latest_snapshot.temperature,
                    threshold_value=self.thresholds['temperature_critical'],
                    message=f"Critical temperature: {latest_snapshot.temperature:.1f}°C",
                    timestamp=datetime.now(),
                    suggestions=[
                        "Reduce CPU performance mode",
                        "Disable intensive AI processing",
                        "Allow device to cool down"
                    ],
                    auto_fix_available=True
                )
                new_alerts.append(alert)
            
            # Battery alerts (if available)
            if (latest_snapshot.battery_level is not None and 
                latest_snapshot.battery_level <= self.thresholds['battery_critical']):
                alert = PerformanceAlert(
                    level=AlertLevel.CRITICAL,
                    category=MetricCategory.BATTERY,
                    metric_name="battery_level",
                    current_value=latest_snapshot.battery_level,
                    threshold_value=self.thresholds['battery_critical'],
                    message=f"Critical battery level: {latest_snapshot.battery_level:.0f}%",
                    timestamp=datetime.now(),
                    suggestions=[
                        "Enable ultra battery saver mode",
                        "Reduce AI processing frequency",
                        "Disable non-essential features"
                    ],
                    auto_fix_available=True
                )
                new_alerts.append(alert)
            
            # Save new alerts to database
            for alert in new_alerts:
                await self._save_alert_to_db(alert)
            
            # Add to active alerts
            self.active_alerts.extend(new_alerts)
            
            return new_alerts
            
        except Exception as e:
            self.logger.error(f"Alert checking failed: {e}")
            return []
    
    async def generate_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Generate AI-powered optimization suggestions"""
        suggestions = []
        
        try:
            if len(self.performance_history) < 10:
                return suggestions
            
            # Analyze recent performance patterns
            recent_snapshots = self.performance_history[-10:]
            
            # CPU optimization suggestions
            avg_cpu = statistics.mean([s.cpu_percent for s in recent_snapshots])
            if avg_cpu > 70:
                suggestions.append({
                    'type': 'cpu_optimization',
                    'priority': 'high',
                    'title': 'High CPU Usage Detected',
                    'description': f'Average CPU usage is {avg_cpu:.1f}%',
                    'actions': [
                        'Reduce AI inference frequency',
                        'Enable CPU performance throttling',
                        'Close background applications'
                    ],
                    'expected_improvement': '15-30%',
                    'auto_applicable': True
                })
            
            # Memory optimization suggestions
            avg_memory = statistics.mean([s.memory_percent for s in recent_snapshots])
            if avg_memory > 80:
                suggestions.append({
                    'type': 'memory_optimization',
                    'priority': 'high',
                    'title': 'High Memory Usage',
                    'description': f'Average memory usage is {avg_memory:.1f}%',
                    'actions': [
                        'Clear system cache',
                        'Reduce AI model memory footprint',
                        'Optimize data structures'
                    ],
                    'expected_improvement': '10-25%',
                    'auto_applicable': True
                })
            
            # AI performance suggestions (simulated)
            ai_times = [s.ai_response_time for s in recent_snapshots if s.ai_response_time is not None]
            if ai_times:
                avg_ai_time = statistics.mean(ai_times)
                if avg_ai_time > 1.5:
                    suggestions.append({
                        'type': 'ai_optimization',
                        'priority': 'medium',
                        'title': 'AI Response Time Optimization',
                        'description': f'Average AI response time is {avg_ai_time:.2f}s',
                        'actions': [
                            'Use smaller model for simple queries',
                            'Implement response caching',
                            'Optimize model quantization'
                        ],
                        'expected_improvement': '30-50%',
                        'auto_applicable': False
                    })
            
            # Battery optimization suggestions
            battery_levels = [s.battery_level for s in recent_snapshots if s.battery_level is not None]
            if battery_levels:
                # Check if battery is declining rapidly
                if len(battery_levels) >= 5:
                    battery_trend = self._calculate_trend(battery_levels)
                    if battery_trend == "decreasing":
                        suggestions.append({
                            'type': 'battery_optimization',
                            'priority': 'medium',
                            'title': 'Battery Optimization',
                            'description': 'Battery level is declining rapidly',
                            'actions': [
                                'Reduce screen brightness',
                                'Enable adaptive performance mode',
                                'Limit background AI processing'
                            ],
                            'expected_improvement': '20-40%',
                            'auto_applicable': True
                        })
            
            self.optimization_suggestions = suggestions
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Optimization suggestion generation failed: {e}")
            return []
    
    async def _save_snapshot_to_db(self, snapshot: SystemSnapshot):
        """Save system snapshot to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO system_snapshots (
                        timestamp, cpu_percent, memory_percent, disk_usage_percent,
                        battery_level, temperature, network_sent, network_recv,
                        active_processes, ai_response_time, device_commands_pending,
                        glyph_status, voice_processing_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.timestamp.isoformat(),
                    snapshot.cpu_percent,
                    snapshot.memory_percent,
                    snapshot.disk_usage_percent,
                    snapshot.battery_level,
                    snapshot.temperature,
                    snapshot.network_sent,
                    snapshot.network_recv,
                    snapshot.active_processes,
                    snapshot.ai_response_time,
                    snapshot.device_commands_pending,
                    snapshot.glyph_status,
                    snapshot.voice_processing_active
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database save error: {e}")
    
    async def _save_alert_to_db(self, alert: PerformanceAlert):
        """Save performance alert to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO performance_alerts (
                        timestamp, level, category, metric_name, current_value,
                        threshold_value, message, suggestions, auto_fix_applied
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.timestamp.isoformat(),
                    alert.level.value,
                    alert.category.value,
                    alert.metric_name,
                    alert.current_value,
                    alert.threshold_value,
                    alert.message,
                    json.dumps(alert.suggestions),
                    False
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Alert save error: {e}")
    
    async def start_continuous_monitoring(self, interval_seconds: float = 5.0):
        """Start continuous performance monitoring"""
        try:
            self.monitoring_active = True
            self.logger.info(f"📊 Starting continuous monitoring (interval: {interval_seconds}s)")
            
            while self.monitoring_active:
                # Collect metrics
                metrics = await self.collect_comprehensive_metrics()
                
                # Check for alerts
                new_alerts = await self.check_performance_alerts()
                if new_alerts:
                    for alert in new_alerts:
                        self.logger.warning(f"🚨 {alert.level.value.upper()}: {alert.message}")
                
                # Generate optimization suggestions periodically
                if len(self.performance_history) % 20 == 0:  # Every 20 samples
                    suggestions = await self.generate_optimization_suggestions()
                    if suggestions:
                        self.logger.info(f"💡 Generated {len(suggestions)} optimization suggestions")
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
            
            self.logger.info("📊 Continuous monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
        finally:
            self.monitoring_active = False
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        self.logger.info("📊 Stopping continuous monitoring...")
    
    async def generate_performance_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            report_snapshots = [
                snapshot for snapshot in self.performance_history
                if snapshot.timestamp >= cutoff_time
            ]
            
            if not report_snapshots:
                return {
                    'success': False,
                    'error': 'No performance data available for report period'
                }
            
            # Calculate summary statistics
            cpu_values = [s.cpu_percent for s in report_snapshots]
            memory_values = [s.memory_percent for s in report_snapshots]
            
            report = {
                'success': True,
                'report_period_hours': hours_back,
                'data_points': len(report_snapshots),
                'generated_at': datetime.now().isoformat(),
                
                'system_performance': {
                    'cpu': {
                        'average': statistics.mean(cpu_values),
                        'max': max(cpu_values),
                        'min': min(cpu_values),
                        'current': cpu_values[-1] if cpu_values else 0
                    },
                    'memory': {
                        'average': statistics.mean(memory_values),
                        'max': max(memory_values),
                        'min': min(memory_values),
                        'current': memory_values[-1] if memory_values else 0
                    }
                },
                
                'alerts_summary': {
                    'total_alerts': len(self.active_alerts),
                    'critical_alerts': len([a for a in self.active_alerts if a.level == AlertLevel.CRITICAL]),
                    'warning_alerts': len([a for a in self.active_alerts if a.level == AlertLevel.WARNING])
                },
                
                'optimization_opportunities': len(self.optimization_suggestions),
                
                'recommendations': self.optimization_suggestions
            }
            
            # Add AI performance summary if available
            ai_times = [s.ai_response_time for s in report_snapshots if s.ai_response_time is not None]
            if ai_times:
                report['ai_performance'] = {
                    'average_response_time': statistics.mean(ai_times),
                    'max_response_time': max(ai_times),
                    'min_response_time': min(ai_times)
                }
            
            # Add temperature summary if available
            temps = [s.temperature for s in report_snapshots if s.temperature is not None]
            if temps:
                report['thermal_performance'] = {
                    'average_temperature': statistics.mean(temps),
                    'max_temperature': max(temps),
                    'min_temperature': min(temps)
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# ========== MAIN EXECUTION ==========

async def main():
    """Main performance analytics execution"""
    try:
        print("\n📊 JARVIS PERFORMANCE ANALYTICS SYSTEM")
        print("=" * 60)
        print("🤖 STANDALONE VERSION - Phase 2 Foundation")
        
        # Initialize analytics system
        analytics = PerformanceAnalytics()
        
        # Collect initial metrics
        print("\n🔍 Collecting initial performance metrics...")
        initial_metrics = await analytics.collect_comprehensive_metrics()
        
        if initial_metrics['success']:
            print("✅ Initial metrics collected successfully")
            
            # Generate performance report
            print("\n📋 Generating performance report...")
            report = await analytics.generate_performance_report(1)  # Last 1 hour
            
            if report['success']:
                print(f"📊 Performance Report Summary:")
                print(f"   CPU Average: {report['system_performance']['cpu']['average']:.1f}%")
                print(f"   Memory Average: {report['system_performance']['memory']['average']:.1f}%")
                print(f"   Data Points: {report['data_points']}")
                print(f"   Active Alerts: {report['alerts_summary']['total_alerts']}")
                print(f"   Optimization Opportunities: {report['optimization_opportunities']}")
            
            # Start monitoring for demo (10 seconds)
            print(f"\n🔄 Starting 10-second monitoring demo...")
            monitoring_task = asyncio.create_task(analytics.start_continuous_monitoring(1.0))
            
            await asyncio.sleep(10)
            
            analytics.stop_monitoring()
            await monitoring_task
            
            print(f"✅ Performance analytics demo completed!")
            
        else:
            print(f"❌ Initial metrics collection failed: {initial_metrics.get('error')}")
        
    except KeyboardInterrupt:
        print("\n👋 Performance analytics interrupted by user")
    except Exception as e:
        print(f"❌ Performance analytics error: {e}")

if __name__ == "__main__":
    print("🔥 JARVIS Performance Analytics System v3.0 - STANDALONE")
    print("🤖 Phase 2 Foundation - Ready for Testing")
    print("=" * 60)
    asyncio.run(main())
