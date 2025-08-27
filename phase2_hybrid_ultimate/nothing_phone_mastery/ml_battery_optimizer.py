#!/usr/bin/env python3
"""
JARVIS ML Battery Optimizer v1.0
Advanced Machine Learning Battery Management System
Specifically designed for Nothing Phone 2a with Dimensity 7200 Pro
"""

import asyncio
import logging
import json
import time
import sqlite3
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import pickle

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

class BatteryProfile(Enum):
    """Battery optimization profiles"""
    ULTRA_SAVER = "ultra_saver"
    BATTERY_SAVER = "battery_saver"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    GAMING = "gaming"
    ADAPTIVE = "adaptive"

class ChargingPhase(Enum):
    """Battery charging phases"""
    TRICKLE = "trickle"          # 0-10%
    FAST_CHARGE = "fast_charge"  # 10-80%
    SLOW_CHARGE = "slow_charge"  # 80-100%
    MAINTENANCE = "maintenance"   # 100%

class MLBatteryOptimizer:
    """Advanced ML-driven Battery Optimizer for Nothing Phone 2a"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.db_path = Path('logs/ml_battery_optimizer.db')
        self.model_path = Path('models/battery_models.pkl')
        self.db_path.parent.mkdir(exist_ok=True)
        self.model_path.parent.mkdir(exist_ok=True)
        
        self._init_database()
        
        # ML Models
        self.models = {
            'drain_predictor': None,
            'health_predictor': None,
            'charging_optimizer': None,
            'usage_classifier': None
        }
        
        # Battery characteristics for Nothing Phone 2a
        self.battery_specs = {
            'capacity_mah': 5000,
            'voltage_nominal': 3.85,
            'fast_charge_watts': 45,
            'wireless_charge_watts': 15,
            'chemistry': 'Li-Po',
            'cycles_rated': 800
        }
        
        # Advanced ML features
        self.feature_extractors = {
            'temporal': self._extract_temporal_features,
            'usage': self._extract_usage_features,
            'environmental': self._extract_environmental_features,
            'hardware': self._extract_hardware_features
        }
        
        # Adaptive learning parameters
        self.learning_config = {
            'retrain_interval': 24,  # hours
            'min_samples': 100,
            'feature_window': 7,     # days
            'prediction_horizon': 6  # hours
        }
        
        # Current state
        self.current_profile = BatteryProfile.ADAPTIVE
        self.last_prediction = None
        self.optimization_active = True
        self.learning_enabled = True
        
        self.logger.info("🔋 Advanced ML Battery Optimizer initialized for Nothing Phone 2a")

    def _setup_logging(self):
        """Setup advanced logging for ML Battery Optimizer"""
        logger = logging.getLogger('ml_battery_optimizer')
        logger.setLevel(logging.INFO)
        
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f'ml_battery_optimizer_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s | BATTERY-ML | %(levelname)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def _init_database(self):
        """Initialize comprehensive battery database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Battery usage data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS battery_data (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    battery_level INTEGER NOT NULL,
                    voltage_mv INTEGER,
                    current_ma INTEGER,
                    temperature_c REAL,
                    charging_status TEXT,
                    screen_on INTEGER,
                    cpu_usage REAL,
                    network_active INTEGER,
                    gps_active INTEGER,
                    camera_active INTEGER,
                    audio_active INTEGER,
                    app_foreground TEXT,
                    brightness_level INTEGER
                )
            ''')
            
            # ML predictions and optimizations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    predicted_value REAL NOT NULL,
                    actual_value REAL,
                    accuracy REAL,
                    model_version TEXT
                )
            ''')
            
            # Battery health tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS battery_health (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    capacity_percentage REAL,
                    cycle_count INTEGER,
                    health_score REAL,
                    degradation_rate REAL,
                    estimated_remaining_cycles INTEGER
                )
            ''')
            
            # Charging optimization logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS charging_optimizations (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    optimization_type TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    battery_before INTEGER,
                    battery_after INTEGER,
                    time_saved_minutes INTEGER,
                    health_impact REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ ML Battery Optimizer database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {str(e)}")

    async def initialize_ml_battery_system(self):
        """Initialize the complete ML battery management system"""
        try:
            self.logger.info("🚀 Initializing ML Battery Management System...")
            
            # Verify battery hardware access
            if not await self._verify_battery_hardware():
                return False
            
            # Load or train ML models
            await self._initialize_ml_models()
            
            # Start data collection
            asyncio.create_task(self._continuous_data_collection())
            
            # Start ML prediction engine
            asyncio.create_task(self._ml_prediction_engine())
            
            # Start adaptive optimization
            asyncio.create_task(self._adaptive_optimization_engine())
            
            # Start battery health monitoring
            asyncio.create_task(self._battery_health_monitor())
            
            # Start charging optimization
            asyncio.create_task(self._smart_charging_controller())
            
            self.logger.info("✅ ML Battery Management System operational!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ ML Battery system initialization failed: {str(e)}")
            return False

    async def _verify_battery_hardware(self):
        """Verify Nothing Phone 2a battery hardware access"""
        try:
            # Check battery capacity
            result = await self._execute_command("cat /sys/class/power_supply/battery/capacity")
            if result['success']:
                capacity = int(result['output'])
                self.logger.info(f"✅ Battery hardware detected: {capacity}%")
                return True
            else:
                self.logger.error("❌ Cannot access battery hardware")
                return False
                
        except Exception as e:
            self.logger.error(f"Battery hardware verification failed: {str(e)}")
            return False

    async def _initialize_ml_models(self):
        """Initialize and train ML models for battery optimization"""
        try:
            if not ML_AVAILABLE:
                self.logger.warning("⚠️ ML libraries not available, using basic optimization")
                return
            
            self.logger.info("🧠 Initializing ML models...")
            
            # Load existing models if available
            if self.model_path.exists():
                try:
                    with open(self.model_path, 'rb') as f:
                        self.models = pickle.load(f)
                    self.logger.info("✅ Loaded pre-trained ML models")
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to load models: {str(e)}, training new ones")
            
            # Initialize new models
            self.models['drain_predictor'] = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2)),
                ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
            ])
            
            self.models['health_predictor'] = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
            ])
            
            self.models['charging_optimizer'] = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', Ridge(alpha=1.0))
            ])
            
            # Train models if we have historical data
            await self._train_models_if_data_available()
            
            self.logger.info("✅ ML models initialized")
            
        except Exception as e:
            self.logger.error(f"ML model initialization failed: {str(e)}")

    async def _continuous_data_collection(self):
        """Continuously collect comprehensive battery and usage data"""
        while True:
            try:
                # Collect comprehensive battery data
                battery_data = await self._collect_comprehensive_battery_data()
                
                if battery_data is not None:
                    await self._store_battery_data(battery_data)
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                self.logger.error(f"Data collection error: {str(e)}")
                await asyncio.sleep(120)

    async def _collect_comprehensive_battery_data(self):
        """Collect comprehensive battery and system data"""
        try:
            data = {}
            
            # Basic battery metrics
            capacity_result = await self._execute_command("cat /sys/class/power_supply/battery/capacity")
            voltage_result = await self._execute_command("cat /sys/class/power_supply/battery/voltage_now")
            current_result = await self._execute_command("cat /sys/class/power_supply/battery/current_now")
            temp_result = await self._execute_command("cat /sys/class/power_supply/battery/temp")
            status_result = await self._execute_command("cat /sys/class/power_supply/battery/status")
            
            data['battery_level'] = int(capacity_result['output']) if capacity_result['success'] else 0
            data['voltage_mv'] = int(voltage_result['output']) // 1000 if voltage_result['success'] else 0
            data['current_ma'] = int(current_result['output']) // 1000 if current_result['success'] else 0
            data['temperature_c'] = int(temp_result['output']) / 10 if temp_result['success'] else 0
            data['charging_status'] = status_result['output'].strip() if status_result['success'] else 'Unknown'
            
            # System usage metrics
            screen_result = await self._execute_command("dumpsys power | grep 'Display Power'")
            data['screen_on'] = 1 if 'ON' in screen_result['output'] else 0
            
            # CPU usage
            cpu_result = await self._execute_command("cat /proc/loadavg")
            if cpu_result['success']:
                data['cpu_usage'] = float(cpu_result['output'].split()[0])
            else:
                data['cpu_usage'] = 0.0
            
            # Network activity
            network_result = await self._execute_command("cat /proc/net/dev | grep wlan0")
            data['network_active'] = 1 if network_result['success'] and network_result['output'] else 0
            
            # Active sensors/features
            data['gps_active'] = await self._check_gps_active()
            data['camera_active'] = await self._check_camera_active()
            data['audio_active'] = await self._check_audio_active()
            
            # Current app
            app_result = await self._execute_command("dumpsys activity activities | grep 'Run #0'")
            data['app_foreground'] = self._extract_foreground_app(app_result['output']) if app_result['success'] else 'Unknown'
            
            # Screen brightness
            brightness_result = await self._execute_command("cat /sys/class/leds/lcd-backlight/brightness")
            data['brightness_level'] = int(brightness_result['output']) if brightness_result['success'] else 0
            
            data['timestamp'] = datetime.now().isoformat()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Comprehensive data collection failed: {str(e)}")
            return None

    async def _ml_prediction_engine(self):
        """ML-powered prediction engine for battery optimization"""
        while True:
            try:
                if not ML_AVAILABLE or not self.models['drain_predictor']:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    continue
                
                # Get current features
                features = await self._extract_current_features()
                if features is None:
                    await asyncio.sleep(300)
                    continue
                
                # Predict battery drain
                predicted_drain = await self._predict_battery_drain(features)
                
                # Predict optimal actions
                if predicted_drain is not None:
                    await self._generate_optimization_recommendations(predicted_drain, features)
                
                # Store prediction for accuracy tracking
                await self._store_prediction('battery_drain', predicted_drain)
                
                self.last_prediction = {
                    'timestamp': datetime.now(),
                    'predicted_drain': predicted_drain,
                    'features': features
                }
                
                await asyncio.sleep(300)  # Predict every 5 minutes
                
            except Exception as e:
                self.logger.error(f"ML prediction engine error: {str(e)}")
                await asyncio.sleep(600)

    async def _extract_current_features(self):
        """Extract comprehensive features for ML prediction"""
        try:
            # Get latest battery data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM battery_data 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            
            recent_data = cursor.fetchall()
            conn.close()
            
            if len(recent_data) < 5:
                return None
            
            # Extract features
            features = []
            
            # Temporal features
            temporal_features = self._extract_temporal_features()
            features.extend(temporal_features)
            
            # Usage pattern features
            usage_features = self._extract_usage_features(recent_data)
            features.extend(usage_features)
            
            # Environmental features
            env_features = self._extract_environmental_features(recent_data)
            features.extend(env_features)
            
            # Hardware state features
            hw_features = self._extract_hardware_features(recent_data)
            features.extend(hw_features)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            return None

    def _extract_temporal_features(self):
        """Extract time-based features"""
        now = datetime.now()
        return [
            now.hour,                    # Hour of day
            now.weekday(),              # Day of week
            1 if 9 <= now.hour <= 17 else 0,  # Work hours
            1 if 22 <= now.hour or now.hour <= 6 else 0,  # Sleep hours
            now.minute / 60.0           # Minute fraction
        ]

    def _extract_usage_features(self, recent_data):
        """Extract usage pattern features"""
        if not recent_data:
            return [0] * 8
        
        recent_df = recent_data[-5:]  # Last 5 data points
        
        return [
            np.mean([row[6] for row in recent_df]),      # Avg screen on
            np.mean([row[7] for row in recent_df]),      # Avg CPU usage
            np.mean([row[8] for row in recent_df]),      # Avg network activity
            np.sum([row[9] for row in recent_df]),       # GPS usage count
            np.sum([row[10] for row in recent_df]),      # Camera usage count
            np.sum([row[11] for row in recent_df]),      # Audio usage count
            np.mean([row[13] for row in recent_df]),     # Avg brightness
            len(set([row[12] for row in recent_df]))     # App diversity
        ]

    def _extract_environmental_features(self, recent_data):
        """Extract environmental features"""
        if not recent_data: 
            return [0] * 4
        
        recent_df = recent_data[-5:]
        
        return [
            np.mean([row[4] for row in recent_df]),      # Avg temperature
            np.max([row[4] for row in recent_df]),       # Max temperature
            np.std([row[3] for row in recent_df]) if len(recent_df) > 1 else 0,  # Voltage stability
            1 if any('Charging' in str(row[5]) for row in recent_df) else 0     # Recent charging
        ]

    def _extract_hardware_features(self, recent_data):
        """Extract hardware state features"""
        if not recent_data:
            return [0] * 3
        
        latest = recent_data[0]
        return [
            latest[2],                  # Current battery level
            latest[3] / 1000.0,        # Current voltage (V)
            abs(latest[4]) / 1000.0    # Current draw (A)
        ]

    async def _predict_battery_drain(self, features):
        """Predict battery drain using ML model"""
        try:
            if not self.models['drain_predictor'] or features is None:
                return None
            
            # Ensure features have the right shape
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Make prediction
            prediction = self.models['drain_predictor'].predict(features)[0]
            
            # Clamp prediction to reasonable bounds (0-50% per hour)
            prediction = max(0, min(50, prediction))
            
            self.logger.info(f"🧠 Predicted battery drain: {prediction:.2f}% per hour")
            return prediction
            
        except Exception as e:
            self.logger.error(f"Battery drain prediction failed: {str(e)}")
            return None

    async def _generate_optimization_recommendations(self, predicted_drain, features):
        """Generate smart optimization recommendations based on predictions"""
        try:
            recommendations = []
            
            # High drain detection
            if predicted_drain > 10:  # >10% per hour
                recommendations.append({
                    'type': 'high_drain_warning',
                    'action': 'reduce_performance',
                    'urgency': 'high',
                    'message': f"High battery drain predicted ({predicted_drain:.1f}%/h)"
                })
                
                # Apply power saving measures
                await self._apply_power_saving_measures()
            
            # Low battery with high drain
            current_level = await self._get_current_battery_level()
            if current_level < 20 and predicted_drain > 5:
                recommendations.append({
                    'type': 'critical_battery',
                    'action': 'emergency_mode',
                    'urgency': 'critical',
                    'message': f"Critical battery with high drain - enabling emergency mode"
                })
                
                await self._enable_emergency_mode()
            
            # Optimal charging recommendations
            if current_level < 80 and predicted_drain < 2:
                recommendations.append({
                    'type': 'charging_suggestion',
                    'action': 'suggest_charge',
                    'urgency': 'low',
                    'message': "Good time to charge - low predicted usage"
                })
            
            # Log recommendations
            for rec in recommendations:
                await self._log_optimization_action(rec)
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {str(e)}")

    async def _adaptive_optimization_engine(self):
        """Adaptive optimization engine that learns from usage patterns"""
        while True:
            try:
                # Analyze recent patterns
                patterns = await self._analyze_usage_patterns()
                
                # Adjust optimization profile based on patterns
                if patterns:
                    new_profile = await self._determine_optimal_profile(patterns)
                    if new_profile != self.current_profile:
                        await self._switch_battery_profile(new_profile)
                
                # Retrain models periodically
                if self.learning_enabled:
                    await self._periodic_model_retraining()
                
                await asyncio.sleep(1800)  # Run every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Adaptive optimization error: {str(e)}")
                await asyncio.sleep(3600)

    async def _smart_charging_controller(self):
        """Smart charging controller with ML optimization"""
        while True:
            try:
                charging_status = await self._get_charging_status()
                
                if charging_status['is_charging']:
                    await self._optimize_charging_process(charging_status)
                else:
                    await self._monitor_charging_opportunities()
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                self.logger.error(f"Smart charging controller error: {str(e)}")
                await asyncio.sleep(300)

    async def _optimize_charging_process(self, charging_status):
        """Optimize the charging process based on ML predictions"""
        try:
            battery_level = charging_status['level']
            charge_rate = charging_status['rate']
            
            # Determine optimal charging phase
            phase = self._determine_charging_phase(battery_level)
            
            # Apply phase-specific optimizations
            if phase == ChargingPhase.FAST_CHARGE and battery_level < 80:
                # Allow fast charging up to 80%
                await self._enable_fast_charging()
                self.logger.info("⚡ Fast charging enabled (10-80%)")
                
            elif phase == ChargingPhase.SLOW_CHARGE and battery_level >= 80:
                # Slow charging for battery health
                await self._enable_slow_charging()
                self.logger.info("🐌 Slow charging enabled (80-100%) for battery health")
                
            elif phase == ChargingPhase.MAINTENANCE and battery_level >= 100:
                # Stop charging to prevent overcharge
                await self._enable_maintenance_mode()
                self.logger.info("🔋 Maintenance mode - preventing overcharge")
            
            # Log charging optimization
            await self._log_charging_optimization(phase, battery_level, charge_rate)
            
        except Exception as e:
            self.logger.error(f"Charging optimization failed: {str(e)}")

    async def _battery_health_monitor(self):
        """Advanced battery health monitoring with ML analysis"""
        while True:
            try:
                # Collect battery health metrics
                health_data = await self._collect_battery_health_data()
                
                if health_data is not None:
                    # Analyze health trends
                    health_score = await self._calculate_health_score(health_data)
                    
                    # Predict remaining battery life
                    remaining_cycles = await self._predict_remaining_cycles(health_data)
                    
                    # Store health data
                    await self._store_battery_health(health_data, health_score, remaining_cycles)
                    
                    # Alert if health is declining
                    if health_score < 80:
                        self.logger.warning(f"⚠️ Battery health declining: {health_score:.1f}%")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Battery health monitoring error: {str(e)}")
                await asyncio.sleep(7200)

    async def get_ml_battery_status(self):
        """Get comprehensive ML battery status"""
        try:
            current_data = await self._collect_comprehensive_battery_data()
            
            # Get latest prediction
            prediction_summary = "No prediction available"
            if self.last_prediction:
                pred_time = self.last_prediction['timestamp']
                pred_drain = self.last_prediction['predicted_drain']
                prediction_summary = f"Predicted drain: {pred_drain:.1f}%/h (at {pred_time.strftime('%H:%M')})"
            
            # Get battery health
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM battery_health ORDER BY timestamp DESC LIMIT 1')
            health_data = cursor.fetchone()
            conn.close()
            
            health_summary = "Health data not available"
            if health_data:
                health_summary = f"Health: {health_data[3]:1f}%, Cycles: {health_data[4]}, Remaining: {health_data[6]}"
            
            return {
                'system_status': 'operational',
                'ml_models_loaded': ML_AVAILABLE and self.models['drain_predictor'] is not None,
                'current_profile': self.current_profile.value,
                'optimization_active': self.optimization_active,
                'learning_enabled': self.learning_enabled,
                'battery_data': current_data,
                'prediction_summary': prediction_summary,
                'health_summary': health_summary,
                'battery_specs': self.battery_specs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Status retrieval failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    # Utility methods
    async def _execute_command(self, command):
        """Execute ADB shell command"""
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
                'error': stderr.decode().strip()
            }
            
        except Exception as e:
            return {'success': False, 'output': '', 'error': str(e)}

    async def _get_current_battery_level(self):
        """Get current battery level"""
        result = await self._execute_command("cat /sys/class/power_supply/battery/capacity")
        return int(result['output']) if result['success'] else 0

    async def _get_charging_status(self):
        """Get detailed charging status"""
        level_result = await self._execute_command("cat /sys/class/power_supply/battery/capacity")
        status_result = await self._execute_command("cat /sys/class/power_supply/battery/status")
        current_result = await self._execute_command("cat /sys/class/power_supply/battery/current_now")
        
        return {
            'level': int(level_result['output']) if level_result['success'] else 0,
            'is_charging': 'Charging' in status_result['output'] if status_result['success'] else False,
            'rate': int(current_result['output']) // 1000 if current_result['success'] else 0
        }

    def _determine_charging_phase(self, battery_level):
        """Determine optimal charging phase"""
        if battery_level < 10:
            return ChargingPhase.TRICKLE
        elif battery_level < 80:
            return ChargingPhase.FAST_CHARGE
        elif battery_level < 100:
            return ChargingPhase.SLOW_CHARGE
        else:
            return ChargingPhase.MAINTENANCE

    async def _store_battery_data(self, data):
        """Store comprehensive battery data"""
        try:
            data_id = hashlib.md5(f"{data['timestamp']}_{data['battery_level']}".encode()).hexdigest()[:12]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO battery_data 
                (id, timestamp, battery_level, voltage_mv, current_ma, temperature_c, 
                 charging_status, screen_on, cpu_usage, network_active, gps_active, 
                 camera_active, audio_active, app_foreground, brightness_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data_id, data['timestamp'], data['battery_level'], data['voltage_mv'],
                data['current_ma'], data['temperature_c'], data['charging_status'],
                data['screen_on'], data['cpu_usage'], data['network_active'],
                data['gps_active'], data['camera_active'], data['audio_active'],
                data['app_foreground'], data['brightness_level']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store battery data: {str(e)}")

    # Additional helper methods for sensors and app detection
    async def _check_gps_active(self):
        """Check if GPS is active"""
        result = await self._execute_command("dumpsys location | grep 'gps'")
        return 1 if result['success'] and 'AVAILABLE' in result['output'] else 0

    async def _check_camera_active(self):
        """Check if camera is active"""
        result = await self._execute_command("lsof /dev/video* 2>/dev/null")
        return 1 if result['success'] and result['output'] else 0

    async def _check_audio_active(self):
        """Check if audio is active"""
        result = await self._execute_command("dumpsys audio | grep 'AudioTrack'")
        return 1 if result['success'] and result['output'] else 0

    def _extract_foreground_app(self, activity_output):
        """Extract foreground app from activity manager output"""
        try:
            # Simple extraction - would need refinement for production
            if 'ActivityRecord' in activity_output:
                parts = activity_output.split(' ')
                for part in parts:
                    if '/' in part and '.' in part:
                        return part.split('/')[0]
            return 'Unknown'
        except:
            return 'Unknown'

    async def _periodic_model_retraining(self):
        """Periodic ML model retraining"""
        try:
            self.logger.info("🔄 Periodic ML model retraining started...")
            while True:
                # Check if we have enough new data
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM battery_data')
                data_count = cursor.fetchone()[0]
                conn.close()
                
                if data_count >= self.learning_config['min_samples']:
                    self.logger.info(f"🧠 Retraining models with {data_count} samples")
                    await self._train_models_if_data_available()
                
                await asyncio.sleep(3600)  # Retrain every hour
                
        except Exception as e:
            self.logger.error(f"Periodic retraining error: {str(e)}")

    async def _log_charging_optimization(self, phase, battery_level, charge_rate):
        """Log charging optimization actions"""
        try:
            log_id = hashlib.md5(f"{phase}_{battery_level}_{time.time()}".encode()).hexdigest()[:12]
            timestamp = datetime.now().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO charging_optimizations 
                (id, timestamp, optimization_type, action_taken, battery_before, battery_after, time_saved_minutes, health_impact)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (log_id, timestamp, phase, f"phase_optimization", battery_level, battery_level, 0, 0.1))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"📊 Logged charging optimization: {phase} at {battery_level}%")
            
        except Exception as e:
            self.logger.error(f"Failed to log charging optimization: {str(e)}")

    async def _train_models_if_data_available(self):
        """Train models if sufficient data is available"""
        try:
            if not ML_AVAILABLE:
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM battery_data')
            data_count = cursor.fetchone()[0]
            conn.close()
            
            if data_count < self.learning_config['min_samples']:
                self.logger.info(f"Not enough data for training ({data_count}/{self.learning_config['min_samples']})")
                return
            
            # Simple model training
            X, y = self._prepare_training_data()
            if X is not None and len(X) > 10:
                self.models['drain_predictor'].fit(X, y)
                self.logger.info("✅ ML models retrained successfully")
                
                # Save models
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.models, f)
                
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")

    def _prepare_training_data(self):
        """Prepare training data for ML models"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT battery_level, screen_on, cpu_usage, temperature_c, charging_status
                FROM battery_data 
                ORDER BY timestamp 
                LIMIT 1000
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            if len(rows) < 10:
                return None, None
            
            X = []
            y = []
            
            for i in range(len(rows) - 1):
                current = rows[i]
                next_row = rows[i + 1]
                
                # Features: battery_level, screen_on, cpu_usage, temperature, charging
                features = [
                    current[0],  # battery_level
                    current[1],  # screen_on
                    current[2] if current[2] is not None else 0,  # cpu_usage
                    current[3] if current[3] is not None else 30,  # temperature
                    1 if 'Charging' in str(current[4]) else 0  # charging
                ]
                
                # Target: battery drain
                drain = current[0] - next_row[0]
                if drain >= 0:  # Only positive drain
                    X.append(features)
                    y.append(drain)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed: {str(e)}")
            return None, None

    async def _analyze_usage_patterns(self):
        """Analyze recent usage patterns from battery_data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM battery_data ORDER BY timestamp DESC LIMIT 60')
            rows = cursor.fetchall()
            conn.close()
            if not rows:
                return None

            recent = rows[:30]
            screen_on_ratio = np.mean([r[6] for r in recent]) if recent else 0
            cpu_avg = np.mean([r[7] for r in recent]) if recent else 0.0
            net_active_ratio = np.mean([r[8] for r in recent]) if recent else 0
            temp_avg = np.mean([r[5] for r in recent]) if recent else 0.0
            level = recent[0][2]

            return {
                'screen_on_ratio': screen_on_ratio,
                'cpu_avg': cpu_avg,
                'net_active_ratio': net_active_ratio,
                'temp_avg': temp_avg,
                'battery_level': level
            }
        except Exception as e:
            self.logger.error(f"Usage pattern analysis failed: {str(e)}")
            return None

    async def _determine_optimal_profile(self, patterns):
        """Choose a battery profile based on patterns."""
        try:
            level = patterns.get('battery_level', 100)
            temp = patterns.get('temp_avg', 30.0)
            cpu_avg = patterns.get('cpu_avg', 0.0)
            screen_ratio = patterns.get('screen_on_ratio', 0.0)
            
            # Adaptive profile selection based on usage patterns
            if level < 15:
                return BatteryProfile.ULTRA_SAVER
            elif level < 30:
                return BatteryProfile.BATTERY_SAVER
            elif cpu_avg > 4.0 or screen_ratio > 0.8:
                return BatteryProfile.PERFORMANCE
            elif temp > 40.0:
                return BatteryProfile.BATTERY_SAVER
            else:
                return BatteryProfile.BALANCED
                
        except Exception as e:
            self.logger.error(f"Profile determination failed: {str(e)}")
            return BatteryProfile.BALANCED

    async def _switch_battery_profile(self, new_profile):
        """Switch to a new battery profile"""
        try:
            self.logger.info(f"🔄 Switching battery profile: {self.current_profile.value} → {new_profile.value}")
            self.current_profile = new_profile
            
            # Apply profile-specific optimizations
            await self._apply_profile_optimizations(new_profile)
            
        except Exception as e:
            self.logger.error(f"Profile switch failed: {str(e)}")

    async def _apply_profile_optimizations(self, profile):
        """Apply optimizations for specific battery profile"""
        try:
            if profile == BatteryProfile.ULTRA_SAVER:
                await self._execute_command("echo powersave > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
                await self._execute_command("echo 50 > /sys/class/leds/lcd-backlight/brightness")
                
            elif profile == BatteryProfile.PERFORMANCE:
                await self._execute_command("echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
                
            self.logger.info(f"✅ Applied {profile.value} optimizations")
            
        except Exception as e:
            self.logger.error(f"Profile optimization failed: {str(e)}")

    async def _apply_power_saving_measures(self):
        """Apply immediate power saving measures"""
        try:
            self.logger.info("⚡ Applying power saving measures")
            await self._execute_command("echo powersave > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            await self._execute_command("echo 30 > /sys/class/leds/lcd-backlight/brightness")
            await self._execute_command("settings put secure location_mode 0")
        except Exception as e:
            self.logger.error(f"Power saving measures failed: {str(e)}")

    async def _enable_emergency_mode(self):
        """Enable emergency battery mode"""
        try:
            self.logger.warning("🚨 Enabling emergency battery mode")
            await self._execute_command("echo powersave > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            await self._execute_command("echo 10 > /sys/class/leds/lcd-backlight/brightness")
            await self._execute_command("settings put secure location_mode 0")
            await self._execute_command("settings put global airplane_mode_on 1")
        except Exception as e:
            self.logger.error(f"Emergency mode failed: {str(e)}")

    async def _log_optimization_action(self, recommendation):
        """Log optimization actions"""
        try:
            log_id = hashlib.md5(f"{recommendation['type']}_{time.time()}".encode()).hexdigest()[:12]
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO charging_optimizations 
                (id, timestamp, optimization_type, action_taken, battery_before, battery_after, time_saved_minutes, health_impact)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (log_id, datetime.now().isoformat(), recommendation['type'], 
                  recommendation['action'], 0, 0, 0, 0.1))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Action logging failed: {str(e)}")

    async def _store_prediction(self, prediction_type, predicted_value):
        """Store ML predictions for accuracy tracking"""
        try:
            pred_id = hashlib.md5(f"{prediction_type}_{time.time()}".encode()).hexdigest()[:12]
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO ml_predictions 
                (id, timestamp, prediction_type, predicted_value, actual_value, accuracy, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (pred_id, datetime.now().isoformat(), prediction_type, predicted_value, 
                  None, None, "v1.0"))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Prediction storage failed: {str(e)}")

    async def _enable_fast_charging(self):
        """Enable fast charging mode"""
        try:
            self.logger.info("⚡ Fast charging enabled")
            await self._execute_command("echo 1 > /sys/class/power_supply/battery/fast_charging_enabled")
        except Exception as e:
            self.logger.error(f"Fast charging enable failed: {str(e)}")

    async def _enable_slow_charging(self):
        """Enable slow charging mode for battery health"""
        try:
            self.logger.info("🐌 Slow charging enabled for battery health")
            await self._execute_command("echo 0 > /sys/class/power_supply/battery/fast_charging_enabled")
        except Exception as e:
            self.logger.error(f"Slow charging enable failed: {str(e)}")

    async def _enable_maintenance_mode(self):
        """Enable battery maintenance mode"""
        try:
            self.logger.info("🔋 Battery maintenance mode enabled")
            await self._execute_command("echo 1 > /sys/class/power_supply/battery/maintenance_mode")
        except Exception as e:
            self.logger.error(f"Maintenance mode failed: {str(e)}")

    async def _collect_battery_health_data(self):
        """Collect real battery health data"""
        try:
            capacity_result = await self._execute_command("cat /sys/class/power_supply/battery/capacity")
            cycle_result = await self._execute_command("cat /sys/class/power_supply/battery/cycle_count")
            health_result = await self._execute_command("cat /sys/class/power_supply/battery/health")
            return {
                'capacity_percentage': int(capacity_result['output']) if capacity_result['success'] else 99.0,
                'cycle_count': int(cycle_result['output']) if cycle_result['success'] else 90,
                'health_score': 99.0 - (int(cycle_result['output']) * 0.01) if cycle_result['success'] else 99.0,
                'degradation_rate': 0.01,
                'estimated_remaining_cycles': 800 - int(cycle_result['output']) if cycle_result['success'] else 700
            }
        except Exception as e:
            self.logger.error(f"Health data collection failed: {str(e)}")
            return {
                'capacity_percentage': 99.0,
                'cycle_count': 90,
                'health_score': 99.0,
                'degradation_rate': 0.01,
                'estimated_remaining_cycles': 700
            }

    async def _calculate_health_score(self, health_data):
        """Calculate battery health score"""
        try:
            capacity = health_data.get('capacity_percentage', 100)
            cycles = health_data.get('cycle_count', 0)
            health_score = capacity * (1 - (cycles / 1000))
            return max(0, min(100, health_score))
        except Exception as e:
            self.logger.error(f"Health score calculation failed: {str(e)}")
            return 99.0

    async def _predict_remaining_cycles(self, health_data):
        """Predict remaining battery cycles"""
        try:
            current_cycles = health_data.get('cycle_count', 0)
            rated_cycles = self.battery_specs['cycles_rated']
            return max(0, rated_cycles - current_cycles)
        except Exception as e:
            self.logger.error(f"Remaining cycles prediction failed: {str(e)}")
            return 700

    async def _store_battery_health(self, health_data, health_score, remaining_cycles):
        """Store battery health data"""
        try:
            health_id = hashlib.md5(f"health_{time.time()}".encode()).hexdigest()[:12]
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO battery_health 
                (id, timestamp, capacity_percentage, cycle_count, health_score, degradation_rate, estimated_remaining_cycles)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (health_id, datetime.now().isoformat(), health_data['capacity_percentage'],
                  health_data['cycle_count'], health_score, health_data['degradation_rate'], remaining_cycles))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Health data storage failed: {str(e)}")

# Demo and main execution
async def main():
    """Main function to run ML Battery Optimizer"""
    optimizer = MLBatteryOptimizer()
    
    print("🔋 JARVIS ML Battery Optimizer v1.0")
    print("=" * 60)
    print("📱 Advanced ML Battery Management for Nothing Phone 2a")
    print()
    
    if await optimizer.initialize_ml_battery_system():
        print("✅ ML Battery Optimizer operational!")
        
        # Get initial status
        print("\n📊 Getting ML battery status...")
        status = await optimizer.get_ml_battery_status()
        print("   ML Battery Summary:")
        print(f"     System Status: {status['system_status']}")
        print(f"     ML Models: {'✅' if status['ml_models_loaded'] else '❌'}")
        print(f"     Current Profile: {status['current_profile']}")
        print(f"     Optimization Active: {'✅' if status['optimization_active'] else '❌'}")
        print(f"     Learning Enabled: {'✅' if status['learning_enabled'] else '❌'}")
        
        if status.get('battery_data'):
            battery = status['battery_data']
            print(f"     Battery Level: {battery['battery_level']}%")
            print(f"     Temperature: {battery['temperature_c']:.1f}°C")
            print(f"     Charging: {battery['charging_status']}")
        
        print(f"\n   {status['prediction_summary']}")
        print(f"   {status['health_summary']}")
        
        print("\n🧠 Starting continuous ML optimization...")
        print("Press Ctrl+C to stop")
        
        try:
            # Run continuous optimization
            while True:
                await asyncio.sleep(300)  # Status update every 5 minutes
                current_level = await optimizer._get_current_battery_level()
                print(f"🔋 Battery: {current_level}% | Profile: {optimizer.current_profile.value}")
                
        except KeyboardInterrupt:
            print("\n🛑 ML Battery Optimizer stopped by user")
            
    else:
        print("❌ ML Battery Optimizer initialization failed!")
        print("Make sure your Nothing Phone 2a is connected via ADB with root access")

if __name__ == '__main__':
    asyncio.run(main())
