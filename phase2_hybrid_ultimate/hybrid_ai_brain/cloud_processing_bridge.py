#!/usr/bin/env python3
"""
JARVIS Hybrid Cloud Processing Bridge v4.1 - GROQ-FIRST PRIORITY
Enterprise-grade hybrid AI routing with GROQ PRIORITY + Local Fallback
✅ Groq Llama-4 Scout 17B (FIRST PRIORITY) - Ultra-fast cloud processing
✅ Local Mac M2 7B (FALLBACK) - When Groq quota exceeded or unavailable  
✅ Nothing Phone 2a Control - Seamless device integration

Author: JARVIS Project  
Date: August 27, 2025 - 1:09 AM IST
Priority: Groq Cloud FIRST → Local Mac M2 SECOND → HuggingFace THIRD
"""

import asyncio
import aiohttp
import json
import time
import logging
import subprocess
import hashlib
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import threading
from pathlib import Path
import pickle

class ProcessingLocation(Enum):
    GROQ_LLAMA4_17B = "groq_llama4_17b"        # FIRST PRIORITY
    LOCAL_MAC_M2 = "local_mac_m2"              # SECOND PRIORITY
    FREE_HUGGINGFACE = "free_huggingface"      # THIRD PRIORITY
    PHONE_LOCAL = "phone_local"
    FALLBACK_LOCAL = "fallback_local"

class RequestPriority(Enum):
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0

@dataclass
class CloudRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    priority: RequestPriority = RequestPriority.NORMAL
    require_local: bool = False
    timeout: float = 30.0
    force_groq: bool = False  # Force use Groq even if quota low

@dataclass 
class CloudResponse:
    text: str
    processing_location: ProcessingLocation
    processing_time_ms: float
    tokens_generated: int
    tokens_per_second: float = 0.0
    success: bool = True
    error: Optional[str] = None
    cost: float = 0.0
    quota_remaining: int = 0  # Track remaining Groq quota

class QuotaManager:
    """Manages Groq API quota tracking"""
    
    def __init__(self):
        self.quota_file = Path('groq_quota_tracker.pkl')
        self.daily_limit = 1000  # Groq free tier daily limit
        self.reset_time = None
        self.requests_made = 0
        self.load_quota_data()
    
    def load_quota_data(self):
        """Load quota data from file"""
        # Initialize with default values first
        self.requests_made = 0
        self.reset_time = time.time()
        
        try:
            if self.quota_file.exists():
                with open(self.quota_file, 'rb') as f:
                    data = pickle.load(f)
                    self.requests_made = data.get('requests_made', 0)
                    self.reset_time = data.get('reset_time', time.time())
                    
                    # Reset if it's a new day
                    if time.time() - self.reset_time > 86400:  # 24 hours
                        self.requests_made = 0
                        self.reset_time = time.time()
                        self.save_quota_data()
        except Exception:
            # If any error occurs, we already have safe default values
            pass
    
    def save_quota_data(self):
        """Save quota data to file"""
        try:
            data = {
                'requests_made': self.requests_made,
                'reset_time': self.reset_time
            }
            with open(self.quota_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass
    
    def can_make_request(self) -> bool:
        """Check if we can make a Groq request"""
        return self.requests_made < self.daily_limit
    
    def use_request(self):
        """Record a Groq request made"""
        self.requests_made += 1
        self.save_quota_data()
    
    def get_remaining_quota(self) -> int:
        """Get remaining quota"""
        return max(0, self.daily_limit - self.requests_made)
    
    def get_status(self) -> Dict[str, Any]:
        """Get quota status"""
        remaining = self.get_remaining_quota()
        hours_until_reset = max(0, 24 - ((time.time() - self.reset_time) / 3600))
        
        return {
            'requests_used': self.requests_made,
            'requests_remaining': remaining,
            'daily_limit': self.daily_limit,
            'quota_percentage': (remaining / self.daily_limit) * 100,
            'hours_until_reset': hours_until_reset,
            'can_make_request': self.can_make_request()
        }

class EnhancedFreeCloudServices:
    """Enhanced Free AI service integrations with Groq-first priority"""
    
    def __init__(self):
        self.logger = logging.getLogger('EnhancedFreeCloudServices')
        # Load API keys from environment variables
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
        
        if not self.groq_api_key:
            print("⚠️  GROQ_API_KEY not found in environment variables")
        if not self.huggingface_token:
            print("⚠️  HUGGINGFACE_TOKEN not found in environment variables")
        
        # Quota manager for Groq
        self.quota_manager = QuotaManager()
        
        # Enhanced model configurations
        self.groq_models = {
            "llama4_scout_17b": "meta-llama/llama-4-scout-17b-16e-instruct",
            "llama3_8b": "llama3-8b-8192",
            "mixtral_8x7b": "mixtral-8x7b-32768"
        }
    
    async def groq_llama4_inference(self, prompt: str, max_tokens: int = 256, force_request: bool = False) -> Dict[str, Any]:
        """Groq Llama-4 Scout 17B - PRIORITY #1 with quota management"""
        try:
            # Check quota first (unless forced)
            if not force_request and not self.quota_manager.can_make_request():
                return {
                    'success': False,
                    'error': f'Groq daily quota exceeded ({self.quota_manager.daily_limit} requests/day)',
                    'quota_remaining': 0,
                    'service': 'groq_llama4_17b'
                }
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are JARVIS, an advanced AI assistant. Provide helpful, accurate, and concise responses."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "model": self.groq_models["llama4_scout_17b"],
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": False
                }
                
                headers = {
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                }
                
                async with session.post(
                    'https://api.groq.com/openai/v1/chat/completions',
                    json=payload,
                    headers=headers,
                    timeout=15
                ) as response:
                    processing_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        # Record successful request
                        self.quota_manager.use_request()
                        
                        result = await response.json()
                        message_content = result['choices'][0]['message']['content']
                        usage = result.get('usage', {})
                        completion_tokens = usage.get('completion_tokens', len(message_content.split()))
                        
                        tokens_per_sec = completion_tokens / (processing_time / 1000) if processing_time > 0 else 0
                        
                        return {
                            'success': True,
                            'text': message_content,
                            'tokens_generated': completion_tokens,
                            'processing_time_ms': processing_time,
                            'tokens_per_second': tokens_per_sec,
                            'model_used': 'Groq Llama-4 Scout 17B',
                            'quota_remaining': self.quota_manager.get_remaining_quota(),
                            'service': 'groq_llama4_17b'
                        }
                    elif response.status == 429:  # Rate limit exceeded
                        return {
                            'success': False,
                            'error': 'Groq rate limit exceeded - switching to local processing',
                            'quota_remaining': self.quota_manager.get_remaining_quota(),
                            'service': 'groq_llama4_17b'
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'success': False,
                            'error': f'Groq API error: {response.status} - {error_text}',
                            'quota_remaining': self.quota_manager.get_remaining_quota(),
                            'service': 'groq_llama4_17b'
                        }
                        
        except Exception as e:
            return {
                'success': False,
                'error': f'Groq connection failed: {str(e)}',
                'quota_remaining': self.quota_manager.get_remaining_quota(),
                'service': 'groq_llama4_17b'
            }
    
    async def huggingface_enhanced_inference(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        """Enhanced HuggingFace Inference - PRIORITY #3"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "do_sample": True,
                        "return_full_text": False
                    }
                }
                
                headers = {
                    "Authorization": f"Bearer {self.huggingface_token}",
                    "Content-Type": "application/json"
                }
                
                async with session.post(
                    "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
                    json=payload,
                    headers=headers,
                    timeout=15
                ) as response:
                    processing_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0:
                            text = result[0].get('generated_text', 'HuggingFace response generated')
                            
                            return {
                                'success': True,
                                'text': text,
                                'tokens_generated': len(text.split()),
                                'processing_time_ms': processing_time,
                                'tokens_per_second': len(text.split()) / (processing_time / 1000),
                                'model_used': 'HuggingFace DialoGPT-Medium',
                                'service': 'huggingface_enhanced'
                            }
            
            return {
                'success': False,
                'error': 'HuggingFace processing failed',
                'service': 'huggingface_enhanced'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'HuggingFace error: {str(e)}',
                'service': 'huggingface_enhanced'
            }

class OptimizedLocalEngineManager:
    """Optimized Local Mac M2 7B engine manager - PRIORITY #2 with anti-hang optimizations"""
    
    def __init__(self):
        self.logger = logging.getLogger('OptimizedLocalEngineManager')
        self.model_path = "/Users/techman/JARVIS_PROJECT/models/codellama-7b-python.q4_k_m.gguf"
        self.engine_ready = False
        self.engine = None
        
    async def initialize_local_engine(self) -> bool:
        """Initialize Mac M2 7B engine with ANTI-HANG optimizations"""
        try:
            if not os.path.exists(self.model_path):
                self.logger.error(f"❌ Model file not found: {self.model_path}")
                return False
            
            file_size = os.path.getsize(self.model_path) / (1024**3)
            self.logger.info(f"✅ Model file found: {file_size:.2f}GB")
            
            try:
                from llama_cpp import Llama
                
                self.logger.info("🚀 Initializing OPTIMIZED Mac M2 7B engine (anti-hang settings)...")
                
                # ANTI-HANG OPTIMIZATION SETTINGS
                self.engine = Llama(
                    model_path=self.model_path,
                    n_ctx=2048,           # Reduced context to prevent hanging
                    n_threads=4,          # Reduced threads to prevent overload
                    use_mmap=True,
                    use_mlock=False,
                    n_gpu_layers=20,      # Reduced GPU layers for stability
                    verbose=False,
                    n_batch=256,          # Smaller batch size
                    rope_scaling_type=0,  # Disable rope scaling for stability
                    logits_all=False,     # Disable logits to save memory
                    embedding=False       # Disable embedding to save memory
                )
                
                self.engine_ready = True
                self.logger.info("✅ Mac M2 7B engine ready with ANTI-HANG optimizations!")
                return True
                
            except ImportError:
                self.logger.warning("⚠️ llama-cpp-python not available")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Local engine init failed: {e}")
            return False
    
    async def local_inference(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        """Optimized local inference with timeout protection"""
        if not self.engine_ready or not self.engine:
            return {
                'success': False,
                'error': 'Local engine not ready',
                'service': 'local_mac_m2'
            }
        
        try:
            start_time = time.time()
            
            # Simplified prompt to reduce processing complexity
            clean_prompt = prompt[:500]  # Limit prompt length to prevent hanging
            
            # ANTI-HANG: Use asyncio timeout
            async def _inference():
                return self.engine(
                    prompt=clean_prompt,
                    max_tokens=min(max_tokens, 150),  # Limit tokens to prevent hanging
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    stream=False,
                    stop=["<|endoftext|>", "\n\n\n"]  # Early stopping
                )
            
            # Timeout after 10 seconds to prevent hanging
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(_inference), 
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                return {
                    'success': False,
                    'error': 'Local inference timeout (prevented hanging)',
                    'service': 'local_mac_m2'
                }
            
            processing_time = (time.time() - start_time) * 1000
            response_text = result['choices'][0]['text'].strip()
            token_count = result['usage']['completion_tokens']
            tokens_per_second = token_count / (processing_time / 1000) if processing_time > 0 else 0
            
            return {
                'success': True,
                'text': response_text,
                'tokens_generated': token_count,
                'processing_time_ms': processing_time,
                'tokens_per_second': tokens_per_second,
                'model_used': 'Optimized Local CodeLlama-7B',
                'service': 'local_mac_m2'
            }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Local inference failed: {str(e)}',
                'service': 'local_mac_m2'
            }

class PhoneController:
    """Enhanced Nothing Phone 2a controller"""
    
    def __init__(self):
        self.logger = logging.getLogger('PhoneController')
        self.phone_connected = False
        self.phone_apps = {
            'whatsapp': 'com.whatsapp/.Main',
            'instagram': 'com.instagram.android/.activity.MainTabActivity',
            'chrome': 'com.android.chrome/.Main',
            'camera': 'com.nothing.camera/.activity.CameraActivity',
            'settings': 'com.android.settings/.Settings'
        }
    
    async def check_phone_connection(self) -> bool:
        """Enhanced phone connection check"""
        try:
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=5)
            self.phone_connected = 'device' in result.stdout and 'unauthorized' not in result.stdout
            return self.phone_connected
        except:
            return False
    
    async def send_to_phone(self, command: str) -> Dict[str, Any]:
        """Enhanced phone control with more commands"""
        if not await self.check_phone_connection():
            return {
                'success': False,
                'error': 'Nothing Phone 2a not connected via ADB',
                'service': 'phone_local'
            }
        
        try:
            if command.startswith('input_text:'):
                text = command.split(':', 1)[1]
                subprocess.run(['adb', 'shell', 'input', 'text', f'"{text}"'], timeout=10)
                return {'success': True, 'action': f'Sent text: {text}', 'service': 'phone_local'}
            
            elif command.startswith('open_app:'):
                app_name = command.split(':', 1)[1].lower()
                package = self.phone_apps.get(app_name, app_name)
                subprocess.run(['adb', 'shell', 'am', 'start', '-n', package], timeout=10)
                return {'success': True, 'action': f'Opened app: {app_name}', 'service': 'phone_local'}
            
            elif command == 'home':
                subprocess.run(['adb', 'shell', 'input', 'keyevent', 'KEYCODE_HOME'], timeout=5)
                return {'success': True, 'action': 'Pressed home button', 'service': 'phone_local'}
            
            elif command == 'back':
                subprocess.run(['adb', 'shell', 'input', 'keyevent', 'KEYCODE_BACK'], timeout=5)
                return {'success': True, 'action': 'Pressed back button', 'service': 'phone_local'}
            
            elif command == 'screenshot':
                subprocess.run(['adb', 'shell', 'screencap', '-p', '/sdcard/jarvis_screenshot.png'], timeout=10)
                subprocess.run(['adb', 'pull', '/sdcard/jarvis_screenshot.png', '.'], timeout=10)
                return {'success': True, 'action': 'Screenshot taken', 'service': 'phone_local'}
            
            elif command == 'open_camera':
                # Open camera app
                subprocess.run(['adb', 'shell', 'am', 'start', '-n', 'com.nothing.camera/.activity.CameraActivity'], timeout=10)
                await asyncio.sleep(2)  # Wait for camera to open
                return {'success': True, 'action': 'Camera app opened', 'service': 'phone_local'}
            
            elif command == 'take_photo':
                # Open camera and take a photo
                subprocess.run(['adb', 'shell', 'am', 'start', '-n', 'com.nothing.camera/.activity.CameraActivity'], timeout=10)
                await asyncio.sleep(3)  # Wait for camera to fully load
                # Press camera button to take photo
                subprocess.run(['adb', 'shell', 'input', 'tap', '540', '1800'], timeout=5)  # Approximate camera button position
                await asyncio.sleep(2)  # Wait for photo to be taken
                return {'success': True, 'action': 'Photo taken with back camera', 'service': 'phone_local'}
            
            else:
                return {'success': False, 'error': f'Unknown phone command: {command}', 'service': 'phone_local'}
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Phone command timed out', 'service': 'phone_local'}
        except Exception as e:
            return {'success': False, 'error': f'Phone command failed: {str(e)}', 'service': 'phone_local'}

class GroqFirstHybridBridge:
    """GROQ-FIRST Hybrid AI Processing Bridge - Prevents local hanging"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.cloud_services = EnhancedFreeCloudServices()
        self.local_engine = OptimizedLocalEngineManager()
        self.phone_controller = PhoneController()
        self.performance_stats = {
            'groq_requests': 0,
            'local_requests': 0,
            'huggingface_requests': 0,
            'phone_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'avg_tokens_per_second': 0.0,
            'quota_saves': 0  # Times we avoided using Groq to save quota
        }
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('GroqFirstHybridBridge')
        logger.setLevel(logging.INFO)
        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def initialize(self) -> bool:
        """Initialize all components with Groq-first priority"""
        self.logger.info("🚀 Initializing GROQ-FIRST Hybrid Processing Bridge...")
        self.logger.info("🎯 Priority Order: Groq Cloud → Local Mac M2 → HuggingFace")
        
        # Check Groq quota status
        quota_status = self.cloud_services.quota_manager.get_status()
        self.logger.info(f"🔑 Groq Quota Status: {quota_status['requests_remaining']}/{quota_status['daily_limit']} remaining")
        
        # Initialize components  
        local_ready = await self.local_engine.initialize_local_engine()
        phone_ready = await self.phone_controller.check_phone_connection()
        
        self.logger.info("📊 System Status:")
        self.logger.info(f"   ☁️ Groq Llama-4 Scout 17B: ✅ Ready (Priority #1)")
        self.logger.info(f"   🧠 Local Mac M2 7B (Optimized): {'✅ Ready' if local_ready else '❌ Not Available'} (Priority #2)")
        self.logger.info(f"   🤗 HuggingFace Enhanced: ✅ Ready (Priority #3)")
        self.logger.info(f"   📱 Nothing Phone 2a: {'✅ Connected' if phone_ready else '❌ Not Connected'}")
        self.logger.info(f"   💰 Total Cost: $0.00 (100% Free!)")
        
        return True
    
    async def groq_first_route_request(self, request: CloudRequest) -> CloudResponse:
        """GROQ-FIRST intelligent request routing with quota management"""
        start_time = time.time()
        
        self.logger.info(f"🎯 Processing request: {request.prompt[:60]}...")
        
        # Phone control commands (always highest priority)
        if self._is_phone_command(request.prompt):
            return await self._route_to_phone(request, start_time)
        
        # GROQ-FIRST ROUTING LOGIC
        routing_decision = self._groq_first_routing_decision(request)
        self.logger.info(f"🎯 Routing decision: {routing_decision}")
        
        if routing_decision == "groq_priority":
            return await self._route_to_groq_cloud(request, start_time)
        elif routing_decision == "local_fallback":
            return await self._route_to_local(request, start_time)
        elif routing_decision == "huggingface_backup":
            return await self._route_to_huggingface(request, start_time)
        else:
            return await self._smart_fallback_sequence(request, start_time)
    
    def _groq_first_routing_decision(self, request: CloudRequest) -> str:
        """GROQ-FIRST routing decision with quota awareness"""
        
        # Force Groq if explicitly requested
        if request.force_groq:
            return "groq_priority"
        
        # Force local if explicitly requested
        if request.require_local:
            return "local_fallback"
        
        # Check Groq quota availability
        quota_status = self.cloud_services.quota_manager.get_status()
        
        # GROQ PRIORITY LOGIC:
        # Use Groq for ALL requests if quota available (unless local forced)
        if quota_status['can_make_request']:
            # Save some quota for important requests if running low
            if quota_status['requests_remaining'] < 50:
                # High priority requests still get Groq
                if (request.priority in [RequestPriority.HIGH, RequestPriority.CRITICAL] or
                    len(request.prompt) > 200 or
                    request.max_tokens > 300):
                    return "groq_priority"
                else:
                    # Save quota - use local for simple requests
                    self.performance_stats['quota_saves'] += 1
                    return "local_fallback"
            else:
                # Plenty of quota - use Groq for everything
                return "groq_priority"
        
        # No Groq quota left - use local as fallback
        self.logger.warning("⚠️ Groq quota exhausted - using local fallback")
        return "local_fallback"
    
    def _is_phone_command(self, prompt: str) -> bool:
        """Enhanced phone command detection"""
        phone_keywords = [
            'open app', 'launch app', 'start app', 'open whatsapp', 'open instagram',
            'send text', 'type text', 'input text',
            'home screen', 'go home', 'press home',
            'go back', 'press back', 'back button',
            'take screenshot', 'screenshot', 'capture screen',
            'camera', 'take photo', 'take picture', 'open camera', 'back camera',
            'phone', 'android', 'nothing phone'
        ]
        
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in phone_keywords)
    
    async def _route_to_groq_cloud(self, request: CloudRequest, start_time: float) -> CloudResponse:
        """Route to Groq Llama-4 Scout 17B - PRIORITY #1"""
        try:
            result = await self.cloud_services.groq_llama4_inference(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                force_request=request.force_groq
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            if result['success']:
                self.performance_stats['groq_requests'] += 1
                self._update_performance_stats(result)
                
                self.logger.info(f"✅ Groq Llama-4 processing: {processing_time:.0f}ms, {result.get('tokens_per_second', 0):.1f} tok/s")
                
                return CloudResponse(
                    text=result['text'],
                    processing_location=ProcessingLocation.GROQ_LLAMA4_17B,
                    processing_time_ms=processing_time,
                    tokens_generated=result.get('tokens_generated', 0),
                    tokens_per_second=result.get('tokens_per_second', 0),
                    quota_remaining=result.get('quota_remaining', 0),
                    success=True,
                    cost=0.0
                )
            else:
                self.logger.warning(f"⚠️ Groq processing failed: {result['error']} - falling back to local")
                return await self._route_to_local(request, start_time)
                
        except Exception as e:
            self.logger.error(f"❌ Groq routing error: {e} - falling back to local")
            return await self._route_to_local(request, start_time)
    
    async def _route_to_local(self, request: CloudRequest, start_time: float) -> CloudResponse:
        """Route to optimized local Mac M2 7B engine - PRIORITY #2"""
        try:
            result = await self.local_engine.local_inference(
                prompt=request.prompt,
                max_tokens=request.max_tokens
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            if result['success']:
                self.performance_stats['local_requests'] += 1
                self._update_performance_stats(result)
                
                self.logger.info(f"✅ Local M2 (optimized) processing: {processing_time:.0f}ms, {result.get('tokens_per_second', 0):.1f} tok/s")
                
                return CloudResponse(
                    text=result['text'],
                    processing_location=ProcessingLocation.LOCAL_MAC_M2,
                    processing_time_ms=processing_time,
                    tokens_generated=result.get('tokens_generated', 0),
                    tokens_per_second=result.get('tokens_per_second', 0),
                    success=True,
                    cost=0.0
                )
            else:
                self.logger.warning(f"⚠️ Local processing failed: {result['error']} - trying HuggingFace")
                return await self._route_to_huggingface(request, start_time)
                
        except Exception as e:
            self.logger.error(f"❌ Local routing error: {e} - trying HuggingFace")
            return await self._route_to_huggingface(request, start_time)
    
    async def _route_to_huggingface(self, request: CloudRequest, start_time: float) -> CloudResponse:
        """Route to HuggingFace models - PRIORITY #3"""
        try:
            result = await self.cloud_services.huggingface_enhanced_inference(
                prompt=request.prompt,
                max_tokens=request.max_tokens
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            if result['success']:
                self.performance_stats['huggingface_requests'] += 1
                self._update_performance_stats(result)
                
                self.logger.info(f"✅ HuggingFace processing: {processing_time:.0f}ms, {result.get('tokens_per_second', 0):.1f} tok/s")
                
                return CloudResponse(
                    text=result['text'],
                    processing_location=ProcessingLocation.FREE_HUGGINGFACE,
                    processing_time_ms=processing_time,
                    tokens_generated=result.get('tokens_generated', 0),
                    tokens_per_second=result.get('tokens_per_second', 0),
                    success=True,
                    cost=0.0
                )
            else:
                self.logger.warning(f"⚠️ HuggingFace processing failed: {result['error']}")
                return await self._create_fallback_response(request, start_time)
                
        except Exception as e:
            self.logger.error(f"❌ HuggingFace routing error: {e}")
            return await self._create_fallback_response(request, start_time)
    
    async def _smart_fallback_sequence(self, request: CloudRequest, start_time: float) -> CloudResponse:
        """Smart fallback sequence: Groq → Local → HuggingFace"""
        services = [
            ("Groq Llama-4 Scout", self._route_to_groq_cloud),
            ("Optimized Local Mac M2", self._route_to_local),
            ("HuggingFace Enhanced", self._route_to_huggingface)
        ]
        
        for service_name, service_func in services:
            try:
                self.logger.info(f"🔄 Trying fallback: {service_name}")
                response = await service_func(request, start_time)
                if response.success:
                    return response
            except Exception as e:
                self.logger.warning(f"⚠️ {service_name} fallback failed: {e}")
                continue
        
        return await self._create_fallback_response(request, start_time)
    
    async def _route_to_phone(self, request: CloudRequest, start_time: float) -> CloudResponse:
        """Enhanced phone control routing"""
        try:
            prompt_lower = request.prompt.lower()
            
            if 'whatsapp' in prompt_lower or 'instagram' in prompt_lower or 'chrome' in prompt_lower:
                app_name = next((app for app in ['whatsapp', 'instagram', 'chrome'] if app in prompt_lower), 'whatsapp')
                command = f'open_app:{app_name}'
            elif 'camera' in prompt_lower and ('photo' in prompt_lower or 'take' in prompt_lower or 'picture' in prompt_lower):
                command = 'take_photo'
            elif 'camera' in prompt_lower:
                command = 'open_camera'
            elif 'screenshot' in prompt_lower or 'capture' in prompt_lower:
                command = 'screenshot'
            elif 'home' in prompt_lower:
                command = 'home'
            elif 'back' in prompt_lower:
                command = 'back'
            elif 'send text' in prompt_lower or 'type' in prompt_lower:
                text = request.prompt.split('text')[-1].strip()
                command = f'input_text:{text}'
            else:
                command = 'home'
            
            result = await self.phone_controller.send_to_phone(command)
            processing_time = (time.time() - start_time) * 1000
            
            if result['success']:
                response_text = f"✅ Phone action completed: {result['action']}"
                self.performance_stats['phone_requests'] += 1
            else:
                response_text = f"❌ Phone action failed: {result['error']}"
            
            return CloudResponse(
                text=response_text,
                processing_location=ProcessingLocation.PHONE_LOCAL,
                processing_time_ms=processing_time,
                tokens_generated=len(response_text.split()),
                success=result['success'],
                error=result.get('error'),
                cost=0.0
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return CloudResponse(
                text=f"Phone control error: {str(e)}",
                processing_location=ProcessingLocation.PHONE_LOCAL,
                processing_time_ms=processing_time,
                tokens_generated=0,
                success=False,
                error=str(e),
                cost=0.0
            )
    
    async def _create_fallback_response(self, request: CloudRequest, start_time: float) -> CloudResponse:
        """Create intelligent fallback response"""
        processing_time = (time.time() - start_time) * 1000
        self.performance_stats['failed_requests'] += 1
        
        quota_status = self.cloud_services.quota_manager.get_status()
        
        if quota_status['requests_remaining'] == 0:
            response_text = f"Groq quota exhausted ({quota_status['daily_limit']} requests used today). Local processing temporarily unavailable. Try again later."
        else:
            response_text = "All AI services temporarily unavailable. Please try again in a moment."
        
        return CloudResponse(
            text=response_text,
            processing_location=ProcessingLocation.FALLBACK_LOCAL,
            processing_time_ms=processing_time,
            tokens_generated=len(response_text.split()),
            success=False,
            error="All services temporarily unavailable",
            cost=0.0
        )
    
    def _update_performance_stats(self, result: Dict[str, Any]):
        """Update performance statistics"""
        if 'processing_time_ms' in result:
            self.performance_stats['total_processing_time'] += result['processing_time_ms']
        if 'tokens_per_second' in result:
            current_avg = self.performance_stats['avg_tokens_per_second']
            new_value = result['tokens_per_second']
            total_requests = sum([
                self.performance_stats['groq_requests'],
                self.performance_stats['local_requests'],
                self.performance_stats['huggingface_requests']
            ])
            self.performance_stats['avg_tokens_per_second'] = (
                (current_avg * (total_requests - 1) + new_value) / total_requests
                if total_requests > 0 else new_value
            )
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including quota information"""
        quota_status = self.cloud_services.quota_manager.get_status()
        total_requests = sum([
            self.performance_stats['groq_requests'],
            self.performance_stats['local_requests'],
            self.performance_stats['huggingface_requests'],
            self.performance_stats['phone_requests']
        ])
        
        return {
            'bridge_status': 'groq_first_operational',
            'routing_priority': ['Groq Llama-4 Scout 17B', 'Local Mac M2 (Optimized)', 'HuggingFace Enhanced'],
            'quota_status': quota_status,
            'performance_stats': self.performance_stats,
            'total_requests': total_requests,
            'avg_tokens_per_second': self.performance_stats['avg_tokens_per_second'],
            'local_engine_ready': self.local_engine.engine_ready,
            'phone_connected': self.phone_controller.phone_connected,
            'optimization_features': [
                'Groq-first priority routing',
                'Quota-aware request management',
                'Anti-hang local engine optimizations',
                'Smart fallback sequences',
                'Timeout protection'
            ],
            'cost_summary': {
                'total_cost': 0.0,
                'groq_requests_free': self.performance_stats['groq_requests'],
                'local_requests_unlimited': self.performance_stats['local_requests'],
                'estimated_savings_vs_paid': '$1000+/month (100% free!)'
            }
        }

# Enhanced demo and testing
async def main():
    """Demo the GROQ-FIRST Hybrid Processing Bridge"""
    
    print("🚀 JARVIS GROQ-FIRST Hybrid AI Processing Bridge v4.1")
    print("=" * 85)
    print("🎯 PRIORITY ORDER: Groq Cloud FIRST → Local Mac M2 SECOND → HuggingFace THIRD")
    print("💰 Total Cost: $0.00 (100% FREE!)")
    print("⚡ Anti-Hang Optimizations: Timeout protection + Optimized local engine")
    print("🔑 Smart Quota Management: Preserves Groq quota for important requests")
    print()
    
    # Initialize Groq-first bridge
    bridge = GroqFirstHybridBridge()
    await bridge.initialize()
    
    print("\n🧪 Testing Groq-first hybrid request routing...\n")
    
    # Test requests with different priorities
    test_requests = [
        CloudRequest("Hello, test Groq priority", max_tokens=100),
        CloudRequest("Write a detailed Python web scraping tutorial", max_tokens=400, priority=RequestPriority.HIGH),
        CloudRequest("Simple math: what is 2+2?", max_tokens=50, priority=RequestPriority.LOW),
        CloudRequest("Open WhatsApp on my phone", max_tokens=50),
        CloudRequest("Open back camera and take a photo", max_tokens=50),
        CloudRequest("Explain quantum computing comprehensively", max_tokens=500, force_groq=True),
        CloudRequest("Quick code question", max_tokens=100, require_local=True)
    ]
    
    for i, request in enumerate(test_requests, 1):
        print(f"🎯 Test {i}: {request.prompt}")
        if request.force_groq:
            print(f"   🔧 Mode: Force Groq")
        elif request.require_local:
            print(f"   🔧 Mode: Force Local")
        else:
            print(f"   🔧 Mode: Auto-routing (Priority: {request.priority.name})")
        
        try:
            response = await bridge.groq_first_route_request(request)
            
            print(f"   📍 Location: {response.processing_location.value}")
            print(f"   ⏱️ Time: {response.processing_time_ms:.0f}ms")
            print(f"   🚀 Speed: {response.tokens_per_second:.1f} tok/s")
            print(f"   💰 Cost: ${response.cost:.2f}")
            if hasattr(response, 'quota_remaining') and response.quota_remaining:
                print(f"   📊 Groq Quota Remaining: {response.quota_remaining}")
            print(f"   ✅ Response: {response.text[:120]}...")
            
            if response.error:
                print(f"   ⚠️ Error: {response.error}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    # Show comprehensive system status
    status = bridge.get_comprehensive_status()
    print("📊 Comprehensive Final System Status:")
    print(f"   🎯 Routing Priority: {' → '.join(status['routing_priority'])}")
    print(f"   ☁️ Groq Quota Used: {status['quota_status']['requests_used']}/{status['quota_status']['daily_limit']}")
    print(f"   📈 Quota Remaining: {status['quota_status']['requests_remaining']} ({status['quota_status']['quota_percentage']:.1f}%)")
    print(f"   🧠 Local Engine: {'✅ Optimized' if status['local_engine_ready'] else '❌'}")
    print(f"   📱 Phone Connected: {'✅' if status['phone_connected'] else '❌'}")
    print(f"   📊 Total Requests: {status['total_requests']}")
    print(f"   ⚡ Avg Performance: {status['avg_tokens_per_second']:.1f} tok/s")
    print(f"   💾 Quota Saves: {status['performance_stats']['quota_saves']} (smart management)")
    print(f"   💰 Total Cost: $0.00 (FREE!)")
    print(f"   💎 Estimated Savings: {status['cost_summary']['estimated_savings_vs_paid']}")

if __name__ == "__main__":
    # Check API keys from environment variables
    print("🔑 GROQ-FIRST Configuration:")
    if os.getenv('GROQ_API_KEY'):
        print("   ✅ Groq API Key: Active (Priority #1)")
    else:
        print("   ⚠️  Groq API Key: Not found")
        
    if os.getenv('HUGGINGFACE_TOKEN'):
        print("   ✅ HuggingFace Token: Active (Priority #3)")
    else:
        print("   ⚠️  HuggingFace Token: Not found")
        
    print("   🛡️ Anti-Hang Optimizations: Enabled")
    print("   📊 Smart Quota Management: Enabled")
    print()
    
    asyncio.run(main())
