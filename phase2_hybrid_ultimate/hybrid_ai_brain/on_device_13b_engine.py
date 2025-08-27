#!/usr/bin/env python3
"""
JARVIS On-Device 7B Engine - MAC M2 OPTIMIZED
Complete implementation with RAM check removed
Optimized for Apple Silicon M2 with Metal acceleration
"""

import asyncio
import logging
import json
import time
import sqlite3
import hashlib
import threading
import psutil
import gc
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict

# Safe imports with fallback
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

class ProcessingPriority(Enum):
    LOW = 3
    NORMAL = 2
    HIGH = 1

class ModelSize(Enum):
    CODELLAMA_7B = "codellama_7b"
    LLAMA_7B = "llama_7b"

@dataclass
class ModelConfig:
    model_path: str
    model_type: ModelSize = ModelSize.CODELLAMA_7B
    context_length: int = 2048
    threads: int = 6  # Optimized for M2
    use_mmap: bool = True
    use_mlock: bool = False
    metal: bool = True  # Enable Metal for M2
    gpu_layers: int = 35  # Use GPU acceleration

@dataclass
class InferenceRequest:
    id: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class InferenceResponse:
    id: str
    text: str
    tokens_generated: int
    processing_time_ms: float
    tokens_per_second: float
    cache_hit: bool = False
    error: Optional[str] = None

class SimpleCache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
            self.cache[key] = value
    
    def size(self) -> int:
        with self.lock:
            return len(self.cache)

class OnDevice7BEngine:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.db_path = Path('logs/on_device_7b_m2.db')
        
        # Create directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Core components
        self.model: Optional[Llama] = None
        self.model_loaded = False
        self.inference_active = False
        
        # Caching and performance
        self.response_cache = SimpleCache(max_size=100)
        self.request_queue = asyncio.PriorityQueue()
        self.active_requests: Dict[str, InferenceRequest] = {}
        
        # Thread safety
        self.model_lock = threading.RLock()
        self.inference_lock = threading.Semaphore(1)
        
        self.logger.info("🧠 Mac M2 Optimized 7B Engine initialized")

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('OnDevice7BEngine')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def _init_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    prompt_length INTEGER,
                    tokens_generated INTEGER,
                    processing_time_ms REAL,
                    tokens_per_second REAL,
                    cache_hit BOOLEAN,
                    gpu_used BOOLEAN
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Database initialized")
            
        except Exception as e:
            self.logger.error(f"Database init failed: {e}")

    async def initialize_engine(self) -> bool:
        try:
            self.logger.info("🚀 Initializing Mac M2 Optimized Engine...")
            
            if not self._verify_requirements():
                return False
            
            if not await self._load_model():
                return False
            
            # Start background tasks
            asyncio.create_task(self._request_processor())
            asyncio.create_task(self._memory_monitor())
            
            self.logger.info("✅ Mac M2 Engine operational with Metal acceleration!")
            return True
            
        except Exception as e:
            self.logger.error(f"Engine init failed: {e}")
            return False

    def _verify_requirements(self) -> bool:
        if not LLAMA_AVAILABLE:
            self.logger.error("❌ llama-cpp-python not installed")
            return False
        
        if not Path(self.config.model_path).exists():
            self.logger.error(f"❌ Model not found: {self.config.model_path}")
            return False
        
        # Check if we're on Apple Silicon
        if sys.platform == 'darwin' and 'arm64' in os.uname().machine:
            self.logger.info("✅ Apple Silicon M2 detected - Metal acceleration available")
        else:
            self.logger.warning("⚠️ Not on Apple Silicon - Metal acceleration not available")
            self.config.metal = False
            self.config.gpu_layers = 0
        
        # No strict RAM checking - let Mac's unified memory handle it
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        self.logger.info(f"✅ Mac M2 with {total_gb:.1f}GB unified memory - Ready to go!")
        
        return True

    async def _load_model(self) -> bool:
        try:
            self.logger.info(f"📥 Loading model: {Path(self.config.model_path).name}")
            self.logger.info("🎮 Using Metal GPU acceleration for M2")
            
            start_time = time.time()
            
            # M2 Optimized parameters
            model_params = {
                'model_path': self.config.model_path,
                'n_ctx': self.config.context_length,
                'n_threads': self.config.threads,
                'use_mmap': self.config.use_mmap,
                'use_mlock': self.config.use_mlock,
                'verbose': False
            }
            
            # Add Metal GPU acceleration for M2
            if self.config.metal and self.config.gpu_layers > 0:
                model_params['n_gpu_layers'] = self.config.gpu_layers
                self.logger.info(f"🚀 Metal acceleration: {self.config.gpu_layers} GPU layers")
            
            with self.model_lock:
                self.model = Llama(**model_params)
                self.model_loaded = True
            
            load_time = time.time() - start_time
            
            self.logger.info(f"✅ Model loaded in {load_time:.2f}s with Metal acceleration")
            
            # Warmup
            await self._warmup_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            # Fallback without GPU layers
            if self.config.gpu_layers > 0:
                self.logger.info("🔄 Retrying without GPU layers...")
                self.config.gpu_layers = 0
                return await self._load_model()
            return False

    async def _warmup_model(self):
        try:
            self.logger.info("🔥 Warming up M2 optimized model...")
            
            warmup_request = InferenceRequest(
                id="warmup",
                prompt="Hello",
                max_tokens=10,
                temperature=0.1
            )
            
            response = await self._execute_inference(warmup_request)
            
            if response and not response.error:
                self.logger.info(f"✅ Warmup complete: {response.tokens_per_second:.1f} tok/s")
            else:
                self.logger.warning("⚠️ Warmup failed")
                
        except Exception as e:
            self.logger.warning(f"Warmup error: {e}")

    async def _request_processor(self):
        self.logger.info("🔄 Request processor started")
        
        while True:
            try:
                priority, request_id, request = await self.request_queue.get()
                
                if request_id in self.active_requests:
                    request = self.active_requests[request_id]
                    response = await self._execute_inference(request)
                    
                    if request_id in self.active_requests:
                        del self.active_requests[request_id]
                
                self.request_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Request processor error: {e}")
                await asyncio.sleep(0.1)

    async def _execute_inference(self, request: InferenceRequest) -> Optional[InferenceResponse]:
        if not self.model_loaded or not self.model:
            return InferenceResponse(
                id=request.id,
                text="",
                tokens_generated=0,
                processing_time_ms=0,
                tokens_per_second=0,
                error="Model not loaded"
            )
        
        try:
            # Check cache
            cache_key = hashlib.sha256(f"{request.prompt}_{request.max_tokens}_{request.temperature}".encode()).hexdigest()[:16]
            cached_response = self.response_cache.get(cache_key)
            
            if cached_response:
                cached_response.cache_hit = True
                self.logger.info(f"⚡ Cache hit for {request.id}")
                return cached_response
            
            # Execute inference
            start_time = time.time()
            
            with self.model_lock:
                result = self.model(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=0.9,
                    top_k=40,
                    stream=False
                )
                
                response_text = result['choices'][0]['text']
                token_count = result['usage']['completion_tokens']
            
            processing_time = (time.time() - start_time) * 1000
            tokens_per_second = token_count / (processing_time / 1000) if processing_time > 0 else 0
            
            response = InferenceResponse(
                id=request.id,
                text=response_text.strip(),
                tokens_generated=token_count,
                processing_time_ms=processing_time,
                tokens_per_second=tokens_per_second,
                cache_hit=False
            )
            
            # Cache response
            self.response_cache.put(cache_key, response)
            
            # Log to database
            await self._log_inference(request, response)
            
            self.logger.info(f"✅ Generated {token_count} tokens in {processing_time:.0f}ms ({tokens_per_second:.1f} tok/s)")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return InferenceResponse(
                id=request.id,
                text="",
                tokens_generated=0,
                processing_time_ms=0,
                tokens_per_second=0,
                error=str(e)
            )

    async def _memory_monitor(self):
        while True:
            try:
                memory = psutil.virtual_memory()
                
                # More lenient memory management for Mac M2
                if memory.percent > 90.0:
                    self.logger.warning(f"⚠️ High memory: {memory.percent:.1f}%")
                    gc.collect()
                    
                    # Clear some cache if really high
                    if memory.percent > 95.0 and self.response_cache.size() > 50:
                        self.response_cache = SimpleCache(max_size=30)
                        self.logger.info("🧹 Cache cleared")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Memory monitor error: {e}")
                await asyncio.sleep(60)

    async def generate_response(self, prompt: str, max_tokens: int = 128, 
                              temperature: float = 0.7, 
                              priority: ProcessingPriority = ProcessingPriority.NORMAL) -> InferenceResponse:
        
        request = InferenceRequest(
            id=hashlib.md5(f"{prompt}_{time.time()}".encode()).hexdigest()[:12],
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority
        )
        
        # Store request
        self.active_requests[request.id] = request
        
        # Add to queue
        await self.request_queue.put((priority.value, request.id, request))
        
        self.logger.info(f"🎯 Queued request {request.id}")
        
        # Wait for processing
        max_wait = 30.0
        start_wait = time.time()
        
        while request.id in self.active_requests:
            if time.time() - start_wait > max_wait:
                if request.id in self.active_requests:
                    del self.active_requests[request.id]
                
                return InferenceResponse(
                    id=request.id,
                    text="",
                    tokens_generated=0,
                    processing_time_ms=max_wait * 1000,
                    tokens_per_second=0,
                    error="Timeout"
                )
            
            await asyncio.sleep(0.1)
        
        # Get response from cache
        cache_key = hashlib.sha256(f"{request.prompt}_{request.max_tokens}_{request.temperature}".encode()).hexdigest()[:16]
        response = self.response_cache.get(cache_key)
        
        return response or InferenceResponse(
            id=request.id,
            text="",
            tokens_generated=0,
            processing_time_ms=0,
            tokens_per_second=0,
            error="Response not found"
        )

    async def get_engine_status(self) -> Dict[str, Any]:
        try:
            memory = psutil.virtual_memory()
            
            # Get database stats
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM inference_logs WHERE timestamp > datetime('now', '-24 hours')")
            daily_inferences = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(tokens_per_second) FROM inference_logs WHERE timestamp > datetime('now', '-1 hour')")
            avg_tokens_per_sec = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'system_status': 'operational' if self.model_loaded else 'initializing',
                'model_loaded': self.model_loaded,
                'model_type': self.config.model_type.value,
                'context_length': self.config.context_length,
                'threads': self.config.threads,
                'metal_acceleration': self.config.metal,
                'gpu_layers': self.config.gpu_layers,
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_percent': memory.percent
                },
                'performance': {
                    'active_requests': len(self.active_requests),
                    'queue_size': self.request_queue.qsize(),
                    'cache_entries': self.response_cache.size(),
                    'daily_inferences': daily_inferences,
                    'avg_tokens_per_sec': avg_tokens_per_sec
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'system_status': 'error', 'error': str(e)}

    async def _log_inference(self, request: InferenceRequest, response: InferenceResponse):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO inference_logs 
                (id, timestamp, prompt_length, tokens_generated, processing_time_ms, tokens_per_second, cache_hit, gpu_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                response.id,
                datetime.now().isoformat(),
                len(request.prompt),
                response.tokens_generated,
                response.processing_time_ms,
                response.tokens_per_second,
                response.cache_hit,
                self.config.gpu_layers > 0
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Logging failed: {e}")

# Main execution
async def main():
    # Mac M2 optimized config
    config = ModelConfig(
        model_path="models/codellama-7b-python.q4_k_m.gguf",
        model_type=ModelSize.CODELLAMA_7B,
        context_length=2048,
        threads=6,  # Optimized for M2
        metal=True,
        gpu_layers=35  # Use GPU acceleration
    )
    
    engine = OnDevice7BEngine(config)
    
    print("🧠 JARVIS On-Device 7B Engine - MAC M2 OPTIMIZED")
    print("=" * 65)
    print("🎮 Metal GPU acceleration enabled for Apple Silicon")
    print("📱 Ready for Nothing Phone 2a integration")
    print()
    
    if await engine.initialize_engine():
        print("✅ Mac M2 Engine operational!")
        
        # Get status
        status = await engine.get_engine_status()
        print("\n📊 Engine Status:")
        print(f"   Model: {status['model_type']}")
        print(f"   Metal Acceleration: {'✅' if status['metal_acceleration'] else '❌'}")
        print(f"   GPU Layers: {status['gpu_layers']}")
        print(f"   Memory: {status['memory']['available_gb']:.1f}GB available / {status['memory']['total_gb']:.1f}GB total")
        print(f"   Usage: {status['memory']['used_percent']:.1f}%")
        print(f"   Cache: {status['performance']['cache_entries']} entries")
        print(f"   Daily Inferences: {status['performance']['daily_inferences']}")
        print(f"   Avg Speed: {status['performance']['avg_tokens_per_sec']:.1f} tok/s")
        
        # Test responses
        print("\n🤖 Testing M2 accelerated responses...")
        
        test_prompts = [
            "Write a Python function to control Android phone via ADB",
            "Explain how JARVIS voice control works",
            "Generate code for Nothing Phone 2a automation"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n   Test {i}: {prompt}")
            try:
                response = await engine.generate_response(
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.7,
                    priority=ProcessingPriority.HIGH
                )
                
                if response and not response.error:
                    print(f"   ✅ Response ({response.tokens_generated} tokens, {response.processing_time_ms:.0f}ms):")
                    print(f"      {response.text[:100]}...")
                    print(f"      Speed: {response.tokens_per_second:.1f} tok/s")
                    print(f"      Cache: {'✅' if response.cache_hit else '❌'}")
                else:
                    print(f"   ❌ Error: {response.error if response else 'No response'}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
        
        print("\n🎉 Mac M2 7B Engine ready for JARVIS!")
        print("🎙️ Next: Add voice interface + Nothing Phone 2a control")
        print("⚡ Your M2 with Metal acceleration will deliver excellent performance!")
        
    else:
        print("❌ Engine initialization failed!")
        print("\n📋 Setup Requirements:")
        print("   1. pip install llama-cpp-python")
        print("   2. Download model: curl -L 'https://huggingface.co/TheBloke/CodeLlama-7B-Python-GGUF/resolve/main/codellama-7b-python.Q4_K_M.gguf' -o models/codellama-7b-python.q4_k_m.gguf")

if __name__ == '__main__':
    asyncio.run(main())
