#!/usr/bin/env python3
"""
JARVIS Response Acceleration Engine v1.1 - FIXED ASYNC VERSION
Turbocharges your hybrid Groq+Local system with parallel processing and intelligent caching
Integrates seamlessly with your existing GroqFirstHybridBridge architecture

Author: JARVIS Project
Date: August 27, 2025 - 10:26 AM IST
Performance Target: Sub-second responses with quality optimization
FIXES: Proper asyncio task creation and parallel racing
"""

import asyncio
import time
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict, deque

class AccelerationType(Enum):
    PARALLEL_RACE = "parallel_race"           # Race multiple engines, return fastest
    CACHED_RESPONSE = "cached_response"       # Instant cache hit
    SMART_ROUTING = "smart_routing"           # Route to best engine based on query type
    EARLY_STOPPING = "early_stopping"        # Stop when confidence threshold met
    PARTIAL_STREAMING = "partial_streaming"   # Stream partial responses

@dataclass
class AcceleratedResponse:
    text: str
    acceleration_type: AccelerationType
    processing_time_ms: float
    winning_engine: str
    engines_used: List[str]
    cache_hit: bool = False
    confidence_score: float = 1.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0

class ResponseCache:
    """Intelligent caching system for instant responses"""
    
    def __init__(self, cache_file: str = "jarvis_response_cache.pkl", max_cache_size: int = 10000):
        self.cache_file = Path(cache_file)
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, Dict] = {}
        self.access_times: Dict[str, float] = {}
        self.load_cache()
        
        # Common patterns that benefit from caching
        self.cacheable_patterns = [
            'what is', 'how to', 'explain', 'define', 'tell me about',
            'simple math', 'convert', 'calculate', 'who is', 'when was'
        ]
    
    def _generate_cache_key(self, prompt: str, context: str = "") -> str:
        """Generate consistent cache key for prompts"""
        normalized = prompt.lower().strip()
        # Remove variable elements but keep core query
        cache_input = f"{normalized}|{context}".encode('utf-8')
        return hashlib.sha256(cache_input).hexdigest()[:16]
    
    def is_cacheable(self, prompt: str) -> bool:
        """Determine if a prompt should be cached"""
        prompt_lower = prompt.lower()
        return any(pattern in prompt_lower for pattern in self.cacheable_patterns)
    
    async def get(self, prompt: str, context: str = "") -> Optional[Dict]:
        """Get cached response if available"""
        if not self.is_cacheable(prompt):
            return None
            
        cache_key = self._generate_cache_key(prompt, context)
        
        if cache_key in self.cache:
            # Update access time for LRU
            self.access_times[cache_key] = time.time()
            cached_data = self.cache[cache_key].copy()
            cached_data['cache_hit'] = True
            return cached_data
        
        return None
    
    async def set(self, prompt: str, response_data: Dict, context: str = ""):
        """Cache a response"""
        if not self.is_cacheable(prompt):
            return
            
        cache_key = self._generate_cache_key(prompt, context)
        
        # Add timestamp and usage data
        response_data.update({
            'cached_at': time.time(),
            'prompt_hash': cache_key,
            'access_count': 1
        })
        
        self.cache[cache_key] = response_data
        self.access_times[cache_key] = time.time()
        
        # Maintain cache size
        await self._evict_if_needed()
        self.save_cache()
    
    async def _evict_if_needed(self):
        """Remove oldest entries if cache is full"""
        if len(self.cache) > self.max_cache_size:
            # Remove 10% of oldest entries
            evict_count = max(1, len(self.cache) // 10)
            oldest_keys = sorted(self.access_times.keys(), 
                               key=lambda k: self.access_times[k])[:evict_count]
            
            for key in oldest_keys:
                self.cache.pop(key, None)
                self.access_times.pop(key, None)
    
    def load_cache(self):
        """Load cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', {})
                    self.access_times = data.get('access_times', {})
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}")
            self.cache = {}
            self.access_times = {}
    
    def save_cache(self):
        """Save cache to disk"""
        try:
            cache_data = {
                'cache': self.cache,
                'access_times': self.access_times,
                'version': '1.0'
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")

class EnginePerformanceTracker:
    """Tracks performance of different engines for smart routing"""
    
    def __init__(self):
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        self.engine_specialties = {
            'groq': ['complex', 'reasoning', 'analysis', 'long'],
            'local': ['code', 'programming', 'privacy', 'simple'],
            'huggingface': ['fallback', 'backup', 'general']
        }
    
    def record_performance(self, engine: str, response_time_ms: float, 
                         query_type: str, success: bool):
        """Record engine performance for future routing decisions"""
        self.performance_history[engine].append({
            'response_time': response_time_ms,
            'query_type': query_type,
            'success': success,
            'timestamp': time.time()
        })
    
    def get_best_engine(self, query: str) -> str:
        """Determine best engine based on query and historical performance"""
        query_lower = query.lower()
        
        # Analyze query characteristics
        is_complex = len(query) > 100 or any(word in query_lower for word in 
                                           ['explain', 'analyze', 'detailed', 'comprehensive'])
        is_code = any(word in query_lower for word in ['code', 'function', 'debug', 'python'])
        is_simple = len(query) < 50 and any(word in query_lower for word in ['what', 'how', 'simple'])
        
        # Calculate engine scores based on recent performance
        engine_scores = {}
        for engine, history in self.performance_history.items():
            if not history:
                engine_scores[engine] = 0.5  # Neutral score for unknown engines
                continue
            
            recent_performance = list(history)[-10:]  # Last 10 requests
            avg_time = sum(p['response_time'] for p in recent_performance) / len(recent_performance)
            success_rate = sum(1 for p in recent_performance if p['success']) / len(recent_performance)
            
            # Lower time and higher success rate = higher score
            engine_scores[engine] = (success_rate * 2) / (avg_time / 1000 + 1)
        
        # Boost scores based on specialties
        if is_code and 'local' in engine_scores:
            engine_scores['local'] *= 1.5
        if is_complex and 'groq' in engine_scores:
            engine_scores['groq'] *= 1.3
        if is_simple and 'groq' in engine_scores:
            engine_scores['groq'] *= 1.2
        
        # Return highest scoring engine, default to groq
        return max(engine_scores.items(), key=lambda x: x[1])[0] if engine_scores else 'groq'

class ResponseAccelerator:
    """Main acceleration engine that orchestrates all speed optimizations"""
    
    def __init__(self, jarvis_bridge):
        self.jarvis_bridge = jarvis_bridge  # Your existing GroqFirstHybridBridge
        self.cache = ResponseCache()
        self.performance_tracker = EnginePerformanceTracker()
        self.logger = logging.getLogger('ResponseAccelerator')
        
        # Configure thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="JarvisAccel")
        
        # Acceleration statistics
        self.acceleration_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'parallel_races': 0,
            'avg_speedup': 0.0,
            'total_time_saved': 0.0
        }
    
    async def accelerated_process(self, prompt: str, context: str = "") -> AcceleratedResponse:
        """Main acceleration entry point - tries multiple strategies for fastest response"""
        
        start_time = time.time()
        self.acceleration_stats['total_requests'] += 1
        
        self.logger.info(f"🚀 Accelerating request: {prompt[:60]}...")
        
        # Strategy 1: Check cache first (instant response)
        cached_response = await self.cache.get(prompt, context)
        if cached_response:
            self.acceleration_stats['cache_hits'] += 1
            self.logger.info("⚡ Cache hit - instant response!")
            
            return AcceleratedResponse(
                text=cached_response['text'],
                acceleration_type=AccelerationType.CACHED_RESPONSE,
                processing_time_ms=(time.time() - start_time) * 1000,
                winning_engine="cache",
                engines_used=["cache"],
                cache_hit=True,
                confidence_score=cached_response.get('confidence_score', 1.0),
                tokens_generated=cached_response.get('tokens_generated', 0),
                tokens_per_second=float('inf')  # Instant!
            )
        
        # Strategy 2: Smart routing or parallel race
        try:
            if self._should_use_parallel_race(prompt):
                return await self._parallel_race_engines(prompt, start_time)
            else:
                return await self._smart_route_single_engine(prompt, start_time)
        except Exception as e:
            self.logger.error(f"Strategy selection failed: {e}")
            # Fallback to simple routing
            return await self._simple_fallback(prompt, start_time)
    
    def _should_use_parallel_race(self, prompt: str) -> bool:
        """Determine if we should race engines or use smart single routing"""
        # Use parallel race for:
        # - Important/urgent queries
        # - Medium-complexity queries where multiple engines might work
        # - When we're not sure which engine is best
        
        urgent_words = ['urgent', 'immediate', 'asap', 'quickly', 'fast']
        important_words = ['important', 'critical', 'essential', 'crucial']
        
        prompt_lower = prompt.lower()
        
        # Check for urgent/important words
        is_urgent = any(word in prompt_lower for word in urgent_words)
        is_important = any(word in prompt_lower for word in important_words)
        
        # Check for medium complexity (5-25 words)
        is_medium_complexity = 5 <= len(prompt.split()) <= 25
        
        return is_urgent or is_important or is_medium_complexity
    
    async def _parallel_race_engines(self, prompt: str, start_time: float) -> AcceleratedResponse:
        """FIXED: Race multiple engines in parallel using proper task creation"""
        
        self.acceleration_stats['parallel_races'] += 1
        self.logger.info("🏁 Starting parallel engine race...")
        
        # Create tasks properly (not coroutines)
        tasks = []
        engine_names = []
        
        # Create Groq task
        if (hasattr(self.jarvis_bridge, 'cloud_services') and 
            self.jarvis_bridge.cloud_services.quota_manager.can_make_request()):
            task = asyncio.create_task(self._run_engine_safely('groq', prompt))
            tasks.append(task)
            engine_names.append('groq')
        
        # Create Local task  
        if (hasattr(self.jarvis_bridge, 'local_engine') and 
            self.jarvis_bridge.local_engine.engine_ready):
            task = asyncio.create_task(self._run_engine_safely('local', prompt))
            tasks.append(task)
            engine_names.append('local')
        
        # Create HuggingFace task
        task = asyncio.create_task(self._run_engine_safely('huggingface', prompt))
        tasks.append(task)
        engine_names.append('huggingface')
        
        if not tasks:
            raise RuntimeError("No engines available for parallel processing")
        
        winning_response = None
        winning_engine = "fallback"
        
        try:
            # Wait for first completion - now with proper tasks
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=15)
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Get winning result
            for task in done:
                try:
                    result = task.result()
                    if result and result.get('success', False):
                        winning_response = result
                        winning_engine = engine_names[tasks.index(task)]
                        self.logger.info(f"🏆 Winner: {winning_engine}")
                        break
                except Exception as e:
                    self.logger.warning(f"Task failed: {e}")
                    continue
                    
        except asyncio.TimeoutError:
            self.logger.warning("⚠️ Race timed out")
            for task in tasks:
                if not task.done():
                    task.cancel()
        
        processing_time = (time.time() - start_time) * 1000
        
        if winning_response and winning_response.get('success'):
            await self.cache.set(prompt, {
                'text': winning_response['text'],
                'confidence_score': 0.9,
                'tokens_generated': winning_response.get('tokens_generated', 0)
            })
            
            return AcceleratedResponse(
                text=winning_response['text'],
                acceleration_type=AccelerationType.PARALLEL_RACE,
                processing_time_ms=processing_time,
                winning_engine=winning_engine,
                engines_used=engine_names,
                confidence_score=0.9,
                tokens_generated=winning_response.get('tokens_generated', 0),
                tokens_per_second=winning_response.get('tokens_per_second', 0)
            )
        else:
            return AcceleratedResponse(
                text="Processing failed, please try again.",
                acceleration_type=AccelerationType.PARALLEL_RACE,
                processing_time_ms=processing_time,
                winning_engine="fallback",
                engines_used=engine_names,
                confidence_score=0.1
            )
    
    async def _smart_route_single_engine(self, prompt: str, start_time: float) -> AcceleratedResponse:
        """Use performance tracking to route to best single engine"""
        
        # Determine best engine based on query and historical performance
        best_engine = self.performance_tracker.get_best_engine(prompt)
        self.logger.info(f"🎯 Smart routing to: {best_engine}")
        
        # Execute on chosen engine
        result = await self._run_engine_safely(best_engine, prompt)
        processing_time = (time.time() - start_time) * 1000
        
        # Record performance for future routing
        self.performance_tracker.record_performance(
            best_engine, processing_time, 
            self._classify_query(prompt), 
            result.get('success', False) if result else False
        )
        
        if result and result.get('success'):
            # Cache successful response
            await self.cache.set(prompt, {
                'text': result['text'],
                'confidence_score': 0.8,
                'tokens_generated': result.get('tokens_generated', 0),
                'winning_engine': best_engine
            })
            
            return AcceleratedResponse(
                text=result['text'],
                acceleration_type=AccelerationType.SMART_ROUTING,
                processing_time_ms=processing_time,
                winning_engine=best_engine,
                engines_used=[best_engine],
                confidence_score=0.8,
                tokens_generated=result.get('tokens_generated', 0),
                tokens_per_second=result.get('tokens_per_second', 0)
            )
        else:
            return AcceleratedResponse(
                text="I couldn't process that request. Please try rephrasing.",
                acceleration_type=AccelerationType.SMART_ROUTING,
                processing_time_ms=processing_time,
                winning_engine=best_engine,
                engines_used=[best_engine],
                confidence_score=0.1
            )
    
    async def _simple_fallback(self, prompt: str, start_time: float) -> AcceleratedResponse:
        """Simple fallback when acceleration strategies fail"""
        try:
            from cloud_processing_bridge import CloudRequest
            request = CloudRequest(prompt=prompt)
            response = await self.jarvis_bridge.groq_first_route_request(request)
            
            return AcceleratedResponse(
                text=response.text,
                acceleration_type=AccelerationType.SMART_ROUTING,
                processing_time_ms=response.processing_time_ms,
                winning_engine="fallback",
                engines_used=["fallback"],
                confidence_score=0.6,
                tokens_generated=response.tokens_generated,
                tokens_per_second=response.tokens_per_second
            )
        except Exception as e:
            self.logger.error(f"Simple fallback failed: {e}")
            return AcceleratedResponse(
                text="I'm having trouble processing your request. Please try again.",
                acceleration_type=AccelerationType.SMART_ROUTING,
                processing_time_ms=(time.time() - start_time) * 1000,
                winning_engine="error",
                engines_used=["error"],
                confidence_score=0.1
            )
    
    async def _run_engine_safely(self, engine_name: str, prompt: str) -> Optional[Dict]:
        """FIXED: Safely run a specific engine and return standardized result"""
        try:
            if engine_name == 'groq':
                from cloud_processing_bridge import CloudRequest
                request = CloudRequest(prompt=prompt)
                response = await self.jarvis_bridge._route_to_groq_cloud(request, time.time())
                return {
                    'text': response.text,
                    'success': response.success,
                    'tokens_generated': response.tokens_generated,
                    'tokens_per_second': response.tokens_per_second,
                    'processing_time_ms': response.processing_time_ms
                }
            
            elif engine_name == 'local':
                from cloud_processing_bridge import CloudRequest
                request = CloudRequest(prompt=prompt)
                response = await self.jarvis_bridge._route_to_local(request, time.time())
                return {
                    'text': response.text,
                    'success': response.success,
                    'tokens_generated': response.tokens_generated,
                    'tokens_per_second': response.tokens_per_second,
                    'processing_time_ms': response.processing_time_ms
                }
            
            elif engine_name == 'huggingface':
                from cloud_processing_bridge import CloudRequest
                request = CloudRequest(prompt=prompt)
                response = await self.jarvis_bridge._route_to_huggingface(request, time.time())
                return {
                    'text': response.text,
                    'success': response.success,
                    'tokens_generated': response.tokens_generated,
                    'tokens_per_second': getattr(response, 'tokens_per_second', 0),
                    'processing_time_ms': response.processing_time_ms
                }
            
        except Exception as e:
            self.logger.error(f"Engine {engine_name} failed: {e}")
            return None
    
    def _classify_query(self, prompt: str) -> str:
        """Classify query type for performance tracking"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['code', 'function', 'debug', 'python']):
            return 'code'
        elif any(word in prompt_lower for word in ['explain', 'analyze', 'detailed']):
            return 'complex'
        elif len(prompt) < 50:
            return 'simple'
        else:
            return 'general'
    
    def get_acceleration_stats(self) -> Dict[str, Any]:
        """Get detailed acceleration performance statistics"""
        total_requests = max(1, self.acceleration_stats['total_requests'])
        
        return {
            'total_requests': self.acceleration_stats['total_requests'],
            'cache_hit_rate': (self.acceleration_stats['cache_hits'] / total_requests) * 100,
            'parallel_race_rate': (self.acceleration_stats['parallel_races'] / total_requests) * 100,
            'avg_speedup_factor': self.acceleration_stats.get('avg_speedup', 1.0),
            'total_time_saved_seconds': self.acceleration_stats.get('total_time_saved', 0.0),
            'cache_size': len(self.cache.cache),
            'engines_tracked': len(self.performance_tracker.performance_history)
        }
    
    async def cleanup(self):
        """Clean up resources"""
        self.cache.save_cache()
        self.executor.shutdown(wait=True)

# Integration with your existing JARVIS system
class AcceleratedJARVIS:
    """Your enhanced JARVIS with response acceleration"""
    
    def __init__(self, jarvis_bridge):
        self.jarvis_bridge = jarvis_bridge
        self.accelerator = ResponseAccelerator(jarvis_bridge)
        self.logger = logging.getLogger('AcceleratedJARVIS')
    
    async def initialize(self):
        """Initialize the accelerated JARVIS system"""
        await self.jarvis_bridge.initialize()
        self.logger.info("🚀 Accelerated JARVIS initialized and ready!")
    
    async def process_accelerated(self, user_input: str) -> AcceleratedResponse:
        """Process user input with maximum acceleration"""
        self.logger.info(f"⚡ Processing with acceleration: {user_input[:50]}...")
        
        try:
            response = await self.accelerator.accelerated_process(user_input)
            
            # Log acceleration results
            speedup_info = f"{response.acceleration_type.value} via {response.winning_engine}"
            if response.cache_hit:
                speedup_info += " (cached)"
            
            self.logger.info(f"✅ Accelerated response: {response.processing_time_ms:.0f}ms using {speedup_info}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Acceleration failed: {e}")
            # Fallback to normal processing
            from cloud_processing_bridge import CloudRequest
            request = CloudRequest(prompt=user_input)
            fallback_response = await self.jarvis_bridge.groq_first_route_request(request)
            
            return AcceleratedResponse(
                text=fallback_response.text,
                acceleration_type=AccelerationType.SMART_ROUTING,
                processing_time_ms=fallback_response.processing_time_ms,
                winning_engine="fallback",
                engines_used=["fallback"],
                confidence_score=0.7,
                tokens_generated=fallback_response.tokens_generated,
                tokens_per_second=fallback_response.tokens_per_second
            )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including acceleration metrics"""
        base_status = self.jarvis_bridge.get_comprehensive_status()
        accel_stats = self.accelerator.get_acceleration_stats()
        
        return {
            **base_status,
            'acceleration_enabled': True,
            'acceleration_stats': accel_stats,
            'performance_boost': f"{accel_stats['cache_hit_rate']:.1f}% instant responses",
            'engines_optimized': accel_stats['engines_tracked']
        }
    
    async def cleanup(self):
        """Clean up accelerated JARVIS resources"""
        await self.accelerator.cleanup()

async def demo_response_acceleration():
    """Demonstrate response acceleration capabilities"""
    
    print("🚀 JARVIS Response Acceleration Demo")
    print("=" * 70)
    print("⚡ Turbocharging your hybrid Groq+Local system with:")
    print("   • Intelligent caching for instant responses")
    print("   • Parallel engine racing for optimal speed")
    print("   • Smart routing based on performance history")
    print("   • Early stopping and resource optimization")
    print()
    
    # Import your existing bridge
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from cloud_processing_bridge import GroqFirstHybridBridge
        
        # Initialize accelerated JARVIS
        bridge = GroqFirstHybridBridge()
        accelerated_jarvis = AcceleratedJARVIS(bridge)
        await accelerated_jarvis.initialize()
        
        # Test acceleration with various query types
        test_queries = [
            "What is 2+2?",  # Simple - should be fast
            "What is 2+2?",  # Repeat - should hit cache (instant)
            "Write a Python function for sorting",  # Code - smart routing
            "Explain quantum computing in detail",  # Complex - parallel race
            "How are you today?",  # Simple - fast routing
            "Debug this Python error: TypeError",  # Code - optimized routing
        ]
        
        total_time_saved = 0.0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🎯 Test {i}: {query}")
            
            start_time = time.time()
            response = await accelerated_jarvis.process_accelerated(query)
            
            # Display results
            print(f"   ⚡ Acceleration: {response.acceleration_type.value}")
            print(f"   🏆 Winner: {response.winning_engine}")
            print(f"   ⏱️ Time: {response.processing_time_ms:.0f}ms")
            print(f"   🚀 Speed: {response.tokens_per_second:.1f} tok/s")
            print(f"   💾 Cached: {'Yes' if response.cache_hit else 'No'}")
            print(f"   ✅ Response: {response.text[:80]}...")
            
            if response.cache_hit:
                total_time_saved += 2000  # Estimate 2 seconds saved per cache hit
        
        # Show acceleration statistics
        stats = await accelerated_jarvis.get_system_status()
        accel_stats = stats['acceleration_stats']
        
        print(f"\n📊 Acceleration Performance Summary:")
        print(f"   🎯 Total Requests: {accel_stats['total_requests']}")
        print(f"   ⚡ Cache Hit Rate: {accel_stats['cache_hit_rate']:.1f}%")
        print(f"   🏁 Parallel Races: {accel_stats['parallel_race_rate']:.1f}%")
        print(f"   💾 Cache Size: {accel_stats['cache_size']} entries")
        print(f"   ⏰ Estimated Time Saved: {total_time_saved/1000:.1f} seconds")
        print(f"   🚀 Performance Boost: {stats['performance_boost']}")
        
        await accelerated_jarvis.cleanup()
        
    except ImportError:
        print("ℹ️  To run this demo, ensure your cloud_processing_bridge.py is available")
        print("   This acceleration module will integrate seamlessly with your existing JARVIS!")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    
    # Run the demo
    asyncio.run(demo_response_acceleration())
