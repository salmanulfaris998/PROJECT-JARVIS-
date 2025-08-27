#!/usr/bin/env python3
"""
Simple test script for JARVIS Response Acceleration
Tests the acceleration system with your real JARVIS setup
"""

import asyncio
import time
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cloud_processing_bridge import GroqFirstHybridBridge
from response_acceleration import AcceleratedJARVIS

async def test_acceleration():
    """Test the acceleration system"""
    
    print("🚀 Testing JARVIS Response Acceleration")
    print("=" * 50)
    
    # Set environment variables
    # Load API keys from environment variables or .env file
    if not os.getenv('GROQ_API_KEY'):
        print("⚠️  GROQ_API_KEY not found in environment variables")
    if not os.getenv('HUGGINGFACE_TOKEN'):
        print("⚠️  HUGGINGFACE_TOKEN not found in environment variables")
    
    try:
        # Initialize accelerated JARVIS
        print("🔧 Initializing accelerated JARVIS...")
        bridge = GroqFirstHybridBridge()
        accelerated_jarvis = AcceleratedJARVIS(bridge)
        await accelerated_jarvis.initialize()
        
        print("✅ Initialization complete!")
        
        # Test queries
        test_queries = [
            "Hello, how are you?",
            "What is 2+2?",
            "Hello, how are you?",  # Should hit cache
            "Write a simple Python function",
            "Open WhatsApp on my phone"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🎯 Test {i}: {query}")
            print("-" * 40)
            
            start_time = time.time()
            response = await accelerated_jarvis.process_accelerated(query)
            
            print(f"✅ Response: {response.text[:100]}...")
            print(f"⚡ Acceleration: {response.acceleration_type.value}")
            print(f"🏆 Engine: {response.winning_engine}")
            print(f"⏱️ Time: {response.processing_time_ms:.0f}ms")
            print(f"💾 Cache hit: {'Yes' if response.cache_hit else 'No'}")
            print(f"🚀 Speed: {response.tokens_per_second:.1f} tok/s")
        
        # Show final stats
        print(f"\n📊 Final Statistics:")
        print("=" * 30)
        status = await accelerated_jarvis.get_system_status()
        accel_stats = status.get('acceleration_stats', {})
        
        print(f"🎯 Total Requests: {accel_stats.get('total_requests', 0)}")
        print(f"⚡ Cache Hit Rate: {accel_stats.get('cache_hit_rate', 0):.1f}%")
        print(f"🏁 Parallel Races: {accel_stats.get('parallel_race_rate', 0):.1f}%")
        print(f"💾 Cache Size: {accel_stats.get('cache_size', 0)} entries")
        
        await accelerated_jarvis.cleanup()
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_acceleration())

