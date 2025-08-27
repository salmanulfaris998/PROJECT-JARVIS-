#!/usr/bin/env python3
"""
JARVIS Live Accelerated System v1.0
Real-time integration of response acceleration with your existing JARVIS
Use this for live, accelerated interactions with your JARVIS system

Author: JARVIS Project
Date: August 27, 2025
"""

import asyncio
import time
import logging
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cloud_processing_bridge import GroqFirstHybridBridge
from response_acceleration import AcceleratedJARVIS

class LiveAcceleratedJARVIS:
    """Real-time accelerated JARVIS system for live interactions"""
    
    def __init__(self):
        self.accelerated_jarvis = None
        self.logger = self._setup_logging()
        self.is_running = False
        
    def _setup_logging(self):
        """Setup logging for live system"""
        logger = logging.getLogger('LiveAcceleratedJARVIS')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def initialize(self):
        """Initialize the live accelerated JARVIS system"""
        self.logger.info("🚀 Initializing Live Accelerated JARVIS...")
        
        try:
            # Initialize your existing bridge
            bridge = GroqFirstHybridBridge()
            
            # Create accelerated JARVIS
            self.accelerated_jarvis = AcceleratedJARVIS(bridge)
            await self.accelerated_jarvis.initialize()
            
            self.logger.info("✅ Live Accelerated JARVIS ready!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            return False
    
    async def start_interactive_mode(self):
        """Start interactive mode for real-time conversations"""
        if not self.accelerated_jarvis:
            self.logger.error("❌ JARVIS not initialized. Run initialize() first.")
            return
        
        self.is_running = True
        self.logger.info("🎤 Starting interactive mode...")
        self.logger.info("💡 Type 'quit', 'exit', or 'bye' to end the session")
        self.logger.info("💡 Type 'status' to see system performance")
        self.logger.info("💡 Type 'stats' to see acceleration statistics")
        self.logger.info("=" * 60)
        
        while self.is_running:
            try:
                # Get user input
                user_input = input("\n🤖 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    await self.shutdown()
                    break
                elif user_input.lower() == 'status':
                    await self.show_system_status()
                    continue
                elif user_input.lower() == 'stats':
                    await self.show_acceleration_stats()
                    continue
                
                # Process with acceleration
                print("⚡ Processing with acceleration...")
                start_time = time.time()
                
                response = await self.accelerated_jarvis.process_accelerated(user_input)
                
                # Display response
                print(f"\n🤖 JARVIS ({response.acceleration_type.value}): {response.text}")
                print(f"   ⏱️ Response time: {response.processing_time_ms:.0f}ms")
                print(f"   🏆 Engine: {response.winning_engine}")
                if response.cache_hit:
                    print(f"   ⚡ Cache hit - instant response!")
                print(f"   🚀 Speed: {response.tokens_per_second:.1f} tok/s")
                
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user")
                await self.shutdown()
                break
            except Exception as e:
                self.logger.error(f"❌ Error processing request: {e}")
                print(f"❌ Error: {e}")
    
    async def show_system_status(self):
        """Display current system status"""
        if not self.accelerated_jarvis:
            print("❌ JARVIS not initialized")
            return
        
        try:
            status = await self.accelerated_jarvis.get_system_status()
            
            print("\n📊 JARVIS System Status:")
            print("=" * 40)
            print(f"🎯 Routing Priority: {' → '.join(status['routing_priority'])}")
            print(f"☁️ Groq Quota: {status['quota_status']['requests_used']}/{status['quota_status']['daily_limit']}")
            print(f"📈 Quota Remaining: {status['quota_status']['requests_remaining']} ({status['quota_status']['quota_percentage']:.1f}%)")
            print(f"🧠 Local Engine: {'✅ Ready' if status['local_engine_ready'] else '❌'}")
            print(f"📱 Phone Connected: {'✅' if status['phone_connected'] else '❌'}")
            print(f"⚡ Acceleration: {'✅ Enabled' if status.get('acceleration_enabled', False) else '❌'}")
            print(f"🚀 Performance Boost: {status.get('performance_boost', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error getting status: {e}")
    
    async def show_acceleration_stats(self):
        """Display acceleration statistics"""
        if not self.accelerated_jarvis:
            print("❌ JARVIS not initialized")
            return
        
        try:
            status = await self.accelerated_jarvis.get_system_status()
            accel_stats = status.get('acceleration_stats', {})
            
            print("\n⚡ Acceleration Statistics:")
            print("=" * 40)
            print(f"🎯 Total Requests: {accel_stats.get('total_requests', 0)}")
            print(f"⚡ Cache Hit Rate: {accel_stats.get('cache_hit_rate', 0):.1f}%")
            print(f"🏁 Parallel Races: {accel_stats.get('parallel_race_rate', 0):.1f}%")
            print(f"💾 Cache Size: {accel_stats.get('cache_size', 0)} entries")
            print(f"⏰ Time Saved: {accel_stats.get('total_time_saved_seconds', 0):.1f} seconds")
            print(f"🔧 Engines Tracked: {accel_stats.get('engines_tracked', 0)}")
            
        except Exception as e:
            print(f"❌ Error getting acceleration stats: {e}")
    
    async def shutdown(self):
        """Shutdown the live system"""
        self.logger.info("🛑 Shutting down Live Accelerated JARVIS...")
        self.is_running = False
        
        if self.accelerated_jarvis:
            await self.accelerated_jarvis.cleanup()
        
        self.logger.info("✅ Shutdown complete")

async def main():
    """Main entry point for live accelerated JARVIS"""
    
    print("🚀 JARVIS Live Accelerated System v1.0")
    print("=" * 50)
    print("⚡ Real-time response acceleration with your JARVIS")
    print("🎯 Groq Cloud + Local Mac M2 + HuggingFace")
    print("💾 Intelligent caching for instant responses")
    print("🏁 Parallel engine racing for optimal speed")
    print()
    
    # Initialize live system
    live_jarvis = LiveAcceleratedJARVIS()
    
    if await live_jarvis.initialize():
        # Start interactive mode
        await live_jarvis.start_interactive_mode()
    else:
        print("❌ Failed to initialize JARVIS. Please check your configuration.")
        return 1
    
    return 0

if __name__ == "__main__":
    # Load API keys from environment variables or .env file
    if not os.getenv('GROQ_API_KEY'):
        print("⚠️  GROQ_API_KEY not found in environment variables")
    if not os.getenv('HUGGINGFACE_TOKEN'):
        print("⚠️  HUGGINGFACE_TOKEN not found in environment variables")
    
    # Run the live system
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

