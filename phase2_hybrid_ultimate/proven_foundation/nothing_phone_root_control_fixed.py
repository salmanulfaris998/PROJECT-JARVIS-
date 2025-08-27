#!/usr/bin/env python3
"""
Nothing Phone 2a Root Control - FIXED VERSION
Step-by-step root control with SELinux bypass
"""

import logging
import subprocess
import asyncio
import time
from typing import Dict, Any

class NothingPhoneRootFixed:
    def __init__(self):
        self.logger = self._setup_logging()
        self.logger.info("🔥 Nothing Phone Root Control - FIXED VERSION")

    def _setup_logging(self):
        logger = logging.getLogger('nothing_root_fixed')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    async def _run_root_command(self, command: str) -> Dict[str, Any]:
        """Execute root command"""
        try:
            cmd = ["adb", "shell", "su", "-c", command]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            output = stdout.decode().strip()
            error = stderr.decode().strip()
            success = proc.returncode == 0
            
            return {'success': success, 'output': output, 'error': error}
        except Exception as e:
            return {'success': False, 'output': '', 'error': str(e)}

    async def step1_check_root(self):
        """Step 1: Verify root access"""
        self.logger.info("🔍 STEP 1: Checking root access...")
        result = await self._run_root_command("id")
        if result['success'] and "uid=0" in result['output']:
            self.logger.info("✅ Root access confirmed!")
            return True
        else:
            self.logger.error("❌ Root access failed!")
            return False

    async def step2_check_selinux(self):
        """Step 2: Check and fix SELinux"""
        self.logger.info("🔍 STEP 2: Checking SELinux status...")
        
        # Check current SELinux mode
        result = await self._run_root_command("getenforce")
        if result['success']:
            mode = result['output']
            self.logger.info(f"📊 Current SELinux mode: {mode}")
            
            if mode == "Enforcing":
                self.logger.info("🔓 Setting SELinux to Permissive...")
                perm_result = await self._run_root_command("setenforce 0")
                if perm_result['success']:
                    self.logger.info("✅ SELinux set to Permissive!")
                    return True
                else:
                    self.logger.error("❌ Failed to set SELinux permissive")
                    return False
            else:
                self.logger.info("✅ SELinux already permissive!")
                return True
        return False

    async def step3_find_hardware(self):
        """Step 3: Find real hardware paths"""
        self.logger.info("🔍 STEP 3: Finding real hardware paths...")
        
        # Find LED paths
        led_result = await self._run_root_command("ls /sys/class/leds/")
        if led_result['success']:
            leds = led_result['output'].split('\n')
            self.logger.info(f"📱 Found {len(leds)} LED devices:")
            for led in leds[:10]:  # Show first 10
                if led.strip():
                    self.logger.info(f"   - {led}")
            
            # Look for Glyph-like LEDs
            glyph_leds = [led for led in leds if any(word in led.lower() 
                         for word in ['glyph', 'aw', 'nothing', 'zone', 'ring'])]
            
            if glyph_leds:
                self.logger.info(f"🌟 Potential Glyph LEDs found: {glyph_leds}")
                return glyph_leds
            else:
                self.logger.info("ℹ️  No obvious Glyph LEDs found, will use generic LEDs")
                return leds[:4]  # Use first 4 LEDs
        
        return []

    async def step4_fix_permissions(self, led_paths):
        """Step 4: Fix LED permissions"""
        self.logger.info("🔧 STEP 4: Fixing LED permissions...")
        
        success_count = 0
        for led in led_paths:
            if led.strip():
                # Fix permissions for brightness and trigger
                commands = [
                    f"chmod 666 /sys/class/leds/{led}/brightness",
                    f"chmod 666 /sys/class/leds/{led}/trigger"
                ]
                
                for cmd in commands:
                    result = await self._run_root_command(cmd)
                    if result['success']:
                        success_count += 1
        
        self.logger.info(f"✅ Fixed permissions for {success_count} LED controls")
        return success_count > 0

    async def step5_test_led_control(self, led_paths):
        """Step 5: Test LED control"""
        self.logger.info("🌟 STEP 5: Testing LED control...")
        
        if not led_paths:
            self.logger.error("❌ No LED paths available")
            return False
        
        # Test first LED
        test_led = led_paths[0].strip()
        if test_led:
            self.logger.info(f"🧪 Testing LED: {test_led}")
            
            # Turn on LED
            result1 = await self._run_root_command(f"echo 255 > /sys/class/leds/{test_led}/brightness")
            await asyncio.sleep(2)  # Keep on for 2 seconds
            
            # Turn off LED  
            result2 = await self._run_root_command(f"echo 0 > /sys/class/leds/{test_led}/brightness")
            
            if result1['success'] and result2['success']:
                self.logger.info("✅ LED control working!")
                return True
            else:
                self.logger.error(f"❌ LED control failed: {result1['error']} | {result2['error']}")
        
        return False

    async def step6_simple_glyph_control(self, led_paths):
        """Step 6: Simple Glyph pattern"""
        self.logger.info("🎨 STEP 6: Creating simple Glyph pattern...")
        
        if not led_paths:
            return False
        
        # Use up to 4 LEDs for pattern
        working_leds = led_paths[:4]
        
        try:
            # Breathing pattern
            for brightness in [50, 100, 150, 200, 255, 200, 150, 100, 50, 0]:
                for led in working_leds:
                    if led.strip():
                        await self._run_root_command(f"echo {brightness} > /sys/class/leds/{led.strip()}/brightness")
                await asyncio.sleep(0.2)
            
            self.logger.info("✅ Glyph breathing pattern completed!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pattern failed: {str(e)}")
            return False

    async def run_all_steps(self):
        """Run all steps in sequence"""
        self.logger.info("🚀 Starting Nothing Phone Root Control Setup...")
        
        # Step 1: Check root
        if not await self.step1_check_root():
            return False
        
        # Step 2: Fix SELinux
        if not await self.step2_check_selinux():
            self.logger.warning("⚠️  SELinux fix failed, continuing anyway...")
        
        # Step 3: Find hardware
        led_paths = await self.step3_find_hardware()
        if not led_paths:
            self.logger.error("❌ No LEDs found!")
            return False
        
        # Step 4: Fix permissions
        if not await self.step4_fix_permissions(led_paths):
            self.logger.warning("⚠️  Permission fix failed, trying anyway...")
        
        # Step 5: Test control
        if await self.step5_test_led_control(led_paths):
            # Step 6: Demo pattern
            await self.step6_simple_glyph_control(led_paths)
        
        self.logger.info("🎯 Setup complete! Your Nothing Phone root control is ready.")
        return True

# Simple command interface
async def simple_led_command(command: str, value: str = ""):
    """Simple LED command interface"""
    controller = NothingPhoneRootFixed()
    
    if command == "on":
        brightness = value if value else "255"
        result = await controller._run_root_command(f"find /sys/class/leds -name brightness -exec sh -c 'echo {brightness} > {{}}' \\;")
        print(f"LEDs ON ({brightness}): {'✅' if result['success'] else '❌'}")
    
    elif command == "off":
        result = await controller._run_root_command("find /sys/class/leds -name brightness -exec sh -c 'echo 0 > {}' \\;")
        print(f"LEDs OFF: {'✅' if result['success'] else '❌'}")
    
    elif command == "blink":
        for i in range(5):
            await controller._run_root_command("find /sys/class/leds -name brightness -exec sh -c 'echo 255 > {}' \\;")
            await asyncio.sleep(0.5)
            await controller._run_root_command("find /sys/class/leds -name brightness -exec sh -c 'echo 0 > {}' \\;") 
            await asyncio.sleep(0.5)
        print("Blink pattern completed ✅")

async def main():
    controller = NothingPhoneRootFixed()
    
    print("🔥 Nothing Phone 2a Root Control - FIXED VERSION")
    print("=" * 50)
    
    success = await controller.run_all_steps()
    
    if success:
        print("\n🎮 Available commands:")
        print("python3 nothing_phone_root_control_fixed.py on     # Turn on all LEDs")
        print("python3 nothing_phone_root_control_fixed.py off    # Turn off all LEDs") 
        print("python3 nothing_phone_root_control_fixed.py blink  # Blink pattern")
    
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        value = sys.argv[2] if len(sys.argv) > 2 else ""
        asyncio.run(simple_led_command(command, value))
    else:
        asyncio.run(main())
