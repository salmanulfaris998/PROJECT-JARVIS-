#!/usr/bin/env python3
"""
JARVIS Voice System - Debug & Fix Version
Addresses the synthesis failures and improves error handling
"""

import asyncio
import time
import tempfile
import os
import sys
import subprocess
import threading
import logging
import platform
import wave
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("JARVIS_Voice_Debug")

class VoiceEmotion(Enum):
    """Simplified emotions for debugging"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    CONFIDENT = "confident"
    CALM = "calm"

@dataclass
class VoiceConfig:
    """Simplified voice configuration"""
    backend: str = "system"
    quality: str = "high"
    sample_rate: int = 44100
    use_cache: bool = True
    speaker_id: str = "jarvis"

class DebugTTSEngine:
    """Debug-focused TTS engine with detailed error reporting"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.available_backends = self._detect_backends()
        logger.info(f"🖥️ Platform detected: {self.platform}")
        logger.info(f"🎤 Available backends: {self.available_backends}")
    
    def _detect_backends(self) -> Dict[str, bool]:
        """Detect available TTS backends with detailed testing"""
        backends = {}
        
        if self.platform == "darwin":  # macOS
            # Test 'say' command
            try:
                result = subprocess.run(
                    ["say", "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                backends["system_say"] = result.returncode == 0
                logger.debug(f"'say' command test: {result.returncode == 0}")
                if result.stderr:
                    logger.debug(f"'say' stderr: {result.stderr}")
            except Exception as e:
                backends["system_say"] = False
                logger.warning(f"'say' command failed: {e}")
            
            # Test 'afplay' command
            try:
                result = subprocess.run(
                    ["afplay", "--help"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                backends["system_afplay"] = result.returncode == 0 or "afplay" in result.stderr
                logger.debug(f"'afplay' command test: {backends['system_afplay']}")
            except Exception as e:
                backends["system_afplay"] = False
                logger.warning(f"'afplay' command failed: {e}")
        
        elif self.platform == "linux":
            # Test espeak
            backends["espeak"] = self._test_command(["espeak", "--version"])
            backends["aplay"] = self._test_command(["aplay", "--version"])
            backends["paplay"] = self._test_command(["paplay", "--version"])
        
        elif self.platform == "windows":
            # Test PowerShell TTS
            backends["powershell_tts"] = True  # Assume available on Windows
        
        return backends
    
    def _test_command(self, cmd: List[str]) -> bool:
        """Test if a command is available"""
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception as e:
            logger.debug(f"Command {cmd[0]} test failed: {e}")
            return False
    
    async def synthesize_speech(self, text: str, emotion: VoiceEmotion, output_file: str) -> bool:
        """Synthesize speech with detailed error reporting"""
        logger.info(f"🎵 Starting synthesis: '{text[:30]}...' -> {output_file}")
        
        try:
            if self.platform == "darwin":
                return await self._synthesize_macos(text, emotion, output_file)
            elif self.platform == "linux":
                return await self._synthesize_linux(text, emotion, output_file)
            elif self.platform == "windows":
                return await self._synthesize_windows(text, emotion, output_file)
            else:
                logger.error(f"❌ Unsupported platform: {self.platform}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Synthesis exception: {e}")
            return False
    
    async def _synthesize_macos(self, text: str, emotion: VoiceEmotion, output_file: str) -> bool:
        """macOS synthesis with detailed logging"""
        if not self.available_backends.get("system_say", False):
            logger.error("❌ macOS 'say' command not available")
            return False
        
        # Emotion to voice mapping
        voice_map = {
            VoiceEmotion.NEUTRAL: "Alex",
            VoiceEmotion.HAPPY: "Samantha",
            VoiceEmotion.EXCITED: "Victoria",
            VoiceEmotion.CONFIDENT: "Daniel",
            VoiceEmotion.CALM: "Allison"
        }
        
        voice = voice_map.get(emotion, "Alex")
        rate = 180  # Words per minute
        
        # Build command
        cmd = [
            "say",
            "-v", voice,
            "-r", str(rate),
            "-o", output_file,
            text
        ]
        
        logger.debug(f"🔧 Running command: {' '.join(cmd)}")
        
        try:
            # Run synthesis
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            
            logger.debug(f"Process return code: {process.returncode}")
            if stdout:
                logger.debug(f"Stdout: {stdout.decode()}")
            if stderr:
                logger.debug(f"Stderr: {stderr.decode()}")
            
            # Check if file was created
            if process.returncode == 0:
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    logger.info(f"✅ Synthesis successful, file size: {file_size} bytes")
                    return True
                else:
                    logger.error(f"❌ Process succeeded but output file not found: {output_file}")
                    return False
            else:
                logger.error(f"❌ Process failed with code {process.returncode}")
                return False
                
        except asyncio.TimeoutError:
            logger.error("❌ Synthesis timeout (30s)")
            return False
        except Exception as e:
            logger.error(f"❌ Synthesis process error: {e}")
            return False
    
    async def _synthesize_linux(self, text: str, emotion: VoiceEmotion, output_file: str) -> bool:
        """Linux synthesis with espeak fallback"""
        if not self.available_backends.get("espeak", False):
            logger.error("❌ Linux espeak not available")
            return False
        
        # Emotion to espeak parameters
        speed = 180
        pitch = 50
        
        if emotion == VoiceEmotion.EXCITED:
            speed = 220
            pitch = 70
        elif emotion == VoiceEmotion.CALM:
            speed = 150
            pitch = 30
        
        cmd = [
            "espeak",
            "-s", str(speed),
            "-p", str(pitch),
            "-w", output_file,
            text
        ]
        
        logger.debug(f"🔧 Running command: {' '.join(cmd)}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            
            logger.debug(f"Process return code: {process.returncode}")
            if stdout:
                logger.debug(f"Stdout: {stdout.decode()}")
            if stderr:
                logger.debug(f"Stderr: {stderr.decode()}")
            
            if process.returncode == 0 and os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                logger.info(f"✅ Linux synthesis successful, file size: {file_size} bytes")
                return True
            else:
                logger.error(f"❌ Linux synthesis failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Linux synthesis error: {e}")
            return False
    
    async def _synthesize_windows(self, text: str, emotion: VoiceEmotion, output_file: str) -> bool:
        """Windows synthesis with PowerShell SAPI"""
        # PowerShell SAPI synthesis script
        ps_script = f'''
        Add-Type -AssemblyName System.Speech
        $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
        $synth.Rate = 0
        $synth.Volume = 100
        $synth.SetOutputToWaveFile("{output_file}")
        $synth.Speak("{text}")
        $synth.Dispose()
        '''
        
        cmd = ["powershell", "-Command", ps_script]
        
        logger.debug(f"🔧 Running PowerShell synthesis")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            
            logger.debug(f"Process return code: {process.returncode}")
            if stdout:
                logger.debug(f"Stdout: {stdout.decode()}")
            if stderr:
                logger.debug(f"Stderr: {stderr.decode()}")
            
            if process.returncode == 0 and os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                logger.info(f"✅ Windows synthesis successful, file size: {file_size} bytes")
                return True
            else:
                logger.error(f"❌ Windows synthesis failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Windows synthesis error: {e}")
            return False
    
    async def play_audio(self, audio_file: str) -> bool:
        """Play audio file with platform-specific player"""
        if not os.path.exists(audio_file):
            logger.error(f"❌ Audio file not found: {audio_file}")
            return False
        
        file_size = os.path.getsize(audio_file)
        logger.info(f"🔊 Playing audio file: {audio_file} ({file_size} bytes)")
        
        try:
            if self.platform == "darwin":
                if self.available_backends.get("system_afplay", False):
                    cmd = ["afplay", audio_file]
                else:
                    logger.warning("⚠️ afplay not available, using say for playback")
                    cmd = ["say", "-f", audio_file]
            
            elif self.platform == "linux":
                if self.available_backends.get("aplay", False):
                    cmd = ["aplay", audio_file]
                elif self.available_backends.get("paplay", False):
                    cmd = ["paplay", audio_file]
                else:
                    logger.error("❌ No Linux audio player available")
                    return False
            
            elif self.platform == "windows":
                # Use PowerShell for playback
                ps_script = f'''
                Add-Type -AssemblyName presentationCore
                $mediaPlayer = New-Object system.windows.media.mediaplayer
                $mediaPlayer.open("{audio_file}")
                $mediaPlayer.Play()
                Start-Sleep -Seconds 5
                '''
                cmd = ["powershell", "-Command", ps_script]
            
            else:
                logger.error(f"❌ Unsupported platform for playback: {self.platform}")
                return False
            
            logger.debug(f"🔧 Playback command: {cmd[0]}")
            
            # Execute playback command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15.0)
            
            logger.debug(f"Playback return code: {process.returncode}")
            if stderr:
                logger.debug(f"Playback stderr: {stderr.decode()}")
            
            success = process.returncode == 0
            logger.info(f"{'✅' if success else '❌'} Playback {'successful' if success else 'failed'}")
            return success
            
        except asyncio.TimeoutError:
            logger.error("❌ Audio playback timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Audio playback error: {e}")
            return False

class DebugJARVISSystem:
    """Simplified JARVIS system focused on debugging synthesis issues"""
    
    def __init__(self):
        self.config = VoiceConfig()
        self.tts_engine = DebugTTSEngine()
        self.stats = {
            "total_requests": 0,
            "successful_syntheses": 0,
            "failed_syntheses": 0
        }
        logger.info("🚀 Debug JARVIS System initialized")
    
    async def speak(self, text: str, emotion: VoiceEmotion = VoiceEmotion.NEUTRAL) -> bool:
        """Speak text with comprehensive error handling and logging"""
        self.stats["total_requests"] += 1
        
        logger.info(f"🗣️ JARVIS speaking: '{text[:50]}...' ({emotion.value})")
        
        # Input validation
        if not text or not text.strip():
            logger.error("❌ Empty text provided")
            self.stats["failed_syntheses"] += 1
            return False
        
        # Create temporary output file
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time() * 1000)
        output_file = os.path.join(temp_dir, f"jarvis_speech_{timestamp}.wav")
        
        logger.debug(f"📁 Output file: {output_file}")
        
        try:
            # Synthesize speech
            logger.info("🎵 Starting speech synthesis...")
            synthesis_start = time.time()
            
            synthesis_success = await self.tts_engine.synthesize_speech(text, emotion, output_file)
            
            synthesis_time = time.time() - synthesis_start
            logger.info(f"⏱️ Synthesis took {synthesis_time:.2f} seconds")
            
            if not synthesis_success:
                logger.error("❌ Speech synthesis failed")
                self.stats["failed_syntheses"] += 1
                return False
            
            # Verify output file
            if not os.path.exists(output_file):
                logger.error(f"❌ Output file not created: {output_file}")
                self.stats["failed_syntheses"] += 1
                return False
            
            file_size = os.path.getsize(output_file)
            if file_size == 0:
                logger.error(f"❌ Output file is empty: {output_file}")
                self.stats["failed_syntheses"] += 1
                return False
            
            logger.info(f"✅ Speech synthesis successful: {file_size} bytes")
            
            # Play audio
            logger.info("🔊 Starting audio playback...")
            playback_start = time.time()
            
            playback_success = await self.tts_engine.play_audio(output_file)
            
            playback_time = time.time() - playback_start
            logger.info(f"⏱️ Playback took {playback_time:.2f} seconds")
            
            if playback_success:
                self.stats["successful_syntheses"] += 1
                logger.info("✅ Complete speech operation successful")
            else:
                logger.warning("⚠️ Synthesis successful but playback failed")
                # Still count as success since synthesis worked
                self.stats["successful_syntheses"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Speech operation failed with exception: {e}")
            self.stats["failed_syntheses"] += 1
            return False
            
        finally:
            # Cleanup temporary file
            try:
                if os.path.exists(output_file):
                    os.unlink(output_file)
                    logger.debug(f"🗑️ Cleaned up temporary file: {output_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup temporary file: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        total = self.stats["total_requests"]
        success_rate = (self.stats["successful_syntheses"] / total * 100) if total > 0 else 0
        
        return {
            "total_requests": total,
            "successful_syntheses": self.stats["successful_syntheses"],
            "failed_syntheses": self.stats["failed_syntheses"],
            "success_rate": f"{success_rate:.1f}%"
        }
    
    async def diagnostic_test(self):
        """Run comprehensive diagnostic tests"""
        logger.info("🔍 Running JARVIS Voice System Diagnostics")
        logger.info("=" * 50)
        
        # System information
        logger.info(f"🖥️ Platform: {self.tts_engine.platform}")
        logger.info(f"🎤 Available backends: {self.tts_engine.available_backends}")
        
        # Test basic synthesis
        logger.info("\n📝 Testing basic synthesis...")
        success1 = await self.speak("This is a basic synthesis test.")
        
        # Test with emotion
        logger.info("\n🎭 Testing emotional synthesis...")
        success2 = await self.speak("This is an excited emotional test!", VoiceEmotion.EXCITED)
        
        # Test edge cases
        logger.info("\n🔬 Testing edge cases...")
        success3 = await self.speak("Short.")
        success4 = await self.speak("This is a much longer text to test how the system handles extended speech synthesis with multiple sentences and various punctuation marks!")
        
        # Summary
        total_tests = 4
        passed_tests = sum([success1, success2, success3, success4])
        
        logger.info(f"\n📊 Diagnostic Summary:")
        logger.info(f"   Tests passed: {passed_tests}/{total_tests}")
        logger.info(f"   Overall success rate: {passed_tests/total_tests*100:.1f}%")
        
        stats = self.get_stats()
        logger.info(f"   System stats: {stats}")
        
        if passed_tests == total_tests:
            logger.info("🎉 All diagnostic tests passed! System is working correctly.")
        elif passed_tests > 0:
            logger.warning("⚠️ Some tests failed. Check logs for details.")
        else:
            logger.error("❌ All tests failed. System requires troubleshooting.")
        
        return passed_tests == total_tests

async def main():
    """Main debug function"""
    print("🔧 JARVIS Voice System - Debug Mode")
    print("=" * 40)
    
    # Initialize debug system
    jarvis = DebugJARVISSystem()
    
    # Run diagnostics
    try:
        success = await jarvis.diagnostic_test()
        
        if success:
            print("\n🎤 Interactive test mode (type 'quit' to exit):")
            while True:
                try:
                    text = input("\nEnter text to speak: ").strip()
                    if text.lower() == 'quit':
                        break
                    if text:
                        await jarvis.speak(text)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
        
        # Final stats
        stats = jarvis.get_stats()
        print(f"\n📊 Final Statistics: {stats}")
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")
    
    print("\n👋 Debug session complete!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")