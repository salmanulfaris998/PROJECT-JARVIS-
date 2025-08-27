#!/usr/bin/env python3
"""
JARVIS Voice Command System v3.0 - Phase 2 Ultimate (STANDALONE)
Advanced Natural Language Processing - Ready to run immediately
Real-time voice recognition, intent processing, and action execution
"""

import asyncio
import speech_recognition as sr
import pyttsx3
import numpy as np
import json
import sqlite3
import logging
import time
import threading
import queue
import wave
import pyaudio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import re
import difflib
from dataclasses import dataclass
from enum import Enum
import contextlib
import sys
import os

class VoiceCommandState(Enum):
    """Voice command system states"""
    INACTIVE = "inactive"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    ERROR = "error"

class IntentCategory(Enum):
    """Command intent categories"""
    DEVICE_CONTROL = "device_control"
    AI_CONVERSATION = "ai_conversation"
    SYSTEM_QUERY = "system_query"
    AUTOMATION = "automation"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"

@dataclass
class VoiceCommand:
    """Voice command data structure"""
    raw_text: str
    confidence: float
    intent: IntentCategory
    entities: Dict[str, Any]
    timestamp: datetime
    processing_time: float
    response_generated: bool = False

class AdvancedVoiceProcessor:
    """Advanced voice processing with multiple engines"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.tts_engine = None
        self.audio_queue = queue.Queue()
        self.is_calibrated = False
        self.noise_threshold = None
        self.setup_audio_systems()
    
    def setup_audio_systems(self):
        """Initialize audio input/output systems"""
        try:
            # Setup microphone
            self.microphone = sr.Microphone()
            
            # Setup TTS engine
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 180)  # Speed
            self.tts_engine.setProperty('volume', 0.9)  # Volume
            
            # Get available voices and set a good one
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voices for JARVIS
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            print("✅ Audio systems initialized")
            
        except Exception as e:
            print(f"❌ Audio setup failed: {e}")
            raise
    
    def calibrate_microphone(self) -> bool:
        """Calibrate microphone for ambient noise"""
        try:
            print("🎤 Calibrating microphone for ambient noise...")
            print("Please remain quiet for 3 seconds...")
            
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=3)
                self.noise_threshold = self.recognizer.energy_threshold
                self.is_calibrated = True
            
            print(f"✅ Microphone calibrated - Noise threshold: {self.noise_threshold}")
            return True
            
        except Exception as e:
            print(f"❌ Microphone calibration failed: {e}")
            return False
    
    async def listen_for_wake_word(self, wake_words: List[str] = ["jarvis", "hey jarvis"]) -> bool:
        """Listen for wake word activation"""
        try:
            with self.microphone as source:
                print("👂 Listening for wake word...")
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
            
            # Process audio in background
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None, self.recognizer.recognize_google, audio
            )
            
            text_lower = text.lower()
            for wake_word in wake_words:
                if wake_word in text_lower:
                    return True
            
            return False
            
        except sr.WaitTimeoutError:
            return False
        except sr.UnknownValueError:
            return False
        except Exception as e:
            print(f"❌ Wake word detection error: {e}")
            return False
    
    async def capture_command(self, timeout: float = 5.0) -> Optional[Tuple[str, float]]:
        """Capture voice command with confidence score"""
        try:
            with self.microphone as source:
                print("🎤 Listening for command...")
                # Listen with timeout
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            # Process with multiple recognition engines for best results
            recognition_results = []
            
            # Google Speech Recognition
            try:
                loop = asyncio.get_event_loop()
                google_text = await loop.run_in_executor(
                    None, self.recognizer.recognize_google, audio
                )
                recognition_results.append(("google", google_text, 0.8))
            except:
                pass
            
            # Sphinx (offline backup)
            try:
                sphinx_text = await loop.run_in_executor(
                    None, self.recognizer.recognize_sphinx, audio
                )
                recognition_results.append(("sphinx", sphinx_text, 0.6))
            except:
                pass
            
            if not recognition_results:
                return None
            
            # Return best result (prefer Google for higher accuracy)
            best_result = max(recognition_results, key=lambda x: x[2])
            return best_result[1], best_result[2]
            
        except sr.WaitTimeoutError:
            print("⏰ Command timeout - no speech detected")
            return None
        except Exception as e:
            print(f"❌ Command capture error: {e}")
            return None
    
    async def speak_response(self, text: str, emotion: str = "neutral") -> bool:
        """Speak response with emotional tone"""
        try:
            # Adjust voice properties based on emotion
            if emotion == "excited":
                self.tts_engine.setProperty('rate', 200)
                self.tts_engine.setProperty('volume', 1.0)
            elif emotion == "calm":
                self.tts_engine.setProperty('rate', 160)
                self.tts_engine.setProperty('volume', 0.8)
            elif emotion == "urgent":
                self.tts_engine.setProperty('rate', 220)
                self.tts_engine.setProperty('volume', 1.0)
            else:  # neutral
                self.tts_engine.setProperty('rate', 180)
                self.tts_engine.setProperty('volume', 0.9)
            
            # Speak in separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._speak_sync, text)
            
            return True
            
        except Exception as e:
            print(f"❌ Speech synthesis error: {e}")
            return False
    
    def _speak_sync(self, text: str):
        """Synchronous speech function for threading"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

class IntentProcessor:
    """Advanced intent recognition and entity extraction"""
    
    def __init__(self):
        self.command_patterns = self._load_command_patterns()
        self.entity_extractors = self._setup_entity_extractors()
    
    def _load_command_patterns(self) -> Dict[IntentCategory, List[Dict[str, Any]]]:
        """Load command patterns for intent recognition"""
        return {
            IntentCategory.DEVICE_CONTROL: [
                {
                    "patterns": [
                        r"(turn on|enable|activate).*(glyph|led|light)",
                        r"(set|change).*(performance|gaming|battery) mode",
                        r"(take|capture).*(photo|picture|screenshot)",
                        r"(record|start).*(video|recording)",
                        r"(open|launch|start).*(camera|gallery|settings)",
                        r"(control|manage).*(wifi|bluetooth|hotspot)",
                        r"(increase|decrease|set).*(brightness|volume)",
                        r"(enable|disable).*(airplane mode|do not disturb)"
                    ],
                    "confidence": 0.9
                }
            ],
            IntentCategory.AI_CONVERSATION: [
                {
                    "patterns": [
                        r"(what|how|why|when|where).*(is|are|do|does|can|will)",
                        r"(tell me|explain|describe).*(about|what|how)",
                        r"(help me|assist me|guide me)",
                        r"(what do you think|your opinion|recommend)",
                        r"(remember|note|save).*(this|that)",
                        r"(good morning|good evening|hello|hi) jarvis"
                    ],
                    "confidence": 0.8
                }
            ],
            IntentCategory.SYSTEM_QUERY: [
                {
                    "patterns": [
                        r"(what.s|check|show).*(battery|temperature|memory|storage)",
                        r"(how.s|what.s).*(performance|speed|cpu|ram)",
                        r"(list|show|display).*(apps|processes|connections)",
                        r"(system|device|phone).*(status|info|stats)",
                        r"(network|wifi|data).*(status|speed|connection)"
                    ],
                    "confidence": 0.85
                }
            ],
            IntentCategory.AUTOMATION: [
                {
                    "patterns": [
                        r"(automate|schedule|set up).*(routine|task|reminder)",
                        r"(when i|if i|whenever).*(arrive|leave|wake up)",
                        r"(create|make|setup).*(automation|rule|trigger)",
                        r"(remind me|alert me|notify me)",
                        r"(smart|intelligent).*(routine|automation)"
                    ],
                    "confidence": 0.8
                }
            ],
            IntentCategory.EMERGENCY: [
                {
                    "patterns": [
                        r"(emergency|help|urgent|critical)",
                        r"(call|contact).*(emergency|911|police|ambulance)",
                        r"(battery|power).*(dying|critical|emergency)",
                        r"(lost|stolen|missing).*(phone|device)",
                        r"(panic|sos|mayday)"
                    ],
                    "confidence": 0.95
                }
            ]
        }
    
    def _setup_entity_extractors(self) -> Dict[str, callable]:
        """Setup entity extraction functions"""
        return {
            "number": self._extract_numbers,
            "app_name": self._extract_app_names,
            "device_feature": self._extract_device_features,
            "time": self._extract_time_entities,
            "location": self._extract_locations
        }
    
    def process_command(self, text: str) -> Tuple[IntentCategory, float, Dict[str, Any]]:
        """Process command to extract intent and entities"""
        text_lower = text.lower()
        
        best_intent = IntentCategory.UNKNOWN
        best_confidence = 0.0
        
        # Intent recognition
        for intent_category, pattern_groups in self.command_patterns.items():
            for pattern_group in pattern_groups:
                for pattern in pattern_group["patterns"]:
                    if re.search(pattern, text_lower):
                        confidence = pattern_group["confidence"]
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_intent = intent_category
        
        # Entity extraction
        entities = {}
        for entity_type, extractor in self.entity_extractors.items():
            extracted = extractor(text_lower)
            if extracted:
                entities[entity_type] = extracted
        
        return best_intent, best_confidence, entities
    
    def _extract_numbers(self, text: str) -> List[int]:
        """Extract numbers from text"""
        numbers = []
        # Written numbers
        word_to_num = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "twenty": 20, "thirty": 30, "fifty": 50,
            "hundred": 100
        }
        
        for word, num in word_to_num.items():
            if word in text:
                numbers.append(num)
        
        # Digit numbers
        digit_matches = re.findall(r'\b\d+\b', text)
        numbers.extend([int(match) for match in digit_matches])
        
        return numbers
    
    def _extract_app_names(self, text: str) -> List[str]:
        """Extract app names from text"""
        common_apps = [
            "camera", "gallery", "settings", "phone", "messages", "contacts",
            "chrome", "youtube", "spotify", "instagram", "whatsapp", "telegram",
            "netflix", "maps", "gmail", "calendar", "calculator", "clock"
        ]
        
        found_apps = []
        for app in common_apps:
            if app in text:
                found_apps.append(app)
        
        return found_apps
    
    def _extract_device_features(self, text: str) -> List[str]:
        """Extract device features from text"""
        features = [
            "glyph", "led", "camera", "flash", "hotspot", "wifi", "bluetooth",
            "performance", "gaming", "battery", "brightness", "volume",
            "airplane mode", "do not disturb", "location", "nfc"
        ]
        
        found_features = []
        for feature in features:
            if feature in text:
                found_features.append(feature)
        
        return found_features
    
    def _extract_time_entities(self, text: str) -> List[str]:
        """Extract time-related entities"""
        time_patterns = [
            r'\b\d{1,2}:\d{2}\b',  # 12:30
            r'\b\d{1,2} (am|pm)\b',  # 3 pm
            r'\b(morning|afternoon|evening|night)\b',
            r'\b(today|tomorrow|yesterday)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        ]
        
        found_times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text)
            found_times.extend(matches)
        
        return found_times
    
    def _extract_locations(self, text: str) -> List[str]:
        """Extract location entities"""
        location_keywords = ["home", "work", "office", "school", "gym", "mall", "park"]
        found_locations = []
        
        for location in location_keywords:
            if location in text:
                found_locations.append(location)
        
        return found_locations

class VoiceCommandSystem:
    """Main voice command system - STANDALONE VERSION"""
    
    def __init__(self):
        self.state = VoiceCommandState.INACTIVE
        self.logger = self._setup_logging()
        self.voice_processor = AdvancedVoiceProcessor()
        self.intent_processor = IntentProcessor()
        
        # Command history
        self.command_history = []
        self.session_stats = {
            "commands_processed": 0,
            "successful_commands": 0,
            "avg_response_time": 0.0,
            "wake_word_detections": 0,
            "session_start": time.time()
        }
        
        # Database
        self.db_path = Path("logs/voice_commands.db")
        self._init_database()
        
        self.logger.info("🎤 Voice Command System v3.0 initialized (STANDALONE)")

    def _setup_logging(self) -> logging.Logger:
        """Setup advanced logging system"""
        logger = logging.getLogger('voice_command_system')
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / f'voice_commands_{datetime.now().strftime("%Y%m%d")}.log')
        file_formatter = logging.Formatter('%(asctime)s | VOICE | %(levelname)s | %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('🎤 %(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _init_database(self):
        """Initialize voice command database"""
        try:
            self.db_path.parent.mkdir(exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS voice_commands (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        raw_text TEXT NOT NULL,
                        confidence REAL,
                        intent TEXT,
                        entities TEXT,
                        processing_time REAL,
                        response_generated BOOLEAN,
                        success BOOLEAN
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS session_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_start TEXT NOT NULL,
                        session_end TEXT,
                        total_commands INTEGER,
                        successful_commands INTEGER,
                        avg_response_time REAL,
                        wake_word_detections INTEGER
                    )
                ''')
                
                conn.commit()
            
            self.logger.info("✅ Voice command database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
    
    async def calibrate_system(self) -> bool:
        """Calibrate voice recognition system"""
        try:
            print("\n🎤 VOICE SYSTEM CALIBRATION")
            print("=" * 50)
            
            # Calibrate microphone
            if not self.voice_processor.calibrate_microphone():
                return False
            
            # Test wake word detection
            print("\n🧪 Testing wake word detection...")
            print("Please say 'Hey JARVIS' when ready:")
            
            wake_detected = False
            for attempt in range(3):
                if await self.voice_processor.listen_for_wake_word():
                    print("✅ Wake word detected successfully!")
                    wake_detected = True
                    break
                print(f"Attempt {attempt + 1}/3 - Please try again")
            
            if not wake_detected:
                print("❌ Wake word detection failed")
                return False
            
            # Test command recognition
            print("\n🧪 Testing command recognition...")
            print("Please say a simple command (e.g., 'What's my battery level?'):")
            
            command_result = await self.voice_processor.capture_command()
            if command_result:
                text, confidence = command_result
                print(f"✅ Command captured: '{text}' (Confidence: {confidence:.2f})")
                
                # Test TTS response
                await self.voice_processor.speak_response(
                    "Voice system calibration successful. JARVIS is ready to assist you."
                )
                
                self.state = VoiceCommandState.INACTIVE
                return True
            else:
                print("❌ Command recognition failed")
                return False
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return False
    
    async def process_voice_command(self, raw_text: str, confidence: float) -> Optional[VoiceCommand]:
        """Process voice command with built-in responses"""
        start_time = time.time()
        
        try:
            # Extract intent and entities
            intent, intent_confidence, entities = self.intent_processor.process_command(raw_text)
            
            # Create voice command object
            voice_cmd = VoiceCommand(
                raw_text=raw_text,
                confidence=confidence,
                intent=intent,
                entities=entities,
                timestamp=datetime.now(),
                processing_time=0.0
            )
            
            self.logger.info(f"Processing command: '{raw_text}' (Intent: {intent.value})")
            
            # Built-in response system
            response = await self._handle_command_builtin(voice_cmd)
            
            # Generate spoken response
            if response and response.get('success'):
                await self.voice_processor.speak_response(
                    response.get('message', 'Command completed'),
                    response.get('emotion', 'neutral')
                )
                voice_cmd.response_generated = True
            
            # Update processing time and stats
            voice_cmd.processing_time = time.time() - start_time
            self.session_stats["commands_processed"] += 1
            if response and response.get('success'):
                self.session_stats["successful_commands"] += 1
            
            # Update average response time
            current_avg = self.session_stats["avg_response_time"]
            total_commands = self.session_stats["commands_processed"]
            self.session_stats["avg_response_time"] = (
                (current_avg * (total_commands - 1) + voice_cmd.processing_time) / total_commands
            )
            
            # Save to database
            await self._save_command_to_db(voice_cmd, response and response.get('success', False))
            
            return voice_cmd
            
        except Exception as e:
            self.logger.error(f"Command processing failed: {e}")
            await self.voice_processor.speak_response(
                "I encountered an error processing that command. Please try again.",
                "apologetic"
            )
            return None
    
    async def _handle_command_builtin(self, cmd: VoiceCommand) -> Dict[str, Any]:
        """Handle commands with built-in responses"""
        try:
            text = cmd.raw_text.lower()
            
            # Device control simulation
            if cmd.intent == IntentCategory.DEVICE_CONTROL:
                if "glyph" in text or "led" in text:
                    return {
                        'success': True,
                        'message': 'Glyph LED control simulated. In Phase 2, this will control real LEDs.',
                        'emotion': 'confident'
                    }
                elif "performance" in text or "gaming" in text:
                    return {
                        'success': True,
                        'message': 'Performance mode change simulated. In Phase 2, this will optimize your device.',
                        'emotion': 'confident'
                    }
                elif "photo" in text or "picture" in text:
                    return {
                        'success': True,
                        'message': 'Photo capture simulated. In Phase 2, this will take real photos.',
                        'emotion': 'pleased'
                    }
                elif "camera" in text:
                    return {
                        'success': True,
                        'message': 'Camera opening simulated. In Phase 2, this will launch the camera app.',
                        'emotion': 'neutral'
                    }
                else:
                    return {
                        'success': True,
                        'message': 'Device control command recognized. Full functionality available in Phase 2.',
                        'emotion': 'neutral'
                    }
            
            # AI conversation
            elif cmd.intent == IntentCategory.AI_CONVERSATION:
                conversation_responses = [
                    "I'm JARVIS, your advanced AI assistant. I'm currently in Phase 2 development.",
                    "Hello! I'm here to help you. My full capabilities will be available once Phase 2 is complete.",
                    "I understand your request. I'm learning and evolving to serve you better.",
                    "That's an interesting question. My AI brain is getting more sophisticated with each phase.",
                    "I'm designed to be your intelligent companion. Phase 2 will bring many new abilities."
                ]
                
                import random
                response_text = random.choice(conversation_responses)
                
                return {
                    'success': True,
                    'message': response_text,
                    'emotion': 'friendly'
                }
            
            # System queries
            elif cmd.intent == IntentCategory.SYSTEM_QUERY:
                if "battery" in text:
                    return {
                        'success': True,
                        'message': 'Battery monitoring is simulated. In Phase 2, I will provide real battery information.',
                        'emotion': 'informative'
                    }
                elif "temperature" in text:
                    return {
                        'success': True,
                        'message': 'Temperature monitoring is simulated. In Phase 2, I will monitor actual device temperature.',
                        'emotion': 'informative'
                    }
                elif "performance" in text or "stats" in text:
                    response_msg = f"Voice system statistics: {self.session_stats['commands_processed']} commands processed with {self.session_stats['avg_response_time']:.2f} second average response time."
                    return {
                        'success': True,
                        'message': response_msg,
                        'emotion': 'proud'
                    }
                else:
                    return {
                        'success': True,
                        'message': 'System information request recognized. Full monitoring available in Phase 2.',
                        'emotion': 'informative'
                    }
            
            # Automation
            elif cmd.intent == IntentCategory.AUTOMATION:
                return {
                    'success': True,
                    'message': 'Automation features are in development. Advanced task automation will be available in Phase 2.',
                    'emotion': 'promising'
                }
            
            # Emergency
            elif cmd.intent == IntentCategory.EMERGENCY:
                self.logger.warning(f"Emergency command detected: {cmd.raw_text}")
                return {
                    'success': True,
                    'message': 'Emergency command detected. Emergency protocols will be fully implemented in Phase 2.',
                    'emotion': 'urgent'
                }
            
            # Unknown commands
            else:
                return {
                    'success': True,
                    'message': 'I recognize your voice but need more context. My understanding will improve in Phase 2.',
                    'emotion': 'curious'
                }
                
        except Exception as e:
            self.logger.error(f"Built-in command handling error: {e}")
            return {
                'success': False,
                'message': 'I encountered an issue processing that command.',
                'emotion': 'apologetic'
            }
    
    async def _save_command_to_db(self, cmd: VoiceCommand, success: bool):
        """Save command to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO voice_commands 
                    (timestamp, raw_text, confidence, intent, entities, processing_time, response_generated, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cmd.timestamp.isoformat(),
                    cmd.raw_text,
                    cmd.confidence,
                    cmd.intent.value,
                    json.dumps(cmd.entities),
                    cmd.processing_time,
                    cmd.response_generated,
                    success
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database save error: {e}")
    
    async def run_voice_session(self):
        """Run main voice command session"""
        try:
            print("\n🎤 JARVIS VOICE COMMAND SESSION STARTING")
            print("=" * 55)
            print("🤖 STANDALONE VERSION - Phase 2 Foundation")
            
            # Calibrate system
            if not await self.calibrate_system():
                print("❌ System calibration failed")
                return
            
            print(f"\n✅ JARVIS Voice System Ready!")
            print(f"Say 'Hey JARVIS' to activate voice commands")
            print(f"Say 'Stop listening' to end session")
            print("=" * 55)
            
            # Main listening loop
            self.state = VoiceCommandState.LISTENING
            
            while self.state != VoiceCommandState.INACTIVE:
                try:
                    # Listen for wake word
                    if await self.voice_processor.listen_for_wake_word():
                        self.session_stats["wake_word_detections"] += 1
                        print("👂 Wake word detected! Listening for command...")
                        
                        # Capture command
                        command_result = await self.voice_processor.capture_command()
                        
                        if command_result:
                            text, confidence = command_result
                            
                            # Check for session termination
                            if "stop listening" in text.lower() or "goodbye jarvis" in text.lower():
                                await self.voice_processor.speak_response("Goodbye! Voice session ending.")
                                break
                            
                            # Process command
                            self.state = VoiceCommandState.PROCESSING
                            await self.process_voice_command(text, confidence)
                            self.state = VoiceCommandState.LISTENING
                        else:
                            print("⚠️ No command captured")
                    
                except KeyboardInterrupt:
                    print("\n👋 Voice session interrupted by user")
                    break
                except Exception as e:
                    self.logger.error(f"Session error: {e}")
                    await asyncio.sleep(1)
            
            # Session cleanup
            await self._save_session_stats()
            print(f"\n📊 Session complete - {self.session_stats['commands_processed']} commands processed")
            
        except Exception as e:
            self.logger.error(f"Voice session failed: {e}")
        finally:
            self.state = VoiceCommandState.INACTIVE
    
    async def _save_session_stats(self):
        """Save session statistics to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO session_stats 
                    (session_start, session_end, total_commands, successful_commands, avg_response_time, wake_word_detections)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.fromtimestamp(self.session_stats["session_start"]).isoformat(),
                    datetime.now().isoformat(),
                    self.session_stats["commands_processed"],
                    self.session_stats["successful_commands"],
                    self.session_stats["avg_response_time"],
                    self.session_stats["wake_word_detections"]
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Session stats save error: {e}")

# ========== MAIN EXECUTION ==========

async def main():
    """Main voice command system execution"""
    try:
        voice_system = VoiceCommandSystem()
        await voice_system.run_voice_session()
        
    except KeyboardInterrupt:
        print("\n👋 Voice system shutdown requested")
    except Exception as e:
        print(f"❌ Voice system error: {e}")

if __name__ == "__main__":
    print("🔥 JARVIS Voice Command System v3.0 - STANDALONE")
    print("🤖 Phase 2 Foundation - Ready for Testing")
    print("=" * 60)
    asyncio.run(main())
