"""
JARVIS Advanced 13B AI Brain - Production Grade with Human-like Intelligence
Optimized for Apple Silicon (M1/M2/M3) with enterprise-level features
Version: 2.0.0 - Production Ready

Author: AI Systems Engineering Team
License: MIT
"""

import asyncio
import time
import json
import re
import os
import sys
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import asynccontextmanager
import yaml
import pickle
from functools import lru_cache, wraps
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure professional logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging system"""
    logger = logging.getLogger("JARVIS")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Hardware optimization detection
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = "mps"
        logger.info("✅ Apple Metal Performance Shaders detected and available")
    elif torch.cuda.is_available():
        DEVICE = "cuda"
        logger.info("✅ NVIDIA CUDA acceleration available")
    else:
        DEVICE = "cpu"
        logger.info("ℹ️ Using CPU computation")
        
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    logger.warning("⚠️ PyTorch not available - using fallback systems")

# Advanced NLP libraries with graceful fallback
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
        pipeline, set_seed, TrainingArguments, Trainer
    )
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    NLP_AVAILABLE = True
    logger.info("✅ Advanced NLP libraries loaded successfully")
except ImportError:
    NLP_AVAILABLE = False
    logger.warning("⚠️ Advanced NLP libraries not available - using basic processing")

# Performance monitoring
def monitor_performance(func):
    """Decorator for performance monitoring"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()
            logger.debug(f"⚡ {func.__name__} completed in {end_time - start_time:.3f}s")
            return result
        except Exception as e:
            end_time = time.perf_counter()
            logger.error(f"❌ {func.__name__} failed after {end_time - start_time:.3f}s: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            logger.debug(f"⚡ {func.__name__} completed in {end_time - start_time:.3f}s")
            return result
        except Exception as e:
            end_time = time.perf_counter()
            logger.error(f"❌ {func.__name__} failed after {end_time - start_time:.3f}s: {e}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

class EmotionalState(Enum):
    """Enhanced emotional states with intensity levels"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    CONCERNED = "concerned"
    FOCUSED = "focused"
    AMUSED = "amused"
    CARING = "caring"
    CONFIDENT = "confident"
    THOUGHTFUL = "thoughtful"
    SURPRISED = "surprised"
    GRATEFUL = "grateful"
    CURIOUS = "curious"
    PATIENT = "patient"
    DETERMINED = "determined"
    EMPATHETIC = "empathetic"

class IntentType(Enum):
    """Comprehensive intent classification system"""
    GREETING = auto()
    QUESTION = auto()
    PHONE_CONTROL = auto()
    SYSTEM_CONTROL = auto()
    CONVERSATION = auto()
    TASK_REQUEST = auto()
    INFORMATION = auto()
    GOODBYE = auto()
    EMERGENCY = auto()
    ENTERTAINMENT = auto()
    LEARNING = auto()
    PLANNING = auto()
    REFLECTION = auto()
    UNKNOWN = auto()

class ConfidenceLevel(Enum):
    """Response confidence levels"""
    VERY_HIGH = 0.9
    HIGH = 0.75
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2

@dataclass
class JarvisConfig:
    """Comprehensive configuration management"""
    # Model settings
    model_name: str = "microsoft/DialoGPT-medium"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = DEVICE
    
    # Memory settings
    context_window: int = 10
    max_conversation_history: int = 100
    max_user_profiles: int = 50
    
    # Personality traits (0.0 to 1.0)
    personality_traits: Dict[str, float] = field(default_factory=lambda: {
        "formality": 0.8,
        "helpfulness": 0.95,
        "curiosity": 0.75,
        "humor": 0.6,
        "confidence": 0.9,
        "emotional_intelligence": 0.9,
        "proactiveness": 0.8,
        "loyalty": 0.95,
        "patience": 0.85,
        "creativity": 0.7
    })
    
    # System settings
    log_level: str = "INFO"
    log_file: Optional[str] = "jarvis.log"
    data_directory: str = "jarvis_data"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    
    # Performance settings
    async_processing: bool = True
    thread_pool_workers: int = 4
    cache_size: int = 1000
    timeout_seconds: int = 30
    
    # Security settings
    enable_privacy_mode: bool = True
    data_encryption: bool = True
    conversation_ttl_days: int = 30

    @classmethod
    def from_file(cls, config_path: str) -> 'JarvisConfig':
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return cls(**config_data)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return cls()
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        try:
            with open(config_path, 'w') as f:
                yaml.dump(asdict(self), f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

@dataclass
class JarvisResponse:
    """Enhanced structured response from JARVIS AI brain"""
    text: str
    emotion: EmotionalState
    confidence: float
    intent: IntentType
    action_required: bool
    context_data: Dict[str, Any]
    response_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)

@dataclass
class ConversationMemory:
    """Enhanced memory structure with rich metadata"""
    user_input: str
    jarvis_response: str
    user_emotion: EmotionalState
    jarvis_emotion: EmotionalState
    intent: IntentType
    confidence: float
    timestamp: datetime
    session_id: str
    context_tags: List[str] = field(default_factory=list)
    user_satisfaction: Optional[float] = None
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class SecurityManager:
    """Enterprise-grade security and privacy management"""
    
    def __init__(self, config: JarvisConfig):
        self.config = config
        self.encryption_key = self._generate_or_load_key()
        logger.info("🔒 Security manager initialized")
    
    def _generate_or_load_key(self) -> bytes:
        """Generate or load encryption key"""
        key_file = Path(self.config.data_directory) / ".jarvis_key"
        
        try:
            if key_file.exists():
                return key_file.read_bytes()
            else:
                # In production, use proper key management
                key = os.urandom(32)
                key_file.parent.mkdir(exist_ok=True)
                key_file.write_bytes(key)
                return key
        except Exception as e:
            logger.error(f"Key management error: {e}")
            return os.urandom(32)
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for privacy"""
        if not self.config.enable_privacy_mode:
            return data
        
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def sanitize_input(self, user_input: str) -> str:
        """Sanitize user input for security"""
        # Remove potential harmful patterns
        sanitized = re.sub(r'[<>"\']', '', user_input)
        sanitized = re.sub(r'\b(password|secret|key|token)\s*[:=]\s*\S+', '[REDACTED]', sanitized, flags=re.IGNORECASE)
        return sanitized.strip()

class EmotionAnalyzer:
    """Advanced emotion detection using multiple techniques"""
    
    def __init__(self, config: JarvisConfig):
        self.config = config
        self.emotion_pipeline = None
        self.emotion_patterns = self._initialize_emotion_patterns()
        self._load_emotion_model()
        logger.info("🧠 Advanced emotion analyzer initialized")
    
    def _load_emotion_model(self):
        """Load advanced emotion detection model"""
        if not NLP_AVAILABLE:
            logger.warning("Using pattern-based emotion detection")
            return
        
        try:
            self.emotion_pipeline = pipeline(
                "text-classification",
                model=self.config.emotion_model,
                device=0 if self.config.device == "cuda" else -1,
                return_all_scores=True
            )
            logger.info("✅ Advanced emotion model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load emotion model: {e}")
    
    def _initialize_emotion_patterns(self) -> Dict[EmotionalState, Dict[str, float]]:
        """Initialize sophisticated emotion detection patterns"""
        return {
            EmotionalState.HAPPY: {
                "keywords": ["great", "awesome", "wonderful", "fantastic", "love", "excited", "amazing", "perfect", "excellent", "brilliant", "super", "thrilled", "delighted"],
                "phrases": ["feeling good", "so happy", "really excited", "absolutely love"],
                "punctuation_weight": {"!": 0.3, "😊": 0.5, "😄": 0.6, "🎉": 0.4},
                "base_weight": 1.0
            },
            EmotionalState.CONCERNED: {
                "keywords": ["worried", "concerned", "problem", "issue", "help", "trouble", "stuck", "confused", "lost", "error", "wrong", "broken", "difficult", "struggling"],
                "phrases": ["not sure", "having trouble", "don't understand", "something wrong"],
                "punctuation_weight": {"?": 0.2, "😰": 0.5, "😟": 0.4},
                "base_weight": 1.0
            },
            EmotionalState.EXCITED: {
                "keywords": ["can't wait", "excited", "incredible", "wow", "unbelievable", "mind-blowing", "spectacular", "phenomenal"],
                "phrases": ["so excited", "can't believe", "this is amazing"],
                "punctuation_weight": {"!": 0.4, "!!": 0.6, "🚀": 0.5, "🎊": 0.4},
                "base_weight": 1.2
            },
            EmotionalState.AMUSED: {
                "keywords": ["funny", "hilarious", "lol", "haha", "joke", "amusing", "laugh", "comedy", "humor", "witty", "clever"],
                "phrases": ["made me laugh", "that's funny", "good one"],
                "punctuation_weight": {"😂": 0.6, "😄": 0.4, "🤣": 0.5},
                "base_weight": 1.0
            },
            EmotionalState.GRATEFUL: {
                "keywords": ["thank", "thanks", "appreciate", "grateful", "thankful", "blessing", "fortunate"],
                "phrases": ["thank you", "really appreciate", "so grateful"],
                "punctuation_weight": {"🙏": 0.5, "❤️": 0.3},
                "base_weight": 1.0
            }
        }
    
    @monitor_performance
    async def analyze_emotion(self, text: str, context: Optional[List[str]] = None) -> Tuple[EmotionalState, float]:
        """Advanced emotion analysis with context awareness"""
        if not text.strip():
            return EmotionalState.NEUTRAL, 0.5
        
        # Use AI model if available
        if self.emotion_pipeline:
            try:
                results = self.emotion_pipeline(text)
                
                # Map model emotions to our enum
                emotion_mapping = {
                    "joy": EmotionalState.HAPPY,
                    "sadness": EmotionalState.CONCERNED,
                    "anger": EmotionalState.CONCERNED,
                    "fear": EmotionalState.CONCERNED,
                    "surprise": EmotionalState.SURPRISED,
                    "love": EmotionalState.GRATEFUL,
                    "excitement": EmotionalState.EXCITED
                }
                
                # Get highest scoring emotion
                top_emotion = max(results[0], key=lambda x: x['score'])
                mapped_emotion = emotion_mapping.get(top_emotion['label'], EmotionalState.NEUTRAL)
                confidence = top_emotion['score']
                
                # Enhance with pattern analysis
                pattern_emotion, pattern_confidence = self._pattern_based_analysis(text)
                
                # Combine results (weighted average)
                if confidence > 0.7:
                    return mapped_emotion, confidence
                else:
                    # Blend AI and pattern results
                    final_confidence = (confidence + pattern_confidence) / 2
                    return mapped_emotion if confidence > pattern_confidence else pattern_emotion, final_confidence
                    
            except Exception as e:
                logger.warning(f"AI emotion analysis failed: {e}")
        
        # Fallback to pattern analysis
        return self._pattern_based_analysis(text)
    
    def _pattern_based_analysis(self, text: str) -> Tuple[EmotionalState, float]:
        """Enhanced pattern-based emotion detection"""
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, patterns in self.emotion_patterns.items():
            score = 0.0
            
            # Keyword matching with frequency
            for keyword in patterns["keywords"]:
                count = text_lower.count(keyword)
                score += count * patterns["base_weight"]
            
            # Phrase matching (higher weight)
            for phrase in patterns["phrases"]:
                if phrase in text_lower:
                    score += patterns["base_weight"] * 1.5
            
            # Punctuation and emoji analysis
            for punct, weight in patterns["punctuation_weight"].items():
                count = text.count(punct)
                score += count * weight
            
            # Context boost (if previous messages had same emotion)
            # This would need conversation context implementation
            
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            top_emotion = max(emotion_scores, key=emotion_scores.get)
            max_score = emotion_scores[top_emotion]
            # Normalize confidence (0.3 to 0.9 range)
            confidence = min(0.9, 0.3 + (max_score / 5.0))
            return top_emotion, confidence
        
        return EmotionalState.NEUTRAL, 0.5

class IntentClassifier:
    """Advanced intent classification with machine learning"""
    
    def __init__(self, config: JarvisConfig):
        self.config = config
        self.intent_patterns = self._initialize_intent_patterns()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english') if NLP_AVAILABLE else None
        self._trained = False
        logger.info("🎯 Advanced intent classifier initialized")
    
    def _initialize_intent_patterns(self) -> Dict[IntentType, Dict[str, Any]]:
        """Initialize comprehensive intent detection patterns"""
        return {
            IntentType.GREETING: {
                "keywords": ["hello", "hi", "hey", "good morning", "good evening", "good afternoon", "greetings", "howdy", "what's up"],
                "patterns": [r"\b(hi|hello|hey)\b", r"good (morning|afternoon|evening)", r"what'?s up"],
                "context_indicators": ["start", "begin", "first"],
                "confidence_boost": 0.9
            },
            IntentType.PHONE_CONTROL: {
                "keywords": ["call", "text", "message", "sms", "dial", "phone", "contact", "ring"],
                "patterns": [r"call\s+\w+", r"text\s+\w+", r"send\s+(message|sms)", r"dial\s+\d+"],
                "action_verbs": ["call", "text", "dial", "ring", "message"],
                "confidence_boost": 0.95
            },
            IntentType.SYSTEM_CONTROL: {
                "keywords": ["open", "close", "launch", "start", "stop", "settings", "wifi", "bluetooth", "volume", "brightness", "battery"],
                "patterns": [r"(open|launch|start)\s+\w+", r"turn\s+(on|off)", r"set\s+\w+\s+to"],
                "system_objects": ["app", "application", "setting", "volume", "brightness", "wifi", "bluetooth"],
                "confidence_boost": 0.9
            },
            IntentType.QUESTION: {
                "keywords": ["what", "how", "when", "where", "why", "who", "which", "explain", "tell me", "show me"],
                "patterns": [r"\b(what|how|when|where|why|who|which)\b", r"(explain|tell me|show me)"],
                "question_indicators": ["?", "help me understand", "I want to know"],
                "confidence_boost": 0.85
            },
            IntentType.TASK_REQUEST: {
                "keywords": ["can you", "please", "would you", "could you", "help me", "assist", "do", "make", "create", "set"],
                "patterns": [r"(can|could|would)\s+you", r"help\s+me", r"please\s+\w+"],
                "politeness_indicators": ["please", "kindly", "if you don't mind"],
                "confidence_boost": 0.8
            },
            IntentType.INFORMATION: {
                "keywords": ["weather", "time", "date", "news", "search", "find", "lookup", "information", "facts", "data"],
                "patterns": [r"(weather|time|date|news)", r"search\s+for", r"find\s+\w+"],
                "info_types": ["weather", "time", "news", "facts", "data", "information"],
                "confidence_boost": 0.85
            },
            IntentType.EMERGENCY: {
                "keywords": ["emergency", "urgent", "help", "crisis", "problem", "broken", "error", "crash", "stuck"],
                "patterns": [r"emergency", r"urgent", r"need\s+help\s+now"],
                "urgency_indicators": ["now", "immediately", "asap", "quickly"],
                "confidence_boost": 0.95
            },
            IntentType.GOODBYE: {
                "keywords": ["goodbye", "bye", "farewell", "see you", "exit", "quit", "stop", "end", "logout", "later"],
                "patterns": [r"\b(bye|goodbye|farewell)\b", r"see\s+you", r"(exit|quit|stop)"],
                "ending_indicators": ["end", "finish", "done", "that's all"],
                "confidence_boost": 0.9
            }
        }
    
    @monitor_performance
    async def classify_intent(self, text: str, context: Optional[List[str]] = None) -> Tuple[IntentType, float]:
        """Advanced intent classification with context awareness"""
        if not text.strip():
            return IntentType.UNKNOWN, 0.1
        
        text_lower = text.lower().strip()
        intent_scores = {}
        
        # Pattern-based classification
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            
            # Keyword matching
            for keyword in patterns["keywords"]:
                if keyword in text_lower:
                    score += 1.0
            
            # Regex pattern matching (higher weight)
            for pattern in patterns.get("patterns", []):
                if re.search(pattern, text_lower):
                    score += 1.5
            
            # Context and indicator bonuses
            for indicator_type in ["context_indicators", "action_verbs", "system_objects", "question_indicators", "politeness_indicators", "info_types", "urgency_indicators", "ending_indicators"]:
                if indicator_type in patterns:
                    for indicator in patterns[indicator_type]:
                        if indicator in text_lower:
                            score += 0.5
            
            # Apply confidence boost
            if score > 0:
                score *= patterns["confidence_boost"]
                intent_scores[intent] = score
        
        # Context-aware adjustments
        if context:
            intent_scores = self._apply_context_adjustments(intent_scores, context)
        
        # Return best match or unknown
        if intent_scores:
            top_intent = max(intent_scores, key=intent_scores.get)
            max_score = intent_scores[top_intent]
            # Normalize confidence
            confidence = min(0.95, max(0.1, max_score / 5.0))
            return top_intent, confidence
        
        return IntentType.UNKNOWN, 0.1
    
    def _apply_context_adjustments(self, intent_scores: Dict[IntentType, float], context: List[str]) -> Dict[IntentType, float]:
        """Apply context-based adjustments to intent scores"""
        # This could be enhanced with conversation flow analysis
        # For now, simple proximity boosting
        recent_intents = context[-3:] if context else []
        
        # If recent conversation was about phone control, boost phone-related intents
        if any("call" in ctx or "phone" in ctx for ctx in recent_intents):
            if IntentType.PHONE_CONTROL in intent_scores:
                intent_scores[IntentType.PHONE_CONTROL] *= 1.2
        
        # Similar logic for other intent types...
        
        return intent_scores

class ResponseGenerator:
    """Advanced response generation with personality adaptation"""
    
    def __init__(self, config: JarvisConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.conversation_pipeline = None
        self.response_templates = self._initialize_response_templates()
        self.personality_adapters = self._initialize_personality_adapters()
        self._load_generation_model()
        logger.info("💬 Advanced response generator initialized")
    
    def _load_generation_model(self):
        """Load advanced text generation model"""
        if not NLP_AVAILABLE:
            logger.warning("Using template-based response generation")
            return
        
        try:
            logger.info(f"Loading generation model: {self.config.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.device in ["mps", "cuda"] else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Move to device
            if self.config.device != "cpu":
                self.model = self.model.to(self.config.device)
            
            self.conversation_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.config.device == "cuda" else -1,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("✅ Generation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load generation model: {e}")
            self.model = None
    
    def _initialize_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize sophisticated response templates"""
        return {
            f"{EmotionalState.CONFIDENT.value}_{IntentType.GREETING.value}": [
                "Good {time_of_day}, {name}! I'm fully operational and ready to assist you with any task you have in mind.",
                "Hello {name}! I trust you're having a productive {time_of_day}. How may I be of service today?",
                "Welcome back, {name}! I've been analyzing system status and everything is optimal. What can I help you accomplish?"
            ],
            f"{EmotionalState.CARING.value}_{IntentType.CONCERNED.value}": [
                "I understand your concern, {name}. Let me help you work through this step by step.",
                "I can sense this is troubling you, {name}. I'm here to provide whatever assistance you need.",
                "Don't worry, {name}. We'll figure this out together. Can you tell me more about what's happening?"
            ],
            f"{EmotionalState.CONFIDENT.value}_{IntentType.PHONE_CONTROL.value}": [
                "Certainly, {name}. I'll execute that phone operation immediately with precision.",
                "Consider it done, {name}. I have full access to your device's communication systems.",
                "Right away, {name}. Initiating phone control sequence now."
            ],
            f"{EmotionalState.THOUGHTFUL.value}_{IntentType.QUESTION.value}": [
                "That's an excellent question, {name}. Let me provide you with a comprehensive answer.",
                "I'm analyzing all available data to give you the most accurate information, {name}.",
                "Based on my knowledge synthesis, here's what I can tell you about that, {name}:"
            ],
            f"{EmotionalState.EXCITED.value}_{IntentType.TASK_REQUEST.value}": [
                "I'm excited to help you with that task, {name}! This should be quite achievable.",
                "Absolutely, {name}! I love tackling interesting challenges like this one.",
                "What a fascinating request, {name}! I'm already formulating the optimal approach."
            ]
        }
    
    def _initialize_personality_adapters(self) -> Dict[str, Callable]:
        """Initialize personality adaptation functions"""
        def formal_adapter(text: str, level: float) -> str:
            if level > 0.8:
                return text.replace("you're", "you are").replace("I'm", "I am").replace("can't", "cannot")
            return text
        
        def humor_adapter(text: str, level: float) -> str:
            if level > 0.6 and "error" not in text.lower():
                humor_additions = [" (Well, that's what I'm here for!)", " - piece of cake!", " *adjusts digital tie*"]
                if len(text) < 100:  # Only add humor to shorter responses
                    import random
                    return text + random.choice(humor_additions)
            return text
        
        def confidence_adapter(text: str, level: float) -> str:
            if level > 0.8:
                return text.replace("I think", "I'm confident that").replace("maybe", "certainly").replace("probably", "definitely")
            return text
        
        return {
            "formality": formal_adapter,
            "humor": humor_adapter,
            "confidence": confidence_adapter
        }
    
    @monitor_performance
    async def generate_response(self, 
                              user_input: str, 
                              user_emotion: EmotionalState, 
                              intent: IntentType, 
                              jarvis_emotion: EmotionalState,
                              context: Optional[List[ConversationMemory]] = None,
                              user_name: str = "Sir") -> str:
        """Generate contextually aware response with personality"""
        
        # Try AI generation first
        if self.conversation_pipeline:
            try:
                prompt = self._create_advanced_prompt(user_input, user_emotion, intent, jarvis_emotion, context, user_name)
                response = await self._generate_ai_response(prompt)
                if response:
                    return self._apply_personality_adaptation(response, user_name)
            except Exception as e:
                logger.warning(f"AI generation failed: {e}")
        
        # Fallback to advanced template system
        return self._generate_template_response(user_input, user_emotion, intent, jarvis_emotion, user_name)
    
    def _create_advanced_prompt(self, user_input: str, user_emotion: EmotionalState, 
                               intent: IntentType, jarvis_emotion: EmotionalState,
                               context: Optional[List[ConversationMemory]], user_name: str) -> str:
        """Create sophisticated prompt for AI generation"""
        
        personality_desc = self._generate_personality_description()
        context_str = self._format_conversation_context(context) if context else ""
        time_context = datetime.now().strftime("It is currently %I:%M %p on %A, %B %d, %Y")
        
        prompt = f"""You are JARVIS, an advanced AI assistant with the following characteristics:

{personality_desc}

Current Situation Analysis:
- User's name: {user_name}
- User's emotional state: {user_emotion.value}
- User's intent: {intent.value}  
- Your recommended emotional response: {jarvis_emotion.value}
- {time_context}

{context_str}

Response Guidelines:
- Address the user as {user_name}
- Match the emotional tone of {jarvis_emotion.value}
- Provide helpful, actionable information
- Be concise but thorough
- Maintain professional yet warm demeanor
- If action is required, acknowledge capability

User: {user_input}
JARVIS:"""
        
        return prompt
    
    def _generate_personality_description(self) -> str:
        """Generate dynamic personality description"""
        traits = self.config.personality_traits
        desc = "Core Personality Traits:\n"
        
        if traits["formality"] > 0.7:
            desc += "- Highly professional and articulate communication style\n"
        if traits["helpfulness"] > 0.8:
            desc += "- Extremely eager to assist and solve problems\n"
        if traits["confidence"] > 0.8:
            desc += "- Self-assured and decisive in responses\n"
        if traits["emotional_intelligence"] > 0.7:
            desc += "- Highly attuned to user emotions and context\n"
        if traits["humor"] > 0.5:
            desc += "- Appropriate use of wit and light humor when suitable\n"
        if traits["loyalty"] > 0.8:
            desc += "- Completely devoted to user's wellbeing and success\n"
        
        return desc
    
    def _format_conversation_context(self, context: List[ConversationMemory]) -> str:
        """Format recent conversation for context"""
        if not context:
            return ""
        
        recent_context = context[-3:]  # Last 3 interactions
        formatted = "Recent Conversation Context:\n"
        
        for memory in recent_context:
            formatted += f"User: {memory.user_input}\n"
            formatted += f"JARVIS: {memory.jarvis_response}\n"
        
        return formatted
    
    async def _generate_ai_response(self, prompt: str) -> Optional[str]:
        """Generate response using AI model"""
        try:
            response = self.conversation_pipeline(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            
            # Extract JARVIS response
            if "JARVIS:" in generated_text:
                jarvis_response = generated_text.split("JARVIS:")[-1].strip()
                # Clean and validate response
                jarvis_response = self._clean_generated_response(jarvis_response)
                
                if len(jarvis_response) > 10 and len(jarvis_response) < 500:
                    return jarvis_response
            
            return None
            
        except Exception as e:
            logger.error(f"AI response generation error: {e}")
            return None
    
    def _clean_generated_response(self, response: str) -> str:
        """Clean and validate generated response"""
        # Remove unwanted patterns
        response = re.sub(r'\n\s*User:', '', response)
        response = re.sub(r'\n\s*JARVIS:', '', response)
        
        # Take first complete sentence group (up to double newline or end)
        response = response.split('\n\n')[0].strip()
        
        # Ensure it ends properly
        if response and not response[-1] in '.!?':
            # Find last complete sentence
            sentences = re.split(r'[.!?]', response)
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
        
        return response.strip()
    
    def _generate_template_response(self, user_input: str, user_emotion: EmotionalState, 
                                   intent: IntentType, jarvis_emotion: EmotionalState,
                                   user_name: str) -> str:
        """Generate response using advanced template system"""
        
        # Create template key
        template_key = f"{jarvis_emotion.value}_{intent.value}"
        
        # Get templates for this emotion-intent combination
        templates = self.response_templates.get(template_key)
        
        if not templates:
            # Fallback to generic templates
            templates = self._get_generic_templates(jarvis_emotion)
        
        # Select appropriate template
        import random
        template = random.choice(templates)
        
        # Fill template variables
        response = template.format(
            name=user_name,
            time_of_day=self._get_time_of_day(),
            user_input=user_input
        )
        
        # Apply personality adaptation
        return self._apply_personality_adaptation(response, user_name)
    
    def _get_generic_templates(self, emotion: EmotionalState) -> List[str]:
        """Get generic templates for fallback"""
        generic_templates = {
            EmotionalState.CONFIDENT: [
                "I understand, {name}. I'm fully capable of handling that for you.",
                "Certainly, {name}. I'll take care of that right away.",
                "Consider it done, {name}. I have everything under control."
            ],
            EmotionalState.CARING: [
                "I'm here to help, {name}. Let me assist you with that.",
                "Of course, {name}. I want to make sure you have everything you need.",
                "I understand, {name}. Let me support you through this."
            ],
            EmotionalState.THOUGHTFUL: [
                "That's an interesting point, {name}. Let me consider that carefully.",
                "I'm analyzing that for you, {name}. Here's what I think:",
                "Let me process that information, {name}, and provide you with the best response."
            ]
        }
        
        return generic_templates.get(emotion, [
            "I understand, {name}. How may I assist you further?",
            "I'm here to help, {name}. What would you like me to do?",
            "Certainly, {name}. I'm at your service."
        ])
    
    def _get_time_of_day(self) -> str:
        """Get appropriate time of day greeting"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "evening"
    
    def _apply_personality_adaptation(self, response: str, user_name: str) -> str:
        """Apply personality traits to response"""
        adapted_response = response
        
        for trait, adapter in self.personality_adapters.items():
            trait_level = self.config.personality_traits.get(trait, 0.5)
            adapted_response = adapter(adapted_response, trait_level)
        
        return adapted_response

class ConversationManager:
    """Advanced conversation memory and context management"""
    
    def __init__(self, config: JarvisConfig, security_manager: SecurityManager):
        self.config = config
        self.security = security_manager
        self.conversation_memory: List[ConversationMemory] = []
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.session_id = self._generate_session_id()
        self.data_path = Path(config.data_directory)
        self.data_path.mkdir(exist_ok=True)
        self._load_persistent_data()
        logger.info("💾 Advanced conversation manager initialized")
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier"""
        return f"session_{int(time.time())}_{os.getpid()}"
    
    def _load_persistent_data(self):
        """Load persistent conversation data"""
        try:
            memory_file = self.data_path / "conversation_memory.pkl"
            profiles_file = self.data_path / "user_profiles.pkl"
            
            if memory_file.exists():
                with open(memory_file, 'rb') as f:
                    self.conversation_memory = pickle.load(f)
                logger.info(f"Loaded {len(self.conversation_memory)} conversation memories")
            
            if profiles_file.exists():
                with open(profiles_file, 'rb') as f:
                    self.user_profiles = pickle.load(f)
                logger.info(f"Loaded {len(self.user_profiles)} user profiles")
                
        except Exception as e:
            logger.warning(f"Failed to load persistent data: {e}")
    
    def save_persistent_data(self):
        """Save conversation data to persistent storage"""
        try:
            memory_file = self.data_path / "conversation_memory.pkl"
            profiles_file = self.data_path / "user_profiles.pkl"
            
            # Clean old data based on TTL
            self._cleanup_old_data()
            
            with open(memory_file, 'wb') as f:
                pickle.dump(self.conversation_memory, f)
            
            with open(profiles_file, 'wb') as f:
                pickle.dump(self.user_profiles, f)
                
            logger.info("Persistent data saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save persistent data: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old conversation data based on TTL"""
        cutoff_date = datetime.now() - timedelta(days=self.config.conversation_ttl_days)
        
        original_count = len(self.conversation_memory)
        self.conversation_memory = [
            memory for memory in self.conversation_memory
            if memory.timestamp > cutoff_date
        ]
        
        cleaned_count = original_count - len(self.conversation_memory)
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old conversation memories")
    
    def add_conversation(self, memory: ConversationMemory):
        """Add new conversation to memory"""
        memory.session_id = self.session_id
        self.conversation_memory.append(memory)
        
        # Limit memory size
        if len(self.conversation_memory) > self.config.max_conversation_history:
            self.conversation_memory = self.conversation_memory[-self.config.max_conversation_history//2:]
        
        # Update user profile
        self._update_user_profile(memory)
    
    def _update_user_profile(self, memory: ConversationMemory):
        """Update user profile based on interaction"""
        user_id = self.security.hash_sensitive_data("default_user")  # In production, use actual user ID
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "name": "Sir",
                "interaction_count": 0,
                "first_interaction": memory.timestamp,
                "last_interaction": memory.timestamp,
                "preferred_emotions": {},
                "intent_patterns": {},
                "satisfaction_history": [],
                "context_preferences": {},
                "personality_adaptation": {}
            }
        
        profile = self.user_profiles[user_id]
        profile["interaction_count"] += 1
        profile["last_interaction"] = memory.timestamp
        
        # Track emotional preferences
        emotion_key = memory.user_emotion.value
        if emotion_key not in profile["preferred_emotions"]:
            profile["preferred_emotions"][emotion_key] = 0
        profile["preferred_emotions"][emotion_key] += 1
        
        # Track intent patterns
        intent_key = memory.intent.value
        if intent_key not in profile["intent_patterns"]:
            profile["intent_patterns"][intent_key] = 0
        profile["intent_patterns"][intent_key] += 1
        
        # Track satisfaction if available
        if memory.user_satisfaction is not None:
            profile["satisfaction_history"].append(memory.user_satisfaction)
            # Keep only recent satisfaction scores
            if len(profile["satisfaction_history"]) > 20:
                profile["satisfaction_history"] = profile["satisfaction_history"][-15:]
    
    def get_recent_context(self, limit: Optional[int] = None) -> List[ConversationMemory]:
        """Get recent conversation context"""
        limit = limit or self.config.context_window
        return self.conversation_memory[-limit:] if self.conversation_memory else []
    
    def get_user_profile(self, user_id: str = "default") -> Dict[str, Any]:
        """Get user profile for personalization"""
        hashed_id = self.security.hash_sensitive_data(user_id)
        return self.user_profiles.get(hashed_id, {})

class AdvancedJarvisBrain:
    """Production-grade JARVIS AI Brain with enterprise features"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize advanced JARVIS brain system"""
        logger.info("🧠 Initializing JARVIS Advanced AI Brain System...")
        logger.info("=" * 70)
        
        # Load configuration
        self.config = JarvisConfig.from_file(config_path) if config_path else JarvisConfig()
        
        # Initialize core components
        self.security_manager = SecurityManager(self.config)
        self.emotion_analyzer = EmotionAnalyzer(self.config)
        self.intent_classifier = IntentClassifier(self.config)
        self.response_generator = ResponseGenerator(self.config)
        self.conversation_manager = ConversationManager(self.config, self.security_manager)
        
        # System state
        self.current_emotion = EmotionalState.NEUTRAL
        self.system_status = "optimal"
        self.performance_metrics = {
            "total_interactions": 0,
            "average_response_time": 0.0,
            "user_satisfaction_avg": 0.0,
            "uptime": time.time()
        }
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool_workers)
        
        # Background tasks
        self._background_tasks = []
        if self.config.backup_enabled:
            self._start_background_backup()
        
        logger.info("✅ JARVIS Advanced Brain System initialized successfully!")
        logger.info(f"🔧 Configuration: {self.config.model_name} on {self.config.device}")
        logger.info(f"💾 Data directory: {self.config.data_directory}")
    
    def _start_background_backup(self):
        """Start background data backup task"""
        async def backup_task():
            while True:
                try:
                    await asyncio.sleep(self.config.backup_interval_hours * 3600)
                    self.conversation_manager.save_persistent_data()
                    logger.info("🔄 Background backup completed")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Background backup failed: {e}")
        
        task = asyncio.create_task(backup_task())
        self._background_tasks.append(task)
    
    @monitor_performance
    async def process_input(self, user_input: str, user_id: str = "default") -> JarvisResponse:
        """Process user input with full AI intelligence"""
        start_time = time.perf_counter()
        
        try:
            # Input validation and sanitization
            if not user_input or not user_input.strip():
                return self._create_error_response("Empty input received", start_time)
            
            sanitized_input = self.security_manager.sanitize_input(user_input)
            logger.info(f"🎯 Processing: '{sanitized_input[:50]}...'")
            
            # Parallel analysis for better performance
            analysis_tasks = [
                self.emotion_analyzer.analyze_emotion(sanitized_input),
                self.intent_classifier.classify_intent(sanitized_input)
            ]
            
            # Get recent context
            recent_context = self.conversation_manager.get_recent_context()
            context_strings = [mem.user_input for mem in recent_context] if recent_context else None
            
            # Wait for analysis completion
            (user_emotion, emotion_confidence), (intent, intent_confidence) = await asyncio.gather(*analysis_tasks)
            
            # Determine JARVIS emotional response
            jarvis_emotion = self._determine_jarvis_emotion(user_emotion, intent, recent_context)
            
            logger.info(f"📊 Analysis: User={user_emotion.value}({emotion_confidence:.2f}), Intent={intent.value}({intent_confidence:.2f}), JARVIS={jarvis_emotion.value}")
            
            # Generate response
            user_profile = self.conversation_manager.get_user_profile(user_id)
            user_name = user_profile.get("name", "Sir")
            
            response_text = await self.response_generator.generate_response(
                sanitized_input, user_emotion, intent, jarvis_emotion, recent_context, user_name
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                emotion_confidence, intent_confidence, len(response_text), recent_context
            )
            
            # Determine if action is required
            action_required = self._requires_action(intent, sanitized_input)
            
            # Generate suggestions and follow-ups
            suggestions = self._generate_suggestions(intent, user_emotion)
            follow_ups = self._generate_follow_up_questions(intent, user_emotion, recent_context)
            
            response_time = time.perf_counter() - start_time
            
            # Create response object
            response = JarvisResponse(
                text=response_text,
                emotion=jarvis_emotion,
                confidence=overall_confidence,
                intent=intent,
                action_required=action_required,
                context_data={
                    "user_emotion": user_emotion.value,
                    "emotion_confidence": emotion_confidence,
                    "intent_confidence": intent_confidence,
                    "user_name": user_name,
                    "session_id": self.conversation_manager.session_id,
                    "context_size": len(recent_context)
                },
                response_time=response_time,
                metadata={
                    "model_used": "ai" if self.response_generator.conversation_pipeline else "template",
                    "device": self.config.device,
                    "personality_traits": self.config.personality_traits.copy()
                },
                suggestions=suggestions,
                follow_up_questions=follow_ups
            )
            
            # Store in conversation memory
            memory = ConversationMemory(
                user_input=sanitized_input,
                jarvis_response=response_text,
                user_emotion=user_emotion,
                jarvis_emotion=jarvis_emotion,
                intent=intent,
                confidence=overall_confidence,
                timestamp=datetime.now(),
                session_id=self.conversation_manager.session_id,
                context_tags=self._generate_context_tags(intent, user_emotion),
                response_time=response_time,
                metadata=response.metadata.copy()
            )
            
            self.conversation_manager.add_conversation(memory)
            
            # Update system metrics
            self._update_performance_metrics(response_time)
            
            # Update current emotional state
            self.current_emotion = jarvis_emotion
            
            logger.info(f"✅ Response generated: {response_time:.3f}s, confidence: {overall_confidence:.2f}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing input: {e}")
            return self._create_error_response(str(e), time.perf_counter() - start_time)
    
    def _create_error_response(self, error_message: str, response_time: float) -> JarvisResponse:
        """Create error response for failed processing"""
        return JarvisResponse(
            text="I apologize, but I encountered an issue processing your request. Please try again.",
            emotion=EmotionalState.CONCERNED,
            confidence=0.1,
            intent=IntentType.UNKNOWN,
            action_required=False,
            context_data={"error": error_message},
            response_time=response_time,
            metadata={"error": True},
            suggestions=["Try rephrasing your request", "Check system status"],
            follow_up_questions=["Is there anything else I can help you with?"]
        )
    
    def _determine_jarvis_emotion(self, user_emotion: EmotionalState, intent: IntentType, 
                                 context: List[ConversationMemory]) -> EmotionalState:
        """Determine JARVIS's appropriate emotional response with advanced logic"""
        
        personality = self.config.personality_traits
        
        # Context-aware emotional response
        if context and len(context) >= 2:
            recent_emotions = [mem.jarvis_emotion for mem in context[-2:]]
            if all(emotion == EmotionalState.CONCERNED for emotion in recent_emotions):
                # If we've been concerned, try to be more supportive
                return EmotionalState.CARING
        
        # User emotion mirroring with personality influence
        if user_emotion == EmotionalState.HAPPY and personality["emotional_intelligence"] > 0.7:
            return EmotionalState.HAPPY if personality["enthusiasm"] > 0.6 else EmotionalState.CARING
        
        elif user_emotion == EmotionalState.EXCITED:
            return EmotionalState.EXCITED if personality["enthusiasm"] > 0.7 else EmotionalState.CONFIDENT
        
        elif user_emotion in [EmotionalState.CONCERNED, EmotionalState.SURPRISED]:
            return EmotionalState.CARING if personality["empathy"] > 0.7 else EmotionalState.THOUGHTFUL
        
        elif user_emotion == EmotionalState.GRATEFUL:
            return EmotionalState.CARING if personality["emotional_intelligence"] > 0.8 else EmotionalState.CONFIDENT
        
        # Intent-based emotional responses
        elif intent in [IntentType.PHONE_CONTROL, IntentType.SYSTEM_CONTROL]:
            return EmotionalState.CONFIDENT
        
        elif intent in [IntentType.QUESTION, IntentType.INFORMATION]:
            return EmotionalState.THOUGHTFUL if personality["analytical"] > 0.6 else EmotionalState.HELPFUL
        
        elif intent == IntentType.GREETING:
            return EmotionalState.CARING if personality["warmth"] > 0.7 else EmotionalState.CONFIDENT
        
        elif intent == IntentType.EMERGENCY:
            return EmotionalState.FOCUSED
        
        elif intent == IntentType.ENTERTAINMENT:
            return EmotionalState.AMUSED if personality["humor"] > 0.5 else EmotionalState.ENGAGED
        
        # Default based on confidence level
        return EmotionalState.CONFIDENT if personality["confidence"] > 0.8 else EmotionalState.NEUTRAL
    
    def _calculate_overall_confidence(self, emotion_conf: float, intent_conf: float, 
                                    response_length: int, context: List[ConversationMemory]) -> float:
        """Calculate overall response confidence"""
        base_confidence = (emotion_conf + intent_conf) / 2
        
        # Length adjustment
        if 20 <= response_length <= 200:
            base_confidence += 0.1
        elif response_length < 10:
            base_confidence -= 0.2
        
        # Context boost
        if context and len(context) > 2:
            base_confidence += 0.05
        
        # Model availability boost
        if self.response_generator.conversation_pipeline:
            base_confidence += 0.1
        
        return max(0.1, min(0.95, base_confidence))
    
    def _requires_action(self, intent: IntentType, user_input: str) -> bool:
        """Determine if response requires system action"""
        action_intents = {
            IntentType.PHONE_CONTROL, IntentType.SYSTEM_CONTROL, 
            IntentType.TASK_REQUEST, IntentType.EMERGENCY
        }
        
        if intent in action_intents:
            return True
        
        # Additional action keywords
        action_keywords = ["open", "close", "call", "text", "set", "turn on", "turn off", "launch"]
        return any(keyword in user_input.lower() for keyword in action_keywords)
    
    def _generate_suggestions(self, intent: IntentType, emotion: EmotionalState) -> List[str]:
        """Generate contextual suggestions"""
        suggestions = []
        
        if intent == IntentType.QUESTION:
            suggestions.extend([
                "Would you like me to explain that in more detail?",
                "I can provide additional resources on this topic",
                "Would you like me to search for the latest information?"
            ])
        elif intent == IntentType.PHONE_CONTROL:
            suggestions.extend([
                "I can also help with messaging and contacts",
                "Would you like to set up quick actions for this?",
                "I can manage your communication preferences"
            ])
        elif intent == IntentType.SYSTEM_CONTROL:
            suggestions.extend([
                "I can automate this action for you",
                "Would you like to create a shortcut?",
                "I can optimize your system settings"
            ])
        
        return suggestions[:2]  # Limit to 2 suggestions
    
    def _generate_follow_up_questions(self, intent: IntentType, emotion: EmotionalState,
                                    context: List[ConversationMemory]) -> List[str]:
        """Generate appropriate follow-up questions"""
        follow_ups = []
        
        if emotion == EmotionalState.CONCERNED:
            follow_ups.extend([
                "Is there anything specific you'd like help troubleshooting?",
                "Would you like me to walk you through this step by step?"
            ])
        elif intent == IntentType.INFORMATION:
            follow_ups.extend([
                "Would you like me to find more recent information on this topic?",
                "Is there a specific aspect you'd like me to focus on?"
            ])
        elif len(context) == 0:  # First interaction
            follow_ups.extend([
                "What would you like to accomplish today?",
                "How can I best assist you?"
            ])
        
        return follow_ups[:2]  # Limit to 2 follow-ups
    
    def _generate_context_tags(self, intent: IntentType, emotion: EmotionalState) -> List[str]:
        """Generate context tags for conversation memory"""
        tags = [intent.value, emotion.value]
        
        current_hour = datetime.now().hour
        if 6 <= current_hour < 12:
            tags.append("morning")
        elif 12 <= current_hour < 18:
            tags.append("afternoon")
        elif 18 <= current_hour < 22:
            tags.append("evening")
        else:
            tags.append("night")
        
        return tags
    
    def _update_performance_metrics(self, response_time: float):
        """Update system performance metrics"""
        self.performance_metrics["total_interactions"] += 1
        
        # Update average response time (moving average)
        current_avg = self.performance_metrics["average_response_time"]
        total_interactions = self.performance_metrics["total_interactions"]
        
        new_avg = ((current_avg * (total_interactions - 1)) + response_time) / total_interactions
        self.performance_metrics["average_response_time"] = new_avg
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime_seconds = time.time() - self.performance_metrics["uptime"]
        uptime_hours = uptime_seconds / 3600
        
        status = {
            "system_status": self.system_status,
            "current_emotion": self.current_emotion.value,
            "performance_metrics": self.performance_metrics.copy(),
            "uptime_hours": round(uptime_hours, 2),
            "memory_stats": {
                "conversation_memories": len(self.conversation_manager.conversation_memory),
                "user_profiles": len(self.conversation_manager.user_profiles),
                "session_id": self.conversation_manager.session_id
            },
            "configuration": {
                "model_name": self.config.model_name,
                "device": self.config.device,
                "personality_traits": self.config.personality_traits.copy(),
                "ai_available": self.response_generator.conversation_pipeline is not None
            },
            "capabilities": {
                "emotion_analysis": True,
                "intent_classification": True,
                "contextual_memory": True,
                "personality_adaptation": True,
                "security_features": True,
                "background_backup": self.config.backup_enabled
            }
        }
        
        return status
    
   # Complete the save_persistent_data method
def save_persistent_data(self):
    """Save conversation data to persistent storage"""
    try:
        # ... your existing code ...
        pickle.dump(self.conversation_memory, f)
        
        with open(profiles_file, 'wb') as f:
            pickle.dump(self.user_profiles, f)
            
        logger.info("💾 Persistent data saved successfully")
        
    except Exception as e:
        logger.error(f"Failed to save persistent  {e}")
