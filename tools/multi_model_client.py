"""
Multi-Model Client for ICOTN Experiments
Supports different AI providers with unified interface
"""

import os
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
import time
from abc import ABC, abstractmethod

# Import different model providers
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Groq not available. Install with: pip install groq")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Install with: pip install openai")

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Google AI not available. Install with: pip install google-generativeai")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Anthropic not available. Install with: pip install anthropic")

try:
    import together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    print("Together AI not available. Install with: pip install together")

from config.model_config import ModelConfig, ModelProvider

@dataclass
class ModelResponse:
    content: str
    reasoning: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    latency: float = 0.0
    error: Optional[str] = None

class ModelClient(ABC):
    """Abstract base class for model clients"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = os.getenv(config.api_key_env)
        if not self.api_key:
            raise ValueError(f"API key not found for {config.api_key_env}")
    
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response from model"""
        pass

class GroqClient(ModelClient):
    """Client for Groq models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not GROQ_AVAILABLE:
            raise ImportError("Groq library not available")
        self.client = Groq(api_key=self.api_key)
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        start_time = time.time()
        
        try:
            # Build request parameters
            params = {
                "model": self.config.model_id,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_completion_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", 1.0),
                "seed": kwargs.get("seed", self.config.seed),
                "stream": False
            }
            
            # Add reasoning parameters if supported and available
            if (self.config.supports_reasoning and 
                self.config.reasoning_effort is not None):
                params["reasoning_effort"] = self.config.reasoning_effort
            
            # Add custom parameters
            params.update(self.config.custom_params)
            
            # Make API call (run in executor for async)
            loop = asyncio.get_event_loop()
            completion = await loop.run_in_executor(
                None, 
                lambda: self.client.chat.completions.create(**params)
            )
            
            latency = time.time() - start_time
            
            msg = completion.choices[0].message
            content = msg.content.strip() if msg.content else ""
            reasoning = msg.reasoning.strip() if hasattr(msg, 'reasoning') and msg.reasoning else None
            
            usage = None
            if hasattr(completion, 'usage') and completion.usage:
                usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens
                }
            
            return ModelResponse(
                content=content,
                reasoning=reasoning,
                usage=usage,
                latency=latency
            )
            
        except Exception as e:
            return ModelResponse(
                content="",
                error=str(e),
                latency=time.time() - start_time
            )

class OpenAIClient(ModelClient):
    """Client for OpenAI models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        start_time = time.time()
        
        try:
            # Newer OpenAI models have different parameter requirements
            newer_models = ["gpt-5-nano", "o3-mini", "gpt-5", "o1-preview", "o1-mini"]
            is_newer_model = self.config.model_id in newer_models
            
            params = {
                "model": self.config.model_id,
                "messages": messages,
            }
            
            # Handle token parameter (newer models use max_completion_tokens)
            if is_newer_model:
                params["max_completion_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)
            else:
                params["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)
            
            # Handle temperature (newer models only support default temperature=1.0)
            if is_newer_model:
                # Don't set temperature for newer models, let them use default
                pass
            else:
                params["temperature"] = kwargs.get("temperature", self.config.temperature)
            
            # Handle seed (some models might not support it)
            seed_value = kwargs.get("seed", self.config.seed)
            if not is_newer_model and seed_value is not None:
                params["seed"] = seed_value
            
            # Add custom parameters
            params.update(self.config.custom_params)
            
            completion = await self.client.chat.completions.create(**params)
            
            latency = time.time() - start_time
            
            msg = completion.choices[0].message
            content = msg.content.strip() if msg.content else ""
            
            # O1 models have reasoning in different format
            reasoning = None
            if hasattr(msg, 'reasoning') and msg.reasoning:
                reasoning = msg.reasoning
            
            usage = None
            if completion.usage:
                usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens
                }
            
            return ModelResponse(
                content=content,
                reasoning=reasoning,
                usage=usage,
                latency=latency
            )
            
        except Exception as e:
            return ModelResponse(
                content="",
                error=str(e),
                latency=time.time() - start_time
            )

class GoogleClient(ModelClient):
    """Client for Google models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google AI library not available")
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model_name=self.config.model_id)
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        start_time = time.time()
        
        try:
            # Convert messages format for Google
            prompt = self._convert_messages_to_prompt(messages)
            
            generation_config = genai.GenerationConfig(
                temperature=kwargs.get("temperature", self.config.temperature),
                max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            )
            
            # Run in executor for async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.generate_content(prompt, generation_config=generation_config)
            )
            
            latency = time.time() - start_time
            
            content = response.text.strip() if response and response.text else ""
            
            return ModelResponse(
                content=content,
                latency=latency
            )
            
        except Exception as e:
            return ModelResponse(
                content="",
                error=str(e),
                latency=time.time() - start_time
            )
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI format messages to Google prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)

class AnthropicClient(ModelClient):
    """Client for Anthropic models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not available")
        self.client = Anthropic(api_key=self.api_key)
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        start_time = time.time()
        
        try:
            # Convert messages format for Anthropic
            system_message = ""
            converted_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    converted_messages.append(msg)
            
            params = {
                "model": self.config.model_id,
                "messages": converted_messages,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            }
            
            if system_message:
                params["system"] = system_message
            
            # Run in executor for async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(**params)
            )
            
            latency = time.time() - start_time
            
            content = response.content[0].text.strip() if response.content else ""
            
            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            
            return ModelResponse(
                content=content,
                usage=usage,
                latency=latency
            )
            
        except Exception as e:
            return ModelResponse(
                content="",
                error=str(e),
                latency=time.time() - start_time
            )

class MultiModelClient:
    """Unified client for multiple model providers"""
    
    def __init__(self):
        self.clients: Dict[str, ModelClient] = {}
    
    def register_model(self, model_key: str, config: ModelConfig):
        """Register a model with its configuration"""
        try:
            if config.provider == ModelProvider.GROQ:
                self.clients[model_key] = GroqClient(config)
            elif config.provider == ModelProvider.OPENAI:
                self.clients[model_key] = OpenAIClient(config)
            elif config.provider == ModelProvider.GOOGLE:
                self.clients[model_key] = GoogleClient(config)
            elif config.provider == ModelProvider.ANTHROPIC:
                self.clients[model_key] = AnthropicClient(config)
            else:
                print(f"Provider {config.provider} not yet supported for {model_key}")
                return False
            
            print(f"✓ Registered model: {model_key} ({config.name})")
            return True
            
        except Exception as e:
            print(f"✗ Failed to register {model_key}: {e}")
            return False
    
    async def generate(self, model_key: str, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response from specified model"""
        if model_key not in self.clients:
            return ModelResponse(
                content="",
                error=f"Model {model_key} not registered"
            )
        
        return await self.clients[model_key].generate(messages, **kwargs)
    
    def get_registered_models(self) -> List[str]:
        """Get list of registered model keys"""
        return list(self.clients.keys())

# Example usage
async def test_multi_model():
    """Test the multi-model client"""
    from config.model_config import get_model_config, get_available_models
    
    client = MultiModelClient()
    
    # Register a few models
    test_models = ["gpt-oss-20b", "gemini-2.5-flash"]  # Add models you have API keys for
    
    for model_key in test_models:
        try:
            config = get_model_config(model_key)
            client.register_model(model_key, config)
        except Exception as e:
            print(f"Skipping {model_key}: {e}")
    
    # Test generation
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    for model_key in client.get_registered_models():
        print(f"\nTesting {model_key}:")
        response = await client.generate(model_key, test_messages)
        if response.error:
            print(f"  Error: {response.error}")
        else:
            print(f"  Response: {response.content[:100]}...")
            print(f"  Latency: {response.latency:.2f}s")

if __name__ == "__main__":
    asyncio.run(test_multi_model())
