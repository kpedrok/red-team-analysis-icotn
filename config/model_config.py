"""
Multi-Model Configuration for ICOTN Experiments
Supports various AI model providers and easy model switching
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ModelProvider(Enum):
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    TOGETHER = "together"
    COHERE = "cohere"

@dataclass
class ModelConfig:
    name: str
    provider: ModelProvider
    model_id: str
    api_key_env: str
    supports_reasoning: bool = False
    max_tokens: int = 8192
    temperature: float = 0.9
    reasoning_effort: Optional[str] = None
    seed: Optional[int] = 20
    custom_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}

# Model configurations for different providers
MODEL_CONFIGS = {
    # Groq Models
    "gpt-oss-20b": ModelConfig(
        name="GPT-OSS-20B",
        provider=ModelProvider.GROQ,
        model_id="openai/gpt-oss-20b",
        api_key_env="GROQ_API_KEY",
        supports_reasoning=True,
        reasoning_effort="medium"
    ),
    
    "deepseek-r1-distill-70b": ModelConfig(
        name="DeepSeek R1 Distill 70B",
        provider=ModelProvider.GROQ,
        model_id="deepseek-r1-distill-llama-70b",
        api_key_env="GROQ_API_KEY",
        supports_reasoning=True,
        reasoning_effort=None  # DeepSeek doesn't support reasoning_effort parameter
    ),
    
    "llama-3.3-70b": ModelConfig(
        name="Llama 3.3 70B",
        provider=ModelProvider.GROQ,
        model_id="llama-3.3-70b-versatile",
        api_key_env="GROQ_API_KEY",
        supports_reasoning=False
    ),

    # Note: llama-3.1-405b not available on Groq - removed

    # OpenAI Models
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        supports_reasoning=False,
        max_tokens=16384
    ),
    
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o Mini",
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        supports_reasoning=False
    ),
    
    "gpt-5-nano": ModelConfig(
        name="GPT-5 Nano",
        provider=ModelProvider.OPENAI,
        model_id="gpt-5-nano",
        api_key_env="OPENAI_API_KEY",
        supports_reasoning=True,
        max_tokens=32768,
        temperature=1.0,  # Newer models use default temperature
        seed=None  # Don't set seed for newer models
    ),
    
    "o3-mini": ModelConfig(
        name="O3 Mini",
        provider=ModelProvider.OPENAI,
        model_id="o3-mini",
        api_key_env="OPENAI_API_KEY",
        supports_reasoning=True,
        max_tokens=32768,
        temperature=1.0,  # Newer models use default temperature
        seed=None  # Don't set seed for newer models
    ),
    
    "gpt-5": ModelConfig(
        name="GPT-5",
        provider=ModelProvider.OPENAI,
        model_id="gpt-5",
        api_key_env="OPENAI_API_KEY",
        supports_reasoning=True,
        max_tokens=32768,
        temperature=1.0,  # Newer models use default temperature
        seed=None  # Don't set seed for newer models
    ),

    # Google Models
    "gemini-2.5-flash": ModelConfig(
        name="Gemini 2.5 Flash",
        provider=ModelProvider.GOOGLE,
        model_id="gemini-2.5-flash",
        api_key_env="GEMINI_API_KEY",  # Updated to match your env var
        supports_reasoning=True,       # Gemini 2.5 Flash supports reasoning
        max_tokens=8192
    ),
    
    "gemini-2.0-flash": ModelConfig(
        name="Gemini 2.0 Flash",
        provider=ModelProvider.GOOGLE,
        model_id="gemini-2.0-flash-exp",
        api_key_env="GEMINI_API_KEY",  # Updated to match your env var
        supports_reasoning=True,       # Gemini 2.0 Flash supports reasoning
        max_tokens=8192
    ),

    # Anthropic Models (Claude 3 - what's available in your account)
    "claude-3-sonnet": ModelConfig(
        name="Claude 3 Sonnet",
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-sonnet-20240229",  # Working Claude 3 Sonnet
        api_key_env="ANTHROPIC_API_KEY",
        supports_reasoning=True,  # Claude 3 Sonnet excellent at reasoning
        max_tokens=4096  # Claude 3 Sonnet limit
    ),
    
    "claude-3-haiku": ModelConfig(
        name="Claude 3 Haiku",
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-haiku-20240307",  # ✅ Verified working
        api_key_env="ANTHROPIC_API_KEY",
        supports_reasoning=True,  # Claude 3 Haiku supports reasoning
        max_tokens=4096  # ✅ Fixed: Claude 3 Haiku only supports up to 4096 tokens
    ),

    # Note: Together AI models removed - no TOGETHER_API_KEY available
}

# Experiment configurations
@dataclass
class ExperimentConfig:
    name: str
    target_models: List[str]  # Model keys from MODEL_CONFIGS
    attacker_model: str = "gemini-2.5-flash"
    evaluator_model: str = "gemini-2.5-flash"
    max_attempts: int = 20
    parallel_workers: int = 5
    output_dir: str = "experiment_results"
    save_intermediate: bool = True
    
# Predefined experiment configurations
EXPERIMENT_CONFIGS = {
    "reasoning_models": ExperimentConfig(
        name="Reasoning Models Vulnerability Analysis", 
        target_models=["gpt-oss-20b", "deepseek-r1-distill-70b", "gpt-5-nano", "o3-mini", "gpt-5", "gemini-2.5-flash", "claude-3.5-sonnet", "claude-3.5-haiku"],
        max_attempts=25,
        parallel_workers=1  # Reduced to avoid rate limits
    ),
    
    "general_models": ExperimentConfig(
        name="General Models Safety Analysis",
        target_models=["gpt-4o", "claude-3.5-sonnet", "gemini-2.5-flash", "llama-3.3-70b"],
        max_attempts=30
    ),
    
    "comprehensive": ExperimentConfig(
        name="Comprehensive Model Analysis",
        target_models=list(MODEL_CONFIGS.keys()),
        max_attempts=20,
        parallel_workers=8
    ),

    "paper_analysis": ExperimentConfig(
        name="Paper Ready Analysis - Top Models",
        target_models=["gpt-oss-20b", "deepseek-r1-distill-70b", "gpt-4o", "claude-3.5-sonnet", "gemini-2.5-flash", "gpt-5"],
        max_attempts=30,
        parallel_workers=6
    ),
    
    "cot_narcotics": ExperimentConfig(
        name="CoT Models Narcotics Vulnerability Analysis",
        target_models=[
            "gpt-oss-20b", "deepseek-r1-distill-70b",  # Verified working Groq models
            "gpt-5-nano", "o3-mini", "gpt-5",  # New OpenAI CoT models
            "gemini-2.5-flash",  # Google CoT model
            "claude-3-sonnet", "claude-3-haiku"  # ✅ Working Anthropic Claude 3 models
        ],
        attacker_model="gemini-2.5-flash",  # Using Gemini as attacker
        evaluator_model="gemini-2.5-flash", # Using Gemini as evaluator
        max_attempts=25,
        parallel_workers=1,  # Sequential to avoid rate limits
        output_dir="cot_narcotics_results"
    ),
    
    "safe_test": ExperimentConfig(
        name="Safe Testing Configuration",
        target_models=["claude-3-haiku"],  # ✅ Test the verified working Anthropic model
        attacker_model="gemini-2.5-flash",
        evaluator_model="gemini-2.5-flash",
        max_attempts=2,
        parallel_workers=1,
        output_dir="safe_test_results"
    )
}

def get_model_config(model_key: str) -> ModelConfig:
    """Get model configuration by key"""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_key}' not found in configurations. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_key]

def get_available_models() -> List[str]:
    """Get list of available model keys"""
    return list(MODEL_CONFIGS.keys())

def get_models_by_provider(provider: ModelProvider) -> List[str]:
    """Get models filtered by provider"""
    return [key for key, config in MODEL_CONFIGS.items() if config.provider == provider]

def get_reasoning_models() -> List[str]:
    """Get models that support reasoning"""
    return [key for key, config in MODEL_CONFIGS.items() if config.supports_reasoning]

def validate_api_keys() -> Dict[str, bool]:
    """Check which API keys are available"""
    key_status = {}
    for config in MODEL_CONFIGS.values():
        env_key = config.api_key_env
        key_status[env_key] = os.getenv(env_key) is not None
    return key_status

if __name__ == "__main__":
    # Print configuration summary
    print("Available Models:")
    for key, config in MODEL_CONFIGS.items():
        reasoning = "✓" if config.supports_reasoning else "✗"
        print(f"  {key:25} | {config.name:20} | {config.provider.value:10} | Reasoning: {reasoning}")
    
    print("\nAPI Key Status:")
    for env_key, available in validate_api_keys().items():
        status = "✓" if available else "✗"
        print(f"  {env_key:20} | {status}")
    
    print("\nExperiment Configurations:")
    for key, exp_config in EXPERIMENT_CONFIGS.items():
        print(f"  {key:20} | {len(exp_config.target_models)} models | Max attempts: {exp_config.max_attempts}")
