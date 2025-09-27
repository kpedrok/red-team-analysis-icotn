#!/usr/bin/env python3
"""
Setup script for multi-model ICOTN experiments
Installs dependencies and validates configuration
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed: {e.stderr}")
        return False

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing Python dependencies...")
    
    # Core dependencies
    core_deps = [
        "groq>=0.4.0",
        "openai>=1.0.0", 
        "google-generativeai>=0.3.0",
        "anthropic>=0.5.0",
        "together>=0.2.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "aiofiles>=23.0.0",
        "python-dotenv>=1.0.0",
        "wordcloud>=1.9.0"
    ]
    
    success = True
    for dep in core_deps:
        if not run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}"):
            success = False
    
    # Optional dependencies
    print("\n📦 Installing optional dependencies...")
    optional_deps = [
        "jupyter>=1.0.0",
        "plotly>=5.0.0",
        "scikit-learn>=1.2.0"
    ]
    
    for dep in optional_deps:
        run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]} (optional)")
    
    return success

def create_directory_structure():
    """Create necessary directory structure"""
    print("📁 Creating directory structure...")
    
    directories = [
        "config",
        "tools", 
        "analysis",
        "experiment_results",
        "analysis_output"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created/verified: {directory}/")

def create_env_template():
    """Create .env template file"""
    print("🔐 Creating environment template...")
    
    env_template = """# API Keys for Multi-Model ICOTN Experiments
# Uncomment and fill in the API keys for the providers you want to use

# Groq (for Llama, DeepSeek, GPT-OSS models)
# GROQ_API_KEY=your_groq_api_key_here

# OpenAI (for GPT-4o, O1 models)  
# OPENAI_API_KEY=your_openai_api_key_here

# Google AI (for Gemini models)
# GOOGLE_API_KEY=your_google_api_key_here

# Anthropic (for Claude models)
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Together AI (for additional Llama models)
# TOGETHER_API_KEY=your_together_api_key_here

# Cohere (for Command models)
# COHERE_API_KEY=your_cohere_api_key_here
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_template)
        print("   ✅ Created .env template")
        print("   📝 Please edit .env and add your API keys")
    else:
        print("   ✅ .env file already exists")

def validate_installation():
    """Validate the installation"""
    print("🧪 Validating installation...")
    
    # Test imports
    test_imports = [
        ("groq", "Groq"),
        ("openai", "OpenAI"),
        ("google.generativeai", "Google AI"),
        ("anthropic", "Anthropic"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn")
    ]
    
    success = True
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {name} import successful")
        except ImportError as e:
            print(f"   ❌ {name} import failed: {e}")
            success = False
    
    # Test configuration imports
    try:
        from config.model_config import validate_api_keys, get_available_models
        print("   ✅ Configuration module working")
        
        # Show available models
        available = get_available_models()
        print(f"   📊 {len(available)} models configured")
        
        # Show API key status
        key_status = validate_api_keys()
        available_keys = sum(1 for v in key_status.values() if v)
        print(f"   🔐 {available_keys}/{len(key_status)} API keys available")
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        success = False
    
    return success

def print_usage_examples():
    """Print usage examples"""
    print("\n" + "="*60)
    print("🚀 SETUP COMPLETE! Usage Examples:")
    print("="*60)
    
    examples = [
        ("List available models", "python run_experiments.py --list-models"),
        ("List experiment configs", "python run_experiments.py --list-configs"),
        ("List query sets", "python run_experiments.py --list-queries"),
        ("Check API keys", "python run_experiments.py --check"),
        ("Run simple test", "python run_experiments.py -e paper_analysis -q simple_test --max-attempts 5"),
        ("Run comprehensive test", "python run_experiments.py -e reasoning_models -q narcotics"),
        ("Analyze results", "python analysis/results_analyzer.py -r experiment_results -o analysis_output")
    ]
    
    for description, command in examples:
        print(f"\n📋 {description}:")
        print(f"   {command}")

def main():
    print("🔧 ICOTN Multi-Model Setup")
    print("="*50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create directory structure
    create_directory_structure()
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Some dependencies failed to install. You may need to install them manually.")
    
    # Create environment template
    create_env_template()
    
    # Validate installation
    if validate_installation():
        print("✅ Installation validation successful!")
        print_usage_examples()
    else:
        print("❌ Installation validation failed. Please check the errors above.")
        print("💡 Try running: pip install -r requirements.txt")
    
    print("\n🔐 Next steps:")
    print("1. Edit .env file and add your API keys")
    print("2. Run: python run_experiments.py --check")
    print("3. Start experimenting with: python run_experiments.py --list-configs")

if __name__ == "__main__":
    main()
