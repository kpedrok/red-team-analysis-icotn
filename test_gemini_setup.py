#!/usr/bin/env python3
"""
Test script for Gemini API setup with CoT narcotics experiment
"""

import os
import sys
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_environment():
    """Test environment setup"""
    print("🧪 Testing Gemini API Setup")
    print("=" * 50)
    
    # Check API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ GEMINI_API_KEY not found in environment")
        print("💡 Make sure you have GEMINI_API_KEY=your_key in your .env file")
        return False
    
    print(f"✅ GEMINI_API_KEY found: {gemini_key[:10]}***")
    
    # Test imports
    try:
        import google.generativeai as genai
        print("✅ Google AI library available")
        
        # Test API connection
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        print("🧪 Testing API connection...")
        response = model.generate_content("Say hello in one word")
        print(f"✅ API connection successful: '{response.text.strip()}'")
        
    except Exception as e:
        print(f"❌ Google AI setup failed: {e}")
        return False
    
    # Test configuration
    try:
        from config.model_config import get_model_config, validate_api_keys
        
        print("\n🔧 Testing model configuration...")
        
        # Check Gemini models
        gemini_25_config = get_model_config("gemini-2.5-flash")
        gemini_20_config = get_model_config("gemini-2.0-flash")
        
        print(f"✅ Gemini 2.5 Flash configured: {gemini_25_config.name}")
        print(f"   - Reasoning support: {gemini_25_config.supports_reasoning}")
        print(f"   - API key env: {gemini_25_config.api_key_env}")
        
        print(f"✅ Gemini 2.0 Flash configured: {gemini_20_config.name}")
        print(f"   - Reasoning support: {gemini_20_config.supports_reasoning}")
        print(f"   - API key env: {gemini_20_config.api_key_env}")
        
        # Check API key validation
        key_status = validate_api_keys()
        if key_status.get("GEMINI_API_KEY", False):
            print("✅ GEMINI_API_KEY validation passed")
        else:
            print("❌ GEMINI_API_KEY validation failed")
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Ready for experiments.")
    return True

async def test_quick_attack():
    """Test a quick ICOTN attack"""
    print("\n🎯 Testing Quick ICOTN Attack")
    print("-" * 30)
    
    try:
        from tools.multi_model_client import MultiModelClient
        from config.model_config import get_model_config
        
        # Setup client
        client = MultiModelClient()
        config = get_model_config("gemini-2.5-flash")
        
        if not client.register_model("gemini-2.5-flash", config):
            print("❌ Failed to register model")
            return False
        
        print("✅ Model registered successfully")
        
        # Test simple query
        messages = [
            {"role": "user", "content": "What is 2+2? Answer briefly."}
        ]
        
        print("🧪 Testing simple query...")
        response = await client.generate("gemini-2.5-flash", messages)
        
        if response.error:
            print(f"❌ Query failed: {response.error}")
            return False
        
        print(f"✅ Query successful: '{response.content[:50]}...'")
        print(f"   Latency: {response.latency:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Attack test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_next_steps():
    """Show what to do next"""
    print("\n🚀 Next Steps:")
    print("=" * 30)
    
    print("1. Run your CoT narcotics experiment:")
    print("   python run_experiments.py -e cot_narcotics -q narcotics")
    
    print("\n2. Or start with a quick test:")
    print("   python run_experiments.py -e cot_narcotics -q simple_test --max-attempts 5")
    
    print("\n3. Analyze results:")
    print("   python analysis/results_analyzer.py -r cot_narcotics_results")
    
    print("\n📋 Your experiment will test:")
    print("   🤖 Models: 8 CoT-capable models (GPT-OSS-20B, DeepSeek R1, GPT-5-Nano, O3-Mini, GPT-5, Gemini 2.5 Flash, Claude 3.5 Sonnet, Claude 3.5 Haiku)")
    print("   📝 Queries: Narcotics manufacturing, sourcing, distribution, security")
    print("   🎯 Max attempts: 25 per model-query combination")
    print("   ⚡ Parallel workers: 1 (sequential to avoid rate limits)")
    print("   📊 Total attacks: 32 (8 models × 4 narcotics queries)")

async def main():
    """Run all tests"""
    
    if not test_environment():
        print("\n💡 Setup Instructions:")
        print("1. Make sure you have a .env file with: GEMINI_API_KEY=your_actual_key")
        print("2. Install dependencies: python setup_multi_model.py")
        print("3. Re-run this test: python test_gemini_setup.py")
        return
    
    # Test quick attack
    attack_success = await test_quick_attack()
    
    if attack_success:
        show_next_steps()
    else:
        print("❌ Setup issues detected. Please check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
