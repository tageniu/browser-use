#!/usr/bin/env python3
"""Test script to verify GAIA dataset loading and basic setup."""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required imports work."""
    try:
        import datasets
        print("✓ datasets imported successfully")
        
        from browser_use import Agent
        print("✓ browser_use Agent imported successfully")
        
        from browser_use.llm.openai.chat import ChatOpenAI
        print("✓ ChatOpenAI imported successfully")
        
        from dotenv import load_dotenv
        print("✓ dotenv imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_gaia_dataset():
    """Test loading a small sample of GAIA dataset."""
    try:
        import datasets
        
        print("Testing GAIA dataset loading...")
        
        # Try to load the real GAIA dataset
        dataset = datasets.load_dataset("gaia-benchmark/GAIA", "2023_level1", split="test", trust_remote_code=True)
        
        # Convert to list to handle different dataset types
        dataset_list = list(dataset)
        
        print(f"✓ Real GAIA dataset loaded successfully with {len(dataset_list)} tasks")
        
        # Show first task example
        if dataset_list:
            first_task = dataset_list[0]
            print(f"✓ First task example:")
            print(f"  - Question: {first_task.get('Question', 'N/A')[:100]}...")
            print(f"  - Level: {first_task.get('Level', 'N/A')}")
            print(f"  - Task ID: {first_task.get('task_id', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ GAIA dataset loading error: {e}")
        print("The GAIA dataset is gated on Hugging Face. Please:")
        print("1. Visit https://huggingface.co/datasets/gaia-benchmark/GAIA")
        print("2. Request access to the dataset")
        print("3. Authenticate with: huggingface-cli login")
        return False

def test_environment():
    """Test environment setup."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print(f"✓ OpenAI API key found (length: {len(api_key)})")
        else:
            print("⚠ OpenAI API key not found in environment")
            print("  Please set OPENAI_API_KEY in your .env file")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment setup error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing GAIA Browser-Use Setup")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Test", test_environment),
        ("GAIA Dataset Test", test_gaia_dataset),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} passed")
            else:
                print(f"✗ {test_name} failed")
        except Exception as e:
            print(f"✗ {test_name} error: {e}")
    
    print(f"\n{'='*40}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to run GAIA tasks.")
        print("Run: python examples/gaia.py")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
