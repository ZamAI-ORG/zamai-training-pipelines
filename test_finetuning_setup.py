#!/usr/bin/env python3
"""
Test script to validate fine-tuning scripts and Spaces setup
"""

import sys
import os
from pathlib import Path

def test_fine_tuning_scripts():
    """Test that fine-tuning scripts are present and valid"""
    print("Testing fine-tuning scripts...")
    
    scripts = [
        "scripts/fine_tune_phi3.py",
        "scripts/fine_tune_mt5.py",
        "scripts/fine_tune_mistral.py"
    ]
    
    for script in scripts:
        path = Path(script)
        if not path.exists():
            print(f"  ❌ Missing: {script}")
            return False
        else:
            print(f"  ✅ Found: {script}")
    
    return True

def test_hf_spaces():
    """Test that HF Spaces are properly configured"""
    print("\nTesting HF Spaces...")
    
    spaces = [
        "hf_spaces/phi3-finetuning",
        "hf_spaces/mt5-finetuning",
        "hf_spaces/voice-assistant",
        "hf_spaces/business-tools"
    ]
    
    required_files = ["app.py", "requirements.txt", "README.md"]
    
    all_good = True
    for space in spaces:
        space_path = Path(space)
        if not space_path.exists():
            print(f"  ❌ Missing Space: {space}")
            all_good = False
            continue
        
        print(f"  📁 {space}")
        for req_file in required_files:
            file_path = space_path / req_file
            if file_path.exists():
                print(f"    ✅ {req_file}")
            else:
                print(f"    ❌ Missing {req_file}")
                all_good = False
    
    return all_good

def test_documentation():
    """Test that documentation files exist"""
    print("\nTesting documentation...")
    
    docs = [
        "FINETUNING_GUIDE.md",
        "VOICE_MODEL_GUIDE.md",
        "README.md"
    ]
    
    for doc in docs:
        path = Path(doc)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✅ {doc} ({size} bytes)")
        else:
            print(f"  ❌ Missing: {doc}")
            return False
    
    return True

def test_deployment_script():
    """Test deployment script"""
    print("\nTesting deployment script...")
    
    script = Path("deploy_finetuning_spaces.sh")
    if not script.exists():
        print(f"  ❌ Missing: deploy_finetuning_spaces.sh")
        return False
    
    if not os.access(script, os.X_OK):
        print(f"  ⚠️  Not executable: deploy_finetuning_spaces.sh")
        print(f"      (Run: chmod +x deploy_finetuning_spaces.sh)")
    else:
        print(f"  ✅ Executable: deploy_finetuning_spaces.sh")
    
    return True

def test_env_configuration():
    """Test environment configuration"""
    print("\nTesting environment configuration...")
    
    env_example = Path(".env.example")
    if not env_example.exists():
        print(f"  ❌ Missing: .env.example")
        return False
    
    # Check for required variables
    required_vars = [
        "HF_TOKEN",
        "HF_ORG",
        "PHI3_BUSINESS_MODEL",
        "MT5_MODEL",
        "WHISPER_MODEL"
    ]
    
    with open(env_example) as f:
        content = f.read()
    
    all_found = True
    for var in required_vars:
        if var in content:
            print(f"  ✅ {var}")
        else:
            print(f"  ❌ Missing: {var}")
            all_found = False
    
    return all_found

def main():
    """Run all tests"""
    print("=" * 60)
    print("ZamAI Fine-tuning Setup Validation")
    print("=" * 60)
    
    tests = [
        ("Fine-tuning Scripts", test_fine_tuning_scripts),
        ("HF Spaces", test_hf_spaces),
        ("Documentation", test_documentation),
        ("Deployment Script", test_deployment_script),
        ("Environment Configuration", test_env_configuration)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ❌ Error in {name}: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Setup is complete.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
