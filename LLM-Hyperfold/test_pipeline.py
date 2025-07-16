#!/usr/bin/env python3
"""
🧪 QUICK TEST SCRIPT
===================
Test the complete pipeline: dataset → training → inference
"""

import os
import sys
import subprocess
import time

def run_command(cmd, description):
    """Run command and track time"""
    print(f"\n⚡ {description}")
    print(f"🔧 Running: {cmd}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Success ({duration:.1f}s)")
            if result.stdout.strip():
                # Show last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    print(f"   {line}")
        else:
            print(f"❌ Failed ({duration:.1f}s)")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False
    
    return True

def check_files():
    """Check if required files exist"""
    print("\n📂 Checking required files...")
    
    required_files = [
        "simple_dataset.py",
        "build.py", 
        "train.py",
        "inference.py",
        "models/basis_hyper.py",
        "models/hyper_llama.py",
        "models/hyper_model.py"
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
            missing.append(file)
    
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    
    print("✅ All required files present")
    return True

def main():
    """Run complete test pipeline"""
    print("🧪 COMPLETE PIPELINE TEST")
    print("=" * 30)
    
    # Check files first
    if not check_files():
        return False
    
    # Step 1: Create dataset
    if not os.path.exists("datasets/combined_dataset.csv"):
        success = run_command(
            "python simple_dataset.py",
            "Creating training dataset (6000 samples)"
        )
        if not success:
            print("❌ Dataset creation failed")
            return False
    else:
        print("\n✅ Dataset already exists")
    
    # Step 2: Quick training test (1 epoch only)
    print("\n🏋️ Testing training script (quick mode)...")
    success = run_command(
        "python -c \"from train import main; import sys; sys.argv = ['train.py', '--epochs', '1', '--quick-test']; main()\"",
        "Running 1 epoch training test"
    )
    
    if not success:
        print("❌ Training test failed")
        return False
    
    # Step 3: Test inference if checkpoint exists
    checkpoint_files = [
        "checkpoints/best_hypernetwork_350M.pt",
        "checkpoints/final_hypernetwork_350M.pt"
    ]
    
    checkpoint_exists = any(os.path.exists(f) for f in checkpoint_files)
    
    if checkpoint_exists:
        success = run_command(
            "python inference.py",
            "Testing inference engine"
        )
        if not success:
            print("❌ Inference test failed")
            return False
    else:
        print("\n⚠️  No checkpoint found, skipping inference test")
        print("   Run full training first: python train.py")
    
    # Final summary
    print("\n🎉 PIPELINE TEST COMPLETE!")
    print("=" * 30)
    print("✅ Dataset creation: Working")
    print("✅ Training script: Working") 
    if checkpoint_exists:
        print("✅ Inference engine: Working")
    else:
        print("⚠️  Inference engine: Needs trained checkpoint")
    
    print("\n🚀 To run full training:")
    print("   python train.py")
    print("\n⚡ To run inference after training:")
    print("   python inference.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
