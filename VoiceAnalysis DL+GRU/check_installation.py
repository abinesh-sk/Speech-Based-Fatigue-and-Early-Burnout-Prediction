"""
Quick verification script to check if all dependencies are installed correctly.
"""

import sys

def check_imports():
    """Check if all required packages can be imported."""
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'torchaudio': 'TorchAudio',
        'librosa': 'Librosa',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
        'soundfile': 'SoundFile',
    }
    
    print("="*60)
    print("CHECKING INSTALLED PACKAGES")
    print("="*60)
    
    all_good = True
    for package, name in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name:<20} version: {version}")
        except ImportError as e:
            print(f"✗ {name:<20} NOT FOUND")
            all_good = False
    
    print("="*60)
    
    if all_good:
        print("✓ All packages installed successfully!")
        
        # Check CUDA availability
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ CUDA not available - using CPU (this is fine for inference)")
        
        print("="*60)
        return True
    else:
        print("✗ Some packages are missing. Please run:")
        print("  pip install -r requirements.txt")
        print("="*60)
        return False


if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)
