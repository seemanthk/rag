"""
System requirements checker for Amazon Product RAG System
Run this before starting to ensure your system is ready
"""

import sys
import os
import platform
from pathlib import Path


def print_section(title):
    """Print section header"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)


def check_python_version():
    """Check Python version"""
    print_section("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print_section("Dependencies")

    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sentence_transformers': 'Sentence Transformers',
        'faiss': 'FAISS (CPU)',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'yaml': 'PyYAML',
    }

    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} not installed")
            missing.append(name)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nTo install:")
        print("  pip install -r requirements.txt")
        return False

    return True


def check_cuda():
    """Check CUDA availability"""
    print_section("GPU/CUDA")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    Memory: {mem:.2f} GB")
            return True
        else:
            print("⚠ CUDA not available - will use CPU")
            print("  GPU is recommended for faster inference")
            return False
    except Exception as e:
        print(f"⚠ Could not check CUDA: {e}")
        return False


def check_memory():
    """Check available RAM"""
    print_section("System Memory")

    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)

        print(f"Total RAM: {total_gb:.2f} GB")
        print(f"Available RAM: {available_gb:.2f} GB")

        if total_gb >= 16:
            print("✓ Sufficient RAM")
            return True
        else:
            print("⚠ Less than 16GB RAM - may experience issues")
            print("  Recommended: 16GB minimum, 32GB preferred")
            return False
    except ImportError:
        print("⚠ psutil not installed - cannot check RAM")
        print("  Install with: pip install psutil")
        return False


def check_disk_space():
    """Check available disk space"""
    print_section("Disk Space")

    try:
        import shutil
        stat = shutil.disk_usage('.')
        free_gb = stat.free / (1024**3)

        print(f"Free disk space: {free_gb:.2f} GB")

        if free_gb >= 20:
            print("✓ Sufficient disk space")
            return True
        else:
            print("⚠ Less than 20GB free - may not be enough")
            print("  Recommended: 20GB minimum, 50GB preferred")
            return False
    except Exception as e:
        print(f"⚠ Could not check disk space: {e}")
        return False


def check_dataset():
    """Check if dataset is available"""
    print_section("Dataset")

    data_dir = Path("data")
    csv_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []

    if csv_files:
        print("✓ CSV file(s) found:")
        for csv_file in csv_files:
            size_mb = csv_file.stat().st_size / (1024**2)
            print(f"  - {csv_file.name} ({size_mb:.2f} MB)")
        return True
    else:
        print("❌ No CSV files found in data/ directory")
        print("\nTo download:")
        print("  python download_dataset.py")
        print("\nOr manually:")
        print("  1. Visit: https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset")
        print("  2. Download and place in data/ folder")
        return False


def check_config():
    """Check if configuration file exists"""
    print_section("Configuration")

    config_file = Path("config.yaml")

    if config_file.exists():
        print("✓ config.yaml found")

        try:
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)

            # Check critical paths
            dataset_path = config.get('dataset', {}).get('path', '')
            if Path(dataset_path).exists():
                print(f"✓ Dataset path valid: {dataset_path}")
            else:
                print(f"⚠ Dataset path not found: {dataset_path}")
                print("  Update config.yaml with correct path")

            return True
        except Exception as e:
            print(f"⚠ Error reading config: {e}")
            return False
    else:
        print("❌ config.yaml not found")
        return False


def check_huggingface():
    """Check HuggingFace access"""
    print_section("HuggingFace")

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        print("✓ HuggingFace Hub accessible")

        # Check if we can access a model
        try:
            api.model_info("microsoft/Phi-3-mini-4k-instruct")
            print("✓ Can access model repositories")
            return True
        except Exception as e:
            print(f"⚠ Cannot access models: {e}")
            print("  Check internet connection")
            return False
    except ImportError:
        print("⚠ huggingface_hub not installed")
        return False
    except Exception as e:
        print(f"⚠ Error checking HuggingFace: {e}")
        return False


def generate_report():
    """Generate complete system check report"""
    print("\n")
    print("*" * 80)
    print(" " * 20 + "SYSTEM REQUIREMENTS CHECK")
    print("*" * 80)

    checks = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "GPU/CUDA": check_cuda(),
        "System Memory": check_memory(),
        "Disk Space": check_disk_space(),
        "Dataset": check_dataset(),
        "Configuration": check_config(),
        "HuggingFace": check_huggingface(),
    }

    # Summary
    print_section("Summary")

    passed = sum(checks.values())
    total = len(checks)

    for check, status in checks.items():
        symbol = "✓" if status else "❌"
        print(f"{symbol} {check}")

    print(f"\nPassed: {passed}/{total} checks")

    if passed == total:
        print("\n🎉 All checks passed! You're ready to run the RAG system.")
        print("\nNext steps:")
        print("  1. python demo.py           # Quick demo")
        print("  2. python main.py           # Full evaluation")
    elif passed >= total - 2:
        print("\n⚠ Most checks passed. You can proceed but may encounter issues.")
        print("\nRecommended:")
        print("  - Address any ❌ items above")
        print("  - Try running demo.py to test")
    else:
        print("\n❌ Several checks failed. Please address issues before proceeding.")
        print("\nRequired:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Download dataset: python download_dataset.py")

    # System info
    print_section("System Information")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.machine()}")
    print(f"Python: {sys.version}")

    print("\n" + "*" * 80 + "\n")


if __name__ == "__main__":
    generate_report()
