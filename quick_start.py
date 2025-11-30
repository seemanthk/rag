"""
Quick Start Script for Air Conditioner RAG System
This script helps you get started quickly with the Ollama-based RAG system
"""

import os
import sys
import yaml
import subprocess

def check_ollama():
    """Check if Ollama is installed and running"""
    print("Checking Ollama installation...")

    try:
        result = subprocess.run(['ollama', '--version'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            print("✗ Ollama command failed")
            return False
    except FileNotFoundError:
        print("✗ Ollama is not installed")
        print("\nPlease install Ollama:")
        print("  Windows: https://ollama.com/download/windows")
        print("  Linux: curl -fsSL https://ollama.com/install.sh | sh")
        print("  macOS: brew install ollama")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False


def check_ollama_running():
    """Check if Ollama server is running"""
    print("\nChecking if Ollama is running...")

    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama server is running")

            models = response.json().get('models', [])
            if models:
                print(f"  Available models: {[m['name'] for m in models]}")
            return True
        else:
            print("✗ Ollama server returned error")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama server")
        print("\nPlease start Ollama:")
        print("  Windows: Ollama should auto-start, check system tray")
        print("  Linux/macOS: Run 'ollama serve' in a terminal")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def check_models():
    """Check if required models are downloaded"""
    print("\nChecking required models...")

    required_models = ['phi3', 'llama3', 'gemma2']

    try:
        result = subprocess.run(['ollama', 'list'],
                              capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            available = result.stdout.lower()

            missing = []
            for model in required_models:
                if model in available:
                    print(f"  ✓ {model}")
                else:
                    print(f"  ✗ {model} (not downloaded)")
                    missing.append(model)

            if missing:
                print("\nTo download missing models:")
                for model in missing:
                    print(f"  ollama pull {model}")
                return False
            else:
                print("\n✓ All required models are available!")
                return True
        else:
            print("✗ Could not list models")
            return False

    except Exception as e:
        print(f"✗ Error checking models: {e}")
        return False


def check_python_packages():
    """Check if required Python packages are installed"""
    print("\nChecking Python packages...")

    required = [
        'yaml',
        'pandas',
        'numpy',
        'torch',
        'transformers',
        'sentence_transformers',
        'faiss',
        'requests'
    ]

    missing = []
    for package in required:
        try:
            if package == 'yaml':
                __import__('yaml')
            elif package == 'faiss':
                __import__('faiss')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing.append(package)

    if missing:
        print("\nTo install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr install all requirements:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages are installed!")
        return True


def check_data():
    """Check if data file exists"""
    print("\nChecking data file...")

    data_path = "data/Air Conditioners.csv"

    if os.path.exists(data_path):
        import pandas as pd
        try:
            df = pd.read_csv(data_path)
            print(f"✓ Data file found: {len(df)} products")
            return True
        except Exception as e:
            print(f"✗ Error reading data: {e}")
            return False
    else:
        print(f"✗ Data file not found: {data_path}")
        print("\nPlease ensure the Air Conditioners.csv file is in the data/ directory")
        return False


def run_demo():
    """Run a quick demo"""
    print("\n" + "="*80)
    print("Running Quick Demo")
    print("="*80)

    try:
        from src.data_loader import AmazonProductDataLoader
        from src.vector_store import VectorStore
        from src.ollama_handler import OllamaMultiLLMManager
        from src.rag_system import RAGSystem

        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        print("\n1. Loading data...")
        data_loader = AmazonProductDataLoader(config['dataset']['path'])
        df = data_loader.load_data()
        df = data_loader.preprocess_data()
        documents = data_loader.create_documents()
        print(f"   Loaded {len(documents)} products")

        print("\n2. Building vector index...")
        vector_store = VectorStore(
            embedding_model_name=config['embedding']['model_name'],
            index_path=config['vector_db']['index_path'],
            use_faiss=True
        )

        # Check if index exists
        if os.path.exists(f"{config['vector_db']['index_path']}.index"):
            print("   Loading existing index...")
            vector_store.load_index()
        else:
            print("   Creating new index (this may take a minute)...")
            embeddings = vector_store.create_embeddings(documents, batch_size=32)
            vector_store.build_index()
            vector_store.save_index()
        print("   ✓ Vector index ready")

        print("\n3. Loading Phi-3 model...")
        llm_manager = OllamaMultiLLMManager(
            llm_configs=config['llms'],
            base_url=config['ollama']['base_url']
        )
        llm_manager.load_llm('phi3')
        print("   ✓ Phi-3 loaded")

        print("\n4. Setting up RAG system...")
        rag_system = RAGSystem(
            vector_store=vector_store,
            llm_manager=llm_manager,
            top_k=3
        )
        print("   ✓ RAG system ready")

        print("\n5. Running test query...")
        test_question = "What are the top-rated 1.5 ton air conditioners under ₹35,000?"
        print(f"   Q: {test_question}")

        result = rag_system.query(
            query=test_question,
            llm_name='phi3',
            return_context=True
        )

        print(f"\n   A: {result['answer']}")
        print(f"\n   (Based on {result['num_retrieved']} retrieved documents)")

        print("\n" + "="*80)
        print("Demo completed successfully! 🎉")
        print("="*80)

        print("\nNext steps:")
        print("  1. Open notebooks/rag_ollama_demo.ipynb for interactive exploration")
        print("  2. Run: python main_ollama.py --query 'your question here'")
        print("  3. Run full evaluation: python main_ollama.py --evaluate")

        return True

    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("AIR CONDITIONER RAG SYSTEM - QUICK START")
    print("="*80)

    # Run all checks
    checks = [
        ("Ollama Installation", check_ollama),
        ("Ollama Server", check_ollama_running),
        ("Ollama Models", check_models),
        ("Python Packages", check_python_packages),
        ("Data File", check_data),
    ]

    all_passed = True
    for name, check_func in checks:
        if not check_func():
            all_passed = False

    print("\n" + "="*80)

    if all_passed:
        print("✓ All checks passed!")
        print("="*80)

        response = input("\nWould you like to run a quick demo? (y/n): ")
        if response.lower() in ['y', 'yes']:
            run_demo()
        else:
            print("\nYou're all set! Start with:")
            print("  python main_ollama.py --help")
            print("  jupyter notebook notebooks/rag_ollama_demo.ipynb")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("="*80)
        print("\nRefer to SETUP_GUIDE.md for detailed instructions")


if __name__ == "__main__":
    main()
