"""
Quick test to verify Streamlit and all dependencies are installed
Run this before launching the Streamlit app
"""

import sys

def check_imports():
    """Check if all required packages are installed"""

    print("=" * 60)
    print("STREAMLIT SETUP VERIFICATION")
    print("=" * 60)
    print()

    required_packages = {
        'streamlit': 'Streamlit (Web framework)',
        'plotly': 'Plotly (Interactive charts)',
        'pandas': 'Pandas (Data processing)',
        'yaml': 'PyYAML (Configuration)',
        'numpy': 'NumPy (Numerical computing)',
        'sentence_transformers': 'Sentence Transformers (Embeddings)',
        'faiss': 'FAISS (Vector search)',
        'requests': 'Requests (HTTP client)',
    }

    all_installed = True

    for package, description in required_packages.items():
        try:
            if package == 'yaml':
                import yaml
            elif package == 'sentence_transformers':
                import sentence_transformers
            elif package == 'faiss':
                import faiss
            else:
                __import__(package)

            print(f"✅ {description:<40} OK")
        except ImportError:
            print(f"❌ {description:<40} NOT INSTALLED")
            all_installed = False

    print()
    print("=" * 60)

    if all_installed:
        print("✅ ALL DEPENDENCIES INSTALLED")
        print()
        print("You can now run the Streamlit app:")
        print("  streamlit run streamlit_app.py")
        print()
        print("Or test basic functionality:")
        test_basic_functionality()
    else:
        print("❌ MISSING DEPENDENCIES")
        print()
        print("Please install missing packages:")
        print("  pip install -r requirements.txt")

    print("=" * 60)

    return all_installed


def test_basic_functionality():
    """Test basic Streamlit functionality"""
    try:
        import streamlit as st
        import plotly.graph_objects as go
        import pandas as pd

        print()
        print("Testing basic functionality...")

        # Test Pandas DataFrame
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        assert len(df) == 2, "Pandas DataFrame test failed"
        print("  ✅ Pandas DataFrame")

        # Test Plotly figure
        fig = go.Figure(data=[go.Bar(x=[1, 2], y=[3, 4])])
        assert fig.data is not None, "Plotly figure test failed"
        print("  ✅ Plotly Figure")

        # Check Streamlit version
        import streamlit
        version = streamlit.__version__
        print(f"  ✅ Streamlit version: {version}")

        print()
        print("All functionality tests passed!")

    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")


def check_ollama():
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            print("\n✅ Ollama is running")

            models = response.json().get('models', [])
            if models:
                print("\nInstalled models:")
                for model in models:
                    print(f"  - {model['name']}")
            else:
                print("⚠️  No models installed. Install with:")
                print("  ollama pull phi3:mini")
                print("  ollama pull llama3.2:latest")
                print("  ollama pull gemma2:2b")
        else:
            print("\n⚠️  Ollama responded but with unexpected status")
    except requests.exceptions.RequestException:
        print("\n⚠️  Ollama is not running")
        print("Start Ollama with: ollama serve")
    except ImportError:
        print("\n⚠️  requests package not installed")


if __name__ == "__main__":
    all_ok = check_imports()

    if all_ok:
        check_ollama()

        print()
        print("=" * 60)
        print("READY TO LAUNCH STREAMLIT APP!")
        print("=" * 60)
        print()
        print("Run: streamlit run streamlit_app.py")
        print()
