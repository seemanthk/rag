# ShopSmart RAG System - Setup Guide
## Air Conditioner E-commerce Intelligence with Ollama

This guide will help you set up and run the RAG system using local Ollama models (Phi-3, Llama3, Gemma2) for analyzing Air Conditioner products from Amazon.

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Downloading Models](#downloading-models)
4. [Running the System](#running-the-system)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **RAM**: 8GB (16GB recommended)
- **Storage**: 20GB free space
- **Python**: 3.8 or higher
- **Internet**: For initial model downloads

### Recommended for Best Performance
- **RAM**: 16GB+
- **CPU**: Multi-core processor (4+ cores)
- **Storage**: SSD with 30GB+ free space

---

## Installation Steps

### Step 1: Install Python Dependencies

Open your terminal/command prompt in the project directory:

```bash
cd C:\Users\USER\Documents\rag
```

Install required Python packages:

```bash
pip install -r requirements.txt
```

**Required packages**:
```
torch
transformers
sentence-transformers
faiss-cpu
pandas
numpy
pyyaml
matplotlib
seaborn
scikit-learn
requests
jupyter
notebook
```

If you don't have a `requirements.txt`, create one with the packages listed above.

### Step 2: Install Ollama

#### Windows
1. Download Ollama from: https://ollama.com/download/windows
2. Run the installer
3. Ollama will start automatically

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### macOS
```bash
brew install ollama
```

### Step 3: Verify Ollama Installation

Check if Ollama is running:

```bash
ollama --version
```

If Ollama is not running, start it:

**Windows**: Ollama should auto-start. Check system tray.

**Linux/macOS**:
```bash
ollama serve
```

---

## Downloading Models

### Download All Three Models

Download the models used in this project (this may take 15-30 minutes depending on your internet speed):

```bash
# Download Phi-3 (smallest, fastest - ~3.8GB)
ollama pull phi3

# Download Llama3 (balanced - ~4.7GB)
ollama pull llama3

# Download Gemma2 (latest Google model - ~5.4GB)
ollama pull gemma2
```

### Verify Models

List downloaded models:

```bash
ollama list
```

You should see:
```
NAME            ID              SIZE
phi3:latest     xxx             3.8 GB
llama3:latest   xxx             4.7 GB
gemma2:latest   xxx             5.4 GB
```

### Alternative: Download Models One at a Time

If you have limited storage, start with just Phi-3:

```bash
ollama pull phi3
```

You can always download others later.

---

## Running the System

### Option 1: Quick Start (Jupyter Notebook)

The easiest way to explore the system:

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open the notebook:
```
notebooks/rag_ollama_demo.ipynb
```

3. Run cells sequentially (Shift+Enter)

### Option 2: Command Line (main_ollama.py)

#### Build the Vector Index (First Time Only)

```bash
python main_ollama.py --build-index
```

This will:
- Load the Air Conditioners CSV
- Create embeddings using sentence-transformers
- Build and save a FAISS vector index

#### Run a Single Query

```bash
python main_ollama.py --query "What are the best ACs under ₹35,000?"
```

#### Run Full Evaluation

Evaluate all 15 questions with all models:

```bash
python main_ollama.py --evaluate
```

Evaluate only 5 questions:

```bash
python main_ollama.py --evaluate --num-questions 5
```

#### Use Specific Models

Use only Phi-3:

```bash
python main_ollama.py --query "Best 1.5 ton inverter AC?" --models phi3
```

Use Phi-3 and Llama3:

```bash
python main_ollama.py --evaluate --models phi3 llama3 --num-questions 3
```

---

## Usage Examples

### Example 1: Interactive Exploration

```python
# In Python or Jupyter

from src.ollama_handler import OllamaHandler

# Initialize model
model = OllamaHandler(model_name="phi3")
model.load_model()

# Generate
response = model.generate("What makes a good air conditioner?")
print(response)
```

### Example 2: Compare Models on Specific Question

```bash
python main_ollama.py \
  --query "Which brand offers best value in budget segment?" \
  --models phi3 llama3 gemma2
```

### Example 3: Evaluate by Category

```bash
# Evaluate only value reasoning questions
python main_ollama.py --evaluate-category value_reasoning
```

### Example 4: Custom Query with Context

```python
from src.rag_system import RAGSystem
from src.vector_store import VectorStore
from src.ollama_handler import OllamaMultiLLMManager
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup
vector_store = VectorStore.load('data/vector_index')
llm_manager = OllamaMultiLLMManager(config['llms'])
llm_manager.load_llm('phi3')

rag = RAGSystem(vector_store, llm_manager, top_k=5)

# Query
result = rag.query(
    "Best energy efficient AC for small bedroom?",
    llm_name='phi3',
    return_context=True
)

print(result['answer'])
print(f"\nBased on {len(result['retrieved_docs'])} documents")
```

---

## Troubleshooting

### Issue 1: "Cannot connect to Ollama"

**Solution**:
1. Check if Ollama is running:
   ```bash
   ollama list
   ```

2. If not, start it:
   - **Windows**: Check system tray, or restart the Ollama app
   - **Linux/macOS**:
     ```bash
     ollama serve
     ```

3. Verify the URL in `config.yaml`:
   ```yaml
   ollama:
     base_url: "http://localhost:11434"
   ```

### Issue 2: "Model not found"

**Solution**:
```bash
# List available models
ollama list

# Pull missing model
ollama pull phi3
ollama pull llama3
ollama pull gemma2
```

### Issue 3: Out of Memory

**Solutions**:
1. Use only one model at a time
2. Use the smallest model (Phi-3)
3. Reduce batch size in config.yaml:
   ```yaml
   embedding:
     batch_size: 16  # Reduce from 32
   ```

### Issue 4: Slow Response Times

**Solutions**:
1. Phi-3 is fastest - use it for testing
2. Reduce `max_new_tokens` in config.yaml:
   ```yaml
   llms:
     phi3:
       max_new_tokens: 256  # Reduce from 512
   ```

### Issue 5: FAISS Installation Issues

**Solution** (Windows):
```bash
pip uninstall faiss-cpu faiss-gpu
pip install faiss-cpu
```

### Issue 6: "No module named 'sentence_transformers'"

**Solution**:
```bash
pip install sentence-transformers
```

### Issue 7: Vector Index Not Found

**Solution**:
```bash
# Rebuild the index
python main_ollama.py --force-rebuild
```

---

## Project Structure

```
rag/
├── config.yaml                    # Configuration file
├── main_ollama.py                 # Main script for Ollama
├── SETUP_GUIDE.md                 # This file
├── requirements.txt               # Python dependencies
│
├── data/
│   ├── Air Conditioners.csv       # Your dataset
│   └── vector_index/              # FAISS index (created)
│
├── src/
│   ├── ollama_handler.py          # Ollama LLM handler
│   ├── data_loader.py             # Data loading & preprocessing
│   ├── vector_store.py            # Vector DB (FAISS)
│   ├── rag_system.py              # Main RAG system
│   ├── evaluation.py              # Evaluation metrics
│   └── questions.py               # 15 evaluation questions
│
├── notebooks/
│   └── rag_ollama_demo.ipynb      # Interactive notebook
│
└── outputs/                       # Results (created)
    ├── evaluation_*.json
    └── summary_*.txt
```

---

## Next Steps

After setup:

1. **Build Index**: `python main_ollama.py --build-index`
2. **Test Query**: `python main_ollama.py --query "Best budget AC?"`
3. **Run Evaluation**: `python main_ollama.py --evaluate --num-questions 5`
4. **Explore Notebook**: Open `notebooks/rag_ollama_demo.ipynb`

---

## Configuration Details

### Key Config Parameters

**Ollama Settings** (`config.yaml`):
```yaml
ollama:
  base_url: "http://localhost:11434"
  timeout: 120  # seconds
```

**LLM Settings**:
```yaml
llms:
  phi3:
    type: "ollama"
    model_name: "phi3"
    max_new_tokens: 512
    temperature: 0.7
    top_p: 0.9
```

**RAG Settings**:
```yaml
rag:
  top_k: 3  # Number of documents to retrieve
  chunk_size: 512
  chunk_overlap: 50
```

---

## Performance Tips

1. **Start Small**: Use Phi-3 first, it's the fastest
2. **Limit Questions**: Test with 3-5 questions before running all 15
3. **Adjust Parameters**:
   - Lower `max_new_tokens` for faster responses
   - Lower `top_k` to retrieve fewer documents
   - Increase `temperature` for more creative answers

4. **Monitor Resources**:
   - Each model uses 4-6GB RAM when loaded
   - Only load one model at a time if RAM is limited

---

## Support & Resources

- **Ollama Documentation**: https://ollama.com/docs
- **Project Issues**: Check error logs in console
- **Model Cards**:
  - Phi-3: https://ollama.com/library/phi3
  - Llama3: https://ollama.com/library/llama3
  - Gemma2: https://ollama.com/library/gemma2

---

## Evaluation Metrics

The system evaluates models on:

1. **Answer Quality**: Relevance and accuracy
2. **Factuality**: Correctness of prices, specs, ratings
3. **Reasoning**: Value analysis, comparisons
4. **Response Time**: Generation speed
5. **Context Usage**: How well models use retrieved docs

Results are saved in `outputs/` directory with timestamps.

---

Good luck with your RAG system! 🚀
