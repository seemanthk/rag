# ShopSmart RAG: E-commerce Intelligence System
## Air Conditioner Product Analysis with Local LLMs

A Retrieval-Augmented Generation (RAG) system that helps make smart shopping decisions by analyzing Amazon Air Conditioner products using three local Ollama models: **Phi-3**, **Llama3**, and **Gemma2**.

---

## 🎯 Project Overview

This project implements an intelligent question-answering system that:

- **Analyzes 50 Air Conditioner products** from Amazon India
- **Combines structured data** (prices, ratings, brands) with **semantic search**
- **Compares 3 open-source LLMs** running locally via Ollama
- **Answers complex queries** about value, features, and recommendations
- **No cloud API costs** - everything runs on your machine!

---

## ✨ Key Features

### Value Reasoning
Not just "cheapest" but "best value for money"
- Combines price + specs + ratings
- Analyzes discount percentages
- Considers customer satisfaction

### Multi-Model Comparison
Compare answers from three different LLMs:
- **Phi-3** (3.8GB) - Fast and efficient
- **Llama3** (4.7GB) - Balanced performance
- **Gemma2** (5.4GB) - Latest Google model

### Domain-Specific Questions
15 evaluation questions across 4 categories:
1. **Structured Queries** - Price, rating filters
2. **Value Reasoning** - Best bang-for-buck analysis
3. **Temporal Analysis** - Review trends and consistency
4. **Combined** - Comprehensive recommendations

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and Start Ollama

**Windows:**
- Download from https://ollama.com/download/windows
- Install and it starts automatically

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

**macOS:**
```bash
brew install ollama
ollama serve
```

### 3. Download Models

```bash
ollama pull phi3
ollama pull llama3
ollama pull gemma2
```

### 4. Run System Check

```bash
python quick_start.py
```

This will:
- ✓ Check all installations
- ✓ Verify models are available
- ✓ Run a demo query

### 5. Build Vector Index

```bash
python main_ollama.py --build-index
```

### 6. Run Your First Query

```bash
python main_ollama.py --query "What are the best ACs under ₹35,000?"
```

---

## 📊 Dataset

**Source**: Amazon India - Air Conditioners
**Products**: 50
**Columns**:
- Product name
- Brand (LG, Samsung, Voltas, etc.)
- Ratings (1-5 stars)
- Number of ratings
- Discount price (₹)
- Actual price (₹)
- Category (Split AC, Window AC, Inverter, etc.)

**Price Range**: ₹26,490 - ₹52,990
**Average Rating**: 3.9 stars

---

## 💡 Usage Examples

### Interactive Notebook
```bash
jupyter notebook notebooks/rag_ollama_demo.ipynb
```

### Single Query with All Models
```bash
python main_ollama.py \
  --query "Which 1.5 ton inverter AC offers best value?" \
  --models phi3 llama3 gemma2
```

### Run Evaluation (5 Questions)
```bash
python main_ollama.py --evaluate --num-questions 5
```

### Full Evaluation (All 15 Questions)
```bash
python main_ollama.py --evaluate
```

### Evaluate Specific Category
```bash
python main_ollama.py --evaluate-category value_reasoning
```

---

## 📝 Sample Questions

1. "What are the top 5 rated air conditioners under ₹40,000?"
2. "Find the best value 1.5 ton AC considering specs and ratings"
3. "Which budget ACs (₹25,000-₹35,000) offer the best bang for buck?"
4. "Compare LG AC at ₹46,000 vs Voltas at ₹32,000 - which is better value?"
5. "Which brands maintain quality over time based on reviews?"

---

## 🏗️ System Architecture

```
USER QUERY
    ↓
VECTOR STORE (FAISS)
    ↓ (Retrieve top-k products)
CONTEXT FORMATION
    ↓
LLM GENERATION (Ollama: Phi-3/Llama3/Gemma2)
    ↓
ANSWER with REASONING
```

### Components:

1. **Data Loader** - Loads and preprocesses CSV
2. **Vector Store** - FAISS with sentence-transformers
3. **Ollama Handler** - Manages local LLMs
4. **RAG System** - Combines retrieval + generation
5. **Evaluator** - Measures answer quality

---

## 📁 Project Structure

```
rag/
├── config.yaml                    # System configuration
├── main_ollama.py                 # Main execution script
├── quick_start.py                 # Setup verification
├── requirements.txt               # Python dependencies
├── README_OLLAMA.md               # This file
├── SETUP_GUIDE.md                 # Detailed setup instructions
│
├── data/
│   ├── Air Conditioners.csv       # Product dataset
│   └── vector_index/              # FAISS index (auto-created)
│
├── src/
│   ├── ollama_handler.py          # Ollama LLM integration
│   ├── data_loader.py             # Data processing
│   ├── vector_store.py            # Vector database
│   ├── rag_system.py              # RAG pipeline
│   ├── evaluation.py              # Metrics
│   └── questions.py               # 15 eval questions
│
├── notebooks/
│   └── rag_ollama_demo.ipynb      # Interactive demo
│
└── outputs/                       # Generated results
    ├── evaluation_*.json
    └── summary_*.txt
```

---

## 🎓 Academic Context

This project fulfills requirements for **DSCI 6004: Natural Language Processing** term project:

### Requirements Met:
✅ RAG system development
✅ Three free/open-source LLMs (Phi-3, Llama3, Gemma2)
✅ 15+ domain-specific questions
✅ Comparative evaluation across models
✅ Analysis of performance, accuracy, and reasoning

### Deliverables:
- ✅ Working code with documentation
- ✅ Jupyter notebook for exploration
- ✅ Evaluation results and comparison
- ✅ Setup guide and README

---

## 📊 Evaluation Metrics

The system evaluates models on:

1. **Answer Relevancy** - How well answers address the question
2. **Faithfulness** - Accuracy to retrieved context
3. **Context Precision** - Quality of retrieval
4. **Response Length** - Conciseness vs completeness
5. **Response Time** - Generation speed
6. **Factual Accuracy** - Correctness of prices/specs/ratings

---

## 🔧 Configuration

Key settings in `config.yaml`:

```yaml
ollama:
  base_url: "http://localhost:11434"
  timeout: 120

llms:
  phi3:
    model_name: "phi3"
    max_new_tokens: 512
    temperature: 0.7

rag:
  top_k: 3
  chunk_size: 512
```

---

## 🐛 Troubleshooting

### Ollama not running
```bash
# Check status
ollama list

# Start server (Linux/macOS)
ollama serve
```

### Model not found
```bash
# Download model
ollama pull phi3
```

### Out of memory
- Use only Phi-3 (smallest)
- Reduce `max_new_tokens` in config
- Close other applications

### Slow responses
- Phi-3 is fastest - use for testing
- Reduce `top_k` to retrieve fewer docs
- Lower `max_new_tokens`

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed troubleshooting.

---

## 📈 Performance Comparison

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| Phi-3 | 3.8GB | ⚡⚡⚡ | ⭐⭐ | Quick testing |
| Llama3 | 4.7GB | ⚡⚡ | ⭐⭐⭐ | Balanced use |
| Gemma2 | 5.4GB | ⚡ | ⭐⭐⭐ | Best quality |

*(Actual results vary based on hardware)*

---

## 🚀 Getting Started (TL;DR)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Ollama (visit ollama.com)

# 3. Download models
ollama pull phi3 && ollama pull llama3 && ollama pull gemma2

# 4. Verify setup
python quick_start.py

# 5. Build index
python main_ollama.py --build-index

# 6. Run query
python main_ollama.py --query "Best budget AC?"

# 7. Full evaluation
python main_ollama.py --evaluate
```

---

**Happy analyzing! 🎉**

For detailed instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)
