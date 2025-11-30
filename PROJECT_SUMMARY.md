# Amazon Product RAG System - Project Summary

## Overview

A complete Retrieval-Augmented Generation (RAG) system for answering questions about Amazon products, comparing performance across three open-source LLMs: Llama-3, Mistral-7B, and Phi-3.

**Project for**: DSCI 6004 - Natural Language Processing
**Milestones**:
- Proposal Slides: 11/09/2025
- Code Demo: 11/30/2025 ⭐ (Today!)
- Final Submission: 12/07/2025

---

## ✅ Project Status

### Completed Components

1. **Data Pipeline** ✓
   - CSV loader with preprocessing
   - Document chunking and formatting
   - Statistics and analysis

2. **Vector Store** ✓
   - FAISS-based similarity search
   - Sentence transformer embeddings
   - Index persistence

3. **Multi-LLM Support** ✓
   - Llama-3 (3B parameters)
   - Mistral-7B (7B parameters)
   - Phi-3 (3.8B parameters)
   - 4-bit quantization for efficiency

4. **RAG System** ✓
   - Query processing
   - Context retrieval
   - Prompt engineering
   - Response generation

5. **Evaluation Framework** ✓
   - 15 domain-specific questions
   - Multiple evaluation metrics
   - Comparative analysis
   - Automated reporting

6. **Documentation** ✓
   - Comprehensive README
   - Quick start guide
   - Code comments
   - Jupyter notebook

---

## 📁 Project Structure

```
rag/
├── src/                          # Core modules
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Dataset loading (350 lines)
│   ├── vector_store.py          # Embeddings & search (230 lines)
│   ├── llm_handler.py           # LLM management (250 lines)
│   ├── rag_system.py            # RAG pipeline (180 lines)
│   ├── evaluation.py            # Metrics & comparison (300 lines)
│   └── questions.py             # 15 evaluation questions
│
├── data/                         # Dataset directory
│   ├── amazon_products.csv      # Dataset (download required)
│   └── vector_index/            # Generated index files
│
├── outputs/                      # Results directory
│   ├── results_*.json           # Detailed results
│   └── summary_*.txt            # Summary reports
│
├── notebooks/                    # Jupyter notebooks
│   └── rag_exploration.ipynb    # Interactive exploration
│
├── config.yaml                   # System configuration
├── requirements.txt              # Python dependencies
├── main.py                       # Full evaluation script
├── demo.py                       # Quick demo script
├── check_system.py              # System requirements checker
├── download_dataset.py          # Dataset downloader
├── setup.py                      # Package setup
├── .gitignore                   # Git ignore rules
├── README.md                    # Full documentation
├── QUICKSTART.md                # Quick start guide
└── PROJECT_SUMMARY.md           # This file
```

**Total Code**: ~1,500+ lines of Python

---

## 🎯 Key Features

### 1. Flexible Data Processing
- Handles various CSV formats
- Automatic column mapping
- Missing value handling
- Product text generation

### 2. Efficient Retrieval
- Sentence transformer embeddings
- FAISS vector index
- Cosine similarity search
- Configurable top-k

### 3. Multi-Model Comparison
- Three different LLMs
- Unified interface
- Model switching
- Batch processing

### 4. Comprehensive Evaluation
- Answer quality metrics
- Context grounding
- Specificity scoring
- Cross-model comparison

### 5. Production-Ready
- Configuration management
- Error handling
- Logging
- Modular design

---

## 📊 Evaluation Questions (15 Total)

### Categories:
1. **Product Search** - Finding specific products
2. **Price Filtering** - Budget-based queries
3. **Comparison** - Feature comparisons
4. **Sentiment Analysis** - Review analysis
5. **Recommendations** - Product suggestions
6. **Feature Extraction** - Identifying features
7. **Aggregation** - Statistical queries
8. **Brand Analysis** - Brand-level insights
9. **Educational** - Concept explanations
10. **Meta Analysis** - Dataset analysis

### Difficulty Levels:
- Easy: 5 questions
- Medium: 8 questions
- Hard: 2 questions

---

## 🚀 Usage Examples

### Quick Demo (5 minutes)
```bash
python demo.py
```

### Full Evaluation (30-60 minutes)
```bash
python main.py --build-index
```

### Specific Models
```bash
python main.py --llms llama3 phi3 --num-questions 5
```

### Interactive Exploration
```bash
jupyter notebook notebooks/rag_exploration.ipynb
```

---

## 📈 Evaluation Metrics

### Answer Quality
- **Length**: Word and character counts
- **Relevancy**: Query keyword overlap
- **Faithfulness**: Context grounding score
- **Specificity**: Numeric/factual content
- **Readability**: Sentence structure

### Comparison
- **Diversity**: Response variety across models
- **Agreement**: Similarity between answers
- **Rankings**: Per-metric model rankings

---

## 💻 Technical Stack

### Core Libraries
- **PyTorch** - Deep learning framework
- **Transformers** - LLM interface
- **Sentence Transformers** - Embeddings
- **FAISS** - Vector search
- **Pandas** - Data processing

### Models
- **Llama-3-3B** (Meta) - General purpose
- **Mistral-7B** (Mistral AI) - Instruction following
- **Phi-3-mini** (Microsoft) - Efficient inference

### Optimization
- 4-bit quantization
- Batch processing
- Index caching
- GPU acceleration

---

## 📋 Project Requirements (Met)

### Core Requirements ✓
1. ✅ RAG system development
2. ✅ Three open-source LLMs
3. ✅ 10+ domain-specific questions (15 included)
4. ✅ Performance evaluation
5. ✅ Comparative analysis

### Implementation ✓
- ✅ Novel retrieval mechanism (FAISS + sentence transformers)
- ✅ Efficient architecture (quantization, caching)
- ✅ Multiple evaluation metrics
- ✅ Comprehensive documentation

### Deliverables ✓
- ✅ GitHub repository
- ✅ User documentation (README.md)
- ✅ Configuration files
- ✅ Evaluation framework
- ✅ Code organization

---

## 🎬 Demo Video Checklist

### What to Show (< 10 minutes)

1. **Introduction** (1 min)
   - Project overview
   - Dataset description
   - System architecture

2. **Code Walkthrough** (3 min)
   - Project structure
   - Key modules
   - Configuration

3. **Live Demo** (4 min)
   - Run demo.py
   - Show retrieval
   - Display answers
   - Compare models

4. **Results Analysis** (2 min)
   - Evaluation metrics
   - Model comparison
   - Key findings

---

## 📝 Final Report Outline

### Structure (5-8 pages)

1. **Abstract & Introduction**
   - Problem statement
   - RAG overview
   - Project objectives

2. **Related Work**
   - RAG systems
   - LLM comparison studies
   - Citations (5-10 papers)

3. **System Design**
   - Architecture
   - Components
   - Implementation details

4. **Experimental Setup**
   - Dataset description
   - Model configurations
   - Evaluation methodology

5. **Results**
   - Quantitative metrics
   - Qualitative analysis
   - Model comparisons

6. **Discussion**
   - Strengths/weaknesses
   - Insights
   - Future work

7. **Conclusion**
   - Summary
   - Contributions

8. **References**

---

## 🔧 System Requirements

### Minimum
- Python 3.8+
- 16GB RAM
- 20GB disk space
- CPU with 4+ cores

### Recommended
- Python 3.10+
- 32GB RAM
- 50GB SSD storage
- NVIDIA GPU (8GB+ VRAM)

### Compatibility
- ✅ Windows 10/11
- ✅ Linux (Ubuntu 20.04+)
- ✅ macOS (Intel/Apple Silicon)

---

## 🎓 Learning Outcomes

This project demonstrates:

1. **RAG Implementation**
   - Vector database design
   - Semantic search
   - Context retrieval

2. **LLM Integration**
   - Model loading
   - Prompt engineering
   - Generation control

3. **Evaluation Design**
   - Metric selection
   - Comparative analysis
   - Result interpretation

4. **Software Engineering**
   - Modular design
   - Configuration management
   - Documentation

---

## 📚 Key References

1. **RAG Paper**: Lewis et al. (2020) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
2. **FAISS**: Johnson et al. (2019) - "Billion-scale similarity search with GPUs"
3. **Sentence-BERT**: Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
4. **Llama**: Touvron et al. (2023) - "Llama 2: Open Foundation and Fine-Tuned Chat Models"
5. **Mistral**: Jiang et al. (2023) - "Mistral 7B"

---

## 🚧 Known Limitations

1. **Memory**: Large models require significant RAM
2. **Speed**: CPU inference is slow
3. **Dataset**: Limited to Amazon products
4. **Metrics**: Basic evaluation metrics only
5. **Context**: Limited context window size

---

## 🔮 Future Enhancements

1. **Technical**
   - Hybrid search (dense + sparse)
   - Reranking mechanism
   - Streaming responses
   - API endpoint

2. **Evaluation**
   - Human evaluation
   - BLEU/ROUGE scores
   - Hallucination detection
   - Factual accuracy check

3. **Features**
   - Web interface
   - Multi-modal RAG
   - Multilingual support
   - Custom datasets

---

## 📧 Contact & Support

- **Issues**: Open GitHub issue
- **Questions**: Check README.md and QUICKSTART.md
- **Documentation**: See notebooks/ for examples

---

## ✨ Acknowledgments

- **Dataset**: Lokesh Parab (Kaggle)
- **Models**: Meta, Mistral AI, Microsoft
- **Frameworks**: HuggingFace, FAISS
- **Course**: DSCI 6004 - NLP

---

**Last Updated**: November 30, 2025
**Status**: ✅ Ready for Demo
**Next Deadline**: Final Submission (12/07/2025)

---

## 🎉 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python download_dataset.py

# 3. Check system
python check_system.py

# 4. Run demo
python demo.py

# 5. Full evaluation
python main.py --build-index
```

---

**Good luck with your demo and final submission! 🚀**
