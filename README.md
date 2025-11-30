# Amazon Product RAG System - LLM Comparison

A Retrieval-Augmented Generation (RAG) system for answering questions about Amazon products using multiple open-source Language Models (LLMs). This project compares the performance of Llama-3, Mistral-7B, and Phi-3 models in a RAG context.

## Project Overview

This system implements a complete RAG pipeline that:
- Loads and preprocesses Amazon product data from Kaggle
- Creates vector embeddings for semantic search
- Retrieves relevant product information based on user queries
- Generates answers using three different open-source LLMs
- Evaluates and compares LLM performance across multiple metrics

## Features

- **Multi-Model Support**: Compare Llama-3, Mistral-7B, and Phi-3 models
- **Efficient Retrieval**: FAISS-based vector search for fast similarity matching
- **Quantization Support**: 4-bit and 8-bit quantization for running on consumer hardware
- **Comprehensive Evaluation**: Multiple metrics for assessing answer quality
- **15 Domain-Specific Questions**: Covering various categories and difficulty levels
- **Batch Processing**: Evaluate all questions across all models automatically

## System Architecture

```
┌─────────────────┐
│  Amazon CSV     │
│  Dataset        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Loader &   │
│ Preprocessing   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document        │
│ Chunking        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│ Sentence        │────▶│ FAISS Vector │
│ Transformer     │     │ Index        │
│ Embeddings      │     └──────────────┘
└─────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│          RAG System                 │
│  ┌──────────────┐  ┌──────────────┐│
│  │  Retrieval   │  │  Generation  ││
│  │  (Vector     │  │  (LLM)       ││
│  │   Search)    │  │              ││
│  └──────────────┘  └──────────────┘│
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Multi-LLM Manager               │
│  ┌─────────┐ ┌─────────┐ ┌────────┐│
│  │ Llama-3 │ │Mistral-7│ │ Phi-3  ││
│  │         │ │    B    │ │        ││
│  └─────────┘ └─────────┘ └────────┘│
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Evaluation &   │
│  Comparison     │
└─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- At least 16GB RAM (32GB recommended for all models)
- 20GB free disk space

### Setup Instructions

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd rag
```

2. **Create virtual environment**:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download the dataset**:
   - Download the Amazon Products dataset from [Kaggle](https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset)
   - Place the CSV file in the `data/` directory
   - Update the path in `config.yaml` if needed

## Dataset

**Source**: [Amazon Products Dataset on Kaggle](https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset)

The dataset contains Amazon product information including:
- Product names and descriptions
- Categories
- Prices and ratings
- Customer reviews
- Product specifications

## Configuration

Edit `config.yaml` to customize:
- Dataset paths
- Embedding model selection
- LLM configurations (model names, quantization, generation parameters)
- RAG parameters (top_k, chunk size)
- Evaluation settings

Example configuration:
```yaml
dataset:
  path: "data/amazon_products.csv"

embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

llms:
  llama3:
    model_name: "meta-llama/Llama-3.2-3B-Instruct"
    quantization: "4bit"
  # ... more LLMs
```

## Usage

### Quick Demo

Run a quick demo with a single question using one LLM:

```bash
python demo.py
```

This will:
1. Build the vector index (first run only)
2. Load the Phi-3 model
3. Answer a sample question
4. Display evaluation metrics

### Full Evaluation

Run the complete evaluation with all LLMs and all questions:

```bash
# Build index and run evaluation
python main.py --build-index

# Run with existing index
python main.py

# Use specific LLMs only
python main.py --llms llama3 phi3

# Evaluate first 5 questions
python main.py --num-questions 5
```

### Command Line Options

- `--config PATH`: Path to configuration file (default: `config.yaml`)
- `--build-index`: Build new vector index from dataset
- `--llms [NAMES]`: Specific LLMs to use (e.g., `--llms llama3 mistral`)
- `--output-dir DIR`: Output directory for results (default: `outputs`)
- `--num-questions N`: Number of questions to evaluate

## Evaluation Questions

The system includes 15 domain-specific questions covering:

1. **Product Search**: Finding products by category and rating
2. **Price Filtering**: Budget-based recommendations
3. **Comparison**: Comparing features across products
4. **Sentiment Analysis**: Analyzing reviews and feedback
5. **Recommendations**: Suggesting products based on requirements
6. **Feature Extraction**: Identifying specific product features
7. **Aggregation**: Computing statistics across products
8. **Brand Analysis**: Analyzing brand presence
9. **Educational**: Explaining product concepts
10. **Meta Analysis**: Dataset-level insights

See [src/questions.py](src/questions.py) for the complete list.

## Evaluation Metrics

The system evaluates LLM outputs using:

### Answer Quality Metrics
- **Answer Length**: Word and character count
- **Query Overlap**: Keyword overlap with question
- **Context Overlap**: Grounding in retrieved context
- **Specificity Score**: Presence of numbers, prices, ratings
- **Readability**: Average sentence length

### Comparison Metrics
- **Diversity Score**: Variety in responses across models
- **Agreement Score**: Similarity between model outputs

## Output

Results are saved in the `outputs/` directory:

### 1. Detailed Results JSON
```json
{
  "question": {...},
  "results": {
    "llama3": {"answer": "...", ...},
    "mistral": {"answer": "...", ...},
    "phi3": {"answer": "...", ...}
  },
  "comparison": {
    "llm_metrics": {...},
    "diversity_score": 0.75,
    "agreement_score": 0.62
  }
}
```

### 2. Summary Report
```
RAG SYSTEM EVALUATION SUMMARY
=====================================
Number of questions: 15
LLMs evaluated: llama3, mistral, phi3

llama3 Performance:
  answer_length: 45.2
  query_overlap: 0.68
  ...
```

## Project Structure

```
rag/
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Dataset loading and preprocessing
│   ├── vector_store.py      # Vector embeddings and FAISS index
│   ├── llm_handler.py       # LLM loading and management
│   ├── rag_system.py        # RAG pipeline implementation
│   ├── evaluation.py        # Evaluation metrics
│   └── questions.py         # Evaluation questions
├── data/
│   ├── amazon_products.csv  # Dataset (download separately)
│   └── vector_index/        # Generated vector index
├── outputs/                 # Evaluation results
├── models/                  # Cached model files
├── config.yaml             # System configuration
├── requirements.txt        # Python dependencies
├── main.py                 # Main evaluation script
├── demo.py                 # Quick demo script
└── README.md              # This file
```

## Models Used

### 1. Llama-3 (Meta)
- **Model**: `meta-llama/Llama-3.2-3B-Instruct`
- **Size**: 3B parameters
- **Strengths**: General-purpose, strong reasoning

### 2. Mistral-7B (Mistral AI)
- **Model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Size**: 7B parameters
- **Strengths**: Efficient, good instruction following

### 3. Phi-3 (Microsoft)
- **Model**: `microsoft/Phi-3-mini-4k-instruct`
- **Size**: 3.8B parameters
- **Strengths**: Small size, fast inference

All models are quantized to 4-bit for efficient inference on consumer hardware.

## Hardware Requirements

### Minimum Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 16GB
- **Storage**: 20GB free space
- **GPU**: Optional but recommended

### Recommended Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **RAM**: 32GB
- **Storage**: SSD with 50GB free space

### Running Without GPU
The system can run on CPU only, but will be significantly slower. Update `config.yaml`:
```yaml
embedding:
  device: "cpu"
```

## Performance Tips

1. **Use 4-bit Quantization**: Already configured by default
2. **Reduce Batch Size**: Lower `batch_size` if running out of memory
3. **Limit Retrieved Documents**: Reduce `top_k` in config
4. **Test with Smaller Model First**: Start with Phi-3 only
5. **Use SSD**: Store index on SSD for faster loading

## Troubleshooting

### Out of Memory Errors
- Reduce batch size in config
- Use 4-bit quantization (default)
- Load one LLM at a time
- Close other applications

### Slow Inference
- Enable GPU if available
- Use smaller models (Phi-3)
- Reduce `max_new_tokens` in config
- Use 4-bit instead of 8-bit quantization

### Missing Dataset
- Download from Kaggle link above
- Place in `data/` directory
- Update path in `config.yaml`

## Future Improvements

- [ ] Add more embedding models comparison
- [ ] Implement reranking for improved retrieval
- [ ] Add support for multilingual queries
- [ ] Include more evaluation metrics (BLEU, ROUGE, etc.)
- [ ] Add web interface for interactive queries
- [ ] Implement caching for faster repeated queries
- [ ] Add support for custom datasets

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational purposes as part of DSCI 6004 course requirements.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{amazon_rag_2025,
  title={Amazon Product RAG System with Multi-LLM Comparison},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={your-repo-url}
}
```

## Acknowledgments

- Dataset: [Lokesh Parab's Amazon Products Dataset](https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset)
- Models: Meta (Llama-3), Mistral AI (Mistral-7B), Microsoft (Phi-3)
- Frameworks: HuggingFace Transformers, Sentence Transformers, FAISS

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email].

---

**Note**: This is a term project for DSCI 6004: Natural Language Processing. The code demo is due 11/30/2025 and final submission is due 12/07/2025.
