# Quick Start Guide

Get up and running with the Amazon Product RAG System in 5 minutes!

## Prerequisites

- Python 3.8+
- 16GB RAM minimum
- 20GB free disk space
- (Optional) CUDA-compatible GPU

## Installation

### 1. Clone and Setup

```bash
cd rag
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download Dataset

**Option A: Manual Download**
1. Go to https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset
2. Download the CSV file
3. Place it in the `data/` folder
4. Update `config.yaml` with the filename

**Option B: Using Kaggle CLI** (Recommended)
```bash
# Install kaggle and setup credentials first
pip install kaggle

# Download dataset
python download_dataset.py
```

### 3. Run Demo

```bash
python demo.py
```

This will:
- Build the vector index (first run only, ~5-10 minutes)
- Load the Phi-3 model
- Answer a sample question
- Display evaluation metrics

## What the Demo Does

The demo script will:

1. **Load Data**: Read the Amazon products CSV
2. **Build Index**: Create vector embeddings and FAISS index (saved for reuse)
3. **Load Model**: Download and load Phi-3 (smallest/fastest model)
4. **Answer Question**: Process a sample question about products
5. **Show Results**: Display retrieved context and generated answer

### Expected Output

```
================================================================================
AMAZON PRODUCT RAG SYSTEM - DEMO
================================================================================

Loading existing vector index...
Vector index loaded!

Loading LLM (this may take a minute)...
Using Phi-3 model for demo (smallest/fastest)...

================================================================================
Demo Question: What are the top-rated electronics products?
================================================================================

Retrieving relevant documents...

Retrieved Context:
[Document 1] Score: 0.845
Product: Sony WH-1000XM4 Wireless Headphones | Rating: 4.7 stars...

Generated Answer:
Based on the retrieved product information, the top-rated electronics include...

Evaluation Metrics:
  answer_length: 42.000
  query_overlap: 0.714
  context_overlap: 0.823
  specificity_score: 0.750
```

## Next Steps

### Run Full Evaluation

Evaluate all 15 questions with all 3 models:

```bash
python main.py --build-index
```

**Note**: This will take 30-60 minutes depending on your hardware.

### Use Specific Models

```bash
# Use only Llama-3 and Phi-3
python main.py --llms llama3 phi3

# Test with first 5 questions
python main.py --num-questions 5
```

### Interactive Exploration

Use the Jupyter notebook:

```bash
jupyter notebook notebooks/rag_exploration.ipynb
```

## Common Issues

### Out of Memory

**Solution**: Edit `config.yaml` to reduce batch size:
```yaml
embedding:
  batch_size: 16  # reduce from 32
```

### Slow Performance

**Solutions**:
1. Use GPU if available (set `device: "cuda"` in config)
2. Start with Phi-3 only (smallest model)
3. Reduce `max_new_tokens` in config

### Dataset Not Found

Make sure:
1. CSV file is in `data/` directory
2. Path in `config.yaml` matches the actual filename
3. File is not corrupted (check file size)

### Model Download Fails

- Check internet connection
- Try again (downloads are cached)
- Use a different model if one fails

## Project Structure

```
rag/
├── src/              # Core modules
├── data/             # Dataset (you add)
├── outputs/          # Results (generated)
├── config.yaml       # Configuration
├── demo.py          # Quick demo
├── main.py          # Full evaluation
└── README.md        # Full documentation
```

## Configuration

Key settings in `config.yaml`:

```yaml
dataset:
  path: "data/amazon_products.csv"  # Update this!

embedding:
  batch_size: 32      # Reduce if OOM
  device: "cuda"      # or "cpu"

rag:
  top_k: 5           # Retrieved documents

llms:
  phi3:
    quantization: "4bit"  # or "8bit"
    max_new_tokens: 512
```

## Customization

### Add Your Own Questions

Edit `src/questions.py`:

```python
EVALUATION_QUESTIONS.append({
    "id": 16,
    "question": "Your question here?",
    "category": "custom",
    "difficulty": "medium"
})
```

### Change Models

Update `config.yaml`:

```yaml
llms:
  custom_model:
    model_name: "your/model/name"
    quantization: "4bit"
```

### Adjust Retrieval

```yaml
rag:
  top_k: 10          # More context
  chunk_size: 1024   # Larger chunks
```

## Getting Help

1. Check [README.md](README.md) for detailed docs
2. Review error messages carefully
3. Check system requirements
4. Verify dataset is downloaded correctly

## Timeline (Project Milestones)

- **Now**: Setup and testing
- **Code Demo**: Due 11/30/2025
- **Final Submission**: Due 12/07/2025

## What to Submit

1. **GitHub Repository** with:
   - All source code
   - This README
   - Configuration files
   - Results in `outputs/`

2. **Video Demo** (< 10 minutes):
   - Running the code
   - Showing results
   - Explaining implementation

3. **Final Report** (5-8 pages):
   - Implementation details
   - Results and analysis
   - LLM comparison
   - References

---

**Happy coding! 🚀**

For detailed documentation, see [README.md](README.md)
