# ShopSmart RAG - Final Project Guide
## Complete Air Conditioner Analysis System with Ollama

---

## 🎯 Project Overview

**Course**: DSCI 6004 - Natural Language Processing
**Project**: RAG System Development and LLM Comparison
**Dataset**: Air Conditioners from Amazon India (50 products)
**Models**: phi3:mini, llama3.2:latest, gemma2:2b (via Ollama)

This system performs intelligent product analysis using Retrieval-Augmented Generation with **comprehensive evaluation metrics** and **interactive HTML dashboards**.

---

## ✅ Project Deliverables Completed

### 1. ✅ RAG System Implementation
- Vector-based retrieval using FAISS
- Semantic search with sentence-transformers
- Three-model comparison architecture

### 2. ✅ Three LLM Models (Ollama)
- **phi3:mini** (2.2GB) - Fast, efficient
- **llama3.2:latest** (2.0GB) - Balanced performance
- **gemma2:2b** (1.6GB) - Google's latest

### 3. ✅ 15 Domain-Specific Questions
Organized into 4 categories:
- **Structured Queries** (4 questions)
- **Value Reasoning** (5 questions)
- **Temporal Analysis** (3 questions)
- **Combined Analysis** (3 questions)

### 4. ✅ Comprehensive Evaluation Metrics
- **Answer Relevancy** - Query overlap analysis
- **Faithfulness** - Context grounding
- **Context Precision** - Retrieval quality
- **Factual Accuracy** - Price/rating verification
- **Completeness** - Question-type specific analysis
- **Specificity** - Presence of concrete details
- **Response Time** - Generation speed

### 5. ✅ HTML Visualization Dashboard
- Interactive charts (Chart.js)
- Model comparison cards
- Detailed question-by-question results
- Summary statistics
- Export-ready format

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Ollama
```bash
# Already done - Ollama is running
ollama serve
```

### Step 2: Verify Models
```bash
ollama list
# Should show: phi3:mini, llama3.2:latest, gemma2:2b
```

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 4A: Run Complete Evaluation (Command Line)
```bash
python run_complete_evaluation.py
```

This will:
1. ✅ Load Air Conditioners dataset (50 products)
2. ✅ Build/load vector index
3. ✅ Load all 3 Ollama models
4. ✅ Evaluate all 15 questions
5. ✅ Compute comprehensive metrics
6. ✅ Generate HTML dashboard
7. ✅ Save JSON results

**Output**: `outputs/dashboard_YYYYMMDD_HHMMSS.html`

### Step 4B: Run Interactive Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```

This launches an interactive web application where you can:
1. 📊 **Upload CSV**: Upload your own product dataset or use existing
2. ❓ **Upload Questions**: Upload questions (JSON/TXT) or use predefined 15 questions
3. 🚀 **Run Analysis**: Execute evaluation with real-time progress tracking
4. 📈 **View Results**: Interactive Plotly charts and detailed metrics
5. 💾 **Export**: Download JSON results or CSV summary

**Access**: Open browser to `http://localhost:8501`

---

## 📊 Evaluation Metrics Explained

### 1. Answer Relevancy (query_overlap)
- **What**: Measures how well answer addresses the question
- **How**: Calculates keyword overlap between query and answer
- **Range**: 0.0 to 1.0 (higher = more relevant)

### 2. Faithfulness (context_overlap)
- **What**: Measures answer grounding in retrieved context
- **How**: Checks if answer content comes from retrieved documents
- **Range**: 0.0 to 1.0 (higher = more faithful)

### 3. Context Precision
- **What**: Quality of retrieved documents
- **How**: Average relevance score of top-K retrieved docs
- **Range**: 0.0 to 1.0 (higher = better retrieval)

### 4. Factual Accuracy
- **What**: Correctness of prices and ratings mentioned
- **How**: Validates mentioned prices/ratings against source documents
- **Range**: 0.0 to 1.0 (1.0 = all facts correct)
- **Example**: If answer says "₹35,000" but doc says "₹32,999", scores lower

### 5. Completeness
- **What**: Does answer fully address the question type?
- **How**: Question-type specific checks:
  - Comparison questions → Should have "vs", "while", "whereas"
  - Value questions → Should have reasoning keywords
  - Price questions → Should mention specific prices
- **Range**: 0.0 to 1.0

### 6. Specificity Score
- **What**: Presence of concrete details
- **How**: Checks for numbers, prices, percentages, ratings
- **Range**: 0.0 to 1.0

---

## 🌟 Interactive Streamlit Web Application

The project includes a complete web-based interface (`streamlit_app.py`) for easy interaction without command line usage.

### Features
1. **🏠 Home Page**
   - System status check (Ollama, models, data)
   - Quick overview of capabilities
   - Configuration summary

2. **📊 Data Upload Page**
   - Upload custom CSV files or use existing Air Conditioners dataset
   - Preview loaded data with statistics
   - Automatic validation and preprocessing

3. **❓ Questions Setup Page**
   - Upload questions from JSON or TXT file
   - Use predefined 15 evaluation questions
   - Manually enter custom questions
   - Preview and edit question list

4. **🚀 Run Analysis Page**
   - Select models to compare (phi3, llama3, gemma2)
   - Real-time progress tracking with status updates
   - Live progress bar showing evaluation status
   - Automatic metric calculation

5. **📈 Results Page**
   - Interactive Plotly charts:
     - Bar chart: Model comparison across all metrics
     - Radar chart: Overall performance visualization
   - Detailed metrics table for each model
   - Question-by-question breakdown
   - Export options:
     - Download JSON results
     - Download CSV summary
     - View raw data

### How to Use
```bash
# Launch the web interface
streamlit run streamlit_app.py

# Open browser to http://localhost:8501
```

**Workflow**:
1. Navigate through pages using sidebar
2. Upload CSV (or use existing data/Air Conditioners.csv)
3. Upload questions (or use predefined ones)
4. Run analysis and watch progress
5. View interactive charts and export results

---

## 📈 Understanding the HTML Dashboard

### Summary Section
- **Total Questions**: Number of questions evaluated
- **Models Compared**: Number of LLMs used
- **Avg Answer Length**: Average words per answer across all models
- **Avg Factual Accuracy**: Overall accuracy percentage

### Model Performance Cards
Each model shows:
- **Query Overlap**: Answer relevancy
- **Context Overlap**: Faithfulness to context
- **Context Precision**: Retrieval quality
- **Factual Accuracy**: Price/rating correctness
- **Completeness**: Question coverage
- **Answer Length**: Average words

### Charts
1. **Answer Relevancy** - Bar chart comparing query overlap
2. **Factual Accuracy** - How well each model cites correct facts
3. **Context Precision** - Retrieval quality comparison
4. **Overall Performance** - Radar chart showing all metrics

### Detailed Results
- Question-by-question breakdown
- All three model answers side-by-side
- Metrics for each answer
- Category and difficulty tags

---

## 📁 Final Project Structure

```
rag/
├── config.yaml                      # System configuration
├── requirements.txt                 # Python dependencies
│
├── FINAL_PROJECT_GUIDE.md          # This file - Complete guide
├── SETUP_GUIDE.md                  # Detailed setup instructions
├── README_OLLAMA.md                # Project overview
│
├── run_complete_evaluation.py      # 🎯 MAIN SCRIPT - Command line evaluation
├── streamlit_app.py                # 🌟 INTERACTIVE WEB UI - Upload CSV & Questions
├── main_ollama.py                  # CLI tool for queries
├── quick_start.py                  # System verification
│
├── data/
│   ├── Air Conditioners.csv        # Dataset (50 products)
│   └── vector_index/               # FAISS index
│
├── src/
│   ├── data_loader.py              # CSV loading & preprocessing
│   ├── vector_store.py             # FAISS vector database
│   ├── ollama_handler.py           # Ollama LLM integration
│   ├── rag_system.py               # RAG pipeline
│   ├── evaluation.py               # ✨ Enhanced metrics
│   ├── html_visualizer.py          # ✨ Dashboard generator
│   └── questions.py                # 15 evaluation questions
│
├── notebooks/
│   └── rag_ollama_demo.ipynb      # Interactive exploration
│
└── outputs/                        # Generated results
    ├── evaluation_*.json           # Raw results
    └── dashboard_*.html            # 📊 Interactive dashboard
```

### Files to Keep
- ✅ `run_complete_evaluation.py` - Main evaluation script (command line)
- ✅ `streamlit_app.py` - Interactive web interface (recommended)
- ✅ `main_ollama.py` - CLI tool
- ✅ `quick_start.py` - System check
- ✅ All files in `src/` directory
- ✅ `config.yaml`
- ✅ Documentation files (`.md`)

### Files to Remove (Old/Unused)
- ❌ `main.py` (old HuggingFace version)
- ❌ `demo.py` (replaced by run_complete_evaluation.py)
- ❌ `download_dataset.py` (not needed)
- ❌ `check_system.py` (replaced by quick_start.py)
- ❌ `quick_test.py` (obsolete)
- ❌ `setup.py` (not needed)
- ❌ `src/llm_handler.py` (old HuggingFace handler)

---

## 🎓 Meeting Project Requirements

### Requirement Checklist

| Requirement | Status | Evidence |
|------------|--------|----------|
| RAG System Development | ✅ | `src/rag_system.py`, `src/vector_store.py` |
| Three Free/Open-Source LLMs | ✅ | Ollama: phi3:mini, llama3.2, gemma2:2b |
| 10+ Domain-Specific Questions | ✅ | 15 questions in `src/questions.py` |
| LLM Response Evaluation | ✅ | Enhanced `src/evaluation.py` with 7 metrics |
| Performance Comparison | ✅ | Side-by-side in HTML dashboard |
| Accuracy Analysis | ✅ | Factual accuracy metric validates prices/ratings |
| Reasoning Analysis | ✅ | Completeness metric checks reasoning quality |
| Working Code | ✅ | All scripts functional |
| Documentation | ✅ | 3 comprehensive .md files |
| Results Visualization | ✅ | Interactive HTML dashboard |

---

## 🎬 Demo Presentation Guide

### For Code Demo (11/30/2025)

**1. Show System Setup (2 min)**
```bash
# Verify Ollama
ollama list

# Show config
cat config.yaml

# Show dataset
head data/Air\ Conditioners.csv
```

**2. Run Quick Query (2 min)**
```bash
python main_ollama.py --query "Best 1.5 ton AC under ₹35,000?" --models phi3
```

**3. Run Full Evaluation (3 min)**
```bash
python run_complete_evaluation.py
```

**4. Show HTML Dashboard (3 min)**
- Open generated `outputs/dashboard_*.html`
- Highlight:
  - Summary statistics
  - Model comparison cards
  - Factual accuracy chart
  - Detailed question results

### Video Script (< 10 min)

```
[0:00-1:00] Introduction
- ShopSmart RAG system for Air Conditioner analysis
- Using Ollama (phi3, llama3, gemma2) - all local, no API costs
- 50 products, 15 questions, comprehensive metrics

[1:00-3:00] System Architecture
- Show code structure
- Explain RAG pipeline: retrieval → context → generation
- Point to enhanced evaluation.py with 7 metrics

[3:00-5:00] Live Demo
- Run complete evaluation
- Show progress logs
- Open HTML dashboard

[5:00-7:00] Results Analysis
- Navigate dashboard
- Compare model performance
- Highlight factual accuracy feature
- Show specific question examples

[7:00-9:00] Technical Highlights
- Factual accuracy: validates prices/ratings
- Completeness: question-type aware
- Context precision: retrieval quality
- All metrics explained in dashboard

[9:00-10:00] Conclusion
- Deliverables met
- Ready for final submission
- Future enhancements possible
```

---

## 📊 Sample Results

Based on the Air Conditioner dataset (50 products):

### Expected Metrics (Approximate)
- **Query Overlap**: 0.40-0.70 (40-70% keyword match)
- **Context Overlap**: 0.50-0.80 (answers grounded in context)
- **Context Precision**: 0.70-0.90 (FAISS retrieves relevant docs)
- **Factual Accuracy**: 0.60-0.90 (most prices/ratings correct)
- **Completeness**: 0.50-0.90 (varies by question type)

### Model Comparison (Expected)
- **llama3.2**: Likely best overall (most balanced)
- **phi3:mini**: Fastest, shorter answers
- **gemma2:2b**: Good accuracy, newer model

---

## 🐛 Troubleshooting

### Issue: Models not found
```bash
# Check exact model names
ollama list

# Update config.yaml with exact names
# phi3:mini NOT phi3
# llama3.2:latest NOT llama3
```

### Issue: Evaluation fails
```bash
# Check if vector index exists
ls -la data/vector_index/

# Rebuild if needed
python main_ollama.py --build-index
```

### Issue: Out of memory
- Run evaluation with fewer questions:
  ```python
  # Edit run_complete_evaluation.py line 61
  questions = questions[:5]  # Only first 5 questions
  ```

---

## 📝 Final Submission Checklist

For 12/07/2025 submission:

### Code
- [ ] `run_complete_evaluation.py` - Main script
- [ ] All `src/` modules
- [ ] `config.yaml`
- [ ] `requirements.txt`

### Documentation
- [ ] This file (FINAL_PROJECT_GUIDE.md)
- [ ] SETUP_GUIDE.md
- [ ] README_OLLAMA.md

### Results
- [ ] Sample `outputs/evaluation_*.json`
- [ ] Sample `outputs/dashboard_*.html`
- [ ] Screenshots of dashboard

### Report (ACL/NeurIPS style paper - 8 pages)
Structure:
1. **Abstract** - RAG system for e-commerce with factual accuracy
2. **Introduction** - Problem, approach, contributions
3. **System Architecture** - RAG pipeline, Ollama integration
4. **Evaluation Methodology** - 7 metrics explained
5. **Results** - Model comparison, metric analysis
6. **Discussion** - Findings, strengths/weaknesses per model
7. **Conclusion** - Summary, future work
8. **References** - Cite RAG papers, Ollama, models

### GitHub Repository
- [ ] Clean code (remove old files)
- [ ] README.md (use README_OLLAMA.md)
- [ ] Usage instructions
- [ ] Sample outputs

---

## 🚀 Running the Final Evaluation

### Complete Command Sequence

```bash
# 1. Verify setup
python quick_start.py

# 2. Build index (first time only)
python main_ollama.py --build-index

# 3. Test single query
python main_ollama.py --query "Best budget AC?" --models phi3

# 4. Run full evaluation (THIS IS THE MAIN ONE!)
python run_complete_evaluation.py

# 5. Open results
# outputs/dashboard_TIMESTAMP.html in your browser
```

### Expected Runtime
- Build index: ~2-3 minutes (first time only)
- Single query: ~5-10 seconds per model
- Full evaluation (15 questions × 3 models): ~10-15 minutes
- HTML generation: ~5 seconds

---

## 🎉 Success Criteria

Your project is complete when:

1. ✅ `python run_complete_evaluation.py` runs without errors
2. ✅ HTML dashboard opens and displays all metrics
3. ✅ All 3 models show results for all 15 questions
4. ✅ Factual accuracy metric shows percentage > 0
5. ✅ Charts render correctly in dashboard
6. ✅ JSON file contains all evaluation data

---

## 📧 Support

If you encounter issues:

1. Check `quick_start.py` output
2. Verify Ollama is running: `ollama list`
3. Check `config.yaml` has correct model names
4. Review error logs in console
5. See SETUP_GUIDE.md for detailed troubleshooting

---

**Project Complete! 🎊**

You now have a fully functional RAG system with:
- ✅ 3 LLM comparison
- ✅ 7 comprehensive metrics
- ✅ Interactive HTML visualization
- ✅ All project requirements met

**Next**: Record demo video and prepare final report!
