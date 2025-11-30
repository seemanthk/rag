# ShopSmart RAG - Interactive Web Interface Guide

## 🌟 Streamlit Application Overview

The Streamlit web application provides an intuitive, interactive interface for running RAG evaluations without using the command line.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

This includes:
- `streamlit>=1.28.0` - Web framework
- `plotly>=5.17.0` - Interactive charts
- All other RAG system dependencies

### 2. Launch the App
```bash
streamlit run streamlit_app.py
```

### 3. Open Browser
The app will automatically open at `http://localhost:8501`

---

## 📱 Application Pages

### 🏠 Page 1: Home
**Purpose**: System overview and status check

**Features**:
- Quick system status (Ollama, models, dataset)
- Project overview
- Configuration summary

**What to Check**:
- ✅ Ollama is running
- ✅ Models are available (phi3:mini, llama3.2:latest, gemma2:2b)
- ✅ Dataset is accessible

---

### 📊 Page 2: Data Upload
**Purpose**: Load product dataset

**Options**:

1. **Use Existing Dataset**
   - Default: `data/Air Conditioners.csv`
   - Click "Use Existing Dataset" button
   - Preview shows 50 Air Conditioner products

2. **Upload Custom CSV**
   - Click "Upload Product CSV"
   - Supported format: CSV with columns like `product_name`, `actual_price`, `ratings`, etc.
   - Automatic validation and preview

**What You'll See**:
- Dataset statistics (number of products, columns)
- Data preview table
- Column names and types

---

### ❓ Page 3: Questions Setup
**Purpose**: Define evaluation questions

**Options**:

1. **Use Predefined Questions** (Recommended)
   - Click "Use Predefined 15 Questions"
   - Automatically loads Air Conditioner-specific questions
   - Categories: Structured, Value Reasoning, Temporal, Combined

2. **Upload Questions File**
   - **JSON Format**:
     ```json
     [
       {
         "id": 1,
         "question": "What are the best rated ACs under ₹40,000?",
         "category": "structured_query",
         "difficulty": "easy"
       }
     ]
     ```
   - **TXT Format** (one question per line):
     ```
     What are the best rated ACs under ₹40,000?
     Compare 1 ton vs 1.5 ton ACs
     ```

3. **Manual Entry**
   - Enter questions in text area (one per line)
   - Automatically assigns IDs and categories

**What You'll See**:
- Number of loaded questions
- Preview of all questions
- Category distribution

---

### 🚀 Page 4: Run Analysis
**Purpose**: Execute RAG evaluation

**Steps**:

1. **Select Models**
   - Choose which models to compare:
     - ☐ phi3:mini (Fast, efficient)
     - ☐ llama3.2:latest (Balanced)
     - ☐ gemma2:2b (Google's latest)
   - Can select 1, 2, or all 3 models

2. **Click "Run Evaluation"**

3. **Watch Progress**
   - Progress bar shows completion percentage
   - Status updates:
     - 📊 Loading data...
     - 🔍 Building vector index...
     - 🤖 Loading models...
     - 📝 Evaluating questions...
   - Estimated time: 10-15 minutes for 15 questions × 3 models

**What Happens**:
- Loads product data and builds FAISS index
- Initializes selected Ollama models
- For each question:
  - Retrieves relevant products
  - Generates answers from each model
  - Calculates 7 evaluation metrics
- Stores results in session state

---

### 📈 Page 5: Results
**Purpose**: View and export evaluation results

**Sections**:

1. **Summary Cards**
   - Total questions evaluated
   - Models compared
   - Average answer length
   - Average factual accuracy

2. **Interactive Charts**

   **Bar Chart: Model Comparison**
   - X-axis: Metrics (Query Overlap, Factual Accuracy, Context Precision, etc.)
   - Y-axis: Score (0-100%)
   - One bar per model
   - Hover for exact values

   **Radar Chart: Overall Performance**
   - Pentagon/hexagon showing all metrics
   - One trace per model
   - Easy visual comparison of strengths/weaknesses

3. **Detailed Metrics Table**
   - Rows: Each model
   - Columns: All 7 metrics
   - Color-coded (green = good, red = poor)

4. **Question-by-Question Results**
   - Expandable sections for each question
   - Shows:
     - Original question
     - Category and difficulty
     - All model answers side-by-side
     - Metrics for each answer
   - Scroll through all 15 questions

5. **Export Options**
   - **Download JSON**: Raw results with all data
   - **Download CSV**: Summary table (models × metrics)
   - **View Raw Data**: Expand to see full JSON structure

**Metrics Explained**:
- **Query Overlap** (0-100%): How well answer addresses the question
- **Context Overlap** (0-100%): Answer grounded in retrieved documents
- **Context Precision** (0-100%): Quality of retrieved documents
- **Factual Accuracy** (0-100%): Correctness of prices/ratings mentioned
- **Completeness** (0-100%): Question fully answered
- **Specificity** (0-100%): Presence of concrete details
- **Answer Length** (words): Response length

---

## 🎯 Complete Workflow Example

### Scenario: Evaluate Air Conditioner RAG System

1. **Start Application**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Home Page**
   - Check status → All green ✅
   - Navigate to "📊 Data Upload"

3. **Data Upload**
   - Click "Use Existing Dataset"
   - See preview: 50 products loaded
   - Navigate to "❓ Questions Setup"

4. **Questions Setup**
   - Click "Use Predefined 15 Questions"
   - Preview shows all questions
   - Navigate to "🚀 Run Analysis"

5. **Run Analysis**
   - Select all 3 models: ☑ phi3, ☑ llama3, ☑ gemma2
   - Click "Run Evaluation"
   - Wait 10-15 minutes (watch progress bar)
   - Success message appears
   - Navigate to "📈 Results"

6. **View Results**
   - Check summary: 15 questions, 3 models
   - Examine bar chart: llama3 has highest factual accuracy
   - View radar chart: gemma2 balanced, phi3 fastest
   - Scroll through question details
   - Download JSON results for report

7. **Export for Report**
   - Click "Download JSON Results"
   - Use in academic paper or presentation
   - Click "Download CSV Summary"
   - Import to Excel for tables

---

## 🔧 Troubleshooting

### Issue: "Ollama is not running"
**Solution**:
```bash
# In separate terminal
ollama serve
```
Refresh the Home page

### Issue: "Models not available"
**Solution**:
```bash
# Check installed models
ollama list

# Pull missing models
ollama pull phi3:mini
ollama pull llama3.2:latest
ollama pull gemma2:2b
```

### Issue: "No dataset found"
**Solution**:
- Ensure `data/Air Conditioners.csv` exists
- Or upload a custom CSV with proper format

### Issue: Charts not rendering
**Solution**:
- Ensure evaluation completed successfully
- Check browser console for errors
- Try refreshing the Results page

### Issue: Evaluation very slow
**Cause**: Running 15 questions × 3 models = 45 LLM calls
**Solutions**:
- Run fewer models (select only 1 or 2)
- Use fewer questions (edit in Questions Setup)
- Expected time: ~15 seconds per question per model

### Issue: Out of memory
**Solution**:
- Close other applications
- Run one model at a time
- Reduce batch size in config.yaml

---

## 💡 Tips and Best Practices

### For Best Results

1. **Start Small**
   - Test with 1 model and 3 questions first
   - Verify everything works
   - Then run full 15 questions × 3 models

2. **Monitor Progress**
   - Watch status messages
   - Don't close browser during evaluation
   - Streamlit maintains session state

3. **Custom Datasets**
   - CSV must have: product names, prices, ratings
   - More products = better retrieval quality
   - Recommended: 30+ products minimum

4. **Custom Questions**
   - Mix question types (comparison, value, factual)
   - Include price ranges for factual accuracy testing
   - Use domain-specific terms

5. **Interpreting Results**
   - **High Query Overlap** = Relevant answers
   - **High Factual Accuracy** = Correct prices/ratings
   - **High Context Precision** = Good retrieval
   - **High Completeness** = Thorough answers

### For Presentations

1. **Screenshots**
   - Home page showing status
   - Data preview
   - Charts (bar and radar)
   - Detailed question results

2. **Live Demo**
   - Prepare by running evaluation beforehand
   - Navigate to Results page
   - Show interactive chart features (hover, zoom)
   - Highlight best-performing model

3. **Export Data**
   - Download JSON for technical details
   - Download CSV for summary tables
   - Use in papers/slides

---

## 📊 Sample Question Files

### JSON Format (`questions.json`)
```json
[
  {
    "id": 1,
    "question": "What are the top 5 rated air conditioners under ₹40,000?",
    "category": "structured_query",
    "difficulty": "easy",
    "type": "factual"
  },
  {
    "id": 2,
    "question": "Compare 1 ton vs 1.5 ton ACs for small rooms",
    "category": "value_reasoning",
    "difficulty": "medium",
    "type": "comparison"
  }
]
```

### TXT Format (`questions.txt`)
```
What are the best rated ACs under ₹40,000?
Compare 1 ton vs 1.5 ton ACs
Which AC has the best energy efficiency rating?
Find budget-friendly ACs with 4+ star ratings
What's the price range for inverter ACs?
```

---

## 🎓 For Academic Project Submission

### What to Submit

1. **Code**
   - `streamlit_app.py`
   - All `src/` files
   - `config.yaml`
   - `requirements.txt`

2. **Documentation**
   - This file (STREAMLIT_GUIDE.md)
   - FINAL_PROJECT_GUIDE.md
   - README_OLLAMA.md

3. **Results**
   - Sample `evaluation_*.json` from outputs/
   - Screenshots of Streamlit UI
   - Charts (exported as images)

4. **Demo**
   - Record screen while using Streamlit app
   - Show: Data upload → Run analysis → View results
   - Highlight interactive features

### Video Demo Script (5 min)

```
[0:00-0:30] Introduction
"I'll demonstrate our RAG evaluation system using the Streamlit interface"

[0:30-1:30] System Setup
- Show Home page with status checks
- Navigate to Data Upload
- Load Air Conditioners dataset
- Preview data

[1:30-2:30] Questions and Execution
- Navigate to Questions Setup
- Load predefined 15 questions
- Navigate to Run Analysis
- Select all 3 models
- Start evaluation
- Show progress bar

[2:30-4:00] Results Analysis
- Navigate to Results page
- Explain summary cards
- Interact with bar chart (hover, zoom)
- Explain radar chart
- Scroll through question details

[4:00-5:00] Conclusion
- Export results (JSON and CSV)
- Highlight key findings
- Summary of metrics
```

---

## 🔗 Related Files

- **Main evaluation script**: `run_complete_evaluation.py` (command line alternative)
- **Configuration**: `config.yaml`
- **Questions**: `src/questions.py` (predefined questions)
- **Evaluation logic**: `src/evaluation.py` (7 metrics implementation)
- **Visualization**: `src/html_visualizer.py` (HTML dashboard alternative)

---

## ✅ Success Checklist

Before final submission, verify:

- [ ] Streamlit app launches without errors
- [ ] All 5 pages load correctly
- [ ] Can upload CSV and questions
- [ ] Evaluation runs and completes
- [ ] Charts render with real data
- [ ] Can export JSON and CSV
- [ ] Screenshots captured
- [ ] Demo video recorded

---

**Happy Evaluating! 🎉**

For issues or questions, refer to:
- FINAL_PROJECT_GUIDE.md (complete project guide)
- SETUP_GUIDE.md (detailed setup)
- README_OLLAMA.md (project overview)
