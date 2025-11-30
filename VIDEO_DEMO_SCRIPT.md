# Video Demo Script - ShopSmart RAG Evaluation System
## For DSCI 6004 Project Submission

---

## 📹 Video Requirements
- **Duration**: 8-10 minutes (maximum 10 minutes)
- **Format**: MP4, AVI, or MOV
- **Resolution**: 1080p recommended (minimum 720p)
- **Audio**: Clear narration explaining each step
- **Tools**: OBS Studio, Loom, or built-in screen recorder

---

## 🎬 Pre-Recording Checklist

### Before You Start Recording:

- [ ] **Ollama Running**: `ollama serve` in terminal
- [ ] **Models Installed**: phi3:mini, llama3.2:latest, gemma2:2b
- [ ] **Dataset Ready**: `data/Air Conditioners.csv` exists (50 products)
- [ ] **Dependencies Installed**: `pip install -r requirements.txt`
- [ ] **Streamlit Working**: Test with `streamlit run streamlit_app.py`
- [ ] **Close Unnecessary Apps**: Clean desktop, close extra tabs
- [ ] **Prepare Script**: Print this document for reference
- [ ] **Test Audio**: Check microphone quality
- [ ] **Browser Ready**: Clear cache, maximize window

### Optional Pre-Run (Recommended):
- [ ] Run evaluation once beforehand to ensure everything works
- [ ] Take note of interesting results to highlight
- [ ] Prepare any custom questions you want to demonstrate

---

## 🎯 Video Structure (10 Minutes)

### **[0:00 - 1:00] Introduction** (60 seconds)

**What to Show**:
- Your name and student ID
- Course: DSCI 6004 - Natural Language Processing
- Project title slide or desktop

**What to Say**:
```
"Hello, I'm [Your Name], and this is my DSCI 6004 final project demonstration.

I've built an interactive RAG (Retrieval-Augmented Generation) system called
ShopSmart that analyzes Air Conditioner products from Amazon India.

The system compares three open-source LLM models - Phi-3, Llama 3.2, and Gemma 2 -
using comprehensive evaluation metrics including factual accuracy validation.

I'll demonstrate the complete workflow using our Streamlit web interface."
```

**Screen Actions**:
- Show project folder structure briefly
- Show README or FINAL_PROJECT_GUIDE.md
- Transition to terminal

---

### **[1:00 - 2:00] System Setup** (60 seconds)

**What to Show**:
- Terminal/command prompt
- Ollama status check
- Launch Streamlit app

**What to Say**:
```
"First, let me verify our system setup. I'm using Ollama, which allows us to run
large language models locally without API costs or data privacy concerns."

[Run: ollama list]

"As you can see, we have three models installed:
- Phi-3 Mini: Fast and efficient at 2.2GB
- Llama 3.2: Balanced performance at 2GB
- Gemma 2: Google's latest at 1.6GB

Now I'll launch the Streamlit web interface."

[Run: streamlit run streamlit_app.py]

"The application opens in our browser automatically."
```

**Screen Actions**:
```bash
# Show in terminal
ollama list

# Launch Streamlit
streamlit run streamlit_app.py

# Browser opens to http://localhost:8501
```

---

### **[2:00 - 3:00] Home Page & System Status** (60 seconds)

**What to Show**:
- Streamlit Home page
- System status indicators
- Project overview

**What to Say**:
```
"The application has five main pages accessible from the sidebar.

On the Home page, we can see the system status:
- Ollama is running [point to green checkmark]
- All three models are available [point to list]
- Our Air Conditioner dataset is loaded with 50 products

The system uses FAISS for vector similarity search and implements seven
comprehensive evaluation metrics including answer relevancy, faithfulness,
context precision, and most importantly, factual accuracy which validates
that prices and ratings mentioned in answers are correct."
```

**Screen Actions**:
- Navigate through Home page
- Point to status indicators
- Highlight key features in sidebar
- Briefly show config summary

---

### **[3:00 - 4:00] Data Upload & Preview** (60 seconds)

**What to Show**:
- Data Upload page
- Dataset preview
- Statistics

**What to Say**:
```
"Moving to the Data Upload page, I'll load our Air Conditioner dataset.

[Click 'Use Existing Dataset']

The system supports custom CSV uploads, but we'll use the provided dataset
of 50 Air Conditioners from Amazon India.

[Show preview table]

As you can see, each product has:
- Product name and brand
- Actual price and discounted price
- Customer ratings (1-5 stars)
- Number of reviews
- Technical specifications

This rich metadata enables our RAG system to retrieve relevant products
and allows the factual accuracy metric to validate price and rating claims."
```

**Screen Actions**:
- Navigate to "📊 Data Upload"
- Click "Use Existing Dataset" button
- Scroll through preview table
- Point out key columns (price, ratings, product_name)
- Show dataset statistics

---

### **[4:00 - 5:00] Questions Setup** (60 seconds)

**What to Show**:
- Questions Setup page
- Predefined questions
- Question categories

**What to Say**:
```
"Now let's set up our evaluation questions.

[Click 'Use Predefined 15 Questions']

I've prepared 15 domain-specific questions organized into four categories:

1. Structured Queries - like 'What are the top 5 rated ACs under ₹40,000?'
2. Value Reasoning - like 'Find the best value 1.5 ton AC considering specs and ratings'
3. Temporal Analysis - questions about price trends and recent models
4. Combined Analysis - complex multi-factor comparisons

[Scroll through questions]

The system also supports uploading custom questions in JSON or TXT format,
or entering them manually. These diverse question types help us evaluate
how well each model handles different reasoning tasks."
```

**Screen Actions**:
- Navigate to "❓ Questions Setup"
- Click "Use Predefined 15 Questions"
- Scroll through question list slowly
- Point out different categories
- Highlight question complexity (easy/medium/hard)

---

### **[5:00 - 6:30] Running the Evaluation** (90 seconds)

**What to Show**:
- Run Analysis page
- Model selection
- Progress tracking
- Real-time updates

**What to Say**:
```
"Now for the main evaluation. On the Run Analysis page, I'll select all
three models for comparison.

[Check all three model checkboxes]

I'm evaluating:
- Phi-3 Mini for efficiency
- Llama 3.2 for balanced performance
- Gemma 2 for Google's latest architecture

[Click 'Run Evaluation']

The system now:
1. Loads the product data
2. Builds a FAISS vector index for semantic search
3. Loads all three Ollama models
4. For each of the 15 questions:
   - Retrieves the 3 most relevant products
   - Generates answers from all models
   - Calculates 7 evaluation metrics

[Show progress bar advancing]

This typically takes 10-15 minutes for 45 total evaluations - that's 15
questions times 3 models. For this demo, I'll [either wait or say 'I ran
this earlier, so let me show you the results']"
```

**Screen Actions**:
- Navigate to "🚀 Run Analysis"
- Select all three models
- Click "Run Evaluation"
- Show progress bar updating
- Point to status messages
- **Option A**: Wait 10-15 minutes (boring for video)
- **Option B**: Say "I pre-ran this, let me show results" and navigate to Results

**Note**: For video, recommend pre-running and jumping to results to save time.

---

### **[6:30 - 8:00] Results Analysis** (90 seconds)

**What to Show**:
- Results page
- Summary cards
- Interactive charts
- Detailed metrics

**What to Say**:
```
"Here are the evaluation results.

[Point to summary cards]

We evaluated 15 questions across 3 models. The average factual accuracy
is [X]%, which measures how often models cite correct prices and ratings.

[Show bar chart]

This bar chart compares all models across our seven metrics.

[Hover over bars to show exact values]

Looking at Query Overlap, we see [model] provides the most relevant answers
to the questions asked.

For Factual Accuracy - the key innovation of this project - [model] performs
best at [X]%, meaning it correctly cites prices and ratings from the source
documents.

[Show radar chart]

The radar chart gives an overall performance view. We can see:
- [Model A] excels at factual accuracy and completeness
- [Model B] has balanced performance across all metrics
- [Model C] provides shorter but highly relevant answers

[Scroll to detailed metrics table]

Looking at the detailed breakdown, [model] has the highest context precision
at [X]%, indicating superior retrieval quality.

[Expand a question result]

For individual questions, we can see all three model answers side-by-side
with their specific metrics. For example, on this question about best value
ACs under ₹35,000, [model] provided a [length] word answer with [X]%
factual accuracy."
```

**Screen Actions**:
- Navigate to "📈 Results"
- Point to summary statistics
- Interact with bar chart (hover to show values)
- Explain radar chart patterns
- Scroll through metrics table
- Expand 2-3 question details
- Show model answers side-by-side
- Highlight factual accuracy scores

---

### **[8:00 - 9:00] Key Findings & Metrics** (60 seconds)

**What to Show**:
- Continue on Results page
- Focus on specific metrics
- Highlight interesting findings

**What to Say**:
```
"Let me highlight some key findings from this evaluation.

[Point to factual accuracy in table]

The factual accuracy metric is particularly important for e-commerce
applications. It uses regex to extract prices and ratings from model
answers, then validates them against the source documents.

For instance, if a model says an AC costs ₹35,000 but the actual price
is ₹32,999, the accuracy score decreases. This prevents hallucination
of product details.

[Point to completeness scores]

The completeness metric is question-type aware. For comparison questions,
it checks for comparative language like 'while' and 'whereas'. For value
questions, it looks for reasoning keywords like 'because' and 'offers'.

[Point to context precision]

Context precision, averaging [X]% across models, shows our FAISS retrieval
system is successfully finding relevant products. Higher scores mean the
retrieved documents actually contain information needed to answer the question.

Overall, [summarize which model performed best and why]."
```

**Screen Actions**:
- Highlight specific metrics in table
- Show examples of high vs low factual accuracy
- Point out completeness variations
- Explain what good vs poor scores mean

---

### **[9:00 - 9:45] Export & Documentation** (45 seconds)

**What to Show**:
- Export functionality
- Downloaded files
- Documentation

**What to Say**:
```
"The application provides multiple export options for further analysis.

[Click 'Download JSON Results']

The JSON file contains complete evaluation data including all answers,
metrics, and metadata - perfect for detailed analysis or inclusion in
research papers.

[Click 'Download CSV Summary']

The CSV summary provides a quick overview table that can be imported to
Excel or used in presentations.

[Show outputs folder]

All results are also saved automatically to the outputs folder with
timestamps, including interactive HTML dashboards.

[Show documentation files]

The project includes comprehensive documentation:
- FINAL_PROJECT_GUIDE for complete setup
- STREAMLIT_GUIDE for web interface usage
- SETUP_GUIDE for installation
- Complete source code in the src directory"
```

**Screen Actions**:
- Click download buttons
- Show downloaded files in file explorer
- Navigate to project folder
- Show outputs/ directory
- Show documentation files
- Briefly show src/ code structure

---

### **[9:45 - 10:00] Conclusion** (15 seconds)

**What to Show**:
- Summary slide or Home page

**What to Say**:
```
"In summary, this project delivers:

✅ A complete RAG system using FAISS and Ollama
✅ Comparison of three open-source LLMs
✅ Seven comprehensive evaluation metrics
✅ Innovative factual accuracy validation
✅ Interactive web interface with real-time visualization
✅ Complete documentation and exportable results

This demonstrates practical application of RAG systems for e-commerce
with measurable quality metrics. Thank you for watching."
```

**Screen Actions**:
- Return to Home page or show summary
- End recording

---

## 📝 Alternative Shorter Script (5 Minutes)

If you need a condensed version:

- **[0:00-0:30]** Quick intro + project overview
- **[0:30-1:00]** System setup (ollama list + launch Streamlit)
- **[1:00-2:00]** Home page + Data Upload (quickly load dataset)
- **[2:00-2:30]** Questions Setup (load predefined questions)
- **[2:30-3:00]** Run Analysis (show starting, mention pre-ran)
- **[3:00-4:30]** Results - focus on charts and key findings
- **[4:30-5:00]** Export + conclusion

---

## 🎥 Recording Tips

### Before Recording:

1. **Practice Run**: Do a complete dry run without recording
2. **Script Notes**: Have this script printed or on second monitor
3. **Clean Environment**:
   - Close all unnecessary apps
   - Hide desktop icons
   - Use simple wallpaper
   - Clear browser bookmarks bar
4. **Check Audio**: Test microphone, minimize background noise
5. **Browser Zoom**: Set to 100% or 110% for readability

### During Recording:

1. **Speak Clearly**: Moderate pace, articulate technical terms
2. **Mouse Movement**: Move mouse slowly and deliberately
3. **Pause Between Sections**: Easier to edit later
4. **Point at UI Elements**: Move cursor to what you're discussing
5. **Show Don't Just Tell**: Actually click buttons, scroll, interact
6. **If You Make a Mistake**: Pause, take a breath, continue (can edit later)

### After Recording:

1. **Review**: Watch entire video
2. **Check Audio**: Ensure narration is clear throughout
3. **Verify Content**: All required elements shown
4. **Edit if Needed**: Cut long pauses, fix mistakes
5. **Export**: High quality MP4 (H.264 codec recommended)
6. **File Size**: Aim for under 500MB (compress if needed)

---

## 🎬 Recording Software Options

### Free Options:

1. **OBS Studio** (Recommended)
   - Download: https://obsproject.com/
   - Features: Professional quality, free, open source
   - Setup:
     - Source: Window Capture (browser) or Display Capture (full screen)
     - Audio: Add microphone
     - Output: MP4, 1080p, 30fps

2. **Loom** (Easy to use)
   - Web: https://www.loom.com/
   - Features: Browser extension, automatic upload
   - Limit: Free tier has 5-minute limit (may need paid)

3. **Windows Game Bar** (Built-in)
   - Shortcut: Win + G
   - Features: Simple, already installed
   - Limitation: Basic features only

4. **Mac QuickTime** (Built-in)
   - Features: Simple screen recording
   - Good quality

### Paid Options:
- Camtasia: Professional editing
- ScreenFlow (Mac): All-in-one solution
- Snagit: Simple interface

---

## ✅ Video Content Checklist

Make sure your video includes:

### Required Elements:
- [ ] Your name and student ID mentioned
- [ ] Course information (DSCI 6004)
- [ ] Project overview and goals
- [ ] System architecture explanation (RAG, Ollama, FAISS)
- [ ] Dataset description (50 Air Conditioners)
- [ ] All three models shown (phi3, llama3, gemma2)
- [ ] 15 evaluation questions displayed
- [ ] Evaluation process demonstrated
- [ ] Results visualization (both charts)
- [ ] Metrics explanation (especially factual accuracy)
- [ ] Export functionality shown
- [ ] Clear conclusion

### Technical Demonstrations:
- [ ] Ollama running and models listed
- [ ] Streamlit app launch
- [ ] Data upload/loading
- [ ] Questions setup
- [ ] Evaluation execution (or pre-run results)
- [ ] Interactive chart interaction (hover, zoom)
- [ ] Question-by-question result details
- [ ] File export (JSON and CSV)

### Quality Checks:
- [ ] Audio is clear and audible
- [ ] Screen is visible and readable
- [ ] No long awkward pauses
- [ ] Smooth transitions between sections
- [ ] Duration is 8-10 minutes
- [ ] File size is reasonable (<500MB)
- [ ] Video plays without errors

---

## 📤 Submission Checklist

Along with your video, include:

### 1. Code Files:
- [ ] Complete `src/` directory
- [ ] `streamlit_app.py`
- [ ] `run_complete_evaluation.py`
- [ ] `config.yaml`
- [ ] `requirements.txt`

### 2. Documentation:
- [ ] FINAL_PROJECT_GUIDE.md
- [ ] STREAMLIT_GUIDE.md
- [ ] SETUP_GUIDE.md
- [ ] README_OLLAMA.md

### 3. Results:
- [ ] Sample `outputs/evaluation_*.json`
- [ ] Sample `outputs/dashboard_*.html`
- [ ] Screenshots of Streamlit UI (5-10 images)

### 4. Report (8-page paper):
- [ ] Abstract
- [ ] Introduction (problem statement, contributions)
- [ ] Related Work (RAG systems, LLM comparison)
- [ ] System Architecture (diagrams of RAG pipeline)
- [ ] Methodology (evaluation metrics explained)
- [ ] Results (tables, charts from Streamlit)
- [ ] Discussion (model comparison, findings)
- [ ] Conclusion (summary, future work)
- [ ] References (cite RAG papers, Ollama, models)

### 5. Video:
- [ ] MP4 format
- [ ] 8-10 minutes
- [ ] High quality (720p minimum)
- [ ] Clear audio
- [ ] All checklist items covered

---

## 🚀 Quick Test Run

Before final recording, do this quick test:

```bash
# 1. Start Ollama
ollama serve

# 2. Verify models
ollama list

# 3. Test Streamlit
streamlit run streamlit_app.py

# 4. In browser:
# - Check Home page loads
# - Load dataset
# - Load questions
# - Run evaluation (just 1 model, 3 questions for speed test)
# - Check results appear
# - Test export buttons

# 5. If all works: READY TO RECORD!
```

---

## 💡 Pro Tips

1. **Time Management**:
   - Don't spend too long on setup (max 2 minutes)
   - Focus on results and metrics (3-4 minutes)
   - Keep intro and conclusion brief

2. **Highlight Innovation**:
   - Emphasize factual accuracy metric
   - Show how it validates prices/ratings
   - Compare to basic metrics (just length/tokens)

3. **Show Interactivity**:
   - Hover over chart bars to show values
   - Expand question details
   - Actually click export buttons

4. **Explain Technical Decisions**:
   - Why Ollama vs HuggingFace (local, no API costs)
   - Why FAISS (fast vector similarity)
   - Why these 3 models (free, diverse sizes)

5. **If Evaluation is Slow**:
   - Run it beforehand
   - Show clicking "Run" button
   - Say "This takes 10-15 minutes, so I pre-ran it"
   - Jump to Results page
   - Mention in narration: "The evaluation processed 45 queries..."

---

## 🎓 Academic Presentation Style

For professional presentation:

1. **Formal Tone**: Use proper terminology (retrieval-augmented generation, semantic similarity, etc.)
2. **Cite Methods**: "Using FAISS for vector indexing..." "Based on RAG architecture..."
3. **Quantify Results**: Always mention actual numbers ("84% factual accuracy", "15 questions", etc.)
4. **Explain Metrics**: Don't just show charts, explain what they mean
5. **Connect to Course**: Relate to NLP concepts learned in DSCI 6004

---

**Good luck with your video recording! 🎥**

Your Streamlit app is working perfectly, and following this script will create a professional, comprehensive demonstration of your RAG evaluation system.
