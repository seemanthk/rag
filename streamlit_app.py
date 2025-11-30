"""
ShopSmart RAG - Interactive Streamlit Web Application
Upload CSV, customize questions, and visualize results in real-time
"""

import streamlit as st
import pandas as pd
import yaml
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time
from datetime import datetime

from src.data_loader import AmazonProductDataLoader
from src.vector_store import VectorStore
from src.ollama_handler import OllamaMultiLLMManager
from src.rag_system import RAGSystem
from src.evaluation import RAGEvaluator

# Page configuration
st.set_page_config(
    page_title="ShopSmart RAG Analysis",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
        border-radius: 5px;
    }
    .question-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'vector_index_built' not in st.session_state:
    st.session_state.vector_index_built = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'questions' not in st.session_state:
    st.session_state.questions = []

# Header
st.markdown('<h1 class="main-header">🛍️ ShopSmart RAG Analysis System</h1>', unsafe_allow_html=True)
st.markdown("### Intelligent Product Analysis with Multi-LLM Comparison")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=ShopSmart+RAG", use_column_width=True)
    st.markdown("## 📋 Navigation")

    page = st.radio(
        "Select Page",
        ["🏠 Home", "📊 Data Upload", "❓ Questions Setup", "🚀 Run Analysis", "📈 Results & Visualization"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    # Load config
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except:
        st.error("Config file not found!")
        config = {}

    top_k = st.slider("Documents to Retrieve", 1, 10, 3)
    selected_models = st.multiselect(
        "Models to Compare",
        ["phi3", "llama3", "gemma2"],
        default=["phi3", "llama3", "gemma2"]
    )

    st.markdown("---")
    st.markdown("### 📊 Status")
    st.success("✓ Ollama Running" if st.session_state.get('models_loaded') else "⚠ Models Not Loaded")
    st.info(f"✓ Data Loaded: {st.session_state.get('data_loaded', False)}")
    st.info(f"✓ Index Built: {st.session_state.get('vector_index_built', False)}")

# Page: Home
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>📁 Upload Data</h2>
            <p>Upload your product CSV file</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>❓ Questions</h2>
            <p>Define evaluation questions</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>🚀 Analyze</h2>
            <p>Run multi-LLM evaluation</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 How It Works")

    st.markdown("""
    1. **Upload Data** - Upload your product CSV file (Air Conditioners, Laptops, etc.)
    2. **Setup Questions** - Upload questions (JSON/TXT) or use predefined ones
    3. **Configure Models** - Select which Ollama models to compare
    4. **Run Analysis** - System will:
       - Build vector index for semantic search
       - Query all selected models
       - Compute comprehensive metrics
       - Generate visualizations
    5. **View Results** - Interactive charts and detailed comparisons
    """)

    st.markdown("---")
    st.markdown("### 📊 Evaluation Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Answer Quality:**
        - ✅ Relevancy (Query Overlap)
        - ✅ Faithfulness (Context Grounding)
        - ✅ Completeness
        """)

    with col2:
        st.markdown("""
        **Factual Accuracy:**
        - ✅ Price Verification
        - ✅ Rating Verification
        - ✅ Context Precision
        """)

# Page: Data Upload
elif page == "📊 Data Upload":
    st.markdown("## 📊 Upload Product Data")

    upload_method = st.radio("Choose upload method:", ["Upload CSV File", "Use Existing File"])

    if upload_method == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload Product CSV", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! {len(df)} products loaded.")

                # Save to temp location
                temp_path = "data/uploaded_products.csv"
                Path("data").mkdir(exist_ok=True)
                df.to_csv(temp_path, index=False)

                st.session_state.csv_path = temp_path
                st.session_state.data_loaded = True

                # Show preview
                st.markdown("### 📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Products", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())

            except Exception as e:
                st.error(f"Error loading file: {e}")

    else:  # Use existing file
        existing_files = list(Path("data").glob("*.csv"))
        if existing_files:
            selected_file = st.selectbox(
                "Select CSV file:",
                [f.name for f in existing_files]
            )

            if st.button("Load Selected File"):
                try:
                    file_path = f"data/{selected_file}"
                    df = pd.read_csv(file_path)
                    st.success(f"✅ Loaded {len(df)} products from {selected_file}")

                    st.session_state.csv_path = file_path
                    st.session_state.data_loaded = True

                    st.dataframe(df.head(10), use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("No CSV files found in data/ directory")

# Page: Questions Setup
elif page == "❓ Questions Setup":
    st.markdown("## ❓ Configure Evaluation Questions")

    question_method = st.radio(
        "Choose method:",
        ["Upload Questions File", "Use Predefined Questions", "Enter Questions Manually"]
    )

    if question_method == "Upload Questions File":
        st.markdown("Upload a JSON or TXT file with your questions")

        uploaded_qs = st.file_uploader("Upload Questions", type=['json', 'txt'])

        if uploaded_qs is not None:
            try:
                if uploaded_qs.name.endswith('.json'):
                    questions = json.load(uploaded_qs)
                    st.session_state.questions = questions
                    st.success(f"✅ Loaded {len(questions)} questions from JSON")
                else:  # TXT file
                    content = uploaded_qs.read().decode('utf-8')
                    questions = [
                        {
                            "id": i+1,
                            "question": q.strip(),
                            "category": "custom",
                            "difficulty": "medium"
                        }
                        for i, q in enumerate(content.split('\n')) if q.strip()
                    ]
                    st.session_state.questions = questions
                    st.success(f"✅ Loaded {len(questions)} questions from TXT")

                # Show questions
                for q in questions:
                    with st.expander(f"Q{q['id']}: {q['question'][:60]}..."):
                        st.write(f"**Full Question:** {q['question']}")
                        st.write(f"**Category:** {q.get('category', 'N/A')}")
                        st.write(f"**Difficulty:** {q.get('difficulty', 'N/A')}")

            except Exception as e:
                st.error(f"Error loading questions: {e}")

    elif question_method == "Use Predefined Questions":
        from src.questions import get_all_questions

        questions = get_all_questions()
        st.session_state.questions = questions

        st.success(f"✅ Loaded {len(questions)} predefined questions")

        # Show by category
        categories = {}
        for q in questions:
            cat = q.get('category', 'Unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(q)

        for cat, qs in categories.items():
            with st.expander(f"📁 {cat.upper()} ({len(qs)} questions)"):
                for q in qs:
                    st.markdown(f"""
                    <div class="question-card">
                        <strong>Q{q['id']}:</strong> {q['question']}
                        <br><small>Difficulty: {q.get('difficulty', 'N/A')}</small>
                    </div>
                    """, unsafe_allow_html=True)

    else:  # Manual entry
        st.markdown("### ✏️ Enter Questions Manually")

        num_questions = st.number_input("Number of questions", 1, 20, 5)

        questions = []
        for i in range(num_questions):
            with st.expander(f"Question {i+1}", expanded=(i==0)):
                question = st.text_area(f"Question {i+1}", key=f"q_{i}")
                col1, col2 = st.columns(2)
                with col1:
                    category = st.selectbox(f"Category",
                        ["structured_query", "value_reasoning", "temporal_analysis", "combined"],
                        key=f"cat_{i}")
                with col2:
                    difficulty = st.selectbox(f"Difficulty",
                        ["easy", "medium", "hard"],
                        key=f"diff_{i}")

                if question:
                    questions.append({
                        "id": i+1,
                        "question": question,
                        "category": category,
                        "difficulty": difficulty
                    })

        if st.button("Save Questions"):
            st.session_state.questions = questions
            st.success(f"✅ Saved {len(questions)} questions")

# Page: Run Analysis
elif page == "🚀 Run Analysis":
    st.markdown("## 🚀 Run Multi-LLM Analysis")

    # Check prerequisites
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
        st.stop()

    if not st.session_state.questions:
        st.warning("⚠️ Please configure questions first!")
        st.stop()

    st.markdown(f"### ✅ Ready to Analyze")
    st.info(f"📊 Data: {st.session_state.csv_path}")
    st.info(f"❓ Questions: {len(st.session_state.questions)}")
    st.info(f"🤖 Models: {', '.join(selected_models)}")

    if st.button("🚀 START ANALYSIS", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Load data
            status_text.text("📊 Loading data...")
            progress_bar.progress(10)

            data_loader = AmazonProductDataLoader(st.session_state.csv_path)
            df = data_loader.load_data()
            df = data_loader.preprocess_data()
            documents = data_loader.create_documents()

            st.success(f"✅ Loaded {len(documents)} products")
            progress_bar.progress(20)

            # Step 2: Build vector index
            status_text.text("🔍 Building vector index...")

            vector_store = VectorStore(
                embedding_model_name=config['embedding']['model_name'],
                index_path="data/temp_vector_index",
                use_faiss=True
            )

            embeddings = vector_store.create_embeddings(documents, batch_size=32)
            vector_store.build_index()

            st.success("✅ Vector index built")
            st.session_state.vector_index_built = True
            progress_bar.progress(40)

            # Step 3: Load models
            status_text.text("🤖 Loading Ollama models...")

            llm_manager = OllamaMultiLLMManager(
                llm_configs=config['llms'],
                base_url=config['ollama']['base_url']
            )

            for model in selected_models:
                llm_manager.load_llm(model)
                st.success(f"✅ Loaded {model}")

            st.session_state.models_loaded = True
            progress_bar.progress(50)

            # Step 4: Setup RAG
            status_text.text("⚙️ Setting up RAG system...")

            rag_system = RAGSystem(
                vector_store=vector_store,
                llm_manager=llm_manager,
                top_k=top_k
            )

            evaluator = RAGEvaluator()
            progress_bar.progress(60)

            # Step 5: Run evaluation
            status_text.text("🔄 Running evaluation...")

            all_results = []
            total_qs = len(st.session_state.questions)

            for i, q in enumerate(st.session_state.questions):
                status_text.text(f"📝 Question {i+1}/{total_qs}: {q['question'][:50]}...")

                question_result = {
                    'question_id': q['id'],
                    'question': q['question'],
                    'category': q.get('category', 'unknown'),
                    'difficulty': q.get('difficulty', 'medium'),
                    'model_results': {}
                }

                # Query all models
                results = rag_system.query_all_llms(
                    query=q['question'],
                    return_context=True
                )

                for model_name, result in results.items():
                    # Evaluate
                    metrics = evaluator.evaluate_answer(
                        answer=result['answer'],
                        context=result.get('context', ''),
                        query=q['question'],
                        retrieved_docs=result.get('retrieved_docs', [])
                    )

                    question_result['model_results'][model_name] = {
                        'answer': result['answer'],
                        'metrics': metrics
                    }

                all_results.append(question_result)
                progress_bar.progress(60 + int(30 * (i+1) / total_qs))

            # Save results
            st.session_state.evaluation_results = all_results

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"outputs/streamlit_results_{timestamp}.json"
            Path("outputs").mkdir(exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            progress_bar.progress(100)
            status_text.text("✅ Analysis Complete!")

            st.success(f"🎉 Analysis completed! Results saved to {output_file}")
            st.balloons()

        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            import traceback
            st.code(traceback.format_exc())

# Page: Results
elif page == "📈 Results & Visualization":
    st.markdown("## 📈 Analysis Results & Visualization")

    if st.session_state.evaluation_results is None:
        st.warning("⚠️ No results yet. Please run analysis first!")
        st.stop()

    results = st.session_state.evaluation_results

    # Calculate summary stats
    total_questions = len(results)
    models = list(results[0]['model_results'].keys())

    # Summary metrics
    st.markdown("### 📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Questions", total_questions)
    with col2:
        st.metric("Models Compared", len(models))
    with col3:
        avg_length = sum(
            sum(r['model_results'][m]['metrics']['answer_length'] for m in models) / len(models)
            for r in results
        ) / len(results)
        st.metric("Avg Answer Length", f"{avg_length:.0f} words")
    with col4:
        avg_accuracy = sum(
            sum(r['model_results'][m]['metrics']['factual_accuracy'] for m in models) / len(models)
            for r in results
        ) / len(results)
        st.metric("Avg Factual Accuracy", f"{avg_accuracy*100:.1f}%")

    st.markdown("---")

    # Aggregate metrics by model
    model_metrics = {}
    for model in models:
        model_metrics[model] = {
            'query_overlap': [],
            'factual_accuracy': [],
            'context_precision': [],
            'completeness': []
        }

        for result in results:
            metrics = result['model_results'][model]['metrics']
            model_metrics[model]['query_overlap'].append(metrics['query_overlap'])
            model_metrics[model]['factual_accuracy'].append(metrics['factual_accuracy'])
            model_metrics[model]['context_precision'].append(metrics['context_precision'])
            model_metrics[model]['completeness'].append(metrics['completeness'])

    # Calculate averages
    model_avg = {}
    for model in models:
        model_avg[model] = {
            k: sum(v) / len(v) * 100 for k, v in model_metrics[model].items()
        }

    # Charts
    st.markdown("### 📊 Performance Comparison")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Detailed Metrics", "📝 Question Results", "💾 Export"])

    with tab1:
        # Bar chart comparison
        metrics_names = ['Query Overlap', 'Factual Accuracy', 'Context Precision', 'Completeness']

        fig = go.Figure()

        for model in models:
            fig.add_trace(go.Bar(
                name=model.upper(),
                x=metrics_names,
                y=[
                    model_avg[model]['query_overlap'],
                    model_avg[model]['factual_accuracy'],
                    model_avg[model]['context_precision'],
                    model_avg[model]['completeness']
                ]
            ))

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Metrics",
            yaxis_title="Score (%)",
            barmode='group',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        fig_radar = go.Figure()

        for model in models:
            fig_radar.add_trace(go.Scatterpolar(
                r=[
                    model_avg[model]['query_overlap'],
                    model_avg[model]['factual_accuracy'],
                    model_avg[model]['context_precision'],
                    model_avg[model]['completeness']
                ],
                theta=metrics_names,
                fill='toself',
                name=model.upper()
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Overall Performance Radar",
            height=500
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        # Detailed metrics table
        st.markdown("### 📋 Detailed Metrics by Model")

        for model in models:
            with st.expander(f"🤖 {model.upper()}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Query Overlap", f"{model_avg[model]['query_overlap']:.1f}%")
                with col2:
                    st.metric("Factual Accuracy", f"{model_avg[model]['factual_accuracy']:.1f}%")
                with col3:
                    st.metric("Context Precision", f"{model_avg[model]['context_precision']:.1f}%")
                with col4:
                    st.metric("Completeness", f"{model_avg[model]['completeness']:.1f}%")

    with tab3:
        # Question-by-question results
        st.markdown("### 📝 Question-by-Question Results")

        for result in results:
            with st.expander(f"Q{result['question_id']}: {result['question'][:80]}..."):
                st.markdown(f"**Full Question:** {result['question']}")
                st.markdown(f"**Category:** {result['category']} | **Difficulty:** {result['difficulty']}")

                st.markdown("---")

                for model, model_result in result['model_results'].items():
                    st.markdown(f"### {model.upper()}")
                    st.write(model_result['answer'])

                    metrics = model_result['metrics']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Relevancy", f"{metrics['query_overlap']*100:.0f}%")
                    with col2:
                        st.metric("Accuracy", f"{metrics['factual_accuracy']*100:.0f}%")
                    with col3:
                        st.metric("Precision", f"{metrics['context_precision']*100:.0f}%")
                    with col4:
                        st.metric("Complete", f"{metrics['completeness']*100:.0f}%")

                    st.markdown("---")

    with tab4:
        # Export options
        st.markdown("### 💾 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Download JSON"):
                json_str = json.dumps(results, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download JSON File",
                    data=json_str,
                    file_name=f"rag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        with col2:
            if st.button("📥 Download Summary CSV"):
                # Create summary DataFrame
                summary_data = []
                for model in models:
                    summary_data.append({
                        'Model': model.upper(),
                        'Query Overlap (%)': f"{model_avg[model]['query_overlap']:.1f}",
                        'Factual Accuracy (%)': f"{model_avg[model]['factual_accuracy']:.1f}",
                        'Context Precision (%)': f"{model_avg[model]['context_precision']:.1f}",
                        'Completeness (%)': f"{model_avg[model]['completeness']:.1f}"
                    })

                summary_df = pd.DataFrame(summary_data)
                csv = summary_df.to_csv(index=False)

                st.download_button(
                    label="Download CSV File",
                    data=csv,
                    file_name=f"rag_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>ShopSmart RAG System | Powered by Ollama | Built with Streamlit</p>
    <p>DSCI 6004 - Natural Language Processing Term Project</p>
</div>
""", unsafe_allow_html=True)
