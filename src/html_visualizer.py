"""
HTML Visualization Generator for RAG Evaluation Results
"""

import json
from typing import Dict, List
from datetime import datetime


class HTMLVisualizer:
    """Generate interactive HTML dashboards for RAG evaluation"""

    def __init__(self):
        """Initialize visualizer"""
        self.results = []

    def generate_dashboard(self, evaluation_results: List[Dict], output_file: str):
        """
        Generate comprehensive HTML dashboard

        Args:
            evaluation_results: List of evaluation results
            output_file: Path to save HTML file
        """

        html = self._generate_html_template(evaluation_results)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

    def _generate_html_template(self, results: List[Dict]) -> str:
        """Generate complete HTML template"""

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(results)

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ShopSmart RAG Evaluation Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🛍️ ShopSmart RAG Evaluation Dashboard</h1>
            <p class="subtitle">Air Conditioner Product Analysis with Ollama Models</p>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        {self._generate_summary_section(summary_stats, results)}
        {self._generate_model_comparison(summary_stats)}
        {self._generate_metrics_charts(summary_stats)}
        {self._generate_detailed_results(results)}
    </div>

    <script>
        {self._get_javascript(summary_stats)}
    </script>
</body>
</html>
"""
        return html

    def _get_css(self) -> str:
        """Generate CSS styles"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .timestamp {
            margin-top: 10px;
            opacity: 0.8;
            font-size: 0.9em;
        }

        .section {
            padding: 30px 40px;
            border-bottom: 1px solid #e0e0e0;
        }

        .section:last-child {
            border-bottom: none;
        }

        .section-title {
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #667eea;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: transform 0.3s;
        }

        .stat-card:hover {
            transform: translateY(-5px);
        }

        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }

        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
        }

        .model-comparison {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .model-card {
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            padding: 20px;
            background: #f9f9f9;
            transition: all 0.3s;
        }

        .model-card:hover {
            border-color: #667eea;
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
        }

        .model-name {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 15px;
            text-transform: uppercase;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #e0e0e0;
        }

        .metric-row:last-child {
            border-bottom: none;
        }

        .metric-label {
            color: #666;
            font-size: 0.95em;
        }

        .metric-value {
            font-weight: bold;
            color: #333;
        }

        .score-bar {
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }

        .score-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.5s;
        }

        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }

        .chart-container {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .chart-title {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #667eea;
            text-align: center;
        }

        .question-card {
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
        }

        .question-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .question-text {
            font-size: 1.1em;
            font-weight: 500;
            color: #333;
        }

        .question-meta {
            display: flex;
            gap: 10px;
        }

        .badge {
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: 500;
        }

        .badge-category {
            background: #e3f2fd;
            color: #1976d2;
        }

        .badge-difficulty {
            background: #fff3e0;
            color: #f57c00;
        }

        .answer-section {
            margin-top: 15px;
        }

        .model-answer {
            background: white;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 3px solid #667eea;
        }

        .model-label {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
        }

        .answer-text {
            color: #333;
            line-height: 1.6;
        }

        .metrics-summary {
            display: flex;
            gap: 15px;
            margin-top: 10px;
            flex-wrap: wrap;
        }

        .metric-badge {
            background: #f0f0f0;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.85em;
        }

        @media (max-width: 768px) {
            .stats-grid, .model-comparison, .charts-grid {
                grid-template-columns: 1fr;
            }

            h1 {
                font-size: 1.8em;
            }

            .section {
                padding: 20px;
            }
        }
        """

    def _generate_summary_section(self, stats: Dict, results: List[Dict]) -> str:
        """Generate summary statistics section"""
        return f"""
        <div class="section">
            <h2 class="section-title">📊 Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Questions</div>
                    <div class="stat-value">{stats['total_questions']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Models Compared</div>
                    <div class="stat-value">{stats['num_models']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Answer Length</div>
                    <div class="stat-value">{stats.get('avg_answer_length', 0):.0f}</div>
                    <div class="stat-label" style="font-size: 0.8em; margin-top: 5px;">words</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Factual Accuracy</div>
                    <div class="stat-value">{stats.get('avg_factual_accuracy', 0)*100:.1f}%</div>
                </div>
            </div>
        </div>
        """

    def _generate_model_comparison(self, stats: Dict) -> str:
        """Generate model comparison section"""
        models_html = ""

        for model_name, model_stats in stats.get('model_metrics', {}).items():
            metrics_html = ""
            for metric_name, value in model_stats.items():
                # Skip non-numeric or internal metrics
                if not isinstance(value, (int, float)):
                    continue

                # Normalize value to 0-1 for progress bar
                if 'accuracy' in metric_name or 'precision' in metric_name:
                    normalized = value
                elif 'overlap' in metric_name:
                    normalized = value
                elif 'length' in metric_name:
                    normalized = min(value / 100, 1.0)
                else:
                    normalized = value if value <= 1 else value / 100

                display_value = f"{value*100:.1f}%" if value <= 1 else f"{value:.1f}"

                metrics_html += f"""
                <div class="metric-row">
                    <span class="metric-label">{metric_name.replace('_', ' ').title()}</span>
                    <span class="metric-value">{display_value}</span>
                </div>
                <div class="score-bar">
                    <div class="score-fill" style="width: {normalized*100}%"></div>
                </div>
                """

            models_html += f"""
            <div class="model-card">
                <div class="model-name">{model_name}</div>
                {metrics_html}
            </div>
            """

        return f"""
        <div class="section">
            <h2 class="section-title">🤖 Model Performance Comparison</h2>
            <div class="model-comparison">
                {models_html}
            </div>
        </div>
        """

    def _generate_metrics_charts(self, stats: Dict) -> str:
        """Generate charts section (placeholder for Chart.js)"""
        return f"""
        <div class="section">
            <h2 class="section-title">📈 Metrics Visualization</h2>
            <div class="charts-grid">
                <div class="chart-container">
                    <div class="chart-title">Answer Relevancy</div>
                    <canvas id="relevancyChart"></canvas>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Factual Accuracy</div>
                    <canvas id="accuracyChart"></canvas>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Context Precision</div>
                    <canvas id="precisionChart"></canvas>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Overall Performance</div>
                    <canvas id="overallChart"></canvas>
                </div>
            </div>
        </div>
        """

    def _generate_detailed_results(self, results: List[Dict]) -> str:
        """Generate detailed results section"""
        questions_html = ""

        for result in results:
            question_id = result.get('question_id', '')
            question = result.get('question', '')
            category = result.get('category', 'Unknown')
            difficulty = result.get('difficulty', 'Unknown')

            answers_html = ""
            for model_name, model_result in result.get('model_results', {}).items():
                answer = model_result.get('answer', 'No answer')
                metrics = model_result.get('metrics', {})

                metrics_badges = ""
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        display_val = f"{value*100:.0f}%" if value <= 1 else f"{value:.0f}"
                        metrics_badges += f'<span class="metric-badge">{metric_name}: {display_val}</span>'

                answers_html += f"""
                <div class="model-answer">
                    <div class="model-label">{model_name.upper()}</div>
                    <div class="answer-text">{answer}</div>
                    <div class="metrics-summary">{metrics_badges}</div>
                </div>
                """

            questions_html += f"""
            <div class="question-card">
                <div class="question-header">
                    <span class="question-text">Q{question_id}: {question}</span>
                    <div class="question-meta">
                        <span class="badge badge-category">{category}</span>
                        <span class="badge badge-difficulty">{difficulty}</span>
                    </div>
                </div>
                <div class="answer-section">
                    {answers_html}
                </div>
            </div>
            """

        return f"""
        <div class="section">
            <h2 class="section-title">📝 Detailed Question Results</h2>
            {questions_html}
        </div>
        """

    def _get_javascript(self, stats: Dict) -> str:
        """Generate JavaScript for charts"""
        models = list(stats.get('model_metrics', {}).keys())

        # Map model names to cleaner display names
        model_display = {
            'phi3': 'Phi-3',
            'llama3': 'Llama 3',
            'gemma2': 'Gemma 2'
        }
        display_names = [model_display.get(m, m.upper()) for m in models]

        model_colors = {
            'phi3': '#FF6384',
            'llama3': '#36A2EB',
            'gemma2': '#FFCE56'
        }

        # Extract metrics for each model
        relevancy_data = []
        accuracy_data = []
        precision_data = []
        completeness_data = []

        for model in models:
            model_metrics = stats['model_metrics'].get(model, {})
            relevancy_data.append(round(model_metrics.get('query_overlap', 0) * 100, 1))
            accuracy_data.append(round(model_metrics.get('factual_accuracy', 0) * 100, 1))
            precision_data.append(round(model_metrics.get('context_precision', 0) * 100, 1))
            completeness_data.append(round(model_metrics.get('completeness', 0) * 100, 1))

        colors = [model_colors.get(m, '#999999') for m in models]

        return f"""
        // Chart.js visualizations
        const modelNames = {json.dumps(display_names)};
        const colors = {json.dumps(colors)};

        // Relevancy Chart
        new Chart(document.getElementById('relevancyChart'), {{
            type: 'bar',
            data: {{
                labels: modelNames,
                datasets: [{{
                    label: 'Query Overlap (%)',
                    data: {json.dumps(relevancy_data)},
                    backgroundColor: colors,
                    borderColor: colors,
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{ display: true }},
                    title: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Accuracy Chart
        new Chart(document.getElementById('accuracyChart'), {{
            type: 'bar',
            data: {{
                labels: modelNames,
                datasets: [{{
                    label: 'Factual Accuracy (%)',
                    data: {json.dumps(accuracy_data)},
                    backgroundColor: colors,
                    borderColor: colors,
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{ display: true }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Precision Chart
        new Chart(document.getElementById('precisionChart'), {{
            type: 'bar',
            data: {{
                labels: modelNames,
                datasets: [{{
                    label: 'Context Precision (%)',
                    data: {json.dumps(precision_data)},
                    backgroundColor: colors,
                    borderColor: colors,
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{ display: true }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Overall Performance (Radar)
        new Chart(document.getElementById('overallChart'), {{
            type: 'radar',
            data: {{
                labels: ['Relevancy', 'Accuracy', 'Precision', 'Completeness'],
                datasets: [
                    {{
                        label: modelNames[0],
                        data: [
                            {json.dumps(relevancy_data)}[0] || 0,
                            {json.dumps(accuracy_data)}[0] || 0,
                            {json.dumps(precision_data)}[0] || 0,
                            {json.dumps(completeness_data)}[0] || 0
                        ],
                        borderColor: colors[0],
                        backgroundColor: colors[0] + '33',
                        pointBackgroundColor: colors[0],
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: colors[0]
                    }},
                    {{
                        label: modelNames[1] || 'Model 2',
                        data: [
                            {json.dumps(relevancy_data)}[1] || 0,
                            {json.dumps(accuracy_data)}[1] || 0,
                            {json.dumps(precision_data)}[1] || 0,
                            {json.dumps(completeness_data)}[1] || 0
                        ],
                        borderColor: colors[1] || '#999',
                        backgroundColor: (colors[1] || '#999') + '33',
                        pointBackgroundColor: colors[1] || '#999',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: colors[1] || '#999'
                    }},
                    {{
                        label: modelNames[2] || 'Model 3',
                        data: [
                            {json.dumps(relevancy_data)}[2] || 0,
                            {json.dumps(accuracy_data)}[2] || 0,
                            {json.dumps(precision_data)}[2] || 0,
                            {json.dumps(completeness_data)}[2] || 0
                        ],
                        borderColor: colors[2] || '#999',
                        backgroundColor: (colors[2] || '#999') + '33',
                        pointBackgroundColor: colors[2] || '#999',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: colors[2] || '#999'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top'
                    }}
                }},
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            stepSize: 20,
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }});

        console.log('Charts initialized successfully');
        console.log('Relevancy data:', {json.dumps(relevancy_data)});
        console.log('Accuracy data:', {json.dumps(accuracy_data)});
        console.log('Precision data:', {json.dumps(precision_data)});
        """

    def _calculate_summary_stats(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics from results"""
        stats = {
            'total_questions': len(results),
            'num_models': 0,
            'model_metrics': {},
            'avg_answer_length': 0,
            'avg_factual_accuracy': 0
        }

        if not results:
            return stats

        # Get all models
        first_result = results[0]
        models = list(first_result.get('model_results', {}).keys())
        stats['num_models'] = len(models)

        # Aggregate metrics for each model
        for model in models:
            model_metrics = {
                'answer_length': [],
                'query_overlap': [],
                'context_overlap': [],
                'context_precision': [],
                'factual_accuracy': [],
                'specificity_score': [],
                'completeness': []
            }

            for result in results:
                model_result = result.get('model_results', {}).get(model, {})
                metrics = model_result.get('metrics', {})

                for key in model_metrics.keys():
                    if key in metrics:
                        model_metrics[key].append(metrics[key])

            # Calculate averages
            stats['model_metrics'][model] = {}
            for key, values in model_metrics.items():
                if values:
                    stats['model_metrics'][model][key] = sum(values) / len(values)

        # Overall averages
        all_lengths = []
        all_accuracies = []
        for model_stats in stats['model_metrics'].values():
            all_lengths.append(model_stats.get('answer_length', 0))
            all_accuracies.append(model_stats.get('factual_accuracy', 0))

        stats['avg_answer_length'] = sum(all_lengths) / len(all_lengths) if all_lengths else 0
        stats['avg_factual_accuracy'] = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0

        return stats
