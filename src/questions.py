"""
Domain-specific questions for Air Conditioner product RAG evaluation
Based on ShopSmart RAG project requirements
"""

# 15 questions covering different aspects - aligned with project requirements
EVALUATION_QUESTIONS = [
    # Category A: Pure Structured (SQL-focused)
    {
        "id": 1,
        "question": "What are the top 5 rated air conditioners under ₹40,000?",
        "category": "structured_query",
        "difficulty": "easy",
        "type": "price_rating_filter"
    },
    {
        "id": 2,
        "question": "Show me all 1.5 ton inverter ACs with ratings above 4.0",
        "category": "structured_query",
        "difficulty": "easy",
        "type": "specs_filter"
    },
    {
        "id": 3,
        "question": "Which brand has the highest average rating for air conditioners?",
        "category": "structured_query",
        "difficulty": "easy",
        "type": "brand_analysis"
    },
    {
        "id": 4,
        "question": "Compare average prices of 3-star ACs vs 5-star ACs",
        "category": "structured_query",
        "difficulty": "easy",
        "type": "price_comparison"
    },

    # Category B: Value Reasoning (Hybrid)
    {
        "id": 5,
        "question": "Find the best value 1.5 ton air conditioner under ₹35,000 considering specs and ratings",
        "category": "value_reasoning",
        "difficulty": "medium",
        "type": "value_optimization"
    },
    {
        "id": 6,
        "question": "Which budget ACs (₹25,000-₹35,000) offer the best bang for buck in terms of features and energy efficiency?",
        "category": "value_reasoning",
        "difficulty": "medium",
        "type": "value_range"
    },
    {
        "id": 7,
        "question": "Show me products where the discount is more than 30% and have good ratings",
        "category": "value_reasoning",
        "difficulty": "medium",
        "type": "discount_value"
    },
    {
        "id": 8,
        "question": "What's the sweet spot price range for inverter ACs based on customer satisfaction?",
        "category": "value_reasoning",
        "difficulty": "hard",
        "type": "price_satisfaction"
    },
    {
        "id": 9,
        "question": "Compare value proposition: LG AC at ₹46,000 with 4.2 rating vs Voltas at ₹32,000 with 4.0 rating",
        "category": "value_reasoning",
        "difficulty": "hard",
        "type": "value_comparison"
    },

    # Category C: Temporal Analysis (simulated with data available)
    {
        "id": 10,
        "question": "Which AC models have declining ratings and what are common complaints?",
        "category": "temporal_analysis",
        "difficulty": "hard",
        "type": "trend_detection"
    },
    {
        "id": 11,
        "question": "Find air conditioners with consistently high satisfaction based on number of reviews",
        "category": "temporal_analysis",
        "difficulty": "medium",
        "type": "consistency_check"
    },
    {
        "id": 12,
        "question": "Which brands maintain quality over time based on review volume and ratings?",
        "category": "temporal_analysis",
        "difficulty": "hard",
        "type": "brand_consistency"
    },

    # Category D: Combined (Value + Temporal)
    {
        "id": 13,
        "question": "Find durable air conditioners (high review count with sustained ratings) with best price-to-performance ratio",
        "category": "combined",
        "difficulty": "hard",
        "type": "durability_value"
    },
    {
        "id": 14,
        "question": "Which budget ACs under ₹35,000 have the best value AND strong customer trust (high review count)?",
        "category": "combined",
        "difficulty": "hard",
        "type": "value_trust"
    },
    {
        "id": 15,
        "question": "Recommend the most reliable inverter AC brands considering price, ratings, and customer engagement",
        "category": "combined",
        "difficulty": "hard",
        "type": "comprehensive_recommendation"
    }
]


def get_questions_by_category(category: str):
    """Get questions filtered by category"""
    return [q for q in EVALUATION_QUESTIONS if q['category'] == category]


def get_questions_by_difficulty(difficulty: str):
    """Get questions filtered by difficulty"""
    return [q for q in EVALUATION_QUESTIONS if q['difficulty'] == difficulty]


def get_question_by_id(question_id: int):
    """Get a specific question by ID"""
    for q in EVALUATION_QUESTIONS:
        if q['id'] == question_id:
            return q
    return None


def get_all_questions():
    """Get all questions"""
    return EVALUATION_QUESTIONS
