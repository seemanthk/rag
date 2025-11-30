"""
Domain-specific questions for Amazon product RAG evaluation
"""

# 10+ questions covering different aspects of Amazon products
EVALUATION_QUESTIONS = [
    {
        "id": 1,
        "question": "What are the top-rated electronics products with a rating above 4.5 stars?",
        "category": "product_search",
        "difficulty": "easy"
    },
    {
        "id": 2,
        "question": "Which products in the Home & Kitchen category are priced under $50?",
        "category": "price_filter",
        "difficulty": "easy"
    },
    {
        "id": 3,
        "question": "Compare the features and prices of wireless earbuds available in the dataset.",
        "category": "comparison",
        "difficulty": "medium"
    },
    {
        "id": 4,
        "question": "What are the most common complaints or negative aspects mentioned in reviews for laptop products?",
        "category": "sentiment_analysis",
        "difficulty": "hard"
    },
    {
        "id": 5,
        "question": "Recommend a budget-friendly smartphone with good camera quality based on the product descriptions.",
        "category": "recommendation",
        "difficulty": "medium"
    },
    {
        "id": 6,
        "question": "What are the key differences between gaming laptops and regular laptops in terms of specifications?",
        "category": "comparison",
        "difficulty": "medium"
    },
    {
        "id": 7,
        "question": "Which fitness trackers or smartwatches have the best battery life according to product descriptions?",
        "category": "feature_extraction",
        "difficulty": "medium"
    },
    {
        "id": 8,
        "question": "What is the average price range for Bluetooth speakers with waterproof features?",
        "category": "aggregation",
        "difficulty": "medium"
    },
    {
        "id": 9,
        "question": "Find products with 'wireless charging' capability and list their key features.",
        "category": "feature_search",
        "difficulty": "easy"
    },
    {
        "id": 10,
        "question": "What are the most popular brands for headphones based on the number of products available?",
        "category": "brand_analysis",
        "difficulty": "easy"
    },
    {
        "id": 11,
        "question": "Explain the differences between various types of chargers (fast charging, wireless, USB-C) available in the dataset.",
        "category": "educational",
        "difficulty": "medium"
    },
    {
        "id": 12,
        "question": "Which products have the most detailed descriptions and what categories do they belong to?",
        "category": "meta_analysis",
        "difficulty": "hard"
    },
    {
        "id": 13,
        "question": "What are the pros and cons of buying refurbished electronics based on product information?",
        "category": "reasoning",
        "difficulty": "hard"
    },
    {
        "id": 14,
        "question": "Find the best value-for-money products in the Computer Accessories category.",
        "category": "recommendation",
        "difficulty": "medium"
    },
    {
        "id": 15,
        "question": "What security features are mentioned in smart home device descriptions?",
        "category": "feature_extraction",
        "difficulty": "medium"
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
