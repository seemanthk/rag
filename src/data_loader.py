"""
Data loader module for Amazon Products dataset
Handles loading, preprocessing, and chunking of product data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AmazonProductDataLoader:
    """Load and preprocess Amazon product data for RAG"""

    def __init__(self, csv_path: str):
        """
        Initialize data loader

        Args:
            csv_path: Path to Amazon products CSV file
        """
        self.csv_path = csv_path
        self.df = None
        self.documents = []

    def load_data(self) -> pd.DataFrame:
        """Load CSV data"""
        logger.info(f"Loading data from {self.csv_path}")
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.df)} products")
            logger.info(f"Columns: {self.df.columns.tolist()}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self) -> pd.DataFrame:
        """Clean and preprocess the data"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Preprocessing data...")

        # Remove duplicates
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        logger.info(f"Removed {initial_count - len(self.df)} duplicates")

        # Handle missing values
        # Fill NaN with empty string for text columns
        text_columns = self.df.select_dtypes(include=['object']).columns
        self.df[text_columns] = self.df[text_columns].fillna('')

        # Fill NaN with 0 for numeric columns
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_columns] = self.df[numeric_columns].fillna(0)

        logger.info("Preprocessing complete")
        return self.df

    def create_documents(self,
                        include_columns: Optional[List[str]] = None) -> List[Dict]:
        """
        Create document chunks from product data

        Args:
            include_columns: Specific columns to include. If None, use all relevant columns

        Returns:
            List of document dictionaries with text and metadata
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Creating document chunks...")

        self.documents = []

        for idx, row in self.df.iterrows():
            # Create a comprehensive text representation of the product
            doc_text = self._create_product_text(row, include_columns)

            # Create metadata
            metadata = {
                'product_id': str(row.get('asin', idx)) if 'asin' in row else str(idx),
                'index': idx
            }

            # Add additional metadata fields if available
            if 'category' in row:
                metadata['category'] = str(row['category'])
            if 'rating' in row:
                metadata['rating'] = float(row['rating']) if pd.notna(row['rating']) else 0.0
            if 'price' in row or 'actual_price' in row:
                price_col = 'price' if 'price' in row else 'actual_price'
                metadata['price'] = float(row[price_col]) if pd.notna(row[price_col]) else 0.0

            document = {
                'text': doc_text,
                'metadata': metadata
            }

            self.documents.append(document)

        logger.info(f"Created {len(self.documents)} documents")
        return self.documents

    def _create_product_text(self, row: pd.Series,
                            include_columns: Optional[List[str]] = None) -> str:
        """
        Create a text representation of a product

        Args:
            row: Product data row
            include_columns: Specific columns to include

        Returns:
            Formatted text string
        """
        text_parts = []

        # Define common column mappings (different datasets may use different names)
        column_mappings = {
            'name': ['product_name', 'name', 'title', 'product_title'],
            'category': ['category', 'main_category', 'category_name'],
            'price': ['price', 'actual_price', 'discounted_price'],
            'rating': ['rating', 'ratings', 'average_rating'],
            'reviews': ['review_content', 'reviews', 'review_text'],
            'description': ['about_product', 'description', 'product_description'],
        }

        # Product name/title
        for col in column_mappings['name']:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                text_parts.append(f"Product: {row[col]}")
                break

        # Category
        for col in column_mappings['category']:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                text_parts.append(f"Category: {row[col]}")
                break

        # Price
        for col in column_mappings['price']:
            if col in row and pd.notna(row[col]):
                text_parts.append(f"Price: ${row[col]}")
                break

        # Rating
        for col in column_mappings['rating']:
            if col in row and pd.notna(row[col]):
                text_parts.append(f"Rating: {row[col]} stars")
                break

        # Description
        for col in column_mappings['description']:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                text_parts.append(f"Description: {row[col]}")
                break

        # Reviews
        for col in column_mappings['reviews']:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                # Truncate very long reviews
                review_text = str(row[col])
                if len(review_text) > 500:
                    review_text = review_text[:500] + "..."
                text_parts.append(f"Reviews: {review_text}")
                break

        # Add any other relevant columns
        if include_columns:
            for col in include_columns:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    if col not in [item for sublist in column_mappings.values() for item in sublist]:
                        text_parts.append(f"{col}: {row[col]}")

        return " | ".join(text_parts)

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        stats = {
            'total_products': len(self.df),
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
        }

        # Add category distribution if available
        if 'category' in self.df.columns:
            stats['category_distribution'] = self.df['category'].value_counts().head(10).to_dict()

        # Add rating distribution if available
        if 'rating' in self.df.columns:
            stats['avg_rating'] = float(self.df['rating'].mean())
            stats['rating_distribution'] = self.df['rating'].value_counts().sort_index().to_dict()

        return stats
