"""
Dataset preprocessing script for LLM fine-tuning.

This script loads raw dataset from data/raw/ and preprocesses it,
saving the cleaned data to data/processed/.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetPreprocessor:
    """Preprocesses raw dataset for LLM fine-tuning."""
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        Initialize the preprocessor.
        
        Args:
            raw_data_path: Path to raw dataset JSON file
            processed_data_path: Path to save processed dataset
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """Load raw dataset from JSON file."""
        logger.info(f"Loading raw data from {self.raw_data_path}")
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.raw_data_path}")
        
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} raw samples")
        return data
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Strip whitespace
        text = text.strip()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate if a sample has required fields.
        
        Args:
            sample: Sample to validate
            
        Returns:
            True if sample is valid, False otherwise
        """
        required_fields = ['instruction', 'output']
        
        for field in required_fields:
            if field not in sample:
                return False
            if not isinstance(sample[field], str):
                return False
            if not sample[field].strip():
                return False
        
        return True
    
    def preprocess_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """
        Preprocess a single sample.
        
        Args:
            sample: Raw sample
            
        Returns:
            Preprocessed sample
        """
        return {
            'instruction': self.clean_text(sample.get('instruction', '')),
            'input': self.clean_text(sample.get('input', '')),
            'output': self.clean_text(sample.get('output', ''))
        }
    
    def preprocess(self) -> Dict[str, Any]:
        """
        Execute full preprocessing pipeline.
        
        Returns:
            Dictionary with preprocessing statistics
        """
        # Load raw data
        raw_data = self.load_raw_data()
        
        # Preprocess samples
        processed_samples = []
        skipped_count = 0
        
        for idx, sample in enumerate(raw_data):
            if not self.validate_sample(sample):
                logger.warning(f"Skipping invalid sample at index {idx}")
                skipped_count += 1
                continue
            
            processed_sample = self.preprocess_sample(sample)
            processed_samples.append(processed_sample)
        
        # Save processed data
        self._save_processed_data(processed_samples)
        
        # Log statistics
        stats = {
            'total_raw_samples': len(raw_data),
            'valid_samples': len(processed_samples),
            'skipped_samples': skipped_count,
            'output_path': str(self.processed_data_path)
        }
        
        self._log_statistics(stats)
        
        return stats
    
    def _save_processed_data(self, processed_samples: List[Dict[str, str]]):
        """Save processed data to JSON file."""
        logger.info(f"Saving {len(processed_samples)} processed samples to {self.processed_data_path}")
        
        with open(self.processed_data_path, 'w', encoding='utf-8') as f:
            json.dump(processed_samples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved processed dataset")
    
    def _log_statistics(self, stats: Dict[str, Any]):
        """Log preprocessing statistics."""
        logger.info("=" * 50)
        logger.info("PREPROCESSING STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total raw samples: {stats['total_raw_samples']}")
        logger.info(f"Valid samples: {stats['valid_samples']}")
        logger.info(f"Skipped samples: {stats['skipped_samples']}")
        logger.info(f"Output path: {stats['output_path']}")
        logger.info("=" * 50)


def main():
    """Main entry point for preprocessing script."""
    parser = argparse.ArgumentParser(
        description='Preprocess dataset for LLM fine-tuning'
    )
    parser.add_argument(
        '--raw-data',
        type=str,
        default='data/raw/dataset.json',
        help='Path to raw dataset JSON file'
    )
    parser.add_argument(
        '--processed-data',
        type=str,
        default='data/processed/dataset.json',
        help='Path to save processed dataset'
    )
    
    args = parser.parse_args()
    
    # Create preprocessor and run
    preprocessor = DatasetPreprocessor(args.raw_data, args.processed_data)
    stats = preprocessor.preprocess()
    
    return stats


if __name__ == '__main__':
    main()
