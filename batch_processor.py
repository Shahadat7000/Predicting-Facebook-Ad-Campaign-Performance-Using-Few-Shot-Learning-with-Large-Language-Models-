"""
Batch Processing Module for Large-Scale Experiments
Handles concurrent processing with rate limiting
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
import queue
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Handles batch processing with rate limiting and error recovery
    """
    
    def __init__(self, 
                 max_workers: int = 3,
                 rate_limit: int = 10,  # requests per minute
                 retry_attempts: int = 3):
        """
        Initialize batch processor
        
        Args:
            max_workers: Maximum concurrent threads
            rate_limit: Maximum requests per minute
            retry_attempts: Number of retry attempts for failed requests
        """
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.retry_attempts = retry_attempts
        
        # Rate limiting
        self.request_times = []
        self.semaphore = Semaphore(max_workers)
        
        # Tracking
        self.successful = 0
        self.failed = 0
        self.retried = 0
        
    def _check_rate_limit(self):
        """Ensure we don't exceed rate limit"""
        now = time.time()
        # Remove requests older than 60 seconds
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        self.request_times.append(now)
    
    def process_item(self, 
                     item: Any, 
                     process_func: Callable,
                     item_id: Optional[str] = None) -> Dict:
        """
        Process a single item with retry logic
        
        Args:
            item: Item to process
            process_func: Function to process item
            item_id: Identifier for the item
            
        Returns:
            Dict: Processing result
        """
        self._check_rate_limit()
        
        for attempt in range(self.retry_attempts):
            try:
                result = process_func(item)
                self.successful += 1
                
                return {
                    'success': True,
                    'item_id': item_id,
                    'result': result,
                    'attempts': attempt + 1
                }
                
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed for {item_id}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
                    self.retried += 1
                else:
                    self.failed += 1
                    logger.error(f"All attempts failed for {item_id}: {e}")
                    
                    return {
                        'success': False,
                        'item_id': item_id,
                        'error': str(e),
                        'attempts': attempt + 1
                    }
    
    def process_batch(self,
                      items: List[Any],
                      process_func: Callable,
                      item_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Process a batch of items concurrently
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            item_ids: Optional list of identifiers
            
        Returns:
            List[Dict]: Processing results
        """
        if item_ids is None:
            item_ids = [f"item_{i}" for i in range(len(items))]
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self.process_item, item, process_func, item_id): (item, item_id)
                for item, item_id in zip(items, item_ids)
            }
            
            # Collect results
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    item, item_id = future_to_item[future]
                    logger.error(f"Unexpected error processing {item_id}: {e}")
                    results.append({
                        'success': False,
                        'item_id': item_id,
                        'error': str(e),
                        'attempts': self.retry_attempts
                    })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'successful': self.successful,
            'failed': self.failed,
            'retried': self.retried,
            'total': self.successful + self.failed,
            'success_rate': self.successful / (self.successful + self.failed) if (self.successful + self.failed) > 0 else 0
        }

class ExperimentCheckpoint:
    """
    Saves and loads experiment checkpoints for recovery
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, 
                        data: Dict, 
                        checkpoint_name: str,
                        metadata: Optional[Dict] = None):
        """
        Save experiment checkpoint
        
        Args:
            data: Data to save
            checkpoint_name: Name of checkpoint
            metadata: Additional metadata
        """
        checkpoint = {
            'data': data,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = self.checkpoint_dir / f"{checkpoint_name}.json"
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=4, default=str)
        
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, checkpoint_name: str) -> Optional[Dict]:
        """
        Load experiment checkpoint
        
        Args:
            checkpoint_name: Name of checkpoint
            
        Returns:
            Optional[Dict]: Checkpoint data if exists
        """
        filepath = self.checkpoint_dir / f"{checkpoint_name}.json"
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Checkpoint loaded from {filepath}")
            return checkpoint
        
        logger.warning(f"No checkpoint found at {filepath}")
        return None
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints"""
        return [f.stem for f in self.checkpoint_dir.glob("*.json")]

def create_batch_processor_for_experiment(experiment, max_workers: int = 3):
    """
    Create a batch processor configured for the experiment
    
    Args:
        experiment: GPTExperiment instance
        max_workers: Maximum concurrent workers
        
    Returns:
        BatchProcessor: Configured batch processor
    """
    processor = BatchProcessor(
        max_workers=max_workers,
        rate_limit=10,  # OpenAI rate limit
        retry_attempts=3
    )
    
    return processor