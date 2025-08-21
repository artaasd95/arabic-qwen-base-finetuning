#!/usr/bin/env python3
"""
Comprehensive Training Script for All Methods
Supports SFT, DPO, KTO, IPO, and CPO training with CUDA optimization
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.sft_trainer import SFTTrainer
from src.training.dpo_trainer import DPOTrainer
from src.training.kto_trainer import KTOTrainer
from src.training.ipo_trainer import IPOTrainer
from src.training.cpo_trainer import CPOTrainer
from src.config.training_config import TrainingConfig
from src.data.data_loader import DataLoader
from src.utils.metrics import MetricsTracker
from src.utils.logger import setup_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveTrainer:
    """Comprehensive trainer for all methods"""
    
    def __init__(self, config_path: str, output_dir: str = "./outputs"):
        self.config = TrainingConfig.from_file(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger(
            name="comprehensive_trainer",
            log_file=self.output_dir / "training.log"
        )
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Training methods mapping
        self.trainers = {
            'sft': SFTTrainer,
            'dpo': DPOTrainer,
            'kto': KTOTrainer,
            'ipo': IPOTrainer,
            'cpo': CPOTrainer
        }
        
        # Results storage
        self.results = {}
        
    def setup_cuda(self):
        """Setup CUDA environment"""
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            self.logger.info(f"CUDA available with {device_count} device(s)")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                self.logger.info(f"Device {i}: {props.name} ({props.total_memory // 1024**3}GB)")
            
            # Set device
            device = torch.device(f"cuda:{self.config.device_id}" if hasattr(self.config, 'device_id') else "cuda:0")
            torch.cuda.set_device(device)
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            self.logger.info(f"Using device: {device}")
            return device
        else:
            self.logger.warning("CUDA not available, using CPU")
            return torch.device("cpu")
    
    def load_datasets(self, method: str) -> Dict[str, Any]:
        """Load datasets for specific method"""
        data_loader = DataLoader(self.config)
        
        if method == 'sft':
            return {
                'train': data_loader.load_sft_data("data/sft/arabic_sft_samples.json"),
                'eval': data_loader.load_sft_data("data/evaluation/arabic_eval_samples.json")
            }
        elif method in ['dpo', 'ipo', 'cpo']:
            return {
                'train': data_loader.load_preference_data("data/dpo/arabic_dpo_samples.json"),
                'eval': data_loader.load_preference_data("data/dpo/arabic_dpo_samples.json")  # Use same for eval
            }
        elif method == 'kto':
            return {
                'train': data_loader.load_kto_data("data/kto/arabic_kto_samples.json"),
                'eval': data_loader.load_kto_data("data/kto/arabic_kto_samples.json")  # Use same for eval
            }
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def train_method(self, method: str, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Train using specific method"""
        self.logger.info(f"Starting {method.upper()} training...")
        
        # Create method-specific output directory
        method_output_dir = self.output_dir / method
        method_output_dir.mkdir(exist_ok=True)
        
        # Update config for this method
        method_config = self.config.copy()
        method_config.output_dir = str(method_output_dir)
        method_config.method = method
        
        # Initialize trainer
        trainer_class = self.trainers[method]
        trainer = trainer_class(method_config)
        
        # Start timing
        start_time = time.time()
        
        try:
            # Train the model
            training_results = trainer.train(
                train_dataset=datasets['train'],
                eval_dataset=datasets['eval']
            )
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Collect results
            results = {
                'method': method,
                'status': 'completed',
                'training_time': training_time,
                'output_dir': str(method_output_dir),
                'metrics': training_results.get('metrics', {}),
                'best_checkpoint': training_results.get('best_checkpoint'),
                'final_loss': training_results.get('final_loss'),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"{method.upper()} training completed in {training_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"{method.upper()} training failed: {str(e)}")
            results = {
                'method': method,
                'status': 'failed',
                'error': str(e),
                'training_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
        
        return results
    
    def run_all_methods(self, methods: list = None) -> Dict[str, Any]:
        """Run training for all specified methods"""
        if methods is None:
            methods = ['sft', 'dpo', 'kto', 'ipo', 'cpo']
        
        self.logger.info(f"Starting comprehensive training for methods: {methods}")
        
        # Setup CUDA
        device = self.setup_cuda()
        
        # Train each method
        for method in methods:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Training Method: {method.upper()}")
            self.logger.info(f"{'='*50}")
            
            try:
                # Load datasets for this method
                datasets = self.load_datasets(method)
                self.logger.info(f"Loaded datasets for {method}")
                
                # Train the method
                results = self.train_method(method, datasets)
                self.results[method] = results
                
                # Save intermediate results
                self.save_results()
                
            except Exception as e:
                self.logger.error(f"Failed to train {method}: {str(e)}")
                self.results[method] = {
                    'method': method,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def save_results(self):
        """Save training results to file"""
        results_file = self.output_dir / "training_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def generate_summary(self):
        """Generate training summary"""
        summary = {
            'total_methods': len(self.results),
            'completed': len([r for r in self.results.values() if r['status'] == 'completed']),
            'failed': len([r for r in self.results.values() if r['status'] == 'failed']),
            'total_training_time': sum([r.get('training_time', 0) for r in self.results.values()]),
            'results_by_method': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_file = self.output_dir / "training_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Print summary
        self.logger.info("\n" + "="*60)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total Methods: {summary['total_methods']}")
        self.logger.info(f"Completed: {summary['completed']}")
        self.logger.info(f"Failed: {summary['failed']}")
        self.logger.info(f"Total Training Time: {summary['total_training_time']:.2f}s")
        
        for method, results in self.results.items():
            status = results['status']
            time_taken = results.get('training_time', 0)
            self.logger.info(f"  {method.upper()}: {status} ({time_taken:.2f}s)")
        
        self.logger.info(f"\nDetailed results saved to: {summary_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Comprehensive training for all methods")
    parser.add_argument("--config", default="configs/qwen3_config.yaml", help="Config file path")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")
    parser.add_argument("--methods", nargs="+", default=None, 
                       choices=['sft', 'dpo', 'kto', 'ipo', 'cpo'],
                       help="Methods to train (default: all)")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device ID")
    
    args = parser.parse_args()
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    
    # Create trainer
    trainer = ComprehensiveTrainer(
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    # Run training
    results = trainer.run_all_methods(args.methods)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE TRAINING COMPLETED")
    print("="*60)
    print(f"Results saved to: {trainer.output_dir}")
    
    # Print quick summary
    completed = len([r for r in results.values() if r['status'] == 'completed'])
    total = len(results)
    print(f"Success Rate: {completed}/{total} ({100*completed/total:.1f}%)")

if __name__ == "__main__":
    main()