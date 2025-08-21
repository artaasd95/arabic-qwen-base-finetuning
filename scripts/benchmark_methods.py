#!/usr/bin/env python3
"""
Benchmarking Script for Training Methods
Analyzes performance metrics and generates comparison reports
"""

import os
import json
import time
import torch
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Performance benchmarking for training methods"""
    
    def __init__(self, results_dir: str = "./outputs"):
        self.results_dir = Path(results_dir)
        self.benchmark_results = {}
        
    def load_training_results(self) -> Dict[str, Any]:
        """Load training results from file"""
        results_file = self.results_dir / "training_results.json"
        
        if not results_file.exists():
            logger.error(f"Training results not found: {results_file}")
            return {}
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded training results for {len(results)} methods")
            return results
        except Exception as e:
            logger.error(f"Failed to load training results: {e}")
            return {}
    
    def measure_system_performance(self) -> Dict[str, Any]:
        """Measure current system performance"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024**3)
            memory_used_gb = memory.used / (1024**3)
            memory_percent = memory.percent
            
            # GPU metrics (if available)
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    memory_total = props.total_memory / (1024**3)
                    
                    gpu_info[f"gpu_{i}"] = {
                        "name": props.name,
                        "total_memory_gb": memory_total,
                        "allocated_memory_gb": memory_allocated,
                        "reserved_memory_gb": memory_reserved,
                        "utilization_percent": (memory_allocated / memory_total) * 100
                    }
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "core_count": cpu_count
                },
                "memory": {
                    "total_gb": memory_total_gb,
                    "used_gb": memory_used_gb,
                    "usage_percent": memory_percent
                },
                "gpu": gpu_info,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to measure system performance: {e}")
            return {}
    
    def benchmark_inference_speed(self, method: str, model_path: str = None) -> Dict[str, float]:
        """Benchmark inference speed for a method"""
        logger.info(f"Benchmarking inference speed for {method}")
        
        try:
            # Create dummy model for testing
            if torch.cuda.is_available():
                device = torch.device("cuda")
                model = torch.nn.Linear(1024, 1024).to(device)
                input_tensor = torch.randn(1, 1024).to(device)
            else:
                device = torch.device("cpu")
                model = torch.nn.Linear(1024, 1024)
                input_tensor = torch.randn(1, 1024)
            
            model.eval()
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(input_tensor)
            
            # Benchmark
            num_runs = 100
            start_time = time.time()
            
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(input_tensor)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_inference = total_time / num_runs
            throughput = num_runs / total_time
            
            return {
                "avg_inference_time_ms": avg_time_per_inference * 1000,
                "throughput_inferences_per_sec": throughput,
                "total_benchmark_time_s": total_time,
                "num_runs": num_runs,
                "device": str(device)
            }
            
        except Exception as e:
            logger.error(f"Inference benchmark failed for {method}: {e}")
            return {
                "avg_inference_time_ms": 0,
                "throughput_inferences_per_sec": 0,
                "error": str(e)
            }
    
    def analyze_training_efficiency(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training efficiency metrics"""
        efficiency_analysis = {}
        
        for method, results in training_results.items():
            if results.get('status') != 'completed':
                continue
            
            training_time = results.get('training_time', 0)
            final_loss = results.get('final_loss', float('inf'))
            data_samples = results.get('data_samples', 0)
            epochs = results.get('epochs', 1)
            
            # Calculate efficiency metrics
            samples_per_second = (data_samples * epochs) / max(training_time, 0.001)
            loss_reduction_rate = 1.0 - final_loss if final_loss < 1.0 else 0
            efficiency_score = samples_per_second * loss_reduction_rate
            
            efficiency_analysis[method] = {
                "training_time_s": training_time,
                "final_loss": final_loss,
                "samples_per_second": samples_per_second,
                "loss_reduction_rate": loss_reduction_rate,
                "efficiency_score": efficiency_score,
                "data_samples": data_samples,
                "epochs": epochs
            }
        
        return efficiency_analysis
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmarking"""
        logger.info("Starting comprehensive benchmarking...")
        
        # Load training results
        training_results = self.load_training_results()
        
        if not training_results:
            logger.error("No training results found for benchmarking")
            return {}
        
        # System performance
        system_performance = self.measure_system_performance()
        
        # Training efficiency analysis
        efficiency_analysis = self.analyze_training_efficiency(training_results)
        
        # Inference benchmarks
        inference_benchmarks = {}
        for method in training_results.keys():
            inference_benchmarks[method] = self.benchmark_inference_speed(method)
        
        # Compile comprehensive results
        benchmark_results = {
            "system_performance": system_performance,
            "training_efficiency": efficiency_analysis,
            "inference_benchmarks": inference_benchmarks,
            "training_results": training_results,
            "benchmark_timestamp": datetime.now().isoformat(),
            "summary": self.generate_benchmark_summary(efficiency_analysis, inference_benchmarks)
        }
        
        # Save results
        self.save_benchmark_results(benchmark_results)
        
        return benchmark_results
    
    def generate_benchmark_summary(self, efficiency_analysis: Dict, inference_benchmarks: Dict) -> Dict[str, Any]:
        """Generate benchmark summary"""
        if not efficiency_analysis:
            return {"error": "No efficiency analysis data"}
        
        # Find best performing methods
        best_efficiency = max(efficiency_analysis.items(), key=lambda x: x[1]['efficiency_score'])
        fastest_training = min(efficiency_analysis.items(), key=lambda x: x[1]['training_time_s'])
        lowest_loss = min(efficiency_analysis.items(), key=lambda x: x[1]['final_loss'])
        
        # Inference performance
        best_inference = None
        if inference_benchmarks:
            valid_benchmarks = {k: v for k, v in inference_benchmarks.items() 
                              if 'throughput_inferences_per_sec' in v and v['throughput_inferences_per_sec'] > 0}
            if valid_benchmarks:
                best_inference = max(valid_benchmarks.items(), 
                                   key=lambda x: x[1]['throughput_inferences_per_sec'])
        
        summary = {
            "best_overall_efficiency": {
                "method": best_efficiency[0],
                "score": best_efficiency[1]['efficiency_score']
            },
            "fastest_training": {
                "method": fastest_training[0],
                "time_s": fastest_training[1]['training_time_s']
            },
            "lowest_final_loss": {
                "method": lowest_loss[0],
                "loss": lowest_loss[1]['final_loss']
            },
            "method_rankings": {
                "by_efficiency": sorted(efficiency_analysis.items(), 
                                       key=lambda x: x[1]['efficiency_score'], reverse=True),
                "by_speed": sorted(efficiency_analysis.items(), 
                                 key=lambda x: x[1]['training_time_s']),
                "by_loss": sorted(efficiency_analysis.items(), 
                                key=lambda x: x[1]['final_loss'])
            }
        }
        
        if best_inference:
            summary["best_inference_speed"] = {
                "method": best_inference[0],
                "throughput": best_inference[1]['throughput_inferences_per_sec']
            }
        
        return summary
    
    def save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to file"""
        benchmark_file = self.results_dir / "benchmark_results.json"
        
        try:
            with open(benchmark_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Benchmark results saved to {benchmark_file}")
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
    
    def create_performance_plots(self, benchmark_results: Dict[str, Any]):
        """Create performance visualization plots"""
        try:
            import matplotlib.pyplot as plt
            
            efficiency_data = benchmark_results.get('training_efficiency', {})
            inference_data = benchmark_results.get('inference_benchmarks', {})
            
            if not efficiency_data:
                logger.warning("No efficiency data for plotting")
                return
            
            # Create plots directory
            plots_dir = self.results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Plot 1: Training Time Comparison
            methods = list(efficiency_data.keys())
            training_times = [efficiency_data[m]['training_time_s'] for m in methods]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(methods, training_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            plt.title('Training Time Comparison by Method', fontsize=14, fontweight='bold')
            plt.xlabel('Training Method')
            plt.ylabel('Training Time (seconds)')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, time_val in zip(bars, training_times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{time_val:.2f}s', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / "training_time_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Final Loss Comparison
            final_losses = [efficiency_data[m]['final_loss'] for m in methods]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(methods, final_losses, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            plt.title('Final Loss Comparison by Method', fontsize=14, fontweight='bold')
            plt.xlabel('Training Method')
            plt.ylabel('Final Loss')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, loss_val in zip(bars, final_losses):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{loss_val:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / "final_loss_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 3: Efficiency Score Comparison
            efficiency_scores = [efficiency_data[m]['efficiency_score'] for m in methods]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(methods, efficiency_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            plt.title('Training Efficiency Score by Method', fontsize=14, fontweight='bold')
            plt.xlabel('Training Method')
            plt.ylabel('Efficiency Score')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars, efficiency_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{score:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / "efficiency_score_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 4: Inference Speed Comparison (if available)
            if inference_data:
                valid_inference = {k: v for k, v in inference_data.items() 
                                 if 'throughput_inferences_per_sec' in v and v['throughput_inferences_per_sec'] > 0}
                
                if valid_inference:
                    inf_methods = list(valid_inference.keys())
                    throughputs = [valid_inference[m]['throughput_inferences_per_sec'] for m in inf_methods]
                    
                    plt.figure(figsize=(10, 6))
                    bars = plt.bar(inf_methods, throughputs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(inf_methods)])
                    plt.title('Inference Throughput Comparison by Method', fontsize=14, fontweight='bold')
                    plt.xlabel('Training Method')
                    plt.ylabel('Throughput (inferences/sec)')
                    plt.xticks(rotation=45)
                    
                    # Add value labels on bars
                    for bar, throughput in zip(bars, throughputs):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                f'{throughput:.1f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.savefig(plots_dir / "inference_throughput_comparison.png", dpi=300, bbox_inches='tight')
                    plt.close()
            
            logger.info(f"Performance plots saved to {plots_dir}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot generation")
        except Exception as e:
            logger.error(f"Failed to create plots: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark training methods")
    parser.add_argument("--results-dir", default="./outputs", help="Results directory")
    parser.add_argument("--create-plots", action="store_true", help="Create visualization plots")
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = PerformanceBenchmark(args.results_dir)
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    if not results:
        print("No benchmark results generated")
        return
    
    # Create plots if requested
    if args.create_plots:
        benchmark.create_performance_plots(results)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    summary = results.get('summary', {})
    
    if 'best_overall_efficiency' in summary:
        best_eff = summary['best_overall_efficiency']
        print(f"Best Overall Efficiency: {best_eff['method']} (score: {best_eff['score']:.2f})")
    
    if 'fastest_training' in summary:
        fastest = summary['fastest_training']
        print(f"Fastest Training: {fastest['method']} ({fastest['time_s']:.2f}s)")
    
    if 'lowest_final_loss' in summary:
        lowest_loss = summary['lowest_final_loss']
        print(f"Lowest Final Loss: {lowest_loss['method']} ({lowest_loss['loss']:.4f})")
    
    if 'best_inference_speed' in summary:
        best_inf = summary['best_inference_speed']
        print(f"Best Inference Speed: {best_inf['method']} ({best_inf['throughput']:.1f} inf/sec)")
    
    print(f"\nDetailed results saved to: {args.results_dir}/benchmark_results.json")
    
    if args.create_plots:
        print(f"Visualization plots saved to: {args.results_dir}/plots/")

if __name__ == "__main__":
    main()