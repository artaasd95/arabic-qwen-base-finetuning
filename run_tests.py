#!/usr/bin/env python3
"""Test Runner Script

This script provides a convenient way to run tests with different configurations.
"""

import argparse
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Error: Command not found. Make sure pytest is installed.")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run tests for Arabic Qwen Base Fine-tuning project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --unit             # Run only unit tests
  python run_tests.py --integration      # Run only integration tests
  python run_tests.py --coverage         # Run tests with coverage
  python run_tests.py --module config    # Run tests for config module
  python run_tests.py --fast             # Run tests without slow tests
  python run_tests.py --verbose          # Run with verbose output
        """
    )
    
    # Test selection options
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="Run only integration tests"
    )
    parser.add_argument(
        "--module", 
        choices=["config", "data", "training", "evaluation", "utils"],
        help="Run tests for specific module"
    )
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Skip slow tests"
    )
    
    # Output options
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run tests with coverage report"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true", 
        help="Quiet output"
    )
    
    # Execution options
    parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--failfast", 
        action="store_true", 
        help="Stop on first failure"
    )
    parser.add_argument(
        "--lf", 
        action="store_true", 
        help="Run last failed tests only"
    )
    
    # Additional options
    parser.add_argument(
        "--install-deps", 
        action="store_true", 
        help="Install test dependencies before running tests"
    )
    parser.add_argument(
        "--clean", 
        action="store_true", 
        help="Clean test artifacts before running"
    )
    
    args = parser.parse_args()
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    if not (project_root / "tests").exists():
        print("Error: tests directory not found. Make sure you're in the project root.")
        sys.exit(1)
    
    # Install dependencies if requested
    if args.install_deps:
        deps_cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"]
        if not run_command(deps_cmd, "Installing test dependencies"):
            print("Failed to install dependencies")
            sys.exit(1)
    
    # Clean artifacts if requested
    if args.clean:
        import shutil
        artifacts = [
            project_root / "reports",
            project_root / ".coverage",
            project_root / ".pytest_cache",
            project_root / "__pycache__"
        ]
        
        for artifact in artifacts:
            if artifact.exists():
                if artifact.is_dir():
                    shutil.rmtree(artifact)
                else:
                    artifact.unlink()
                print(f"Cleaned: {artifact}")
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add test selection
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    elif args.module:
        cmd.append(f"tests/test_{args.module}.py")
    
    # Add markers
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    # Add output options
    if args.verbose:
        cmd.append("-v")
    elif args.quiet:
        cmd.append("-q")
    
    # Add execution options
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    if args.failfast:
        cmd.append("-x")
    
    if args.lf:
        cmd.append("--lf")
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=html:reports/coverage",
            "--cov-report=term-missing",
            "--cov-report=xml:reports/coverage.xml"
        ])
    
    # Ensure reports directory exists
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Run tests
    success = run_command(cmd, "Running tests")
    
    if success:
        print("\n" + "="*60)
        print("✅ All tests passed successfully!")
        
        if args.coverage:
            coverage_html = reports_dir / "coverage" / "index.html"
            if coverage_html.exists():
                print(f"📊 Coverage report: {coverage_html.absolute()}")
        
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Some tests failed!")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()