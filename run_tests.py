#!/usr/bin/env python3
"""
Quick test runner to validate dynamic steering implementation.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --fast             # Skip slow tests
    python run_tests.py --verbose          # Verbose output
    python run_tests.py --coverage         # With coverage report
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_tests(fast=False, verbose=False, coverage=False, specific_test=None):
    """Run the test suite."""
    
    cmd = ["pytest", "tests/test_dynamic_steering.py"]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if fast:
        cmd.extend(["-m", "not slow"])
        print("⚡ Running fast tests only (skipping integration tests)")
    
    if coverage:
        cmd.extend([
            "--cov=src.activation_steering",
            "--cov-report=term-missing",
            "--cov-report=html"
        ])
        print("📊 Running with coverage analysis")
    
    if specific_test:
        cmd.append(f"::{specific_test}")
        print(f"🎯 Running specific test: {specific_test}")
    
    cmd.append("--tb=short")
    
    print("=" * 70)
    print("🧪 Running Dynamic Steering Tests")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode
    except FileNotFoundError:
        print("\n❌ Error: pytest not found. Install with:")
        print("   pip install pytest pytest-cov")
        return 1


def run_quick_validation():
    """Run a quick validation without pytest."""
    print("=" * 70)
    print("🔍 Quick Validation (without pytest)")
    print("=" * 70)
    
    try:
        # Test imports
        print("\n1. Testing imports...")
        from src.activation_steering import ActivationSteering, SteeringConfig
        print("   ✓ ActivationSteering imported")
        
        from src.config import get_config
        print("   ✓ Config imported")
        
        # Test basic initialization
        print("\n2. Testing basic initialization...")
        from unittest.mock import MagicMock
        import torch
        
        mock_model = MagicMock()
        mock_model.decoder.block = [MagicMock() for _ in range(12)]
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.tensor([1.0]))]
        
        mock_tokenizer = MagicMock()
        
        # Test single mode
        steerer_single = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="single",
            device=torch.device("cpu")
        )
        print("   ✓ Single-layer mode initialized")
        assert steerer_single.steering_mode == "single"
        
        # Test dynamic mode
        steerer_dynamic = ActivationSteering(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer=6,
            steering_mode="dynamic",
            layer_range=(3, 7),
            device=torch.device("cpu")
        )
        print("   ✓ Dynamic mode initialized")
        assert steerer_dynamic.steering_mode == "dynamic"
        
        # Test layer selection
        print("\n3. Testing layer selection...")
        steerer_dynamic.steering_vectors[5] = torch.randn(768)
        steerer_dynamic.steering_vectors[6] = torch.randn(768)
        
        layers = steerer_dynamic.select_layers({"retrieval_score": 0.9})
        print(f"   ✓ High confidence selects: {layers}")
        assert 5 in layers or 6 in layers
        
        layers = steerer_dynamic.select_layers({"retrieval_score": 0.4})
        print(f"   ✓ Low confidence selects: {layers}")
        
        # Test time decay
        print("\n4. Testing time-aware decay...")
        mult_early = steerer_dynamic.get_layer_multiplier(6, 2.0, 0)
        mult_late = steerer_dynamic.get_layer_multiplier(6, 2.0, 60)
        print(f"   ✓ Early multiplier: {mult_early:.2f}")
        print(f"   ✓ Late multiplier: {mult_late:.2f}")
        assert mult_early > mult_late
        
        # Test vector management
        print("\n5. Testing vector management...")
        vectors = {4: torch.randn(768), 5: torch.randn(768)}
        steerer_dynamic.set_steering_vectors(vectors)
        print(f"   ✓ Set {len(vectors)} vectors")
        assert len(steerer_dynamic.steering_vectors) >= 2
        
        print("\n" + "=" * 70)
        print("✅ Quick validation passed!")
        print("=" * 70)
        print("\nRun full test suite with: python run_tests.py")
        return 0
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run dynamic steering tests")
    parser.add_argument("--fast", action="store_true", help="Skip slow integration tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--test", "-t", type=str, help="Run specific test (e.g., TestDynamicLayerSelection)")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick validation without pytest")
    
    args = parser.parse_args()
    
    if args.quick:
        return run_quick_validation()
    else:
        return run_tests(
            fast=args.fast,
            verbose=args.verbose,
            coverage=args.coverage,
            specific_test=args.test
        )


if __name__ == "__main__":
    sys.exit(main())
