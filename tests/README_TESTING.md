# Testing Dynamic Steering Implementation

## Running Tests

### Quick Start

```bash
# Run all tests
pytest tests/test_dynamic_steering.py -v

# Run with coverage
pytest tests/test_dynamic_steering.py --cov=src.activation_steering --cov-report=html

# Run specific test class
pytest tests/test_dynamic_steering.py::TestDynamicLayerSelection -v

# Run fast tests only (skip integration tests with real models)
pytest tests/test_dynamic_steering.py -v -m "not slow"
```

### Test Organization

The test file is organized into test classes:

1. **TestBackwardCompatibility** - Ensures single-layer mode works unchanged
2. **TestDynamicLayerSelection** - Tests adaptive layer selection logic
3. **TestTimeAwareDecay** - Tests steering strength decay over generation
4. **TestHookManagement** - Tests memory safety (hook registration/removal)
5. **TestMultiLayerVectors** - Tests multi-layer vector operations
6. **TestCheckpointSaveLoad** - Tests persistence and checkpoint format
7. **TestEdgeCases** - Tests error handling and edge cases
8. **TestIntegration** - Integration tests with real T5-small model (slow)
9. **TestSteeringHook** - Tests the steering hook function itself

### Test Coverage

**What's Tested:**

✅ Single-layer mode (backward compatibility)  
✅ Multi-layer dynamic mode  
✅ Layer selection for high/medium/low confidence  
✅ Time-aware steering decay  
✅ Layer-specific multipliers  
✅ Hook registration and cleanup  
✅ Memory safety (hooks always removed)  
✅ Checkpoint save/load (single and multi-layer)  
✅ Backward compatibility with old checkpoints  
✅ Edge cases (no vectors, out-of-range layers, empty selection)  
✅ Integration with real T5 model  

### Expected Results

All tests should pass. Example output:

```
tests/test_dynamic_steering.py::TestBackwardCompatibility::test_single_layer_initialization PASSED
tests/test_dynamic_steering.py::TestBackwardCompatibility::test_single_layer_returns_default PASSED
tests/test_dynamic_steering.py::TestDynamicLayerSelection::test_high_confidence_selection PASSED
tests/test_dynamic_steering.py::TestDynamicLayerSelection::test_medium_confidence_selection PASSED
...
========================= 30 passed in 2.45s =========================
```

### Running Integration Tests

Integration tests use real T5-small model and are marked as `@pytest.mark.slow`:

```bash
# Run only integration tests
pytest tests/test_dynamic_steering.py::TestIntegration -v

# Skip integration tests (faster)
pytest tests/test_dynamic_steering.py -v -m "not slow"
```

**Note:** Integration tests download T5-small (~240MB) on first run.

### Debugging Failed Tests

If tests fail, use verbose output:

```bash
# Full traceback
pytest tests/test_dynamic_steering.py -v --tb=long

# Stop on first failure
pytest tests/test_dynamic_steering.py -x

# Run specific test
pytest tests/test_dynamic_steering.py::TestDynamicLayerSelection::test_high_confidence_selection -v
```

### Prerequisites

Install test dependencies:

```bash
pip install pytest pytest-cov
```

### Key Test Scenarios

#### 1. Backward Compatibility
```python
def test_single_layer_returns_default():
    # Single mode always returns default layer
    steerer = ActivationSteering(..., steering_mode="single")
    layers = steerer.select_layers({"retrieval_score": 0.9})
    assert layers == [6]  # Always default
```

#### 2. Dynamic Layer Selection
```python
def test_high_confidence_selection():
    # High confidence → middle layers only
    runtime_state = {"retrieval_score": 0.85}
    layers = steerer.select_layers(runtime_state)
    assert set(layers) == {5, 6}
```

#### 3. Memory Safety
```python
def test_hooks_removed_on_exception():
    # Hooks removed even if exception occurs
    try:
        with steerer.apply(multiplier=2.0):
            raise ValueError("Test")
    except ValueError:
        pass
    assert len(steerer.hook_handles) == 0  ✅
```

#### 4. Checkpoint Compatibility
```python
def test_backward_compat_old_checkpoint():
    # Old single-layer checkpoint loads in new code
    checkpoint = {"steering_vector": vector, "layer": 6}
    steerer.load(checkpoint_path)
    assert steerer.steering_vector is not None  ✅
    assert 6 in steerer.steering_vectors  ✅
```

### CI/CD Integration

Add to GitHub Actions workflow:

```yaml
- name: Run dynamic steering tests
  run: |
    pytest tests/test_dynamic_steering.py -v --cov=src.activation_steering
    pytest tests/test_dynamic_steering.py -m "not slow"  # Skip slow tests in CI
```

### Manual Testing

For interactive testing:

```python
# In Python REPL
from tests.test_dynamic_steering import *
import pytest

# Run single test
test_obj = TestDynamicLayerSelection()
test_obj.test_high_confidence_selection(mock_model(), mock_tokenizer())
```

### Troubleshooting

**Issue: "No module named 'src'"**
```bash
# Run from project root
cd e:\code\contribution\B-PLIS-rag
pytest tests/test_dynamic_steering.py
```

**Issue: "Mock object has no attribute 'register_forward_hook'"**
- This is expected for mocked models
- Real hooks tested in integration tests
- Unit tests focus on logic, not PyTorch internals

**Issue: Integration tests timeout**
```bash
# Increase timeout
pytest tests/test_dynamic_steering.py::TestIntegration --timeout=300
```

### Performance Benchmarks

Approximate test execution times:

| Test Class | Duration | Notes |
|-----------|----------|-------|
| Unit tests (all except integration) | ~2s | Fast |
| Integration tests | ~30s | Downloads T5-small once |
| Full suite | ~35s | Includes all tests |

### Test Metrics

Expected coverage: **>95%** of `src/activation_steering.py`

Critical paths covered:
- ✅ `select_layers()` - All branches
- ✅ `get_layer_multiplier()` - All conditions
- ✅ `apply()` context manager - Normal and exception paths
- ✅ `_steering_hook()` - Active and inactive states
- ✅ `save()`/`load()` - All checkpoint formats

### Next Steps

After tests pass:

1. **Run on real data:**
   ```bash
   python examples/dynamic_steering_example.py
   ```

2. **Benchmark performance:**
   ```bash
   python scripts/evaluate.py --use-steering --steering-mode dynamic
   ```

3. **Compare single vs dynamic:**
   ```bash
   # Run A/B test script (create if needed)
   python scripts/compare_steering_modes.py
   ```

---

**Questions?** Check [DYNAMIC_STEERING_GUIDE.md](DYNAMIC_STEERING_GUIDE.md) for usage details.
