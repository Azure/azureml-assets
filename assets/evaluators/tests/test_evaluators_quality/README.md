# Quality Tests for Evaluators

This directory contains quality tests for evaluators that execute real LLM flows without mocking.

## Overview

Quality tests validate the actual behavior of evaluators by making real calls to Azure OpenAI models, ensuring that the evaluators work correctly in production scenarios.

## Key Differences from Behavior Tests

| Aspect | Behavior Tests | Quality Tests |
|--------|----------------|---------------|
| Flow Execution | Mocked with `MagicMock` | Real LLM calls |
| Purpose | Test input validation and error handling | Test actual evaluation quality |
| Speed | Fast | Slower (due to API calls) |
| Dependencies | None (all mocked) | Requires valid Azure OpenAI credentials |

## Structure

### Base Classes

- **`BaseEvaluatorRunner`** (`common/base_evaluator_runner.py`): 
  - Core base class for all evaluator tests
  - Controlled by `use_mocking` flag (default: `True`)
  - Handles both mocked and real evaluator initialization
  - Provides assertion methods: `assert_pass()`, `assert_fail()`, `assert_pass_or_fail()`

- **`BaseQualityEvaluatorRunner`** (`common/base_quality_evaluator_runner.py`): 
  - Thin wrapper around `BaseEvaluatorRunner`
  - Sets `use_mocking = False` for real LLM calls
  - Use this for quality tests to avoid repeating the flag and for clearer intent

### Test Files

- **`test_groundedness_evaluator_quality.py`**: Quality tests for Groundedness evaluator
- **`test_coherence_evaluator_quality.py`**: Quality tests for Coherence evaluator (stub)
- **`test_relevance_evaluator_quality.py`**: Quality tests for Relevance evaluator (stub)

## Prerequisites

### Environment Variables

Set the following environment variables before running quality tests:

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"
```

### Authentication

Quality tests use `DefaultAzureCredential()` for authentication. Ensure you have valid Azure credentials configured (e.g., via Azure CLI login, managed identity, or environment variables).

## Running Tests

### Run all quality tests
```bash
pytest assets/evaluators/tests/test_evaluators_quality/ -m quality
```

### Run specific evaluator quality tests
```bash
pytest assets/evaluators/tests/test_evaluators_quality/test_groundedness_evaluator_quality.py -m quality
```

### Run with verbose output
```bash
pytest assets/evaluators/tests/test_evaluators_quality/ -m quality -v
```

## Writing New Quality Tests

To create a new quality test:

1. **Create test file**: `test_<evaluator_name>_evaluator_quality.py`

2. **Import required classes**:
```python
import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner
from ...builtin.<evaluator_name>.evaluator._<evaluator_name> import <EvaluatorClass>
```

3. **Define test class**:
```python
@pytest.mark.quality
class Test<EvaluatorName>EvaluatorQuality(BaseQualityEvaluatorRunner):
    evaluator_type = <EvaluatorClass>
    # No need to set use_mocking - it's already False in the base class
    
    def test_case_name(self):
        # Prepare test data
        query = [...]
        response = [...]
        
        # Run evaluation (real LLM call)
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Test case description")
        
        # Assert expected outcome
        self.assert_pass(result_data)  # or assert_fail, assert_pass_or_fail
```

4. **Update `__init__.py`**: Add your test class to the exports

## Test Case Guidelines

### Input Format

Use standard conversation format:
```python
query = [
    {"role": "system", "content": "System message"},
    {"role": "user", "content": [{"type": "text", "text": "User message"}]}
]

response = [
    {"role": "assistant", "content": [{"type": "text", "text": "Assistant response"}]}
]
```

### Assertions

- **`assert_pass(result_data)`**: Expects `label == "pass"` and `score >= 1.0`
- **`assert_fail(result_data)`**: Expects `label == "fail"` and `score == 0.0`
- **`assert_pass_or_fail(result_data)`**: Accepts either pass or fail with valid score

### Test Markers

All quality tests should be marked with `@pytest.mark.quality` to enable selective test execution.

## Example Test Cases

### Groundedness - Hallucinated Response (Expected: Fail)
```python
def test_fail_hallucinated_response(self):
    query = [...]  # User asks about ungrounded information
    response = [...]  # Assistant provides hallucinated content
    
    results = self._run_evaluation(query=query, response=response)
    result_data = self._extract_and_print_result(results, "FAIL-Hallucinated response")
    
    self.assert_fail(result_data)
```

### Groundedness - Grounded Response (Expected: Pass)
```python
def test_success_grounded_response(self):
    query = [...]  # User asks a question
    response = [...]  # Assistant provides grounded response with tool results
    
    results = self._run_evaluation(query=query, response=response)
    result_data = self._extract_and_print_result(results, "SUCCESS-Grounded response")
    
    self.assert_pass(result_data)
```

## Troubleshooting

### Authentication Errors
- Ensure you're logged in to Azure CLI: `az login`
- Verify your credentials have access to the Azure OpenAI resource
- Check that environment variables are correctly set

### API Errors
- Verify the deployment name matches your Azure OpenAI deployment
- Ensure the API version is supported by your deployment
- Check rate limits and quotas for your Azure OpenAI resource

### Timeout Issues
- Quality tests make real API calls and may take longer to execute
- Consider increasing pytest timeout: `pytest --timeout=300`

## Best Practices

1. **Use descriptive test names**: Clearly indicate what is being tested and expected outcome
2. **Print results**: Use `_extract_and_print_result()` for debugging and visibility
3. **Test edge cases**: Include tests for boundary conditions and error scenarios
4. **Document test cases**: Add docstrings explaining the purpose and expected behavior
5. **Keep tests focused**: Each test should validate a specific aspect of evaluator behavior
6. **Use realistic data**: Test with actual conversation formats and realistic content

## Contributing

When adding new quality tests:
1. Follow the existing structure and naming conventions
2. Add comprehensive test cases covering various scenarios
3. Update this README with any new patterns or considerations
4. Ensure tests pass locally before submitting
