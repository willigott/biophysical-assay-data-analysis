# BADA Package Tests

This directory contains tests for the BADA (Biophysical Assay Data Analysis) package.

## Running Tests

To run all tests:

```bash
python -m pytest
```

To run tests with coverage information:

```bash
python -m pytest --cov=bada
```

To generate an HTML coverage report:

```bash
python -m pytest --cov=bada --cov-report=html
```

## Test Structure

- `conftest.py`: Contains shared fixtures for use across all tests
- `test_init.py`: Tests for proper package imports
- `test_models.py`: Tests for the data models
- `test_parsers.py`: Tests for file parsers
- `test_processing.py`: Tests for data processing functions
- `test_visualization.py`: Tests for visualization functions

## Adding New Tests

When adding new functionality to the package, please also add corresponding tests:

1. If the functionality fits into an existing module, add tests to the appropriate test file
2. If creating a new module, create a new test file named `test_<module_name>.py`
3. Use fixtures from `conftest.py` where appropriate
4. Follow the existing patterns for test organization 