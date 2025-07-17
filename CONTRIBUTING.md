# Contributing to Transformers Attention Viz

Thank you for your interest in contributing! We welcome contributions from everyone.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/transformers-attention-viz.git
   cd transformers-attention-viz
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```

2. Make your changes and add tests

3. Run tests:
   ```bash
   pytest tests/
   ```

4. Run linting:
   ```bash
   black attention_viz/
   flake8 attention_viz/
   isort attention_viz/
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

6. Push and create a pull request

## Code Style

- We use Black for code formatting
- Follow PEP 8
- Add type hints where possible
- Document all public functions

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage

## Adding New Visualizations

To add a new visualization type:

1. Create a new file in `attention_viz/visualizers/`
2. Inherit from base visualization class
3. Implement the `create()` method
4. Add to `__init__.py` exports
5. Add tests and documentation

## Adding Model Support

To support a new model:

1. Create adapter in `attention_viz/extractors/model_adapters.py`
2. Implement attention module detection
3. Handle model-specific attention patterns
4. Add tests with mock model

## Documentation

- Update docstrings for all changes
- Add examples to the notebook
- Update README if adding features

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add entry to CHANGELOG (if exists)
4. Request review from maintainers

## Questions?

Feel free to open an issue for any questions!