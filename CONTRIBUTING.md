# Contributing to SIGtor

Contributions are welcome and appreciated! This document provides guidelines for contributing to SIGtor.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/solomontesema/sigtor.git`
3. Create a branch for your changes: `git checkout -b feature/your-feature-name`
4. Install in development mode: `pip install -e ".[dev]"`

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings for all public functions and classes
- Keep functions focused and modular

## Making Changes

1. **Small, focused changes**: Keep pull requests focused on a single feature or bug fix
2. **Write tests**: Add tests for new features or bug fixes
3. **Update documentation**: Update README.md or relevant docs if needed
4. **Test your changes**: Run the test suite before submitting

## Submitting Changes

1. Commit your changes with clear, descriptive commit messages
2. Push to your fork: `git push origin feature/your-feature-name`
3. Open a Pull Request on GitHub
4. Provide a clear description of your changes

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=. tests/
```

## Reporting Issues

When reporting issues, please include:
- Description of the problem
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)

## Feature Requests

Feature requests are welcome! Please open an issue describing:
- The feature you'd like to see
- Use case or motivation
- Potential implementation approach (if you have ideas)

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

Thank you for contributing to SIGtor!

