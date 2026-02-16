# Contributing Guide

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork locally
   ```bash
   git clone https://github.com/afshinmohammadi/truck-productivity-simulation.git
   cd truck-productivity-simulation
   ```
3. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes
   - Follow the existing code style
   - Add docstrings to new functions and classes
   - Include type hints where appropriate

3. Run tests
   ```bash
   pytest tests/
   ```

4. Run linting
   ```bash
   black src/
   flake8 src/
   ```

5. Commit your changes
   ```bash
   git commit -m "feat: description of your changes"
   ```

6. Push to your fork and create a pull request

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Write docstrings for all public functions and classes
- Keep functions focused and under 50 lines when possible

## Project Structure

```
truck-productivity-simulation/
├── src/
│   ├── agents/          # Agent implementations
│   ├── environment/     # Network and traffic models
│   ├── optimization/    # Parameter sweep tools
│   └── utils/           # Utilities and visualization
├── notebooks/           # Jupyter notebooks
├── configs/             # Configuration files
├── tests/               # Unit tests
└── examples/            # Example scripts
```

## Adding New Features

### New Agent Types

1. Create a new file in `src/agents/`
2. Inherit from `BaseAgent`
3. Implement `step()` and `decide()` methods
4. Add to `src/agents/__init__.py`

### New Network Types

1. Add to `NetworkType` enum in `road_network.py`
2. Implement creation method (e.g., `_create_xxx_network()`)
3. Update `from_config()` method

### New Routing Policies

1. Add to `RoutingPolicy` enum in `truck_agent.py`
2. Implement logic in `calculate_route()` method

## Testing

- Write unit tests for new functionality
- Place tests in `tests/` directory
- Use pytest fixtures for common setup
- Aim for >80% code coverage

## Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions
- Update type hints
- Add examples in `examples/` or `notebooks/`

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about implementation

Thank you for contributing!
