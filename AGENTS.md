# AGENTS.md

Guidelines for AI coding agents working in this repository.

## Build/Lint/Test Commands

This project uses `uv` for Python package management and Ruff for linting/formatting.

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run linting
ruff check .

# Fix auto-fixable linting issues
ruff check --fix .

# Format code
ruff format .

# Check and format in one go
ruff check --fix . && ruff format .
```

**Note**: There is no test suite configured for this project. Tests should be run manually via Python scripts.

## Code Style Guidelines

### Imports

Follow this import order (separated by blank lines):
1. Standard library imports
2. Third-party imports (torch, numpy, matplotlib, etc.)
3. Local module imports (from datasets, from models, from utils)

```python
from dataclasses import dataclass
from typing import Callable, Tuple, List

import torch
from torch import nn
from torchvision import transforms

from datasets import get_dataset
from models import get_transforms
```

### Formatting

- Use **Ruff** for formatting (configured in GitHub Actions)
- Line length: default (88 characters)
- Use double quotes for strings
- No trailing commas in multi-line imports

### Type Hints

- Use type hints for function parameters and return values
- Use `typing` module imports (Callable, Tuple, List, Optional, etc.)
- Example: `def get_dataset(dataset: str) -> Tuple[FedDataset, Dataset, List[int]]:`

### Naming Conventions

- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private functions**: `_leading_underscore`

### Docstrings

- Use triple double quotes `"""` for docstrings
- Document function purpose, args, and return values

### Error Handling

- Use `assert` statements for runtime checks (especially CUDA availability)
- Use specific exception types (e.g., `FileNotFoundError`)
- Example: `assert torch.cuda.is_available(), "CUDA not available"`

### Project Structure

- `datasets.py` - Dataset loading and configuration
- `models.py` - Neural network architectures
- `utils/` - Utility modules (colored_mnist, waterbirds, fedprox, fednova, etc.)
- Main experiment scripts at root level (flex_crp.py, contrastive_cosine.py, irm_w.py)

### PyTorch Conventions

- Check for CUDA at module level: `device = "cuda"`
- Use `nn.Module` for all model classes
- Call `super().__init__()` in model constructors

## Pre-commit Checklist

Before committing:
1. Run `ruff check --fix .` to fix linting issues
2. Run `ruff format .` to ensure consistent formatting
3. Verify the code follows import ordering conventions
4. Ensure type hints are present for public functions
