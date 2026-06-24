# ContextChecker

> A framework for evaluating LLM contexts.

## Installation

```bash
# Core package
uv pip install contextchecker

# With test suite
uv pip install "contextchecker[test]"
```

### Development install

```bash
uv pip install -e ".[test]"
```

## Quick Start

```python
from contextchecker import __version__

print(__version__)  # 0.5.0
```

## CLI

```bash
contextchecker --help
```

## Architecture

See [architecture.md](architecture.md) for the full design document.

## License

MIT
