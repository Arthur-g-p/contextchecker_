# Logging Architecture

How logging works in contextchecker, and how to use it when building new services.

## The System

Every module uses Python's standard `logging` via `settings.get_logger(__name__)`. By default, output is **silent** (NullHandler). The CLI activates pretty output at startup via `enable_logging()`.

```
contextchecker                         ← NullHandler (silent) + CLI attaches StreamHandler here
├── contextchecker.cli                 ← box header, output path
├── contextchecker.services.extraction ← validation, skip, config, results, done line
├── contextchecker.workers.extractor   ← extraction progress, retries
└── contextchecker.llmclient           ← connection test, strategy discovery (future migration)
```

All child loggers bubble up to the parent `contextchecker` logger. One handler on the parent controls all output.

## Two Modes

| Mode | Activated by | Formatter | What it shows |
|------|-------------|-----------|---------------|
| Normal | `enable_logging()` (default) | `PrettyFormatter` | Just the message |
| Debug | `enable_logging(debug=True)` or `--debug` | `DebugFormatter` | `HH:MM:SS \| module \| message` |

No separate "verbose" mode exists.

## For Library Users

```python
import contextchecker
from contextchecker.settings import enable_logging
from contextchecker.services.extraction import ExtractionService

# Silent by default — no output
service = ExtractionService(model="gemini-2.0-flash")
result = service.run_sync(data)

# Opt in to output:
enable_logging()             # pretty
enable_logging(debug=True)   # with timestamps + module prefix
```

## For CLI Users

Output is automatic. The `--debug` flag enables debug mode:

```bash
contextchecker extract data.json                  # pretty output
contextchecker extract data.json --debug          # + timestamps & module names
```

## Rules for New Services

When adding a new service (e.g. `CheckingService`), follow these rules:

### 1. Use `logger = settings.get_logger(__name__)` at module level

```python
from contextchecker import settings
logger = settings.get_logger(__name__)
```

Never call `print()`. Never create your own handlers.

### 2. Each layer logs what it naturally owns

| Layer | What it logs | Example |
|-------|-------------|---------|
| **CLI** | Box header, output file path | `📁 results/output.json` |
| **Service** | Validation, filtering, config, results summary, done line | `📂 Validation`, `✅ Done: 10 items → 34 claims` |
| **Worker** | Execution progress, retries | `Extracting: 100%`, `♻️ Retry: 3 failed items` |

The CLI does NOT format validation/results. Services do NOT print file paths. Workers do NOT log configuration.

### 3. Use proper indentation

All sections follow a consistent indentation pattern:

```python
# Section header: 1 space + emoji
logger.info(" 📂 Validation")

# Section content: 4 spaces
logger.info("    Total:      %d items", total)

# Tree items: 4 spaces + tree char
logger.info("    ├─ abstain:  %d  (empty response)", count)
logger.info("    └─ valid:   %d items", valid)

# Section separator
logger.info("")
```

### 4. Hide zero-count lines

Don't show lines where the count is zero. Skip entire sections when they're irrelevant:

```python
# Good: only show when there's something to show
if abstained > 0:
    logger.info("    ├─ abstain:  %d  (empty response)", abstained)

# Good: skip entire section when nothing happened
if skipped == 0:
    return
```

### 5. stats.py is pure data — no printing

`PhaseStats` and `TokenStats` are data containers. They do NOT format or print output. The service reads them and logs the formatted summary via `logger.info()`.

```python
# ✅ Correct: service formats and logs
logger.info(" 🌐 API-Request summary:")
logger.info("    %d (pending) input items", stats.total_items)
if stats.prefilter > 0:
    logger.info("     ├─ 🔇 %d prefiltered", stats.prefilter)

# ❌ Wrong: stats module prints directly
def print_api_summary(stats):
    print(f"🌐 API-Request summary:")  # NEVER do this
```

### 6. Consistent emoji vocabulary

| Emoji | Meaning | Used by |
|-------|---------|---------|
| 📂 | Input validation | Service |
| 🔄 | Skip/filter | Service |
| ⚙️ | Configuration | Service |
| 📡 | Connection test | LLMClient |
| 🔬 | Strategy discovery | LLMClient |
| 🔒 | Strategy locked | LLMClient |
| ⚠️ | Warning (non-fatal) | Any |
| ❌ | Error/failure | Any |
| ✅ | Success | Any |
| ♻️ | Retry | Worker |
| 📊 | Statistics | Service |
| 📁 | Output file | CLI |
| 🌐 | API summary | Service |
| 📝 | Business logic results | Service |
| 🔇 | Prefiltered | Service |
| 📏 | Context too long | Service |
| 🛡️ | Content blocked | Service |

### 7. Section separators

Use `── TITLE ──` for major phase boundaries:

```python
logger.info("── Extraction ────────────────────────────────────────")
```

These mark transitions between phases (e.g. filtering → extraction → results).
