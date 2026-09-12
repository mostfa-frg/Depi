# Support-ticket LLM pipeline

This repository is a course project that builds a small local language-model
application for customer-support tickets. It has four connected layers:

1. **Inference** (`src/infrence/`) loads a Hugging Face causal language model,
   tokenizes prompts, generates tokens, and implements temperature, top-k, and
   top-p decoding.
2. **Prompt engineering** (`prompts/`) builds reusable prompts for ticket
   classification and summarization.
3. **Structured output** (`structured/`) constrains model output to the
   `TicketOutput` Pydantic schema: category, sentiment, urgency, and summary.
4. **Evaluation** (`evaluation/`) runs a generator over `data/tickets.csv` and
   reports validity, failure, category, sentiment, and urgency metrics.

## Setup

Use Python 3.10+ from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The default model is `distilgpt2`, which is intentionally small but still
requires a first-run download from Hugging Face. To use another compatible
causal language model, create a local `.env` file containing:

```text
MODEL_NAME=your-model-name
```

Do not commit `.env` or model weights.

## Run the lightweight checks

```powershell
python -m pytest -q
```

These tests cover prompt construction, schema validation, evaluation metrics,
and decoding without loading a model. The full model demos are separate because
they require model files and can be slow on CPU.

## Run the model demos

```powershell
python -m src.inferenece
python -m structured.llama
```

The first command demonstrates free-form generation. The second demonstrates
structured ticket classification through Outlines. The model must support
causal generation; encoder-only or sequence-to-sequence models will not work
with the current loader.

## Evaluation workflow

The intended next step is to connect `TicketEvaluator` to a
`StructuredGeneration` instance:

```python
import pandas as pd
from evaluation import TicketEvaluator

tickets = pd.read_csv("data/tickets.csv")
evaluator = TicketEvaluator(generator)
results = evaluator.evaluate(tickets)
```

Inspect each `EvaluationResult.error` first, then compare the four aggregate
metrics. This separates infrastructure failures (invalid JSON, schema errors,
model exceptions) from actual classification mistakes.

The ready-to-run version is:

```powershell
python -m evaluation.run_evaluation
```

## Learning roadmap

1. Python modules, imports, virtual environments, and dependency management.
2. Hugging Face tokenizers and causal language models.
3. Tokenization, tensors, attention masks, devices, and KV caching.
4. Autoregressive decoding and sampling: temperature, top-k, and top-p.
5. Prompt templates and constraints for reliable model behavior.
6. Pydantic models, `Literal` values, JSON parsing, and structured generation.
7. Dataset handling with pandas and labeled evaluation data.
8. Evaluation design: validity rate, failure count, per-field accuracy, and
   error classification.
9. Debugging workflow: reproduce, isolate the layer, inspect the exception,
   fix one cause, and rerun the smallest relevant check.
