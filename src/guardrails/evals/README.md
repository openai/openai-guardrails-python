# Guardrails Evaluation (`evals/`)

Core components for running guardrail evaluations and benchmarking.

## Quick Start

### Demo
Test the evaluation system with included demo files:
```bash
# Evaluation mode
python guardrail_evals.py \
  --config-path eval_demo/demo_config.json \
  --dataset-path eval_demo/demo_data.jsonl

# Benchmark mode
python guardrail_evals.py \
  --config-path eval_demo/demo_config.json \
  --dataset-path eval_demo/demo_data.jsonl \
  --mode benchmark \
  --models gpt-5 gpt-5-mini gpt-5-nano
```

### Basic Evaluation
```bash
python guardrail_evals.py \
  --config-path guardrails_config.json \
  --dataset-path data.jsonl
```

### Benchmark Mode
```bash
python guardrail_evals.py \
  --config-path guardrails_config.json \
  --dataset-path data.jsonl \
  --mode benchmark \
  --models gpt-5 gpt-5-mini gpt-5-nano
```

## Core Components

- **`guardrail_evals.py`** - Main evaluation script
- **`core/`** - Evaluation engine, metrics, and reporting
  - `async_engine.py` - Batch evaluation engine
  - `calculator.py` - Precision, recall, F1 metrics
  - `benchmark_calculator.py` - ROC AUC, precision at recall thresholds
  - `benchmark_reporter.py` - Benchmark results and tables
  - `latency_tester.py` - End-to-end guardrail latency testing
  - `visualizer.py` - Performance charts and graphs
  - `types.py` - Core data models and protocols

## Features

### Evaluation Mode
- Multi-stage pipeline evaluation (pre_flight, input, output)
- Automatic stage detection and validation
- Batch processing with configurable batch size
- JSON/JSONL output with organized results

### Benchmark Mode
- Model performance comparison across multiple LLMs
- Advanced metrics: ROC AUC, precision at recall thresholds
- End-to-end latency testing with dataset samples
- Automatic visualization generation
- Performance and latency summary tables

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--config-path` | ✅ | Pipeline configuration file |
| `--dataset-path` | ✅ | Evaluation dataset (JSONL) |
| `--mode` | ❌ | `evaluate` (default) or `benchmark` |
| `--stages` | ❌ | Specific stages to evaluate |
| `--models` | ❌ | Models for benchmark mode |
| `--batch-size` | ❌ | Parallel processing batch size (default: 32) |
| `--latency-iterations` | ❌ | Latency test samples (default: 50) |
| `--output-dir` | ❌ | Results directory (default: `results/`) |

## Output Structure

### Evaluation Mode
```
results/
└── eval_run_YYYYMMDD_HHMMSS/
    ├── eval_results_{stage}.jsonl
    ├── eval_metrics.json
    └── run_summary.txt
```

### Benchmark Mode
```
results/
└── benchmark_{guardrail}_YYYYMMDD_HHMMSS/
    ├── results/
    │   ├── eval_results_{guardrail}_{model}.jsonl
    │   ├── performance_metrics.json
    │   ├── latency_results.json
    │   └── benchmark_summary_tables.txt
    ├── graphs/
    │   ├── {guardrail}_roc_curves.png
    │   ├── {guardrail}_basic_metrics.png
    │   ├── {guardrail}_advanced_metrics.png
    │   └── latency_comparison.png
    └── benchmark_summary.txt
```

## Dataset Format

JSONL file with each line containing:
```json
{
  "id": "sample_1",
  "data": "Text to evaluate",
  "expected_triggers": {
    "guardrail_name": true/false
  }
}
```

## Dependencies

### Basic
```bash
pip install -e .
```

### Benchmark Mode
```bash
pip install -r requirements-benchmark.txt
```

## Notes

- Automatically evaluates all stages found in configuration
- Latency testing measures end-to-end guardrail performance
- All evaluation is asynchronous with progress tracking
- Invalid stages are automatically skipped with warnings
