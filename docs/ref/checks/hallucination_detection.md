# Hallucination Detection

Detects potential hallucinations in AI-generated text by validating factual claims against reference documents using [OpenAI's FileSearch API](https://platform.openai.com/docs/guides/tools-file-search). Analyzes text for factual claims that can be validated, flags content that is contradicted or unsupported by your knowledge base, and provides confidence scores and reasoning for detected issues.

## Hallucination Detection Definition

Flags model text containing factual claims that are clearly contradicted or not supported by your reference documents (via File Search). Does not flag opinions, questions, or supported claims. Sensitivity is controlled by a confidence threshold.

## Configuration

```json
{
    "name": "Hallucination Detection",
    "config": {
        "model": "gpt-4.1-mini",
        "confidence_threshold": 0.7,
        "knowledge_source": "vs_abc123"
    }
}
```

### Parameters

- **`model`** (required): OpenAI model (required) to use for validation (e.g., "gpt-4.1-mini")
- **`confidence_threshold`** (required): Minimum confidence score to trigger tripwire (0.0 to 1.0)
- **`knowledge_source`** (required): OpenAI vector store ID starting with "vs_" containing reference documents

### Tuning guidance

- Start at 0.7. Increase toward 0.8–0.9 to avoid borderline flags; decrease toward 0.6 to catch more subtle errors.
- Quality and relevance of your vector store strongly influence precision/recall. Prefer concise, authoritative sources over large, noisy corpora.

## Implementation

### Prerequisites: Create a Vector Store

- Use the [OpenAI Dashboard](https://platform.openai.com/storage/vector_stores) to create and manage vector stores; or
- Use the utility script to upload files:

```bash
python src/guardrails/utils/create_vector_store.py your_document.pdf
```

Save the returned vector store ID (e.g., `vs_abc123`).

### Configure Guardrails

```python
bundle = {
    "version": 1,
    "output": {
        "version": 1,
        "guardrails": [
            {
            "name": "Hallucination Detection",
            "config": {
                "model": "gpt-5",
                "confidence_threshold": 0.7,
                "knowledge_source": "vs_abc123",
            },
            },
        ],
    },
}
```

### Use with Guardrails Client

```python
from guardrails import GuardrailsAsyncOpenAI

client = GuardrailsAsyncOpenAI(config=bundle)
response = await client.responses.create(
    model="gpt-5",
    input="Microsoft's revenue in 2023 was $500 billion."
)

# Guardrails automatically validate against your reference documents
print(response.llm_response.output_text)
```

### How It Works

- Input: LLM response text
- Validation: Uses [OpenAI's FileSearch API](https://platform.openai.com/docs/guides/tools-file-search) to check claims against your vector store documents
- Output: Triggers if claims can't be verified or are contradicted
- Confidence: Adjustable threshold for detection sensitivity

### Complete Example

See [`examples/hallucination_detection/`](https://github.com/openai/openai-guardrails-python/tree/main/examples/hallucination_detection) for the full implementation.

### Notes

- Uses OpenAI's FileSearch API which incurs additional [costs](https://platform.openai.com/docs/pricing#built-in-tools)
- Only flags clear contradictions or unsupported claims; it does not flag opinions, questions, or supported claims

#### Error handling

- If the model returns malformed or non-JSON output, the guardrail returns a safe default with `flagged=false`, `confidence=0.0`, and an `error` message in `info`.
- If a vector store ID is missing or invalid (must start with `vs_`), an error is thrown during execution.

## What It Returns

Returns a `GuardrailResult` with the following `info` dictionary:

```json
{
    "guardrail_name": "Hallucination Detection",
    "flagged": true,
    "confidence": 0.95,
    "reasoning": "The claim about pricing contradicts the documented information",
    "hallucination_type": "factual_error",
    "hallucinated_statements": ["Our premium plan costs $299/month"],
    "verified_statements": ["We offer customer support"],
    "threshold": 0.7,
    "checked_text": "Our premium plan costs $299/month and we offer customer support"
}
```

- **`flagged`**: Whether the content was flagged as potentially hallucinated
- **`confidence`**: Confidence score (0.0 to 1.0) for the detection
- **`reasoning`**: Explanation of why the content was flagged
- **`hallucination_type`**: Type of issue detected (e.g., "factual_error", "unsupported_claim")
- **`hallucinated_statements`**: Specific statements that are contradicted or unsupported
- **`verified_statements`**: Statements that are supported by your documents
- **`threshold`**: The confidence threshold that was configured
- **`checked_text`**: Original input text

Tip: `hallucination_type` is typically one of `factual_error`, `unsupported_claim`, or `none`.

## Benchmark Results

### Dataset Description

This benchmark evaluates model performance on factual claims validation:

#### Knowledge Source
The knowledge base consists of 15 publicly available SEC filings from three major companies (5 from each company):

- **Microsoft Corporation**: Annual reports (10-K) and quarterly reports (10-Q) containing financial statements, business operations, risk factors, and management discussion
- **Oracle Corporation**: SEC filings including financial performance, revenue breakdowns, cloud services metrics, and corporate governance information  
- **Ford Motor Company**: Automotive industry reports covering vehicle sales, manufacturing operations, financial results, and market analysis

These documents provide diverse coverage of financial metrics, business strategies, operational details, and corporate information that can be used to validate factual claims.

#### Evaluation Set
The evaluation dataset contains 300 carefully crafted statements designed to test the hallucination detection capabilities:

- **150 positive examples**: Statements containing factual claims that are _clearly contradicted_ or _completely unsupported_ by the knowledge source documents.
- **150 negative examples**: Statements that are either _supported_ by the documents or contain no verifiable factual claims (and therefore do not need to be fact checked).

The statements cover various types of factual claims including:

- Financial figures (revenue, profit, growth rates)
- Business metrics (employee count, market share, product details)
- Operational information (facilities, partnerships, timelines)
- Corporate facts (executives, policies, strategic initiatives)

**Total n = 300; positive class prevalence = 150 (50.0%)**

### Results

**Precision** measures how many of the statements flagged by the guardrail as hallucinations were actually unsupported or contradicted by the knowledge source (i.e., correctly identified as hallucinations). 

**Recall** measures how many of the total hallucinated statements in the evaluation dataset the model were correctly flagged by the guardrail. High precision indicates the model avoids false positives; high recall indicates the model catches most hallucinations.

#### ROC Curve

![ROC Curve](../../benchmarking/hallucination_detection_roc_curves.png)

#### Model Performance Table

| Model         | ROC AUC | Prec@R=0.80 | Prec@R=0.90 | Prec@R=0.95 |
|--------------|---------|-------------|-------------|-------------|
| gpt-5         | 0.854   | 0.732       | 0.686       | 0.670       |
| gpt-5-mini    | 0.934   | 0.813       | 0.813       | 0.770       |
| gpt-5-nano    | 0.566   | 0.540       | 0.540       | 0.533       |
| gpt-4.1       | 0.870   | 0.785       | 0.785       | 0.785       |
| gpt-4.1-mini (default) | 0.876   | 0.806       | 0.789       | 0.789       |
| gpt-4.1-nano  | 0.537   | 0.526       | 0.526       | 0.526       |

**Notes:**
- ROC AUC: Area under the ROC curve (higher is better)
- Prec@R: Precision at the specified recall threshold

### Latency Performance

The following table shows latency measurements for each model using the hallucination detection guardrail with OpenAI's File Search tool:

| Model         | TTC P50 (ms) | TTC P95 (ms) |
|--------------|--------------|--------------|
| gpt-5         | 34,135       | 525,854      |
| gpt-5-mini    | 23,013       | 59,316       |
| gpt-5-nano    | 17,079       | 26,317       |
| gpt-4.1       | 7,126        | 33,464       |
| gpt-4.1-mini (default) | 7,069        | 43,174       |
| gpt-4.1-nano  | 4,809        | 6,869        |

- **TTC P50**: Median time to completion (50% of requests complete within this time)
- **TTC P95**: 95th percentile time to completion (95% of requests complete within this time)
- All measurements include file search processing time using OpenAI's File Search tool

### Vector Store Scaling Analysis

In addition to the above evaluations which use a 3 MB sized vector store, the hallucination detection guardrail was tested across various vector store sizes to understand the impact of knowledge base scale on performance and latency:

#### Vector Store Configurations
- **Small (1 MB)**: 1 document each from Microsoft, Oracle, and Ford (3 total documents)
- **Medium (3 MB)**: 5 documents each from Microsoft, Oracle, and Ford (15 total documents)  
- **Large (11 MB)**: Medium configuration plus 8MB of additional financial documents from an open source [Kaggle dataset](https://www.kaggle.com/datasets/finnhub/reported-financials?resource=download)
- **Extra Large (105 MB)**: An extension of the large vector store with additional documents from the kaggle dataset

#### Latency Scaling

| Model         | Small (1 MB) P50/P95 | Medium (3 MB) P50/P95 | Large (11 MB) P50/P95 | Extra Large (105 MB) P50/P95 |
|--------------|---------------------|----------------------|---------------------|---------------------------|
| gpt-5         | 28,762 / 396,472    | 34,135 / 525,854     | 37,104 / 75,684     | 40,909 / 645,025          |
| gpt-5-mini    | 19,240 / 39,526     | 23,013 / 59,316      | 24,217 / 65,904     | 37,314 / 118,564          |
| gpt-5-nano    | 13,436 / 22,032     | 17,079 / 26,317      | 17,843 / 35,639     | 21,724 / 37,062           |
| gpt-4.1       | 7,437 / 15,721      | 7,126 / 33,464       | 6,993 / 30,315      | 6,688 / 127,481           |
| gpt-4.1-mini (default) | 6,661 / 14,827      | 7,069 / 43,174       | 7,032 / 46,354      | 7,374 / 37,769            |
| gpt-4.1-nano  | 4,296 / 6,378       | 4,809 / 6,869        | 4,171 / 6,609       | 4,650 / 6,201             |

- **Vector store size impact varies by model**: GPT-4.1 series shows minimal latency impact across vector store sizes, while GPT-5 series shows significant increases.

#### Performance Scaling

![ROC Curve](../../benchmarking/vector_store_performance_scaling.png)

### Complete Performance Metrics Across All Vector Store Sizes

| Model | Vector Store | ROC AUC | Prec@R=0.80 | Prec@R=0.90 | Prec@R=0.95 |
|-------|-------------|---------|-------------|-------------|-------------|
| **gpt-5** | Small (1 MB) | 0.847 | 0.713 | 0.649 | 0.645 |
| | Medium (3 MB) | 0.854 | 0.732 | 0.686 | 0.670 |
| | Large (11 MB) | 0.814 | 0.649 | 0.633 | 0.633 |
| | Extra Large (105 MB) | 0.866 | 0.744 | 0.684 | 0.683 |
| **gpt-5-mini** | Small (1 MB) | 0.939 | 0.821 | 0.821 | 0.821 |
| | Medium (3 MB) | 0.934 | 0.813 | 0.813 | 0.770 |
| | Large (11 MB) | 0.919 | 0.817 | 0.817 | 0.817 |
| | Extra Large (105 MB) | 0.909 | 0.793 | 0.793 | 0.711 |
| **gpt-5-nano** | Small (1 MB) | 0.590 | 0.547 | 0.545 | 0.536 |
| | Medium (3 MB) | 0.566 | 0.540 | 0.540 | 0.533 |
| | Large (11 MB) | 0.564 | 0.534 | 0.532 | 0.507 |
| | Extra Large (105 MB) | 0.603 | 0.570 | 0.558 | 0.550 |
| **gpt-4.1** | Small (1 MB) | 0.907 | 0.839 | 0.839 | 0.839 |
| | Medium (3 MB) | 0.870 | 0.785 | 0.785 | 0.785 |
| | Large (11 MB) | 0.846 | 0.753 | 0.753 | 0.753 |
| | Extra Large (105 MB) | 0.837 | 0.743 | 0.743 | 0.743 |
| **gpt-4.1-mini (default)** | Small (1 MB) | 0.914 | 0.851 | 0.851 | 0.851 |
| | Medium (3 MB) | 0.876 | 0.806 | 0.789 | 0.789 |
| | Large (11 MB) | 0.862 | 0.791 | 0.757 | 0.757 |
| | Extra Large (105 MB) | 0.802 | 0.722 | 0.722 | 0.722 |
| **gpt-4.1-nano** | Small (1 MB) | 0.605 | 0.528 | 0.528 | 0.528 |
| | Medium (3 MB) | 0.537 | 0.526 | 0.526 | 0.526 |
| | Large (11 MB) | 0.618 | 0.531 | 0.531 | 0.531 |
| | Extra Large (105 MB) | 0.636 | 0.528 | 0.528 | 0.528 |

**Key Insights:**

- **Best Performance**: gpt-5-mini consistently achieves the highest ROC AUC scores across all vector store sizes (0.909-0.939)
- **Best Latency**: gpt-4.1-nano shows the most consistent and lowest latency across all scales (4,171-4,809ms P50) but shows poor performance
- **Most Stable**: gpt-4.1-mini (default) maintains relatively stable performance across vector store sizes with good accuracy-latency balance
- **Scale Sensitivity**: gpt-5 shows the most variability in performance across vector store sizes, with performance dropping significantly at larger scales
- **Performance vs Scale**: Most models show decreasing performance as vector store size increases, with gpt-5-mini being the most resilient

**Why Performance Decreases with Scale:**

- **Signal-to-noise ratio degradation**: Larger vector stores contain more irrelevant documents that may not be relevant to the specific factual claims being validated
- **Semantic search limitations**: File search retrieves semantically similar documents, but with a large diverse knowledge source, these may not always be factually relevant
- **Document quality matters more than quantity**: The relevance and accuracy of documents is more important than the total number of documents
- **Performance plateaus**: Beyond a certain size (11 MB), the performance impact becomes less severe