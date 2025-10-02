# NSFW Text Check Benchmark Results

## Dataset Description

This benchmark evaluates model performance on a balanced set of social media posts:

- Open Source [Toxicity dataset](https://github.com/surge-ai/toxicity/blob/main/toxicity_en.csv)
- 500 NSFW (true) and 500 non-NSFW (false) samples
- All samples are sourced from real social media platforms

**Total n = 1,000; positive class prevalence = 500 (50.0%)**

## Results

### ROC Curve

![ROC Curve](./NSFW_roc_curve.png)

### Metrics Table

| Model         | ROC AUC | Prec@R=0.80 | Prec@R=0.90 | Prec@R=0.95 | Recall@FPR=0.01 |
|--------------|---------|-------------|-------------|-------------|-----------------|
| gpt-4.1      | 0.989   | 0.976       | 0.962       | 0.962       | 0.717           |
| gpt-4.1-mini | 0.984   | 0.977       | 0.977       | 0.943       | 0.653           |
| gpt-4.1-nano | 0.952   | 0.972       | 0.823       | 0.823       | 0.429           |
| gpt-4o-mini  | 0.965   | 0.977       | 0.955       | 0.945       | 0.842           |

#### Notes
- ROC AUC: Area under the ROC curve (higher is better)
- Prec@R: Precision at the specified recall threshold
- Recall@FPR=0.01: Recall when the false positive rate is 1%
