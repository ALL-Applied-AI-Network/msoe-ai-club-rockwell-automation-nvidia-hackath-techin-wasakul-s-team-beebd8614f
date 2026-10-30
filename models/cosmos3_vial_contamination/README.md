# Cosmos 3 Vial Contamination Detection — Original V2

## Selected model

Selected adapter: `original_v2/`

Base model: `PourSightAI/CosmosReason3Hackathon`
Base model subfolder: `cosmos3-nano`

This LoRA adapter assesses chronological vial images for possible visual liquid carryover between filling operations.

## Final independent synthetic evaluation

| Metric | Original V2 |
|---|---:|
| Evaluation examples | 540 |
| True positives | 411 |
| False positives | 24 |
| True negatives | 68 |
| False negatives | 37 |
| Precision | 94.48% |
| Recall | 91.74% |
| Specificity | 73.91% |
| F1 score | 93.09% |
| Balanced accuracy | 82.83% |

The evaluation used 540 eligible examples from an independently generated synthetic dataset. It did not use real-world production data.

## Model selection

Original V2 was selected to reduce false alarms while retaining strong carryover detection. It makes a trade-off: some carryover cases may be missed.

## Method

The model compares chronological camera images using visual color and fill-level consistency. It does not receive hidden recipe IDs, tank identities, ingredient lists, or simulator labels as inputs.

Its output is an assessment of a possible visual carryover signal, not proof of contamination.

## Repository contents

- `original_v2/` — selected LoRA adapter, configuration, tokenizer, processor, metrics, and validation predictions.
- `training_scripts/` — scripts for the original and refined training experiments.
- `evaluation/final_test/` — independent final synthetic test report.
- `evaluation/additional_test/` — earlier synthetic comparison report.
- `experiment_metrics/` — metrics from comparison experiments.

## Limitations

All reported evaluation results are from synthetic data. They do not establish real-world production performance. False positives and false negatives remain possible. Lighting, reflections, camera position, and visually similar legitimate recipes may affect predictions. Human review and further validation are recommended before operational use.

## Reproducibility

Training scripts are included for transparency. Reproducing training requires the compatible base model, Python dependencies, and generated synthetic datasets. The datasets are not included in this repository.
