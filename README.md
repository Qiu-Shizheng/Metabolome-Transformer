# Metabolome Transformer

Metabolome Transformer provides command-line tools for metabolomics missing value imputation and binary disease prediction.

## Installation

```bash
pip install -e .
```

## Input

Your input table should contain one sample ID column and the 249 metabolite columns listed in [`assets/pretrained/metabolite_columns.txt`](assets/pretrained/metabolite_columns.txt).

Example input:

```text
sample_id,Total Cholesterol,Total Cholesterol Minus HDL-C,...
sample_001,4.60,3.20,...
sample_002,,3.60,...
```

## Impute Missing Values

```bash
metabolome-transformer impute \
  --input examples/example_input.csv \
  --output results/imputed.csv \
  --id-column sample_id \
  --batch-size 12
```

Use CPU explicitly with `--device cpu`.

Example output:

```text
sample_id,Total Cholesterol,Total Cholesterol Minus HDL-C,...
sample_001,4.60,3.20,...
sample_002,4.82,3.60,...
```

## Fine-Tune Current Disease Prediction

The training file should contain the sample ID column, metabolite columns, and a binary label column.

Example training input:

```text
sample_id,label,Total Cholesterol,Total Cholesterol Minus HDL-C,...
sample_001,0,4.60,3.20,...
sample_002,1,5.10,3.60,...
```

```bash
metabolome-transformer train-binary \
  --input train.csv \
  --output-dir results/current_task \
  --id-column sample_id \
  --label-column label \
  --task current \
  --epochs 10 \
  --batch-size 12
```

## Fine-Tune Future Disease Prediction

```bash
metabolome-transformer train-binary \
  --input train.csv \
  --output-dir results/future_task \
  --id-column sample_id \
  --label-column label \
  --task future \
  --epochs 10 \
  --device cpu
```

## Predict With A Fine-Tuned Classifier

```bash
metabolome-transformer predict-binary \
  --input test.csv \
  --classifier results/current_task/classifier.pt \
  --output results/predictions.csv \
  --id-column sample_id \
  --batch-size 12
```

Prediction output contains the ID column, `probability`, and `prediction`.

Example prediction output:

```text
sample_id,probability,prediction
sample_001,0.18,0
sample_002,0.83,1
```
