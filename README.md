# Metabolome Transformer

Metabolome Transformer is a pretrained transformer model for metabolomics data. This repository provides a clean deployment package for downstream use of the pretrained model.

## Features

This package supports four practical downstream tasks:

1. **Load the pretrained Metabolome Transformer**
2. **Impute missing metabolomics values**
3. **Compute Health Deviation Score (HDS)**
4. **Generate future disease risk reports from HDS**


---

## What this package does

### 1. Missing value imputation
- Accepts user metabolomics files in `csv`, `txt`, or `tsv`
- Keeps observed values unchanged
- Predicts only missing values

### 2. HDS scoring
- Computes a single HDS per sample
- Reports the approximate percentile rank relative to the UK Biobank reference distribution

### 3. Disease risk reporting
- Uses HDS together with age, sex, and BMI
- Generates one formal risk report style per participant
- Saves HTML reports

---

## Installation
'''
git clone https://github.com/yourname/metabolome-transformer.git
cd metabolome-transformer
pip install -e .
pip install -r requirements.txt
'''
