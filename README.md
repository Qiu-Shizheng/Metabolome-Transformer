# Metabolome Transformer

<p align="center">
  <b>Metabolome Transformer</b> is a pretrained transformer model for metabolomics data.<br>
  This repository provides a deployment package for downstream use of the pretrained model.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c" alt="PyTorch 2.4.1">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License MIT">
</p>

---

## Overview

**Metabolome Transformer** provides a deployment package for downstream use of a pretrained transformer model for metabolomics data.

It supports missing value imputation, Health Deviation Score calculation, disease risk report generation, and custom downstream fine-tuning tasks.

---

## Features

This package supports five practical downstream tasks:

1. Load the pretrained **Metabolome Transformer**
2. Impute missing metabolomics values
3. Compute **Health Deviation Score**, or **HDS**
4. Generate future disease risk reports from HDS
5. Support other custom fine-tuning tasks, such as disease prediction

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
- Generates one formal risk-report-style HTML file per participant
- Saves HTML reports and summary outputs

---

## Installation

```bash
git clone https://github.com/yourname/metabolome-transformer.git
cd metabolome-transformer

pip install -e .
pip install -r requirements.txt
