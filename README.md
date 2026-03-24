# Metabolome Transformer

Metabolome Transformer is a pretrained transformer model for metabolomics data. This repository provides a deployment package for downstream use of the pretrained model.

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
```
git clone https://github.com/yourname/metabolome-transformer.git
cd metabolome-transformer
pip install -e .
pip install -r requirements.txt
```


## Input format
The input metabolomics file should contain:

**an ID column**  
**metabolite columns**  

**optionally contain:**  
age  
sex_binary  
BMI  

**sex_binary encoding**  
0 = Female  
1 = Male  

---

## Command-line usage
### 1. Impute missing metabolomics values
```
metabolome-transformer impute \
  --input data/my_cohort.csv \
  --output results/imputed_metabolomics.csv \
  --device cuda \
  --batch-size 16
```

### 2. Compute HDS

```
metabolome-transformer hds \
  --input data/my_cohort.csv \
  --output results/hds_results.csv \
  --device cuda \
  --batch-size 16
```


### 3. Generate risk reports
The input file must contain:

eid  
metabolite columns  
age  
sex_binary  
BMI  

```
metabolome-transformer report \
  --input data/my_cohort_with_clinical.csv \
  --output-dir results/reports \
  --device cuda \
  --batch-size 16
```


### 4. Run the full pipeline
This runs:

imputation  
HDS scoring  
disease risk report generation  
```
metabolome-transformer run-all \
  --input data/my_cohort_with_clinical.csv \
  --output-dir results/full_run \
  --device cuda \
  --batch-size 16
```

---

## Output files
### Imputation output
imputed_metabolomics.csv  
### HDS output
hds_results.csv 

**Columns:**  
eid  
HDS  
HDS_percentile  

### Risk report output
input_with_hds.csv  
one HTML report per participant  
one CSV file with all predicted outcomes per participant  
risk_report_summary.csv

---

## GPU not found
Use:
```
--device cpu
```
instead of:
```
--device cuda
```
