# Predicting Facebook Ad Campaign Performance Using Few-Shot Learning with Large Language Models

## Overview

This repository contains the implementation, experiments, and analysis for the master’s thesis:

**“Predicting Facebook Ad Campaign Performance Using Few-Shot Learning with Large Language Models”**

The project investigates whether modern Large Language Models (LLMs) can accurately predict Facebook advertising campaign success using Few-Shot Learning techniques without traditional gradient-based model training.

The study evaluates multiple LLMs on structured tabular marketing data under different prompting configurations and compares their effectiveness in campaign performance prediction.

---

# Research Objectives

The main objectives of this thesis are:

* Predict Facebook advertising campaign success using LLMs
* Evaluate Few-Shot Learning on structured tabular data
* Compare multiple Large Language Models under different shot settings
* Compare LLM-based approaches with retrieval-based baselines
* Analyze the impact of prompt engineering and example selection strategies

---

# Models Evaluated

The following Large Language Models were evaluated:

* GPT OSS 120B
* Qwen 3.5
* DeepSeek v3.2

Each model was tested using:

* Zero-shot prompting
* One-shot prompting
* Three-shot prompting
* Five-shot prompting

---

# Dataset

The experiments use the Facebook Ad Campaign Performance dataset.

The dataset contains advertising campaign statistics and demographic targeting information.

## Features Include

* Age group
* Gender
* Interest categories
* Impressions
* Clicks
* Amount spent
* CTR (Click Through Rate)
* CPC (Cost Per Click)
* Total conversions
* Approved conversions
* Conversion Rate

---

# Success Definition

A campaign is classified as **successful** if at least **two out of three** of the following conditions are satisfied:

* CTR > 0.90%
* CPC < $1.72
* Conversion Rate > 3.0%

This rule-based definition was designed to balance engagement quality, advertising cost efficiency, and conversion effectiveness.

---

# Methodology

## Data Processing Pipeline

The experimental workflow includes:

1. Raw dataset cleaning
2. Corruption detection and removal
3. Feature engineering
4. Performance metric calculation
5. Leakage-free train-test split
6. Success label generation
7. Few-shot prompt construction
8. Multi-model LLM evaluation
9. Statistical analysis
10. Visualization and error analysis

---

# Few-Shot Learning Settings

The following prompting configurations were evaluated:

* 0-shot
* 1-shot
* 3-shot
* 5-shot

Few-shot examples were selected using similarity-based retrieval strategies.

---

# Prompting Techniques

The experiments incorporate:

* Chain-of-Thought (CoT) prompting
* Similarity-based example selection
* Structured tabular prompts
* Leakage-free evaluation methodology

---

# Repository Structure

```text
.
├── data/                                   # Processed datasets
├── logs/                                   # Experiment logs
├── results/                                # Experimental outputs and evaluation results
├── visualizations/                         # Generated charts and plots
│
├── clean_data.py                           # Dataset corruption detection and cleaning
├── data_preprocessing.py                   # Data preprocessing and feature engineering
├── train_test_split.py                     # Leakage-free train/test split generation
├── multi_model_experiment.py               # Multi-model few-shot LLM experiments
├── retrieval_baseline.py                   # Retrieval-based baseline implementation
├── extract_errors_all_models.py            # Error extraction and analysis
├── h3_all_models.py                        # Statistical hypothesis testing
├── run_experiment.py                       # Master pipeline runner
│
└── README.md
```

---

# Experimental Pipeline

The complete experiment can be executed using the master runner script.

## 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 2. Clean the Raw Dataset

```powershell
python clean_data.py
```

This script:

* Detects corrupted rows
* Removes shifted or invalid records
* Generates cleaned datasets
* Creates corruption reports

Generated files:

```text
data/data_clean.csv
data/bad_rows_report.csv
```

---

## 3. Run Data Preprocessing

```powershell
python data_preprocessing.py
```

This step performs:

* Feature engineering
* Metric calculation
* Success label generation
* Data validation
* Leakage prevention checks

---

## 4. Run the Full Experiment Pipeline

```powershell
python run_experiment.py
```

This master script automatically executes:

* Data preprocessing
* Train-test split generation
* Multi-model experiments
* Statistical evaluation
* Visualization generation

---

# Data Cleaning Results

Initial dataset size:

```text
1143 rows
```

After corruption detection and cleaning:

```text
761 clean rows
382 corrupted rows removed
```

Corruption rate detected:

```text
33.42%
```

The cleaning process removes shifted demographic fields and structurally corrupted records.

---

# Dataset Distribution

## Success Distribution

* Successful campaigns: 167 (21.94%)
* Unsuccessful campaigns: 594 (78.06%)

## Gender Distribution

* Male: 63.9%
* Female: 36.1%

## Age Distribution

* 30–34: 43.0%
* 35–39: 23.7%
* 40–44: 18.3%
* 45–49: 15.1%

---

# Evaluation Metrics

The models were evaluated using:

* Accuracy
* Balanced Accuracy
* Precision
* Recall
* F1-Score
* Matthews Correlation Coefficient (MCC)
* Confusion Matrix

---

# Key Findings

* GPT OSS 120B achieved the strongest overall performance
* Few-shot prompting significantly improved prediction quality
* Chain-of-Thought prompting improved reasoning consistency
* LLMs performed effectively on structured tabular marketing data
* Similarity-based example selection improved few-shot performance
* Retrieval-based approaches provided useful baseline comparisons

---

# Generated Outputs

The project automatically generates:

## Results

* Prediction outputs
* Evaluation reports
* Statistical summaries
* Error analysis reports

## Visualizations

* Success distribution plots
* Metric distribution plots
* Model comparison charts
* Confusion matrices

---

# Statistical Analysis

Statistical hypothesis testing is implemented in:

```text
h3_all_models.py
```

The analysis compares:

* Shot configurations
* Model performance differences
* Statistical significance across experiments

---

# Error Analysis

Error extraction and analysis are implemented in:

```text
extract_errors_all_models.py
```

This script identifies:

* Incorrect predictions
* Common failure patterns
* Model-specific weaknesses

---

# Retrieval Baseline

The repository includes a retrieval-based baseline:

```text
retrieval_baseline.py
```

This baseline is used for comparison against LLM-based prediction approaches.

---

# Limitations

* Dataset is limited to Facebook advertising campaigns
* Dataset originates from 2017 advertising data
* No multimodal data (images/videos/text creatives) included
* Results may vary across different API versions and LLM updates

---

# Future Work

Possible future extensions include:

* Multimodal advertisement analysis
* Integration of image and video campaign data
* Evaluation of newer LLM architectures
* Hybrid ML + LLM prediction systems
* Real-time campaign optimization frameworks

---

# Author

**Shahadat Hussain**
Master’s Thesis
Faculty of Informatics
Masaryk University
Brno, Czech Republic
2026

---

# Keywords

Few-Shot Learning, Large Language Models, Facebook Ads, Campaign Prediction, Prompt Engineering, Tabular Data, Marketing Analytics
