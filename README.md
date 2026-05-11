# Predicting Facebook Ad Campaign Performance Using Few-Shot Learning with Large Language Models

##  Overview
This thesis project investigates the effectiveness of Few-Shot Learning with Large Language Models (LLMs) for predicting Facebook advertising campaign performance.

The research evaluates whether modern LLMs can accurately predict campaign success using only a small number of examples, without traditional model training.


##  Objectives
- Predict Facebook ad campaign success using LLMs
- Compare multiple LLMs under few-shot settings
- Evaluate performance on structured tabular marketing data
- Compare LLMs with traditional machine learning baselines


##  Models Evaluated
- GPT OSS 120B
- Qwen3.5
- DeepSeek v3.2


##  Dataset
The project uses the Facebook Ad Campaign Performance dataset.

### Features include:
- Age
- Gender
- Interest scores
- Impressions
- Clicks
- Spend
- CTR (Click Through Rate)
- CPC (Cost Per Click)
- Conversion Rate


## ⚙️ Methodology

### Few-Shot Settings
- Zero-shot
- One-shot
- Three-shot
- Five-shot

### Techniques Used
- Chain-of-Thought Prompting
- Similarity-based Example Selection
- Stratified Train-Test Split
- Statistical Evaluation


##  Success Criteria
A campaign is classified as successful if at least 2 of the following conditions are satisfied:

- CTR > 0.90%
- CPC < $1.72
- Conversion Rate > 3.0%


##  Project Structure

```text
data/                   # Dataset files
results/                # Experimental outputs
visualizations/         # Generated charts and plots

clean_data.py
data_preprocessing.py
multi_model_experiment.py
prompt_engineering.py
run_experiment.py
statistical_analysis.py
success_metrics.py
train_test_split.py
visualization.py

## How to Run
Install dependencies
pip install -r requirements.txt

## Run experiments
python run_experiment.py

## Key Findings
GPT OSS 120B achieved the best performance
Few-shot learning significantly outperformed traditional ML methods
Chain-of-thought prompting improved reasoning quality
LLMs handled class imbalance better than classical approaches


## Evaluation Metrics
Accuracy
Balanced Accuracy
Precision
Recall
F1 Score
MCC (Matthews Correlation Coefficient)

## Limitations
Dataset is limited to Facebook advertising campaigns
Dataset originates from 2017
No multimodal data (images/videos) included


## Future Work
Integrate multimodal ad data
Evaluate newer LLMs
Explore hybrid ML + LLM systems
Real-time adaptive prediction systems


## Author

Shahadat Hussain
Master’s Thesis
Faculty of Informatics
Brno, 2026

## Keywords

Few-shot Learning, Large Language Models, Facebook Ads, Campaign Prediction, Prompt Engineering, Tabular Data
