# Facebook Ad Performance Prediction Using Few-Shot Learning with GPT Models

## Thesis Research Project

This repository contains the complete code and documentation for a master's thesis investigating the application of few-shot learning with GPT models to predict Facebook ad campaign performance.

## 📋 Overview

The research explores whether Large Language Models (LLMs) can effectively predict ad campaign success from minimal examples, addressing the cold-start problem in marketing analytics. The study compares zero-shot, one-shot, and few-shot learning approaches across 1,000 Facebook ad campaigns.

## 🔬 Research Questions

- **RQ1**: Can GPT models achieve statistically significant accuracy (>50%) in predicting ad success?
- **RQ2**: How does accuracy vary across shot levels (0,1,3,5)?
- **RQ3**: Which metrics (CTR, CPC, Conversion Rate) are most predictable?
- **RQ4**: What are optimal prompt engineering strategies?
- **RQ5**: How does GPT compare to traditional ML (Logistic Regression, Random Forest)?

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key
- Dataset from [Kaggle](https://www.kaggle.com/datasets/madislemsalu/facebook-ad-campaign)

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd facebook-ad-prediction-thesis

# Run setup script
chmod +x setup_environment.sh
./setup_environment.sh

# Add your OpenAI API key to .env file
nano .env

# Download dataset and place in data/data.csv

# Run complete experiment
python run_experiment.py