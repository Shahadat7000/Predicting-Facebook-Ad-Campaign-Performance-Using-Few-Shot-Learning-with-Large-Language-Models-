#!/bin/bash

# Facebook Ad Prediction Thesis - Complete Environment Setup

echo "========================================="
echo "Setting up Facebook Ad Prediction Thesis Environment"
echo "========================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv thesis_env

# Activate environment
echo "Activating environment..."
source thesis_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Create requirements.txt
echo "Creating requirements.txt..."

cat > requirements.txt << 'EOF'
# Core Data Science
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.10.1
statsmodels==0.14.0

# Visualization
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.14.1

# API Integration
openai==1.3.0
requests==2.31.0
python-dotenv==1.0.0
tenacity==8.2.3

# Utilities
jupyter==1.0.0
ipython==8.14.0
tqdm==4.65.0

# Testing
pytest==7.4.0
pytest-cov==4.1.0
EOF

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create directory structure
echo "Creating project directories..."
mkdir -p data
mkdir -p results
mkdir -p visualizations
mkdir -p logs
mkdir -p notebooks
mkdir -p src

# Create .env file
echo "Creating .env file template..."
cat > .env << 'EOF'
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
PRIMARY_MODEL=gpt-4o-mini
SECONDARY_MODEL=gpt-3.5-turbo
TEMPERATURE=0.0
MAX_TOKENS=150

# Paths
DATA_PATH=data/data.csv
RESULTS_PATH=results/
LOG_PATH=logs/experiment.log

# Experiment Configuration
RANDOM_SEED=42
TEST_SIZE=0.2
BATCH_SIZE=10
EOF

echo "========================================="
echo "Environment setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Add your OpenAI API key to the .env file"
echo "2. Download dataset from Kaggle and place in data/data.csv"
echo "3. Activate environment: source thesis_env/bin/activate"
echo "4. Run experiments: python src/main_analysis.py"
echo "========================================="