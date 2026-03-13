# Iris Classification Lab - MLflow Experiment Tracking
### MLOps (IE-7374) - Experiment Tracking Lab

This lab demonstrates MLflow experiment tracking using the **Iris dataset** instead of the original wine quality dataset. It covers data preprocessing, model training, model registration, and inference using MLflow and scikit-learn.

## Modification
The original lab uses wine quality CSV datasets. This lab uses the **Iris dataset** from sklearn (built-in, no external files needed), making it more portable and reproducible.

## Lab Steps
1. **Data Exploration** - Load and explore the Iris dataset
2. **Data Visualization** - Distribution plots and EDA box plots
3. **Missing Data Check** - Verify data quality
4. **Data Splitting** - Train/validation/test split (60/20/20)
5. **Model Training** - Random Forest classifier with MLflow tracking
6. **Feature Importance** - Identify key predictors
7. **Model Registration** - Register model in MLflow Model Registry
8. **Production Deployment** - Transition model to Production stage
9. **Inference** - Load production model and make predictions

## Results
- **Model:** Random Forest Classifier (n_estimators=10)
- **Accuracy:** 90%
- **Registered Model:** iris_quality (Production)

## Project Structure
```
mlops-mlflow-lab/
├── Lab.ipynb          # Main notebook
├── requirements.txt   # Dependencies
└── README.md         # Documentation
```

## Setup

### Create and activate virtual environment
```bash
python -m venv lab_03
.\lab_03\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the notebook
Open `Lab.ipynb` in VS Code and run all cells.

### Launch MLflow UI
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser.

## Dependencies
- mlflow
- scikit-learn
- pandas
- numpy
- seaborn
- matplotlib
- cloudpickle