# %% [markdown]
# # Iris Classification Lab - MLflow Experiment Tracking
# This lab demonstrates MLflow experiment tracking using the Iris dataset instead of wine quality data.

# %%
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from mlflow.models.signature import infer_signature
import cloudpickle
import time

# %% [markdown]
# ## Step 1: Data Exploration
# Visualizing the distribution of Iris classes.

# %%
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="quality")
X.head()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize distribution of target classes
sns.histplot(y, kde=False)
plt.title("Distribution of Iris Classes")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# ## Step 2: Exploratory Data Analysis (EDA)
# Box plots to identify key features for classification.

# %%
dims = (2, 2)
f, axes = plt.subplots(dims[0], dims[1], figsize=(12, 8))
axis_i, axis_j = 0, 0

for col in X.columns:
    sns.boxplot(x=y, y=X[col], ax=axes[axis_i, axis_j])
    axis_j += 1
    if axis_j == dims[1]:
        axis_i += 1
        axis_j = 0
plt.tight_layout()
plt.show()

# %% [markdown]
# Petal length and petal width appear to be the strongest predictors of Iris species.

# %% [markdown]
# ## Step 3: Check Missing Data

# %%
X.isna().any()

# %% [markdown]
# ## Step 4: Data Splitting

# %%
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# %% [markdown]
# ## Step 5: Build Baseline Model with MLflow Tracking

# %%
with mlflow.start_run(run_name='iris_random_forest'):
    n_estimators = 10
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    predictions_test = model.predict(X_test)
    acc = accuracy_score(y_test, predictions_test)
    
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_metric('accuracy', acc)
    
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "iris_rf_model", signature=signature)
    
    print(f"Accuracy: {acc:.4f}")
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")

# %% [markdown]
# ## Step 6: Feature Importance Analysis

# %%
feature_importances = pd.DataFrame(
    model.feature_importances_,
    index=X_train.columns.tolist(),
    columns=['importance']
)
feature_importances.sort_values('importance', ascending=False)

# %% [markdown]
# ## Step 7: Model Registration

# %%
model_name = "iris_quality"
model_version = mlflow.register_model(f"runs:/{run_id}/iris_rf_model", model_name)
time.sleep(10)
print(f"Model registered: version {model_version.version}")

# %% [markdown]
# ## Step 8: Load Production Model & Inference

# %%
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/1")
preds = loaded_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

# %%



