# Employee Salary Prediction

Employee Salary Prediction is a Machine Learning project that predicts whether an employee earns more than $50K or less than/equal to $50K annually based on demographic, educational, and employment-related attributes.

The project uses the Adult Census Income dataset and applies data preprocessing, feature engineering, and machine learning techniques for classification.

---

# Features

* Employee Salary Classification
* Data Cleaning and Preprocessing
* Handling Missing Values
* Exploratory Data Analysis (EDA)
* Feature Encoding
* Machine Learning Model Training
* Income Prediction (>50K or <=50K)
* Accuracy Evaluation

---

# Dataset Information

The project uses the Adult Census Income Dataset containing employee-related information such as:

* Age
* Workclass
* Education
* Occupation
* Marital Status
* Race
* Gender
* Working Hours
* Capital Gain/Loss
* Native Country
* Income Category

Dataset Shape:

```text
48842 Rows × 15 Columns
```

Target Variable:

```text
income
```

Classes:

* `<=50K`
* `>50K`

---

# Technologies Used

## Programming Language

* Python

## Libraries

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Environment

* Google Colab / Jupyter Notebook

---

# Project Workflow

## 1. Data Collection

The Adult Census dataset is loaded using Pandas.

```python
import pandas as pd

data = pd.read_csv("adult.csv")
```

---

## 2. Data Preprocessing

### Handling Missing Values

Missing categorical values represented by `?` are replaced with meaningful labels.

Example:

```python
data.workclass.replace({'?':'Others'}, inplace=True)
```

### Feature Cleaning

* Removed inconsistencies
* Handled categorical values
* Checked null values
* Prepared dataset for modeling

---

## 3. Exploratory Data Analysis (EDA)

Performed:

* Data shape analysis
* Distribution analysis
* Salary distribution visualization
* Feature relationship analysis

---

## 4. Feature Engineering

Categorical features converted into numerical format using encoding techniques.

Examples:

* Label Encoding
* One-Hot Encoding

---

## 5. Model Training

Machine Learning classification models are trained to predict employee salary categories.

Possible algorithms:

* Logistic Regression
* Decision Tree
* Random Forest
* Naive Bayes

---

## 6. Model Evaluation

Evaluation metrics include:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1 Score

---

# Project Structure

```bash
EmployeeSalaryPrediction/
│
├── EmployeeSalaryPrediction.ipynb
├── adult.csv
├── README.md
└── requirements.txt
```

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd EmployeeSalaryPrediction
```

---

# Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Packages

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

# Run Project

## Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
EmployeeSalaryPrediction.ipynb
```

---

# Sample Features

Input Features:

* age
* workclass
* education
* occupation
* marital-status
* relationship
* race
* gender
* capital-gain
* capital-loss
* hours-per-week
* native-country

Output:

```text
<=50K or >50K
```

---

# Key Insights

* Education level strongly impacts salary prediction.
* Capital gain is highly correlated with higher income.
* Marital status and occupation influence salary classification.
* Working hours contribute significantly to prediction accuracy.

---

# Future Improvements

* Deploy using Flask or Django
* Build Streamlit Web App
* Add Real-Time Prediction UI
* Hyperparameter Tuning
* Improve Accuracy using Ensemble Models
* Add Model Explainability

---

# Author

Developed by Sai Chaitanya

---

# License

This project is created for educational and learning purposes.
