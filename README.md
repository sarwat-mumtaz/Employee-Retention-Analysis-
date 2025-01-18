**README: Employee Retention Analysis Using Logistic Regression**

---

### **Project Overview**
This project aims to assist the HR department in understanding factors that influence employee retention and predicting the likelihood of employees leaving the company. By analyzing the dataset collected by the HR team and building a predictive model using Logistic Regression, this project provides actionable insights to improve employee satisfaction and retention.

---

### **Problem Statement**
Employee turnover is costly and time-consuming for organizations. The HR department seeks to identify patterns and factors associated with employee attrition, enabling them to take proactive measures to enhance employee satisfaction and retention. The key question addressed in this project is:

> **"What factors are likely to make employees leave the company?"**

---

### **Dataset**
The dataset includes employee information such as:
- Satisfaction Level
- Last Evaluation Score
- Number of Projects
- Average Monthly Hours
- Time Spent at the Company
- Work Accident History
- Promotion in the Last 5 Years
- Department
- Salary Level

Target Variable:
- **Attrition** (1 = Employee left the company, 0 = Employee stayed)

---

### **Steps to Solution**
1. **Data Exploration and Visualization**
   - Understand the dataset.
   - Identify trends, patterns, and correlations using visualizations.
2. **Data Preprocessing**
   - Handle missing values, encode categorical variables, and scale numerical features.
3. **Model Building**
   - Train a Logistic Regression model to predict employee attrition.
4. **Evaluation**
   - Assess model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
5. **Recommendations**
   - Provide actionable insights based on the findings.

---

### **Key Dependencies**
- Python (>=3.8)
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

Install required libraries using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

### **Code Implementation**

#### **1. Importing Libraries**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
```

#### **2. Loading and Exploring Data**
```python
data = pd.read_csv('employee_data.csv')
print(data.head())
print(data.info())
print(data.describe())
```

#### **3. Data Visualization**
```python
sns.countplot(x='Attrition', data=data)
plt.title('Employee Attrition Distribution')
plt.show()

sns.boxplot(x='Attrition', y='SatisfactionLevel', data=data)
plt.title('Satisfaction Level vs Attrition')
plt.show()
```

#### **4. Data Preprocessing**
```python
# Encoding categorical variables
data = pd.get_dummies(data, columns=['Department', 'Salary'], drop_first=True)

# Splitting data into features and target
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### **5. Logistic Regression Model**
```python
# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
```

#### **6. Model Evaluation**
```python
# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Classification Report
print(classification_report(y_test, y_pred))

# ROC-AUC Score
roc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"ROC-AUC Score: {roc_score:.2f}")
```

---

### **Findings and Recommendations**
1. **Key Factors Influencing Attrition**:
   - Low satisfaction levels.
   - High average monthly working hours.
   - Lack of promotions.
   - Low salary levels.

2. **Recommendations**:
   - Conduct regular employee satisfaction surveys.
   - Promote a better work-life balance by monitoring working hours.
   - Create opportunities for career growth and recognition.
   - Review compensation strategies to ensure competitive salaries.

---

### **Conclusion**
By leveraging data analytics and Logistic Regression, this project provides a robust framework to identify key drivers of employee attrition. The insights gained can empower HR teams to implement targeted interventions, ultimately enhancing employee satisfaction and retention.

-
