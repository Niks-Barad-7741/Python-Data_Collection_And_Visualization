#  Data Cleaning & Preprocessing Pipeline

This project demonstrates a **complete data preprocessing pipeline**
used in real-world **Machine Learning and Data Science workflows**.

The dataset contains messy HR employee data including missing values,
duplicates, outliers, and categorical variables that need transformation
before training ML models.

------------------------------------------------------------------------

#  DATA CLEANING

## 1️⃣ Handling Missing Values

### What is it?

Handling missing values means **filling empty/null cells in the
dataset**.

### Why does it happen in real life?

-   Employee didn't fill the form
-   System crashed during data entry
-   Data was never collected

### Example from dataset

  Column   Missing Values
  -------- ----------------
  Age      18
  Salary   8
  Bonus    22

### Why it matters for Machine Learning

ML models **cannot work with missing values**.

Problems caused by missing values: - Models crash - Incorrect
predictions - Biased training

After handling missing values → **model can train properly** 

### Implementation

``` python
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())
```

------------------------------------------------------------------------

## 2️⃣ Removing Duplicates

### What is it?

Removing rows that appear **multiple times** in the dataset.

### Why does it happen?

-   Data entered twice
-   Systems merged together
-   Form submitted multiple times

### Dataset Example

Original dataset:

210 rows

Duplicates detected:

10 rows

After cleaning:

200 rows

### Why it matters for ML

Duplicates cause **model bias and overfitting**.

The model may learn the **same pattern multiple times**, which leads to
incorrect predictions.

### Implementation

``` python
df = df.drop_duplicates(subset=['Employee_ID'])
```

------------------------------------------------------------------------

## 3️⃣ Outlier Detection & Treatment

### What are Outliers?

Outliers are values that are **far away from normal values**.

### Why do they occur?

-   Data entry mistakes
-   System bugs
-   Rare extreme cases

### Dataset Examples

  Column   Outlier Example
  -------- -----------------
  Age      -10, 150, 999
  Salary   500, 999999
  Bonus    -50000

These values are **not realistic**.

### Why it matters for ML

Outliers can **destroy model accuracy**.

### Implementation (IQR Method)

``` python
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df[col] = df[col].clip(lower, upper)
```

------------------------------------------------------------------------

#  DATA TRANSFORMATION

## 4️⃣ Normalization

Normalization scales values **between 0 and 1**.

Formula:

(value - min) / (max - min)

### Implementation

``` python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalized[cols] = scaler.fit_transform(df_normalized[cols])
```

Best for: - Neural Networks - KNN - Image processing

------------------------------------------------------------------------

##  Standardization

Standardization transforms data so:

Mean = 0\
Standard Deviation = 1

Formula:

(value - mean) / std

### Implementation

``` python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_standardized[cols] = scaler.fit_transform(df_standardized[cols])
```

Best for: - Linear Regression - Logistic Regression - SVM

------------------------------------------------------------------------

##  Encoding Categorical Variables

Machine learning models cannot understand text values like:

Male, Female, Mumbai, Delhi.

These must be converted to numbers.

### Example Columns

-   Gender
-   Department
-   City
-   Education
-   Remote_Work

### Implementation

``` python
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
```

This converts text categories into numeric columns.

------------------------------------------------------------------------

##  Feature Scaling

Feature Scaling ensures all features have a similar range.

Feature Scaling includes:

-   Normalization
-   Standardization

### Why it matters

Without scaling, models assume larger numeric values are more important.

Scaling ensures models learn **patterns instead of magnitude**.

Algorithms that require scaling:

-   KNN
-   SVM
-   Neural Networks
-   Linear Regression
-   Logistic Regression

Algorithms that do NOT require scaling:

-   Decision Trees
-   Random Forest
-   XGBoost

------------------------------------------------------------------------

#  Generated Datasets

The preprocessing pipeline produces multiple datasets:

-   Cleaned_Messy_HR_Dataset.csv
-   Normalized_HR_Dataset.csv
-   Standardized_HR_Dataset.csv
-   Final_Preprocessed_HR_Dataset.csv

------------------------------------------------------------------------

#  Final Result

This project demonstrates a **complete ML preprocessing pipeline**
including:

-   Missing value handling
-   Duplicate removal
-   Outlier detection & treatment
-   Data transformation
-   Normalization
-   Standardization
-   Categorical encoding
-   Feature scaling
