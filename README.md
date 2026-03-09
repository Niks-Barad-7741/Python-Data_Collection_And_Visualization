# Data Cleaning, EDA & Data Integration Pipeline

This project demonstrates a **complete data preprocessing and exploratory analysis pipeline**
used in real-world **Machine Learning and Data Science workflows**.

The dataset contains messy HR employee data including missing values,
duplicates, outliers, and categorical variables that need transformation
before training ML models.

---

# DATA CLEANING

## 1️⃣ Handling Missing Values

### What is it?
Handling missing values means **filling empty/null cells in the dataset**.

### Why does it happen in real life?
- Employee didn't fill the form
- System crashed during data entry
- Data was never collected

### Example from dataset

| Column | Missing Values |
|--------|---------------|
| Age | 18 |
| Salary | 8 |
| Bonus | 22 |

### Why it matters for Machine Learning
ML models **cannot work with missing values**.

Problems caused by missing values:
- Models crash
- Incorrect predictions
- Biased training

After handling missing values → **model can train properly** ✅

### Implementation
```python
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())
```

---

## 2️⃣ Removing Duplicates

### What is it?
Removing rows that appear **multiple times** in the dataset.

### Why does it happen?
- Data entered twice
- Systems merged together
- Form submitted multiple times

### Dataset Example

| Stage | Rows |
|-------|------|
| Original dataset | 210 |
| Duplicates detected | 10 |
| After cleaning | 200 |

### Why it matters for ML
Duplicates cause **model bias and overfitting**.
The model learns the same pattern multiple times → incorrect predictions.

### Implementation
```python
df = df.drop_duplicates(subset=['Employee_ID'])
```

---

## 3️⃣ Outlier Detection & Treatment

### What are Outliers?
Outliers are values that are **far away from normal values**.

### Why do they occur?
- Data entry mistakes
- System bugs
- Rare extreme cases

### Dataset Examples

| Column | Outlier Example |
|--------|----------------|
| Age | -10, 150, 999 |
| Salary | 500, 999999 |
| Bonus | -50000 |

### Why it matters for ML
Outliers can **destroy model accuracy**.

### Implementation (IQR Method)
```python
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df[col] = df[col].clip(lower, upper)
```

---

# DATA TRANSFORMATION

## 4️⃣ Normalization

Normalization scales values **between 0 and 1**.

**Formula:** `(value - min) / (max - min)`

### Implementation
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalized[cols] = scaler.fit_transform(df_normalized[cols])
```

**Best for:** Neural Networks, KNN, Image processing

---

## 5️⃣ Standardization

Standardization transforms data so **Mean = 0** and **Standard Deviation = 1**.

**Formula:** `(value - mean) / std`

### Implementation
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_standardized[cols] = scaler.fit_transform(df_standardized[cols])
```

**Best for:** Linear Regression, Logistic Regression, SVM

---

## 6️⃣ Encoding Categorical Variables

ML models cannot understand text values like Male, Female, Mumbai, Delhi.
These must be converted to numbers.

### Example Columns
- Gender → Label Encoding (Male=1, Female=0)
- Department → One Hot Encoding (no order)
- City → One Hot Encoding (no order)
- Education → Ordinal Encoding (Diploma < Bachelor < Master < PhD)
- Remote_Work → Binary Encoding (Yes=1, No=0)

### Implementation
```python
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
```

`drop_first=True` removes redundant columns and prevents the **Dummy Variable Trap**.

---

## 7️⃣ Feature Scaling

Feature Scaling ensures all features have a **similar numeric range**.

Feature Scaling includes:
- Normalization (0 to 1)
- Standardization (mean=0, std=1)

### Why it matters
Without scaling, models assume larger numeric values are more important.
Scaling ensures models learn **patterns instead of magnitude**.

**Algorithms that require scaling:** KNN, SVM, Neural Networks, Linear Regression, Logistic Regression

**Algorithms that do NOT require scaling:** Decision Trees, Random Forest, XGBoost

---

# EXPLORATORY DATA ANALYSIS (EDA)

## 8️⃣ Summary Statistics

### What is it?
Getting a **quick numerical overview** of the dataset without reading every row.

### Key functions used

```python
df.describe()       # count, mean, std, min, max, quartiles
df.info()           # column names, data types, null counts
df.shape            # (rows, columns)
df.value_counts()   # frequency of each category
```

### Key findings from HR Dataset

| Metric | Value |
|--------|-------|
| Total Employees | 200 |
| Average Age | 40.8 years |
| Average Salary | 75,536 |
| Average Bonus | 10,419 |
| Remote Workers | 35% (70 employees) |
| Largest Department | HR (39 employees) |
| Most Common City | Bangalore (43 employees) |

---

## 9️⃣ Data Visualization

### What is it?
Converting numbers into **charts and graphs** so patterns become visible instantly.

### Libraries used
- **Matplotlib** — basic plots, full control
- **Seaborn** — statistical plots, beautiful defaults

### Charts created

| Chart Type | Purpose | Finding |
|-----------|---------|---------|
| Bar Chart | Employees by Department | HR most (39), IT least (30) |
| Histogram | Age Distribution | Peak at 40–45 age group |
| Box Plot | Salary by Department | Finance has widest spread |
| Pie Chart | Remote Work | 65% office, 35% remote |
| Scatter Plot | Experience vs Salary | No clear pattern |
| Bar Chart | Avg Salary by Education | All levels earn ~75k equally |

### Implementation
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# Bar, Histogram, Box Plot, Pie, Scatter, Bar
plt.tight_layout()
plt.savefig('HR_EDA_Visualization.png', dpi=150)
plt.show()
```

---

## 🔟 Identifying Patterns & Correlations

### What is it?
Finding **relationships between columns** using statistical correlation.

**Correlation scale:** -1 (perfect negative) → 0 (no relation) → +1 (perfect positive)

### Implementation
```python
corr_matrix = df[numeric_cols].corr()

sns.heatmap(corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0)
plt.title('Correlation Heatmap')
plt.show()
```

### Key findings

| Pair | Correlation | Meaning |
|------|------------|---------|
| Age vs Experience | ~0.00 | No relation (random dataset) |
| Salary vs Bonus | ~0.06 | Very weak |
| All columns | ~0.00 | Dataset was randomly generated |

> **Real world datasets** would typically show Experience ↑ → Salary ↑ (0.65+)

---

# DATA INTEGRATION

## 1️⃣1️⃣ Merging & Joining Datasets

### What is it?
Combining **two or more tables** into one — just like SQL JOIN.

### Why needed?
Real world data is never in one file. HR system, Salary system, and Attendance system are all separate.

### 4 types of joins

| Join Type | Result | Use Case |
|-----------|--------|---------|
| INNER | Only matching rows | Employees WITH salary data |
| LEFT | All left + matched right | All employees, NaN if no salary |
| RIGHT | All right + matched left | All salary records |
| OUTER | Everything from both | Complete data from both tables |

### Implementation
```python
# INNER JOIN — only matched rows (150)
inner = pd.merge(df_employees, df_salary_partial, on='Employee_ID', how='inner')

# LEFT JOIN — all left rows (200), NaN where unmatched
left = pd.merge(df_employees, df_salary_partial, on='Employee_ID', how='left')

# OUTER JOIN — everything (200)
outer = pd.merge(df_employees, df_salary_partial, on='Employee_ID', how='outer')
```

### Dataset result

| Join Type | Rows | Notes |
|-----------|------|-------|
| INNER | 150 | Only employees with salary |
| LEFT | 200 | 50 rows had NaN salary |
| OUTER | 200 | All data retained |

---

## 1️⃣2️⃣ Data Aggregation

### What is it?
**Summarizing data by groups** — like Excel PivotTables in Python.

### Implementation
```python
# Average salary by department
df.groupby('Department')['Salary'].mean()

# Multiple aggregations at once
df.groupby('Department').agg(
    Total_Employees  = ('Employee_ID', 'count'),
    Avg_Salary       = ('Salary', 'mean'),
    Max_Salary       = ('Salary', 'max'),
    Avg_Performance  = ('Performance_Score', 'mean')
)

# Pivot table — salary by Department AND Gender
df.pivot_table(values='Salary', index='Department', columns='Gender', aggfunc='mean')
```

### Key findings

| Department | Avg Salary | Employees |
|-----------|-----------|----------|
| FINANCE | 81,268 | 31 |
| HR | 78,713 | 39 |
| MARKETING | 76,850 | 32 |
| SALES | 74,244 | 31 |
| OPERATIONS | 73,226 | 37 |
| IT | 68,269 | 30 |

**Top earner:** Nikhil Singh — FINANCE — 1,56,248 🏆

---

## 1️⃣3️⃣ Handling Different Data Formats

### CSV (Comma Separated Values)

```python
# Read
df = pd.read_csv("file.csv")

# Write
df.to_csv("file.csv", index=False)
```

**Best for:** Flat tabular data, Excel exports, Data Science workflows

---

### JSON (JavaScript Object Notation)

```python
# Read
df = pd.read_json("file.json")

# Write
df.to_json("file.json", orient='records', indent=2)
```

**Best for:** APIs, web applications, nested/hierarchical data

---

### SQL (Structured Query Language)

```python
import sqlite3

conn = sqlite3.connect("HR_Database.db")

# Write to SQL
df.to_sql("employees", conn, if_exists='replace', index=False)

# Read with query
df = pd.read_sql("SELECT * FROM employees WHERE Department = 'IT'", conn)

# Aggregation in SQL
pd.read_sql("""
    SELECT Department, COUNT(*) as Total, ROUND(AVG(Salary), 2) as Avg_Salary
    FROM employees
    GROUP BY Department
    ORDER BY Avg_Salary DESC
""", conn)

conn.close()
```

**Best for:** Production systems, large datasets, multi-user environments

---

### Format Comparison

| Format | File Size | Read Function | Supports Nested | Best For |
|--------|----------|--------------|----------------|---------|
| CSV | 18 KB ✅ | read_csv() | No | Flat data |
| JSON | 69 KB ❌ | read_json() | Yes ✅ | APIs, web |
| SQL | 28 KB | read_sql() | No | Production |

> JSON is ~4x larger than CSV because **column keys repeat for every single row**.

---

# Generated Datasets

The preprocessing pipeline produces multiple output datasets:

| File | Description |
|------|-------------|
| Cleaned_Messy_HR_Dataset.csv | After cleaning (200 rows, 14 cols) |
| Normalized_HR_Dataset.csv | MinMax scaled 0–1 |
| Standardized_HR_Dataset.csv | Z-score scaled mean=0 std=1 |
| Final_Preprocessed_HR_Dataset.csv | Encoded, ML-ready (200 rows, 24 cols) |
| HR_Export.csv | CSV format export |
| HR_Export.json | JSON format export |
| HR_Database.db | SQLite database |

---

# Complete Pipeline Overview

```
RAW DATA (Messy CSV)
        ↓
┌─────────────────────┐
│    DATA CLEANING    │
│  Missing Values     │  → fillna(median)
│  Duplicates         │  → drop_duplicates()
│  Outliers           │  → IQR + clip()
│  Inconsistent Text  │  → str methods
└─────────────────────┘
        ↓
┌─────────────────────┐
│  DATA TRANSFORMATION│
│  Encoding           │  → get_dummies()
│  Normalization      │  → MinMaxScaler
│  Standardization    │  → StandardScaler
└─────────────────────┘
        ↓
┌─────────────────────┐
│        EDA          │
│  Summary Statistics │  → describe()
│  Visualization      │  → Matplotlib/Seaborn
│  Correlations       │  → corr() + heatmap
└─────────────────────┘
        ↓
┌─────────────────────┐
│  DATA INTEGRATION   │
│  Merging/Joining    │  → pd.merge()
│  Aggregation        │  → groupby(), agg()
│  CSV / JSON / SQL   │  → read/write all formats
└─────────────────────┘
        ↓
  ML-READY DATA 🚀
```

---

# Technologies Used

- **Python 3.11**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib** — data visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — preprocessing (MinMaxScaler, StandardScaler, LabelEncoder)
- **SQLite3** — SQL database operations
