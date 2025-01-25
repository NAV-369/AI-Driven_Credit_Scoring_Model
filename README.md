# Task 3: Feature Engineering

## Overview
Feature engineering transforms raw data into more predictive features, crucial for enhancing model performance in credit risk assessment.

## Objectives

- Create aggregate features to summarize customer behavior.
- Extract time-based features to capture temporal patterns.
- Encode categorical variables for machine learning compatibility.
- Handle missing values to maintain data quality.
- Normalize or standardize numerical features for consistent scaling.
- Implement WOE and IV for feature transformation and selection.

## Implementation Details

### # Create Aggregate Features
- **Total Transaction Amount:** Sum of transactions per customer.
  ```python
  df['TotalTransactionAmount'] = df.groupby('AccountId')['Amount'].transform('sum')
Average Transaction Amount: Mean transaction amount per customer.
python
df['AverageTransactionAmount'] = df.groupby('AccountId')['Amount'].transform('mean')
Transaction Count: Number of transactions per customer.
python
df['TransactionCount'] = df.groupby('AccountId')['TransactionId'].transform('count')
Standard Deviation of Transaction Amounts: Variability in transaction amounts.
python
df['StdTransactionAmount'] = df.groupby('AccountId')['Amount'].transform('std')

# Extract Features
Time-Based Features: From transaction timestamps for pattern analysis.
python
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
df['TransactionHour'] = df['TransactionStartTime'].dt.hour
df['TransactionDay'] = df['TransactionStartTime'].dt.day
df['TransactionMonth'] = df['TransactionStartTime'].dt.month
df['TransactionYear'] = df['TransactionStartTime'].dt.year

# Encode Categorical Variables
One-Hot Encoding: For ChannelId.
python
df = pd.get_dummies(df, columns=['ChannelId'], prefix=['Channel'])
Label Encoding: For ProductCategory when necessary.
python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['ProductCategory_encoded'] = le.fit_transform(df['ProductCategory'])

# Handle Missing Values
Imputation: Mean or mode strategy.
python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
Removal: If the missing data is insignificant.

# Normalize/Standardize Numerical Features
Normalization: To scale between 0 and 1.
python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
Standardization: To center around 0 with unit variance.
python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Feature Engineering with WOE and IV
Weight of Evidence (WOE) and Information Value (IV): For feature enhancement and selection.
python
def woe_iv(X, y, event=1):
    # Simplified WOE/IV calculation
    ...
X = df['ProductCategory']
y = df['FraudResult']
woe_df = woe_iv(X, y)
df = df.merge(woe_df['WOE'], left_on='ProductCategory', right_index=True, how='left', suffixes=('', '_WOE'))

### How to Use
Jupyter Notebook: Execute ./notebooks/03_feature_engineering.ipynb for this task's code.
Interpret: Review visualizations and statistics for insights.

### Next Steps
Model Training: Utilize these features in credit risk models.
Feature Selection: Prioritize based on IV for model inclusion.

### References
WOE and IV Explanation (link_to_woe_iv_explanation)
Feature Engineering Techniques (link_to_feature_engineering)

```
