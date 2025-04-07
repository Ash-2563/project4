# project4

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load Dataset
df = pd.read_csv(r"C:\Users\hp\Downloads\archive (1)\data.csv" )  # Replace with actual filename
print(df.head())

# 3. Check for Missing Data
print(df.isnull().sum())

# 4. Drop unnecessary columns (like street, date, country if not helpful)
df.drop(columns=['date', 'street', 'country'], inplace=True)

# 5. Feature Selection
features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view',
            'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'city', 'statezip']
target = 'price'

X = df[features]
y = df[target]

# 6. Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Preprocessing (Categorical + Numeric)
numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'sqft_above',
                    'sqft_basement', 'yr_built', 'yr_renovated']
categorical_features = ['waterfront', 'view', 'condition', 'city', 'statezip']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 8. Pipeline with Linear Regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 9. Train the Model
model.fit(X_train, y_train)

# 10. Predict
y_pred = model.predict(X_test)

# 11. Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {round(rmse, 2)}")
print(f"R-squared: {round(r2, 4)}")

# 12. Actual vs Predicted
results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print(results_df.head())

# 13. Optional Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()
