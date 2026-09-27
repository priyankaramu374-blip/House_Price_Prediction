import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
df = pd.read_csv("dataset/archive (1)/house_data.xlsx.csv")

print("Dataset loaded successfully!")
print(df.head())


# Remove ID column
df = df.drop("Id", axis=1)


# Input and output
X = df.drop("Price", axis=1)
y = df["Price"]


# Categorical columns
categorical_columns = [
    "Location",
    "Condition",
    "Garage"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ML model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train
model.fit(X_train, y_train)

print("Model trained successfully!")


# Predict
y_pred = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


# Save model
joblib.dump(model, "house_price_model.pkl")

print("Model saved successfully!")