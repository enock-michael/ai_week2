import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load dataset
# -----------------------------
FILE_PATH = "CO2_Emissions_1960-2018.csv"
df = pd.read_csv(FILE_PATH)

# Rename for consistency
df = df.rename(columns={'Country Name': 'country'})

# Melt the data from wide → long format
df_long = df.melt(id_vars='country', var_name='year', value_name='co2')

# Convert 'year' to numeric
df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce')

# Drop missing CO2 values
df_long = df_long.dropna(subset=['co2'])

print(" Data loaded and transformed successfully!")
print(df_long.head())

# -----------------------------
# 2. Select example country
# -----------------------------
country_name = "Tanzania"
country_data = df_long[df_long['country'] == country_name].copy()

if country_data.empty:
    raise ValueError(f"Country '{country_name}' not found in dataset.")

# -----------------------------
# 3. Prepare data for regression
# -----------------------------
train = country_data[country_data['year'] <= 2013]
test = country_data[country_data['year'] > 2013]

X_train, y_train = train[['year']], train['co2']
X_test, y_test = test[['year']], test['co2']

# -----------------------------
# 4. Train Polynomial Regression model
# -----------------------------
model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# -----------------------------
# 5. Evaluate model
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n Model Evaluation for {country_name}:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# -----------------------------
# 6. Forecast future emissions
# -----------------------------
future_years = np.arange(2019, 2031)
future_pred = model.predict(future_years.reshape(-1, 1))

# -----------------------------
# 7. Visualization (Save as image)
# -----------------------------
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_test, y_pred, color='red', label='Predicted (Test)')
plt.plot(future_years, future_pred, color='orange',
         linestyle='--', label='Forecast (2019–2030)')
plt.title(f"CO₂ Emission Forecast for {country_name}")
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions (metric tons per capita)")
plt.legend()
plt.grid(True)

# Save figure
plot_file = f"{country_name}_CO2_forecast.png"
plt.savefig(plot_file)
print(f"Chart saved to {plot_file}")

# -----------------------------
# 8. Save forecast results
# -----------------------------
forecast_df = pd.DataFrame({
    'year': future_years,
    'predicted_co2': future_pred
})
forecast_csv = f"{country_name}_CO2_forecast.csv"
forecast_df.to_csv(forecast_csv, index=False)

print(f"Forecast data saved to {forecast_csv}")
