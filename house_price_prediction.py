import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = {
    'Area (m2)': [50, 60, 80, 100, 120, 150, 200, 250, 300, 350],
    'Bedrooms': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'Bathrooms': [1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
    'Price (million VND)': [500, 600, 800, 1000, 1200, 1500, 2000, 2500, 3000, 3500]
}
df = pd.DataFrame(data)
print("Sample Data:")
print(df.head())


X = df[['Area (m2)', 'Bedrooms', 'Bathrooms']]  # Features
y = df['Price (million VND)']  # House prices


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")


print("\nHouse price predictions on test set:")
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)


plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Price')
plt.plot(range(len(y_pred)), y_pred, color='red', label='Predicted Price')
plt.xlabel('Sample')
plt.ylabel('Price (million VND)')
plt.title('Comparison of Actual vs Predicted Prices')
plt.legend()
plt.show()


import joblib
joblib.dump(model, 'house_price_model.pkl')
print("Model saved to 'house_price_model.pkl'")