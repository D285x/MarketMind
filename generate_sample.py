"""Run once to generate sample_data/retail_sample.csv"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 90
dates = pd.date_range("2024-01-01", periods=n, freq="D")

trend = np.linspace(40000, 72000, n)
seasonal = 6000 * np.sin(np.linspace(0, 4 * np.pi, n))
noise = np.random.normal(0, 2000, n)
revenue = np.maximum(trend + seasonal + noise, 5000)

ad_spend = revenue * np.random.uniform(0.08, 0.14, n)
orders = (revenue / np.random.uniform(85, 115, n)).astype(int)
avg_order_value = revenue / orders
returns = (orders * np.random.uniform(0.02, 0.09, n)).astype(int)
conversion_rate = np.random.uniform(0.021, 0.058, n)
discount_spend = revenue * np.random.uniform(0.03, 0.08, n)
sessions = (orders / conversion_rate).astype(int)
new_customers = (orders * np.random.uniform(0.3, 0.6, n)).astype(int)
repeat_customers = orders - new_customers

df = pd.DataFrame({
    "date": dates,
    "revenue": revenue.round(2),
    "orders": orders,
    "avg_order_value": avg_order_value.round(2),
    "ad_spend": ad_spend.round(2),
    "discount_spend": discount_spend.round(2),
    "returns": returns,
    "conversion_rate": (conversion_rate * 100).round(3),
    "sessions": sessions,
    "new_customers": new_customers,
    "repeat_customers": repeat_customers,
})

df.to_csv("sample_data/retail_sample.csv", index=False)
print("✅  sample_data/retail_sample.csv written")
print(df.head(3).to_string())
