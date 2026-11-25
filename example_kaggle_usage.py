"""
Example: How to Use Kaggle Datasets with Your Trading Bot

This is just an example - you probably won't use this for UK stocks!
"""

# STEP 1: Install Kaggle API
# pip install kaggle

# STEP 2: Authenticate
# Place your kaggle.json file in:
# Windows: C:\Users\YourUsername\.kaggle\kaggle.json

# STEP 3: Search for datasets
# Example: Search for stock data
"""
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Search
datasets = api.dataset_list(search='stock price historical')

for dataset in datasets:
    print(f"{dataset.ref}: {dataset.title}")
"""

# STEP 4: Download dataset
"""
# Example: US stock dataset
api.dataset_download_files(
    'borismarjanovic/price-volume-data-for-all-us-stocks-etfs',
    path='data/kaggle/',
    unzip=True
)
"""

# STEP 5: Use the data
"""
import pandas as pd

# Load downloaded data
data = pd.read_csv('data/kaggle/prices.csv')

# See what stocks are available
print(data['symbol'].unique()[:10])  # First 10 stocks

# Filter for specific stock
aapl = data[data['symbol'] == 'AAPL']

# Convert to datetime
aapl['date'] = pd.to_datetime(aapl['date'])

# Plot price chart
import matplotlib.pyplot as plt
plt.plot(aapl['date'], aapl['close'])
plt.title('AAPL Close Price')
plt.show()
"""

# STEP 6: Train your model (practice!)
"""
# Use similar features as UK stocks
from src.feature_engineering.indicators import TechnicalIndicators

indicators = TechnicalIndicators()
features = indicators.calculate_all(aapl)

# Train model on US data
from src.models.trainer import ModelTrainer

trainer = ModelTrainer()
model = trainer.train_model(features, target_column='close')
"""

# LIMITATION: This is US data, not UK!
# Use this for practice, not live trading!

print("""
SUMMARY:
- Kaggle has US stock datasets
- Useful for practice/learning
- NOT for UK 5-minute trading
- Stick with Yahoo Finance for UK!
""")

