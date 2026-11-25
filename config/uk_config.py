"""
UK Stock Trading Configuration
"""
import os

# UK stock directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UK_DATA_DIR = os.path.join(BASE_DIR, "data/uk")
UK_RAW_DIR = os.path.join(UK_DATA_DIR, "raw")
UK_PROCESSED_DIR = os.path.join(UK_DATA_DIR, "processed")
UK_MODELS_DIR = os.path.join(BASE_DIR, "src/models/uk_models")

# Create directories
for directory in [UK_DATA_DIR, UK_RAW_DIR, UK_PROCESSED_DIR, UK_MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# FTSE 100 UK Stocks
UK_TICKERS = [
    "BP.L",      # BP (Energy)
    "SHEL.L",    # Shell (Energy)
    "GSK.L",     # GlaxoSmithKline (Pharma)
    "AZN.L",     # AstraZeneca (Pharma)
    "ULVR.L",    # Unilever (Consumer)
    "DGE.L",     # Diageo (Beverages)
    "RIO.L",     # Rio Tinto (Mining)
    "BHP.L",     # BHP (Mining)
    "VOD.L",     # Vodafone (Telecom)
    "BT-A.L",    # BT Group (Telecom)
]

# Day trading parameters
INTRADAY_INTERVAL = "1h"  # 1-hour bars for day trading
DAYS_BACK = 60  # 60 days of intraday data (~1500 bars)
TARGET_FORWARD_HOURS = 2  # Predict 2 hours ahead

# Validation
MIN_INTRADAY_POINTS = 500  # Need at least 500 hours of data

