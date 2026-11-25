"""
Process raw data and extract features for ML model training
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from tqdm import tqdm
from src.feature_engineering.feature_extractor import FeatureExtractor
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, DEFAULT_PERIOD, DEFAULT_INTERVAL

def process_all_tickers():
    """Process all ticker data files and add features"""
    
    print("=" * 60)
    print("🔧 Feature Engineering - Processing Data")
    print("=" * 60)
    print()
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Find all CSV files in raw data directory
    pattern = os.path.join(RAW_DATA_DIR, f"*_{DEFAULT_PERIOD}_{DEFAULT_INTERVAL}.csv")
    data_files = glob.glob(pattern)
    
    if not data_files:
        print(f"❌ No data files found in {RAW_DATA_DIR}")
        return
    
    print(f"📁 Found {len(data_files)} data files to process\n")
    
    processed_count = 0
    failed_count = 0
    
    for filepath in tqdm(data_files, desc="Processing"):
        try:
            # Load raw data
            df = pd.read_csv(filepath)
            
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            
            # Extract ticker from filename
            filename = os.path.basename(filepath)
            ticker = filename.split('_')[0]
            
            print(f"\n📊 Processing {ticker}...")
            
            # Add all features
            df_features = extractor.add_all_features(df)
            
            # Add target variable (predict next day direction)
            df_features = extractor.add_target_variable(df_features, forward_period=1, threshold=0.01)
            
            # Save processed data
            output_filename = f"{ticker}_features.csv"
            output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
            df_features.to_csv(output_path, index=False)
            
            # Print summary
            feature_count = len(df_features.columns) - len(df.columns)
            valid_rows = df_features.dropna().shape[0]
            
            print(f"   ✓ Added {feature_count} features")
            print(f"   ✓ {valid_rows} valid rows (after removing NaN)")
            print(f"   ✓ Saved to: {output_path}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {filepath}: {str(e)}")
            failed_count += 1
    
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"✅ Successfully processed: {processed_count}/{len(data_files)}")
    print(f"❌ Failed: {failed_count}/{len(data_files)}")
    
    if processed_count > 0:
        print(f"\n🎯 Ready for ML model training!")
        print(f"📁 Processed data saved to: {PROCESSED_DATA_DIR}")
    print()

def show_feature_summary():
    """Show summary of available features"""
    
    print("\n" + "=" * 60)
    print("📋 Available Features")
    print("=" * 60)
    
    # Load one processed file as example
    processed_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*.csv"))
    
    if not processed_files:
        print("⚠️  No processed files found. Run processing first.")
        return
    
    sample_file = processed_files[0]
    df = pd.read_csv(sample_file)
    
    print(f"\n📊 Sample file: {os.path.basename(sample_file)}")
    print(f"📈 Total columns: {len(df.columns)}")
    print(f"📏 Rows: {len(df)}")
    
    # Categorize features
    feature_categories = {
        'Price Data': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'Moving Averages': [col for col in df.columns if 'SMA_' in col or 'EMA_' in col],
        'Oscillators': [col for col in df.columns if any(ind in col for ind in ['RSI', 'Stoch', 'Williams', 'CCI'])],
        'Trend': [col for col in df.columns if any(ind in col for ind in ['MACD', 'ADX', 'ATR'])],
        'Bands': [col for col in df.columns if 'BB_' in col],
        'Volume': [col for col in df.columns if 'Volume' in col and col != 'Volume'],
        'Features': [col for col in df.columns if any(ind in col for ind in ['Cross', 'Position', 'Change', 'Momentum', 'Shadow', 'Gap'])]
    }
    
    for category, features in feature_categories.items():
        if features:
            print(f"\n{category} ({len(features)}):")
            for feat in features[:10]:  # Show first 10
                print(f"  • {feat}")
            if len(features) > 10:
                print(f"  ... and {len(features) - 10} more")

if __name__ == "__main__":
    process_all_tickers()
    show_feature_summary()

