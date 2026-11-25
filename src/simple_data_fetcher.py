"""
Simple Real-Time Data Fetcher - No Broker Needed!

Uses Yahoo Finance (already installed) for live market data.
Works in UK, completely FREE!
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class SimpleDataFetcher:
    """Fetch market data directly - no broker needed"""
    
    def __init__(self):
        """Initialize data fetcher"""
        print("✅ Using Yahoo Finance for FREE market data")
    
    def get_current_price(self, symbol):
        """Get current price"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            return price
        except:
            return None
    
    def get_quote(self, symbol):
        """Get real-time quote"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            quote = {
                'symbol': symbol,
                'price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'volume': info.get('volume'),
                'market_cap': info.get('marketCap')
            }
            return quote
        except:
            return None
    
    def get_history(self, symbol, period="5d", interval="5m"):
        """Get historical data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except:
            return None
    
    def get_intraday_data(self, symbol, days=1):
        """Get intraday data (5-minute bars)"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get recent intraday data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval="5m"
            )
            
            return data
        except:
            return None


def test_fetcher():
    """Test the simple data fetcher"""
    print("=" * 60)
    print("🧪 Testing Simple Data Fetcher (No Broker!)")
    print("=" * 60)
    
    fetcher = SimpleDataFetcher()
    
    # Test symbols (UK and US)
    test_symbols = ['AAPL', 'MSFT', 'TSLA', 'BT.L', 'VOD.L']
    
    print("\n📊 Testing Current Prices:\n")
    
    for symbol in test_symbols:
        try:
            price = fetcher.get_current_price(symbol)
            if price:
                print(f"✅ {symbol}: ${price:.2f}")
            else:
                print(f"❌ {symbol}: No data")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    # Test getting intraday data
    print("\n📊 Testing Intraday Data:\n")
    
    try:
        data = fetcher.get_intraday_data('AAPL', days=1)
        if data is not None and not data.empty:
            print(f"✅ Got {len(data)} bars for AAPL")
            print(f"   Latest: {data.index[-1]}")
            print(f"   Price: ${data['Close'].iloc[-1]:.2f}")
        else:
            print("❌ Could not get intraday data")
    except Exception as e:
        print(f"❌ Error getting intraday data: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Data Fetcher Ready!")
    print("=" * 60)
    print("\n📝 No broker needed!")
    print("📝 FREE market data!")
    print("📝 Works in UK!")
    print("📝 Use this for testing!")


if __name__ == "__main__":
    test_fetcher()

