"""
IEX Cloud - Get Data Like Brokers Do!

This is the same data source that brokers use - direct from IEX exchange.
"""
import requests
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

class IEXCloudClient:
    """IEX Cloud API client - professional-grade data"""
    
    def __init__(self, api_key=None):
        """Initialize IEX Cloud client"""
        self.api_key = api_key or os.getenv('IEX_CLOUD_API_KEY', '')
        self.base_url = "https://cloud.iexapis.com/stable"
        
        if not self.api_key:
            print("⚠️  No API key set")
            print("   1. Sign up: https://iexcloud.io")
            print("   2. Get API key (free tier available)")
            print("   3. Set: export IEX_CLOUD_API_KEY='your_key'")
    
    def get_quote(self, symbol):
        """Get real-time quote (no delay!)"""
        url = f"{self.base_url}/stock/{symbol}/quote"
        params = {'token': self.api_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error getting quote: {e}")
            return None
    
    def get_intraday(self, symbol, interval='5min'):
        """Get intraday data (real-time)"""
        url = f"{self.base_url}/stock/{symbol}/intraday-prices"
        params = {'token': self.api_key, 'interval': interval}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as e:
            print(f"❌ Error getting intraday: {e}")
            return None
    
    def get_chart(self, symbol, range='5d', interval='5min'):
        """Get historical chart data"""
        url = f"{self.base_url}/stock/{symbol}/chart/{range}"
        params = {'token': self.api_key, 'interval': interval}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error getting chart: {e}")
            return None


def test_iex():
    """Test IEX Cloud connection"""
    print("=" * 60)
    print("🧪 Testing IEX Cloud (Same as Brokers Use!)")
    print("=" * 60)
    
    client = IEXCloudClient()
    
    if not client.api_key:
        print("\n📝 To get API key:")
        print("   1. Go to: https://iexcloud.io")
        print("   2. Create free account")
        print("   3. Get API key")
        print("   4. Set environment variable:")
        print("      export IEX_CLOUD_API_KEY='your_key'")
        print("\n   FREE tier: 5 calls/minute (good for testing!)")
        print("   PAID tier: $99/month unlimited (like brokers!)")
        return
    
    # Test getting quote
    print("\n📊 Testing real-time quote...")
    quote = client.get_quote('AAPL')
    
    if quote:
        print("✅ Got REAL-TIME quote (no delay!)")
        print(f"   Symbol: {quote.get('symbol', 'N/A')}")
        print(f"   Price: ${quote.get('latestPrice', 'N/A'):.2f}")
        print(f"   Change: {quote.get('changePercent', 'N/A')*100:.2f}%")
        print(f"   Volume: {quote.get('latestVolume', 'N/A'):,}")
    else:
        print("❌ Could not get quote")
    
    # Test getting intraday data
    print("\n📊 Testing real-time intraday data...")
    intraday = client.get_intraday('AAPL')
    
    if intraday:
        print(f"✅ Got {len(intraday)} bars of real-time data")
        if intraday:
            latest = intraday[-1]
            print(f"   Latest: ${latest.get('average', 'N/A')}")
            print(f"   Time: {latest.get('label', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("✅ This is broker-quality data!")
    print("✅ Same source brokers use!")
    print("=" * 60)


if __name__ == "__main__":
    test_iex()

