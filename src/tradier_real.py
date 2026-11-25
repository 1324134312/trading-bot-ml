"""
Tradier Real-Time Data Integration

Uses Tradier's REST API for real-time market data and paper trading.
"""
import requests
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

class TradierAPI:
    """Tradier API client for real-time data"""
    
    def __init__(self, access_token=None, account_id=None, sandbox=True):
        """Initialize Tradier API"""
        self.access_token = access_token or os.getenv('TRADIER_ACCESS_TOKEN', '')
        self.account_id = account_id or os.getenv('TRADIER_ACCOUNT_ID', '')
        
        # Use sandbox for paper trading
        if sandbox:
            self.base_url = "https://sandbox.tradier.com"
        else:
            self.base_url = "https://api.tradier.com"
        
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }
    
    def _request(self, method, endpoint, params=None):
        """Make API request"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=self.headers, params=params)
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, json=params)
            else:
                return None
            
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            print(f"❌ API Error: {e}")
            return None
    
    def get_account_info(self):
        """Get account information"""
        endpoint = "/v1/accounts"
        data = self._request('GET', endpoint)
        
        if data and 'accounts' in data:
            return data['accounts']['account']
        return None
    
    def get_quote(self, symbol):
        """Get real-time quote"""
        endpoint = f"/v1/markets/quotes"
        params = {'symbols': symbol}
        data = self._request('GET', endpoint, params=params)
        
        if data and 'quotes' in data:
            quotes = data['quotes'].get('quote', [])
            if isinstance(quotes, list) and quotes:
                return quotes[0]
            return quotes
        return None
    
    def get_history(self, symbol, interval='daily', days=30):
        """Get historical data"""
        endpoint = f"/v1/markets/history"
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'days': days
        }
        
        data = self._request('GET', endpoint, params=params)
        
        if data and 'history' in data:
            return data['history']
        return None
    
    def place_order(self, symbol, quantity, order_type='market', side='buy', limit_price=None):
        """Place paper trade order"""
        endpoint = f"/v1/accounts/{self.account_id}/orders"
        
        params = {
            'symbol': symbol,
            'quantity': quantity,
            'side': side,
            'type': order_type,
            'duration': 'day'
        }
        
        if order_type == 'limit' and limit_price:
            params['price'] = limit_price
        
        data = self._request('POST', endpoint, params=params)
        return data
    
    def get_positions(self):
        """Get current positions"""
        endpoint = f"/v1/accounts/{self.account_id}/positions"
        data = self._request('GET', endpoint)
        
        if data and 'positions' in data:
            positions = data['positions']
            if positions and 'position' in positions:
                pos = positions['position']
                return pos if isinstance(pos, list) else [pos]
        return []
    
    def get_orders(self):
        """Get order history"""
        endpoint = f"/v1/accounts/{self.account_id}/orders"
        data = self._request('GET', endpoint)
        
        if data and 'orders' in data:
            orders = data['orders']
            if orders and 'order' in orders:
                ord = orders['order']
                return ord if isinstance(ord, list) else [ord]
        return []


def test_tradier():
    """Test Tradier connection"""
    print("=" * 60)
    print("🧪 Testing Tradier Connection")
    print("=" * 60)
    
    # Check for credentials
    access_token = os.getenv('TRADIER_ACCESS_TOKEN', '')
    
    if not access_token:
        print("\n📝 SETUP REQUIRED:")
        print("1. Go to: https://developer.tradier.com")
        print("2. Create free account")
        print("3. Get sandbox API token")
        print("4. Set environment variable:")
        print("   export TRADIER_ACCESS_TOKEN='your_token_here'")
        print("   export TRADIER_ACCOUNT_ID='your_account_here'")
        print("\n   Or edit src/tradier_real.py with your credentials")
        return None
    
    # Create client
    client = TradierAPI(sandbox=True)
    
    # Test account
    print("\n📊 Testing account...")
    account = client.get_account_info()
    
    if account:
        print("✅ Connected to Tradier!")
        if isinstance(account, list):
            print(f"Accounts: {len(account)}")
        else:
            print(f"Account: {account.get('account_number', 'N/A')}")
    
    # Test quote
    print("\n📊 Testing real-time quote...")
    quote = client.get_quote('AAPL')
    
    if quote:
        print("✅ Real-time data working!")
        print(f"AAPL: ${quote.get('last', 'N/A')}")
        print(f"Bid: ${quote.get('bid', 'N/A')}")
        print(f"Ask: ${quote.get('ask', 'N/A')}")
    else:
        print("⚠️  Could not get quotes")
    
    print("\n" + "=" * 60)
    print("✅ Tradier Ready!")
    print("=" * 60)
    
    return client


if __name__ == "__main__":
    client = test_tradier()
    
    if client:
        print("\n📚 USAGE IN YOUR BOT:")
        print("from src.tradier_real import TradierAPI")
        print("client = TradierAPI()")
        print("quote = client.get_quote('AAPL')")
        print("")

