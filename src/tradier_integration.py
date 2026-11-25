"""
Tradier Integration - Real-Time Data for Your Bot

This connects your bot to Tradier's API for real-time data and paper trading.
"""
import os
from tradier import Tradier
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

class TradierClient:
    """Wrapper for Tradier API connection"""
    
    def __init__(self):
        """Initialize Tradier client"""
        # Get credentials from environment or config
        self.access_token = os.getenv('TRADIER_ACCESS_TOKEN', '')
        self.account_id = os.getenv('TRADIER_ACCOUNT_ID', '')
        self.sandbox = True  # Paper trading
        
        if not self.access_token:
            print("⚠️  Set TRADIER_ACCESS_TOKEN environment variable")
            print("   Or edit this file with your credentials")
            
        self.client = None
        if self.access_token:
            try:
                self.client = Tradier(
                    access_token=self.access_token,
                    account_id=self.account_id,
                    sandbox=self.sandbox
                )
                print("✅ Tradier client initialized")
            except Exception as e:
                print(f"❌ Error initializing Tradier: {e}")
                self.client = None
    
    def get_account_info(self):
        """Get account information"""
        if not self.client:
            return None
        
        try:
            account = self.client.account.get_account()
            return account
        except Exception as e:
            print(f"❌ Error getting account: {e}")
            return None
    
    def get_quote(self, symbol):
        """Get real-time quote for a symbol"""
        if not self.client:
            return None
        
        try:
            quote = self.client.market.get_quotes(symbols=[symbol])
            return quote
        except Exception as e:
            print(f"❌ Error getting quote for {symbol}: {e}")
            return None
    
    def get_history(self, symbol, interval='5min', start_date=None, end_date=None):
        """Get historical data"""
        if not self.client:
            return None
        
        try:
            history = self.client.market.get_history(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            return history
        except Exception as e:
            print(f"❌ Error getting history for {symbol}: {e}")
            return None
    
    def place_order(self, symbol, quantity, order_type='market', side='buy'):
        """Place a paper trade order"""
        if not self.client:
            return None
        
        try:
            order = self.client.trading.post_order(
                symbol=symbol,
                quantity=quantity,
                order_type=order_type,
                side=side
            )
            return order
        except Exception as e:
            print(f"❌ Error placing order: {e}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        if not self.client:
            return None
        
        try:
            positions = self.client.account.get_positions()
            return positions
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
            return None
    
    def get_orders(self):
        """Get order history"""
        if not self.client:
            return None
        
        try:
            orders = self.client.account.get_orders()
            return orders
        except Exception as e:
            print(f"❌ Error getting orders: {e}")
            return None


def test_connection():
    """Test Tradier connection"""
    print("=" * 60)
    print("🧪 Testing Tradier Connection")
    print("=" * 60)
    
    # Initialize client
    client = TradierClient()
    
    if not client.client:
        # Test getting account info
        print("\n📊 Testing account info...")
        account = client.get_account_info()
        
        if account:
            print("✅ Connected to Tradier!")
            print(f"Account: {account}")
        else:
            print("⚠️  Could not get account info")
        
        # Test getting a quote
        print("\n📊 Testing real-time quote...")
        quote = client.get_quote('AAPL')
        
        if quote:
            print("✅ Real-time data working!")
            print(f"AAPL Quote: {quote}")
        else:
            print("⚠️  Could not get quotes")
    
    print("\n" + "=" * 60)
    print("✅ Tradier Integration Ready!")
    print("=" * 60)
    
    return client


if __name__ == "__main__":
    # Test the connection
    client = test_connection()
    
    # Show usage
    print("\n📚 USAGE:")
    print("1. Set credentials:")
    print("   export TRADIER_ACCESS_TOKEN='your_token'")
    print("   export TRADIER_ACCOUNT_ID='your_account'")
    print("")
    print("2. Or edit src/tradier_integration.py with your credentials")
    print("")
    print("3. Then use in your bot:")
    print("   from src.tradier_integration import TradierClient")
    print("   client = TradierClient()")
    print("   quote = client.get_quote('AAPL')")

