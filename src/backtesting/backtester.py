"""
Backtesting Engine
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predictor import ModelPredictor
from src.strategies.base_strategy import TradeSignal, Position


class Backtester:
    """Backtest trading strategies on historical data"""
    
    def __init__(
        self,
        model_path: str,
        initial_capital: float = 10000,
        min_confidence: float = 0.7,
        stop_loss_pct: float = 0.005,
        take_profit_pct: float = 0.01,
        max_positions: int = 3
    ):
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.min_confidence = min_confidence
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        
        self.positions = {}
        self.trade_history = []
        self.equity_curve = []
        
        # Load model - handle both US model (direct) and UK model (dict) formats
        try:
            # Try US model format (direct XGBClassifier)
            self.model = self._load_model_direct(model_path)
            self.feature_cols = self._load_features()
            self.scaler = self._load_scaler()
        except:
            # Fall back to UK model format (via ModelPredictor)
            self.predictor = ModelPredictor(model_path)
            if hasattr(self.predictor, 'feature_names'):
                self.feature_cols = self.predictor.feature_names
            else:
                self.feature_cols = []
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            data: Historical data with features
            
        Returns:
            Dictionary with backtest results
        """
        print("\n" + "=" * 70)
        print("🚀 Starting Backtest")
        print("=" * 70)
        
        # Sort by datetime
        data = data.sort_values('Datetime').reset_index(drop=True)
        
        # Get unique dates/bars
        unique_bars = data['Datetime'].unique()
        total_bars = len(unique_bars)
        
        print(f"📊 Total bars: {len(data)}")
        print(f"📊 Unique time steps: {len(unique_bars)}")
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print()
        
        # Process each time step
        for i, current_time in enumerate(unique_bars):
            current_data = data[data['Datetime'] == current_time].copy()
            
            # Get current prices for all positions
            current_prices = {}
            for ticker in self.positions.keys():
                ticker_data = current_data[current_data['Ticker'] == ticker]
                if len(ticker_data) > 0:
                    current_prices[ticker] = ticker_data['Close'].iloc[0]
            
            # Update positions and check exits
            exited = []
            time_obj = pd.to_datetime(current_time)
            
            for ticker, position in self.positions.items():
                if ticker in current_prices:
                    position.current_price = current_prices[ticker]
                    
                    # Check stop loss
                    if current_prices[ticker] <= position.stop_loss:
                        exited.append(ticker)
                        continue
                    
                    # Check take profit
                    if current_prices[ticker] >= position.take_profit:
                        exited.append(ticker)
                        continue
                    
                    # Check end of day (EOD) - force close at 16:30 UK time
                    if time_obj.hour >= 16 and time_obj.minute >= 30:
                        exited.append(ticker)
                        continue
            
            # Close exited positions
            for ticker in exited:
                self._close_position(ticker, current_prices.get(ticker, current_data[current_data['Ticker'] == ticker]['Close'].iloc[0] if len(current_data[current_data['Ticker'] == ticker]) > 0 else 0), current_time)
            
            # Check if we can open new positions
            if len(self.positions) < self.max_positions:
                # Generate signals for this bar
                signals = self._generate_signals(current_data)
                
                # Open positions
                for signal in signals[:self.max_positions - len(self.positions)]:
                    self._open_position(signal, current_time)
            
            # Track equity curve
            current_value = self.cash
            for ticker, position in self.positions.items():
                if ticker in current_prices:
                    current_value += position.quantity * current_prices[ticker]
                else:
                    current_value += position.quantity * position.current_price
            
            self.equity_curve.append({
                'time': current_time,
                'value': current_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })
            
            # Progress update
            if (i + 1) % 100 == 0 or (i + 1) == len(unique_bars):
                print(f"Progress: {i+1}/{len(unique_bars)} bars | Current value: ${current_value:,.2f}")
        
        # Close all remaining positions
        final_data = data.tail(1)
        for ticker in list(self.positions.keys()):
            ticker_data = final_data[final_data['Ticker'] == ticker]
            if len(ticker_data) > 0:
                self._close_position(ticker, ticker_data['Close'].iloc[0], final_data['Datetime'].iloc[0])
        
        # Calculate final metrics
        results = self._calculate_metrics()
        
        return results
    
    def _generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate trading signals from data"""
        signals = []
        
        # Get predictions
        predictions = self.predictor.predict_direction(data.copy())
        
        # Filter by confidence
        high_confidence = predictions[
            (predictions['Prediction'] == 1) & 
            (predictions['Confidence'] >= self.min_confidence)
        ]
        
        for _, row in high_confidence.iterrows():
            signal = TradeSignal(
                ticker=row['Ticker'],
                action='buy',
                confidence=row['Confidence'],
                entry_price=row['Close'],
                timestamp=str(row['Datetime']),
                stop_loss=row['Close'] * (1 - self.stop_loss_pct),
                take_profit=row['Close'] * (1 + self.take_profit_pct),
                position_size=None
            )
            signals.append(signal)
        
        return signals
    
    def _open_position(self, signal: TradeSignal, current_time):
        """Open a new position"""
        if signal.ticker in self.positions:
            return
        
        # Calculate position size using proper risk management
        risk_pct = getattr(self, 'initial_risk_pct', 0.01)  # Use custom or default 1%
        risk_amount = self.initial_capital * risk_pct
        
        # Calculate stop loss distance
        stop_loss_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        if stop_loss_distance == 0:
            stop_loss_distance = 0.005  # Default 0.5%
        
        # Position value = risk_amount / stop_loss_distance
        position_value = risk_amount / stop_loss_distance
        
        # Adjust by confidence
        position_value = position_value * signal.confidence
        
        # Limit to based on risk_pct (adjustable max)
        max_pct = min(risk_pct * 10, 0.20)  # Max 10x risk or 20% of capital
        max_position = self.cash * max_pct
        position_value = min(position_value, max_position)
        
        if position_value > self.cash or position_value < 10:
            return
        
        # Calculate quantity
        quantity = position_value / signal.entry_price
        
        position = Position(
            ticker=signal.ticker,
            entry_price=signal.entry_price,
            entry_time=str(current_time),
            current_price=signal.entry_price,
            current_time=str(current_time),
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0
        )
        
        self.positions[signal.ticker] = position
        self.cash -= position_value
    
    def _close_position(self, ticker: str, exit_price: float, current_time):
        """Close a position"""
        if ticker not in self.positions:
            return
        
        position = self.positions[ticker]
        
        # Calculate P&L
        pnl = (exit_price - position.entry_price) * position.quantity
        pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
        
        # Close position
        position_value = position.quantity * exit_price
        self.cash += position_value
        
        # Record trade
        self.trade_history.append({
            'ticker': ticker,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_time,
            'exit_time': str(current_time),
            'quantity': position.quantity,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'holding_bars': (pd.to_datetime(current_time) - pd.to_datetime(position.entry_time)).total_seconds() / 300  # 5 min bars
        })
        
        del self.positions[ticker]
    
    def _calculate_metrics(self) -> Dict:
        """Calculate backtest metrics"""
        final_value = self.cash
        
        # Calculate total return
        total_return = (final_value / self.initial_capital) - 1
        
        if not self.trade_history:
            return {
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'total_trades': 0,
                'initial_capital': self.initial_capital,
                'final_value': final_value
            }
        
        df_trades = pd.DataFrame(self.trade_history)
        
        # Calculate metrics
        total_pnl = df_trades['pnl'].sum()
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        win_rate = winning_trades / len(df_trades) if len(df_trades) > 0 else 0
        
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Calculate equity curve metrics
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['return'] = (equity_df['value'] / equity_df['value'].shift(1) - 1).fillna(0)
        
        sharpe_ratio = equity_df['return'].mean() / equity_df['return'].std() * np.sqrt(252 * 288) if equity_df['return'].std() > 0 else 0  # 288 = 24 hours * 12 (5-min bars per hour)
        max_drawdown = ((equity_df['value'] / equity_df['value'].cummax()) - 1).min()
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_pnl': total_pnl,
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_trades': len(df_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_bars': df_trades['holding_bars'].mean(),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

