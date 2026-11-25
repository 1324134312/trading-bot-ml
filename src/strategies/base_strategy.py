"""
Base Trading Strategy
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass

@dataclass
class TradeSignal:
    """Represents a trading signal"""
    ticker: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    entry_price: float
    timestamp: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None

@dataclass
class Position:
    """Represents an open position"""
    ticker: str
    entry_price: float
    entry_time: str
    current_price: float
    current_time: str
    quantity: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

class BaseStrategy(ABC):
    """Base class for trading strategies"""
    
    def __init__(self, model_path: str, initial_capital: float = 10000):
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # ticker -> Position
        self.trade_history = []
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate trading signals from data"""
        pass
    
    @abstractmethod
    def should_exit(self, position: Position, current_data: pd.DataFrame) -> bool:
        """Determine if should exit a position"""
        pass
    
    def execute_trade(self, signal: TradeSignal) -> Optional[Position]:
        """Execute a trade based on signal"""
        
        if signal.action == 'buy':
            return self._open_position(signal)
        elif signal.action == 'sell':
            return self._close_position(signal)
        
        return None
    
    def _open_position(self, signal: TradeSignal) -> Optional[Position]:
        """Open a new position"""
        
        if signal.ticker in self.positions:
            return None  # Already have position
        
        # Calculate position size based on risk
        position_value = self._calculate_position_size(
            signal, 
            entry_price=signal.entry_price
        )
        
        if position_value > self.cash:
            return None  # Not enough cash
        
        # Open position
        position = Position(
            ticker=signal.ticker,
            entry_price=signal.entry_price,
            entry_time=signal.timestamp,
            current_price=signal.entry_price,
            current_time=signal.timestamp,
            quantity=position_value / signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0
        )
        
        self.positions[signal.ticker] = position
        self.cash -= position_value
        
        return position
    
    def _close_position(self, signal: TradeSignal) -> Optional[Position]:
        """Close an existing position"""
        
        if signal.ticker not in self.positions:
            return None
        
        position = self.positions[signal.ticker]
        
        # Update with current price
        position.current_price = signal.entry_price  # Closing price
        position.current_time = signal.timestamp
        
        # Calculate P&L
        position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
        position.unrealized_pnl_pct = ((position.current_price - position.entry_price) / position.entry_price) * 100
        
        # Close position
        position_value = position.quantity * position.current_price
        self.cash += position_value
        
        # Record trade
        self.trade_history.append({
            'ticker': position.ticker,
            'entry_price': position.entry_price,
            'exit_price': position.current_price,
            'entry_time': position.entry_time,
            'exit_time': position.current_time,
            'quantity': position.quantity,
            'pnl': position.unrealized_pnl,
            'pnl_pct': position.unrealized_pnl_pct
        })
        
        del self.positions[signal.ticker]
        
        return position
    
    def _calculate_position_size(self, signal: TradeSignal, entry_price: float) -> float:
        """Calculate position size based on risk"""
        
        # Simple position sizing: risk 2% of capital per trade
        risk_pct = 0.02
        base_position = self.initial_capital * risk_pct
        
        # Adjust by confidence
        confidence_multiplier = signal.confidence
        position_value = base_position * confidence_multiplier
        
        return min(position_value, self.cash * 0.5)  # Max 50% of cash per trade
    
    def update_positions(self, current_prices: Dict[str, float]):
        """Update open positions with current prices"""
        
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                position.current_price = current_prices[ticker]
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                position.unrealized_pnl_pct = ((position.current_price - position.entry_price) / position.entry_price) * 100
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        
        total = self.cash
        
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                total += position.quantity * current_prices[ticker]
            else:
                total += position.quantity * position.current_price
        
        return total
    
    def get_positions_summary(self) -> pd.DataFrame:
        """Get summary of open positions"""
        
        if not self.positions:
            return pd.DataFrame()
        
        summary = []
        for ticker, position in self.positions.items():
            summary.append({
                'Ticker': ticker,
                'Quantity': position.quantity,
                'Entry Price': position.entry_price,
                'Current Price': position.current_price,
                'P&L $': position.unrealized_pnl,
                'P&L %': position.unrealized_pnl_pct,
                'Stop Loss': position.stop_loss,
                'Take Profit': position.take_profit
            })
        
        return pd.DataFrame(summary)
    
    def get_performance_metrics(self, final_value: float) -> Dict:
        """Calculate performance metrics"""
        
        if not self.trade_history:
            return {}
        
        df_trades = pd.DataFrame(self.trade_history)
        
        total_return = final_value / self.initial_capital - 1
        total_pnl = df_trades['pnl'].sum()
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        win_rate = winning_trades / len(df_trades) if len(df_trades) > 0 else 0
        
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_pnl': total_pnl,
            'total_trades': len(df_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }

