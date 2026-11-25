"""
US Swing Trading Strategy for ETFs
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from .base_strategy import BaseStrategy, TradeSignal, Position

class SwingTradingStrategy(BaseStrategy):
    """Swing trading strategy for daily bars (ETFs)"""
    
    def __init__(
        self,
        model_path: str,
        initial_capital: float = 10000,
        min_confidence: float = 0.7,
        stop_loss_pct: float = 0.02,  # 2% stop loss (more for swing)
        take_profit_pct: float = 0.05,  # 5% take profit
        hold_period_days: int = 5  # Target hold period
    ):
        super().__init__(model_path, initial_capital)
        self.min_confidence = min_confidence
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.hold_period_days = hold_period_days
        
        # Load model
        from src.models.predictor import ModelPredictor
        self.predictor = ModelPredictor(model_path)
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """
        Generate signals for swing trading
        
        Args:
            data: DataFrame with daily bars
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Get predictions
        predictions = self.predictor.predict_direction(data.copy())
        
        # Filter by confidence and focus on ETFs
        etf_data = predictions[predictions['Ticker'].isin(['SPY', 'QQQ'])]
        high_confidence = etf_data[
            (etf_data['Prediction'] == 1) & 
            (etf_data['Confidence'] >= self.min_confidence)
        ]
        
        for _, row in high_confidence.iterrows():
            signal = TradeSignal(
                ticker=row.get('Ticker', 'ETF'),
                action='buy',
                confidence=row['Confidence'],
                entry_price=row['Close'],
                timestamp=str(row.get('Date', datetime.now())),
                stop_loss=row['Close'] * (1 - self.stop_loss_pct),
                take_profit=row['Close'] * (1 + self.take_profit_pct),
                position_size=None
            )
            signals.append(signal)
        
        return signals
    
    def should_exit(self, position: Position, current_data: pd.DataFrame) -> bool:
        """
        Determine if should exit swing position
        
        Check:
        - Hit stop loss?
        - Hit take profit?
        - Model says sell?
        - Hold period exceeded?
        """
        if len(current_data) > 0:
            current_price = current_data['Close'].iloc[-1]
            position.current_price = current_price
            
            # Check stop loss
            if current_price <= position.stop_loss:
                return True
            
            # Check take profit
            if current_price >= position.take_profit:
                return True
            
            # Check hold period
            entry_time = pd.to_datetime(position.entry_time)
            if 'Date' in current_data.columns:
                current_time = pd.to_datetime(current_data['Date'].iloc[-1])
                hold_days = (current_time - entry_time).days
                
                if hold_days >= self.hold_period_days:
                    return True
            
            # Check model signal
            predictions = self.predictor.predict_direction(current_data.copy())
            recent_signal = predictions['Prediction'].iloc[-1]
            
            if recent_signal == 0:  # Model says sell
                return True
        
        return False

