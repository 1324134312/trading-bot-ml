"""
UK 5-Minute Day Trading Strategy
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from .base_strategy import BaseStrategy, TradeSignal, Position

class DayTradingStrategy(BaseStrategy):
    """Day trading strategy for 5-minute bars"""
    
    def __init__(
        self,
        model_path: str,
        initial_capital: float = 10000,
        min_confidence: float = 0.7,
        stop_loss_pct: float = 0.005,  # 0.5% stop loss
        take_profit_pct: float = 0.01,  # 1% take profit
        max_positions: int = 3  # Max 3 concurrent positions
    ):
        super().__init__(model_path, initial_capital)
        self.min_confidence = min_confidence
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        
        # Load model
        from src.models.predictor import ModelPredictor
        self.predictor = ModelPredictor(model_path)
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """
        Generate trading signals from 5-minute data
        
        Args:
            data: DataFrame with 5-minute bars
            
        Returns:
            List of trading signals
        """
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
                ticker=row.get('Ticker', 'UK_STOCK'),
                action='buy',
                confidence=row['Confidence'],
                entry_price=row['Close'],
                timestamp=str(row.get('Datetime', datetime.now())),
                stop_loss=row['Close'] * (1 - self.stop_loss_pct),
                take_profit=row['Close'] * (1 + self.take_profit_pct),
                position_size=None
            )
            signals.append(signal)
        
        return signals
    
    def should_exit(self, position: Position, current_data: pd.DataFrame) -> bool:
        """
        Determine if should exit a position
        
        Check:
        - Hit stop loss?
        - Hit take profit?
        - Time-based exit (end of day)?
        - Model says sell?
        """
        # Update position with current price
        if len(current_data) > 0:
            current_price = current_data['Close'].iloc[-1]
            position.current_price = current_price
            
            # Check stop loss
            if current_price <= position.stop_loss:
                return True
            
            # Check take profit
            if current_price >= position.take_profit:
                return True
            
            # Check market close (UK market: 16:30)
            current_time = pd.to_datetime(current_data['Datetime'].iloc[-1])
            if current_time.hour >= 16 and current_time.minute >= 30:
                return True
            
            # Check model signal
            predictions = self.predictor.predict_direction(current_data.copy())
            recent_signal = predictions['Prediction'].iloc[-1]
            
            if recent_signal == 0:  # Model says sell
                return True
        
        return False
    
    def filter_signals(self, signals: List[TradeSignal], current_positions: Dict) -> List[TradeSignal]:
        """
        Filter signals based on portfolio constraints
        
        Rules:
        - Max 3 positions at once
        - Don't add to existing position
        - Only highest confidence signals
        """
        # Don't trade if already have maximum positions
        if len(current_positions) >= self.max_positions:
            return []
        
        # Filter out signals for stocks we already hold
        filtered = [s for s in signals if s.ticker not in current_positions]
        
        # Sort by confidence and take top N
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit to available slots
        max_new = self.max_positions - len(current_positions)
        return filtered[:max_new]
    
    def process_bar(self, data: pd.DataFrame, current_positions: Dict) -> List[TradeSignal]:
        """
        Process a single 5-minute bar and generate actionable signals
        
        Args:
            data: Latest bar(s) of data
            current_positions: Current open positions
            
        Returns:
            List of signals to execute
        """
        # Generate all signals
        all_signals = self.generate_signals(data)
        
        # Filter based on portfolio constraints
        actionable_signals = self.filter_signals(all_signals, current_positions)
        
        return actionable_signals

