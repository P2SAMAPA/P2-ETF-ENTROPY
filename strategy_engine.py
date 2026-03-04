"""
Strategy Engine
Implements the full trading logic:
1. Cross-sectional ETF selection (highest expected return)
2. Trailing Stop Loss (TSL)
3. Z-score re-entry logic
4. Transaction cost friction
"""

import pandas as pd
import numpy as np
from datetime import timedelta


class StrategyEngine:
    """
    Implements the full trading strategy with TSL and Z-score controls
    """
    
    def __init__(self, etf_list, tsl_pct=0.15, transaction_cost_bps=25, 
                 z_score_threshold=1.0, cash_etf='3MTBILL'):
        """
        Parameters:
        -----------
        etf_list : list
            List of 7 ETFs to trade
        tsl_pct : float
            Trailing stop loss percentage (0.10 to 0.25)
        transaction_cost_bps : int
            Transaction cost in basis points (10 to 75)
        z_score_threshold : float
            Z-score threshold for re-entry (default 1.0)
        cash_etf : str
            Column name for risk-free rate/cash
        """
        self.etf_list = etf_list
        self.tsl_pct = tsl_pct / 100  # Convert to decimal
        self.transaction_cost = transaction_cost_bps / 10000  # Convert bps to decimal
        self.z_threshold = z_score_threshold
        self.cash_etf = cash_etf
        
        # State variables
        self.current_position = None  # Current ETF held or 'CASH'
        self.entry_price = None
        self.peak_price = None
        self.in_cash = False
        self.cash_start_date = None
        
    def compute_z_score(self, predicted_ma_diff, history_window=20):
        """
        Compute Z-score of predicted MA_Diff
        Z = (Predicted - Rolling Mean) / Rolling Std
        """
        if len(predicted
