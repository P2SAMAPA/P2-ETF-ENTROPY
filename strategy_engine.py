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
        if len(predicted_ma_diff) < history_window:
            # Not enough history, use simple standardization
            mean = predicted_ma_diff.mean()
            std = predicted_ma_diff.std()
        else:
            # Rolling window (OOS only, no lookahead)
            mean = predicted_ma_diff.rolling(window=history_window).mean().iloc[-1]
            std = predicted_ma_diff.rolling(window=history_window).std().iloc[-1]
        
        if std == 0 or pd.isna(std):
            return 0
        
        latest_pred = predicted_ma_diff.iloc[-1]
        z_score = (latest_pred - mean) / std
        
        return z_score
    
    def select_best_etf(self, predictions, prices, date):
        """
        Select ETF with highest predicted expected return
        Expected Return ≈ Predicted_MA_Diff / Current_Price
        """
        expected_returns = {}
        
        for etf in self.etf_list:
            if etf in predictions and etf in prices.columns:
                pred_diff = predictions[etf]
                if isinstance(pred_diff, pd.Series):
                    pred_diff = pred_diff.loc[date] if date in pred_diff.index else pred_diff.iloc[-1]
                
                price = prices.loc[date, etf] if date in prices.index else prices[etf].iloc[-1]
                
                if pd.notna(pred_diff) and pd.notna(price) and price > 0:
                    expected_ret = pred_diff / price
                    expected_returns[etf] = expected_ret
        
        if not expected_returns:
            return None, 0
        
        # Select highest expected return
        best_etf = max(expected_returns, key=expected_returns.get)
        best_return = expected_returns[best_etf]
        
        return best_etf, best_return
    
    def check_tsl_trigger(self, returns_2day):
        """
        Check if 2-day cumulative return triggers TSL
        """
        if returns_2day < -self.tsl_pct:
            return True
        return False
    
    def should_switch(self, current_etf, new_etf, expected_gain):
        """
        Apply transaction cost friction
        Only switch if expected gain > transaction cost
        """
        if current_etf == new_etf:
            return False
        
        # Minimum gain to justify switch (2x cost for round-trip consideration)
        min_gain = self.transaction_cost * 2
        
        if expected_gain > min_gain:
            return True
        
        return False
    
    def generate_signals(self, predictions_dict, price_df, dates):
        """
        Generate trading signals for entire period
        
        Parameters:
        -----------
        predictions_dict : dict
            {etf: pd.Series of predictions} for each ETF
        price_df : pd.DataFrame
            Price data including all ETFs
        dates : pd.DatetimeIndex
            Trading dates to process
            
        Returns:
        --------
        signals : pd.DataFrame
            Columns: ['selected_etf', 'expected_return', 'z_score', 'in_cash', 'switch_reason']
        """
        signals = []
        
        for i, date in enumerate(dates):
            signal = {
                'date': date,
                'selected_etf': None,
                'expected_return': 0,
                'z_score': None,
                'in_cash': False,
                'switch_reason': None
            }
            
            # Get predictions for all ETFs at this date
            current_preds = {}
            pred_histories = {}
            
            for etf in self.etf_list:
                if etf in predictions_dict:
                    pred_series = predictions_dict[etf]
                    if isinstance(pred_series, pd.Series) and date in pred_series.index:
                        current_preds[etf] = pred_series.loc[date]
                        # Get history up to this point for Z-score
                        pred_histories[etf] = pred_series.loc[:date]
            
            # Step 1: Check if we need to apply TSL (if in position)
            if self.current_position is not None and self.current_position != 'CASH':
                # Calculate 2-day return if we have history
                if i >= 2:
                    price_today = price_df.loc[date, self.current_position]
                    price_2days_ago = price_df.iloc[i-2][self.current_position]
                    
                    if pd.notna(price_today) and pd.notna(price_2days_ago):
                        ret_2day = (price_today / price_2days_ago) - 1
                        
                        if self.check_tsl_trigger(ret_2day):
                            self.in_cash = True
                            self.cash_start_date = date
                            self.current_position = 'CASH'
                            signal['in_cash'] = True
                            signal['switch_reason'] = 'TSL_TRIGGERED'
            
            # Step 2: If in CASH, check Z-score re-entry condition
            if self.in_cash:
                # Check all ETFs for Z-score > threshold
                eligible_etfs = []
                
                for etf, pred_hist in pred_histories.items():
                    if len(pred_hist) >= 2:
                        z = self.compute_z_score(pred_hist)
                        if z > self.z_threshold:
                            eligible_etfs.append((etf, current_preds.get(etf, 0), z))
                
                if eligible_etfs:
                    # Select highest expected return among eligible
                    best = max(eligible_etfs, key=lambda x: x[1])
                    selected_etf = best[0]
                    expected_ret = best[1] / price_df.loc[date, selected_etf] if price_df.loc[date, selected_etf] > 0 else 0
                    
                    if self.should_switch('CASH', selected_etf, expected_ret):
                        self.in_cash = False
                        self.cash_start_date = None
                        self.current_position = selected_etf
                        self.entry_price = price_df.loc[date, selected_etf]
                        signal['switch_reason'] = 'Z_SCORE_REENTRY'
                else:
                    # Stay in cash
                    signal['in_cash'] = True
                    signal['selected_etf'] = 'CASH'
            
            # Step 3: Normal selection (if not in cash or just re-entered)
            if not self.in_cash and self.current_position != 'CASH':
                # Select best ETF by expected return
                best_etf, best_return = self.select_best_etf(current_preds, price_df, date)
                
                if best_etf is not None:
                    # Check if all predictions are negative -> go to cash
                    all_negative = all(p < 0 for p in current_preds.values() if pd.notna(p))
                    
                    if all_negative:
                        self.in_cash = True
                        self.current_position = 'CASH'
                        signal['in_cash'] = True
                        signal['switch_reason'] = 'ALL_NEGATIVE'
                    elif best_return > 0:
                        # Consider switching if better opportunity
                        if self.should_switch(self.current_position, best_etf, 
                                            best_return - (0 if self.current_position is None else 0)):
                            if self.current_position != best_etf:
                                signal['switch_reason'] = 'BETTER_OPPORTUNITY'
                            self.current_position = best_etf
                            self.entry_price = price_df.loc[date, best_etf]
                    
                    signal['selected_etf'] = self.current_position
                    signal['expected_return'] = best_return
            
            # Set final signal
            signal['selected_etf'] = self.current_position if self.current_position else 'CASH'
            
            # Calculate Z-score for selected ETF if available
            if self.current_position in pred_histories:
                signal['z_score'] = self.compute_z_score(pred_histories[self.current_position])
            
            signals.append(signal)
        
        return pd.DataFrame(signals).set_index('date')
    
    def reset(self):
        """Reset strategy state"""
        self.current_position = None
        self.entry_price = None
        self.peak_price = None
        self.in_cash = False
        self.cash_start_date = None
