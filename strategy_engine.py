"""
Strategy Engine
Implements:

1. Cross-sectional ETF selection (highest expected return)
2. Proper peak-based Trailing Stop Loss (TSL)
3. Z-score re-entry logic
4. Transaction cost friction
"""

import pandas as pd
import numpy as np


class StrategyEngine:

    def __init__(
        self,
        etf_list,
        tsl_pct=15,                     # Slider passes 10–25
        transaction_cost_bps=25,
        z_score_threshold=1.0,
        cash_etf="3MTBILL"
    ):
        """
        tsl_pct: percentage value (15 means 15%)
        """

        self.etf_list = etf_list
        self.tsl_pct = tsl_pct / 100.0              # Convert once here
        self.transaction_cost = transaction_cost_bps / 10000.0
        self.z_threshold = z_score_threshold
        self.cash_etf = cash_etf

        self.reset()

    # ----------------------------------------------------
    # Utility Functions
    # ----------------------------------------------------

    def compute_z_score(self, series, window=20):
        """
        Proper rolling OOS Z-score
        """
        if len(series) < 5:
            return 0.0

        rolling_mean = series.rolling(window=min(window, len(series))).mean().iloc[-1]
        rolling_std = series.rolling(window=min(window, len(series))).std().iloc[-1]

        if pd.isna(rolling_std) or rolling_std == 0:
            return 0.0

        return (series.iloc[-1] - rolling_mean) / rolling_std

    def select_best_etf(self, predictions_dict, price_df, date):
        """
        Rank ETFs by expected return = predicted_diff / price
        """
        expected = {}

        for etf in self.etf_list:
            if etf in predictions_dict and date in predictions_dict[etf].index:

                pred = predictions_dict[etf].loc[date]
                price = price_df.loc[date, etf]

                if pd.notna(pred) and pd.notna(price) and price > 0:
                    expected[etf] = pred / price

        if not expected:
            return None, 0.0, {}

        best = max(expected, key=expected.get)
        return best, expected[best], expected

    # ----------------------------------------------------
    # Core Engine
    # ----------------------------------------------------

    def generate_signals(self, predictions_dict, price_df, dates):

        signals = []

        for i, date in enumerate(dates):

            switch_reason = None
            selected = self.current_position

            # --------------------------------------------
            # 1. Apply Trailing Stop (peak based)
            # --------------------------------------------
            if self.current_position not in [None, "CASH"]:

                price_today = price_df.loc[date, self.current_position]

                # Update peak
                if self.peak_price is None:
                    self.peak_price = price_today

                self.peak_price = max(self.peak_price, price_today)

                drawdown = (price_today / self.peak_price) - 1

                if drawdown < -self.tsl_pct:
                    selected = "CASH"
                    self.in_cash = True
                    self.peak_price = None
                    switch_reason = "TSL_TRIGGERED"

            # --------------------------------------------
            # 2. Rank ETFs
            # --------------------------------------------
            best_etf, best_ret, expected_dict = self.select_best_etf(
                predictions_dict,
                price_df,
                date
            )

            all_negative = (
                len(expected_dict) > 0 and
                all(v <= 0 for v in expected_dict.values())
            )

            # --------------------------------------------
            # 3. All negative → go to cash
            # --------------------------------------------
            if all_negative:
                if self.current_position != "CASH":
                    selected = "CASH"
                    self.in_cash = True
                    switch_reason = "ALL_NEGATIVE"

            # --------------------------------------------
            # 4. Z-score re-entry (from cash)
            # --------------------------------------------
            elif self.in_cash and best_etf is not None:

                hist = predictions_dict[best_etf].loc[:date]
                z = self.compute_z_score(hist)

                if z > self.z_threshold:
                    selected = best_etf
                    self.in_cash = False
                    self.peak_price = price_df.loc[date, best_etf]
                    switch_reason = "Z_SCORE_REENTRY"

            # --------------------------------------------
            # 5. Normal switching logic
            # --------------------------------------------
            elif not self.in_cash and best_etf is not None:

                if self.current_position is None:
                    selected = best_etf
                    self.peak_price = price_df.loc[date, best_etf]
                    switch_reason = "INITIAL_ENTRY"

                elif best_etf != self.current_position:

                    gain_diff = best_ret - expected_dict.get(self.current_position, 0)

                    if gain_diff > (self.transaction_cost * 2):
                        selected = best_etf
                        self.peak_price = price_df.loc[date, best_etf]
                        switch_reason = "BETTER_OPPORTUNITY"

            # --------------------------------------------
            # Detect Switch
            # --------------------------------------------
            switch_flag = selected != self.current_position

            self.current_position = selected

            signals.append({
                "date": date,
                "selected_etf": selected if selected else "CASH",
                "expected_return": best_ret,
                "z_score": None if best_etf is None else (
                    self.compute_z_score(
                        predictions_dict[best_etf].loc[:date]
                    )
                ),
                "in_cash": self.in_cash,
                "switch_reason": switch_reason,
                "switch_flag": switch_flag
            })

        return pd.DataFrame(signals).set_index("date")

    # ----------------------------------------------------

    def reset(self):
        self.current_position = None
        self.peak_price = None
        self.in_cash = False
