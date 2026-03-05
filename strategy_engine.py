"""
Strategy Engine — TUNED
Changes vs original:
  - ALL_NEGATIVE cash trap REMOVED: always invest in best-ranked ETF.
    CASH only triggered by TSL or z-score re-entry filter.
  - gain_diff switching threshold: tx_cost*2 → tx_cost*0.5
    (old threshold was 3-50× larger than typical pred/price differences)
  - Re-entry z-score window: 20→60 days (more stable statistics)
  - Initial entry: no z-score gate (first trade enters freely)
  - Position switch: rank-change triggers switch if gain_diff > tx_cost*0.5
"""

import pandas as pd
import numpy as np


class StrategyEngine:

    def __init__(
        self,
        etf_list,
        tsl_pct=15,
        transaction_cost_bps=25,
        z_score_threshold=1.0,
        cash_etf="3MTBILL",
    ):
        self.etf_list         = etf_list
        self.tsl_pct          = tsl_pct / 100.0
        self.transaction_cost = transaction_cost_bps / 10000.0
        self.z_threshold      = z_score_threshold
        self.cash_etf         = cash_etf
        self.reset()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def compute_z_score(self, series, window=60):
        """60-day rolling z-score (was 20) for more stable re-entry signal."""
        if len(series) < 5:
            return 0.0
        w    = min(window, len(series))
        mean = series.rolling(w).mean().iloc[-1]
        std  = series.rolling(w).std().iloc[-1]
        if pd.isna(std) or std < 1e-10:
            return 0.0
        return float((series.iloc[-1] - mean) / std)

    def select_best_etf(self, predictions_dict, price_df, date):
        """
        Always returns the ETF with the highest pred/price ratio.
        Never returns None — there is always a best relative choice.
        """
        expected = {}
        for etf in self.etf_list:
            if (etf in predictions_dict
                    and date in predictions_dict[etf].index
                    and date in price_df.index
                    and etf in price_df.columns):
                pred  = predictions_dict[etf].loc[date]
                price = price_df.loc[date, etf]
                if pd.notna(pred) and pd.notna(price) and price > 0:
                    expected[etf] = float(pred) / float(price)

        if not expected:
            return None, 0.0, {}

        best = max(expected, key=expected.get)
        return best, expected[best], expected

    # ── Core signal generation ─────────────────────────────────────────────────

    def generate_signals(self, predictions_dict, price_df, dates):

        signals = []

        for date in dates:

            prev_pos      = self.current_position
            selected      = prev_pos
            switch_reason = None

            # ── 1. Trailing Stop Loss ─────────────────────────────────────────
            if prev_pos not in (None, "CASH"):
                if date in price_df.index:
                    price_today = price_df.loc[date, prev_pos]
                    if pd.notna(price_today):
                        if self.peak_price is None:
                            self.peak_price = price_today
                        self.peak_price = max(self.peak_price, price_today)
                        drawdown = (price_today / self.peak_price) - 1
                        if drawdown < -self.tsl_pct:
                            selected      = "CASH"
                            self.in_cash  = True
                            self.peak_price = None
                            switch_reason = "TSL_TRIGGERED"

            # ── 2. Rank ETFs — always pick best ───────────────────────────────
            best_etf, best_ret, expected_dict = self.select_best_etf(
                predictions_dict, price_df, date
            )

            # ── 3. No ETF data at all → stay put ──────────────────────────────
            if best_etf is None:
                pass  # selected unchanged

            # ── 4. Re-entry from TSL cash (z-score gate) ──────────────────────
            elif self.in_cash and selected == "CASH":
                hist = predictions_dict[best_etf].loc[:date]
                z    = self.compute_z_score(hist)
                if z > self.z_threshold:
                    selected      = best_etf
                    self.in_cash  = False
                    self.peak_price = (price_df.loc[date, best_etf]
                                       if date in price_df.index else None)
                    switch_reason = "Z_SCORE_REENTRY"
                # else: remain in CASH until z-score threshold met

            # ── 5. Initial entry (no z-score gate on first trade) ─────────────
            elif prev_pos is None:
                selected      = best_etf
                self.peak_price = (price_df.loc[date, best_etf]
                                   if date in price_df.index else None)
                switch_reason = "INITIAL_ENTRY"

            # ── 6. Normal switching — lowered threshold ────────────────────────
            elif not self.in_cash and best_etf != prev_pos:
                gain_diff = best_ret - expected_dict.get(prev_pos, 0.0)
                # Threshold: tx_cost * 0.5 (was tx_cost * 2)
                if gain_diff > self.transaction_cost * 0.5:
                    selected      = best_etf
                    self.peak_price = (price_df.loc[date, best_etf]
                                       if date in price_df.index else None)
                    switch_reason = "BETTER_OPPORTUNITY"

            # ── Record ────────────────────────────────────────────────────────
            switch_flag           = (selected != prev_pos)
            self.current_position = selected if selected else "CASH"

            z_display = None
            if best_etf and best_etf in predictions_dict:
                z_display = self.compute_z_score(
                    predictions_dict[best_etf].loc[:date]
                )

            signals.append({
                "date":            date,
                "selected_etf":    self.current_position,
                "expected_return": best_ret,
                "signal_z":        z_display,
                "in_cash":         self.in_cash,
                "switch_reason":   switch_reason,
                "switch_flag":     switch_flag,
            })

        return pd.DataFrame(signals).set_index("date")

    def reset(self):
        self.current_position = None
        self.peak_price       = None
        self.in_cash          = False
