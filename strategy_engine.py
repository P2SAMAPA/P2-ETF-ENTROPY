"""
Strategy Engine — TUNED v2
Changes vs previous:
  - ALL_NEGATIVE cash trap RE-ADDED with a sensible threshold:
    if ALL ETFs have negative expected return AND the best is below
    -tx_cost, go to CASH. This prevents forced allocation when the
    model has no positive signal (e.g. GLD at -0.0004% selected over
    clearly bad alternatives).
  - expected_return stored as percentage (×100) for readable audit display
  - gain_diff threshold: tx_cost*0.5 (unchanged)
  - Re-entry z-score window: 60 days (unchanged)
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
        """60-day rolling z-score for stable re-entry signal."""
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
        Returns the ETF with the highest pred/price ratio.
        Returns (None, 0.0, {}) only if no data available.
        expected dict values are in % (×100) for readable display.
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
                    # Store as percentage for readable audit trail
                    expected[etf] = float(pred) / float(price) * 100.0

        if not expected:
            return None, 0.0, {}

        best     = max(expected, key=expected.get)
        best_ret = expected[best]

        return best, best_ret, expected

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
                            selected        = "CASH"
                            self.in_cash    = True
                            self.peak_price = None
                            switch_reason   = "TSL_TRIGGERED"

            # ── 2. Rank ETFs ───────────────────────────────────────────────────
            best_etf, best_ret, expected_dict = self.select_best_etf(
                predictions_dict, price_df, date
            )

            # ── 3. No ETF data → stay put ─────────────────────────────────────
            if best_etf is None:
                pass

            # ── 4. ALL returns negative → go to CASH ──────────────────────────
            # FIX: when every ETF has a negative expected return below the cost
            # threshold, the model has no conviction — hold CASH instead of
            # forcing the "least bad" allocation (e.g. GLD at -0.0004%).
            # Threshold: best return must exceed -tx_cost_pct to stay invested.
            elif (not self.in_cash
                  and selected != "CASH"
                  and switch_reason != "TSL_TRIGGERED"
                  and best_ret < -(self.transaction_cost * 100)
                  and all(v < 0 for v in expected_dict.values())):
                selected        = "CASH"
                self.in_cash    = True
                self.peak_price = None
                switch_reason   = "ALL_NEGATIVE"

            # ── 5. Re-entry from CASH (z-score gate) ──────────────────────────
            elif self.in_cash and selected == "CASH":
                hist = predictions_dict[best_etf].loc[:date]
                z    = self.compute_z_score(hist)
                if z > self.z_threshold:
                    selected        = best_etf
                    self.in_cash    = False
                    self.peak_price = (price_df.loc[date, best_etf]
                                       if date in price_df.index else None)
                    switch_reason   = "Z_SCORE_REENTRY"

            # ── 6. Initial entry (no z-score gate) ────────────────────────────
            elif prev_pos is None:
                selected        = best_etf
                self.peak_price = (price_df.loc[date, best_etf]
                                   if date in price_df.index else None)
                switch_reason   = "INITIAL_ENTRY"

            # ── 7. Normal switching ────────────────────────────────────────────
            elif not self.in_cash and best_etf != prev_pos:
                gain_diff = best_ret - expected_dict.get(prev_pos, 0.0)
                if gain_diff > self.transaction_cost * 0.5 * 100:
                    selected        = best_etf
                    self.peak_price = (price_df.loc[date, best_etf]
                                       if date in price_df.index else None)
                    switch_reason   = "BETTER_OPPORTUNITY"

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
                "expected_return": round(best_ret, 6) if best_ret else 0.0,
                "signal_z":        round(z_display, 3) if z_display else None,
                "in_cash":         self.in_cash,
                "switch_reason":   switch_reason if switch_reason else "",
                "switch_flag":     switch_flag,
            })

        return pd.DataFrame(signals).set_index("date")

    def reset(self):
        self.current_position = None
        self.peak_price       = None
        self.in_cash          = False
