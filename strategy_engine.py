"""
Strategy Engine — v3
Key changes vs v2:
  - ALL_NEGATIVE cash trap REMOVED — model always picks best ETF even if
    all expected returns are negative. Eliminates flatline equity curves.
  - Z-score re-entry gate REMOVED — after TSL triggers CASH, re-entry
    happens on the very next day using best available ETF. No more
    multi-week CASH lockout from a single stop-loss event.
  - TSL now sets a 1-day CASH flag only — next iteration re-enters immediately.
  - gain_diff switching threshold lowered: tx_cost * 0.1 (was 0.5) — allows
    more frequent switching to winners, reduces long flatline hold periods.
  - CASH is only held for 1 day after TSL, never as a permanent allocation.
"""

import pandas as pd
import numpy as np


class StrategyEngine:

    def __init__(
        self,
        etf_list,
        tsl_pct=15,
        transaction_cost_bps=25,
        z_score_threshold=1.0,   # kept as param for API compat, no longer used
        cash_etf="3MTBILL",
    ):
        self.etf_list         = etf_list
        self.tsl_pct          = tsl_pct / 100.0
        self.transaction_cost = transaction_cost_bps / 10000.0
        self.z_threshold      = z_score_threshold  # unused, kept for compat
        self.cash_etf         = cash_etf
        self.reset()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def select_best_etf(self, predictions_dict, price_df, date):
        """
        Returns ETF with highest pred/price ratio.
        ALWAYS returns the best ETF — even if all returns are negative.
        No CASH trap. Model conviction, not avoidance.
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
            # TSL triggers 1-day CASH only. Next day re-enters best ETF.
            tsl_fired = False
            if prev_pos not in (None, "CASH"):
                if date in price_df.index and prev_pos in price_df.columns:
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
                            tsl_fired       = True

            # ── 2. Get best ETF ───────────────────────────────────────────────
            best_etf, best_ret, expected_dict = self.select_best_etf(
                predictions_dict, price_df, date)

            if best_etf is None:
                # No data — stay put
                pass

            elif tsl_fired:
                # TSL just fired — hold CASH for exactly this 1 day
                # (next iteration will re-enter via step 3 below)
                pass

            elif self.in_cash:
                # ── 3. Re-entry from CASH — immediate, next day ───────────────
                # No z-score gate. Just pick best ETF and go.
                selected        = best_etf
                self.in_cash    = False
                self.peak_price = (price_df.loc[date, best_etf]
                                   if date in price_df.index
                                   and best_etf in price_df.columns else None)
                switch_reason   = "TSL_REENTRY"

            elif prev_pos is None:
                # ── 4. Initial entry ──────────────────────────────────────────
                selected        = best_etf
                self.peak_price = (price_df.loc[date, best_etf]
                                   if date in price_df.index
                                   and best_etf in price_df.columns else None)
                switch_reason   = "INITIAL_ENTRY"

            elif best_etf != prev_pos:
                # ── 5. Switch — lowered threshold for more responsiveness ─────
                # Was: tx_cost * 0.5 — too conservative, caused long flatlines
                # Now: tx_cost * 0.1 — switches when new ETF beats current by >1bp
                gain_diff = best_ret - expected_dict.get(prev_pos, 0.0)
                if gain_diff > self.transaction_cost * 0.1 * 100:
                    selected        = best_etf
                    self.peak_price = (price_df.loc[date, best_etf]
                                       if date in price_df.index
                                       and best_etf in price_df.columns else None)
                    switch_reason   = "BETTER_OPPORTUNITY"

            # ── Record ────────────────────────────────────────────────────────
            switch_flag           = (selected != prev_pos)
            self.current_position = selected if selected else "CASH"

            signals.append({
                "date":            date,
                "selected_etf":    self.current_position,
                "expected_return": round(best_ret, 6) if best_ret else 0.0,
                "signal_z":        None,   # removed z-score, kept col for compat
                "in_cash":         self.in_cash,
                "switch_reason":   switch_reason if switch_reason else "",
                "switch_flag":     switch_flag,
            })

        return pd.DataFrame(signals).set_index("date")

    def reset(self):
        self.current_position = None
        self.peak_price       = None
        self.in_cash          = False
