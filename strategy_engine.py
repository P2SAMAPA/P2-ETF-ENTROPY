"""
Strategy Engine
Implements:
  1. Cross-sectional ETF selection (best predicted MA_Diff / price)
  2. Peak-based trailing stop loss
  3. Z-score re-entry gate (computed on normalised signal history)
  4. Transaction cost friction (switch only if gain > 2× cost)

Key fix vs original:
  - Z-score was computed on raw MA_Diff predictions (dollar amounts).
    Those are not comparable across ETFs and the 1.0 threshold was
    arbitrary and meaningless.
  - Now Z-score is computed on the ETF's own normalised expected-return
    signal history (pred / price), which is dimensionless and comparable
    to the user-set threshold.
"""

import pandas as pd
import numpy as np


class StrategyEngine:

    def __init__(
        self,
        etf_list: list,
        tsl_pct: float = 15.0,
        transaction_cost_bps: float = 25.0,
        z_score_threshold: float = 1.0,
        z_score_window: int = 20,
        cash_etf: str = "3MTBILL",
    ):
        self.etf_list         = etf_list
        self.tsl_pct          = tsl_pct / 100.0
        self.transaction_cost = transaction_cost_bps / 10_000.0
        self.z_threshold      = z_score_threshold
        self.z_window         = z_score_window
        self.cash_etf         = cash_etf
        self.reset()

    # ──────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────

    def _compute_z_score(self, series: pd.Series) -> float:
        """
        Z-score of the most recent value against a rolling window.
        Series should contain NORMALISED expected-return values
        (pred / price), not raw MA_Diff dollar amounts.
        """
        n = len(series)
        if n < 5:
            return 0.0
        w    = min(self.z_window, n)
        tail = series.iloc[-w:]
        mu   = tail.mean()
        sig  = tail.std()
        if pd.isna(sig) or sig < 1e-10:
            return 0.0
        return float((series.iloc[-1] - mu) / sig)

    def _expected_returns(
        self,
        predictions_dict: dict,
        price_df: pd.DataFrame,
        date,
    ) -> dict:
        """
        Compute expected normalised return for each ETF on a given date.
        expected_return[etf] = MA_Diff_pred / current_price  (dimensionless)
        """
        expected = {}
        for etf in self.etf_list:
            if (
                etf in predictions_dict
                and date in predictions_dict[etf].index
                and date in price_df.index
                and etf in price_df.columns
            ):
                pred  = predictions_dict[etf].loc[date]
                price = price_df.loc[date, etf]
                if pd.notna(pred) and pd.notna(price) and price > 0:
                    expected[etf] = float(pred) / float(price)
        return expected

    # ──────────────────────────────────────────────────────────────────
    # Signal generation
    # ──────────────────────────────────────────────────────────────────

    def generate_signals(
        self,
        predictions_dict: dict,
        price_df: pd.DataFrame,
        dates,
    ) -> pd.DataFrame:
        """
        Walk forward through dates and generate position signals.

        Parameters
        ----------
        predictions_dict : {etf: pd.Series of MA_Diff predictions}
        price_df         : DataFrame of ETF prices
        dates            : ordered sequence of trading dates

        Returns
        -------
        pd.DataFrame indexed by date with columns:
            selected_etf, expected_return, signal_z, in_cash,
            switch_reason, switch_flag
        """
        # Pre-compute expected return series for Z-score history
        # {etf: list of (date, norm_return)} built incrementally
        norm_ret_history: dict = {etf: pd.Series(dtype=float) for etf in self.etf_list}

        signals = []

        for date in dates:
            prev_position = self.current_position
            selected      = prev_position
            switch_reason = None

            # ── 0. Build expected returns for today ───────────────────
            exp_ret = self._expected_returns(predictions_dict, price_df, date)

            # Update running history of normalised returns (for Z-score)
            for etf, er in exp_ret.items():
                norm_ret_history[etf] = pd.concat(
                    [norm_ret_history[etf], pd.Series([er], index=[date])]
                )

            all_negative = len(exp_ret) > 0 and all(v <= 0 for v in exp_ret.values())
            best_etf     = max(exp_ret, key=exp_ret.get) if exp_ret else None
            best_ret     = exp_ret.get(best_etf, 0.0)

            # ── 1. Trailing stop loss ─────────────────────────────────
            if prev_position not in (None, "CASH"):
                if date in price_df.index and prev_position in price_df.columns:
                    price_today = price_df.loc[date, prev_position]
                    if pd.notna(price_today):
                        if self.peak_price is None:
                            self.peak_price = price_today
                        self.peak_price = max(self.peak_price, price_today)
                        drawdown = (price_today / self.peak_price) - 1.0
                        if drawdown < -self.tsl_pct:
                            selected      = "CASH"
                            self.in_cash  = True
                            self.peak_price = None
                            switch_reason = "TSL_TRIGGERED"

            # ── 2. All signals negative → go to cash ──────────────────
            if all_negative and selected != "CASH":
                selected      = "CASH"
                self.in_cash  = True
                switch_reason = "ALL_NEGATIVE"

            # ── 3. Re-entry from cash via Z-score gate ────────────────
            elif self.in_cash and best_etf is not None and not all_negative:
                # Z-score on NORMALISED expected-return history
                z = self._compute_z_score(norm_ret_history[best_etf])
                if z > self.z_threshold:
                    selected           = best_etf
                    self.in_cash       = False
                    self.peak_price    = (
                        price_df.loc[date, best_etf]
                        if (date in price_df.index and best_etf in price_df.columns)
                        else None
                    )
                    switch_reason = f"Z_REENTRY(z={z:.2f})"

            # ── 4. Normal switching ───────────────────────────────────
            elif not self.in_cash and best_etf is not None:
                if prev_position is None:
                    selected        = best_etf
                    self.peak_price = (
                        price_df.loc[date, best_etf]
                        if (date in price_df.index and best_etf in price_df.columns)
                        else None
                    )
                    switch_reason = "INITIAL_ENTRY"

                elif best_etf != prev_position:
                    gain_diff = best_ret - exp_ret.get(prev_position, 0.0)
                    # Switch only if incremental gain exceeds round-trip cost
                    if gain_diff > self.transaction_cost * 2:
                        selected        = best_etf
                        self.peak_price = (
                            price_df.loc[date, best_etf]
                            if (date in price_df.index and best_etf in price_df.columns)
                            else None
                        )
                        switch_reason = "BETTER_OPPORTUNITY"

            # ── Record ────────────────────────────────────────────────
            switch_flag           = (selected != prev_position)
            self.current_position = selected

            # Z-score of the selected ETF for audit trail
            sig_z = (
                self._compute_z_score(norm_ret_history[selected])
                if selected not in (None, "CASH") and selected in norm_ret_history
                else None
            )

            signals.append({
                "date":            date,
                "selected_etf":    selected if selected else "CASH",
                "expected_return": best_ret,
                "signal_z":        sig_z,
                "in_cash":         self.in_cash,
                "switch_reason":   switch_reason,
                "switch_flag":     switch_flag,
            })

        return pd.DataFrame(signals).set_index("date")

    # ──────────────────────────────────────────────────────────────────

    def reset(self):
        self.current_position = None
        self.peak_price       = None
        self.in_cash          = False
