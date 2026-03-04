"""
Streamlit App - ETF Transfer Voting Strategy
Main UI for the Entropy Paper-based ETF Allocation Engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download, HfApi
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import our modules
from data_loader import HF_DATASET_REPO, ETF_LIST, load_metadata, seed_dataset, load_dataset
from update_data import incremental_update
from utils import get_latest_trading_day, get_nyse_trading_days
from transfer_voting import TransferVotingModel
from strategy_engine import StrategyEngine
from backtest import run_backtest
from metrics import calculate_metrics, format_metrics_for_display
from feature_engineering import prepare_all_features

# Page config
st.set_page_config(
    page_title="ETF Transfer Voting Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        border: 2px solid #1f77b4;
    }
    .etf-badge {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .methodology {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Load trained model and artifacts from HF"""
    try:
        os.makedirs("artifacts", exist_ok=True)
        
        meta_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="models/best_model.json",
            repo_type="dataset",
            local_dir="artifacts"
        )
        
        with open(meta_path, 'r') as f:
            best_info = json.load(f)
        
        best_ma = best_info['best_ma_window']
        
        model_file = f"transfer_voting_MA{best_ma}.pkl"
        model_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=f"models/{model_file}",
            repo_type="dataset",
            local_dir="artifacts"
        )
        
        model = TransferVotingModel([], best_ma, "artifacts")
        model.load(model_path)
        
        return model, best_info
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_data
def load_data():
    """Load dataset from HF"""
    try:
        df = load_dataset()
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


@st.cache_data
def run_full_backtest(model, df, best_ma, tsl_pct, tx_cost, z_threshold):
    """
    Run full backtest with selected parameters
    Returns backtest results and metrics
    """
    etf_list = ['TLT', 'VNQ', 'GLD', 'SLV', 'VCIT', 'HYG', 'LQD']
    
    # Prepare features for best MA window
    data_dict = prepare_all_features(df, ma_window=best_ma)
    features_dict = data_dict['features']
    targets_dict = data_dict['targets']
    
    # Find common dates
    common_dates = None
    for etf in etf_list:
        if etf in features_dict:
            dates = features_dict[etf].index
            if common_dates is None:
                common_dates = dates
            else:
                common_dates = common_dates.intersection(dates)
    
    # Split: 80% train, 10% val, 10% test (OOS)
    n = len(common_dates)
    test_start = int(n * 0.9)
    
    test_dates = common_dates[test_start:]
    
    # Generate predictions for test period
    predictions_dict = {}
    for etf in etf_list:
        if etf in features_dict:
            X_test = features_dict[etf].loc[test_dates]
            preds = model.predict_single_etf(X_test, etf)
            predictions_dict[etf] = pd.Series(preds, index=test_dates)
    
    # Run strategy backtest
    strategy_engine = StrategyEngine(
        etf_list=etf_list,
        tsl_pct=tsl_pct,
        transaction_cost_bps=tx_cost,
        z_score_threshold=z_threshold
    )
    
    results = run_backtest(
        predictions_dict=predictions_dict,
        price_df=df[etf_list],
        tbill_df=df['3MTBILL'],
        strategy_engine=strategy_engine,
        test_dates=test_dates
    )
    
    # Calculate metrics
    metrics = calculate_metrics(
        equity_curve=results['equity_curve']['strategy'],
        returns=results['returns'],
        risk_free=results['risk_free'],
        audit_trail=results['audit_trail']
    )
    
    return results, metrics


def check_data_freshness():
    """Check if dataset is up to date"""
    try:
        metadata = load_metadata()
        if metadata is None:
            return False, "No dataset found", None
        
        last_update = pd.to_datetime(metadata['last_data_update'])
        latest_trading = pd.to_datetime(get_latest_trading_day())
        
        if last_update >= latest_trading:
            return True, f"Dataset already updated till {last_update.date()}", last_update.date()
        else:
            return False, f"Dataset stale: {last_update.date()} (Latest: {latest_trading.date()})", last_update.date()
    except Exception as e:
        return False, f"Error checking freshness: {e}", None


def generate_next_day_prediction(model, df, best_ma):
    """Generate prediction for next trading day"""
    etf_list = ['TLT', 'VNQ', 'GLD', 'SLV', 'VCIT', 'HYG', 'LQD']
    
    # Get latest data
    latest_date = df.index[-1]
    
    # Get next trading day
    nyse_days = get_nyse_trading_days(start_date=latest_date.strftime('%Y-%m-%d'))
    future_days = nyse_days[nyse_days > latest_date]
    
    if len(future_days) == 0:
        return None, None, "No future trading days available"
    
    next_trading_day = future_days[0]
    
    # Prepare features for prediction
    data_dict = prepare_all_features(df, ma_window=best_ma)
    features_dict = data_dict['features']
    
    # Generate predictions for all ETFs
    predictions = {}
    for etf in etf_list:
        if etf in features_dict:
            # Get latest features
            X_latest = features_dict[etf].iloc[-1:]
            pred = model.predict_single_etf(X_latest, etf)
            predictions[etf] = pred[0]
    
    # Select best ETF by expected return
    expected_returns = {}
    for etf in etf_list:
        if etf in predictions:
            price = df.loc[latest_date, etf]
            expected_returns[etf] = predictions[etf] / price if price > 0 else 0
    
    best_etf = max(expected_returns, key=expected_returns.get)
    best_return = expected_returns[best_etf]
    
    return next_trading_day, best_etf, best_return


def plot_equity_curves(equity_df):
    """Plot strategy vs benchmarks"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(equity_df.index, equity_df['strategy'], label='Strategy', linewidth=2, color='#1f77b4')
    
    if 'SPY' in equity_df.columns:
        ax.plot(equity_df.index, equity_df['SPY'], label='SPY', linewidth=1.5, alpha=0.7, color='gray')
    if 'AGG' in equity_df.columns:
        ax.plot(equity_df.index, equity_df['AGG'], label='AGG', linewidth=1.5, alpha=0.7, color='orange')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Equity')
    ax.set_title('Strategy Performance vs Benchmarks (OOS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


def main():
    st.markdown('<div class="main-header">📈 ETF Transfer Voting Engine</div>', unsafe_allow_html=True)
    st.markdown("*Implementation of Entropy Paper: Flexible Target Prediction with Transfer Learning*")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Strategy Parameters")
    
    year_start = st.sidebar.slider(
        "Start Year",
        min_value=2008,
        max_value=2025,
        value=2008,
        step=1
    )
    
    tsl_pct = st.sidebar.slider(
        "Trailing Stop Loss (%)",
        min_value=10,
        max_value=25,
        value=15,
        step=1
    ) / 100
    
    tx_cost = st.sidebar.slider(
        "Transaction Cost (bps)",
        min_value=10,
        max_value=75,
        value=25,
        step=5
    )
    
    z_threshold = st.sidebar.slider(
        "Z-Score Re-entry Threshold",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔄 Data Management")
    
    is_fresh, freshness_msg, last_date = check_data_freshness()
    st.sidebar.info(f"**Dataset Status:**\n{freshness_msg}")
    
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        with st.spinner("Checking and updating data..."):
            if is_fresh:
                st.sidebar.success(f"✅ {freshness_msg}")
            else:
                try:
                    incremental_update()
                    st.sidebar.success("✅ Data refreshed successfully!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"❌ Error refreshing data: {e}")
    
    # Load data and model
    df = load_data()
    model, best_info = load_model_artifacts()
    
    if df is None:
        st.warning("⚠️ No dataset found. Please seed the dataset first.")
        if st.button("Initialize Dataset (2008-Present)"):
            with st.spinner("Downloading historical data..."):
                seed_dataset()
                st.success("Dataset initialized!")
                st.rerun()
        return
    
    # Run backtest if model loaded
    results = None
    metrics = None
    
    if model is not None:
        with st.spinner("Running backtest... This may take a minute."):
            try:
                results, metrics = run_full_backtest(
                    model, df, best_info['best_ma_window'], 
                    tsl_pct, tx_cost, z_threshold
                )
            except Exception as e:
                st.error(f"Backtest error: {e}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📋 Audit Trail", "📖 Methodology"])
    
    with tab1:
        st.subheader("Next Trading Day Prediction")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if model is not None:
                next_day, pred_etf, exp_return = generate_next_day_prediction(
                    model, df, best_info.get('best_ma_window', 3)
                )
                
                if next_day is not None:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style="font-size: 1.2rem; color: #666; margin-bottom: 0.5rem;">
                            Next Trading Day: {next_day.strftime('%Y-%m-%d')}
                        </div>
                        <div class="etf-badge">
                            {pred_etf}
                        </div>
                        <div style="font-size: 1rem; color: #666; margin-top: 0.5rem;">
                            Model: Transfer Voting – MA({best_info.get('best_ma_window', 'N/A')})
                        </div>
                        <div style="font-size: 0.9rem; color: #888; margin-top: 0.3rem;">
                            Expected Return: {exp_return:.2%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Unable to generate prediction")
            else:
                st.info("Model not loaded. Run training first via GitHub Actions.")
        
        with col2:
            st.metric("Latest Data", last_date.strftime('%Y-%m-%d') if last_date else "N/A")
            st.metric("Active MA Window", f"MA({best_info.get('best_ma_window', 'N/A')})" if best_info else "N/A")
        
        with col3:
            st.metric("ETFs Tracked", 7)
            st.metric("Strategy Status", "Active" if model else "Training Needed")
        
        st.markdown("---")
        st.subheader("Performance Metrics (OOS Period)")
        
        if metrics is not None:
            metrics_cols = st.columns(5)
            display_metrics = {
                'Annualized Return': f"{metrics['annualized_return']:.2%}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
                'Worst Daily DD': f"{metrics['worst_daily_return']:.2%}",
                'Hit Ratio (15d)': f"{metrics['hit_ratio_15d']:.1%}" if metrics['hit_ratio_15d'] else "N/A"
            }
            
            for i, (metric, value) in enumerate(display_metrics.items()):
                with metrics_cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; color: #666;">{metric}</div>
                        <div style="font-size: 1.5rem; font-weight: bold;">{value}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Run backtest to see metrics")
        
        st.markdown("---")
        st.subheader("Equity Curve")
        
        if results is not None:
            fig = plot_equity_curves(results['equity_curve'])
            st.pyplot(fig)
            
            # Max DD details
            dd_col1, dd_col2 = st.columns(2)
            with dd_col1:
                st.metric("Max Drawdown (Peak to Trough)", f"{metrics['max_drawdown']:.2%}")
            with dd_col2:
                st.metric("Max DD Date", str(metrics['max_drawdown_date'])[:10])
        else:
            st.info("Run backtest to see equity curve")
    
    with tab2:
        st.subheader("Last 15 Trading Days Audit Trail")
        
        if results is not None and 'audit_trail' in results:
            audit_df = results['audit_trail'].tail(15)[['date', 'predicted_etf', 'actual_return', 'strategy_return', 'in_cash']]
            audit_df.columns = ['Date', 'Predicted ETF', 'Actual Return', 'Strategy Return', 'In Cash']
            st.dataframe(audit_df, use_container_width=True)
        else:
            st.info("Run backtest to see audit trail")
    
    with tab3:
        st.subheader("Methodology")
        
        st.markdown("""
        <div class="methodology">
        
        <h4>📄 Reference Paper</h4>
        <p><strong>Flexible Target Prediction for Quantitative Trading in the American Stock Market: 
        A Hybrid Framework Integrating Ensemble Models, Fusion Models and Transfer Learning</strong><br>
        <em>Journal: Entropy (2026)</em></p>
        
        <h4>🎯 Core Innovation</h4>
        <p>Instead of predicting raw closing prices (high entropy), this engine predicts 
        <strong>Moving Average Differences (MA_Diff)</strong> which smooths noise and improves 
        predictability.</p>
        
        <h4>🧠 Model Architecture</h4>
        <ol>
            <li><strong>Base Models:</strong> RandomForest, XGBoost, LightGBM, AdaBoost, DecisionTree</li>
            <li><strong>Voting:</strong> Simple average of base model predictions</li>
            <li><strong>Transfer Voting:</strong> Weight predictions using DTW (Dynamic Time Warping) 
            similarity between ETFs</li>
        </ol>
        
        <h4>📊 Target Definition</h4>
        <p>MA_Diff(t+1) = MA(t+1) - MA(t)</p>
        <p>We optimize between MA(3) and MA(5) windows, selecting the one with higher 
        out-of-sample annualized return.</p>
        
        <h4>⚡ Trading Logic</h4>
        <ol>
            <li><strong>Selection:</strong> Pick ETF with highest predicted expected return 
            (Predicted_MA_Diff / Price)</li>
            <li><strong>All Negative:</strong> If all predictions negative → go to CASH (3M T-Bill)</li>
            <li><strong>TSL:</strong> Trailing Stop Loss triggers on 2-day cumulative return < -TSL%</li>
            <li><strong>Re-entry:</strong> Z-score > 1.0 on predicted MA_Diff to exit CASH</li>
            <li><strong>Friction:</strong> Transaction costs prevent excessive flipping</li>
        </ol>
        
        <h4>🔄 Automation</h4>
        <ul>
            <li><strong>Daily Update:</strong> 00:30 UTC (after US market close)</li>
            <li><strong>Weekly Retraining:</strong> Full model retraining with latest data</li>
            <li><strong>Data Storage:</strong> HuggingFace Dataset repository</li>
            <li><strong>CI/CD:</strong> GitHub Actions for automated pipeline</li>
        </ul>
        
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
