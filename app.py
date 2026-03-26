import os

try:
    import yfinance
except ModuleNotFoundError:
    os.system("pip install yfinance pandas numpy plotly ta scikit-learn")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from sklearn.metrics import accuracy_score

# --- 1. CONFIG & UI SETUP ---
st.set_page_config(page_title="Live ML Backtester | IIMA Quant", page_icon="⚡", layout="wide")
st.title("⚡ Momentum + XGBoost Backtester")
st.caption("Live NSE 1-Min Data | Machine Learning Execution | #QuantFinance")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ 1. Asset & Timeframe")
    ticker = st.selectbox("NSE Ticker", ["RELIANCE.NS", "IRCON.NS", "JWL.NS", "^NSEI"])
    days = st.slider("Lookback Period (Days)", 1, 7, 7, help="Yahoo Finance 1m limit is 7 days.")
    
    st.markdown("---")
    st.header("🎛️ 2. Signal Parameters")
    st.caption("Loosen these to increase trade frequency")
    rsi_buy = st.slider("RSI Buy Threshold (Below)", 20, 50, 45)
    rsi_sell = st.slider("RSI Sell Threshold (Above)", 50, 80, 55)
    use_ma_filter = st.checkbox("Require MA Trend Alignment", value=False, help="If unchecked, only relies on RSI + XGBoost")
    cost_bps = st.number_input("Slippage/Cost (bps)", value=10.0, step=1.0) / 10000
    
    st.markdown("---")
    st.header("🛡️ 3. Risk Management")
    sl_pct = st.number_input("Stop-Loss (%)", value=0.15, step=0.05, format="%.2f") / 100
    tp_pct = st.number_input("Take-Profit (%)", value=0.30, step=0.05, format="%.2f") / 100

    st.markdown("---")
    if st.button("🔄 Refresh Live Data", use_container_width=True):
        st.cache_data.clear()

# --- 3. DATA ENGINE & FEATURE ENGINEERING ---
@st.cache_data(show_spinner="Fetching Live NSE Data...")
def load_data(tkr, d):
    df = yf.download(tkr, period=f"{d}d", interval="1m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

data = load_data(ticker, days)
if data.empty:
    st.error("No data returned. The market might be closed or the ticker is invalid.")
    st.stop()

df = data[['Close', 'Volume']].copy()
df['ret'] = df['Close'].pct_change()
df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
df['MA20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
df['MA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
df['MA_ratio'] = df['MA20'] / df['MA50']
df['vol20'] = df['ret'].rolling(20).std()

for i in range(1, 4):
    df[f'lag_ret{i}'] = df['ret'].shift(i)

df['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
df.dropna(inplace=True)

# --- 4. XGBOOST MODEL ---
features = ['RSI', 'MA_ratio', 'vol20', 'lag_ret1', 'lag_ret2', 'lag_ret3']
split = int(len(df) * 0.8)
X_train, X_test = df[features].iloc[:split], df[features].iloc[split:]
y_train, y_test = df['target'].iloc[:split], df['target'].iloc[split:]

model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)
df.loc[df.index[split]:, 'pred'] = model.predict(X_test)

# --- 5. BACKTESTING LOGIC (STATE MACHINE FOR SL/TP) ---
test_df = df.iloc[split:].copy()

# Dynamic Signal Generation based on UI
if use_ma_filter:
    test_df['Buy_Sig'] = ((test_df['RSI'] < rsi_buy) & (test_df['MA20'] > test_df['MA50']) & (test_df['pred'] == 1)).astype(int)
    test_df['Sell_Sig'] = ((test_df['RSI'] > rsi_sell) & (test_df['MA20'] < test_df['MA50']) & (test_df['pred'] == 0)).astype(int)
else:
    test_df['Buy_Sig'] = ((test_df['RSI'] < rsi_buy) & (test_df['pred'] == 1)).astype(int)
    test_df['Sell_Sig'] = ((test_df['RSI'] > rsi_sell) & (test_df['pred'] == 0)).astype(int)

# Fast numpy arrays for loop state management
closes = test_df['Close'].values
buys = test_df['Buy_Sig'].values
sells = test_df['Sell_Sig'].values

positions = np.zeros(len(test_df))
current_pos = 0
entry_price = 0.0

for i in range(len(test_df)):
    price = closes[i]
    
    if current_pos == 1:
        if price <= entry_price * (1 - sl_pct) or price >= entry_price * (1 + tp_pct):
            current_pos = 0 
    elif current_pos == -1:
        if price >= entry_price * (1 + sl_pct) or price <= entry_price * (1 - tp_pct):
            current_pos = 0 

    if current_pos == 0:
        if buys[i] == 1:
            current_pos = 1
            entry_price = price
        elif sells[i] == 1:
            current_pos = -1
            entry_price = price
    else:
        if current_pos == -1 and buys[i] == 1:
            current_pos = 1
            entry_price = price
        elif current_pos == 1 and sells[i] == 1:
            current_pos = -1
            entry_price = price
            
    positions[i] = current_pos

test_df['position'] = positions
test_df['trades'] = test_df['position'].diff().abs().fillna(0)

test_df['strat_ret'] = test_df['position'].shift(1) * test_df['ret'] - (test_df['trades'] * cost_bps)
test_df['bh_ret'] = test_df['ret']

test_df['cum_strat'] = (1 + test_df['strat_ret']).cumprod() - 1
test_df['cum_bh'] = (1 + test_df['bh_ret']).cumprod() - 1

# --- 6. HERO KPIs ---
acc = accuracy_score(y_test, test_df['pred'])
tot_ret = test_df['cum_strat'].iloc[-1]
bh_ret = test_df['cum_bh'].iloc[-1]
active_trades = test_df[test_df['strat_ret'] != 0]
win_rate = len(active_trades[active_trades['strat_ret'] > 0]) / len(active_trades) if not active_trades.empty else 0
roll_max = (1 + test_df['strat_ret']).cumprod().cummax()
max_dd = (((1 + test_df['strat_ret']).cumprod() / roll_max) - 1).min()

st.markdown("### 📊 Strategy Performance vs. Benchmark")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Strategy Return", f"{tot_ret*100:.2f}%", f"{(tot_ret - bh_ret)*100:.2f}% Alpha vs B&H")
c2.metric("ML Model Accuracy", f"{acc*100:.2f}%")
c3.metric("Trade Win Rate", f"{win_rate*100:.2f}%")
c4.metric("Maximum Drawdown", f"{max_dd*100:.2f}%", delta_color="inverse")

# --- 7. MASTER PLOTLY DASHBOARD ---
st.markdown("### 🖥️ Execution Analytics")
fig = make_subplots(rows=2, cols=2, row_heights=[0.5, 0.5], 
                    specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
                    subplot_titles=("Cumulative Equity Curve", "XGBoost Feature Importance", 
                                    "Trade Execution Overlay", "Trade Return Distribution"))

fig.add_trace(go.Scatter(x=test_df.index, y=test_df['cum_strat']*100, name='Strategy', line=dict(color='gold', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=test_df.index, y=test_df['cum_bh']*100, name='Buy & Hold', line=dict(color='cyan', width=2, dash='dot')), row=1, col=1)

imp = pd.DataFrame({'Feature': features, 'Weight': model.feature_importances_}).sort_values('Weight', ascending=True)
fig.add_trace(go.Bar(x=imp['Weight'], y=imp['Feature'], orientation='h', marker=dict(color=imp['Weight'], colorscale='Viridis'), name='Importance'), row=1, col=2)

fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Close'], name='Price', line=dict(color='#555', width=1)), row=2, col=1)

buy_entries = test_df[(test_df['position'] == 1) & (test_df['position'].shift(1) != 1)]
sell_entries = test_df[(test_df['position'] == -1) & (test_df['position'].shift(1) != -1)]
exits = test_df[(test_df['position'] == 0) & (test_df['position'].shift(1) != 0)]

fig.add_trace(go.Scatter(x=buy_entries.index, y=buy_entries['Close'], mode='markers', marker=dict(color='lime', size=10, symbol='triangle-up'), name='Go Long'), row=2, col=1)
fig.add_trace(go.Scatter(x=sell_entries.index, y=sell_entries['Close'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Go Short'), row=2, col=1)
fig.add_trace(go.Scatter(x=exits.index, y=exits['Close'], mode='markers', marker=dict(color='white', size=6, symbol='x'), name='Exit Trade (SL/TP)'), row=2, col=1)

fig.add_trace(go.Histogram(x=active_trades['strat_ret']*100, nbinsx=40, marker_color='gold', name='Returns'), row=2, col=2)

fig.update_layout(height=750, template="plotly_dark", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# --- 8. TRADE LOG & STYLED DATAFRAME ---
st.markdown("### 📜 Recent Signals & Trade Log")

log_df = test_df[['Close', 'RSI', 'MA_ratio', 'pred', 'position', 'strat_ret']].tail(15).copy()
log_df['strat_ret'] = log_df['strat_ret'] * 100 

def color_returns(val):
    if pd.isna(val) or val == 0:
        color = 'grey'
    else:
        color = '#2e8b57' if val > 0 else '#d62728'
    return f'color: {color}; font-weight: bold'

styled_log = log_df.style.format({
    'Close': '₹{:,.2f}',
    'RSI': '{:.1f}',
    'MA_ratio': '{:.3f}',
    'pred': '{:.0f}',
    'position': '{:.0f}',
    'strat_ret': '{:.3f}%'
}).map(color_returns, subset=['strat_ret'])

st.dataframe(styled_log, use_container_width=True)

csv = test_df.to_csv().encode('utf-8')
st.download_button("📥 Download Full Trade Log CSV", data=csv, file_name='xgb_backtest_trades.csv', mime='text/csv')
