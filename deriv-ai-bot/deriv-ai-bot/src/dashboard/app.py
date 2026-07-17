import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Deriv AI Bot Dashboard", layout="wide")
st.title("🚀 Deriv AI Trading Bot Live Dashboard")

# Sidebar
st.sidebar.header("Controls")
mode = st.sidebar.selectbox("Mode", ["Demo", "Live"])
pause = st.sidebar.button("Pause Trading")

# Main metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Balance", "$1,245.67", "+$23.45")
col2.metric("Win Rate", "62%", "↑ 3%")
col3.metric("Trades Today", "18")
col4.metric("Active Markets", "4")

# Live trades
st.subheader("Recent Trades")
trades = pd.DataFrame({
    'Time': [datetime.now()],
    'Symbol': ['R_100'],
    'Type': ['DIGITOVER'],
    'Stake': [2.0],
    'Result': ['WIN'],
    'P&L': [1.8]
})
st.dataframe(trades, use_container_width=True)

# Performance chart
st.subheader("Equity Curve")
fig = px.line(x=[1,2,3], y=[1000,1020,1245], title="Balance Over Time")
st.plotly_chart(fig, use_container_width=True)

if st.button("Refresh Data"):
    st.rerun()

st.caption("Connected to Deriv API • AI Confidence: 78%")
