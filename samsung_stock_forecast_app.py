import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import itertools
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Samsung Stock Price Analysis & Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #0323f0;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #000428 0%, #004e92 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-bottom: 5px solid #00d4ff;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #004e92;
        padding: 0.8rem;
        border-bottom: 3px solid #004e92;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .info-box {
        background-color: black;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: lightgreen;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: black;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #000428 0%, #004e92 100%);
        color: white;
        border-radius: 15px;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING AND CACHING
# ============================================
@st.cache_data
def load_data():
    """Load Samsung stock data"""
    try:
        df = pd.read_csv('005930.KS.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = df.reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.error("❌ Could not find '005930.KS.csv'. Please make sure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

@st.cache_data
def prepare_time_series(df, price_column='close'):
    """Prepare time series data for analysis"""
    ts_data = df[['Date', price_column]].copy()
    ts_data.columns = ['ds', 'y']
    ts_data = ts_data.dropna()
    ts_data.set_index('ds', inplace=True)
    return ts_data

@st.cache_data
def calculate_moving_averages(df, windows=[20, 50, 200]):
    """Calculate moving averages"""
    df_ma = df.copy()
    for window in windows:
        df_ma[f'MA_{window}'] = df_ma['close'].rolling(window=window).mean()
    return df_ma

# ============================================
# STATIONARITY TEST
# ============================================
def test_stationarity(timeseries, title):
    """Perform Augmented Dickey-Fuller test for stationarity"""
    result = adfuller(timeseries.dropna())
    
    output = {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'Is Stationary': result[1] < 0.05
    }
    
    return output

# ============================================
# FIXED ARIMA MODEL FUNCTIONS
# ============================================
def find_best_arima_params(series, p_range, d_range, q_range, test_size=30):
    """Find best ARIMA parameters using AIC"""
    train = series[:-test_size]
    
    best_aic = float('inf')
    best_order = None
    best_model = None
    
    progress_bar = st.progress(0)
    total_combinations = len(p_range) * len(d_range) * len(q_range)
    count = 0
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(train, order=(p, d, q))
                    model_fit = model.fit()
                    
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        best_model = model_fit
                except:
                    continue
                finally:
                    count += 1
                    progress_bar.progress(count / total_combinations)
    
    progress_bar.empty()
    return best_order, best_model, best_aic

def fit_arima_model(series, order):
    """Fit ARIMA model with given order"""
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

# FIXED: Proper date handling for forecast
def forecast_arima(model_fit, steps, last_date):
    """Generate forecasts with proper date handling"""
    forecast = model_fit.forecast(steps=steps)
    
    # Create forecast dates (business days)
    forecast_dates = []
    current_date = last_date
    
    for i in range(steps):
        # Skip weekends (Saturday and Sunday)
        next_date = current_date + timedelta(days=1)
        while next_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            next_date += timedelta(days=1)
        forecast_dates.append(next_date)
        current_date = next_date
    
    return forecast, forecast_dates

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/null/samsung.png", width=100)
    st.title("📈 Samsung Stock Analysis")
    st.markdown("---")
    
    # Navigation
    st.subheader("Navigation")
    page = st.radio(
        "Go to",
        ["📊 Data Overview", 
         "📈 Trend Analysis", 
         "🔄 Stationarity & ACF/PACF",
         "🤖 ARIMA Model",
         "🔮 Price Forecasting"]
    )
    
    st.markdown("---")
    
    # Parameters for ARIMA
    if page in ["🤖 ARIMA Model", "🔮 Price Forecasting"]:
        st.subheader("⚙️ ARIMA Parameters")
        
        # Parameter ranges
        p_range = st.multiselect(
            "AR order (p) - Try [0,1,2]",
            options=[0, 1, 2, 3, 4, 5],
            default=[0, 1, 2]
        )
        
        d_range = st.multiselect(
            "Difference order (d) - Try [0,1]",
            options=[0, 1, 2],
            default=[0, 1]
        )
        
        q_range = st.multiselect(
            "MA order (q) - Try [0,1,2]",
            options=[0, 1, 2, 3, 4, 5],
            default=[0, 1, 2]
        )
        
        st.markdown("---")
        
        # Forecast parameters
        forecast_days = st.slider(
            "Forecast Days",
            min_value=7,
            max_value=90,
            value=30,
            step=1
        )
        
        test_size = st.slider(
            "Test Size (days for validation)",
            min_value=10,
            max_value=60,
            value=30,
            step=5
        )
    
    st.markdown("---")
    
    # About section
    st.markdown("### About")
    st.info("""
    **📊 Samsung Electronics Stock Analysis**
    
    This app analyzes Samsung stock price data and forecasts future prices using ARIMA modeling.
    
    **Features:**
    - 📈 Stock price trends & moving averages
    - 🔄 Stationarity testing
    - 📊 ACF/PACF analysis
    - 🤖 Automatic ARIMA parameter selection
    - 🔮 30-day price forecasting
    """)

# ============================================
# MAIN CONTENT
# ============================================
st.markdown('<div class="main-header">📈 Samsung Electronics (005930.KS) Stock Analysis & Forecasting</div>', unsafe_allow_html=True)

# Load data
df = load_data()

if df is None:
    st.stop()

# Prepare time series data
ts_data = prepare_time_series(df, 'close')
df_ma = calculate_moving_averages(df)

# ============================================
# PAGE 1: DATA OVERVIEW
# ============================================
if page == "📊 Data Overview":
    st.markdown('<div class="sub-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Days", f"{df.shape[0]:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        change = ((current_price - prev_price) / prev_price) * 100
        st.metric("Current Price", f"₩{current_price:,.0f}", f"{change:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Price", f"₩{df['close'].mean():,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Volume", f"{df['volume'].sum():,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Price Statistics
    st.subheader("📈 Price Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stats_df = pd.DataFrame({
            'Metric': ['Open', 'High', 'Low', 'Close'],
            'Mean': [f"₩{df['open'].mean():,.0f}", f"₩{df['high'].mean():,.0f}", 
                    f"₩{df['low'].mean():,.0f}", f"₩{df['close'].mean():,.0f}"],
            'Min': [f"₩{df['open'].min():,.0f}", f"₩{df['high'].min():,.0f}", 
                   f"₩{df['low'].min():,.0f}", f"₩{df['close'].min():,.0f}"],
            'Max': [f"₩{df['open'].max():,.0f}", f"₩{df['high'].max():,.0f}", 
                   f"₩{df['low'].max():,.0f}", f"₩{df['close'].max():,.0f}"],
            'Std': [f"₩{df['open'].std():,.0f}", f"₩{df['high'].std():,.0f}", 
                   f"₩{df['low'].std():,.0f}", f"₩{df['close'].std():,.0f}"]
        })
        st.table(stats_df)
    
    with col2:
        # Volume distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['volume']/1e6, bins=30, color='#004e92', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Volume (Millions)')
        ax.set_ylabel('Frequency')
        ax.set_title('Trading Volume Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Data Preview
    st.subheader("📋 Data Preview")
    
    tab1, tab2 = st.tabs(["Latest Data", "Raw Data"])
    
    with tab1:
        st.dataframe(df.tail(10), use_container_width=True)
    
    with tab2:
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="samsung_stock_data.csv",
            mime="text/csv"
        )

# ============================================
# PAGE 2: TREND ANALYSIS
# ============================================
elif page == "📈 Trend Analysis":
    st.markdown('<div class="sub-header">📈 Stock Trend & Moving Averages</div>', unsafe_allow_html=True)
    
    # Moving average parameters
    col1, col2 = st.columns(2)
    
    with col1:
        ma_windows = st.multiselect(
            "Select Moving Average Windows",
            options=[5, 10, 20, 50, 100, 200],
            default=[20, 50, 200]
        )
    
    with col2:
        date_range = st.date_input(
            "Select Date Range",
            value=(df['Date'].min(), df['Date'].max()),
            min_value=df['Date'].min(),
            max_value=df['Date'].max()
        )
    
    # Filter data by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
        df_filtered = df[mask].copy()
        df_ma_filtered = calculate_moving_averages(df_filtered, ma_windows)
    else:
        df_filtered = df
        df_ma_filtered = calculate_moving_averages(df, ma_windows)
    
    # Interactive Plotly chart
    st.subheader("📊 Interactive Stock Price Chart")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_filtered['Date'],
            open=df_filtered['open'],
            high=df_filtered['high'],
            low=df_filtered['low'],
            close=df_filtered['close'],
            name="Price",
            showlegend=True
        )
    )
    
    # Add moving averages
    colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#6BCB77', '#9B59B6']
    for i, window in enumerate(ma_windows):
        fig.add_trace(
            go.Scatter(
                x=df_ma_filtered['Date'],
                y=df_ma_filtered[f'MA_{window}'],
                name=f"MA-{window}",
                line=dict(color=colors[i % len(colors)], width=2)
            )
        )
    
    # Add volume as bar chart on secondary y-axis
    fig.add_trace(
        go.Bar(
            x=df_filtered['Date'],
            y=df_filtered['volume']/1e6,
            name="Volume (M)",
            marker_color='lightgray',
            opacity=0.5
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Samsung Stock Price with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (KRW)",
        template="plotly_white",
        height=600,
        hovermode='x unified'
    )
    
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_yaxes(title_text="Price (KRW)", secondary_y=False)
    fig.update_yaxes(title_text="Volume (Millions)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Daily Returns Analysis
    st.subheader("📉 Daily Returns Analysis")
    
    df_filtered['returns'] = df_filtered['close'].pct_change() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Returns distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_filtered['returns'].dropna(), bins=50, color='#004e92', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.axvline(df_filtered['returns'].mean(), color='green', linestyle='--', 
                  linewidth=2, label=f"Mean: {df_filtered['returns'].mean():.2f}%")
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Returns over time
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_filtered['Date'], df_filtered['returns'], color='#004e92', alpha=0.7, linewidth=0.5)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.axhline(df_filtered['returns'].mean(), color='green', linestyle='--', 
                  linewidth=1, label=f"Mean: {df_filtered['returns'].mean():.2f}%")
        ax.set_xlabel('Date')
        ax.set_ylabel('Daily Return (%)')
        ax.set_title('Daily Returns Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
    
    # Returns statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Daily Return", f"{df_filtered['returns'].mean():.2f}%")
    with col2:
        st.metric("Std Daily Return", f"{df_filtered['returns'].std():.2f}%")
    with col3:
        st.metric("Max Daily Return", f"{df_filtered['returns'].max():.2f}%")
    with col4:
        st.metric("Min Daily Return", f"{df_filtered['returns'].min():.2f}%")

# ============================================
# PAGE 3: STATIONARITY & ACF/PACF
# ============================================
elif page == "🔄 Stationarity & ACF/PACF":
    st.markdown('<div class="sub-header">🔄 Stationarity Test & ACF/PACF Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>📌 What is Stationarity?</h4>
    <p>A time series is stationary if its statistical properties (mean, variance) do not change over time.
    ARIMA models require the time series to be stationary.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select series type
        series_type = st.radio(
            "Select Series Type",
            ["Original Close Price", "Log Returns", "First Difference"],
            index=0
        )
        
        if series_type == "Original Close Price":
            series = ts_data['y']
            title = "Original Close Price"
        elif series_type == "Log Returns":
            series = np.log(ts_data['y']).diff()
            title = "Log Returns"
        else:
            series = ts_data['y'].diff()
            title = "First Difference"
    
    with col2:
        st.markdown("### 📊 Stationarity Test Results")
        
        # Perform ADF test
        stationarity_result = test_stationarity(series, title)
        
        result_df = pd.DataFrame({
            'Metric': ['ADF Statistic', 'p-value', 'Is Stationary?'],
            'Value': [
                f"{stationarity_result['Test Statistic']:.4f}",
                f"{stationarity_result['p-value']:.4f}",
                '✅ Yes' if stationarity_result['Is Stationary'] else '❌ No'
            ]
        })
        st.table(result_df)
        
        if stationarity_result['Is Stationary']:
            st.success(f"✅ The {title} series is stationary! Suitable for ARIMA modeling.")
        else:
            st.warning(f"⚠️ The {title} series is NOT stationary. Try differencing.")
    
    st.markdown("---")
    
    # ACF and PACF plots
    st.subheader("📈 ACF & PACF Plots")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_acf(series.dropna(), lags=40, ax=ax)
        ax.set_title(f'Autocorrelation Function (ACF) - {title}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_pacf(series.dropna(), lags=40, ax=ax, method='ywm')
        ax.set_title(f'Partial Autocorrelation Function (PACF) - {title}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Partial Autocorrelation')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("""
    <div class="info-box">
    <h4>🔍 How to interpret ACF/PACF for ARIMA:</h4>
    <ul>
        <li><strong>AR(p):</strong> PACF cuts off after lag p, ACF decays gradually</li>
        <li><strong>MA(q):</strong> ACF cuts off after lag q, PACF decays gradually</li>
        <li><strong>ARMA(p,q):</strong> Both ACF and PACF decay gradually</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE 4: ARIMA MODEL
# ============================================
elif page == "🤖 ARIMA Model":
    st.markdown('<div class="sub-header">🤖 ARIMA Model Fitting & Optimization</div>', unsafe_allow_html=True)
    
    # Check if parameters are selected
    if not p_range or not d_range or not q_range:
        st.warning("⚠️ Please select ARIMA parameters in the sidebar.")
        st.stop()
    
    # Prepare data
    series = ts_data['y']
    
    # Split into train and test
    train = series[:-test_size]
    test = series[-test_size:]
    
    st.markdown(f"""
    <div class="info-box">
    <h4>📊 Data Split</h4>
    <ul>
        <li><strong>Training period:</strong> {train.index[0].strftime('%Y-%m-%d')} to {train.index[-1].strftime('%Y-%m-%d')}</li>
        <li><strong>Testing period:</strong> {test.index[0].strftime('%Y-%m-%d')} to {test.index[-1].strftime('%Y-%m-%d')}</li>
        <li><strong>Training samples:</strong> {len(train)} days</li>
        <li><strong>Testing samples:</strong> {len(test)} days</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Find best ARIMA parameters
    with st.spinner("🔍 Searching for optimal ARIMA parameters... This may take a minute..."):
        best_order, best_model, best_aic = find_best_arima_params(
            series, p_range, d_range, q_range, test_size
        )
    
    if best_model is None:
        st.error("❌ Could not find suitable ARIMA parameters. Try different parameter ranges.")
        st.stop()
    
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown(f"### ✅ Best ARIMA Model Found!")
    st.markdown(f"""
    - **Order (p,d,q):** {best_order}
    - **AIC Value:** {best_aic:.2f}
    - **BIC Value:** {best_model.bic:.2f}
    - **Log Likelihood:** {best_model.llf:.2f}
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model summary
    with st.expander("📋 View ARIMA Model Summary"):
        st.text(str(best_model.summary()))
    
    st.markdown("---")
    
    # Residual analysis
    st.subheader("📊 Residual Diagnostics")
    
    residuals = best_model.resid
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Residuals over time
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(residuals.index, residuals.values, color='#004e92', alpha=0.7)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Date')
        ax.set_ylabel('Residual')
        ax.set_title('ARIMA Model Residuals', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Residual distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(residuals, bins=30, color='#004e92', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Residual')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    # ACF of residuals
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_acf(residuals.dropna(), lags=40, ax=ax)
    ax.set_title('Autocorrelation of Residuals', fontsize=14, fontweight='bold')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    # Ljung-Box test for residual autocorrelation
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals.dropna(), lags=[10, 20, 30], return_df=True)
    
    st.subheader("📋 Ljung-Box Test for Residual Autocorrelation")
    st.dataframe(lb_test.style.format({'lb_stat': '{:.4f}', 'lb_pvalue': '{:.4f}'}))

# ============================================
# PAGE 5: PRICE FORECASTING (COMPLETELY FIXED)
# ============================================
else:
    st.markdown('<div class="sub-header">🔮 Stock Price Forecasting</div>', unsafe_allow_html=True)
    
    # Check if parameters are selected
    if not p_range or not d_range or not q_range:
        st.warning("⚠️ Please select ARIMA parameters in the sidebar.")
        st.stop()
    
    # Prepare data
    series = ts_data['y']
    last_date = series.index[-1]
    
    # Split into train and test
    train = series[:-test_size]
    test = series[-test_size:]
    
    # Find best ARIMA parameters
    with st.spinner("🔍 Training ARIMA model..."):
        best_order, best_model, best_aic = find_best_arima_params(
            series, p_range, d_range, q_range, test_size
        )
    
    if best_model is None:
        st.error("❌ Could not find suitable ARIMA parameters. Try different parameter ranges.")
        st.stop()
    
    st.markdown(f"""
    <div class="success-box">
    <h4>✅ Model Trained Successfully</h4>
    <p>ARIMA{best_order} | AIC: {best_aic:.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate forecasts using FIXED function
    forecast_values, forecast_dates = forecast_arima(best_model, forecast_days, last_date)
    
    # Test set predictions
    test_pred = best_model.forecast(steps=len(test))
    
    # Calculate metrics on test set
    mae = mean_absolute_error(test, test_pred)
    rmse = np.sqrt(mean_squared_error(test, test_pred))
    mape = np.mean(np.abs((test - test_pred) / test)) * 100
    
    # Display test metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("MAE", f"₩{mae:,.0f}")
    with col2:
        st.metric("RMSE", f"₩{rmse:,.0f}")
    with col3:
        st.metric("MAPE", f"{mape:.2f}%")
    
    st.markdown("---")
    
    # Historical vs Forecast plot - USING MATPLOTLIB INSTEAD OF PLOTLY
    st.subheader("📈 Historical Prices & Forecast")
    
    # Create figure with matplotlib (avoids all Plotly timestamp issues)
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot historical data
    ax.plot(series.index, series.values, color='#004e92', linewidth=2, label='Historical Price')
    
    # Add vertical line for train/test split
    ax.axvline(x=train.index[-1], color='gray', linestyle='--', linewidth=1.5, label='Train/Test Split')
    
    # Plot test predictions
    ax.plot(test.index, test_pred, color='#FF6B6B', linestyle='--', linewidth=2, label='Test Predictions')
    
    # Plot forecast
    ax.plot(forecast_dates, forecast_values, color='#4ECDC4', linewidth=3, label=f'{forecast_days}-Day Forecast')
    
    # Add confidence interval
    ax.fill_between(
        forecast_dates,
        forecast_values - 1.96 * np.std(best_model.resid),
        forecast_values + 1.96 * np.std(best_model.resid),
        color='#4ECDC4',
        alpha=0.2,
        label='95% Confidence Interval'
    )
    
    # Customize the plot
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (KRW)', fontsize=12)
    ax.set_title(f'Samsung Stock Price Forecast - Next {forecast_days} Days', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Format y-axis with commas for thousands
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₩{x:,.0f}'))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Display the plot
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Alternative Plotly version using numeric indices (if you prefer interactive plots)
    with st.expander("📊 View Interactive Plotly Chart"):
        # Create a dataframe with all data for Plotly
        plot_data = pd.DataFrame()
        
        # Historical data
        hist_df = pd.DataFrame({
            'Date': series.index,
            'Value': series.values,
            'Type': 'Historical'
        })
        
        # Test predictions
        test_df = pd.DataFrame({
            'Date': test.index,
            'Value': test_pred,
            'Type': 'Test Prediction'
        })
        
        # Forecast
        forecast_df_plot = pd.DataFrame({
            'Date': forecast_dates,
            'Value': forecast_values,
            'Type': 'Forecast'
        })
        
        # Confidence interval
        ci_df = pd.DataFrame({
            'Date': forecast_dates,
            'Lower': forecast_values - 1.96 * np.std(best_model.resid),
            'Upper': forecast_values + 1.96 * np.std(best_model.resid)
        })
        
        # Combine for plotting
        all_data = pd.concat([hist_df, test_df, forecast_df_plot], ignore_index=True)
        
        # Create Plotly figure with numeric indices for x-axis
        fig2 = go.Figure()
        
        # Add traces with index as x-axis to avoid date issues
        fig2.add_trace(go.Scatter(
            x=list(range(len(hist_df))),
            y=hist_df['Value'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#004e92', width=2),
            hovertemplate='Index: %{x}<br>Price: ₩%{y:,.0f}<extra></extra>'
        ))
        
        fig2.add_trace(go.Scatter(
            x=list(range(len(hist_df), len(hist_df) + len(test_df))),
            y=test_df['Value'],
            mode='lines',
            name='Test Predictions',
            line=dict(color='#FF6B6B', width=2, dash='dot'),
            hovertemplate='Index: %{x}<br>Price: ₩%{y:,.0f}<extra></extra>'
        ))
        
        fig2.add_trace(go.Scatter(
            x=list(range(len(hist_df) + len(test_df), len(hist_df) + len(test_df) + len(forecast_df_plot))),
            y=forecast_df_plot['Value'],
            mode='lines',
            name=f'{forecast_days}-Day Forecast',
            line=dict(color='#4ECDC4', width=3),
            hovertemplate='Index: %{x}<br>Price: ₩%{y:,.0f}<extra></extra>'
        ))
        
        # Add confidence interval
        fig2.add_trace(go.Scatter(
            x=list(range(len(hist_df) + len(test_df), len(hist_df) + len(test_df) + len(ci_df))) * 2,
            y=list(ci_df['Upper']) + list(ci_df['Lower'][::-1]),
            fill='toself',
            fillcolor='rgba(78, 205, 196, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True,
            hovertemplate='Upper: ₩%{y:,.0f}<extra></extra>'
        ))
        
        # Add vertical line for split
        fig2.add_vline(
            x=len(hist_df) - len(test) - 1,
            line_dash="dash",
            line_color="gray",
            annotation_text="Train/Test Split",
            annotation_position="top"
        )
        
        # Update layout
        fig2.update_layout(
            title=f"Samsung Stock Price Forecast - Next {forecast_days} Days",
            xaxis_title="Time (Trading Days)",
            yaxis_title="Price (KRW)",
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        
        fig2.update_yaxes(tickformat=",.0f", ticksuffix=" ₩")
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.info("""
        **Note:** The interactive chart uses trading day indices instead of dates to avoid timestamp issues.
        For actual dates, refer to the forecast table below.
        """)
    
    st.markdown("---")
    
    # Forecast table
    st.subheader("📊 Forecast Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Format forecast dataframe
        display_forecast = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'Forecast': [f"₩{x:,.0f}" for x in forecast_values],
            'Lower CI': [f"₩{(x - 1.96 * np.std(best_model.resid)):,.0f}" for x in forecast_values],
            'Upper CI': [f"₩{(x + 1.96 * np.std(best_model.resid)):,.0f}" for x in forecast_values]
        })
        
        st.dataframe(display_forecast, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Summary Statistics")
        
        last_price = series.iloc[-1]
        first_forecast = forecast_values.iloc[0]
        last_forecast = forecast_values.iloc[-1]
        
        forecast_change = ((last_forecast - last_price) / last_price) * 100
        forecast_range = (forecast_values.max() - forecast_values.min()) / forecast_values.mean() * 100
        
        st.metric("Current Price", f"₩{last_price:,.0f}")
        st.metric(f"Forecast ({forecast_days} days)", f"₩{last_forecast:,.0f}", 
                 f"{forecast_change:.2f}%")
        st.metric("Forecast Range", f"₩{forecast_values.min():,.0f} - ₩{forecast_values.max():,.0f}")
        st.metric("Forecast Volatility", f"{forecast_range:.2f}%")
    
    st.markdown("---")
    
    # Download forecast
    download_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
        'Forecast': forecast_values,
        'Lower_CI': forecast_values - 1.96 * np.std(best_model.resid),
        'Upper_CI': forecast_values + 1.96 * np.std(best_model.resid)
    })
    
    csv_forecast = download_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast Data",
        data=csv_forecast,
        file_name=f"samsung_forecast_{forecast_days}days.csv",
        mime="text/csv"
    )
    
    # Investment recommendation
    st.subheader("💡 Investment Insight")
    
    if forecast_change > 5:
        recommendation = "STRONG BUY"
        color = "#28a745"
        icon = "🚀"
    elif forecast_change > 2:
        recommendation = "BUY"
        color = "#5cb85c"
        icon = "📈"
    elif forecast_change > -2:
        recommendation = "HOLD"
        color = "#f0ad4e"
        icon = "⚖️"
    elif forecast_change > -5:
        recommendation = "SELL"
        color = "#d9534f"
        icon = "📉"
    else:
        recommendation = "STRONG SELL"
        color = "#c9302c"
        icon = "⚠️"
    
    st.markdown(f"""
    <div style="background-color: {color}20; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {color};">
        <h3 style="color: {color}; margin-bottom: 0.5rem;">{icon} {recommendation}</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
            Based on ARIMA model forecast, the stock price is expected to 
            <strong style="color: {color};">
                {'increase' if forecast_change > 0 else 'decrease'} by {abs(forecast_change):.2f}%
            </strong>
            over the next {forecast_days} trading days.
        </p>
        <p style="color: #666; margin-bottom: 0;">
            Confidence Level: 95% | Model: ARIMA{best_order}
        </p>
    </div>
    """, unsafe_allow_html=True)
      
# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">📈 Samsung Electronics Stock Analysis & Forecasting</p>
    <p style="opacity: 0.9;">Built with Streamlit, Statsmodels, and Plotly | ARIMA Time Series Forecasting</p>
    <p style="opacity: 0.7; font-size: 0.9rem; margin-top: 0.5rem;">
        ⚠️ This is for educational purposes only. Not financial advice.
    </p>
</div>
""", unsafe_allow_html=True)