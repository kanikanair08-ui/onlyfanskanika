import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
from datetime import datetime, timedelta
import itertools

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        border: 1px solid #2e2e3e;
    }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 Stock Price Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">5-Year Historical Data · ARIMA Forecast to June 2027</div>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    ticker = st.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        help="Enter a valid Yahoo Finance ticker (e.g. AAPL, MSFT, RELIANCE.NS)",
    ).upper().strip()

    price_col = st.selectbox(
        "Price Column",
        ["Close", "Open", "High", "Low", "Adj Close"],
        index=0,
    )

    st.markdown("---")
    st.subheader("ARIMA Parameters")
    auto_arima = st.checkbox("Auto-select best (p,d,q)", value=True)

    if not auto_arima:
        p = st.slider("p  (AR order)", 0, 5, 1)
        d = st.slider("d  (differencing)", 0, 2, 1)
        q = st.slider("q  (MA order)", 0, 5, 1)
    else:
        p, d, q = None, None, None

    run_btn = st.button("🚀 Run Forecast", use_container_width=True, type="primary")

# ── Helper functions ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, years: int = 5) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")
    return df


def adf_test(series: pd.Series) -> int:
    """Return the minimum d needed to make series stationary (0, 1, or 2)."""
    for d_val in range(3):
        result = adfuller(series.dropna(), autolag="AIC")
        if result[1] < 0.05:
            return d_val
        series = series.diff()
    return 2


def best_arima_order(series: pd.Series, d: int):
    """Grid search over p in [0..4] and q in [0..4] for lowest AIC."""
    best_aic, best_order = np.inf, (1, d, 1)
    for p_val, q_val in itertools.product(range(5), range(5)):
        try:
            m = ARIMA(series, order=(p_val, d, q_val)).fit()
            if m.aic < best_aic:
                best_aic, best_order = m.aic, (p_val, d, q_val)
        except Exception:
            pass
    return best_order


def forecast_to_june2027(series: pd.Series, order: tuple):
    """Fit ARIMA and forecast monthly closing prices through June 2027."""
    model = ARIMA(series, order=order).fit()

    # Build a monthly date index from the last data point to June 2027
    last_date = series.index[-1]
    target = pd.Timestamp("2027-06-30")
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthEnd(1),
                                 end=target,
                                 freq="ME")
    steps = len(future_dates)

    forecast_res = model.get_forecast(steps=steps)
    forecast_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)

    forecast_df = pd.DataFrame({
        "forecast": forecast_mean.values,
        "lower":    conf_int.iloc[:, 0].values,
        "upper":    conf_int.iloc[:, 1].values,
    }, index=future_dates)

    return model, forecast_df


# ── Main logic ─────────────────────────────────────────────────────────────────
if run_btn or ticker:
    try:
        with st.spinner(f"Fetching 5-year data for **{ticker}** …"):
            df = fetch_data(ticker)

        # Use monthly resampled close for ARIMA (less noise, faster)
        price_series = df[price_col].resample("ME").last().dropna()

        # ── Key metrics row ──────────────────────────────────────────────────
        latest_price = float(df[price_col].iloc[-1])
        start_price  = float(df[price_col].iloc[0])
        pct_change   = (latest_price - start_price) / start_price * 100
        high_5y      = float(df["High"].max())
        low_5y       = float(df["Low"].min())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price",   f"${latest_price:,.2f}")
        c2.metric("5-Year Change",   f"{pct_change:+.1f}%")
        c3.metric("5-Year High",     f"${high_5y:,.2f}")
        c4.metric("5-Year Low",      f"${low_5y:,.2f}")

        st.markdown("---")

        # ── ARIMA order ──────────────────────────────────────────────────────
        with st.spinner("Fitting ARIMA model …"):
            if auto_arima:
                d_val = adf_test(price_series)
                with st.spinner(f"Grid-searching best (p,{d_val},q) — this may take ~30 s …"):
                    order = best_arima_order(price_series, d_val)
            else:
                order = (p, d, q)

            model_fit, forecast_df = forecast_to_june2027(price_series, order)

        # Predicted June 2027 value
        june_2027_price = forecast_df.loc[forecast_df.index.month == 6].iloc[-1]["forecast"]

        # ── ARIMA summary info ───────────────────────────────────────────────
        st.success(
            f"✅ ARIMA{order} fitted successfully  •  AIC: {model_fit.aic:.1f}  •  "
            f"Predicted June 2027: **${june_2027_price:,.2f}**"
        )

        # ── Chart 1 — Historical line chart ─────────────────────────────────
        st.subheader(f"📊 {ticker} — 5-Year {price_col} Price")

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=df.index, y=df[price_col],
            mode="lines",
            name=f"{price_col} Price",
            line=dict(color="#1f77b4", width=1.8),
            hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>",
        ))
        # 50-day & 200-day MAs
        fig_hist.add_trace(go.Scatter(
            x=df.index, y=df[price_col].rolling(50).mean(),
            mode="lines", name="50-day MA",
            line=dict(color="#ff7f0e", width=1.4, dash="dot"),
        ))
        fig_hist.add_trace(go.Scatter(
            x=df.index, y=df[price_col].rolling(200).mean(),
            mode="lines", name="200-day MA",
            line=dict(color="#2ca02c", width=1.4, dash="dash"),
        ))
        fig_hist.update_layout(
            template="plotly_dark",
            height=420,
            xaxis_title="Date",
            yaxis_title=f"Price (USD)",
            legend=dict(orientation="h", y=1.08),
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Chart 2 — ARIMA Forecast ─────────────────────────────────────────
        st.subheader(f"🔮 ARIMA{order} Forecast — Through June 2027")

        fig_fore = go.Figure()

        # Historical (monthly)
        fig_fore.add_trace(go.Scatter(
            x=price_series.index, y=price_series.values,
            mode="lines+markers",
            name="Historical (monthly)",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=3),
            hovertemplate="%{x|%b %Y}<br>$%{y:,.2f}<extra></extra>",
        ))

        # Confidence band
        fig_fore.add_trace(go.Scatter(
            x=pd.concat([forecast_df.index.to_series(), forecast_df.index.to_series()[::-1]]),
            y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(255,127,14,0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
            hoverinfo="skip",
        ))

        # Forecast line
        fig_fore.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#ff7f0e", width=2.5, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
            hovertemplate="%{x|%b %Y}<br>Forecast: $%{y:,.2f}<extra></extra>",
        ))

        # Vertical divider at last historical point
        fig_fore.add_vline(
            x=price_series.index[-1],
            line_dash="dot", line_color="gray",
            annotation_text="Forecast start",
            annotation_position="top right",
        )

        # Annotation for June 2027
        fig_fore.add_annotation(
            x=forecast_df.index[-1],
            y=june_2027_price,
            text=f"June 2027<br><b>${june_2027_price:,.2f}</b>",
            showarrow=True, arrowhead=2,
            ax=-60, ay=-40,
            font=dict(color="#ff7f0e", size=12),
        )

        fig_fore.update_layout(
            template="plotly_dark",
            height=460,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend=dict(orientation="h", y=1.08),
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_fore, use_container_width=True)

        # ── Forecast table ───────────────────────────────────────────────────
        with st.expander("📋 View Full Forecast Table"):
            display_df = forecast_df.copy()
            display_df.index = display_df.index.strftime("%b %Y")
            display_df.columns = ["Forecast ($)", "Lower 95% CI ($)", "Upper 95% CI ($)"]
            display_df = display_df.applymap(lambda x: f"${x:,.2f}")
            st.dataframe(display_df, use_container_width=True)

        # ── Model diagnostics ────────────────────────────────────────────────
        with st.expander("🔬 ARIMA Model Diagnostics"):
            st.text(str(model_fit.summary()))

    except ValueError as ve:
        st.error(f"⚠️ {ve}")
    except Exception as ex:
        st.error(f"❌ Unexpected error: {ex}")
        st.exception(ex)

else:
    st.info("👈 Enter a ticker in the sidebar and click **Run Forecast** to get started.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data via Yahoo Finance · Forecasting via ARIMA (statsmodels) · Charts via Plotly")
