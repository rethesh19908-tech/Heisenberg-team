"""
Personal Expense Forecasting System
------------------------------------
Merged single-file version. This used to be two files:
  - app.py   (the Streamlit web app)
  - test.py  (a standalone script that checked forecast accuracy on
              sample data)

They're combined here because the app and the accuracy check share the
same backend logic (cleaning the CSV, building the category x month
table). Keeping one copy of that logic means the number you see in the
"Forecast Accuracy" section of the app is guaranteed to describe the
same forecast the app actually shows you -- not a slightly different
copy that could drift out of sync.

Run modes:
  streamlit run app.py        -> launches the web app (normal use)
  python app.py --selftest    -> runs the accuracy check on built-in
                                  sample data and prints results to the
                                  terminal. No Streamlit needed for this,
                                  so it's a fast way to sanity-check the
                                  forecasting logic while developing.
"""

import sys
import numpy as np
import pandas as pd

# =========================================================
# BACKEND LOGIC (shared by the app and the accuracy checker)
# =========================================================


def _prepare_expenses(df):
    """Common cleanup: parse dates, keep only expense rows (amount < 0),
    and flip them positive so they're easier to sum and chart."""
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["Month_Year"] = df["transaction_date"].dt.to_period("M")

    expenses = df[df["amount"] < 0].copy()
    expenses["amount"] = -expenses["amount"]
    return df, expenses


def weighted_forecast(category_monthly):
    """
    Forecast next month's spend per category using a WEIGHTED moving
    average instead of a plain average.

    Why: a plain average treats a transaction from 4 months ago exactly
    the same as one from last month. If spending is genuinely rising or
    falling, a plain average reacts slowly and the forecast lags behind
    reality. Giving more recent months a bigger weight makes the
    forecast track recent behaviour more closely, without needing a
    heavier model (like Prophet) that this amount of data doesn't
    really justify yet.

    category_monthly: DataFrame, rows = category, columns = months
                       (in chronological order), values = amount spent.
    Returns: Series, category -> forecasted amount for the next month.
    """
    months = list(category_monthly.columns)
    n = len(months)
    if n == 0:
        return pd.Series(dtype=float)

    # weights 1, 2, 3, ... n -> the most recent month gets the largest
    # weight, the oldest month gets the smallest.
    weights = np.arange(1, n + 1)
    weighted_sum = category_monthly.mul(weights, axis=1).sum(axis=1)
    return weighted_sum / weights.sum()


def simple_average_forecast(category_monthly):
    """The original baseline: a plain average across all months. Kept
    around so the accuracy checker can show how much the weighted
    version actually improves on it."""
    return category_monthly.mean(axis=1)


def detect_trending_up(category_monthly, min_months=3):
    """
    Flags a category as "trending up" if it has an overall rising trend
    across the months available, using the slope of a best-fit line
    instead of requiring every single month to beat the previous one.

    Why the change: requiring a strict month-over-month increase means
    one ordinary dip (very common in real spending data) breaks the
    streak, even when spending is clearly climbing overall. A trend
    line tolerates normal noise while still catching real trends, and
    needs at least `min_months` data points before it says anything.
    """
    trending = []
    for cat, row in category_monthly.iterrows():
        vals = row.values.astype(float)
        if len(vals) < min_months:
            continue
        x = np.arange(len(vals))
        slope = np.polyfit(x, vals, 1)[0]
        # Rising by more than 5% of the category's average monthly
        # spend, per month, counts as a real trend rather than noise.
        if vals.mean() > 0 and slope > 0.05 * vals.mean():
            trending.append(cat)
    return trending


def process_expenses(df):
    """Processes the uploaded CSV dataframe using Pandas and produces
    everything the Streamlit UI needs to display."""
    df, expenses = _prepare_expenses(df)

    total_spending = float(expenses["amount"].sum())

    cat_totals = expenses.groupby("category")["amount"].sum()
    highest_category = str(cat_totals.idxmax()) if not cat_totals.empty else "None"

    category_breakdown = cat_totals.reset_index()
    category_breakdown.columns = ["Category", "Total Spent"]

    monthly_totals = expenses.groupby("Month_Year")["amount"].sum().reset_index()
    monthly_totals["Month_Year"] = monthly_totals["Month_Year"].astype(str)
    monthly_totals.columns = ["Month", "Total Spent"]

    category_monthly = expenses.groupby(["category", "Month_Year"])["amount"].sum().unstack(fill_value=0)
    category_monthly = category_monthly[sorted(category_monthly.columns)]  # chronological order

    forecast_val = float(weighted_forecast(category_monthly).sum())
    trending_up = detect_trending_up(category_monthly)

    return {
        "forecast": forecast_val,
        "total_spending": total_spending,
        "highest_category": highest_category,
        "monthly_breakdown": monthly_totals,
        "category_breakdown": category_breakdown,
        "trending_up": trending_up,
        "category_monthly": category_monthly,  # reused by the backtest section below
    }


# =========================================================
# ACCURACY CHECK (this used to be the whole of test.py) --
# now a reusable function the app calls on the user's own data
# =========================================================


def evaluate_forecast(df, verbose=False):
    """
    Backtests the forecast: trains on every month except the most
    recent one, predicts that held-out month, and compares the
    prediction to what actually happened. Returns MAE / RMSE / MAPE for
    both the weighted-average forecast and the plain-average baseline,
    so the improvement can be shown with numbers instead of just
    claimed.
    """
    _, expenses = _prepare_expenses(df)
    category_monthly = expenses.groupby(["category", "Month_Year"])["amount"].sum().unstack(fill_value=0)
    months = sorted(category_monthly.columns)

    if len(months) < 2:
        return None

    train_months = months[:-1]
    test_month = months[-1]
    train_data = category_monthly[train_months]
    actual = category_monthly[test_month]

    def _score(predicted):
        errors = predicted - actual
        mae = errors.abs().mean()
        rmse = (errors**2).mean() ** 0.5
        mape = (errors.abs() / actual.replace(0, pd.NA)).dropna().mean() * 100
        return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}

    weighted_scores = _score(weighted_forecast(train_data))
    baseline_scores = _score(simple_average_forecast(train_data))

    if verbose:
        print("Category x Month table:\n", category_monthly, "\n")
        print("Train months:", list(train_months), " Test month:", test_month, "\n")
        print("Actual (test month):\n", actual, "\n")
        print("Weighted-average forecast accuracy:", weighted_scores)
        print("Plain-average (baseline) accuracy: ", baseline_scores)

    return {
        "test_month": str(test_month),
        "weighted": weighted_scores,
        "baseline": baseline_scores,
    }


def _sample_dataframe():
    """Small built-in sample dataset (same columns as a real bank
    statement CSV) used for the quick self-test."""
    sample_data = {
        "transaction_id": [f"TXN{i:03d}" for i in range(1, 13)],
        "transaction_date": [
            "2026-06-05", "2026-06-10", "2026-06-15", "2026-06-20",
            "2026-07-05", "2026-07-10", "2026-07-15", "2026-07-20",
            "2026-08-05", "2026-08-10", "2026-08-15", "2026-08-20",
        ],
        "category": [
            "Food", "Transport", "Shopping", "Entertainment",
            "Food", "Transport", "Shopping", "Entertainment",
            "Food", "Transport", "Shopping", "Entertainment",
        ],
        "amount": [
            -3000, -1500, -2000, -800,
            -3200, -1600, -2500, -750,
            -3100, -1700, -3000, -900,
        ],
    }
    return pd.DataFrame(sample_data)


def _run_selftest():
    df = _sample_dataframe()
    result = evaluate_forecast(df, verbose=True)
    print("\nReturned dict:", result)
    if result:
        w, b = result["weighted"]["mae"], result["baseline"]["mae"]
        if w < b:
            print(f"\nWeighted average is more accurate: MAE {w:.2f} vs {b:.2f} (baseline)")
        elif w > b:
            print(f"\nBaseline happens to score better on this tiny sample: MAE {b:.2f} vs {w:.2f} (weighted)")
        else:
            print("\nBoth methods score the same on this sample.")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--selftest":
    _run_selftest()
    sys.exit(0)


# =========================================================
# FRONTEND: everything below only runs for `streamlit run app.py`
# (streamlit is imported down here, not at the top, so
# `python app.py --selftest` doesn't need it installed at all)
# =========================================================

import streamlit as st

st.set_page_config(
    page_title="Personal Expense Forecasting System",
    page_icon="\U0001F4B0",
    layout="wide",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink:        #0B0E14;
    --surface:    #141922;
    --surface-2:  #1B2130;
    --hairline:   #2B3243;
    --gold:       #C9A227;
    --gold-soft:  rgba(201, 162, 39, 0.14);
    --emerald:    #3FA796;
    --rose:       #C1666B;
    --text:       #ECE9E2;
    --text-muted: #8B93A7;
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #151B27 0%, var(--ink) 55%);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] { background: var(--surface); }

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.brand-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.35em;
    font-size: 0.68rem;
    color: var(--gold);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

h1 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    font-size: 2.6rem !important;
    color: var(--text) !important;
    letter-spacing: -0.01em;
    margin-bottom: 0.2rem !important;
}

.stApp > header + div .block-container p {
    color: var(--text-muted);
}

.brand-rule {
    height: 1px;
    width: 100%;
    margin: 1.4rem 0 2rem 0;
    background: linear-gradient(90deg, var(--gold) 0%, var(--hairline) 35%, transparent 100%);
}

h2, h3 {
    font-family: 'Fraunces', serif !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    letter-spacing: -0.005em;
}

h3 {
    font-size: 1.05rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em !important;
    color: var(--gold) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1px dashed var(--hairline) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--gold) !important;
}

.stButton > button {
    background: linear-gradient(180deg, #D4B84A 0%, var(--gold) 100%) !important;
    color: #14100A !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.4rem !important;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 16px var(--gold-soft);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px var(--gold-soft);
    color: #14100A !important;
}

.stDownloadButton > button {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    border-color: var(--gold) !important;
    color: var(--gold) !important;
}

div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 12px;
    padding: 1.1rem 1.3rem 1rem 1.3rem;
    position: relative;
    overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--gold);
}
div[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    color: var(--text) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--hairline);
    border-radius: 10px;
    overflow: hidden;
}

div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid var(--hairline) !important;
    font-family: 'Inter', sans-serif !important;
}

hr {
    border-color: var(--hairline) !important;
    margin: 1.8rem 0 !important;
}

details {
    background: var(--surface) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 10px !important;
}

.stSpinner > div {
    color: var(--gold) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="brand-eyebrow">Statement &amp; Forecast &middot; Private Ledger</div>', unsafe_allow_html=True)
st.title("\U0001F4B0 Personal Expense Forecasting System")

st.write(
    "Upload your bank statement (CSV) to analyze monthly spending, "
    "view category-wise expenses, and forecast next month's expenses."
)
st.markdown('<div class="brand-rule"></div>', unsafe_allow_html=True)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Choose a bank statement CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

st.success("CSV uploaded successfully!")

df = pd.read_csv(uploaded_file)
df = pd.read_csv(uploaded_file)

# Normalize column names: strip whitespace, lowercase, then rename back to expected names
df.columns = df.columns.str.strip()

required_cols = {"transaction_date", "category", "amount"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Your CSV is missing required column(s): {', '.join(missing)}. "
              f"Found columns: {', '.join(df.columns)}")
    st.stop()
with st.expander("Preview uploaded data"):
    st.dataframe(df.head())

# -----------------------------
# Income Detection & Override
# -----------------------------
df_temp = df.copy()
df_temp["transaction_date"] = pd.to_datetime(df_temp["transaction_date"])
salary_df = df_temp[(df_temp["amount"] > 0) & (df_temp["category"].str.lower() == "salary")]

default_income = 0.0
if not salary_df.empty:
    latest_date = salary_df["transaction_date"].max()
    default_income = float(salary_df[salary_df["transaction_date"] == latest_date]["amount"].sum())

st.subheader("Income Verification")
user_income = st.number_input("Detected Monthly Income (Edit if necessary):", value=default_income, step=100.0)

# -----------------------------
# Analyze Button
# -----------------------------
if st.button("Analyze Expenses", type="primary"):

    try:
        with st.spinner("Processing your expenses..."):
            results = process_expenses(df)
            backtest = evaluate_forecast(df)

        st.success("Analysis completed!")

        # -----------------------------
        # Summary Metrics
        # -----------------------------
        st.subheader("Summary")

        projected_savings = user_income - results["forecast"]
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Next Month Forecast", f"\u20B9{results['forecast']:,.0f}")
        with col2:
            st.metric("Projected Savings", f"\u20B9{projected_savings:,.0f}")
        with col3:
            st.metric("Total Spending", f"\u20B9{results['total_spending']:,.0f}")
        with col4:
            st.metric("Highest Category", results["highest_category"])

        if results["forecast"] > user_income:
            st.error(
                f"\U0001F6A8 **Budget Alert!** Your forecasted expenses "
                f"(\u20B9{results['forecast']:,.0f}) exceed your monthly income (\u20B9{user_income:,.0f})."
            )
        else:
            st.success(
                f"\u2705 **Looking Good!** Based on your forecast, you are on track to save "
                f"\u20B9{projected_savings:,.0f} next month."
            )

        st.divider()

        # -----------------------------
        # Monthly Spending (Side-by-Side)
        # -----------------------------
        st.subheader("Monthly Spending Breakdown")

        monthly_df = results["monthly_breakdown"]
        col_table, col_chart = st.columns([1, 2])

        with col_table:
            st.dataframe(monthly_df, use_container_width=True, hide_index=True)

        with col_chart:
            if "Month" in monthly_df.columns:
                chart_df = monthly_df.set_index("Month")
                st.bar_chart(chart_df, color="#C9A227")

        st.divider()

        # -----------------------------
        # Category Spending (Side-by-Side)
        # -----------------------------
        st.subheader("Category-wise Spending")

        category_df = results["category_breakdown"]
        col_table2, col_chart2 = st.columns([1, 2])

        with col_table2:
            st.dataframe(category_df, use_container_width=True, hide_index=True)

        with col_chart2:
            if "Category" in category_df.columns:
                st.bar_chart(category_df.set_index("Category"), color="#3FA796")

        st.divider()

        # -----------------------------
        # Trending Categories
        # -----------------------------
        st.subheader("Trending Up")

        if results["trending_up"]:
            st.warning(
                "These categories show a rising trend and are worth keeping an eye on: "
                + ", ".join(results["trending_up"])
            )
        else:
            st.info("No category shows a clear rising trend yet (need at least 3 months of data per category).")

        st.divider()

        # -----------------------------
        # Forecast Accuracy (Backtest) -- merged in from test.py
        # -----------------------------
        st.subheader("Forecast Accuracy (Backtest)")

        if backtest is None:
            st.info("Upload at least 2 months of data to see a forecast accuracy check.")
        else:
            st.write(
                f"Trained on all months before **{backtest['test_month']}**, then checked the "
                f"prediction against what was actually spent that month."
            )
            acc_col1, acc_col2 = st.columns(2)
            with acc_col1:
                st.markdown("**Weighted average (used for the forecast above)**")
                st.metric("MAE", f"\u20B9{backtest['weighted']['mae']:,.0f}")
                st.metric("RMSE", f"\u20B9{backtest['weighted']['rmse']:,.0f}")
                st.metric("MAPE", f"{backtest['weighted']['mape']:.1f}%")
            with acc_col2:
                st.markdown("**Plain average (old baseline, for comparison)**")
                st.metric("MAE", f"\u20B9{backtest['baseline']['mae']:,.0f}")
                st.metric("RMSE", f"\u20B9{backtest['baseline']['rmse']:,.0f}")
                st.metric("MAPE", f"{backtest['baseline']['mape']:.1f}%")

        st.divider()

        # -----------------------------
        # Download Results
        # -----------------------------
        st.subheader("Download Results")

        download_df = monthly_df.copy()
        csv = download_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Monthly Report",
            data=csv,
            file_name="expense_forecast_results.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
