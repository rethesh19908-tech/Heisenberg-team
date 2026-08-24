import streamlit as st
import pandas as pd

# -----------------------------
# BACKEND LOGIC (Merged)
# -----------------------------
def process_expenses(df):
    """Processes the uploaded CSV dataframe using Pandas instead of MySQL."""
    # Ensure date is datetime type
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['Year'] = df['transaction_date'].dt.year
    df['Month'] = df['transaction_date'].dt.month
    df['Month_Year'] = df['transaction_date'].dt.to_period('M')
    
    # Isolate expenses (where amount is negative) and convert to positive for aggregation
    expenses = df[df['amount'] < 0].copy()
    expenses['amount'] = -expenses['amount']
    
    # Total Spending
    total_spending = float(expenses['amount'].sum())
    
    # Category Totals (All time)
    cat_totals = expenses.groupby('category')['amount'].sum()
    highest_category = str(cat_totals.idxmax()) if not cat_totals.empty else "None"
    
    category_breakdown = cat_totals.reset_index()
    category_breakdown.columns = ['Category', 'Total Spent']
    
    # Monthly Spending Totals
    monthly_totals = expenses.groupby('Month_Year')['amount'].sum().reset_index()
    monthly_totals['Month_Year'] = monthly_totals['Month_Year'].astype(str)
    monthly_totals.columns = ['Month', 'Total Spent']
    
    # Forecast Logic: Calculate the monthly average for each category 
    category_monthly = expenses.groupby(['category', 'Month_Year'])['amount'].sum().unstack(fill_value=0)
    forecast_val = float(category_monthly.mean(axis=1).sum())
    
    # Trend Detection Logic
    trending_up = []
    for cat, row in category_monthly.iterrows():
        vals = row.values
        if len(vals) >= 2: # Need at least 2 months to detect a trend
            # Check if each month's spend is greater than the previous month
            is_increasing = all(vals[i] < vals[i+1] for i in range(len(vals)-1))
            if is_increasing:
                trending_up.append(cat)

    return {
        "forecast": forecast_val,
        "total_spending": total_spending,
        "highest_category": highest_category,
        "monthly_breakdown": monthly_totals,
        "category_breakdown": category_breakdown,
        "trending_up": trending_up
    }

# -----------------------------
# FRONTEND: Page Configuration & CSS
# -----------------------------
st.set_page_config(
    page_title="Personal Expense Forecasting System",
    page_icon="💰",
    layout="wide"
)

st.markdown("""
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

/* App background */
.stApp {
    background: radial-gradient(circle at 15% 0%, #151B27 0%, var(--ink) 55%);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] { background: var(--surface); }

/* Hide default Streamlit chrome for a cleaner, bespoke feel */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* -------- Header / Letterhead -------- */
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

/* -------- File uploader -------- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1px dashed var(--hairline) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--gold) !important;
}

/* -------- Buttons -------- */
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

/* -------- Metrics as ledger cards -------- */
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

/* -------- Dataframes -------- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--hairline);
    border-radius: 10px;
    overflow: hidden;
}

/* -------- Alerts (info / success / warning / error) -------- */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid var(--hairline) !important;
    font-family: 'Inter', sans-serif !important;
}

/* -------- Dividers -------- */
hr {
    border-color: var(--hairline) !important;
    margin: 1.8rem 0 !important;
}

/* -------- Expander -------- */
details {
    background: var(--surface) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 10px !important;
}

/* -------- Spinner text -------- */
.stSpinner > div {
    color: var(--gold) !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="brand-eyebrow">Statement &amp; Forecast · Private Ledger</div>', unsafe_allow_html=True)
st.title("💰 Personal Expense Forecasting System")

st.write(
    "Upload your bank statement (CSV) to analyze monthly spending, "
    "view category-wise expenses, and forecast next month's expenses."
)
st.markdown('<div class="brand-rule"></div>', unsafe_allow_html=True)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose a bank statement CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

st.success("CSV uploaded successfully!")

# Load the dataframe immediately so we can read it for income and the backend
df = pd.read_csv(uploaded_file)

# Optional preview[cite: 1]
with st.expander("Preview uploaded data"):
    st.dataframe(df.head())

# -----------------------------
# Income Detection & Override
# -----------------------------
# Detect latest salary from the dataframe
df_temp = df.copy()
df_temp['transaction_date'] = pd.to_datetime(df_temp['transaction_date'])
salary_df = df_temp[(df_temp['amount'] > 0) & (df_temp['category'].str.lower() == 'salary')]

default_income = 0.0
if not salary_df.empty:
    latest_date = salary_df['transaction_date'].max()
    default_income = float(salary_df[salary_df['transaction_date'] == latest_date]['amount'].sum())

st.subheader("Income Verification")
user_income = st.number_input("Detected Monthly Income (Edit if necessary):", value=default_income, step=100.0)

# -----------------------------
# Analyze Button
# -----------------------------
# -----------------------------
# Analyze Button
# -----------------------------
if st.button("Analyze Expenses", type="primary"):

    try:
        with st.spinner("Processing your expenses..."):
            # Pass the dataframe directly to the combined backend logic
            results = process_expenses(df)

        st.success("Analysis completed!")

        # -----------------------------
        # Summary Metrics
        # -----------------------------
        st.subheader("Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Next Month Forecast", f"₹{results['forecast']:,.0f}")
        with col2:
            st.metric("Total Spending", f"₹{results['total_spending']:,.0f}")
        with col3:
            st.metric("Highest Category", results["highest_category"])

        # --- NEW BUDGET ALERT ---
        if results['forecast'] > user_income:
            st.error(f"🚨 **Budget Alert!** Your forecasted expenses (₹{results['forecast']:,.0f}) exceed your monthly income (₹{user_income:,.0f}).")

        st.divider()

        # -----------------------------
        # Monthly Spending (Side-by-Side)
        # -----------------------------
        st.subheader("Monthly Spending Breakdown")
        
        monthly_df = results["monthly_breakdown"]
        
        # Create columns: Table on the left, Chart on the right
        col_table, col_chart = st.columns([1, 2])
        
        with col_table:
            st.dataframe(monthly_df, use_container_width=True, hide_index=True)
            
        with col_chart:
            if "Month" in monthly_df.columns:
                chart_df = monthly_df.set_index("Month")
                # Apply the gold color from your CSS
                st.bar_chart(chart_df, color="#C9A227") 

        st.divider()

        # -----------------------------
        # Category Spending (Side-by-Side)
        # -----------------------------
        st.subheader("Category-wise Spending")

        category_df = results["category_breakdown"]
        
        # Create columns: Table on the left, Chart on the right
        col_table2, col_chart2 = st.columns([1, 2])

        with col_table2:
            st.dataframe(category_df, use_container_width=True, hide_index=True)
            
        with col_chart2:
            if "Category" in category_df.columns:
                # Apply the emerald color from your CSS for contrast
                st.bar_chart(category_df.set_index("Category"), color="#3FA796")

        st.divider()

        # -----------------------------
        # Trending Categories (Combined Alert)
        # -----------------------------
        # -----------------------------
        # Summary Metrics
        # -----------------------------
        st.subheader("Summary")

        # Changed to 4 columns to fit the new Savings metric
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Next Month Forecast", f"₹{results['forecast']:,.0f}")
        
        with col2:
            # Calculate Projected Savings (Income - Forecast)
            projected_savings = user_income - results['forecast']
            st.metric("Projected Savings", f"₹{projected_savings:,.0f}")

        with col3:
            st.metric("Total Spending", f"₹{results['total_spending']:,.0f}")
            
        with col4:
            st.metric("Highest Category", results["highest_category"])

        # --- BUDGET ALERT ---
        if results['forecast'] > user_income:
            st.error(f"🚨 **Budget Alert!** Your forecasted expenses (₹{results['forecast']:,.0f}) exceed your monthly income (₹{user_income:,.0f}).")
        else:
            st.success(f"✅ **Looking Good!** Based on your forecast, you are on track to save ₹{projected_savings:,.0f} next month.")

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
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
