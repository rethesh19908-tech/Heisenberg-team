import streamlit as st
import pandas as pd
from datetime import date

# -----------------------------
# CSV VALIDATION
# -----------------------------
REQUIRED_COLUMNS = {"transaction_date", "amount", "description"}

def validate_csv(df):
    """Returns a list of missing required columns (empty list = valid)."""
    return sorted(REQUIRED_COLUMNS - set(df.columns))

# -----------------------------
# AUTO-CATEGORIZATION (fallback when 'category' is missing/blank)
# -----------------------------
CATEGORY_KEYWORDS = {
    "Food": ["restaurant", "swiggy", "zomato", "cafe", "food"],
    "Groceries": ["grocery", "supermarket", "bigbasket", "dmart", "reliance fresh"],
    "Shopping": ["amazon", "flipkart", "myntra", "mall"],
    "Transport": ["uber", "ola", "petrol", "fuel", "metro"],
    "Utilities": ["electricity", "water bill", "recharge", "broadband", "wifi"],
    "Entertainment": ["netflix", "spotify", "movie", "bookmyshow"],
    "Healthcare": ["pharmacy", "hospital", "clinic", "medical"],
    "Salary": ["salary", "payroll"],
}

def auto_categorize(description):
    """Guesses a category from the transaction description text."""
    text = str(description).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "Other"

def ensure_categories(df):
    """Fills in a 'category' column using auto_categorize() if it's missing or has blanks."""
    df = df.copy()
    if "category" not in df.columns:
        df["category"] = df["description"].apply(auto_categorize)
    elif df["category"].isna().any():
        blank_mask = df["category"].isna()
        df.loc[blank_mask, "category"] = df.loc[blank_mask, "description"].apply(auto_categorize)
    return df

# -----------------------------
# BACKEND LOGIC (unchanged)
# -----------------------------
def process_expenses(df):
    """Processes the uploaded CSV dataframe using Pandas instead of MySQL."""
    df = ensure_categories(df)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['Year'] = df['transaction_date'].dt.year
    df['Month'] = df['transaction_date'].dt.month
    df['Month_Year'] = df['transaction_date'].dt.to_period('M')

    expenses = df[df['amount'] < 0].copy()
    expenses['amount'] = -expenses['amount']

    total_spending = float(expenses['amount'].sum())

    cat_totals = expenses.groupby('category')['amount'].sum()
    highest_category = str(cat_totals.idxmax()) if not cat_totals.empty else "None"

    category_breakdown = cat_totals.reset_index()
    category_breakdown.columns = ['Category', 'Total Spent']

    monthly_totals = expenses.groupby('Month_Year')['amount'].sum().reset_index()
    monthly_totals['Month_Year'] = monthly_totals['Month_Year'].astype(str)
    monthly_totals.columns = ['Month', 'Total Spent']

    category_monthly = expenses.groupby(['category', 'Month_Year'])['amount'].sum().unstack(fill_value=0)
    forecast_val = float(category_monthly.mean(axis=1).sum())

    trending_up = []
    for cat, row in category_monthly.iterrows():
        vals = row.values
        if len(vals) >= 2:
            is_increasing = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
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
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,500&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink:        #0B0E14;
    --surface:    #141922;
    --surface-2:  #1B2130;
    --hairline:   #2B3243;
    --gold:       #C9A227;
    --gold-soft:  rgba(201, 162, 39, 0.14);
    --emerald:    #3FA796;
    --emerald-soft: rgba(63, 167, 150, 0.14);
    --rose:       #C1666B;
    --rose-soft:  rgba(193, 102, 107, 0.14);
    --text:       #ECE9E2;
    --text-muted: #8B93A7;
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #151B27 0%, var(--ink) 55%);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--hairline);
}
section[data-testid="stSidebar"] .block-container { padding-top: 2.2rem; }

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
    font-size: 2.7rem !important;
    color: var(--text) !important;
    letter-spacing: -0.01em;
    margin-bottom: 0.2rem !important;
}
h1 em {
    font-style: italic;
    color: var(--gold);
}

.hero-sub {
    color: var(--text-muted);
    font-size: 1rem;
    max-width: 640px;
    line-height: 1.55;
}

.hero-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 1rem;
}
.hero-stamp {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    border: 1px solid var(--hairline);
    border-radius: 100px;
    padding: 0.35rem 0.9rem;
    white-space: nowrap;
}
.hero-stamp b { color: var(--gold); font-weight: 500; }

.brand-rule {
    height: 1px;
    width: 100%;
    margin: 1.4rem 0 2.2rem 0;
    background: linear-gradient(90deg, var(--gold) 0%, var(--hairline) 35%, transparent 100%);
}

h2, h3 {
    font-family: 'Fraunces', serif !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    letter-spacing: -0.005em;
}
h3 {
    font-size: 1.0rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em !important;
    color: var(--gold) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] h3 {
    font-size: 0.78rem !important;
    color: var(--text) !important;
}

/* -------- File uploader -------- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface-2) !important;
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
    width: 100%;
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

/* -------- Bordered section containers -------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--hairline) !important;
    border-radius: 14px !important;
    background: linear-gradient(180deg, var(--surface) 0%, rgba(20,25,34,0.6) 100%);
}

/* -------- Metrics as ledger cards -------- */
div[data-testid="stMetric"] {
    background: var(--surface-2);
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
    font-size: 0.7rem !important;
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

/* -------- Alerts -------- */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid var(--hairline) !important;
    font-family: 'Inter', sans-serif !important;
}

hr { border-color: var(--hairline) !important; margin: 1.8rem 0 !important; }

details {
    background: var(--surface) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 10px !important;
}

.stSpinner > div { color: var(--gold) !important; }

/* -------- Signature element: budget seal -------- */
.seal-wrap { display: flex; justify-content: center; margin: 0.4rem 0 1.6rem 0; }
.seal {
    width: 116px; height: 116px;
    border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    text-align: center;
    animation: sealIn 0.5s ease-out;
}
.seal-ok { border: 2px solid var(--emerald); background: var(--emerald-soft); color: var(--emerald); }
.seal-warn { border: 2px solid var(--rose); background: var(--rose-soft); color: var(--rose); }
.seal .seal-title { font-size: 0.62rem; opacity: 0.85; }
.seal .seal-amount { font-size: 0.95rem; font-weight: 600; margin-top: 0.15rem; font-family: 'Fraunces', serif; letter-spacing: 0; }

/* -------- Trend pills -------- */
.pill-row { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.4rem 0.85rem;
    border-radius: 100px;
    border: 1px solid var(--rose);
    color: var(--rose);
    background: var(--rose-soft);
}
.pill::before { content: "▲ "; }

@keyframes sealIn {
    from { opacity: 0; transform: scale(0.85); }
    to { opacity: 1; transform: scale(1); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
div[data-testid="stVerticalBlockBorderWrapper"] { animation: fadeInUp 0.4s ease-out; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Hero
# -----------------------------
st.markdown('<div class="brand-eyebrow">Statement &amp; Forecast · Private Ledger</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="hero-row">
        <div>
            <h1>Expense <em>Forecasting</em> System</h1>
            <p class="hero-sub">Upload a bank statement to see monthly spending, category
            breakdowns, and a forecast of next month's expenses.</p>
        </div>
        <div class="hero-stamp">Rendered <b>{date.today().strftime('%d %b %Y')}</b></div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="brand-rule"></div>', unsafe_allow_html=True)

# -----------------------------
# Sidebar: Statement Setup
# -----------------------------
with st.sidebar:
    st.markdown("### Statement Setup")
    uploaded_file = st.file_uploader("Bank statement CSV", type=["csv"])

    df = None
    user_income = 0.0

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        missing_cols = validate_csv(df)
        if missing_cols:
            st.error(
                f"This CSV is missing required column(s): {', '.join(missing_cols)}. "
                "Please upload a statement with transaction_date, amount, and description."
            )
            st.stop()

        df = ensure_categories(df)

        with st.expander("Preview data"):
            st.dataframe(df.head(), hide_index=True)

        df_temp = df.copy()
        df_temp['transaction_date'] = pd.to_datetime(df_temp['transaction_date'])
        salary_df = df_temp[(df_temp['amount'] > 0) & (df_temp['category'].str.lower() == 'salary')]

        default_income = 0.0
        if not salary_df.empty:
            latest_date = salary_df['transaction_date'].max()
            default_income = float(salary_df[salary_df['transaction_date'] == latest_date]['amount'].sum())

        st.markdown("### Monthly Income")
        user_income = st.number_input(
            "Detected from latest salary — edit if needed",
            value=default_income,
            step=100.0
        )

        analyze_clicked = st.button("Analyze Expenses", type="primary")
    else:
        st.info("Upload a CSV to begin.")
        analyze_clicked = False

# -----------------------------
# Main area
# -----------------------------
if uploaded_file is None:
    st.markdown(
        '<p class="hero-sub">Waiting on a statement — use the panel on the left '
        'to upload a CSV and set your income.</p>',
        unsafe_allow_html=True
    )
    st.stop()

if analyze_clicked:
    try:
        with st.spinner("Processing your expenses..."):
            results = process_expenses(df)

        forecast = results["forecast"]
        projected_savings = user_income - forecast
        over_budget = forecast > user_income

        # -----------------------------
        # Budget seal + summary metrics
        # -----------------------------
        with st.container(border=True):
            seal_class = "seal-warn" if over_budget else "seal-ok"
            seal_title = "Over Budget" if over_budget else "On Track"
            st.markdown(
                f"""
                <div class="seal-wrap">
                    <div class="seal {seal_class}">
                        <span class="seal-title">{seal_title}</span>
                        <span class="seal-amount">₹{abs(projected_savings):,.0f}</span>
                        <span class="seal-title">{'over' if over_budget else 'saved'}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Next Month Forecast", f"₹{forecast:,.0f}")
            col2.metric("Projected Savings", f"₹{projected_savings:,.0f}")
            col3.metric("Total Spending", f"₹{results['total_spending']:,.0f}")
            col4.metric("Highest Category", results["highest_category"])

            if over_budget:
                st.error(
                    f"Forecasted expenses (₹{forecast:,.0f}) exceed monthly income "
                    f"(₹{user_income:,.0f})."
                )
            else:
                st.success(
                    f"On track to save ₹{projected_savings:,.0f} next month."
                )

        # -----------------------------
        # Monthly spending
        # -----------------------------
        with st.container(border=True):
            st.subheader("Monthly Spending")
            monthly_df = results["monthly_breakdown"]
            col_table, col_chart = st.columns([1, 2])
            with col_table:
                st.dataframe(monthly_df, use_container_width=True, hide_index=True)
            with col_chart:
                if "Month" in monthly_df.columns:
                    st.bar_chart(monthly_df.set_index("Month"), color="#C9A227")

        # -----------------------------
        # Category spending
        # -----------------------------
        with st.container(border=True):
            st.subheader("Category-wise Spending")
            category_df = results["category_breakdown"]
            col_table2, col_chart2 = st.columns([1, 2])
            with col_table2:
                st.dataframe(category_df, use_container_width=True, hide_index=True)
            with col_chart2:
                if "Category" in category_df.columns:
                    st.bar_chart(category_df.set_index("Category"), color="#3FA796")

        # -----------------------------
        # Trending up categories (restored)
        # -----------------------------
        with st.container(border=True):
            st.subheader("Trending Up")
            if results["trending_up"]:
                pills = "".join(f'<span class="pill">{c}</span>' for c in results["trending_up"])
                st.markdown(f'<div class="pill-row">{pills}</div>', unsafe_allow_html=True)
                st.caption("These categories increased every month in the uploaded data.")
            else:
                st.success("No categories show a consistent upward trend.")

        # -----------------------------
        # Download
        # -----------------------------
        with st.container(border=True):
            st.subheader("Download Report")
            csv = monthly_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Monthly Report",
                data=csv,
                file_name="expense_forecast_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.markdown(
        '<p class="hero-sub">Statement loaded. Click <b>Analyze Expenses</b> in the '
        'sidebar to generate the forecast.</p>',
        unsafe_allow_html=True
    )
