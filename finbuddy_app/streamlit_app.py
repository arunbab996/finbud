import streamlit as st
import pandas as pd
import numpy as np
import io, re, uuid
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from dateutil import parser as dateparser
import openai

st.set_page_config(page_title="FinBuddy", layout="wide")

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def normalize_amount(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    s = s.replace("₹", "").replace(",", "")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s)
    except:
        return np.nan

def deduce_merchant(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()

    # Try UPI/NAME format
    m = re.search(r"UPI/([^/,\n]+)", text, flags=re.I)
    if m:
        return m.group(1).strip().title()

    # Try simple word extraction
    cleaned = re.sub(r"[^A-Za-z0-9 @.&]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    parts = cleaned.split()
    if len(parts) >= 1:
        return " ".join(parts[:2]).title()
    return cleaned.title()


CATEGORY_RULES = {
    "Food & Dining": ["RESTAURANT", "CAFE", "FOOD", "DINING", "SHANKARAN", "NANDHANA"],
    "Groceries": ["GROCERY", "BLINKIT"],
    "Drinks": ["SCOTCH", "BAR"],
    "Education": ["UDEMY"],
    "Home Care": ["URBAN COMPANY"],
    "Software": ["GOOGLE CLOUD"],
    "Water": ["DRINKPRIME"],
    "Investments": ["OID", "SIP"],
    "Credit Card": ["CRED"],
}

def assign_category(merchant, desc):
    M = merchant.upper()
    D = desc.upper()
    for cat, kws in CATEGORY_RULES.items():
        for kw in kws:
            if kw in M or kw in D:
                return cat
    if "OPENING BALANCE" in D or "CLOSING BALANCE" in D:
        return "Ignore"
    return "Other"


def normalize_date(s):
    if pd.isna(s):
        return None
    try:
        return dateparser.parse(str(s), dayfirst=True, fuzzy=True).date()
    except:
        return None


def robust_read_csv(file_bytes):
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            txt = file_bytes.decode(enc)
            df = pd.read_csv(io.StringIO(txt), engine="python", dtype=str)
            return df
        except:
            pass
    raise ValueError("CSV could not be parsed")


# -------------------------------------------------------
# UI
# -------------------------------------------------------

st.title("📊 FinBuddy — Personal Finance Dashboard")

uploaded = st.file_uploader("Upload your bank CSV", type=["csv"])

if not uploaded:
    st.info("Upload a CSV exported from your bank.")
    st.stop()

# -------------------------------------------------------
# Parse CSV
# -------------------------------------------------------

raw_bytes = uploaded.read()

try:
    df = robust_read_csv(raw_bytes)
except Exception as e:
    st.error(f"CSV Read Error: {e}")
    st.stop()

# Detect columns
col_date = None
col_amount = None
col_desc = None

for c in df.columns:
    lc = c.lower()
    if "date" in lc and col_date is None:
        col_date = c
    if ("amount" in lc or "debit" in lc or "credit" in lc) and col_amount is None:
        col_amount = c
    if ("desc" in lc or "narration" in lc or "details" in lc or "particular" in lc) and col_desc is None:
        col_desc = c

if col_desc is None:
    col_desc = df.columns[0]

parsed = []
for _, row in df.iterrows():
    d = normalize_date(row[col_date]) if col_date else None
    a = normalize_amount(row[col_amount]) if col_amount else None
    desc = str(row[col_desc])
    merchant = deduce_merchant(desc)
    cat = assign_category(merchant, desc)

    parsed.append({
        "id": str(uuid.uuid4()),
        "date": d,
        "amount": a,
        "description": desc,
        "merchant": merchant,
        "category": cat
    })

df2 = pd.DataFrame(parsed)

st.success(f"Parsed {len(df2)} transactions.")

st.dataframe(df2.head(20))

# -------------------------------------------------------
# Charts
# -------------------------------------------------------

dfc = df2[df2["category"] != "Ignore"].copy()
dfc = dfc.dropna(subset=["date"])

if dfc.empty:
    st.warning("No valid dated transactions for charts.")
else:
    dfc["month"] = dfc["date"].dt.to_period("M").astype(str)

    st.header("📈 Monthly Trend")
    monthly = dfc.groupby("month")["amount"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8,4))
    sns.lineplot(data=monthly, x="month", y="amount", marker="o", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    st.header("🍩 Category Split")
    cat_sum = dfc.groupby("category")["amount"].sum()
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.pie(cat_sum.values, labels=cat_sum.index, autopct='%1.1f%%')
    st.pyplot(fig2)

    st.header("🏪 Top Merchants")
    merch = dfc.groupby("merchant")["amount"].sum().abs().sort_values(ascending=False).head(15)
    fig3, ax3 = plt.subplots(figsize=(8,6))
    merch.plot(kind="barh", ax=ax3)
    st.pyplot(fig3)

# -------------------------------------------------------
# Chat
# -------------------------------------------------------

st.header("💬 Chat with your finances")

q = st.text_input("Ask something like: 'How much did I spend on Food in 2025-11?'")

if st.button("Ask") and q:
    df_local = df2.copy()
    tokens = q.lower()

    # Simple filters
    year_month = re.findall(r"20\d{2}-\d{2}", tokens)
    category = None
    for c in df_local["category"].unique():
        if c.lower() in tokens:
            category = c

    filtered = df_local.copy()

    if year_month:
        filtered = filtered[filtered["date"].astype(str).str.startswith(year_month[0])]
    if category:
        filtered = filtered[filtered["category"] == category]

    total = filtered["amount"].sum()
    count = len(filtered)

    prompt = f"User query: {q}\nTotal: {total:.2f}\nCount: {count}\nAnswer politely."

    key = st.secrets.get("OPENAI_API_KEY")
    if key:
        openai.api_key = key
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            ans = resp.choices[0].message["content"]
        except Exception as e:
            ans = f"(LLM failed) total = {total} count={count}. Error: {e}"
    else:
        ans = f"You spent ₹{total:.2f} across {count} transactions."

    st.write(ans)
