# streamlit_app.py
"""
FinBuddy - Streamlit prototype (with robust amount parsing for Indian-formatted CSVs)
Drop-in replacement for your existing streamlit_app.py
"""

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
sns.set_theme(style="whitegrid")

# ----------------------------
# Utilities
# ----------------------------
CURRENCY_RE = re.compile(r'[^\d\-\.\,()]')

def normalize_date_str(s):
    if pd.isna(s) or s is None:
        return None
    try:
        return dateparser.parse(str(s), dayfirst=True, fuzzy=True).date().isoformat()
    except:
        return None

def extract_amount_tokens(s):
    """
    Return a list of numeric-like tokens found in a raw string.
    """
    if not isinstance(s, str):
        return []
    # pick tokens like 1,23,456.78 or 123456.78 or (1,234.00)
    tokens = re.findall(r'-?\(?[0-9\.,]+\)?', s)
    return tokens

def clean_numeric_token(tok):
    """
    Convert a token that may contain Indian commas and parentheses into float.
    Returns float or np.nan.
    """
    if tok is None:
        return np.nan
    t = str(tok).strip()
    # parentheses indicate negative amounts like (100.00)
    if t.startswith('(') and t.endswith(')'):
        t = '-' + t[1:-1]
    # remove currency symbols and letters
    t = re.sub(r'[^\d\.\-\,]', '', t)
    # remove commas (Indian or thousand separators)
    t = t.replace(',', '')
    if t == '' or t in ['.', '-', '-.', '.-']:
        return np.nan
    try:
        return float(t)
    except:
        return np.nan

def parse_amount_from_row(row, amount_col_candidates=None):
    """
    Given a pandas Series (row), attempt to return a numeric amount float.
    Logic:
      1) If amount_col_candidates provided, try those columns first (clean them)
      2) Otherwise inspect columns 'amount', 'amt', 'amount_raw', 'raw_line' etc.
      3) If value contains DR/CR or trailing tokens, handle sign accordingly.
      4) If still not found, attempt to extract last numeric token from raw_line/description.
    """
    # 1) try provided columns
    if amount_col_candidates:
        for c in amount_col_candidates:
            if c in row.index and pd.notna(row[c]):
                val = row[c]
                v = clean_numeric_token(val)
                if not np.isnan(v):
                    return v

    # 2) check common column names
    for c in ['amount_num', 'amount', 'amt', 'debit', 'credit', 'value', 'amount_raw', 'raw_amount']:
        if c in row.index and pd.notna(row[c]):
            v = clean_numeric_token(row[c])
            if not np.isnan(v):
                return v

    # 3) inspect raw_line/description for numeric tokens and DR/CR
    raw_candidates = []
    for c in ['raw_line', 'description', 'narration', 'particulars', 'details', 'merchant_raw']:
        if c in row.index:
            raw_candidates.append(str(row[c]))

    joined = " | ".join([x for x in raw_candidates if x and x != 'None'])
    if joined:
        tokens = extract_amount_tokens(joined)
        if tokens:
            # take last token as amount candidate (common pattern)
            amt_token = tokens[-1]
            amt_val = clean_numeric_token(amt_token)
            # detect DR/CR context - if 'DR' appears near token, it's debit (negative); if 'CR' it's positive
            # We'll inspect trailing words
            trailing = joined.split(amt_token)[-1][:10].upper() if amt_token in joined else ''
            leading = joined.split(amt_token)[0][-10:].upper() if amt_token in joined else ''
            sign = 1
            if re.search(r'\bDR\b', trailing) or re.search(r'\bDR\b', leading) or ' DR' in joined.upper():
                sign = -1
            if re.search(r'\bCR\b', trailing) or re.search(r'\bCR\b', leading) or ' CR' in joined.upper():
                sign = 1
            if not np.isnan(amt_val):
                return sign * amt_val

    return np.nan

def guess_columns(df):
    """
    Return candidate column names for date, amount, description based on header heuristics
    """
    date_col = None
    amount_cols = []
    desc_col = None
    for c in df.columns:
        lc = c.lower()
        if 'date' in lc and date_col is None:
            date_col = c
        if any(x in lc for x in ('amount','amt','debit','credit','value')) and c not in amount_cols:
            amount_cols.append(c)
        if any(x in lc for x in ('desc','narration','particular','remark','details','merchant')) and desc_col is None:
            desc_col = c
    return date_col, amount_cols, desc_col

def deduce_merchant(desc):
    if not isinstance(desc, str) or desc.strip()=='':
        return ''
    s = desc.strip()
    # simple common patterns
    # UPI/NAME or UPI/NAME/...
    m = re.search(r'UPI/([^/,\n]+)', s, flags=re.I)
    if m:
        return m.group(1).strip().title()
    # patterns like "S LOKESH" or "SHREE SHANKARAN/391..."
    m2 = re.search(r'([A-Z][A-Z ]{2,})', s)
    if m2:
        return m2.group(1).strip().title()
    # fallback: first two words
    parts = re.split(r'[,/]', s)
    first = parts[0].strip()
    # clean trailing tokens like 'Sent using Payt'
    first = re.sub(r'Sent using.*', '', first, flags=re.I).strip()
    return first.title()

def assign_category(merchant, desc):
    # simple rule-based mapping - extend as needed
    m = (merchant or '').upper()
    d = (desc or '').upper()
    if any(x in m or x in d for x in ['BLINKIT','GROCERY','KUNCHUM','STORE']):
        return 'Groceries'
    if any(x in m or x in d for x in ['SHREE','CHEFMASTER','NANDHANA','RESTAURANT','FOOD','DINE','ZOMATO','SWIGGY']):
        return 'Food & Dining'
    if any(x in m or x in d for x in ['SCOTCH','BAR','DENNYS','DRINK']):
        return 'Drinks'
    if 'UDEMY' in m or 'UDEMY' in d:
        return 'Education'
    if 'GOOGLE' in m or 'CLOUD' in d:
        return 'Software'
    if 'CRED' in m or 'CREDIT' in d:
        return 'Creditcard Payment'
    if 'OID' in m or 'SIP' in d or 'NACH' in d:
        # if it's a debit and NACH -> investment, if credit -> dividend; we'll set 'Investments' and handle later
        return 'Investments'
    if 'OPENING' in d or 'CLOSING' in d:
        return 'Ignore'
    return 'Other'

# ----------------------------
# UI / App
# ----------------------------
st.title("FinBuddy — Personal Finance Dashboard")
st.caption("Upload a CSV or PDF (CSV strongly recommended). This version includes robust Indian-format amount parsing.")

uploaded = st.file_uploader("Upload CSV (bank statement) or PDF", type=["csv","pdf"])
if not uploaded:
    st.info("Upload a CSV exported from your bank (recommended).")
    st.stop()

raw_bytes = uploaded.read()

# ---------- Load into a dataframe ----------
df_raw = None
if uploaded.type == "application/pdf" or uploaded.name.lower().endswith('.pdf'):
    try:
        import pdfplumber
        rows = []
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for p in pdf.pages:
                text = p.extract_text() or ''
                for line in text.splitlines():
                    # heuristic - skip short lines
                    if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', line) and re.search(r'\d', line):
                        rows.append({'raw_line': line})
        if rows:
            df_raw = pd.DataFrame(rows)
        else:
            st.warning("PDF parsing didn't find rows - upload CSV if possible.")
            st.stop()
    except Exception as e:
        st.error("PDF parsing requires pdfplumber. Try uploading CSV instead. Error: " + str(e))
        st.stop()
else:
    # try many encodings & delimiters
    parsed = None
    tried = []
    for enc in ('utf-8','latin1','cp1252'):
        try:
            text = raw_bytes.decode(enc)
        except:
            continue
        import csv
        # try to detect delimiter
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(text[:4096])
            sep = dialect.delimiter
        except Exception:
            sep = None
        candidates = [sep, ',', ';', '\t', '|'] if sep else [',',';','\t','|']
        for csep in candidates:
            try:
                df_candidate = pd.read_csv(io.StringIO(text), sep=csep, engine='python', dtype=str, keep_default_na=False, na_values=[""])
                parsed = df_candidate
                break
            except Exception as e:
                tried.append((enc, csep, str(e)))
        if parsed is not None:
            break
    if parsed is None:
        st.error("Failed to parse CSV. Try re-exporting the CSV from your bank. Parsing attempts:\n" + json.dumps(tried[:5]) if 'json' in globals() else "Failed")
        st.stop()
    df_raw = parsed

# ----------------------------
# Detect columns and parse rows
# ----------------------------
date_col, amount_col_candidates, desc_col = guess_columns(df_raw)

# fallback if no desc column found
if desc_col is None:
    # prefer first textual column
    for c in df_raw.columns:
        if c != date_col and c not in amount_col_candidates:
            desc_col = c
            break

records = []
for idx, row in df_raw.iterrows():
    row = row.fillna('')
    # compose raw_line for extra context
    raw_line = " | ".join([str(row[c]) for c in df_raw.columns if str(row[c]).strip()!=''])
    date_val = None
    if date_col:
        date_val = normalize_date_str(row[date_col])
    # parse amount robustly
    amt = parse_amount_from_row(row, amount_col_candidates)
    # If amt is NaN, try extracting from raw_line
    if np.isnan(amt):
        tokens = extract_amount_tokens(raw_line)
        if tokens:
            amt = clean_numeric_token(tokens[-1])
    desc = row[desc_col] if desc_col in row.index else raw_line
    merchant = deduce_merchant(desc)
    cat = assign_category(merchant, desc)

    records.append({
        'transaction_id': str(uuid.uuid4()),
        'date': date_val,
        'amount_num': amt,
        'description': str(desc),
        'merchant': merchant,
        'raw_line': raw_line,
        'category': cat
    })

df = pd.DataFrame(records)

# ensure numeric type
df['amount_num'] = pd.to_numeric(df['amount_num'], errors='coerce')

st.success(f"Parsed {len(df)} rows.")
st.write("Preview (first 30 rows):")
st.dataframe(df[['date','merchant','category','amount_num','description']].head(30))

# store for chat/interaction
st.session_state['transactions'] = df

# ----------------------------
# Charts (safe guards)
# ----------------------------
st.header("Charts")
df_chart = df[df['category'] != 'Ignore'].copy()
df_chart['date_dt'] = pd.to_datetime(df_chart['date'], errors='coerce')
df_chart = df_chart.dropna(subset=['date_dt'])
if df_chart.empty:
    st.warning("No dated transactions available for charts. Check the preview above.")
else:
    df_chart['month'] = df_chart['date_dt'].dt.to_period('M').astype(str)

    # Monthly net
    monthly = df_chart.groupby('month')['amount_num'].sum().reset_index()
    if monthly['amount_num'].dropna().empty or monthly['amount_num'].abs().sum()==0:
        st.warning("No numeric monthly data to plot.")
    else:
        fig, ax = plt.subplots(figsize=(10,4))
        sns.lineplot(data=monthly, x='month', y='amount_num', marker='o', ax=ax)
        ax.set_title("Monthly Net Flow")
        ax.set_ylabel("Amount")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Category donut
    cat_sum = df_chart.groupby('category')['amount_num'].sum().sort_values(ascending=False)
    if cat_sum.abs().sum() == 0 or cat_sum.empty:
        st.warning("Category chart has no numeric data.")
    else:
        fig2, ax2 = plt.subplots(figsize=(6,6))
        wedges, texts, autotexts = ax2.pie(cat_sum.values, labels=cat_sum.index, autopct='%1.1f%%', startangle=140)
        centre = plt.Circle((0,0),0.70,fc='white')
        fig2.gca().add_artist(centre)
        ax2.set_title("Category Share")
        st.pyplot(fig2)

    # Top merchants
    top_merchants = df_chart.groupby('merchant')['amount_num'].sum().abs().sort_values(ascending=False).head(20)
    if top_merchants.empty or top_merchants.sum()==0:
        st.warning("No merchant spend to plot.")
    else:
        fig3, ax3 = plt.subplots(figsize=(8,6))
        top_merchants.plot(kind='barh', ax=ax3)
        ax3.set_title("Top Merchants by Spend (abs value)")
        st.pyplot(fig3)

    # Treemap
    tree_data = df_chart.groupby(['category','merchant'])['amount_num'].sum().abs().reset_index()
    if not tree_data.empty and tree_data['amount_num'].sum() > 0:
        sizes = tree_data['amount_num'].values
        labels = (tree_data['category'] + "\n" + tree_data['merchant']).values
        fig4 = plt.figure(figsize=(14,8))
        squarify.plot(sizes=sizes, label=labels, alpha=0.8)
        plt.axis('off')
        st.pyplot(fig4)

# ----------------------------
# Chat (simple)
# ----------------------------
st.header("Chat (experimental)")
st.write("Ask things like: 'How much did I spend on Food in 2025-11' or 'List NACH investments'")

user_q = st.text_input("Ask a question about these transactions:")

def compute_answer(df_local, q):
    ql = q.lower()
    months = re.findall(r'20\d{2}-\d{2}', ql)
    years = re.findall(r'20\d{2}', ql)
    cat = None
    for c in df_local['category'].unique():
        if c and c.lower() in ql:
            cat = c
            break
    dfq = df_local.copy()
    if months:
        dfq = dfq[dfq['date'].str.startswith(months[0])]
    elif years:
        dfq = dfq[dfq['date'].str.startswith(years[0])]
    if cat:
        dfq = dfq[dfq['category'] == cat]
    total = dfq['amount_num'].sum()
    count = len(dfq)
    return total, count, dfq

if st.button("Ask") and user_q:
    df_local = st.session_state.get('transactions', df)
    total, count, dfq = compute_answer(df_local, user_q)
    prompt = f"User asked: {user_q}\nComputed: total={total:.2f}, count={count}\nPlease produce a short, friendly answer."
    openai_key = st.secrets.get("OPENAI_API_KEY") or None
    if openai_key:
        try:
            openai.api_key = openai_key
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"You are a concise assistant that reports the provided numbers exactly."},
                    {"role":"user","content":prompt}
                ],
                max_tokens=150,
                temperature=0.2
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            answer = f"(LLM failed) Computed total ₹{total:,.2f} across {count} transactions. Error: {e}"
    else:
        answer = f"Computed total ₹{total:,.2f} across {count} transactions. (No OPENAI_API_KEY set in Streamlit secrets.)"
    st.markdown("**Answer:**")
    st.write(answer)
    if count>0:
        with st.expander("Show matching transactions"):
            st.dataframe(dfq[['date','merchant','category','amount_num']].sort_values('date', ascending=False).head(200))

# ----------------------------
# Done
# ----------------------------
st.write("Tip: If charts are empty, download the parsed CSV (use the preview) and inspect amount/raw_line columns for formatting. I can tune the parser if you paste 3 raw lines here.")
