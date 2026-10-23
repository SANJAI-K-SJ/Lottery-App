import os

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Lottery Number Generator", page_icon="🎟️", layout="centered")

MONTH_MAPPING = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Put your full dataset in lottery_data.csv (same folder) with columns:
# Year, Day, Month, Lottery_Number
SAMPLE_DATA = [
    {"Year": 2012, "Day": 1, "Month": "JAN", "Lottery_Number": "479506"},
    {"Year": 2012, "Day": 2, "Month": "JAN", "Lottery_Number": "113374"},
    {"Year": 2012, "Day": 3, "Month": "JAN", "Lottery_Number": "201638"},
    {"Year": 2012, "Day": 4, "Month": "JAN", "Lottery_Number": "524375"},
    {"Year": 2012, "Day": 5, "Month": "JAN", "Lottery_Number": "118696"},
]


@st.cache_resource
def train_model():
    if os.path.exists("lottery_data.csv"):
        df = pd.read_csv("lottery_data.csv", dtype={"Lottery_Number": str})
        using_sample = False
    else:
        df = pd.DataFrame(SAMPLE_DATA)
        using_sample = True

    df = df.dropna(subset=["Lottery_Number"])
    df = df[df["Lottery_Number"] != "******"]
    df["Lottery_Number"] = df["Lottery_Number"].astype(int)
    df["Month_Num"] = df["Month"].str.upper().map(MONTH_MAPPING)
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month_Num"].astype(str) + "-" + df["Day"].astype(str),
        errors="coerce",
    )
    df = df.dropna(subset=["Date"])
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Year_Num"] = df["Year"]

    X = df[["Year_Num", "Month_Num", "Day", "DayOfWeek", "DayOfYear"]]
    y = df["Lottery_Number"]
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    return model, len(df), using_sample


def predict_numbers(model, d, count):
    ts = pd.Timestamp(d)
    X_target = pd.DataFrame([{
        "Year_Num": ts.year,
        "Month_Num": ts.month,
        "Day": ts.day,
        "DayOfWeek": ts.dayofweek,
        "DayOfYear": ts.dayofyear,
    }])
    preds = [tree.predict(X_target.values)[0] for tree in model.estimators_[:count]]
    return [str(int(round(p))).zfill(6) for p in preds]


st.title("🎟️ Lottery Number Generator")
st.caption("Pick a date, choose how many numbers you want, and get them from a Random Forest model trained on past results.")

model, n_rows, using_sample = train_model()
if using_sample:
    st.warning("Using only 5 sample rows. Add `lottery_data.csv` to use your full dataset.")
else:
    st.info(f"Model trained on {n_rows:,} past draws.")

chosen = st.date_input("Select a date", value=pd.Timestamp("2026-11-15"),
                       min_value=pd.Timestamp("2000-01-01"), max_value=pd.Timestamp("2100-12-31"))

count = st.number_input("How many lottery numbers do you want?", min_value=1,
                        max_value=len(model.estimators_), value=10, step=1)

if st.button("Generate numbers", type="primary"):
    numbers = predict_numbers(model, chosen, int(count))
    st.subheader(f"{len(numbers)} numbers for {chosen.strftime('%d %b %Y')}")
    for i in range(0, len(numbers), 5):
        row = numbers[i:i + 5]
        cols = st.columns(5)
        for col, num in zip(cols, row):
            col.metric("", num)

st.divider()
st.caption(
    "⚠️ For entertainment only. Lottery draws are random, so no model can reliably "
    "predict them. These numbers have no better chance of winning than any others."
)
