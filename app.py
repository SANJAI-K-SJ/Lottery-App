import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Lottery Number Generator", page_icon="🎟️", layout="centered")

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
MAXV = 999999.0
WINDOW = 14
ERR = "Avg error (lower is better)"


# ================================================================ data
def load_data(path="lottery_data.csv"):
    """Needs lottery_data.csv with columns: Year, Day, Month, Lottery_Number"""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={"Lottery_Number": str})
    df = df.dropna(subset=["Lottery_Number"])
    df = df[df["Lottery_Number"].astype(str).str.fullmatch(r"\d+")]
    df["Lottery_Number"] = df["Lottery_Number"].astype(int)
    df["Month_Num"] = df["Month"].astype(str).str.upper().str[:3].map(MONTHS)
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month_Num"].astype(str) + "-" + df["Day"].astype(str),
        errors="coerce",
    )
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df.drop_duplicates("Date", keep="last").reset_index(drop=True)


def feats(dates):
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    return pd.DataFrame({
        "Year_Num": d.dt.year,
        "Month_Num": d.dt.month,
        "Day": d.dt.day,
        "DayOfWeek": d.dt.dayofweek,
        "DayOfYear": d.dt.dayofyear,
    })


def fmt(values):
    v = np.clip(np.round(np.asarray(values, dtype=float)), 0, MAXV).astype(int)
    return [str(x).zfill(6) for x in v]


# ======================================================== random forest
def train_rf(df):
    X = feats(df["Date"]).values
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(
        X, df["Lottery_Number"].values
    )


def rf_generate(model, date, n):
    X = feats([pd.Timestamp(date)]).values
    return fmt([t.predict(X)[0] for t in model.estimators_[:n]])


# ========================================================== time series
def _regular_series(df):
    s = df.set_index("Date")["Lottery_Number"].astype(float)
    return s.asfreq("D").interpolate()


def train_ts(df):
    from statsmodels.tsa.arima.model import ARIMA

    s = _regular_series(df).tail(1500)
    res = ARIMA(s, order=(1, 0, 1)).fit()
    return {"res": res, "last": s.index[-1], "sd": float(np.std(res.resid))}


def ts_generate(m, date, n):
    date = pd.Timestamp(date)
    steps = max(1, (date - m["last"]).days)
    point = float(m["res"].forecast(steps=steps).iloc[-1])
    rng = np.random.default_rng(int(date.strftime("%Y%m%d")))
    return fmt(point + rng.normal(0, m["sd"] * 0.5, n))


# ===================================================================== LSTM
def _build_net():
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.drop = nn.Dropout(0.2)
            self.fc = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.drop(out[:, -1, :]))

    return Net()


def train_lstm(df, epochs=15):
    import torch

    v = (df["Lottery_Number"].values / MAXV).astype("float32")[-3000:]
    if len(v) < WINDOW + 20:
        raise ValueError("LSTM needs at least ~35 rows of data.")
    torch.manual_seed(42)
    X = np.stack([v[i:i + WINDOW] for i in range(len(v) - WINDOW)])[..., None]
    y = v[WINDOW:][:, None]
    X, y = torch.tensor(X), torch.tensor(y)
    net = _build_net()
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    loss_fn = torch.nn.MSELoss()
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(net(X[idx]), y[idx]).backward()
            opt.step()
    return net


def lstm_generate(net, df, date, n):
    import torch

    v = (df["Lottery_Number"].values / MAXV).astype("float32")
    steps = int(np.clip((pd.Timestamp(date) - df["Date"].iloc[-1]).days, 1, 60))
    rng = np.random.default_rng(int(pd.Timestamp(date).strftime("%Y%m%d")))
    win = np.tile(v[-WINDOW:], (n, 1))
    net.train()  # keep dropout on so each of the n runs differs
    with torch.no_grad():
        for _ in range(steps):
            p = net(torch.tensor(win[..., None])).numpy()[:, 0]
            p = np.clip(p + rng.normal(0, 0.02, n), 0, 1).astype("float32")
            win = np.concatenate([win[:, 1:], p[:, None]], axis=1)
    return fmt(win[:, -1] * MAXV)


# ============================================================== backtest
def backtest(df, k=60):
    """Train on all but the last k draws, then score each model on those k draws."""
    train, test = df.iloc[:-k], df.iloc[-k:]
    actual = test["Lottery_Number"].values
    preds, notes = {}, []

    rf = train_rf(train)
    preds["Random Forest"] = rf.predict(feats(test["Date"]).values)

    try:
        from statsmodels.tsa.arima.model import ARIMA

        s_full = _regular_series(df)
        s_tr = _regular_series(train).tail(1500)
        res = ARIMA(s_tr, order=(1, 0, 1)).fit().apply(s_full)
        fit = res.fittedvalues.reindex(test["Date"])
        preds["Time Series (ARIMA)"] = fit.fillna(s_tr.mean()).values
    except Exception as e:
        notes.append(f"Time Series (ARIMA) was skipped: {type(e).__name__}: {e}")

    try:
        import torch

        net = train_lstm(train)
        net.eval()
        v = (df["Lottery_Number"].values / MAXV).astype("float32")
        start = len(df) - k
        X = np.stack([v[i - WINDOW:i] for i in range(start, len(df))])[..., None]
        with torch.no_grad():
            preds["LSTM"] = net(torch.tensor(X)).numpy()[:, 0] * MAXV
    except Exception as e:
        notes.append(f"LSTM was skipped: {type(e).__name__}: {e}")

    rng = np.random.default_rng(0)
    preds["Random guess (baseline)"] = rng.integers(0, 1_000_000, size=k).astype(float)
    # Always guessing the same middle value needs no skill. A model is only useful if it beats this.
    preds["Constant middle guess (baseline)"] = np.full(k, float(train["Lottery_Number"].mean()))

    rows = []
    for name, p in preds.items():
        p = np.clip(np.round(p), 0, MAXV).astype(int)
        rows.append({
            "Model": name,
            ERR: int(np.mean(np.abs(p - actual))),
            "Last-digit hit % (random = 10%)": round(100 * float(np.mean(p % 10 == actual % 10)), 1),
        })
    out = pd.DataFrame(rows).sort_values(ERR).reset_index(drop=True)
    return out, notes


# ================================================================== UI
st.title("🎟️ Lottery Number Generator")
st.caption("Pick a date and how many numbers you want. Three models each generate numbers, "
           "and a test on past draws shows which one scored best.")


@st.cache_data
def get_data():
    return load_data("lottery_data.csv")


df = get_data()
if df is None or len(df) < 100:
    st.error("Add your full `lottery_data.csv` (columns: Year, Day, Month, Lottery_Number, "
             "at least 100 rows) to the same GitHub folder as this app, then reload.")
    st.stop()


@st.cache_resource
def get_rf():
    return train_rf(df)


@st.cache_resource
def get_ts():
    return train_ts(df)


@st.cache_resource
def get_lstm():
    return train_lstm(df)


@st.cache_data
def get_backtest():
    return backtest(df, k=min(60, max(10, len(df) // 5)))


st.info(f"Models trained on {len(df):,} past draws "
        f"({df['Date'].min():%d %b %Y} to {df['Date'].max():%d %b %Y}).")

col1, col2 = st.columns(2)
chosen = col1.date_input("Select a date", value=pd.Timestamp.today() + pd.Timedelta(days=1),
                         min_value=pd.Timestamp("2000-01-01"), max_value=pd.Timestamp("2100-12-31"))
count = col2.number_input("How many numbers per model?", min_value=1, max_value=100, value=10, step=1)

with st.spinner("Testing each model on past draws (first run takes a minute)..."):
    score, notes = get_backtest()
real = score[~score["Model"].str.contains("baseline")]
best = real.iloc[0]["Model"]
naive_err = float(score.loc[score["Model"].str.contains("Constant middle"), ERR].iloc[0])
best_beats_naive = float(real.iloc[0][ERR]) < naive_err * 0.95


def show(numbers):
    for i in range(0, len(numbers), 5):
        cols = st.columns(5)
        for c, n in zip(cols, numbers[i:i + 5]):
            c.code(n, language=None)


if st.button("Generate numbers", type="primary"):
    n = int(count)
    results = {}
    try:
        results["Random Forest"] = rf_generate(get_rf(), chosen, n)
    except Exception as e:
        st.warning(f"Random Forest failed: {e}")
    try:
        results["LSTM"] = lstm_generate(get_lstm(), df, chosen, n)
    except ModuleNotFoundError:
        st.warning("LSTM needs PyTorch, which is not installed.")
    except Exception as e:
        st.warning(f"LSTM failed: {e}")
    try:
        results["Time Series (ARIMA)"] = ts_generate(get_ts(), chosen, n)
    except Exception as e:
        st.warning(f"Time Series failed: {e}")

    st.subheader(f"Numbers for {chosen:%d %b %Y}")
    for name, nums in results.items():
        star = "  ⭐ best on past test" if (name == best and best_beats_naive) else ""
        st.markdown(f"**{name}**{star}")
        show(nums)

st.divider()
st.subheader("Which model was best on past draws?")
st.write("Each model was trained without the most recent draws, then asked to guess them. "
         "Two no-skill baselines (a random guess and a constant middle guess) are included "
         "for comparison.")
st.dataframe(score, hide_index=True)
for note in notes:
    st.warning(note)
if best_beats_naive:
    st.success(f"Best model in this test: **{best}**. It scored better than the no-skill baselines.")
else:
    st.warning(f"**{best}** had the lowest error among the models, but it did not clearly beat a "
               "simple constant guess. That means no model is really predicting these numbers, "
               "so there is no true 'best' model.")

st.caption(
    "⚠️ For entertainment only. Lottery draws are random. If a model's error is close to the "
    "random guess, it is not really predicting anything. The numbers have no better chance of "
    "winning than any others."
)
