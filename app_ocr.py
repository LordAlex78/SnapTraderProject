import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Gold Trading Signal Analyzer")
st.write("Upload a CSV file to get SMA/RSI-based signals (for educational use only).")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file (with columns: Date, Close)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data", df.tail())

    # --- Simple Algorithm ---
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()

    # RSI
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Generate Signals
    df["Signal"] = np.where((df["SMA_10"] > df["SMA_30"]) & (df["RSI"] < 70), "BUY",
                    np.where((df["SMA_10"] < df["SMA_30"]) & (df["RSI"] > 30), "SELL", "HOLD"))

    st.write("### Generated Signals", df[["Date", "Close", "SMA_10", "SMA_30", "RSI", "Signal"]].tail())

    # --- Plot Chart ---
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"], label="Close Price", linewidth=1)
    ax.plot(df["Date"], df["SMA_10"], label="SMA 10", linewidth=1)
    ax.plot(df["Date"], df["SMA_30"], label="SMA 30", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Signals CSV", csv, "signals.csv", "text/csv")



