import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gold Signal OCR", layout="wide")
st.title("📈 Gold Trading Signal Analyzer + OCR Screenshot Reader")
st.write("Upload a screenshot or CSV. This is for educational/simulated signals only.")

# --------------------
# OCR / image upload
# --------------------
st.header("📷 Upload Screenshot (PNG/JPG)")
uploaded_image = st.file_uploader("Upload a trading screenshot", type=["png", "jpg", "jpeg"], key="img")

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Screenshot", use_column_width=True)
        text = pytesseract.image_to_string(image)
        st.subheader("OCR extracted text (raw):")
        st.text(text[:1000])  # show only first 1000 chars
    except Exception as e:
        st.error(f"OCR failed: {e}")

# --------------------
# CSV upload and signals
# --------------------
st.header("📂 Upload CSV (must contain columns: Date, Close)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.write("### Data preview")
    st.dataframe(df.tail())

    # verify columns
    if "Close" not in df.columns:
        st.error("CSV must include a 'Close' column (case-sensitive).")
        st.stop()

    # compute indicators (safe: handle small datasets)
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["SMA_30"] = df["Close"].rolling(window=30, min_periods=1).mean()

    # RSI (simple implementation)
    delta = df["Close"].diff().fillna(0)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs)).fillna(50)

    # signals
    df["Signal"] = np.where(
        (df["SMA_10"] > df["SMA_30"]) & (df["RSI"] < 70), "BUY",
        np.where((df["SMA_10"] < df["SMA_30"]) & (df["RSI"] > 30), "SELL", "HOLD")
    )

    st.write("### Generated signals (last 10 rows)")
    st.dataframe(df[["Date", "Close", "SMA_10", "SMA_30", "RSI", "Signal"]].tail(10))

    # plot
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df["Date"], df["Close"], label="Close")
    ax.plot(df["Date"], df["SMA_10"], label="SMA 10")
    ax.plot(df["Date"], df["SMA_30"], label="SMA 30")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # download CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download signals CSV", data=csv_bytes, file_name="signals.csv", mime="text/csv")




