from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import yfinance as yf
from app.dashboard_service import analyze, demo_data

st.set_page_config(page_title="BTC Market Intelligence", page_icon="📊", layout="wide")
st.title("📊 BTC Market Intelligence")
st.caption("ทดสอบ AI กับข้อมูลย้อนหลัง • แบ่งข้อมูลตามเวลา • PROVE BEFORE TRADE")
st.info("ผลแสดงความแม่นยำย้อนหลัง ไม่ใช่สัญญาณซื้อขายหรือการรับประกันผลตอบแทน")

@st.cache_data(ttl=3600, show_spinner=False)
def download_btc():
    df = yf.download("BTC-USD", period="5y", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("โหลดข้อมูล BTC ไม่สำเร็จ ลองอีกครั้งหรืออัปโหลด CSV")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    # Exclude today's candle, which may still be changing.
    dates = pd.to_datetime(df.Date, utc=True)
    return df.loc[dates.dt.date < datetime.now(timezone.utc).date()]

source = st.radio("แหล่งข้อมูล", ["ทดลองด้วยข้อมูลจำลอง", "โหลด BTC จาก Yahoo Finance", "อัปโหลด CSV"], horizontal=True)
upload = None
if source == "อัปโหลด CSV":
    st.caption("CSV รายวันต่อเนื่อง อย่างน้อย 180 แถว: Date, Close, High, Low, Volume")
    upload = st.file_uploader("เลือกไฟล์ CSV", type=["csv"])
    st.download_button("ดาวน์โหลดตัวอย่าง CSV (ข้อมูลจำลอง)", demo_data().to_csv(index=False).encode(), "sample_market_data.csv", "text/csv")
source_key = (source, upload.getvalue() if upload is not None else None)
if st.session_state.get("source_key") != source_key:
    st.session_state.pop("analysis", None)
    st.session_state["source_key"] = source_key
if st.button("เริ่มวิเคราะห์", type="primary"):
    st.session_state.pop("analysis", None)
    try:
        with st.spinner("กำลังเตรียมข้อมูลและทดสอบโมเดล..."):
            if source == "ทดลองด้วยข้อมูลจำลอง":
                raw = demo_data()
            elif source == "โหลด BTC จาก Yahoo Finance":
                raw = download_btc()
            else:
                if upload is None:
                    raise ValueError("กรุณาเลือกไฟล์ CSV ก่อน")
                raw = pd.read_csv(upload)
            st.session_state.analysis = analyze(raw)
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        st.error("วิเคราะห์ไม่สำเร็จ กรุณาตรวจไฟล์หรือลองใหม่ หากโหลดข้อมูลไม่ได้ ให้ใช้โหมดจำลองหรือ CSV")

result = st.session_state.get("analysis")
if result is not None:
    if source == "ทดลองด้วยข้อมูลจำลอง":
        st.warning("ข้อมูลจำลองสำหรับทดลองหน้าจอ ไม่ใช่ราคา BTC จริง")
    cols = st.columns(3)
    cols[0].metric("ความแม่นยำ AI", f"{result['accuracy']:.1%}")
    cols[1].metric("ความแม่นยำวิธีพื้นฐาน", f"{result['baseline_accuracy']:.1%}")
    cols[2].metric("จำนวนวันที่ทดสอบ", len(result["ledger"]))
    difference = result["accuracy"] - result["baseline_accuracy"]
    st.write(f"AI ต่างจากวิธีพื้นฐาน {difference * 100:+.2f} จุดเปอร์เซ็นต์")
    st.caption(f"ฝึก {result['train_samples']} แถว ถึง {result['train_end'].date()} • เว้น 1 แถว • ทดสอบตั้งแต่ {result['test_start'].date()}")
    st.caption("วิธีพื้นฐาน: ทายทุกวันด้วยประเภทที่พบบ่อยที่สุดในชุดฝึก (ขึ้น หรือไม่ขึ้น)")
    st.subheader("ราคาย้อนหลัง")
    st.line_chart(result["data"].set_index("Date")[["Close"]])
    st.subheader("ผลทดสอบรายวัน")
    st.caption("prediction / actual: 1 = ขึ้น, 0 = ไม่ขึ้น • probability_up เป็นค่าประเมินของโมเดล ซึ่งยังไม่ได้สอบเทียบ")
    st.dataframe(result["ledger"], use_container_width=True, hide_index=True)
    st.download_button("ดาวน์โหลดผลทดสอบ CSV", result["ledger"].to_csv(index=False).encode(), "prediction_ledger.csv", "text/csv")
    with st.expander("ข้อมูลการประเมินเพิ่มเติม"):
        st.write({"balanced_accuracy": result["balanced_accuracy"], "log_loss": result["log_loss"]})
