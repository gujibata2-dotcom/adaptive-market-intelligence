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
    st.session_state.pop("paper_result", None)
    st.session_state["source_key"] = source_key
if st.button("เริ่มวิเคราะห์", type="primary"):
    st.session_state.pop("analysis", None)
    st.session_state.pop("paper_result", None)
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

if result is not None:
    from app.paper_trading import simulate
    st.divider()
    st.subheader("บัญชีจำลองซื้อ–ขายย้อนหลัง")
    st.caption("BTC/USD • เงินหน่วย USD • ถือซื้อหรือเงินสด • ไม่ใช้ leverage")
    st.info("สัญญาณวันนี้ส่งคำสั่งที่ราคาปิดวันถัดไป พร้อมค่าธรรมเนียมและ slippage; จุดหยุดขาดทุนตรวจเฉพาะราคาปิด จึงอาจขาดทุนเกินค่าที่ตั้งได้")
    with st.form("paper_settings"):
        cols = st.columns(3)
        capital = cols[0].number_input("เงินเริ่มต้น (USD)", min_value=100., max_value=10000000., value=10000.)
        allocation = cols[1].slider("ใช้เงินต่อการซื้อ (%)", 1, 100, 25)
        stop = cols[2].slider("หยุดขาดทุนจากราคาซื้อ (%)", 1, 50, 10)
        cols = st.columns(4)
        buy = cols[0].slider("ซื้อเมื่อ probability_up ≥", .51, .99, .60)
        sell = cols[1].slider("ขายเมื่อ probability_up ≤", .01, .49, .40)
        fee = cols[2].number_input("ค่าธรรมเนียมต่อข้าง (%)", min_value=0., max_value=5., value=.1, step=.01)
        slip = cols[3].number_input("Slippage ต่อข้าง (%)", min_value=0., max_value=5., value=.1, step=.01)
        submitted = st.form_submit_button("รันบัญชีจำลอง")
    if submitted:
        try:
            paper = simulate(result['data'], result['ledger'], capital, allocation/100, fee/100, slip/100, buy, sell, stop/100)
            st.session_state['paper_result'] = paper
        except ValueError as exc:
            st.error(str(exc))
    paper = st.session_state.get('paper_result')
    if paper is not None:
        cols = st.columns(4)
        cols[0].metric("เงินสุดท้าย (USD)", f"{paper['final_equity']:,.2f}")
        cols[1].metric("กำไร/ขาดทุนสุทธิ (USD)", f"{paper['net_pnl']:+,.2f}")
        cols[2].metric("ผลตอบแทนสุทธิ", f"{paper['total_return']:.2%}")
        cols[3].metric("เงินทุนลดลงสูงสุด", f"{paper['max_drawdown']:.2%}")
        st.caption(f"ปิดการซื้อขาย {paper['closed_trades']} ครั้ง • ค่าธรรมเนียมรวม {paper['fees']:.2f} USD • ซื้อแล้วถือด้วยเงินเต็มจำนวน: {paper['benchmark_return']:.2%}")
        if paper['win_rate'] is not None:
            st.write(f"สัดส่วนการซื้อขายที่กำไร: {paper['win_rate']:.1%}")
        st.caption("ผลนี้เป็นค่าตั้งจากรอบที่กดรันล่าสุด เปลี่ยนค่าตั้งแล้วกดรันใหม่ • การปรับค่าซ้ำบนชุดทดสอบอาจทำให้ผลดูดีเกินจริง")
        st.line_chart(paper['curve'].set_index('Date')[['equity']])
        st.dataframe(paper['trades'], hide_index=True, use_container_width=True)
        st.download_button("ดาวน์โหลดรายการซื้อ–ขายจำลอง", paper['trades'].to_csv(index=False).encode(), "paper_trades.csv", "text/csv")
        st.download_button("ดาวน์โหลดประวัติเงินทุน", paper['curve'].to_csv(index=False).encode(), "paper_equity.csv", "text/csv")
