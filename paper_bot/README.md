# PROVE BEFORE TRADE BOT

ระบบทดลองซื้อขายอัตโนมัติแบบ **Paper Trading First** สำหรับ Bitkub Spot

## สถานะปัจจุบัน
**Level 2 → Level 3: Paper Trading Engine + Feature Lab**

เงินจริงยังถูกล็อก ระบบ V1 ไม่มี live-order executor

## Paper Trading
- ดึงราคาจริง BTC/THB จาก Bitkub
- กลยุทธ์ SMA crossover เป็น baseline เท่านั้น
- เงินจำลองเริ่มต้น 1,000 บาท
- จำกัดขนาด Position
- Stop Loss / Take Profit
- Daily Loss Kill Switch
- บันทึก Signal / Order / Equity ลง CSV

## Feature Lab V1
ทดสอบตัวแปรโดยไม่สมมติว่าตัวไหนต้องใช้ได้:
- Return 1/3/7 วัน
- Distance from SMA20 / SMA50
- RSI 14
- ATR 14 normalized
- Price range
- Volatility 7 / 30 วัน
- Volume change / volume z-score
- ETH return
- Nasdaq return
- DXY return
- VIX return

Feature Lab ใช้ chronological 70/30 train/test และห้าม shuffle time series
เพื่อช่วยลด look-ahead bias

ผลลัพธ์:
- `experiments/feature_lab_scores.csv`
- `experiments/feature_lab_latest_features.csv`

สถานะ `CANDIDATE` หมายถึง "ควรทดสอบต่อ" ไม่ใช่ "มี Edge พิสูจน์แล้ว"

## Proof Gate / Live Gate V1
ก่อนพิจารณาเงินจริง ระบบต้องผ่านอย่างน้อย:
- Paper Trading >= 60 วัน
- Closed trades >= 50
- Net return หลังต้นทุน > 0
- Max drawdown <= 10%
- Profit factor >= 1.20
- Out-of-Sample PASS
- Walk-forward PASS
- Fee + slippage stress PASS
- ทดสอบอย่างน้อย 3 market regimes
- Regime robustness PASS
- Prediction ledger ถูกล็อกก่อนรู้ผล

**สำคัญ:** ถึงหลักฐานครบทุกข้อ V1 ก็ยังคง `LIVE_LOCKED`
เพราะ `LIVE_TRADING_ENABLED = False` และยังไม่มีโค้ดส่งคำสั่งเงินจริง

## Google Colab
เปิดไฟล์ `PROVE_BEFORE_TRADE_COLAB.ipynb` แล้ว Run All

> PROVE BEFORE TRADE — ตัวชี้วัดทุกตัวเป็นหลักฐานให้ทดสอบต่อ ไม่ใช่คำแนะนำการลงทุน
