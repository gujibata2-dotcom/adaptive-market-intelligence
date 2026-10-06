# PROVE BEFORE TRADE BOT

ระบบทดลองซื้อขายอัตโนมัติแบบ **Paper Trading First** สำหรับ Bitkub Spot

## สิ่งที่ทำได้
- ดึงราคาจริง BTC/THB จาก Bitkub
- กลยุทธ์ตัวอย่าง SMA crossover
- เงินจำลองเริ่มต้น 1,000 บาท
- จำกัดขนาด Position
- Stop Loss / Take Profit
- Daily Loss Kill Switch
- บันทึก Signal / Order / Equity ลง CSV
- ค่าเริ่มต้น **ไม่ส่งคำสั่งเงินจริง**

## Google Colab
เปิดไฟล์ `PROVE_BEFORE_TRADE_COLAB.ipynb` แล้ว Run All

> กลยุทธ์ SMA เป็น baseline สำหรับทดสอบระบบ ไม่ใช่คำแนะนำการลงทุน
