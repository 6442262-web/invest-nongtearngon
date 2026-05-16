# คู่มือตั้งค่า — Daily Investment Brief

## ขั้นตอนทั้งหมด (ใช้เวลา ~10 นาที)

---

## ขั้นที่ 1: สร้าง GitHub Repository (Private)

1. ไปที่ https://github.com/new
2. ตั้งค่าดังนี้:
   - **Repository name:** `invest-brief` (หรือชื่ออื่น)
   - **Visibility:** `Private` ← สำคัญ! ห้าม Public
   - **Add a README:** ไม่ต้องติ๊ก
3. คลิก **Create repository**

---

## ขั้นที่ 2: อัปโหลดไฟล์ขึ้น GitHub

เปิด PowerShell ใน folder `invest test` แล้วรันทีละบรรทัด:

```powershell
# ไปที่ folder
cd "C:\Users\User\Desktop\invest test"

# เริ่ม git
git init
git add invest_analysis.py requirements.txt .github

# commit
git commit -m "Add daily investment brief script"

# เชื่อมต่อ GitHub (เปลี่ยน YOUR_USERNAME เป็นชื่อ GitHub ของคุณ)
git remote add origin https://github.com/YOUR_USERNAME/invest-brief.git
git branch -M main
git push -u origin main
```

---

## ขั้นที่ 3: ตั้ง Email Credentials (GitHub Secrets)

1. ไปที่ repo ของคุณบน GitHub
2. คลิก **Settings** → **Secrets and variables** → **Actions**
3. คลิก **New repository secret** แล้วเพิ่ม 3 ค่า:

| Secret Name | ค่า |
|---|---|
| `EMAIL_SENDER` | Gmail ของคุณ เช่น `myname@gmail.com` |
| `EMAIL_PASSWORD` | **App Password** (ดูขั้นที่ 4) |
| `EMAIL_RECIPIENT` | `6442262@schoolptk.ac.th` |

---

## ขั้นที่ 4: สร้าง Gmail App Password

> ⚠️ ต้องใช้ App Password เท่านั้น ห้ามใช้ password จริง

1. ไปที่ https://myaccount.google.com/security
2. เปิด **2-Step Verification** (ถ้ายังไม่ได้เปิด)
3. ไปที่ https://myaccount.google.com/apppasswords
4. เลือก **App:** `Mail` → **Device:** `Other` → พิมพ์ `invest-script`
5. คลิก **Generate** → คัดลอก password 16 ตัว (เช่น `xxxx xxxx xxxx xxxx`)
6. นำไปใส่ใน Secret `EMAIL_PASSWORD` (ไม่ต้องมีช่องว่าง)

---

## ขั้นที่ 5: ทดสอบรัน Manual

1. ไปที่ tab **Actions** ใน GitHub repo
2. คลิก **Daily Investment Brief** ทางซ้าย
3. คลิก **Run workflow** → **Run workflow**
4. รอ ~2 นาที แล้วเช็คอีเมล

---

## ตาราง Schedule

Script จะรันอัตโนมัติทุก **วันจันทร์–ศุกร์ เวลา 7:00 AM** (Bangkok)

ดู logs และ snapshot ได้ที่ tab **Actions** บน GitHub

---

## แก้ปัญหา

| ปัญหา | วิธีแก้ |
|---|---|
| Email ไม่ส่ง | ตรวจสอบ App Password และ 2FA เปิดอยู่ |
| Script error | ดู logs ใน Actions tab |
| ข้อมูลขาด | yfinance อาจ timeout — รัน manual ใหม่ |
| ต้องการเพิ่มหุ้น | แก้ `GROWTH_STOCKS` ใน `invest_analysis.py` |
