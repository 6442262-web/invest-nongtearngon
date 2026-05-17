# -*- coding: utf-8 -*-
"""
Webull Paper Trading + Email & Telegram Notify
พอร์ต ฿5,000 | แจ้งเตือนทางอีเมล (+ Telegram ถ้าตั้งค่าไว้) | Fractional Shares
รันทุกวัน 20:00 น. ไทย (ก่อนตลาด US เปิด 30 นาที)

Setup:
  pip install webull requests yfinance pandas numpy python-dotenv
  แล้วกรอก .env ตาม .env.example

แจ้งเตือน:
  - Email (Gmail SMTP) — ตั้งค่า EMAIL_SENDER / EMAIL_PASSWORD / EMAIL_RECIPIENT
  - Telegram Bot       — ตั้งค่า TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (optional)
"""

import os, json, time, warnings, math, smtplib
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ── Webull Credentials ────────────────────────────────────
WB_EMAIL    = os.getenv("WEBULL_EMAIL", "")
WB_PASSWORD = os.getenv("WEBULL_PASSWORD", "")
WB_TRADE_PIN= os.getenv("WEBULL_TRADE_PIN", "")   # 6-digit trading PIN

# ── Email (Gmail SMTP) ───────────────────────────────────
# Gmail App Password: myaccount.google.com/apppasswords
EMAIL_SENDER    = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD  = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "")
SMTP_HOST       = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT       = int(os.getenv("SMTP_PORT", "587"))

# ── Telegram Bot (optional) ──────────────────────────────
# สร้าง Bot: คุยกับ @BotFather ใน Telegram → /newbot
# หา Chat ID: คุยกับ @userinfobot → copy id ตัวเลข
TG_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")   # จาก @BotFather
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",  "")    # จาก @userinfobot

# ── Portfolio (THB) ───────────────────────────────────────
PORTFOLIO_THB   = float(os.getenv("PORTFOLIO_THB", "5000"))
THB_USD_RATE    = None   # auto-fetch from Yahoo Finance

# ── Risk Profile (AGGRESSIVE — อายุน้อย) ─────────────────
SL_PCT       = 0.10   # Stop Loss 10%
TP_PCT       = 0.25   # Take Profit 25%
TRAIL_PCT    = 0.07   # Trailing Stop 7%
MAX_SLOTS    = 3      # ถือได้ 3 ตัวพร้อมกัน
STOCK_ALLOC  = 0.30   # 30% ต่อตัว (3x30% = 90% invested, 10% cash)
GOLD_ALLOC   = 0.15   # ทอง 15% (ถ้า signal ดี)
MIN_SCORE    = 3

WATCHLIST = [
    "NVDA","MSFT","AAPL","META","GOOGL","AMZN",
    "AVGO","NOW","PANW","ADBE","CRM",
    "DDOG","NET","ZS","TSLA","MRVL","AXON","TTD",
]
MACRO_TICKERS = ["DX-Y.NYB","^VIX","^TNX","HYG","QQQ","GLD","^GSPC","THB=X"]


# ─────────────────────────────────────────────────────────
# 1. Exchange Rate
# ─────────────────────────────────────────────────────────

def get_thb_rate():
    """
    ดึงอัตราแลกเปลี่ยน THB/USD จาก Yahoo Finance
    THB=X = USDTHB (จำนวนบาทต่อ 1 ดอลลาร์) → ใช้ค่าตรงๆ
    """
    global THB_USD_RATE
    for symbol in ["THB=X", "USDTHB=X"]:
        try:
            df = yf.download(symbol, period="5d", progress=False, auto_adjust=True)
            if df.empty:
                continue
            rate = float(df["Close"].dropna().iloc[-1])
            # THB=X ให้ค่าประมาณ 33-36 (บาทต่อดอลลาร์)
            if 25 < rate < 50:
                THB_USD_RATE = rate
                return THB_USD_RATE
        except:
            continue
    # fallback
    if not THB_USD_RATE or not (25 < THB_USD_RATE < 50):
        THB_USD_RATE = 35.0
    return THB_USD_RATE

def thb_to_usd(thb): return thb / THB_USD_RATE
def usd_to_thb(usd): return usd * THB_USD_RATE
def fmt_thb(thb):    return f"฿{thb:,.0f}"
def fmt_usd(usd):    return f"${usd:,.2f}"


# ─────────────────────────────────────────────────────────
# 2. Webull Paper Trading
# ─────────────────────────────────────────────────────────

class WebullPaper:
    """Wrapper สำหรับ Webull Paper Trading API"""

    def __init__(self):
        self.wb = None
        self.connected = False
        self.account_id = None

    def connect(self):
        try:
            from webull import paper_webull
            self.wb = paper_webull()

            if not WB_EMAIL or not WB_PASSWORD:
                print("  [!] Webull credentials not set (WEBULL_EMAIL / WEBULL_PASSWORD)")
                return False

            print(f"  Logging in to Webull Paper ({WB_EMAIL})...")

            # Login — อาจต้องใส่ MFA code ครั้งแรก
            result = self.wb.login(WB_EMAIL, WB_PASSWORD)

            if "data" in str(result).lower() or result:
                self.connected = True
                print("  Webull Paper: connected")
                return True
            else:
                print(f"  Webull login failed: {result}")
                return False

        except ImportError:
            print("  [!] webull not installed: pip install webull")
            return False
        except Exception as e:
            print(f"  Webull error: {e}")
            return False

    def get_account(self):
        """ดึงข้อมูลบัญชี paper"""
        if not self.connected: return {}
        try:
            acct = self.wb.get_paper_finance()
            return acct
        except Exception as e:
            print(f"  get_account error: {e}")
            return {}

    def get_positions(self):
        """ดึง positions ปัจจุบัน"""
        if not self.connected: return []
        try:
            pos = self.wb.get_paper_positions()
            return pos if pos else []
        except Exception as e:
            print(f"  get_positions error: {e}")
            return []

    def get_orders(self):
        """ดึง pending orders"""
        if not self.connected: return []
        try:
            return self.wb.get_paper_orders(status="Working") or []
        except Exception as e:
            return []

    def cancel_all_orders(self):
        """ยกเลิก orders ที่ค้างอยู่ทั้งหมด"""
        if not self.connected: return
        try:
            orders = self.get_orders()
            for o in orders:
                try:
                    oid = o.get("orderId") or o.get("id")
                    if oid:
                        self.wb.cancel_paper_order(oid)
                        time.sleep(0.2)
                except:
                    pass
            if orders:
                print(f"  Cancelled {len(orders)} pending orders")
        except Exception as e:
            print(f"  cancel_all error: {e}")

    def place_market_order(self, ticker, side, usd_amount):
        """
        ส่ง market order แบบ dollar-based (fractional shares)
        side = 'BUY' or 'SELL'
        """
        if not self.connected: return None
        try:
            # Webull fractional: ระบุ quant เป็น dollar
            result = self.wb.place_order(
                stock=ticker,
                action=side,
                orderType="MKT",
                enforce="DAY",
                quant=0,             # 0 = dollar-based
                extendedHours=False,
                stpPrice=None,
                trial_value=round(usd_amount, 2),
                trial_type="DOLLAR",
            )
            return result
        except Exception as e:
            print(f"  order error {ticker}: {e}")
            return None

    def place_order_with_targets(self, ticker, side, usd_amount, sl_price, tp_price):
        """
        ส่ง order พร้อม Stop Loss และ Take Profit
        Webull paper: ส่งแยกเป็น 3 orders (entry, stop, limit)
        """
        if not self.connected: return {}
        results = {}

        # Entry order (market)
        entry = self.place_market_order(ticker, side, usd_amount)
        results["entry"] = entry
        print(f"  Entry {side} {ticker} ${usd_amount:.2f}: {entry}")
        time.sleep(0.5)

        return results

    def get_portfolio_value(self):
        """คืนค่า total portfolio value เป็น USD"""
        if not self.connected: return PORTFOLIO_THB / THB_USD_RATE
        try:
            acct = self.get_account()
            # Webull paper finance structure varies
            for key in ["totalMarketValue","netLiquidation","totalAssets","portfolioValue"]:
                if key in str(acct):
                    val = acct.get(key) or acct.get("data",{}).get(key)
                    if val:
                        return float(val)
            return PORTFOLIO_THB / THB_USD_RATE
        except:
            return PORTFOLIO_THB / THB_USD_RATE


# ─────────────────────────────────────────────────────────
# 3. Market Data & Signals (same engine as morning_signal)
# ─────────────────────────────────────────────────────────

def fetch_data():
    all_tk = list(set(WATCHLIST + MACRO_TICKERS))
    raw = yf.download(all_tk, period="2mo", interval="1d",
                      progress=False, auto_adjust=True, group_by="ticker")
    result = {}
    for tk in all_tk:
        try:
            df = raw[tk].copy() if len(all_tk)>1 else raw.copy()
            df = df.dropna(subset=["Close","Open","High","Low"])
            if len(df) > 5: result[tk] = df
        except: pass
    return result

def lc(df,n):
    c = df["Close"].dropna()
    return c.values[-n:] if len(c)>=n else None

def lv(df,n):
    v = df["Volume"].dropna()
    return v.values[-n:] if len(v)>=n else None

def pct(arr,back):
    if arr is None or len(arr)<=back: return 0.0
    b = float(arr[-(back+1)])
    return (float(arr[-1])-b)/b*100 if b>0 else 0.0

def get_macro(data):
    score=0; info={}
    def a(tk,n): return lc(data[tk],n) if tk in data else None

    v=a("^VIX",3)
    if v is not None:
        vix=float(v[-1]); info["vix"]=round(vix,1)
        score+=(3 if vix<15 else 2 if vix<18 else 1 if vix<22
                else -1 if vix<28 else -2 if vix<35 else -4)

    h=a("HYG",6)
    if h is not None:
        h5=pct(h,5); info["hyg_5d"]=round(h5,2)
        score+=(2 if h5>0.5 else 1 if h5>0 else -2 if h5<-1 else 0)

    q=a("QQQ",11)
    if q is not None:
        q5=pct(q,5); info["qqq_5d"]=round(q5,2)
        score+=(2 if q5>2 else 1 if q5>0 else -2 if q5<-3 else 0)

    d=a("DX-Y.NYB",6)
    if d is not None:
        info["dxy"]=round(float(d[-1]),2)
        score+=(-1 if pct(d,5)>1 else 1 if pct(d,5)<-1 else 0)

    y=a("^TNX",6)
    if y is not None:
        info["yield_10y"]=round(float(y[-1]),2)

    sp=a("^GSPC",6)
    if sp is not None:
        info["sp500"]=round(float(sp[-1]),0)
        info["sp500_1d"]=round(pct(sp,1),2)

    g=a("GLD",6)
    if g is not None:
        info["gold_5d"]=round(pct(g,5),2)
        score+=(-1 if pct(g,5)>3 else 1 if pct(g,5)<-2 else 0)

    # อัตราแลกเปลี่ยน THB/USD (อัปเดตทุกรอบจาก data ที่ดึงมา)
    t=a("THB=X",2)
    if t is not None:
        live_rate = float(t[-1])
        if 25 < live_rate < 50:
            global THB_USD_RATE
            THB_USD_RATE = live_rate
    info["thb_usd"] = round(THB_USD_RATE, 2)

    info["score"]=score
    if   score>=4:  regime="STRONG_ON"
    elif score>=1:  regime="RISK_ON"
    elif score>=-1: regime="NEUTRAL"
    elif score>=-3: regime="RISK_OFF"
    else:           regime="CRISIS"
    return regime, score, info

def score_stock(tk, df, data):
    c2=lc(df,2); c6=lc(df,6); c11=lc(df,11); c21=lc(df,21); vol=lv(df,25)
    if c2 is None: return None
    cn=float(c2[-1]); cp=float(c2[-2])
    if cn<=0 or cp<=0: return None
    m1=pct(c2,1); m5=pct(c6,5) if c6 is not None else 0
    m10=pct(c11,10) if c11 is not None else 0
    m20=pct(c21,20) if c21 is not None else 0
    vr=1.0
    if vol is not None and len(vol)>=20:
        vr=float(vol[-5:].mean())/float(vol[-20:].mean())
    above_20=(cn>float(c21.mean())) if c21 is not None else False

    sc=0
    sc+=(3 if m1>3 else 2 if m1>1.5 else 1 if m1>0.5 else -2 if m1<-1.5 else 0)
    sc+=(3 if m5>6 else 2 if m5>3   else 1 if m5>1   else -2 if m5<-4   else 0)
    sc+=(2 if m10>10 else 1 if m10>5 else 0)
    sc+=(2 if m20>15 else 1 if m20>8 else -1 if m20<-5 else 0)
    sc+=(1 if above_20 else 0)
    sc+=(2 if vr>2 else 1 if vr>1.3 else 0)

    return dict(ticker=tk, score=sc, price=round(cn,2),
                mom_1d=round(m1,2), mom_5d=round(m5,2),
                mom_10d=round(m10,2), vol_ratio=round(vr,2),
                above_20ma=above_20,
                strength=("แข็งมาก" if sc>=11 else "แข็ง" if sc>=8
                          else "ปานกลาง" if sc>=5 else "อ่อน" if sc>=2 else "ขาลง"))

def score_gold(data):
    df=data.get("GLD")
    if df is None: return 0,{}
    c6=lc(df,6); c21=lc(df,21)
    if c6 is None: return 0,{}
    cn=float(c6[-1])
    m5=pct(c6,5); m20=pct(c21,20) if c21 is not None else 0
    sc=(3 if m5>3 else 2 if m5>1 else 1 if m5>0 else -1 if m5<-2 else 0)
    sc+=(2 if m20>10 else 1 if m20>4 else 0)
    return sc, dict(price=round(cn,2), mom_5d=round(m5,2), mom_20d=round(m20,2))


# ─────────────────────────────────────────────────────────
# 3b. News Fetcher — Economic & Tech
# ─────────────────────────────────────────────────────────

# คีย์เวิร์ดกรองข่าวเศรษฐกิจ
_ECON_KEYS = [
    "fed","rate","inflation","gdp","jobs","payroll","cpi","ppi",
    "interest","economy","recession","yield","tariff","treasury",
    "fomc","powell","employment","fiscal","deficit","trade war",
    "dollar","dxy","เฟด","ดอกเบี้ย","เงินเฟ้อ","เศรษฐกิจ",
]
# คีย์เวิร์ดกรองข่าวเทค
_TECH_KEYS = [
    "ai","chip","semiconductor","cloud","software","tech","cyber",
    "earnings","revenue","guidance","nvidia","microsoft","google",
    "apple","meta","amazon","tesla","broadcom","crowdstrike",
    "datadog","zscaler","palo alto","snowflake","salesforce",
    "profit","beat","miss","outlook","forecast","ipo","acquisition",
]

def _extract_news_item(raw):
    """รองรับ yfinance structure ทั้งเก่าและใหม่"""
    if "content" in raw:
        # yfinance >= 0.2.54 format
        c      = raw["content"]
        title  = c.get("title","") or c.get("headline","")
        pub    = (c.get("provider") or {}).get("displayName","")
        link   = (c.get("canonicalUrl") or {}).get("url","") or \
                 (c.get("clickThroughUrl") or {}).get("url","")
        ts_str = c.get("pubDate","")        # "2026-05-16T11:44:39Z"
        try:
            # parse ISO 8601
            ts_str_clean = ts_str.replace("Z","+00:00")
            from datetime import timezone
            dt = datetime.fromisoformat(ts_str_clean)
            ts = dt.timestamp()
        except:
            ts = 0
    else:
        # yfinance รุ่นเก่า
        title = raw.get("title","")
        pub   = raw.get("publisher","")
        link  = raw.get("link","")
        ts    = raw.get("providerPublishTime", 0)
    return title.strip(), pub, link, ts

def fetch_news():
    """
    ดึงข่าวจาก yfinance แบ่งเป็น:
      - economic_news : ข่าวเศรษฐกิจ/macro
      - tech_news     : ข่าวหุ้นเทค
    คืนค่า (economic_news[:6], tech_news[:8])
    """
    cutoff = datetime.now().timestamp() - 72 * 3600   # 72 ชม. ล่าสุด
    seen   = set()
    econ, tech = [], []

    # ── ดึงจาก macro tickers ──
    for sym in ["SPY","QQQ","TLT","GLD","^TNX","HYG"]:
        try:
            items = yf.Ticker(sym).news or []
            for raw in items:
                title, pub, link, ts = _extract_news_item(raw)
                if not title or title in seen or ts < cutoff:
                    continue
                seen.add(title)
                tl = title.lower()
                if any(k in tl for k in _ECON_KEYS):
                    econ.append(dict(title=title, pub=pub, link=link, ts=ts, sym=sym))
                elif any(k in tl for k in _TECH_KEYS):
                    tech.append(dict(title=title, pub=pub, link=link, ts=ts, sym=sym))
                else:
                    econ.append(dict(title=title, pub=pub, link=link, ts=ts, sym=sym))
        except:
            pass

    # ── ดึงจากหุ้นเทคใน watchlist ──
    for sym in ["NVDA","MSFT","AAPL","META","GOOGL","AMZN",
                "PANW","DDOG","ZS","NET","AVGO","NOW","TSLA"]:
        try:
            items = yf.Ticker(sym).news or []
            for raw in items[:4]:   # แค่ 4 ข่าวต่อตัว
                title, pub, link, ts = _extract_news_item(raw)
                if not title or title in seen or ts < cutoff:
                    continue
                seen.add(title)
                tech.append(dict(title=title, pub=pub, link=link, ts=ts, sym=sym))
        except:
            pass

    # sort newest first
    econ.sort(key=lambda x: x["ts"], reverse=True)
    tech.sort(key=lambda x: x["ts"], reverse=True)

    # dedup tech by sym: ไม่เกิน 2 ข่าวต่อตัว
    sym_count, tech_filtered = {}, []
    for n in tech:
        cnt = sym_count.get(n["sym"], 0)
        if cnt < 2:
            tech_filtered.append(n)
            sym_count[n["sym"]] = cnt + 1

    econ_final = econ[:6]
    tech_final = tech_filtered[:10]

    # แปลข่าวเป็นภาษาไทย
    print("  กำลังแปลข่าว EN → TH ...")
    _translate_news(econ_final)
    _translate_news(tech_final)

    return econ_final, tech_final

def _fmt_time_ago(ts):
    diff = datetime.now().timestamp() - ts
    if diff < 3600:   return f"{int(diff/60)} นาทีที่แล้ว"
    if diff < 86400:  return f"{int(diff/3600)} ชม.ที่แล้ว"
    return f"{int(diff/86400)} วันที่แล้ว"


def _translate_news(news_list):
    """
    แปลชื่อข่าวจาก EN → TH แบบ batch
    คืนค่า list เดิมพร้อม key 'title_th' เพิ่มเข้ามา
    """
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="en", target="th")
        titles = [n["title"] for n in news_list]

        # batch แปลทีละ 5 (หลีกเลี่ยง rate limit)
        translated = []
        for i in range(0, len(titles), 5):
            batch = titles[i:i+5]
            for t in batch:
                try:
                    th = translator.translate(t) or ""
                    translated.append(th)
                except:
                    translated.append("")
            if i + 5 < len(titles):
                time.sleep(0.3)

        for n, th in zip(news_list, translated):
            n["title_th"] = th
    except Exception as e:
        print(f"  [translate] {e}")
        for n in news_list:
            n["title_th"] = ""
    return news_list


# ─────────────────────────────────────────────────────────
# 4. Position Management
# ─────────────────────────────────────────────────────────

def load_state():
    """โหลด portfolio state จากไฟล์ (ตำแหน่งที่เปิดอยู่)"""
    try:
        with open("paper_state.json") as f:
            return json.load(f)
    except:
        return {
            "positions": {},
            "cash_thb": PORTFOLIO_THB,
            "cash_usd": thb_to_usd(PORTFOLIO_THB),
            "trades_history": [],
            "created": datetime.now().isoformat(),
        }

def save_state(state):
    with open("paper_state.json","w") as f:
        json.dump(state, f, indent=2, default=str)

def build_sell_reason(reason_code, pos, cur_price, peak, regime, sig):
    """สร้างเหตุผลการขายแบบละเอียด"""
    entry     = pos["entry_price"]
    pnl_pct   = (cur_price - entry) / entry * 100
    hold_days = pos.get("hold_days", 0)
    parts     = []

    if "STOP LOSS" in reason_code:
        parts.append(f"ราคาร่วงลง {abs(pnl_pct):.1f}% ถึงจุด Stop Loss ที่ ${cur_price:.2f}")
        parts.append(f"ตัดขาดทุนเพื่อปกป้องพอร์ต (SL ตั้งไว้ {SL_PCT*100:.0f}%)")

    elif "TAKE PROFIT" in reason_code:
        parts.append(f"ราคาขึ้น {pnl_pct:.1f}% ถึงเป้า Take Profit ที่ ${cur_price:.2f}")
        parts.append(f"ล็อคกำไรตามแผน (TP ตั้งไว้ {TP_PCT*100:.0f}%)")

    elif "TRAILING" in reason_code:
        drop_from_peak = (cur_price - peak) / peak * 100
        parts.append(f"ราคาร่วงจากจุดสูงสุด ${peak:.2f} ลงมา {abs(drop_from_peak):.1f}%")
        parts.append(f"Trailing Stop เตะที่ {TRAIL_PCT*100:.0f}% จาก peak — ล็อคกำไรบางส่วน")

    elif "REGIME" in reason_code:
        parts.append(f"สภาวะตลาดเปลี่ยนเป็น {regime} — ความเสี่ยงสูงขึ้น")
        parts.append(f"ถือมาแล้ว {hold_days} วัน ออกก่อนตลาดแย่ลงต่อ")

    elif "SIGNAL" in reason_code:
        sc = sig["score"] if sig else "?"
        parts.append(f"Signal กลับทิศ — score ลดลงเหลือ {sc} (ต่ำกว่า 0)")
        parts.append(f"แรงซื้อหายไป momentum อ่อนแล้ว")

    parts.append(f"ถือ {hold_days} วัน | ซื้อ ${entry:.2f} → ขาย ${cur_price:.2f}")
    return parts


def build_buy_reason(sig, regime, mac_score):
    """สร้างเหตุผลการซื้อแบบละเอียด"""
    parts = []

    # 1. Macro
    regime_th = {"STRONG_ON":"แข็งแกร่งมาก","RISK_ON":"ขาขึ้น",
                 "NEUTRAL":"ทรงตัว"}.get(regime, regime)
    parts.append(f"ตลาดอยู่ในโหมด {regime_th} (macro score {mac_score:+d})")

    # 2. Momentum
    mom_parts = []
    if sig.get("mom_1d", 0) > 0.5:
        mom_parts.append(f"วันนี้ +{sig['mom_1d']:.1f}%")
    if sig.get("mom_5d", 0) > 1:
        mom_parts.append(f"5 วัน +{sig['mom_5d']:.1f}%")
    if sig.get("mom_10d", 0) > 3:
        mom_parts.append(f"10 วัน +{sig['mom_10d']:.1f}%")
    if mom_parts:
        parts.append("Momentum ขาขึ้น: " + " | ".join(mom_parts))

    # 3. Volume
    vr = sig.get("vol_ratio", 1)
    if vr > 2:
        parts.append(f"ปริมาณซื้อขายพุ่ง {vr:.1f}x เหนือค่าเฉลี่ย — มีแรงซื้อเข้าแรง")
    elif vr > 1.3:
        parts.append(f"ปริมาณซื้อขายสูงกว่าปกติ {vr:.1f}x")

    # 4. MA20
    if sig.get("above_20ma"):
        parts.append("ราคาอยู่เหนือ MA20 — trend ยังเป็นขาขึ้น")

    # 5. Score summary
    strength = sig.get("strength", "")
    parts.append(f"Signal strength: {strength} (score {sig['score']})")

    return parts


def check_exits(state, signals_map, regime):
    """ตรวจสอบว่า position ไหนควรปิด"""
    exits = []
    for tk, pos in list(state["positions"].items()):
        entry = pos["entry_price"]
        peak  = pos.get("peak_price", entry)
        sig   = signals_map.get(tk)
        cur_price = sig["price"] if sig else entry

        # Update peak
        if cur_price > peak:
            pos["peak_price"] = cur_price

        reason_code = None
        if cur_price <= entry*(1-SL_PCT):
            reason_code = f"STOP LOSS (-{SL_PCT*100:.0f}%)"
        elif cur_price >= entry*(1+TP_PCT):
            reason_code = f"TAKE PROFIT (+{TP_PCT*100:.0f}%)"
        elif cur_price <= peak*(1-TRAIL_PCT):
            reason_code = f"TRAILING STOP ({TRAIL_PCT*100:.0f}% จาก peak)"
        elif regime in ("RISK_OFF","CRISIS") and pos.get("hold_days",0) >= 2:
            reason_code = "REGIME_EXIT (ตลาดเป็นขาลง)"
        elif sig and sig["score"] < -1:
            reason_code = "SIGNAL_REVERSAL (สัญญาณกลับ)"

        hold_days = (datetime.now() - datetime.fromisoformat(pos["entry_date"])).days
        pos["hold_days"] = hold_days

        if reason_code:
            pnl_pct = (cur_price - entry)/entry*100
            pnl_thb = pos["size_thb"] * pnl_pct/100
            reason_detail = build_sell_reason(reason_code, pos, cur_price, peak, regime, sig)
            exits.append({
                "ticker": tk, "reason": reason_code,
                "reason_detail": reason_detail,
                "entry": entry, "exit_price": cur_price,
                "pnl_pct": round(pnl_pct,2),
                "pnl_thb": round(pnl_thb,0),
                "hold_days": hold_days,
                "size_thb": pos["size_thb"],
            })

    return exits

def apply_exits(state, exits, wb):
    """ปิด positions และอัปเดต state"""
    for ex in exits:
        tk = ex["ticker"]
        if tk not in state["positions"]: continue

        # Webull: ขาย
        pos = state["positions"][tk]
        if wb.connected:
            wb.place_market_order(tk, "SELL", thb_to_usd(pos["size_thb"]))
            time.sleep(0.5)

        # คืนเงิน (ประมาณ, ไม่หักค่า commission paper trade)
        returned = pos["size_thb"] * (1 + ex["pnl_pct"]/100)
        state["cash_thb"] = state.get("cash_thb",0) + returned

        state["trades_history"].append({
            **ex,
            "date_exit": datetime.now().isoformat(),
            "date_entry": state["positions"][tk]["entry_date"],
        })
        del state["positions"][tk]
        print(f"  CLOSED {tk}: {ex['pnl_pct']:+.2f}% | {fmt_thb(ex['pnl_thb'])}")

def apply_entries(state, picks, data, wb, regime="NEUTRAL", mac_score=0):
    """เปิด positions ใหม่"""
    entries = []
    for sig in picks:
        tk = sig["ticker"]
        if tk in state["positions"]: continue

        size_thb = PORTFOLIO_THB * STOCK_ALLOC
        if state["cash_thb"] < size_thb * 0.9:
            print(f"  ไม่พอ cash สำหรับ {tk} (ต้องการ {fmt_thb(size_thb)})")
            continue

        size_usd = thb_to_usd(size_thb)

        # Webull: ซื้อ
        if wb.connected:
            result = wb.place_order_with_targets(
                tk, "BUY", size_usd,
                sl_price=round(sig["price"]*(1-SL_PCT),2),
                tp_price=round(sig["price"]*(1+TP_PCT),2),
            )
        else:
            result = {"entry": "SIMULATED"}

        reason_detail = build_buy_reason(sig, regime, mac_score)

        state["cash_thb"] -= size_thb
        state["positions"][tk] = {
            "entry_price": sig["price"],
            "entry_date":  datetime.now().isoformat(),
            "size_thb":    size_thb,
            "size_usd":    size_usd,
            "peak_price":  sig["price"],
            "hold_days":   0,
            "score":       sig["score"],
            "sl_price":    round(sig["price"]*(1-SL_PCT),2),
            "tp_price":    round(sig["price"]*(1+TP_PCT),2),
            "order_result": str(result),
        }
        entries.append({
            "ticker": tk, "price": sig["price"],
            "size_thb": size_thb, "score": sig["score"],
            "reason_detail": reason_detail,
            "mom_1d": sig.get("mom_1d",0), "mom_5d": sig.get("mom_5d",0),
            "vol_ratio": sig.get("vol_ratio",1), "strength": sig.get("strength",""),
        })
        print(f"  BOUGHT {tk} @ ${sig['price']:.2f} | {fmt_thb(size_thb)} (score={sig['score']})")
        time.sleep(0.3)

    return entries


# ─────────────────────────────────────────────────────────
# 5. Notify — Email + Telegram (optional)
# ─────────────────────────────────────────────────────────

# ── 5a. Email ────────────────────────────────────────────

def build_email_html(state, regime, mac_score, macro_info,
                     exits, entries, all_signals, gold_score, gold_info,
                     econ_news=None, tech_news=None):
    """สร้าง HTML email สวยงาม พร้อมตาราง portfolio + ข่าว"""
    econ_news = econ_news or []
    tech_news = tech_news or []
    now_bkk = datetime.utcnow() + timedelta(hours=7)

    regime_color = {
        "STRONG_ON":"#16a34a","RISK_ON":"#2563eb",
        "NEUTRAL":"#6b7280","RISK_OFF":"#d97706","CRISIS":"#dc2626"
    }.get(regime,"#6b7280")
    regime_th = {
        "STRONG_ON":"แข็งแกร่งมาก 🚀","RISK_ON":"ขาขึ้น 📈",
        "NEUTRAL":"ทรงตัว ➡️","RISK_OFF":"ระวัง ⚠️","CRISIS":"วิกฤต 🚨"
    }.get(regime, regime)

    total_invested = sum(p["size_thb"] for p in state["positions"].values())
    total_value    = state.get("cash_thb", 0) + total_invested
    total_ret_pct  = (total_value - PORTFOLIO_THB) / PORTFOLIO_THB * 100
    ret_color      = "#16a34a" if total_ret_pct >= 0 else "#dc2626"
    ret_sign       = "+" if total_ret_pct >= 0 else ""

    # ── styles ──
    css = """
    body{font-family:'Segoe UI',Arial,sans-serif;background:#f0f4f8;margin:0;padding:20px}
    .card{background:#fff;border-radius:12px;padding:24px;margin-bottom:16px;
          box-shadow:0 1px 4px rgba(0,0,0,.08)}
    h2{margin:0 0 4px;font-size:20px} h3{margin:8px 0 12px;font-size:15px;color:#374151}
    .tag{display:inline-block;padding:4px 12px;border-radius:20px;color:#fff;
         font-weight:600;font-size:13px}
    table{width:100%;border-collapse:collapse;font-size:13px}
    th{background:#f9fafb;color:#6b7280;font-weight:600;padding:8px 10px;
       text-align:left;border-bottom:2px solid #e5e7eb}
    td{padding:8px 10px;border-bottom:1px solid #f3f4f6}
    tr:hover td{background:#fafafa}
    .pos{color:#16a34a;font-weight:600} .neg{color:#dc2626;font-weight:600}
    .metric{display:inline-block;min-width:120px;margin:4px 8px 4px 0}
    .mval{font-size:18px;font-weight:700;display:block}
    .mlbl{font-size:11px;color:#9ca3af}
    .footer{font-size:11px;color:#9ca3af;text-align:center;padding-top:8px}
    """

    def pnl_cls(v): return "pos" if v >= 0 else "neg"
    def sign(v):    return "+" if v >= 0 else ""

    # ── Portfolio metrics row ──
    metrics_html = f"""
    <span class="metric">
      <span class="mval">{fmt_thb(PORTFOLIO_THB)}</span>
      <span class="mlbl">เริ่มต้น</span>
    </span>
    <span class="metric">
      <span class="mval" style="color:{ret_color}">{fmt_thb(total_value)}</span>
      <span class="mlbl">ปัจจุบัน ({ret_sign}{total_ret_pct:.2f}%)</span>
    </span>
    <span class="metric">
      <span class="mval" style="color:{ret_color}">{fmt_thb(total_value-PORTFOLIO_THB)}</span>
      <span class="mlbl">กำไร/ขาดทุน</span>
    </span>
    <span class="metric">
      <span class="mval">{fmt_thb(state.get('cash_thb',0))}</span>
      <span class="mlbl">เงินสด</span>
    </span>
    <span class="metric">
      <span class="mval">฿{THB_USD_RATE:.2f}</span>
      <span class="mlbl">1 USD</span>
    </span>
    """

    # ── Macro row ──
    macro_html = f"""
    <span class="metric">
      <span class="mval">{macro_info.get('vix','?')}</span>
      <span class="mlbl">VIX</span>
    </span>
    <span class="metric">
      <span class="mval">{sign(macro_info.get('sp500_1d',0))}{macro_info.get('sp500_1d',0):.2f}%</span>
      <span class="mlbl">S&amp;P500 1d</span>
    </span>
    <span class="metric">
      <span class="mval">{sign(macro_info.get('qqq_5d',0))}{macro_info.get('qqq_5d',0):.2f}%</span>
      <span class="mlbl">QQQ 5d</span>
    </span>
    <span class="metric">
      <span class="mval">{sign(macro_info.get('gold_5d',0))}{macro_info.get('gold_5d',0):.2f}%</span>
      <span class="mlbl">ทอง 5d</span>
    </span>
    <span class="metric">
      <span class="mval">{macro_info.get('yield_10y','?')}%</span>
      <span class="mlbl">10Y Yield</span>
    </span>
    <span class="metric">
      <span class="mval">{macro_info.get('dxy','?')}</span>
      <span class="mlbl">DXY</span>
    </span>
    <span class="metric" style="background:#fef9c3;padding:4px 8px;border-radius:6px">
      <span class="mval" style="color:#92400e">฿{macro_info.get('thb_usd', THB_USD_RATE):.2f}</span>
      <span class="mlbl">💱 1 USD (live)</span>
    </span>
    """

    # ── Positions table ──
    if state["positions"]:
        pos_rows = ""
        for tk, pos in state["positions"].items():
            sig     = next((s for s in all_signals if s and s["ticker"]==tk), None)
            cur     = sig["price"] if sig else pos["entry_price"]
            pnl_pct = (cur - pos["entry_price"]) / pos["entry_price"] * 100
            pnl_thb = pos["size_thb"] * pnl_pct / 100
            hold    = pos.get("hold_days", 0)
            peak    = pos.get("peak_price", cur)
            trail   = (cur - peak) / peak * 100
            pos_rows += f"""
            <tr>
              <td><b>{tk}</b></td>
              <td>${pos['entry_price']:.2f}</td>
              <td>${cur:.2f}</td>
              <td class="{pnl_cls(pnl_pct)}">{sign(pnl_pct)}{pnl_pct:.2f}%</td>
              <td class="{pnl_cls(pnl_thb)}">{fmt_thb(pnl_thb)}</td>
              <td>{hold} วัน</td>
              <td>${pos['sl_price']} / ${pos['tp_price']}</td>
            </tr>"""
        pos_html = f"""
        <table>
          <tr><th>หุ้น</th><th>ราคาซื้อ</th><th>ปัจจุบัน</th>
              <th>%</th><th>กำไร (฿)</th><th>ถือ</th><th>SL / TP</th></tr>
          {pos_rows}
        </table>"""
    else:
        pos_html = "<p style='color:#6b7280'>ไม่มี position เปิดอยู่</p>"

    # ── Exits table ──
    # ── Exits ──
    exits_html = ""
    if exits:
        ex_cards = ""
        for ex in exits:
            icon      = "✅" if ex["pnl_thb"] >= 0 else "❌"
            border    = "#16a34a" if ex["pnl_thb"] >= 0 else "#dc2626"
            detail_li = "".join(f"<li>{d}</li>" for d in ex.get("reason_detail",[]))
            ex_cards += f"""
            <div style="border-left:4px solid {border};padding:10px 14px;
                        margin-bottom:12px;background:#f9fafb;border-radius:0 8px 8px 0">
              <div style="font-size:15px;font-weight:700">
                {icon} {ex['ticker']}
                <span class="{pnl_cls(ex['pnl_pct'])}" style="margin-left:8px">
                  {sign(ex['pnl_pct'])}{ex['pnl_pct']:.2f}%
                </span>
                <span style="color:#6b7280;font-size:13px;font-weight:400;margin-left:6px">
                  {fmt_thb(ex['pnl_thb'])}
                </span>
              </div>
              <div style="font-size:12px;color:#6b7280;margin:2px 0 6px">
                ซื้อ ${ex['entry']:.2f} → ขาย ${ex['exit_price']:.2f}
                &nbsp;·&nbsp; ถือ {ex['hold_days']} วัน
                &nbsp;·&nbsp; <b>{ex['reason']}</b>
              </div>
              <ul style="margin:4px 0 0;padding-left:16px;font-size:13px;color:#374151;line-height:1.8">
                {detail_li}
              </ul>
            </div>"""
        exits_html = f"""
        <div class="card">
          <h3>🔔 ขายออก / ปิด Position วันนี้ ({len(exits)} ตัว)</h3>
          {ex_cards}
        </div>"""

    # ── New entries ──
    entries_html = ""
    if entries:
        en_cards = ""
        for en in entries:
            detail_li = "".join(f"<li>{d}</li>" for d in en.get("reason_detail",[]))
            en_cards += f"""
            <div style="border-left:4px solid #2563eb;padding:10px 14px;
                        margin-bottom:12px;background:#eff6ff;border-radius:0 8px 8px 0">
              <div style="font-size:15px;font-weight:700;color:#1e40af">
                🛒 BUY {en['ticker']}
                <span style="font-size:13px;font-weight:400;color:#374151;margin-left:8px">
                  @ ${en['price']:.2f}
                  &nbsp;·&nbsp; {fmt_thb(en['size_thb'])} (~{fmt_usd(thb_to_usd(en['size_thb']))})
                </span>
              </div>
              <div style="font-size:12px;color:#6b7280;margin:2px 0 6px">
                🛑 SL ${en['price']*(1-SL_PCT):.2f}
                &nbsp;·&nbsp; 🎯 TP ${en['price']*(1+TP_PCT):.2f}
                &nbsp;·&nbsp; 📉 Trail {int(TRAIL_PCT*100)}%
                &nbsp;·&nbsp; Score {en['score']} ({en.get('strength','')})
              </div>
              <div style="font-size:13px;font-weight:600;color:#1e40af;margin-bottom:4px">
                📋 เหตุผลที่ซื้อ:
              </div>
              <ul style="margin:0;padding-left:16px;font-size:13px;color:#374151;line-height:1.8">
                {detail_li}
              </ul>
            </div>"""
        entries_html = f"""
        <div class="card">
          <h3>🛒 ซื้อใหม่วันนี้ — รอตลาดเปิด 20:30 น.</h3>
          {en_cards}
        </div>"""
    else:
        no_entry_note = ""
        if regime in ("RISK_OFF","CRISIS"):
            no_entry_note = f"<p style='color:#d97706'>ตลาด{regime_th} — ถือเงินสดไว้ก่อน</p>"
        entries_html = f"""
        <div class="card">
          <h3>💤 ไม่มีการซื้อใหม่วันนี้</h3>
          {no_entry_note}
        </div>"""

    # ── Top watchlist ──
    top = sorted([s for s in all_signals if s and s["score"] >= MIN_SCORE],
                 key=lambda x: x["score"], reverse=True)[:8]
    watch_rows = "".join(f"""
    <tr>
      <td><b>{'📌 ' if s['ticker'] in state['positions'] else ''}{s['ticker']}</b></td>
      <td>{s['score']}</td>
      <td class="{pnl_cls(s['mom_1d'])}">{sign(s['mom_1d'])}{s['mom_1d']:.1f}%</td>
      <td class="{pnl_cls(s['mom_5d'])}">{sign(s['mom_5d'])}{s['mom_5d']:.1f}%</td>
      <td>{s['vol_ratio']:.1f}x</td>
      <td>${s['price']:.2f}</td>
      <td>{'✅' if s['above_20ma'] else '—'}</td>
    </tr>""" for s in top) if top else "<tr><td colspan='7' style='color:#9ca3af'>ไม่มีหุ้นผ่านเกณฑ์</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{css}</style></head>
<body>
  <div class="card" style="background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff">
    <h2>🌅 Morning Signal — Paper Trading</h2>
    <p style="margin:4px 0;opacity:.85">{now_bkk.strftime('%A %d %B %Y  %H:%M น. (เวลาไทย)')}</p>
    <span class="tag" style="background:{regime_color};margin-top:8px">
      ตลาด: {regime_th}  (score {mac_score:+d})
    </span>
  </div>

  <div class="card">
    <h3>🌐 ภาวะตลาดโลก</h3>
    {macro_html}
  </div>

  <div class="card">
    <h3>💼 Portfolio Paper Trading</h3>
    {metrics_html}
  </div>

  <div class="card">
    <h3>📂 Position ที่ถือ ({len(state['positions'])} ตัว)</h3>
    {pos_html}
  </div>

  {exits_html}
  {entries_html}

  <div class="card">
    <h3>🔍 หุ้นน่าสนใจ (score ≥ {MIN_SCORE})</h3>
    <table>
      <tr><th>หุ้น</th><th>Score</th><th>1d</th><th>5d</th><th>Volume</th><th>ราคา</th><th>&gt;MA20</th></tr>
      {watch_rows}
    </table>
  </div>

  {_build_news_html(econ_news, tech_news)}

  <div class="footer">
    ⚙️ AGGRESSIVE  SL={int(SL_PCT*100)}%  TP={int(TP_PCT*100)}%  Trail={int(TRAIL_PCT*100)}%
    &nbsp;|&nbsp; 🤖 invest-nongtearngon  |  Paper {fmt_thb(PORTFOLIO_THB)}
  </div>
</body></html>"""
    return html


def _build_news_html(econ_news, tech_news):
    """สร้าง HTML สำหรับส่วนข่าว"""
    if not econ_news and not tech_news:
        return ""

    def news_li(n):
        ago      = _fmt_time_ago(n["ts"])
        pub      = n.get("pub","")
        link     = n.get("link","")
        title_en = n["title"]
        title_th = n.get("title_th","")

        title_html = (f'<a href="{link}" style="color:#1d4ed8;text-decoration:none;font-weight:600">{title_en}</a>'
                      if link else f'<b>{title_en}</b>')
        th_html = (f'<span style="display:block;color:#374151;font-size:13px;margin-top:2px">'
                   f'📌 {title_th}</span>' if title_th else "")
        meta = f'<span style="font-size:11px;color:#9ca3af">{pub}  ·  {ago}</span>' if pub else \
               f'<span style="font-size:11px;color:#9ca3af">{ago}</span>'

        return (f'<li style="margin-bottom:12px;line-height:1.5">'
                f'{title_html}'
                f'{th_html}'
                f'<span style="display:block;margin-top:2px">{meta}</span>'
                f'</li>')

    econ_html = ""
    if econ_news:
        items = "".join(news_li(n) for n in econ_news)
        econ_html = f"""
        <div class="card">
          <h3>📰 ข่าวเศรษฐกิจสำคัญ</h3>
          <ul style="padding-left:18px;margin:0">{items}</ul>
        </div>"""

    tech_html = ""
    if tech_news:
        items = "".join(news_li(n) for n in tech_news)
        tech_html = f"""
        <div class="card">
          <h3>💻 ข่าวหุ้นเทค</h3>
          <ul style="padding-left:18px;margin:0">{items}</ul>
        </div>"""

    return econ_html + tech_html


def send_email(subject, html_body):
    """ส่ง HTML email ผ่าน Gmail SMTP"""
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECIPIENT:
        print("  [!] Email ยังไม่ได้ตั้งค่า (EMAIL_SENDER / EMAIL_PASSWORD / EMAIL_RECIPIENT)")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_RECIPIENT
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())

        print(f"  Email: ส่งแล้ว → {EMAIL_RECIPIENT} ✓")
        return True
    except Exception as e:
        print(f"  Email error: {e}")
        return False


# ── 5b. Telegram (optional) ──────────────────────────────

def send_telegram(message):
    """
    ส่งข้อความผ่าน Telegram Bot API
    รองรับ Markdown: *bold* _italic_ `code`
    """
    if not TG_TOKEN or not TG_CHAT_ID:
        print("  [!] Telegram ยังไม่ได้ตั้งค่า")
        print("      TELEGRAM_BOT_TOKEN และ TELEGRAM_CHAT_ID ใน .env")
        print(f"\n--- Telegram Preview ---\n{message}\n---")
        return False
    try:
        url  = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = {
            "chat_id":    TG_CHAT_ID,
            "text":       message,
            "parse_mode": "HTML",      # รองรับ <b> <i> <code>
            "disable_web_page_preview": True,
        }
        r = requests.post(url, json=data, timeout=15)
        ok = r.status_code == 200
        if not ok:
            print(f"  Telegram FAIL: {r.status_code} {r.text[:200]}")
        else:
            print("  Telegram: ส่งแล้ว ✓")
        return ok
    except Exception as e:
        print(f"  Telegram error: {e}")
        return False

def send_telegram_chunks(message):
    """ถ้าข้อความยาวเกิน 4096 ตัว (Telegram limit) → แบ่งส่ง"""
    MAX = 4000
    if len(message) <= MAX:
        return send_telegram(message)
    parts = [message[i:i+MAX] for i in range(0, len(message), MAX)]
    ok = True
    for i, part in enumerate(parts, 1):
        header = f"<b>[{i}/{len(parts)}]</b>\n" if len(parts) > 1 else ""
        ok = send_telegram(header + part) and ok
        time.sleep(0.5)
    return ok

def build_telegram_message(state, regime, mac_score, macro_info,
                           exits, entries, all_signals, gold_score, gold_info,
                           econ_news=None, tech_news=None):
    """
    สร้างข้อความสำหรับ Telegram (HTML format)
    <b>bold</b>  <i>italic</i>  <code>code</code>
    """
    econ_news = econ_news or []
    tech_news = tech_news or []
    now_bkk = datetime.utcnow() + timedelta(hours=7)

    regime_emoji = {
        "STRONG_ON":"🚀","RISK_ON":"📈","NEUTRAL":"➡️",
        "RISK_OFF":"⚠️","CRISIS":"🚨"
    }.get(regime,"📊")
    regime_th = {
        "STRONG_ON":"แข็งแกร่งมาก","RISK_ON":"ขาขึ้น",
        "NEUTRAL":"ทรงตัว","RISK_OFF":"ระวัง","CRISIS":"วิกฤต"
    }.get(regime, regime)

    # Portfolio value
    total_invested = sum(p["size_thb"] for p in state["positions"].values())
    total_value    = state.get("cash_thb",0) + total_invested
    total_ret_pct  = (total_value - PORTFOLIO_THB) / PORTFOLIO_THB * 100
    ret_emoji      = "🟢" if total_ret_pct >= 0 else "🔴"

    L = []   # lines

    # ── Header
    L.append(f"🌅 <b>Morning Signal</b>  {now_bkk.strftime('%d %b %Y  %H:%M น.')}")
    L.append("━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ── Macro
    L.append(f"\n{regime_emoji} <b>ตลาด: {regime_th}</b>  (score <code>{mac_score:+d}</code>)")
    L.append(f"   VIX <b>{macro_info.get('vix','?')}</b>"
             f"  │  S&amp;P500 <b>{macro_info.get('sp500_1d',0):+.2f}%</b>")
    L.append(f"   QQQ 5d <b>{macro_info.get('qqq_5d',0):+.2f}%</b>"
             f"  │  ทอง 5d <b>{macro_info.get('gold_5d',0):+.2f}%</b>")
    L.append(f"   10Y Yield <b>{macro_info.get('yield_10y','?')}%</b>"
             f"  │  DXY <b>{macro_info.get('dxy','?')}</b>")
    L.append(f"   💱 1 USD = <b>฿{macro_info.get('thb_usd', THB_USD_RATE):.2f}</b>"
             f"  (live)")

    # ── Portfolio
    L.append(f"\n💼 <b>Portfolio Paper Trading</b>")
    L.append(f"   เริ่มต้น   : <b>{fmt_thb(PORTFOLIO_THB)}</b>")
    L.append(f"   ปัจจุบัน  : {ret_emoji} <b>{fmt_thb(total_value)}</b>  ({total_ret_pct:+.2f}%)")
    L.append(f"   กำไร/ขาดทุน: <b>{fmt_thb(total_value - PORTFOLIO_THB)}</b>")
    L.append(f"   เงินสด     : <b>{fmt_thb(state.get('cash_thb',0))}</b>")
    L.append(f"   1 USD = ฿{THB_USD_RATE:.2f}")

    # ── Current positions
    if state["positions"]:
        L.append(f"\n📂 <b>Position ที่ถือ</b>  ({len(state['positions'])} ตัว)")
        for tk, pos in state["positions"].items():
            sig     = next((s for s in all_signals if s and s["ticker"]==tk), None)
            cur     = sig["price"] if sig else pos["entry_price"]
            pnl_pct = (cur - pos["entry_price"]) / pos["entry_price"] * 100
            pnl_thb = pos["size_thb"] * pnl_pct / 100
            hold    = pos.get("hold_days", 0)
            icon    = "🟢" if pnl_pct >= 0 else "🔴"
            peak    = pos.get("peak_price", cur)
            trail_from_peak = (cur - peak) / peak * 100

            L.append(f"\n   {icon} <b>{tk}</b>  {pnl_pct:+.2f}%  ({fmt_thb(pnl_thb)})")
            L.append(f"      ซื้อ <code>${pos['entry_price']:.2f}</code>"
                     f" → ปัจจุบัน <code>${cur:.2f}</code>  ({hold}วัน)")
            L.append(f"      🛑 SL <code>${pos['sl_price']}</code>"
                     f"  🎯 TP <code>${pos['tp_price']}</code>")
            L.append(f"      Peak <code>${peak:.2f}</code>  ({trail_from_peak:+.1f}% จาก peak)")
    else:
        L.append(f"\n📂 <b>ไม่มี Position เปิดอยู่</b>")

    # ── Exits today
    if exits:
        L.append(f"\n🔔 <b>ขายออก / ปิด Position วันนี้</b>  ({len(exits)} ตัว)")
        for ex in exits:
            icon = "✅" if ex["pnl_thb"] >= 0 else "❌"
            L.append(f"\n   {icon} <b>{ex['ticker']}</b>  {ex['pnl_pct']:+.2f}%"
                     f"  │  {fmt_thb(ex['pnl_thb'])}")
            L.append(f"      ซื้อ ${ex['entry']:.2f} → ขาย ${ex['exit_price']:.2f}"
                     f"  ({ex['hold_days']} วัน)")
            L.append(f"      <b>สาเหตุ: {ex['reason']}</b>")
            for d in ex.get("reason_detail", []):
                L.append(f"      • {d}")

    # ── New entries
    if entries:
        L.append(f"\n🛒 <b>ซื้อใหม่วันนี้</b>  (ตลาดเปิด 20:30 น.)")
        for en in entries:
            L.append(f"\n   ✨ <b>{en['ticker']}</b>"
                     f"  @  <code>${en['price']:.2f}</code>")
            L.append(f"      {fmt_thb(en['size_thb'])}"
                     f"  (~{fmt_usd(thb_to_usd(en['size_thb']))})")
            L.append(f"      🛑 SL <code>${en['price']*(1-SL_PCT):.2f}</code>"
                     f"  🎯 TP <code>${en['price']*(1+TP_PCT):.2f}</code>"
                     f"  📉 Trail {int(TRAIL_PCT*100)}%")
            L.append(f"      <b>📋 เหตุผลที่ซื้อ:</b>")
            for d in en.get("reason_detail", []):
                L.append(f"      • {d}")
    else:
        L.append(f"\n💤 <b>ไม่มีการซื้อใหม่</b>")
        if regime in ("RISK_OFF","CRISIS"):
            L.append(f"   ตลาด{regime_th} — ถือเงินสดไว้ก่อน")

    # ── Top signals watchlist
    top = sorted([s for s in all_signals if s and s["score"] >= MIN_SCORE],
                 key=lambda x: x["score"], reverse=True)[:6]
    if top:
        L.append(f"\n🔍 <b>หุ้นน่าสนใจวันนี้</b>")
        for s in top:
            held = "📌" if s["ticker"] in state["positions"] else "  "
            bar  = "█" * min(s["score"], 8) + "░" * (8 - min(s["score"], 8))
            L.append(f"   {held}<b>{s['ticker']}</b> <code>{bar}</code> {s['score']}")
            L.append(f"      1d <b>{s['mom_1d']:+.1f}%</b>"
                     f"  5d <b>{s['mom_5d']:+.1f}%</b>"
                     f"  vol <b>{s['vol_ratio']:.1f}x</b>")

    # ── Gold
    if gold_score > 0:
        L.append(f"\n🥇 <b>ทอง GLD</b>  score={gold_score}"
                 f"  │  5d {gold_info.get('mom_5d',0):+.1f}%"
                 f"  20d {gold_info.get('mom_20d',0):+.1f}%")

    # ── Economic News
    if econ_news:
        L.append(f"\n📰 <b>ข่าวเศรษฐกิจสำคัญ</b>")
        for n in econ_news[:5]:
            ago    = _fmt_time_ago(n["ts"])
            title_th = n.get("title_th","")
            L.append(f"   • {n['title']}")
            if title_th:
                L.append(f"     📌 <i>{title_th}</i>")
            L.append(f"     <i>{n.get('pub','')}  ·  {ago}</i>")

    # ── Tech News
    if tech_news:
        L.append(f"\n💻 <b>ข่าวหุ้นเทค</b>")
        for n in tech_news[:6]:
            ago      = _fmt_time_ago(n["ts"])
            sym_tag  = f"[<b>{n['sym']}</b>] " if n.get("sym") else ""
            title_th = n.get("title_th","")
            L.append(f"   • {sym_tag}{n['title']}")
            if title_th:
                L.append(f"     📌 <i>{title_th}</i>")
            L.append(f"     <i>{n.get('pub','')}  ·  {ago}</i>")

    # ── Footer
    L.append(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━")
    L.append(f"⚙️ AGGRESSIVE  SL={int(SL_PCT*100)}%  TP={int(TP_PCT*100)}%  Trail={int(TRAIL_PCT*100)}%")
    L.append(f"🤖 invest-nongtearngon  |  Paper {fmt_thb(PORTFOLIO_THB)}")

    return "\n".join(L)


# ─────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────

def run():
    print("\n" + "="*60)
    print("  Webull Paper Trading + Telegram Notify")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')} | พอร์ต {fmt_thb(PORTFOLIO_THB)}")
    print("="*60)

    # Exchange rate
    print("\n[1/6] อัตราแลกเปลี่ยน...")
    get_thb_rate()
    port_usd = thb_to_usd(PORTFOLIO_THB)
    print(f"  1 USD = ฿{THB_USD_RATE:.2f} | {fmt_thb(PORTFOLIO_THB)} = {fmt_usd(port_usd)}")

    # Fetch data
    print("\n[2/6] ดึงข้อมูลตลาด...")
    data = fetch_data()
    print(f"  โหลด {len(data)} tickers")

    # Macro
    print("\n[3/6] วิเคราะห์ macro...")
    regime, mac_score, macro_info = get_macro(data)
    print(f"  Regime: {regime} ({mac_score:+d}) | VIX={macro_info.get('vix','?')}")

    # Signals
    print("\n[4/6] คำนวณ signal...")
    signals = [score_stock(tk, data[tk], data) for tk in WATCHLIST if tk in data]
    signals = [s for s in signals if s]
    gold_sc, gold_info = score_gold(data)
    signals_map = {s["ticker"]:s for s in signals}

    eligible = [s for s in signals if s["score"]>=MIN_SCORE]
    print(f"  eligible: {len(eligible)}/{len(signals)} | gold score={gold_sc}")

    # Load state
    print("\n[5/6] จัดการ portfolio...")
    state = load_state()

    # Connect Webull (optional)
    wb = WebullPaper()
    if WB_EMAIL and WB_PASSWORD:
        wb.connect()
        wb.cancel_all_orders()

    # Check exits
    exits = check_exits(state, signals_map, regime)
    if exits:
        print(f"  ปิด {len(exits)} positions...")
        apply_exits(state, exits, wb)

    # Select new picks
    picks = []
    if regime not in ("RISK_OFF","CRISIS"):
        already_held = set(state["positions"].keys())
        candidates   = sorted(
            [s for s in eligible if s["ticker"] not in already_held],
            key=lambda x:(x["score"], x["mom_5d"]), reverse=True
        )
        slots = MAX_SLOTS - len(state["positions"])
        picks = candidates[:slots]

    entries = apply_entries(state, picks, data, wb, regime, mac_score)

    # Update portfolio value
    state["last_updated"] = datetime.now().isoformat()
    state["last_regime"]  = regime
    save_state(state)
    print(f"  State saved | Positions: {len(state['positions'])} | Cash: {fmt_thb(state['cash_thb'])}")

    # Fetch News
    print("\n[5.5/6] ดึงข่าว...")
    econ_news, tech_news = fetch_news()
    print(f"  ข่าวเศรษฐกิจ {len(econ_news)} | ข่าวเทค {len(tech_news)}")

    # Notify — Email + Telegram
    print("\n[6/6] ส่งการแจ้งเตือน...")
    now_bkk = datetime.utcnow() + timedelta(hours=7)
    subject  = (f"📊 Paper Trading Signal  {now_bkk.strftime('%d/%m/%Y')}"
                f"  |  {regime}  ({mac_score:+d})"
                f"  |  {len(entries)} Buy / {len(exits)} Close")

    # Email (primary)
    html_body = build_email_html(
        state, regime, mac_score, macro_info,
        exits, entries, signals, gold_sc, gold_info,
        econ_news, tech_news
    )
    send_email(subject, html_body)

    # Telegram (optional — ถ้ามี token)
    if TG_TOKEN and TG_CHAT_ID:
        tg_msg = build_telegram_message(
            state, regime, mac_score, macro_info,
            exits, entries, signals, gold_sc, gold_info,
            econ_news, tech_news
        )
        send_telegram_chunks(tg_msg)

    # Save snapshot
    snapshot = dict(
        datetime=datetime.now().isoformat(),
        rate_thb_usd=THB_USD_RATE,
        regime=regime, macro_score=mac_score,
        positions=state["positions"],
        exits=exits, entries=entries,
        signals=sorted(signals, key=lambda x:x["score"], reverse=True)[:10],
        econ_news=[{"title":n["title"],"pub":n["pub"]} for n in econ_news],
        tech_news=[{"title":n["title"],"sym":n["sym"]} for n in tech_news],
    )
    with open("webull_snapshot.json","w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    print("\n" + "="*60)
    print("  เสร็จแล้ว! ดู email สำหรับ summary")
    print("="*60 + "\n")


if __name__ == "__main__":
    run()
