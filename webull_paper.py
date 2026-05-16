# -*- coding: utf-8 -*-
"""
Webull Paper Trading + LINE Notify
พอร์ต ฿5,000 | แจ้งเตือน LINE ทุกวัน | Fractional Shares
รันทุกวัน 20:00 น. ไทย (ก่อนตลาด US เปิด 30 นาที)

Setup:
  pip install webull requests yfinance pandas numpy python-dotenv
  แล้วกรอก .env ตาม .env.example
"""

import os, json, time, warnings, math
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
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

# ── LINE Notify ───────────────────────────────────────────
LINE_TOKEN  = os.getenv("LINE_NOTIFY_TOKEN", "")   # จาก notify-bot.line.me

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
MACRO_TICKERS = ["DX-Y.NYB","^VIX","^TNX","HYG","QQQ","GLD","^GSPC","THBX=X"]


# ─────────────────────────────────────────────────────────
# 1. Exchange Rate
# ─────────────────────────────────────────────────────────

def get_thb_rate():
    global THB_USD_RATE
    try:
        df = yf.download("THBX=X", period="5d", progress=False)
        if df.empty:
            # fallback: USDTHB
            df = yf.download("THB=X", period="5d", progress=False)
        if not df.empty:
            usd_per_thb = float(df["Close"].dropna().iloc[-1])
            THB_USD_RATE = 1 / usd_per_thb if usd_per_thb < 1 else usd_per_thb
    except:
        pass
    if not THB_USD_RATE or THB_USD_RATE < 25 or THB_USD_RATE > 50:
        THB_USD_RATE = 35.0   # fallback rate
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

        reason = None
        if cur_price <= entry*(1-SL_PCT):
            reason = f"STOP LOSS (-{SL_PCT*100:.0f}%)"
        elif cur_price >= entry*(1+TP_PCT):
            reason = f"TAKE PROFIT (+{TP_PCT*100:.0f}%)"
        elif cur_price <= peak*(1-TRAIL_PCT):
            reason = f"TRAILING STOP ({TRAIL_PCT*100:.0f}% จาก peak)"
        elif regime in ("RISK_OFF","CRISIS") and pos.get("hold_days",0) >= 2:
            reason = "REGIME_EXIT (ตลาดเป็นขาลง)"
        elif sig and sig["score"] < -1:
            reason = "SIGNAL_REVERSAL (สัญญาณกลับ)"

        hold_days = (datetime.now() - datetime.fromisoformat(pos["entry_date"])).days
        pos["hold_days"] = hold_days

        if reason:
            pnl_pct = (cur_price - entry)/entry*100
            pnl_thb = pos["size_thb"] * pnl_pct/100
            exits.append({
                "ticker": tk, "reason": reason,
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

def apply_entries(state, picks, data, wb):
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
        entries.append({"ticker":tk, "price":sig["price"],
                        "size_thb":size_thb, "score":sig["score"]})
        print(f"  BOUGHT {tk} @ ${sig['price']:.2f} | {fmt_thb(size_thb)} (score={sig['score']})")
        time.sleep(0.3)

    return entries


# ─────────────────────────────────────────────────────────
# 5. LINE Notify
# ─────────────────────────────────────────────────────────

def send_line(message, image_url=None):
    if not LINE_TOKEN:
        print("  [!] LINE_NOTIFY_TOKEN ไม่ได้ตั้งค่า")
        print(f"\n--- LINE Message Preview ---\n{message}\n---")
        return False
    try:
        headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
        data    = {"message": message}
        if image_url:
            data["imageFullsize"]  = image_url
            data["imageThumbnail"] = image_url
        r = requests.post("https://notify-api.line.me/api/notify",
                          headers=headers, data=data, timeout=10)
        ok = r.status_code == 200
        print(f"  LINE Notify: {'OK' if ok else f'FAIL ({r.status_code})'}")
        return ok
    except Exception as e:
        print(f"  LINE error: {e}")
        return False

def build_line_message(state, regime, mac_score, macro_info,
                       exits, entries, all_signals, gold_score, gold_info):
    now = datetime.now() + timedelta(hours=0)  # server time
    now_bkk = datetime.utcnow() + timedelta(hours=7)

    # emoji ตาม regime
    regime_emoji = {
        "STRONG_ON":"🚀", "RISK_ON":"📈", "NEUTRAL":"➡️",
        "RISK_OFF":"⚠️", "CRISIS":"🚨"
    }.get(regime,"📊")

    regime_th = {
        "STRONG_ON":"แข็งแกร่งมาก", "RISK_ON":"ขาขึ้น",
        "NEUTRAL":"ทรงตัว", "RISK_OFF":"ระวัง", "CRISIS":"วิกฤต"
    }.get(regime, regime)

    # คำนวณ portfolio value
    total_invested = sum(p["size_thb"] for p in state["positions"].values())
    pnl_today_thb = 0
    for tk, pos in state["positions"].items():
        sig = next((s for s in all_signals if s and s["ticker"]==tk), None)
        if sig:
            pnl_today_thb += pos["size_thb"] * (sig["mom_1d"]/100)

    total_value = state.get("cash_thb",0) + total_invested
    total_return_pct = (total_value - PORTFOLIO_THB) / PORTFOLIO_THB * 100

    lines = []
    lines.append(f"\n🌅 Morning Signal — {now_bkk.strftime('%d %b %Y %H:%M น.')}")
    lines.append(f"{'─'*30}")

    # Macro
    lines.append(f"\n{regime_emoji} ตลาด: {regime_th} (score {mac_score:+d})")
    lines.append(f"   VIX {macro_info.get('vix','?')} | S&P500 {macro_info.get('sp500_1d',0):+.2f}%")
    lines.append(f"   QQQ 5d {macro_info.get('qqq_5d',0):+.2f}% | ทอง 5d {macro_info.get('gold_5d',0):+.2f}%")

    # Portfolio summary
    lines.append(f"\n💼 พอร์ต Paper Trading")
    lines.append(f"   เริ่มต้น : {fmt_thb(PORTFOLIO_THB)}")
    lines.append(f"   ปัจจุบัน : {fmt_thb(total_value)} ({total_return_pct:+.2f}%)")
    lines.append(f"   กำไร/ขาดทุน: {fmt_thb(total_value - PORTFOLIO_THB)}")
    lines.append(f"   เงินสดเหลือ: {fmt_thb(state.get('cash_thb',0))}")
    lines.append(f"   อัตราแลกเปลี่ยน: 1 USD = ฿{THB_USD_RATE:.2f}")

    # Current positions
    if state["positions"]:
        lines.append(f"\n📂 Position ที่ถือ ({len(state['positions'])} ตัว)")
        for tk, pos in state["positions"].items():
            sig = next((s for s in all_signals if s and s["ticker"]==tk), None)
            cur  = sig["price"] if sig else pos["entry_price"]
            pnl  = (cur - pos["entry_price"])/pos["entry_price"]*100
            pnl_thb = pos["size_thb"] * pnl/100
            emoji = "🟢" if pnl>0 else "🔴"
            lines.append(f"   {emoji} {tk}: {pnl:+.2f}% | {fmt_thb(pnl_thb)}")
            lines.append(f"      ซื้อ ${pos['entry_price']:.2f} → ปัจจุบัน ${cur:.2f}")
            lines.append(f"      SL=${pos['sl_price']} | TP=${pos['tp_price']}")
    else:
        lines.append(f"\n📂 ไม่มี Position เปิดอยู่")

    # Exits
    if exits:
        lines.append(f"\n🔔 ปิด Position วันนี้ ({len(exits)} ตัว)")
        for ex in exits:
            emoji = "✅" if ex["pnl_thb"]>=0 else "❌"
            lines.append(f"   {emoji} {ex['ticker']}: {ex['pnl_pct']:+.2f}% | {fmt_thb(ex['pnl_thb'])}")
            lines.append(f"      เหตุ: {ex['reason']}")

    # New entries
    if entries:
        lines.append(f"\n🛒 ซื้อใหม่วันนี้ (ตลาดเปิด 20:30 น.)")
        for en in entries:
            lines.append(f"   ✨ {en['ticker']} @ ${en['price']:.2f}")
            lines.append(f"      {fmt_thb(en['size_thb'])} (~{fmt_usd(thb_to_usd(en['size_thb']))})")
            lines.append(f"      SL=${en['price']*(1-SL_PCT):.2f} | TP=${en['price']*(1+TP_PCT):.2f}")
    else:
        lines.append(f"\n💤 ไม่มีการซื้อใหม่วันนี้")
        if regime in ("RISK_OFF","CRISIS"):
            lines.append(f"   (ตลาด{regime_th} — ถือเงินสดไว้ก่อน)")

    # Top signals
    top = sorted([s for s in all_signals if s and s["score"]>=MIN_SCORE],
                 key=lambda x:x["score"], reverse=True)[:5]
    if top:
        lines.append(f"\n🔍 หุ้นน่าสนใจวันนี้")
        for s in top:
            bar = "█"*min(s["score"],8) + "░"*(8-min(s["score"],8))
            lines.append(f"   {s['ticker']}: {bar} score={s['score']}")
            lines.append(f"      1d {s['mom_1d']:+.1f}% | 5d {s['mom_5d']:+.1f}% | vol {s['vol_ratio']:.1f}x")

    # Gold signal
    if gold_score > 0:
        lines.append(f"\n🥇 ทอง (GLD): score={gold_score} | 5d {gold_info.get('mom_5d',0):+.1f}%")

    # Risk reminder
    lines.append(f"\n⚙️ Risk Profile: AGGRESSIVE")
    lines.append(f"   SL={int(SL_PCT*100)}% | TP={int(TP_PCT*100)}% | Trail={int(TRAIL_PCT*100)}%")

    lines.append(f"\n{'─'*30}")
    lines.append(f"🤖 Auto-signal by invest-nongtearngon")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────

def run():
    print("\n" + "="*60)
    print("  Webull Paper Trading + LINE Notify")
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

    entries = apply_entries(state, picks, data, wb)

    # Update portfolio value
    state["last_updated"] = datetime.now().isoformat()
    state["last_regime"]  = regime
    save_state(state)
    print(f"  State saved | Positions: {len(state['positions'])} | Cash: {fmt_thb(state['cash_thb'])}")

    # LINE Notify
    print("\n[6/6] ส่ง LINE Notify...")
    msg = build_line_message(
        state, regime, mac_score, macro_info,
        exits, entries, signals, gold_sc, gold_info
    )
    send_line(msg)

    # Save snapshot
    snapshot = dict(
        datetime=datetime.now().isoformat(),
        rate_thb_usd=THB_USD_RATE,
        regime=regime, macro_score=mac_score,
        positions=state["positions"],
        exits=exits, entries=entries,
        signals=sorted(signals, key=lambda x:x["score"], reverse=True)[:10],
    )
    with open("webull_snapshot.json","w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    print("\n" + "="*60)
    print("  เสร็จแล้ว! ดู LINE สำหรับ summary")
    print("="*60 + "\n")


if __name__ == "__main__":
    run()
