# -*- coding: utf-8 -*-
"""
Morning Signal -- 3-Asset Aggressive (Stocks + Gold + Cash)
Risk Profile: HIGH (young investor, long time horizon)
Auto-Order  : Alpaca API (bracket orders = entry + SL + TP in 1 order)

รันก่อนตลาดเปิด → วิเคราะห์ → ส่ง order อัตโนมัติ → อีเมลสรุป
US Open: 9:30 AM ET = 20:30 น. ไทย
"""

import os, sys, json, smtplib, warnings, time
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ── Broker: Alpaca ────────────────────────────────────────
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

ALPACA_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # paper by default
ALPACA_LIVE   = os.getenv("ALPACA_LIVE", "false").lower() == "true"

# ── Email ─────────────────────────────────────────────────
EMAIL_SENDER    = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD  = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "6442262@schoolptk.ac.th")
SMTP_HOST       = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT       = int(os.getenv("SMTP_PORT", "587"))

# ── Portfolio & Risk Profile (AGGRESSIVE) ─────────────────
PORTFOLIO_USD    = float(os.getenv("PORTFOLIO_USD", "5000"))
RISK_PROFILE     = "AGGRESSIVE"   # CONSERVATIVE / MODERATE / AGGRESSIVE

#  ┌─────────────────────────────────────────────────────┐
#  │  Risk Profile Comparison                            │
#  │  Conservative : 25% / pos,  SL 5%,  TP 10%         │
#  │  Moderate     : 35% / pos,  SL 7%,  TP 15%         │
#  │  Aggressive   : 45% / pos,  SL 10%, TP 25%  <-- YOU│
#  └─────────────────────────────────────────────────────┘
PROFILES = {
    "CONSERVATIVE": dict(
        max_slots=2, stock_pct=0.25, gold_pct=0.20,
        sl=0.05, tp=0.10, trail=0.04, vix_limit=25,
        min_score=5, allow_regime={"STRONG_ON","RISK_ON"}
    ),
    "MODERATE": dict(
        max_slots=2, stock_pct=0.35, gold_pct=0.25,
        sl=0.07, tp=0.15, trail=0.05, vix_limit=30,
        min_score=4, allow_regime={"STRONG_ON","RISK_ON","NEUTRAL"}
    ),
    "AGGRESSIVE": dict(
        max_slots=3,       # ถือได้ 3 หุ้นพร้อมกัน
        stock_pct=0.45,    # 45% ต่อตัว
        gold_pct=0.20,     # ทองน้อยลง เน้นหุ้นมากขึ้น
        sl=0.10,           # SL 10% (ให้ breathing room มากขึ้น)
        tp=0.25,           # TP 25% (ปล่อยให้วิ่งไกลขึ้น)
        trail=0.07,        # Trailing 7%
        vix_limit=35,      # ทนความกลัวได้มากขึ้น
        min_score=3,       # สัญญาณปานกลางก็เข้าได้
        allow_regime={"STRONG_ON","RISK_ON","NEUTRAL"}  # ไม่ถอยแม้ตลาด neutral
    ),
}
P = PROFILES[RISK_PROFILE]

GROWTH_STOCKS = [
    # Mega-cap quality (core)
    "NVDA","MSFT","AAPL","META","GOOGL","AMZN",
    # High-growth tech
    "AVGO","NOW","ADBE","CRM","PANW",
    # High-beta growth (aggressive picks)
    "DDOG","NET","ZS","TSLA","MRVL","AXON","TTD","SNOW",
]

MACRO_TICKERS = ["DX-Y.NYB","^VIX","^TNX","HYG","QQQ","XLK","GLD","^GSPC","IWF"]


# ── Data Fetch ────────────────────────────────────────────

def fetch_data():
    print("  Fetching market data...")
    all_tickers = list(set(GROWTH_STOCKS + MACRO_TICKERS))
    raw = yf.download(all_tickers, period="2mo", interval="1d",
                      progress=False, auto_adjust=True, group_by="ticker")
    result = {}
    for tk in all_tickers:
        try:
            df = raw[tk].copy() if len(all_tickers) > 1 else raw.copy()
            df = df.dropna(subset=["Close","Open","High","Low"])
            if len(df) > 5:
                result[tk] = df
        except:
            pass
    print(f"  Loaded: {len(result)}/{len(all_tickers)} tickers")
    return result


def lc(df, n):
    c = df["Close"].dropna()
    return c.values[-n:] if len(c) >= n else None

def lv(df, n):
    v = df["Volume"].dropna()
    return v.values[-n:] if len(v) >= n else None

def pct(arr, back):
    if arr is None or len(arr) <= back: return 0.0
    b = float(arr[-(back+1)])
    return (float(arr[-1]) - b) / b * 100 if b > 0 else 0.0


# ── Macro Regime ──────────────────────────────────────────

def get_macro(all_data):
    def arr(tk, n):
        df = all_data.get(tk)
        return lc(df, n) if df is not None else None

    info = {}; score = 0

    # VIX
    v = arr("^VIX", 3)
    if v is not None:
        vix = float(v[-1]); info["vix"] = round(vix,1)
        if   vix < 15: score += 3
        elif vix < 18: score += 2
        elif vix < 22: score += 1
        elif vix < 28: score -= 1
        elif vix < 35: score -= 2
        else:          score -= 4

    # HYG
    h = arr("HYG", 6)
    if h is not None:
        h5 = pct(h,5); info["hyg_5d"] = round(h5,2)
        score += 2 if h5>0.5 else (1 if h5>0 else (-2 if h5<-1 else 0))

    # QQQ
    q = arr("QQQ", 11)
    if q is not None:
        q5 = pct(q,5); q10 = pct(q,10)
        info["qqq_5d"] = round(q5,2)
        score += 2 if q5>2 else (1 if q5>0 else (-2 if q5<-3 else 0))
        score += 1 if q10>4 else (-1 if q10<-5 else 0)

    # DXY
    d = arr("DX-Y.NYB", 6)
    if d is not None:
        info["dxy"] = round(float(d[-1]),2); info["dxy_5d"] = round(pct(d,5),2)
        score += -1 if pct(d,5)>1 else (1 if pct(d,5)<-1 else 0)

    # 10Y yield
    y = arr("^TNX", 6)
    if y is not None:
        info["yield_10y"] = round(float(y[-1]),2)
        score += -1 if pct(y,5)>5 else (1 if pct(y,5)<-5 else 0)

    # Gold
    g = arr("GLD", 6)
    if g is not None:
        info["gold_5d"] = round(pct(g,5),2)
        score += -1 if pct(g,5)>3 else (1 if pct(g,5)<-2 else 0)

    # S&P500
    sp = arr("^GSPC", 6)
    if sp is not None:
        info["sp500"] = round(float(sp[-1]),0)
        info["sp500_5d"] = round(pct(sp,5),2)

    info["score"] = score

    if   score >= 4:  regime = "STRONG_ON"
    elif score >= 1:  regime = "RISK_ON"
    elif score >= -1: regime = "NEUTRAL"
    elif score >= -3: regime = "RISK_OFF"
    else:             regime = "CRISIS"

    return regime, score, info


# ── Stock & Gold Scoring ──────────────────────────────────

def score_stock(tk, df, market_data):
    c2 = lc(df,2); c6 = lc(df,6); c11 = lc(df,11); c21 = lc(df,21)
    vol = lv(df,25)
    if c2 is None: return None
    c_now = float(c2[-1]); c_prev = float(c2[-2])
    if c_now<=0 or c_prev<=0: return None

    m1  = pct(c2,1)
    m5  = pct(c6,5)   if c6  is not None else 0
    m10 = pct(c11,10) if c11 is not None else 0
    m20 = pct(c21,20) if c21 is not None else 0

    vr = 1.0
    if vol is not None and len(vol)>=20:
        vr = float(vol[-5:].mean()) / float(vol[-20:].mean())

    above_10 = (c_now > float(c11.mean())) if c11 is not None else False
    above_20 = (c_now > float(c21.mean())) if c21 is not None else False

    # Relative strength vs QQQ
    qqq_df = market_data.get("QQQ")
    rs_bonus = 0
    if qqq_df is not None:
        q6 = lc(qqq_df,6)
        if q6 is not None:
            qqq_5d = pct(q6,5)
            if m5 > qqq_5d + 2: rs_bonus = 2
            elif m5 > qqq_5d:   rs_bonus = 1

    sc = 0
    sc += 3 if m1>3   else (2 if m1>1.5  else (1 if m1>0.5  else (-2 if m1<-1.5 else 0)))
    sc += 3 if m5>6   else (2 if m5>3    else (1 if m5>1    else (-2 if m5<-4   else 0)))
    sc += 2 if m10>10 else (1 if m10>5   else 0)
    sc += 2 if m20>15 else (1 if m20>8   else (-1 if m20<-5 else 0))
    sc += 1 if above_10 else 0
    sc += 1 if above_20 else 0
    sc += 2 if vr>2.0 else (1 if vr>1.3 else 0)
    sc += rs_bonus

    # Aggressive: bonus for high-beta names
    high_beta = {"TSLA","MRVL","AXON","TTD","SNOW","DDOG","NET"}
    if tk in high_beta and sc >= P["min_score"]: sc += 1

    strength = ("VERY STRONG" if sc>=11 else "STRONG" if sc>=8
                else "MODERATE" if sc>=5 else "WEAK" if sc>=2 else "BEARISH")

    return dict(ticker=tk, score=sc, strength=strength,
                price=round(c_now,2), prev_close=round(c_prev,2),
                mom_1d=round(m1,2), mom_5d=round(m5,2),
                mom_10d=round(m10,2), mom_20d=round(m20,2),
                vol_ratio=round(vr,2), above_20ma=above_20,
                rs_bonus=rs_bonus)


def score_gold(all_data):
    df = all_data.get("GLD")
    if df is None: return 0, {}
    c6=lc(df,6); c11=lc(df,11); c21=lc(df,21)
    if c6 is None: return 0, {}
    c_now = float(c6[-1])
    m5=pct(c6,5); m10=pct(c11,10) if c11 is not None else 0
    m20=pct(c21,20) if c21 is not None else 0
    above_20=(c_now>float(c21.mean())) if c21 is not None else False

    sc = 0
    sc += 3 if m5>3   else (2 if m5>1   else (1 if m5>0  else (-1 if m5<-2 else 0)))
    sc += 2 if m10>5  else (1 if m10>2  else 0)
    sc += 2 if m20>10 else (1 if m20>4  else 0)
    sc += 1 if above_20 else 0

    return sc, dict(price=round(c_now,2), mom_5d=round(m5,2),
                    mom_10d=round(m10,2), mom_20d=round(m20,2),
                    above_20ma=above_20)


# ── Action Plan ───────────────────────────────────────────

def build_actions(regime, mac_score, signals, gold_score, gold_info):
    actions = []
    total_pct = 0.0

    # Stock entries
    if regime in P["allow_regime"]:
        ranked = sorted(
            [s for s in signals if s and s["score"] >= P["min_score"]],
            key=lambda x: (x["score"], x["mom_5d"]), reverse=True
        )[:P["max_slots"]]

        for s in ranked:
            # Aggressive: scale position size by score
            multiplier = 1.2 if s["score"] >= 11 else 1.0
            alloc = min(P["stock_pct"] * multiplier, 0.50)
            usd   = PORTFOLIO_USD * alloc
            actions.append(dict(
                action="BUY", asset=s["ticker"], type="STOCK",
                price=s["price"], score=s["score"], strength=s["strength"],
                alloc_pct=round(alloc*100,1), alloc_usd=round(usd,0),
                shares=round(usd/s["price"],6) if s["price"]>0 else 0,
                sl_price=round(s["price"]*(1-P["sl"]),2),
                tp_price=round(s["price"]*(1+P["tp"]),2),
                trail_pct=int(P["trail"]*100),
                reasons=[
                    f"Momentum 1d: {s['mom_1d']:+.1f}% | 5d: {s['mom_5d']:+.1f}%",
                    f"Volume surge: {s['vol_ratio']:.1f}x avg",
                    f"Relative strength bonus: +{s['rs_bonus']}",
                    f"Above 20-day MA: {'YES' if s['above_20ma'] else 'NO'}",
                ],
                order_type="bracket",  # entry + SL + TP ใน order เดียว
            ))
            total_pct += alloc

    # Gold (always consider, even for aggressive profile)
    if gold_score >= 1 and total_pct < 0.85:
        if regime in ("RISK_OFF","CRISIS"):
            gold_alloc = P["gold_pct"] * 2.0
        elif regime == "NEUTRAL":
            gold_alloc = P["gold_pct"] * 1.0
        else:
            gold_alloc = P["gold_pct"] * 0.5  # aggressive = small gold hedge

        gold_alloc = min(gold_alloc, 1.0 - total_pct - 0.10)
        if gold_alloc > 0.05 and gold_info.get("price",0) > 0:
            usd = PORTFOLIO_USD * gold_alloc
            actions.append(dict(
                action="BUY", asset="GLD", type="GOLD",
                price=gold_info["price"], score=gold_score, strength="HEDGE",
                alloc_pct=round(gold_alloc*100,1), alloc_usd=round(usd,0),
                shares=round(usd/gold_info["price"],6),
                sl_price=round(gold_info["price"]*0.95,2),
                tp_price=round(gold_info["price"]*1.12,2),
                trail_pct=4,
                reasons=[
                    f"Gold 5d: {gold_info['mom_5d']:+.1f}% | 20d: {gold_info['mom_20d']:+.1f}%",
                    f"Above 20MA: {'YES' if gold_info.get('above_20ma') else 'NO'}",
                    f"Regime: {regime} → hedge allocation",
                ],
                order_type="bracket",
            ))
            total_pct += gold_alloc

    # Cash remainder
    cash_pct = max(0, 1.0 - total_pct)
    actions.append(dict(
        action="HOLD_CASH", asset="USD CASH", type="CASH",
        alloc_pct=round(cash_pct*100,1), alloc_usd=round(PORTFOLIO_USD*cash_pct,0),
        price=None, score=None, strength=None, shares=None,
        sl_price=None, tp_price=None, trail_pct=None, order_type=None,
        reasons=["รอสัญญาณดีกว่านี้", "Buffer ป้องกัน margin call"],
    ))

    return actions


# ── Alpaca Order Placement ────────────────────────────────

def place_orders_alpaca(actions):
    """
    ส่ง bracket orders ผ่าน Alpaca API
    Bracket order = 1 คำสั่ง ครอบคลุม Entry + Stop Loss + Take Profit
    """
    if not ALPACA_AVAILABLE:
        print("  [!] alpaca-trade-api not installed")
        print("      pip install alpaca-trade-api")
        return []

    if not ALPACA_KEY or not ALPACA_SECRET:
        print("  [!] Alpaca API keys not set in .env")
        print("      ALPACA_API_KEY=...")
        print("      ALPACA_SECRET_KEY=...")
        return []

    env_label = "LIVE" if ALPACA_LIVE else "PAPER"
    base_url  = ("https://api.alpaca.markets"
                 if ALPACA_LIVE else "https://paper-api.alpaca.markets")

    print(f"\n  Connecting to Alpaca [{env_label}]...")
    try:
        api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, base_url, api_version="v2")
        acct = api.get_account()
        buying_power = float(acct.buying_power)
        print(f"  Account: ${float(acct.equity):,.2f} equity | ${buying_power:,.2f} buying power")
    except Exception as e:
        print(f"  [!] Alpaca connection failed: {e}")
        return []

    # Cancel all pending orders first (clean slate)
    try:
        api.cancel_all_orders()
        print("  Cancelled pending orders")
    except:
        pass

    # Check market status
    clock = api.get_clock()
    is_open = clock.is_open
    next_open = clock.next_open
    print(f"  Market {'OPEN' if is_open else 'CLOSED'} | Next open: {next_open}")

    results = []
    buy_actions = [a for a in actions if a["action"] == "BUY"]

    for a in buy_actions:
        ticker = a["asset"]
        notional = a["alloc_usd"]  # dollar amount

        if notional < 1:
            continue

        print(f"\n  Placing order: {ticker} ${notional:,.0f} "
              f"[{a['order_type']}] SL=${a['sl_price']} TP=${a['tp_price']}")

        try:
            # Bracket order: market buy + stop loss + take profit
            order = api.submit_order(
                symbol=ticker,
                notional=notional,          # dollar-based (fractional shares)
                side="buy",
                type="market",
                time_in_force="day",        # expire at end of day
                order_class="bracket",
                stop_loss={"stop_price": str(a["sl_price"])},
                take_profit={"limit_price": str(a["tp_price"])},
            )

            result = dict(
                ticker=ticker,
                order_id=order.id,
                status=order.status,
                notional=notional,
                sl=a["sl_price"],
                tp=a["tp_price"],
                submitted_at=str(order.submitted_at),
                env=env_label,
            )
            results.append(result)
            print(f"  OK  {ticker}: order_id={order.id} status={order.status}")
            time.sleep(0.3)  # rate limit

        except Exception as e:
            err = dict(ticker=ticker, error=str(e), env=env_label)
            results.append(err)
            print(f"  FAIL {ticker}: {e}")

    return results


# ── Email ─────────────────────────────────────────────────

REGIME_CFG = {
    "STRONG_ON": ("27ae60","STRONG RISK-ON","ตลาดแข็งแกร่งมาก"),
    "RISK_ON":   ("2ecc71","RISK-ON",       "ตลาดขาขึ้น"),
    "NEUTRAL":   ("f39c12","NEUTRAL",        "ตลาดทรงตัว"),
    "RISK_OFF":  ("e67e22","RISK-OFF",       "ระวัง — เน้นทอง"),
    "CRISIS":    ("e74c3c","CRISIS",         "วิกฤต — ถือเงินสด"),
}

def build_email(regime, mac_score, macro_info, actions, signals, order_results):
    now_bkk = datetime.utcnow() + timedelta(hours=7)
    date_str = now_bkk.strftime("%A %d %B %Y  |  %H:%M น. (ไทย)")
    rc = REGIME_CFG.get(regime, ("95a5a6","?","?"))
    reg_color, reg_en, reg_th = rc

    # ── Profile badge
    profile_colors = {"CONSERVATIVE":"#3498db","MODERATE":"#f39c12","AGGRESSIVE":"#e74c3c"}
    p_color = profile_colors.get(RISK_PROFILE,"#95a5a6")

    # ── Order status section
    order_html = ""
    if order_results:
        rows = ""
        for r in order_results:
            if "error" in r:
                rows += f"<tr><td style='padding:5px 8px;color:#e74c3c'>{r['ticker']}</td><td colspan='4' style='color:#e74c3c'>{r['error']}</td></tr>"
            else:
                rows += f"""<tr>
                  <td style='padding:5px 8px;font-weight:bold'>{r['ticker']}</td>
                  <td style='padding:5px 8px'>${r['notional']:,.0f}</td>
                  <td style='padding:5px 8px;color:#e74c3c'>${r['sl']}</td>
                  <td style='padding:5px 8px;color:#27ae60'>${r['tp']}</td>
                  <td style='padding:5px 8px'><span style='background:#27ae60;color:white;
                    padding:2px 8px;border-radius:4px;font-size:11px'>{r['status'].upper()}</span></td>
                </tr>"""
        env = order_results[0].get("env","PAPER")
        env_color = "#e74c3c" if env=="LIVE" else "#3498db"
        order_html = f"""
        <div style='margin:16px 0;border:2px solid {env_color};border-radius:10px;overflow:hidden'>
          <div style='background:{env_color};color:white;padding:10px 16px;font-weight:bold'>
            Orders Placed via Alpaca [{env}]
          </div>
          <table style='width:100%;border-collapse:collapse;font-size:13px;background:white'>
            <tr style='background:#f8f9fa'>
              <th style='padding:6px 8px;text-align:left'>Ticker</th>
              <th style='padding:6px 8px;text-align:left'>Amount</th>
              <th style='padding:6px 8px;text-align:left'>Stop Loss</th>
              <th style='padding:6px 8px;text-align:left'>Take Profit</th>
              <th style='padding:6px 8px;text-align:left'>Status</th>
            </tr>
            {rows}
          </table>
        </div>"""
    else:
        order_html = """<div style='background:#fef9e7;border:1px solid #f39c12;
          border-radius:8px;padding:12px;margin:12px 0;font-size:13px;color:#856404'>
          Alpaca ไม่ได้ตั้งค่า — ดู action plan แล้วสั่งเองในแอป broker
        </div>"""

    # ── Action cards
    cards = ""
    for a in actions:
        if a["type"] == "CASH":
            bg, hdr = "#f8f9fa", "#7f8c8d"
        elif a["type"] == "GOLD":
            bg, hdr = "#fef9e7", "#e67e22"
        else:
            bg, hdr = "#eafaf1", "#27ae60"

        reasons_html = "".join(f"<li style='margin:3px 0;font-size:12px'>{r}</li>"
                               for r in (a.get("reasons") or []))

        if a["type"] == "CASH":
            body = f"""<div style='font-size:20px;font-weight:bold;color:#7f8c8d'>
                        {a['alloc_pct']}% = ${a['alloc_usd']:,.0f}
                       </div>
                       <p style='font-size:12px;color:#888'>รอโอกาส / ลด drawdown</p>"""
        else:
            body = f"""
            <table style='width:100%;font-size:12px;border-collapse:collapse'>
              <tr>
                <td style='padding:3px 8px;color:#666'>ราคาปัจจุบัน</td>
                <td style='font-weight:bold'>${a['price']:,.2f}</td>
                <td style='padding:3px 8px;color:#666'>จัดสรร</td>
                <td style='font-weight:bold'>{a['alloc_pct']}% = ${a['alloc_usd']:,.0f}</td>
              </tr><tr>
                <td style='padding:3px 8px;color:#666'>จำนวนหุ้น</td>
                <td style='font-weight:bold'>~{a['shares']:.4f} หุ้น</td>
                <td style='padding:3px 8px;color:#666'>Score</td>
                <td style='font-weight:bold'>{a['score']} ({a['strength']})</td>
              </tr><tr>
                <td style='padding:3px 8px;color:#e74c3c'>Stop Loss (-{P['sl']*100:.0f}%)</td>
                <td style='color:#e74c3c;font-weight:bold'>${a['sl_price']}</td>
                <td style='padding:3px 8px;color:#27ae60'>Take Profit (+{P['tp']*100:.0f}%)</td>
                <td style='color:#27ae60;font-weight:bold'>${a['tp_price']}</td>
              </tr><tr>
                <td style='padding:3px 8px;color:#666'>Trailing Stop</td>
                <td colspan='3' style='font-weight:bold'>{a['trail_pct']}% จาก peak (ล็อกกำไรอัตโนมัติ)</td>
              </tr>
            </table>
            <ul style='margin:8px 0;padding-left:18px'>{reasons_html}</ul>"""

        cards += f"""
        <div style='border-radius:10px;margin:10px 0;overflow:hidden;
                    box-shadow:0 2px 6px rgba(0,0,0,0.08)'>
          <div style='background:{hdr};padding:10px 16px;color:white;
                      display:flex;justify-content:space-between;align-items:center'>
            <span style='font-size:20px;font-weight:bold'>{a["action"]}  {a["asset"]}</span>
            <span style='opacity:0.85;font-size:12px'>{a["type"]}</span>
          </div>
          <div style='background:{bg};padding:12px 16px'>{body}</div>
        </div>"""

    # ── Watchlist table
    top = sorted([s for s in signals if s], key=lambda x:x["score"], reverse=True)[:10]
    watch_rows = ""
    for s in top:
        mc = "#27ae60" if s["mom_1d"]>0 else "#e74c3c"
        m5c= "#27ae60" if s["mom_5d"]>0 else "#e74c3c"
        tag_c = "#27ae60" if s["score"]>=P["min_score"] else "#95a5a6"
        tag   = "BUY"   if s["score"]>=P["min_score"]   else "WATCH"
        bar   = min(int(max(s["score"],0)/14*70),70)
        watch_rows += f"""<tr style='border-bottom:1px solid #eee'>
          <td style='padding:5px 8px;font-weight:bold'>{s['ticker']}</td>
          <td style='padding:5px 8px'>${s['price']:,.2f}</td>
          <td style='padding:5px 8px;color:{mc}'>{s['mom_1d']:+.2f}%</td>
          <td style='padding:5px 8px;color:{m5c}'>{s['mom_5d']:+.2f}%</td>
          <td style='padding:5px 8px'>{s['vol_ratio']:.1f}x</td>
          <td style='padding:5px 8px'>{s['score']}</td>
          <td style='padding:5px 8px'>
            <span style='background:{tag_c};color:white;padding:2px 7px;
                         border-radius:4px;font-size:11px'>{tag}</span>
          </td>
        </tr>"""

    # ── Macro table
    macro_rows = ""
    for label, key, suffix in [
        ("S&P 500","sp500",""), ("VIX","vix",""), ("10Y Yield","yield_10y","%"),
        ("DXY","dxy",""), ("HYG 5d","hyg_5d","%"), ("QQQ 5d","qqq_5d","%"),
        ("Gold 5d","gold_5d","%"),
    ]:
        val = macro_info.get(key,"N/A")
        macro_rows += f"<tr><td style='padding:5px 10px;color:#666'>{label}</td><td style='padding:5px 10px;font-weight:bold'>{val}{suffix}</td></tr>"

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'/></head>
<body style='font-family:Segoe UI,Arial,sans-serif;max-width:700px;margin:auto;
             color:#333;padding:20px;background:#f5f6fa'>

  <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;
              padding:22px;border-radius:12px;margin-bottom:16px'>
    <div style='display:flex;justify-content:space-between;align-items:flex-start'>
      <div>
        <p style='margin:0 0 4px;font-size:12px;opacity:0.7'>
          ซื้อตอนตลาดเปิด 9:30 AM ET | 20:30 น. ไทย
        </p>
        <h1 style='margin:0;font-size:22px'>Morning Signal</h1>
        <p style='margin:4px 0 0;font-size:12px;opacity:0.7'>{date_str}</p>
      </div>
      <div style='text-align:right'>
        <span style='background:{p_color};color:white;padding:4px 12px;
                     border-radius:12px;font-size:12px;font-weight:bold'>
          {RISK_PROFILE}
        </span>
        <div style='font-size:11px;opacity:0.6;margin-top:4px'>
          SL={int(P["sl"]*100)}% | TP={int(P["tp"]*100)}% | Trail={int(P["trail"]*100)}%
        </div>
      </div>
    </div>
  </div>

  <div style='background:#{reg_color};color:white;padding:14px 20px;
              border-radius:10px;margin-bottom:16px;
              display:flex;justify-content:space-between;align-items:center'>
    <div>
      <div style='font-size:12px;opacity:0.85'>MACRO REGIME</div>
      <div style='font-size:26px;font-weight:bold'>{reg_en}</div>
      <div style='font-size:13px;opacity:0.9'>{reg_th}</div>
    </div>
    <div style='text-align:right'>
      <div style='font-size:42px;font-weight:bold'>{mac_score:+d}</div>
      <div style='font-size:11px;opacity:0.8'>Macro Score</div>
    </div>
  </div>

  <h2 style='border-left:5px solid #{reg_color};padding-left:10px;color:#1a1a2e'>
    Action Plan — ซื้อที่ Open Price
  </h2>
  {order_html}
  {cards}

  <h2 style='border-left:5px solid #3498db;padding-left:10px;color:#1a1a2e;margin-top:24px'>
    Macro Dashboard
  </h2>
  <table style='width:100%;border-collapse:collapse;font-size:13px;
                background:white;border-radius:8px;overflow:hidden'>
    {macro_rows}
  </table>

  <h2 style='border-left:5px solid #9b59b6;padding-left:10px;color:#1a1a2e;margin-top:24px'>
    Growth Stock Watchlist
  </h2>
  <table style='width:100%;border-collapse:collapse;font-size:13px;
                background:white;border-radius:8px;overflow:hidden'>
    <tr style='background:#f8f9fa'>
      <th style='padding:7px 8px;text-align:left'>Ticker</th>
      <th style='padding:7px 8px;text-align:left'>Price</th>
      <th style='padding:7px 8px;text-align:left'>1D</th>
      <th style='padding:7px 8px;text-align:left'>5D</th>
      <th style='padding:7px 8px;text-align:left'>Vol</th>
      <th style='padding:7px 8px;text-align:left'>Score</th>
      <th style='padding:7px 8px;text-align:left'>Signal</th>
    </tr>
    {watch_rows}
  </table>

  <div style='background:#fdf2f8;border:1px solid #e74c3c;border-radius:8px;
              padding:14px;margin-top:20px;font-size:12px'>
    <b style='color:#e74c3c'>Risk Profile: {RISK_PROFILE}</b><br>
    SL={int(P["sl"]*100)}% | TP={int(P["tp"]*100)}% | Trailing={int(P["trail"]*100)}% |
    Max {P["max_slots"]} positions | {int(P["stock_pct"]*100)}% per slot<br>
    <span style='color:#888'>ข้อมูลจาก Yahoo Finance | ไม่ใช่คำแนะนำการลงทุน</span>
  </div>
</body></html>"""

    return html


def build_plain(regime, mac_score, actions, order_results):
    lines = [
        f"=== Morning Signal [{RISK_PROFILE}] | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} ===",
        f"Regime: {regime} (score={mac_score:+d})",
        f"SL={int(P['sl']*100)}%  TP={int(P['tp']*100)}%  Trail={int(P['trail']*100)}%",
        "", "ACTION PLAN (9:30 AM ET / 20:30 ไทย):",
    ]
    for a in actions:
        if a["type"] == "CASH":
            lines.append(f"  CASH   {a['alloc_pct']}% = ${a['alloc_usd']:,.0f}")
        else:
            lines.append(f"  {a['action']} {a['asset']:<5}  {a['alloc_pct']}% = ${a['alloc_usd']:,.0f}"
                         f"  ~{a['shares']:.4f} shares @ ~${a['price']:,.2f}"
                         f"  | SL=${a['sl_price']}  TP=${a['tp_price']}")
    if order_results:
        lines.append(""); lines.append("ALPACA ORDERS:")
        for r in order_results:
            if "error" in r:
                lines.append(f"  FAIL {r['ticker']}: {r['error']}")
            else:
                lines.append(f"  OK   {r['ticker']} [{r['env']}] id={r['order_id']} {r['status']}")
    return "\n".join(lines)


def send_email(subject, html, plain):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("  [!] Email not configured (EMAIL_SENDER / EMAIL_PASSWORD in .env)")
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_RECIPIENT
        msg.attach(MIMEText(plain,"plain","utf-8"))
        msg.attach(MIMEText(html, "html", "utf-8"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as srv:
            srv.ehlo(); srv.starttls()
            srv.login(EMAIL_SENDER, EMAIL_PASSWORD)
            srv.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
        print(f"  Email sent -> {EMAIL_RECIPIENT}")
    except Exception as e:
        print(f"  Email error: {e}")


# ── Main ──────────────────────────────────────────────────

def run():
    print("\n" + "="*62)
    print(f"  Morning Signal [{RISK_PROFILE}] -- Stocks + Gold + Cash")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | Portfolio: ${PORTFOLIO_USD:,.0f}")
    print("="*62)

    # 1. Fetch data
    print("\n[1/4] Fetching data...")
    all_data = fetch_data()

    # 2. Macro
    print("[2/4] Macro analysis...")
    regime, mac_score, macro_info = get_macro(all_data)
    print(f"  Regime: {regime} ({mac_score:+d}) | VIX={macro_info.get('vix','?')}")

    # 3. Signals
    print("[3/4] Signal scoring...")
    signals = [score_stock(tk, all_data[tk], all_data)
               for tk in GROWTH_STOCKS if tk in all_data]
    signals = [s for s in signals if s is not None]

    gold_score, gold_info = score_gold(all_data)
    eligible = sum(1 for s in signals if s["score"] >= P["min_score"])
    print(f"  Stocks eligible (score>={P['min_score']}): {eligible}/{len(signals)}")
    print(f"  Gold score: {gold_score} | Price: ${gold_info.get('price',0):,.2f}")

    actions = build_actions(regime, mac_score, signals, gold_score, gold_info)

    # Print action plan
    print(f"\n{'='*62}")
    print(f"  ACTION PLAN  |  {RISK_PROFILE}  |  SL={int(P['sl']*100)}%  TP={int(P['tp']*100)}%")
    print(f"{'='*62}")
    for a in actions:
        if a["type"] == "CASH":
            print(f"  [CASH ] {a['alloc_pct']}% = ${a['alloc_usd']:,.0f}")
        elif a["type"] == "GOLD":
            print(f"  [GOLD ] BUY GLD  {a['alloc_pct']}% = ${a['alloc_usd']:,.0f} "
                  f"~{a['shares']:.4f}sh @ ${a['price']:,.2f} | "
                  f"SL=${a['sl_price']} TP=${a['tp_price']} Trail={a['trail_pct']}%")
        else:
            print(f"  [STOCK] BUY {a['asset']:<5} {a['alloc_pct']}% = ${a['alloc_usd']:,.0f} "
                  f"~{a['shares']:.4f}sh @ ${a['price']:,.2f} | "
                  f"SL=${a['sl_price']} TP=${a['tp_price']} Trail={a['trail_pct']}%")
    print(f"{'='*62}")

    # 4. Place orders
    print("\n[4/4] Placing orders (Alpaca)...")
    order_results = place_orders_alpaca(actions)

    # Save snapshot
    snapshot = dict(
        datetime_utc=datetime.utcnow().isoformat(),
        risk_profile=RISK_PROFILE,
        regime=regime, macro_score=mac_score, macro_info=macro_info,
        actions=actions, order_results=order_results,
        gold_score=gold_score, gold_info=gold_info,
        signals=sorted(signals, key=lambda x:x["score"], reverse=True),
    )
    with open("morning_signal_latest.json","w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    # Email
    buys = [a for a in actions if a["action"]=="BUY"]
    names = " + ".join(a["asset"] for a in buys) if buys else "CASH ONLY"
    subject = f"[{RISK_PROFILE}] Morning Signal {datetime.utcnow().strftime('%d %b %Y')} | {regime} | {names}"

    html  = build_email(regime, mac_score, macro_info, actions, signals, order_results)
    plain = build_plain(regime, mac_score, actions, order_results)
    send_email(subject, html, plain)

    print("\n  Done. Snapshot -> morning_signal_latest.json\n")


if __name__ == "__main__":
    run()
