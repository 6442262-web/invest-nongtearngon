# -*- coding: utf-8 -*-
"""
Morning Signal -- 3-Asset (Stocks + Gold + Cash)
รันก่อนตลาดเปิด แล้วส่งอีเมลบอกว่า "ซื้ออะไร / ถือทอง / ถือเงินสด"
US Market Open: 9:30 AM ET = 20:30 น. ไทย

Usage:
  python morning_signal.py              # run once now
  python morning_signal.py --test       # test email without live data
"""

import os, sys, json, smtplib, warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────
EMAIL_SENDER    = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD  = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "6442262@schoolptk.ac.th")
SMTP_HOST       = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT       = int(os.getenv("SMTP_PORT", "587"))

PORTFOLIO_USD   = float(os.getenv("PORTFOLIO_USD", "5000"))

# Allocation limits
MAX_STOCK_SLOTS  = 2
STOCK_ALLOC_PCT  = 0.35     # 35% per stock
GOLD_ALLOC_PCT   = 0.25     # up to 25% gold
# rest = cash

# Thresholds
MIN_STOCK_SCORE  = 4
MIN_GOLD_SCORE   = 2

GROWTH_STOCKS = [
    "NVDA","MSFT","AAPL","META","GOOGL",
    "AMZN","TSLA","AVGO","NOW",
    "ADBE","CRM","PANW","DDOG","NET","ZS",
]

MACRO_TICKERS = ["DX-Y.NYB","^VIX","^TNX","HYG","QQQ","XLK","GLD","^GSPC"]

# ── Data Fetch ────────────────────────────────────────────

def fetch(tickers, period="1mo"):
    raw = yf.download(tickers, period=period, interval="1d",
                      progress=False, auto_adjust=True, group_by="ticker")
    result = {}
    for tk in tickers:
        try:
            df = raw[tk].copy() if len(tickers) > 1 else raw.copy()
            df = df.dropna(subset=["Close","Open","High","Low"])
            if len(df) > 5:
                result[tk] = df
        except:
            pass
    return result


def last_c(df, n):
    c = df["Close"].dropna()
    return c.values[-n:] if len(c) >= n else None

def last_v(df, n):
    v = df["Volume"].dropna()
    return v.values[-n:] if len(v) >= n else None

def pct(arr, back):
    if arr is None or len(arr) <= back: return 0.0
    b = float(arr[-(back+1)])
    return (float(arr[-1]) - b) / b * 100 if b > 0 else 0.0


# ── Macro Regime ──────────────────────────────────────────

def get_macro(macro_data):
    info  = {}
    score = 0

    # VIX
    vix_arr = last_c(macro_data.get("^VIX", pd.DataFrame()), 3)
    if vix_arr is not None:
        vix = float(vix_arr[-1])
        info["vix"] = round(vix, 1)
        if   vix < 15: score += 3
        elif vix < 18: score += 2
        elif vix < 22: score += 1
        elif vix < 28: score -= 1
        elif vix < 35: score -= 2
        else:          score -= 4

    # HYG (credit / risk appetite)
    hyg_arr = last_c(macro_data.get("HYG", pd.DataFrame()), 6)
    if hyg_arr is not None:
        h5 = pct(hyg_arr, 5)
        info["hyg_5d"] = round(h5, 2)
        if   h5 >  0.5: score += 2
        elif h5 >  0:   score += 1
        elif h5 < -1:   score -= 2

    # QQQ trend
    qqq_arr = last_c(macro_data.get("QQQ", pd.DataFrame()), 11)
    if qqq_arr is not None:
        q5  = pct(qqq_arr, 5)
        q10 = pct(qqq_arr, 10)
        info["qqq_5d"] = round(q5, 2)
        if   q5 >  2: score += 2
        elif q5 >  0: score += 1
        elif q5 < -3: score -= 2
        if   q10 > 4: score += 1
        elif q10 < -5:score -= 1

    # DXY
    dxy_arr = last_c(macro_data.get("DX-Y.NYB", pd.DataFrame()), 6)
    if dxy_arr is not None:
        d5 = pct(dxy_arr, 5)
        d1 = pct(dxy_arr, 1)
        info["dxy"] = round(float(dxy_arr[-1]), 2)
        info["dxy_5d"] = round(d5, 2)
        if d5 >  1: score -= 1
        if d5 < -1: score += 1

    # 10Y yield
    y10_arr = last_c(macro_data.get("^TNX", pd.DataFrame()), 6)
    if y10_arr is not None:
        info["yield_10y"] = round(float(y10_arr[-1]), 2)
        y5 = pct(y10_arr, 5)
        if y5 >  5: score -= 1   # yields rising fast = valuation pressure
        if y5 < -5: score += 1

    # Gold trend
    gld_arr = last_c(macro_data.get("GLD", pd.DataFrame()), 6)
    if gld_arr is not None:
        g5 = pct(gld_arr, 5)
        info["gold_5d"] = round(g5, 2)
        if g5 >  3: score -= 1
        if g5 < -2: score += 1

    # S&P500
    sp_arr = last_c(macro_data.get("^GSPC", pd.DataFrame()), 6)
    if sp_arr is not None:
        info["sp500"] = round(float(sp_arr[-1]), 0)
        info["sp500_5d"] = round(pct(sp_arr, 5), 2)

    info["score"] = score

    if   score >= 4:  regime = "STRONG_ON";  color = "#27ae60"
    elif score >= 1:  regime = "RISK_ON";    color = "#2ecc71"
    elif score >= -1: regime = "NEUTRAL";    color = "#f39c12"
    elif score >= -3: regime = "RISK_OFF";   color = "#e67e22"
    else:             regime = "CRISIS";     color = "#e74c3c"

    return regime, score, color, info


# ── Stock Scoring ─────────────────────────────────────────

def score_stock(tk, df):
    c2  = last_c(df, 2);  c6  = last_c(df, 6)
    c11 = last_c(df, 11); c21 = last_c(df, 21)
    vol = last_v(df, 25)

    if c2 is None: return None
    c_now = float(c2[-1]); c_prev = float(c2[-2])
    if c_now <= 0 or c_prev <= 0: return None

    m1  = pct(c2,  1)
    m5  = pct(c6,  5)  if c6  is not None else 0
    m10 = pct(c11, 10) if c11 is not None else 0
    m20 = pct(c21, 20) if c21 is not None else 0

    vr = 1.0
    if vol is not None and len(vol) >= 20:
        vr = float(vol[-5:].mean()) / float(vol[-20:].mean())

    above_10ma = (c_now > float(c11.mean())) if c11 is not None else False
    above_20ma = (c_now > float(c21.mean())) if c21 is not None else False

    sc = 0
    if   m1  >  3:  sc += 3
    elif m1  >  1.5:sc += 2
    elif m1  >  0.5:sc += 1
    elif m1  < -1.5:sc -= 2

    if   m5  >  6:  sc += 3
    elif m5  >  3:  sc += 2
    elif m5  >  1:  sc += 1
    elif m5  < -4:  sc -= 2

    if   m10 > 10:  sc += 2
    elif m10 >  5:  sc += 1

    if   m20 > 15:  sc += 2
    elif m20 >  8:  sc += 1
    elif m20 < -5:  sc -= 1

    if above_10ma:  sc += 1
    if above_20ma:  sc += 1
    if   vr > 2.0:  sc += 2
    elif vr > 1.3:  sc += 1

    # Signal strength label
    if   sc >= 10: strength = "VERY STRONG"
    elif sc >= 7:  strength = "STRONG"
    elif sc >= 4:  strength = "MODERATE"
    elif sc >= 1:  strength = "WEAK"
    else:          strength = "BEARISH"

    return {
        "ticker":    tk,
        "score":     sc,
        "strength":  strength,
        "price":     round(c_now, 2),
        "mom_1d":    round(m1, 2),
        "mom_5d":    round(m5, 2),
        "mom_10d":   round(m10, 2),
        "mom_20d":   round(m20, 2),
        "vol_ratio": round(vr, 2),
        "above_20ma":above_20ma,
    }


def score_gold(gld_df):
    c6  = last_c(gld_df, 6)
    c11 = last_c(gld_df, 11)
    c21 = last_c(gld_df, 21)
    if c6 is None: return 0, {}

    c_now = float(c6[-1])
    m5  = pct(c6,  5)
    m10 = pct(c11, 10) if c11 is not None else 0
    m20 = pct(c21, 20) if c21 is not None else 0
    above_20ma = (c_now > float(c21.mean())) if c21 is not None else False

    sc = 0
    if   m5  >  3: sc += 3
    elif m5  >  1: sc += 2
    elif m5  >  0: sc += 1
    elif m5  < -2: sc -= 1
    if   m10 >  5: sc += 2
    elif m10 >  2: sc += 1
    if   m20 > 10: sc += 2
    elif m20 >  4: sc += 1
    if above_20ma: sc += 1

    return sc, {
        "price":   round(c_now, 2),
        "mom_5d":  round(m5, 2),
        "mom_10d": round(m10, 2),
        "mom_20d": round(m20, 2),
        "above_20ma": above_20ma,
    }


# ── Action Plan Builder ───────────────────────────────────

def build_action_plan(regime, mac_score, stock_signals, gold_score, gold_info, macro_info):
    """
    Returns list of ACTION dicts:
    { action, asset, type, alloc_pct, alloc_usd, reason }
    """
    actions = []
    total_alloc = 0.0

    # ── Stock actions
    buy_stocks = regime not in ("RISK_OFF", "CRISIS")
    if buy_stocks:
        ranked = sorted(
            [s for s in stock_signals if s and s["score"] >= MIN_STOCK_SCORE],
            key=lambda x: x["score"], reverse=True
        )[:MAX_STOCK_SLOTS]

        for s in ranked:
            alloc = STOCK_ALLOC_PCT
            usd   = PORTFOLIO_USD * alloc
            actions.append({
                "action":    "BUY",
                "asset":     s["ticker"],
                "type":      "STOCK",
                "price":     s["price"],
                "score":     s["score"],
                "strength":  s["strength"],
                "alloc_pct": round(alloc*100, 0),
                "alloc_usd": round(usd, 0),
                "shares_approx": round(usd / s["price"], 4) if s["price"] > 0 else 0,
                "reasons": [
                    f"Momentum 1d: {s['mom_1d']:+.1f}%",
                    f"Momentum 5d: {s['mom_5d']:+.1f}%",
                    f"Volume surge: {s['vol_ratio']:.1f}x",
                    f"Above 20-day MA: {'YES' if s['above_20ma'] else 'NO'}",
                ],
                "sl_price":  round(s["price"] * 0.93, 2),
                "tp_price":  round(s["price"] * 1.15, 2),
                "trail_pct": 5,
            })
            total_alloc += alloc

    # ── Gold action
    gold_df_present = gold_info.get("price", 0) > 0
    buy_gold = False
    gold_reason = ""

    if regime in ("RISK_OFF", "CRISIS"):
        buy_gold = True
        gold_reason = f"Regime={regime}: defensive allocation"
    elif gold_score >= MIN_GOLD_SCORE and total_alloc < 0.75:
        buy_gold = True
        gold_reason = f"Gold momentum strong (score={gold_score}, 5d={gold_info.get('mom_5d',0):+.1f}%)"
    elif mac_score < 0 and gold_score >= 1:
        buy_gold = True
        gold_reason = f"Macro weak (score={mac_score}), gold hedge"

    if buy_gold and gold_df_present:
        alloc = GOLD_ALLOC_PCT if regime not in ("RISK_OFF","CRISIS") else 0.40
        alloc = min(alloc, 1.0 - total_alloc - 0.10)  # keep min 10% cash
        if alloc > 0.05:
            usd = PORTFOLIO_USD * alloc
            actions.append({
                "action":    "BUY",
                "asset":     "GLD",
                "type":      "GOLD",
                "price":     gold_info.get("price", 0),
                "score":     gold_score,
                "strength":  "HEDGE" if regime in ("RISK_OFF","CRISIS") else "MOMENTUM",
                "alloc_pct": round(alloc*100, 0),
                "alloc_usd": round(usd, 0),
                "shares_approx": round(usd / gold_info["price"], 4) if gold_info.get("price",0) > 0 else 0,
                "reasons": [gold_reason,
                             f"Gold 5d: {gold_info.get('mom_5d',0):+.1f}%",
                             f"Gold 20d: {gold_info.get('mom_20d',0):+.1f}%"],
                "sl_price":  round(gold_info["price"] * 0.95, 2),
                "tp_price":  round(gold_info["price"] * 1.12, 2),
                "trail_pct": 4,
            })
            total_alloc += alloc

    # ── Cash remainder
    cash_pct = max(0, 1.0 - total_alloc)
    actions.append({
        "action":    "HOLD_CASH",
        "asset":     "USD CASH",
        "type":      "CASH",
        "alloc_pct": round(cash_pct*100, 0),
        "alloc_usd": round(PORTFOLIO_USD * cash_pct, 0),
        "reasons":   ["Buffer for better entries", "Reduces drawdown risk"],
        "price": None, "score": None, "strength": None,
        "shares_approx": None, "sl_price": None, "tp_price": None, "trail_pct": None,
    })

    return actions


# ── HTML Email Builder ────────────────────────────────────

REGIME_CONFIG = {
    "STRONG_ON": ("27ae60", "STRONG RISK-ON", "ตลาดแข็งแกร่ง"),
    "RISK_ON":   ("2ecc71", "RISK-ON",        "ตลาดขาขึ้น"),
    "NEUTRAL":   ("f39c12", "NEUTRAL",         "ตลาดทรงตัว"),
    "RISK_OFF":  ("e67e22", "RISK-OFF",        "ระวัง — เน้นทอง/เงินสด"),
    "CRISIS":    ("e74c3c", "CRISIS",          "ตลาดวิกฤต — ถือเงินสด"),
}

def build_email(regime, mac_score, macro_info, actions, stock_signals):
    now_bkk = datetime.utcnow() + timedelta(hours=7)
    date_str = now_bkk.strftime("%A %d %B %Y  |  %H:%M น. (ไทย)")
    rc = REGIME_CONFIG.get(regime, ("95a5a6","?","?"))
    reg_color, reg_en, reg_th = rc

    # ── Action cards
    action_cards = ""
    for a in actions:
        if a["action"] == "HOLD_CASH":
            card_color = "#ecf0f1"
            header_bg  = "#7f8c8d"
            badge = "CASH"
            badge_bg = "#95a5a6"
        elif a["type"] == "GOLD":
            card_color = "#fef9e7"
            header_bg  = "#f39c12"
            badge = "GOLD"
            badge_bg = "#f39c12"
        else:
            card_color = "#eafaf1"
            header_bg  = "#27ae60"
            badge = "STOCK"
            badge_bg = "#27ae60"

        reasons_html = "".join(
            f"<li style='margin:3px 0'>{r}</li>" for r in a.get("reasons",[])
        )

        if a["action"] == "HOLD_CASH":
            detail = f"""
            <div style='font-size:22px;font-weight:bold;color:#7f8c8d'>
              {a['alloc_pct']:.0f}% = ${a['alloc_usd']:,.0f}
            </div>
            <p style='color:#666;margin:6px 0'>รอโอกาส / ลด drawdown</p>
            """
        else:
            detail = f"""
            <table style='width:100%;font-size:13px;border-collapse:collapse'>
              <tr>
                <td style='padding:3px 8px;color:#666'>ราคาปัจจุบัน</td>
                <td style='padding:3px 8px;font-weight:bold'>${a['price']:,.2f}</td>
                <td style='padding:3px 8px;color:#666'>จำนวนหุ้นโดยประมาณ</td>
                <td style='padding:3px 8px;font-weight:bold'>{a['shares_approx']} หุ้น</td>
              </tr>
              <tr>
                <td style='padding:3px 8px;color:#666'>จัดสรร</td>
                <td style='padding:3px 8px;font-weight:bold'>{a['alloc_pct']:.0f}% = ${a['alloc_usd']:,.0f}</td>
                <td style='padding:3px 8px;color:#666'>Signal Score</td>
                <td style='padding:3px 8px;font-weight:bold'>{a['score']} ({a['strength']})</td>
              </tr>
              <tr>
                <td style='padding:3px 8px;color:#e74c3c'>Stop Loss</td>
                <td style='padding:3px 8px;color:#e74c3c;font-weight:bold'>${a['sl_price']:,.2f}</td>
                <td style='padding:3px 8px;color:#27ae60'>Take Profit</td>
                <td style='padding:3px 8px;color:#27ae60;font-weight:bold'>${a['tp_price']:,.2f}</td>
              </tr>
              <tr>
                <td style='padding:3px 8px;color:#666'>Trailing Stop</td>
                <td style='padding:3px 8px;font-weight:bold' colspan='3'>{a['trail_pct']}% จาก peak</td>
              </tr>
            </table>
            """

        action_cards += f"""
        <div style='border:2px solid #{reg_color};border-radius:10px;
                    margin:12px 0;overflow:hidden'>
          <div style='background:#{header_bg};padding:10px 16px;
                      display:flex;justify-content:space-between;align-items:center'>
            <div>
              <span style='color:white;font-size:18px;font-weight:bold'>{a['action']}</span>
              <span style='color:white;font-size:22px;font-weight:bold;margin-left:12px'>{a['asset']}</span>
            </div>
            <span style='background:rgba(255,255,255,0.2);color:white;
                         padding:3px 10px;border-radius:12px;font-size:12px'>{badge}</span>
          </div>
          <div style='background:{card_color};padding:12px 16px'>
            {detail}
            <ul style='margin:8px 0;padding-left:20px;font-size:12px;color:#555'>
              {reasons_html}
            </ul>
          </div>
        </div>
        """

    # ── Macro summary table
    macro_rows = ""
    macro_display = [
        ("S&P 500",   macro_info.get("sp500","N/A"), macro_info.get("sp500_5d")),
        ("VIX",       macro_info.get("vix","N/A"),   None),
        ("10Y Yield", f"{macro_info.get('yield_10y','N/A')}%", None),
        ("DXY",       macro_info.get("dxy","N/A"),   macro_info.get("dxy_5d")),
        ("HYG 5d",    f"{macro_info.get('hyg_5d','N/A')}%", None),
        ("QQQ 5d",    f"{macro_info.get('qqq_5d','N/A')}%", None),
        ("Gold 5d",   f"{macro_info.get('gold_5d','N/A')}%", None),
    ]
    for label, val, change in macro_display:
        change_html = ""
        if change is not None:
            c_color = "#27ae60" if float(change) > 0 else "#e74c3c"
            change_html = f"<span style='color:{c_color};margin-left:8px'>({change:+.2f}%)</span>"
        macro_rows += f"""
        <tr>
          <td style='padding:5px 10px;color:#666'>{label}</td>
          <td style='padding:5px 10px;font-weight:bold'>{val}{change_html}</td>
        </tr>
        """

    # ── Top stock watchlist
    watch_rows = ""
    top_stocks = sorted(
        [s for s in stock_signals if s is not None],
        key=lambda x: x["score"], reverse=True
    )[:8]
    for s in top_stocks:
        bar_w = min(int(max(s["score"],0)/12*80), 80)
        s_color = "#27ae60" if s["score"] >= MIN_STOCK_SCORE else "#e74c3c"
        s_tag   = "BUY" if s["score"] >= MIN_STOCK_SCORE else "WATCH"
        m1_col  = "#27ae60" if s["mom_1d"] > 0 else "#e74c3c"
        m5_col  = "#27ae60" if s["mom_5d"] > 0 else "#e74c3c"
        watch_rows += f"""
        <tr style='border-bottom:1px solid #eee'>
          <td style='padding:6px 10px;font-weight:bold'>{s['ticker']}</td>
          <td style='padding:6px 10px'>${s['price']:,.2f}</td>
          <td style='padding:6px 10px;color:{m1_col}'>{s['mom_1d']:+.2f}%</td>
          <td style='padding:6px 10px;color:{m5_col}'>{s['mom_5d']:+.2f}%</td>
          <td style='padding:6px 10px'>{s['vol_ratio']:.1f}x</td>
          <td style='padding:6px 10px'>
            <div style='background:#eee;border-radius:4px;height:8px;width:80px'>
              <div style='background:{s_color};border-radius:4px;height:8px;width:{bar_w}px'></div>
            </div>
          </td>
          <td style='padding:6px 10px'>
            <span style='background:{s_color};color:white;padding:2px 8px;
                         border-radius:4px;font-size:11px'>{s_tag}</span>
          </td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'/></head>
<body style='font-family:Segoe UI,Arial,sans-serif;max-width:680px;
             margin:auto;color:#333;padding:20px;background:#f5f6fa'>

  <!-- Header -->
  <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
              color:white;padding:22px;border-radius:12px;margin-bottom:20px'>
    <div style='font-size:13px;opacity:0.7;margin-bottom:4px'>
      Morning Signal | ซื้อตอนตลาดเปิด 9:30 AM ET (20:30 น. ไทย)
    </div>
    <h1 style='margin:0;font-size:22px'>3-Asset Signal Brief</h1>
    <p style='margin:6px 0 0;opacity:0.7;font-size:13px'>{date_str}</p>
  </div>

  <!-- Macro Regime Banner -->
  <div style='background:#{reg_color};color:white;padding:14px 20px;
              border-radius:10px;margin-bottom:20px;
              display:flex;justify-content:space-between;align-items:center'>
    <div>
      <div style='font-size:12px;opacity:0.85'>MACRO REGIME</div>
      <div style='font-size:24px;font-weight:bold'>{reg_en}</div>
      <div style='font-size:14px;opacity:0.9'>{reg_th}</div>
    </div>
    <div style='text-align:right'>
      <div style='font-size:40px;font-weight:bold'>{mac_score:+d}</div>
      <div style='font-size:11px;opacity:0.8'>Macro Score</div>
    </div>
  </div>

  <!-- Action Plan -->
  <h2 style='border-left:5px solid #{reg_color};padding-left:10px;
             color:#1a1a2e;margin-top:24px'>
    สิ่งที่ต้องทำตอนตลาดเปิด
  </h2>
  <p style='color:#666;font-size:13px;margin-top:-8px'>
    Portfolio ${PORTFOLIO_USD:,.0f} | ซื้อที่ราคา Open 9:30 AM ET
  </p>
  {action_cards}

  <!-- Macro Table -->
  <h2 style='border-left:5px solid #3498db;padding-left:10px;color:#1a1a2e;margin-top:28px'>
    Macro Indicators
  </h2>
  <table style='width:100%;border-collapse:collapse;font-size:13px;
                background:white;border-radius:8px;overflow:hidden'>
    {macro_rows}
  </table>

  <!-- Stock Watchlist -->
  <h2 style='border-left:5px solid #9b59b6;padding-left:10px;color:#1a1a2e;margin-top:28px'>
    Growth Stock Watchlist (Top 8)
  </h2>
  <table style='width:100%;border-collapse:collapse;font-size:13px;
                background:white;border-radius:8px;overflow:hidden'>
    <tr style='background:#f8f9fa'>
      <th style='padding:8px 10px;text-align:left'>Ticker</th>
      <th style='padding:8px 10px;text-align:left'>Price</th>
      <th style='padding:8px 10px;text-align:left'>1D</th>
      <th style='padding:8px 10px;text-align:left'>5D</th>
      <th style='padding:8px 10px;text-align:left'>Volume</th>
      <th style='padding:8px 10px;text-align:left'>Score</th>
      <th style='padding:8px 10px;text-align:left'>Signal</th>
    </tr>
    {watch_rows}
  </table>

  <!-- Disclaimer -->
  <div style='background:#f8f9fa;border-radius:8px;padding:12px;
              margin-top:24px;font-size:11px;color:#888'>
    <b>หมายเหตุ:</b> Signal นี้วิเคราะห์จากข้อมูลวันก่อน ไม่ใช่คำแนะนำการลงทุน
    ราคา Open จริงอาจต่างจากราคาที่แสดง ควรตั้ง Stop Loss ทุกครั้ง
    ข้อมูลจาก Yahoo Finance (yfinance)
  </div>

</body></html>"""

    return html


def build_plain(regime, actions, macro_info):
    lines = [
        f"=== Morning Signal | {datetime.utcnow().strftime('%Y-%m-%d')} ===",
        f"Regime: {regime} | Macro Score: {macro_info.get('score','?')}",
        f"VIX: {macro_info.get('vix','N/A')} | QQQ 5d: {macro_info.get('qqq_5d','N/A')}%",
        "",
        "ACTION PLAN (execute at market open 9:30 AM ET):",
    ]
    for a in actions:
        if a["action"] == "HOLD_CASH":
            lines.append(f"  CASH  {a['alloc_pct']:.0f}% = ${a['alloc_usd']:,.0f}")
        else:
            lines.append(
                f"  {a['action']} {a['asset']:<6} {a['alloc_pct']:.0f}% = ${a['alloc_usd']:,.0f} "
                f"@ ~${a['price']:,.2f} | SL=${a['sl_price']} TP=${a['tp_price']}"
            )
    return "\n".join(lines)


# ── Email Send ────────────────────────────────────────────

def send_email(subject, html_body, text_body):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("  [!] Email not configured. Set EMAIL_SENDER and EMAIL_PASSWORD in .env")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_RECIPIENT
        msg.attach(MIMEText(text_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html",  "utf-8"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as srv:
            srv.ehlo(); srv.starttls()
            srv.login(EMAIL_SENDER, EMAIL_PASSWORD)
            srv.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
        print(f"  Email sent to {EMAIL_RECIPIENT}")
        return True
    except Exception as e:
        print(f"  Email error: {e}")
        return False


# ── Main ──────────────────────────────────────────────────

def run():
    print("\n" + "="*60)
    print("  Morning Signal -- 3-Asset (Stocks + Gold + Cash)")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("="*60)

    print("\n[1/3] Loading market data...")
    all_tickers = GROWTH_STOCKS + MACRO_TICKERS
    all_data    = fetch(all_tickers, period="2mo")

    stock_data = {tk: all_data[tk] for tk in GROWTH_STOCKS if tk in all_data}
    macro_data = {tk: all_data[tk] for tk in MACRO_TICKERS if tk in all_data}

    print(f"  Stocks loaded: {len(stock_data)}/{len(GROWTH_STOCKS)}")
    print(f"  Macro loaded:  {len(macro_data)}/{len(MACRO_TICKERS)}")

    print("\n[2/3] Analyzing signals...")
    regime, mac_score, reg_color, macro_info = get_macro(macro_data)

    stock_signals = []
    for tk in GROWTH_STOCKS:
        df = stock_data.get(tk)
        if df is not None:
            sig = score_stock(tk, df)
            if sig:
                stock_signals.append(sig)

    gld_df = macro_data.get("GLD")
    gold_score, gold_info = score_gold(gld_df) if gld_df is not None else (0, {})

    print(f"  Macro Regime : {regime} (score={mac_score:+d})")
    print(f"  VIX          : {macro_info.get('vix','N/A')}")
    print(f"  Gold Score   : {gold_score}")
    print(f"  Stocks with signal >= {MIN_STOCK_SCORE}: "
          f"{sum(1 for s in stock_signals if s['score'] >= MIN_STOCK_SCORE)}")

    print("\n[3/3] Building action plan...")
    actions = build_action_plan(regime, mac_score, stock_signals, gold_score, gold_info, macro_info)

    print("\n" + "-"*60)
    print("  ACTION PLAN (execute at 9:30 AM ET / 20:30 น. ไทย)")
    print("-"*60)
    for a in actions:
        if a["action"] == "HOLD_CASH":
            print(f"  [CASH ] {a['alloc_pct']:.0f}% = ${a['alloc_usd']:,.0f}  (do nothing)")
        elif a["type"] == "GOLD":
            print(f"  [GOLD ] BUY GLD  {a['alloc_pct']:.0f}% = ${a['alloc_usd']:,.0f}"
                  f"  ~{a['shares_approx']} shares @ ${a['price']:,.2f}")
            print(f"          SL=${a['sl_price']}  TP=${a['tp_price']}  Trail={a['trail_pct']}%")
        else:
            print(f"  [STOCK] BUY {a['asset']:<5} {a['alloc_pct']:.0f}% = ${a['alloc_usd']:,.0f}"
                  f"  ~{a['shares_approx']} shares @ ${a['price']:,.2f}")
            print(f"          SL=${a['sl_price']}  TP=${a['tp_price']}  Trail={a['trail_pct']}%")
    print("-"*60)

    # Save snapshot
    snapshot = {
        "datetime_utc": datetime.utcnow().isoformat(),
        "regime": regime, "macro_score": mac_score,
        "macro_info": macro_info,
        "actions": actions,
        "gold_score": gold_score, "gold_info": gold_info,
        "stock_signals": sorted(stock_signals, key=lambda x: x["score"], reverse=True),
    }
    with open("morning_signal_latest.json","w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print("\n  Snapshot saved -> morning_signal_latest.json")

    # Send email
    buy_items = [a for a in actions if a["action"] == "BUY"]
    if buy_items:
        names = " + ".join(a["asset"] for a in buy_items)
        subject = f"Morning Signal {datetime.utcnow().strftime('%d %b %Y')} | {regime} | BUY: {names}"
    else:
        subject = f"Morning Signal {datetime.utcnow().strftime('%d %b %Y')} | {regime} | HOLD CASH"

    html  = build_email(regime, mac_score, macro_info, actions, stock_signals)
    plain = build_plain(regime, actions, macro_info)

    print("\n  Sending email...")
    send_email(subject, html, plain)
    print("\n  Done.\n")


if __name__ == "__main__":
    run()
