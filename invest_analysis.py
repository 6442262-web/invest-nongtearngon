"""
Global Money Flow & Growth Stock Daily Analysis
Runs every morning, sends summary email to configured recipient.

Requirements: pip install -r requirements.txt
Setup: copy .env.example to .env and fill in credentials
"""

import os
import json
import smtplib
import traceback
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
EMAIL_SENDER    = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD  = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "6442262@schoolptk.ac.th")
SMTP_HOST       = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT       = int(os.getenv("SMTP_PORT", "587"))

# Sector ETFs for rotation analysis
SECTOR_ETFS = {
    "Technology":       "XLK",
    "Financials":       "XLF",
    "Energy":           "XLE",
    "Healthcare":       "XLV",
    "Industrials":      "XLI",
    "ConsumerDisc":     "XLY",
    "ConsumerStap":     "XLP",
    "Utilities":        "XLU",
    "RealEstate":       "XLRE",
    "Materials":        "XLB",
    "Communication":    "XLC",
}

# Growth ETF proxies and key growth baskets
GROWTH_ETFS = {
    "Russell1000Growth": "IWF",
    "NasdaqGrowth":      "QQQ",
    "SmallCapGrowth":    "IWO",
    "MegaCapGrowth":     "MGK",
}

# Individual high-quality growth stocks to track
GROWTH_STOCKS = [
    "NVDA", "MSFT", "AAPL", "META", "GOOGL",
    "AMZN", "TSLA", "AVGO", "ASML", "NOW",
    "ADBE", "CRM", "PANW", "MRVL", "TTD",
    "AXON", "DDOG", "NET", "ZS", "SNOW",
]

# Global macro tickers
MACRO_TICKERS = {
    "DXY":      "DX-Y.NYB",
    "Gold":     "GC=F",
    "Oil":      "CL=F",
    "SP500":    "^GSPC",
    "VIX":      "^VIX",
    "10Y_Yield":"^TNX",
    "2Y_Yield": "^IRX",   # closest available proxy (13-wk)
    "5Y_Yield": "^FVX",
}

# Bond/liquidity ETFs for money flow
FLOW_ETFS = {
    "TLT":  "Long Bonds (20Y+)",
    "HYG":  "High Yield (Risk-On)",
    "LQD":  "Investment Grade",
    "BIL":  "T-Bills (Cash-like)",
    "GLD":  "Gold ETF",
    "USO":  "Oil ETF",
    "UUP":  "USD Bull ETF",
    "EEM":  "Emerging Markets",
    "VT":   "World Equities",
}


# ─────────────────────────────────────────────
# Data Fetching
# ─────────────────────────────────────────────

def fetch(tickers: list[str], period: str = "1mo", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Fetch price data for a list of tickers."""
    result = {}
    for tk in tickers:
        try:
            df = yf.download(tk, period=period, interval=interval, progress=False, auto_adjust=True)
            if not df.empty:
                result[tk] = df
        except Exception:
            pass
    return result


def pct_change(df: pd.DataFrame, days: int = 1) -> float:
    """Return % change over last `days` trading days."""
    try:
        closes = df["Close"].dropna()
        if len(closes) < days + 1:
            return float("nan")
        return float((closes.iloc[-1] - closes.iloc[-(days + 1)]) / closes.iloc[-(days + 1)] * 100)
    except Exception:
        return float("nan")


def avg_volume_ratio(df: pd.DataFrame, short: int = 5, long: int = 20) -> float:
    """Volume surge ratio: recent avg vs longer avg (flow proxy)."""
    try:
        vols = df["Volume"].dropna()
        if len(vols) < long:
            return float("nan")
        return float(vols.iloc[-short:].mean() / vols.iloc[-long:].mean())
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────
# Macro Analysis
# ─────────────────────────────────────────────

def analyze_macro() -> dict:
    tickers = list(MACRO_TICKERS.values())
    data = fetch(tickers, period="3mo")

    macro = {}
    for name, tk in MACRO_TICKERS.items():
        df = data.get(tk)
        if df is None or df.empty:
            macro[name] = {"1d": None, "1w": None, "1m": None, "current": None}
            continue
        macro[name] = {
            "current": round(float(df["Close"].dropna().iloc[-1]), 4),
            "1d": round(pct_change(df, 1), 2),
            "1w": round(pct_change(df, 5), 2),
            "1m": round(pct_change(df, 20), 2),
        }

    # Yield curve spread: 10Y - 2Y (inversion warning)
    y10 = macro.get("10Y_Yield", {}).get("current")
    y2  = macro.get("2Y_Yield", {}).get("current")
    spread = round(y10 - y2, 3) if (y10 and y2) else None
    macro["YieldCurve_10Y_minus_2Y"] = spread

    return macro


def interpret_macro(macro: dict) -> list[str]:
    signals = []
    dxy_1d = macro.get("DXY", {}).get("1d")
    vix     = macro.get("VIX", {}).get("current")
    spread  = macro.get("YieldCurve_10Y_minus_2Y")
    y10_1m  = macro.get("10Y_Yield", {}).get("1m")

    if dxy_1d is not None:
        if dxy_1d > 0.5:
            signals.append("USD strengthening — headwind for EM & commodities, watch dollar-sensitive growth names")
        elif dxy_1d < -0.5:
            signals.append("USD weakening — tailwind for international revenues, EM, commodities")

    if vix is not None:
        if vix > 30:
            signals.append(f"VIX={vix} (high fear) — risk-off; prefer quality/cash-rich growth over speculative")
        elif vix < 15:
            signals.append(f"VIX={vix} (low fear) — risk-on environment; growth / momentum favored")
        else:
            signals.append(f"VIX={vix} — neutral volatility regime")

    if spread is not None:
        if spread < 0:
            signals.append(f"Yield curve INVERTED ({spread}%) — recession watch; favor defensive growth (MSFT, GOOGL)")
        elif spread < 0.3:
            signals.append(f"Yield curve flat ({spread}%) — late cycle; quality premium rises")
        else:
            signals.append(f"Yield curve positive ({spread}%) — growth-friendly rate environment")

    if y10_1m is not None:
        if y10_1m > 5:
            signals.append("10Y yields rising fast — valuation compression risk for long-duration growth")
        elif y10_1m < -5:
            signals.append("10Y yields falling — supports growth stock multiples (DCF tailwind)")

    return signals


# ─────────────────────────────────────────────
# ETF Flow Analysis
# ─────────────────────────────────────────────

def analyze_etf_flows() -> dict:
    tickers = list(FLOW_ETFS.keys())
    data = fetch(tickers, period="1mo")

    flows = {}
    for tk, label in FLOW_ETFS.items():
        df = data.get(tk)
        if df is None or df.empty:
            flows[tk] = None
            continue
        flows[tk] = {
            "label":        label,
            "1d_pct":       round(pct_change(df, 1), 2),
            "1w_pct":       round(pct_change(df, 5), 2),
            "vol_ratio":    round(avg_volume_ratio(df), 2),  # 5d vs 20d volume
        }
    return flows


def interpret_flows(flows: dict) -> list[str]:
    signals = []

    def safe_get(tk, field):
        item = flows.get(tk)
        return item.get(field) if item else None

    hyd_1w = safe_get("HYG", "1w_pct")
    tlt_1w = safe_get("TLT", "1w_pct")
    bil_1w = safe_get("BIL", "1w_pct")
    eem_1w = safe_get("EEM", "1w_pct")

    # Risk-on / risk-off
    if hyd_1w is not None and tlt_1w is not None:
        if hyd_1w > 0.5 and tlt_1w < -0.5:
            signals.append("Flow: HYG rising, TLT falling → money rotating into risk assets (RISK-ON)")
        elif hyd_1w < -0.5 and tlt_1w > 0.5:
            signals.append("Flow: HYG falling, TLT rising → flight to safety (RISK-OFF)")

    if bil_1w is not None and bil_1w > 0.2:
        signals.append("BIL inflows elevated → cash hoarding, investors waiting for entry")

    if eem_1w is not None:
        if eem_1w > 1.5:
            signals.append("EEM surging → global risk appetite expanding, USD weakness likely")
        elif eem_1w < -1.5:
            signals.append("EEM declining → global risk-off, favor US domestic growers")

    # Volume surges as flow proxy
    for tk, info in flows.items():
        if info and info.get("vol_ratio", 0) > 1.8:
            signals.append(f"{tk} ({info['label']}): volume surge {info['vol_ratio']}x — unusual flow detected")

    return signals


# ─────────────────────────────────────────────
# Sector Rotation Analysis
# ─────────────────────────────────────────────

def analyze_sector_rotation() -> dict:
    tickers = list(SECTOR_ETFS.values())
    data = fetch(tickers, period="1mo")

    sectors = {}
    for name, tk in SECTOR_ETFS.items():
        df = data.get(tk)
        if df is None or df.empty:
            continue
        sectors[name] = {
            "ticker":    tk,
            "1d":        round(pct_change(df, 1), 2),
            "1w":        round(pct_change(df, 5), 2),
            "1m":        round(pct_change(df, 20), 2),
            "vol_ratio": round(avg_volume_ratio(df), 2),
        }

    # Rank by 1-week performance
    ranked = sorted(sectors.items(), key=lambda x: x[1].get("1w", -999), reverse=True)
    return {"sectors": sectors, "ranked_1w": [r[0] for r in ranked]}


def interpret_rotation(rotation: dict) -> list[str]:
    signals = []
    ranked = rotation.get("ranked_1w", [])
    if not ranked:
        return signals

    top3    = ranked[:3]
    bottom3 = ranked[-3:]

    signals.append(f"Top sectors (1W): {', '.join(top3)} — momentum here")
    signals.append(f"Lagging sectors (1W): {', '.join(bottom3)} — potential value traps or contrarian setup")

    # Growth-friendly rotation check
    growth_sectors = {"Technology", "Communication", "ConsumerDisc"}
    defensive      = {"Utilities", "ConsumerStap", "RealEstate"}

    top_set = set(top3)
    if growth_sectors & top_set:
        signals.append("Growth sectors leading — favorable for growth stock positioning")
    if defensive & top_set:
        signals.append("Defensive sectors leading — market rotating to safety, be selective in growth")

    return signals


# ─────────────────────────────────────────────
# Microeconomic Framework: Growth Stock Analysis
# ─────────────────────────────────────────────

def fetch_fundamentals(ticker: str) -> dict:
    """Pull key fundamental metrics for microeconomic scoring."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        rev_growth     = info.get("revenueGrowth")          # YoY revenue growth
        earnings_growth = info.get("earningsGrowth")         # YoY EPS growth
        gross_margin   = info.get("grossMargins")            # pricing power proxy
        op_margin      = info.get("operatingMargins")        # operational efficiency
        ps_ratio       = info.get("priceToSalesTrailing12Months")
        pe_ratio       = info.get("trailingPE")
        peg_ratio      = info.get("pegRatio")               # growth-adjusted PE
        roe            = info.get("returnOnEquity")          # barriers / moat proxy
        debt_equity    = info.get("debtToEquity")
        fcf_yield_raw  = info.get("freeCashflow")
        market_cap     = info.get("marketCap")
        beta           = info.get("beta")

        fcf_yield = None
        if fcf_yield_raw and market_cap and market_cap > 0:
            fcf_yield = round(fcf_yield_raw / market_cap * 100, 2)

        return {
            "rev_growth":      round(rev_growth * 100, 1) if rev_growth else None,
            "earnings_growth": round(earnings_growth * 100, 1) if earnings_growth else None,
            "gross_margin":    round(gross_margin * 100, 1) if gross_margin else None,
            "op_margin":       round(op_margin * 100, 1) if op_margin else None,
            "ps_ratio":        round(ps_ratio, 2) if ps_ratio else None,
            "pe_ratio":        round(pe_ratio, 2) if pe_ratio else None,
            "peg_ratio":       round(peg_ratio, 2) if peg_ratio else None,
            "roe":             round(roe * 100, 1) if roe else None,
            "debt_equity":     round(debt_equity, 2) if debt_equity else None,
            "fcf_yield_pct":   fcf_yield,
            "beta":            round(beta, 2) if beta else None,
            "market_cap_B":    round(market_cap / 1e9, 1) if market_cap else None,
        }
    except Exception:
        return {}


def score_growth_stock(ticker: str, fundamentals: dict) -> dict:
    """
    Microeconomic scoring:
      - Pricing Power    : gross margin + margin trend
      - Barriers to Entry: ROE + market cap (scale moat)
      - Demand Elasticity: revenue growth vs price (PS ratio)
      - Growth Quality   : FCF yield + PEG ratio
    """
    score = 0
    flags = []

    gm  = fundamentals.get("gross_margin")
    rev = fundamentals.get("rev_growth")
    roe = fundamentals.get("roe")
    fcf = fundamentals.get("fcf_yield_pct")
    peg = fundamentals.get("peg_ratio")
    ps  = fundamentals.get("ps_ratio")
    op  = fundamentals.get("op_margin")
    cap = fundamentals.get("market_cap_B")

    # Pricing Power (gross margin = ability to raise prices without losing customers)
    if gm is not None:
        if gm >= 70:
            score += 3; flags.append("Strong pricing power (GM≥70%)")
        elif gm >= 50:
            score += 2; flags.append("Good pricing power (GM≥50%)")
        elif gm >= 35:
            score += 1; flags.append("Moderate pricing power")
        else:
            flags.append("Weak pricing power (commodity-like)")

    # Revenue Growth (demand elasticity — growing despite potential price increases)
    if rev is not None:
        if rev >= 25:
            score += 3; flags.append(f"High revenue growth ({rev}%)")
        elif rev >= 15:
            score += 2; flags.append(f"Solid revenue growth ({rev}%)")
        elif rev >= 8:
            score += 1; flags.append(f"Moderate growth ({rev}%)")
        else:
            flags.append(f"Slow revenue growth ({rev}%)")

    # Barriers to Entry (ROE = return on capital protected by moat)
    if roe is not None:
        if roe >= 30:
            score += 3; flags.append(f"High moat indicator (ROE={roe}%)")
        elif roe >= 15:
            score += 2; flags.append(f"Decent moat (ROE={roe}%)")
        elif roe >= 5:
            score += 1
        else:
            flags.append(f"Low ROE ({roe}%) — moat concern")

    # Scale advantage
    if cap and cap >= 50:
        score += 1; flags.append(f"Scale moat (Mkt Cap ${cap}B)")

    # Growth Quality (FCF + PEG)
    if fcf is not None:
        if fcf >= 3:
            score += 2; flags.append(f"FCF generative ({fcf}% yield)")
        elif fcf >= 1:
            score += 1; flags.append(f"Positive FCF ({fcf}% yield)")
        else:
            flags.append("FCF negative or minimal")

    if peg is not None and 0 < peg < 2:
        score += 2; flags.append(f"Attractive PEG ({peg}) — growth at reasonable price")
    elif peg and peg >= 4:
        flags.append(f"Stretched PEG ({peg}) — priced for perfection")

    # Operating leverage (elasticity of profits to revenue)
    if op is not None and op >= 20:
        score += 1; flags.append(f"High operating leverage (OP={op}%)")

    # Valuation sanity check
    if ps and ps > 30:
        score -= 1; flags.append(f"Very high P/S ({ps}) — valuation risk")

    return {"score": score, "flags": flags, "max": 15}


def analyze_growth_stocks() -> list[dict]:
    results = []
    for ticker in GROWTH_STOCKS:
        fund = fetch_fundamentals(ticker)
        scored = score_growth_stock(ticker, fund)

        # Also get 1-week price momentum
        price_data = fetch([ticker], period="1mo")
        df = price_data.get(ticker)
        momentum_1w = round(pct_change(df, 5), 2) if df is not None else None
        momentum_1m = round(pct_change(df, 20), 2) if df is not None else None

        results.append({
            "ticker":      ticker,
            "score":       scored["score"],
            "max_score":   scored["max"],
            "flags":       scored["flags"],
            "fundamentals": fund,
            "momentum_1w": momentum_1w,
            "momentum_1m": momentum_1m,
        })

    # Sort by composite: fundamental score + momentum
    def composite(r):
        m = r["momentum_1w"] or 0
        return r["score"] + (m / 5)  # normalize momentum contribution

    return sorted(results, key=composite, reverse=True)


# ─────────────────────────────────────────────
# Report Formatting
# ─────────────────────────────────────────────

def star(score: int, max_score: int) -> str:
    pct = score / max_score if max_score else 0
    filled = round(pct * 5)
    return "★" * filled + "☆" * (5 - filled)


def fmt_num(val, suffix="", prefix="") -> str:
    if val is None:
        return "N/A"
    return f"{prefix}{val}{suffix}"


def build_html_report(
    macro: dict,
    macro_signals: list[str],
    flows: dict,
    flow_signals: list[str],
    rotation: dict,
    rotation_signals: list[str],
    growth_stocks: list[dict],
) -> str:
    date_str = datetime.now().strftime("%A, %d %B %Y")

    def section(title: str, color: str) -> str:
        return f'<h2 style="border-left:5px solid {color};padding-left:10px;color:#1a1a2e">{title}</h2>'

    def signal_list(signals: list[str]) -> str:
        if not signals:
            return "<p>No significant signals today.</p>"
        items = "".join(f"<li style='margin:6px 0'>{s}</li>" for s in signals)
        return f"<ul style='line-height:1.7'>{items}</ul>"

    # ── Macro table
    macro_rows = ""
    for name, vals in macro.items():
        if name == "YieldCurve_10Y_minus_2Y":
            continue
        if isinstance(vals, dict):
            c = vals.get("current", "N/A")
            d = vals.get("1d", "N/A")
            w = vals.get("1w", "N/A")
            m = vals.get("1m", "N/A")
            def col(v):
                if v in (None, "N/A"): return f"<td style='padding:4px 8px'>N/A</td>"
                color = "#27ae60" if float(v) > 0 else "#e74c3c"
                return f"<td style='padding:4px 8px;color:{color}'>{v:+.2f}%</td>"
            macro_rows += f"<tr><td style='padding:4px 8px;font-weight:bold'>{name}</td><td style='padding:4px 8px'>{c}</td>{col(d)}{col(w)}{col(m)}</tr>"

    spread = macro.get("YieldCurve_10Y_minus_2Y")
    spread_color = "#e74c3c" if spread and spread < 0 else "#27ae60"
    macro_rows += f"<tr><td style='padding:4px 8px;font-weight:bold'>Yield Curve (10Y-2Y)</td><td style='padding:4px 8px;color:{spread_color}'>{fmt_num(spread,'%')}</td><td colspan='3'></td></tr>"

    # ── Sector rotation table
    sectors = rotation.get("sectors", {})
    ranked  = rotation.get("ranked_1w", [])
    sector_rows = ""
    for name in ranked:
        info = sectors[name]
        def scol(v):
            if v is None: return "<td style='padding:4px 8px'>N/A</td>"
            color = "#27ae60" if v > 0 else "#e74c3c"
            return f"<td style='padding:4px 8px;color:{color}'>{v:+.2f}%</td>"
        vr = info.get("vol_ratio", "N/A")
        vr_str = f"{vr}x" if isinstance(vr, float) else "N/A"
        sector_rows += f"<tr><td style='padding:4px 8px;font-weight:bold'>{name}</td><td style='padding:4px 8px'>{info['ticker']}</td>{scol(info.get('1d'))}{scol(info.get('1w'))}{scol(info.get('1m'))}<td style='padding:4px 8px'>{vr_str}</td></tr>"

    # ── Top growth stocks cards
    stock_cards = ""
    for rank, st in enumerate(growth_stocks[:10], 1):
        fund  = st["fundamentals"]
        score = st["score"]
        maxsc = st["max_score"]
        stars = star(score, maxsc)
        m1w   = st["momentum_1w"]
        m1m   = st["momentum_1m"]
        mcolor = "#27ae60" if (m1w or 0) > 0 else "#e74c3c"

        flags_html = "".join(f"<span style='display:inline-block;background:#eaf4fb;border-radius:3px;padding:2px 6px;margin:2px;font-size:12px'>{f}</span>" for f in st["flags"])

        fund_table = f"""
        <table style='font-size:12px;border-collapse:collapse;width:100%'>
          <tr><td style='padding:2px 6px'>Rev Growth</td><td style='padding:2px 6px'><b>{fmt_num(fund.get('rev_growth'),'%')}</b></td>
              <td style='padding:2px 6px'>Gross Margin</td><td style='padding:2px 6px'><b>{fmt_num(fund.get('gross_margin'),'%')}</b></td></tr>
          <tr><td style='padding:2px 6px'>ROE</td><td style='padding:2px 6px'><b>{fmt_num(fund.get('roe'),'%')}</b></td>
              <td style='padding:2px 6px'>PEG</td><td style='padding:2px 6px'><b>{fmt_num(fund.get('peg_ratio'))}</b></td></tr>
          <tr><td style='padding:2px 6px'>FCF Yield</td><td style='padding:2px 6px'><b>{fmt_num(fund.get('fcf_yield_pct'),'%')}</b></td>
              <td style='padding:2px 6px'>P/S</td><td style='padding:2px 6px'><b>{fmt_num(fund.get('ps_ratio'),'x')}</b></td></tr>
          <tr><td style='padding:2px 6px'>Mkt Cap</td><td style='padding:2px 6px'><b>${fmt_num(fund.get('market_cap_B'),'B')}</b></td>
              <td style='padding:2px 6px'>Beta</td><td style='padding:2px 6px'><b>{fmt_num(fund.get('beta'))}</b></td></tr>
        </table>
        """

        stock_cards += f"""
        <div style='border:1px solid #dce3ef;border-radius:8px;padding:14px;margin:10px 0;background:#fafcff'>
          <div style='display:flex;justify-content:space-between;align-items:center'>
            <div>
              <span style='font-size:20px;font-weight:bold;color:#1a1a2e'>#{rank} {st['ticker']}</span>
              &nbsp;<span style='font-size:16px;color:#f39c12'>{stars}</span>
              &nbsp;<span style='font-size:13px;background:#1a1a2e;color:white;border-radius:4px;padding:2px 8px'>{score}/{maxsc}</span>
            </div>
            <div style='text-align:right;font-size:14px'>
              1W: <span style='color:{mcolor};font-weight:bold'>{fmt_num(m1w,'%','+' if (m1w or 0)>0 else '')}</span>
              &nbsp; 1M: {fmt_num(m1m,'%','+' if (m1m or 0)>0 else '')}
            </div>
          </div>
          <div style='margin:8px 0'>{flags_html}</div>
          {fund_table}
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"/></head>
    <body style='font-family:Segoe UI,Arial,sans-serif;max-width:750px;margin:auto;color:#333;padding:20px'>

      <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;padding:20px;border-radius:10px;margin-bottom:24px'>
        <h1 style='margin:0;font-size:22px'>Global Money Flow & Growth Stock Daily Brief</h1>
        <p style='margin:6px 0 0;opacity:0.8'>{date_str} | Powered by yfinance + Microeconomic Framework</p>
      </div>

      {section("1. Global Macro Environment","#3498db")}
      <table style='border-collapse:collapse;width:100%;font-size:13px'>
        <tr style='background:#f0f4f8'><th style='padding:6px 8px;text-align:left'>Indicator</th><th>Current</th><th>1D</th><th>1W</th><th>1M</th></tr>
        {macro_rows}
      </table>
      <br>
      {signal_list(macro_signals)}

      {section("2. ETF Money Flow (Risk-On/Off)","#e67e22")}
      {signal_list(flow_signals)}
      <table style='border-collapse:collapse;width:100%;font-size:13px'>
        <tr style='background:#f0f4f8'><th style='padding:6px 8px;text-align:left'>ETF</th><th>Label</th><th>1D</th><th>1W</th><th>Vol Ratio</th></tr>
        {"".join(
            f"<tr><td style='padding:4px 8px;font-weight:bold'>{tk}</td><td style='padding:4px 8px'>{info['label'] if info else 'N/A'}</td>"
            + (f"<td style='padding:4px 8px;color:{'#27ae60' if info['1d_pct']>0 else '#e74c3c'}'>{info['1d_pct']:+.2f}%</td>"
               f"<td style='padding:4px 8px;color:{'#27ae60' if info['1w_pct']>0 else '#e74c3c'}'>{info['1w_pct']:+.2f}%</td>"
               f"<td style='padding:4px 8px'>{info['vol_ratio']}x</td>" if info else "<td colspan='3'>N/A</td>")
            + "</tr>"
            for tk, info in flows.items()
        )}
      </table>

      {section("3. Sector Rotation (1W Ranked)","#9b59b6")}
      <table style='border-collapse:collapse;width:100%;font-size:13px'>
        <tr style='background:#f0f4f8'><th style='padding:6px 8px;text-align:left'>Sector</th><th>ETF</th><th>1D</th><th>1W</th><th>1M</th><th>Vol Ratio</th></tr>
        {sector_rows}
      </table>
      <br>
      {signal_list(rotation_signals)}

      {section("4. Growth Stock Rankings (Microeconomic Score)","#27ae60")}
      <p style='font-size:13px;color:#666'>
        Scoring: Pricing Power (Gross Margin) + Demand Elasticity (Rev Growth) + Barriers to Entry (ROE/Scale) + Growth Quality (FCF/PEG)
      </p>
      {stock_cards}

      <div style='background:#f8f9fa;border-radius:8px;padding:14px;margin-top:24px;font-size:12px;color:#666'>
        <b>Disclaimer:</b> This report is generated automatically for informational purposes only.
        Not financial advice. Always do your own research before making investment decisions.
        Data sourced from Yahoo Finance via yfinance.
      </div>
    </body>
    </html>
    """
    return html


def build_text_report(growth_stocks: list[dict], macro_signals: list, flow_signals: list, rotation_signals: list) -> str:
    lines = [f"=== Daily Investment Brief — {datetime.now().strftime('%Y-%m-%d')} ===\n"]
    lines.append("MACRO SIGNALS:")
    lines.extend(f"  • {s}" for s in macro_signals)
    lines.append("\nFLOW SIGNALS:")
    lines.extend(f"  • {s}" for s in flow_signals)
    lines.append("\nROTATION SIGNALS:")
    lines.extend(f"  • {s}" for s in rotation_signals)
    lines.append("\nTOP 10 GROWTH STOCKS:")
    for i, st in enumerate(growth_stocks[:10], 1):
        m = st['momentum_1w']
        lines.append(f"  {i}. {st['ticker']} (Score {st['score']}/{st['max_score']}, 1W: {m:+.2f}%)" if m else f"  {i}. {st['ticker']} (Score {st['score']}/{st['max_score']})")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Email Sender
# ─────────────────────────────────────────────

def send_email(subject: str, html_body: str, text_body: str):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("Email credentials not set — skipping send. Check EMAIL_SENDER and EMAIL_PASSWORD in .env")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECIPIENT

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html",  "utf-8"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())

    print(f"Email sent to {EMAIL_RECIPIENT}")


# ─────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────

def run():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting daily analysis...")

    print("  → Fetching macro data...")
    macro = analyze_macro()
    macro_signals = interpret_macro(macro)

    print("  → Analyzing ETF flows...")
    flows = analyze_etf_flows()
    flow_signals = interpret_flows(flows)

    print("  → Analyzing sector rotation...")
    rotation = analyze_sector_rotation()
    rotation_signals = interpret_rotation(rotation)

    print("  → Scoring growth stocks (this may take ~30s)...")
    growth_stocks = analyze_growth_stocks()

    print("  → Building report...")
    html = build_html_report(macro, macro_signals, flows, flow_signals, rotation, rotation_signals, growth_stocks)
    text = build_text_report(growth_stocks, macro_signals, flow_signals, rotation_signals)

    subject = f"Investment Brief {datetime.now().strftime('%d %b %Y')} — Top: {', '.join(s['ticker'] for s in growth_stocks[:3])}"

    print(text)
    print("\n  → Sending email...")
    send_email(subject, html, text)

    # Save JSON snapshot for debugging / backtesting
    snapshot = {
        "date":     datetime.now().isoformat(),
        "macro":    macro,
        "flows":    flows,
        "rotation": rotation.get("ranked_1w"),
        "growth_top10": [
            {"ticker": s["ticker"], "score": s["score"], "momentum_1w": s["momentum_1w"]}
            for s in growth_stocks[:10]
        ],
    }
    with open("daily_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)
    print("  → Snapshot saved to daily_snapshot.json")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    run()
