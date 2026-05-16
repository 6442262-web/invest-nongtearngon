# -*- coding: utf-8 -*-
"""
3-Asset Swing Trade Backtest -- Stocks + Gold + Cash
Portfolio : $5,000
Logic     : Dynamic allocation across 3 buckets simultaneously
            STOCKS : strong momentum signals
            GOLD   : risk-off or gold uptrend
            CASH   : no clear signal / high VIX / wait
"""

import json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────
INITIAL_CAPITAL   = 5_000.0
START_DATE        = "2026-02-14"
END_DATE          = "2026-05-14"

# Allocation caps (can hold all 3 simultaneously)
MAX_STOCK_POS     = 2           # max 2 stock positions at once
STOCK_ALLOC_PCT   = 0.35        # 35% per stock slot
GOLD_ALLOC_PCT    = 0.25        # up to 25% in gold
# Remainder stays as cash automatically

# Exit rules (stock)
STOCK_SL_PCT      = 0.07        # -7% stop loss
STOCK_TP_PCT      = 0.15        # +15% take profit
STOCK_TRAIL_PCT   = 0.05        # 5% trailing from peak
MAX_HOLD_DAYS     = 25

# Exit rules (gold)
GOLD_SL_PCT       = 0.05        # -5% stop loss on gold
GOLD_TP_PCT       = 0.12        # +12% take profit on gold
GOLD_TRAIL_PCT    = 0.04        # 4% trailing

COMMISSION        = 0.001       # 0.1% per side

# Signal thresholds
MIN_STOCK_SCORE   = 4
MIN_GOLD_SCORE    = 2           # lower bar for gold (safe haven)

GROWTH_STOCKS = [
    "NVDA","MSFT","AAPL","META","GOOGL",
    "AMZN","TSLA","AVGO","NOW",
    "ADBE","CRM","PANW","DDOG","NET","ZS",
]

MACRO_TICKERS = ["DX-Y.NYB","^VIX","^TNX","HYG","QQQ","XLK","GLD"]


# ── Data Loading ──────────────────────────────────────────

def load_all(tickers, start, end):
    print(f"  Downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start, end=end,
                      interval="1d", progress=False, auto_adjust=True,
                      group_by="ticker")
    result = {}
    for tk in tickers:
        try:
            df = raw[tk].copy() if len(tickers) > 1 else raw.copy()
            df = df.dropna(subset=["Close","Open","High","Low"])
            if len(df) > 10:
                result[tk] = df
        except:
            pass
    print(f"  Loaded: {len(result)}/{len(tickers)} OK")
    return result


# ── Helpers ───────────────────────────────────────────────

def last_closes(df, signal_date, n):
    sub = df.loc[:signal_date, "Close"].dropna()
    return sub.values[-n:] if len(sub) >= n else None

def last_vols(df, signal_date, n):
    sub = df.loc[:signal_date, "Volume"].dropna()
    return sub.values[-n:] if len(sub) >= n else None

def mom(arr, back):
    if arr is None or len(arr) < back + 1: return 0
    b = float(arr[-(back+1)])
    return (float(arr[-1]) - b) / b * 100 if b > 0 else 0


# ── Macro Regime ──────────────────────────────────────────

def macro_regime(macro_data, signal_date):
    """Score overall market environment. Returns score + label."""
    score = 0
    info  = {}

    vix_c = last_closes(macro_data.get("^VIX"), signal_date, 3)
    if vix_c is not None:
        vix = float(vix_c[-1])
        info["vix"] = round(vix, 1)
        if   vix < 15: score += 3
        elif vix < 18: score += 2
        elif vix < 22: score += 1
        elif vix < 28: score -= 1
        elif vix < 35: score -= 2
        else:          score -= 4

    hyg_c = last_closes(macro_data.get("HYG"), signal_date, 6)
    if hyg_c is not None:
        h5 = mom(hyg_c, 5)
        info["hyg_5d"] = round(h5, 2)
        if   h5 >  0.5: score += 2
        elif h5 >  0:   score += 1
        elif h5 < -1:   score -= 2

    qqq_c = last_closes(macro_data.get("QQQ"), signal_date, 11)
    if qqq_c is not None:
        q5  = mom(qqq_c, 5)
        q10 = mom(qqq_c, 10)
        info["qqq_5d"] = round(q5, 2)
        if   q5 >  2: score += 2
        elif q5 >  0: score += 1
        elif q5 < -3: score -= 2
        if   q10 > 4: score += 1
        elif q10 < -5:score -= 1

    dxy_c = last_closes(macro_data.get("DX-Y.NYB"), signal_date, 6)
    if dxy_c is not None:
        d5 = mom(dxy_c, 5)
        info["dxy_5d"] = round(d5, 2)
        if d5 >  1: score -= 1
        if d5 < -1: score += 1

    gld_c = last_closes(macro_data.get("GLD"), signal_date, 6)
    if gld_c is not None:
        g5 = mom(gld_c, 5)
        info["gold_5d"] = round(g5, 2)
        # gold rallying = risk-off indicator
        if g5 >  3: score -= 1
        if g5 < -2: score += 1

    info["macro_score"] = score

    if   score >= 4:  label = "STRONG_ON"
    elif score >= 1:  label = "RISK_ON"
    elif score >= -1: label = "NEUTRAL"
    elif score >= -3: label = "RISK_OFF"
    else:             label = "CRISIS"

    return label, score, info


# ── Allocation Decision ───────────────────────────────────

def decide_allocation(macro_label, macro_score, stock_candidates, gold_score):
    """
    Returns target allocation buckets:
    { "stocks": pct, "gold": pct, "cash": pct }
    Based on macro regime + available signals.
    """
    has_stocks = len(stock_candidates) > 0
    has_gold   = gold_score >= MIN_GOLD_SCORE

    if macro_label == "STRONG_ON":
        # Full risk-on: max stocks, small gold, little cash
        stock_pct = STOCK_ALLOC_PCT * MAX_STOCK_POS if has_stocks else 0
        gold_pct  = GOLD_ALLOC_PCT * 0.3 if has_gold else 0
    elif macro_label == "RISK_ON":
        stock_pct = STOCK_ALLOC_PCT * MAX_STOCK_POS if has_stocks else 0
        gold_pct  = GOLD_ALLOC_PCT * 0.5 if has_gold else 0
    elif macro_label == "NEUTRAL":
        # Balanced: moderate stocks, moderate gold
        stock_pct = STOCK_ALLOC_PCT * 1 if has_stocks else 0   # max 1 position
        gold_pct  = GOLD_ALLOC_PCT * 0.7 if has_gold else 0
    elif macro_label == "RISK_OFF":
        # Defensive: no new stocks, lean into gold + cash
        stock_pct = 0
        gold_pct  = GOLD_ALLOC_PCT * 1.2 if has_gold else 0    # overweight gold
    else:  # CRISIS
        # Max cash, small gold hedge
        stock_pct = 0
        gold_pct  = GOLD_ALLOC_PCT * 0.5 if has_gold else 0

    cash_pct = max(0, 1.0 - stock_pct - gold_pct)
    return {"stocks": round(stock_pct,3), "gold": round(gold_pct,3), "cash": round(cash_pct,3)}


# ── Stock Signal ──────────────────────────────────────────

def stock_signal(tk, stock_data, macro_data, signal_date):
    df = stock_data.get(tk)
    if df is None: return None

    c2  = last_closes(df, signal_date, 2)
    c6  = last_closes(df, signal_date, 6)
    c11 = last_closes(df, signal_date, 11)
    c21 = last_closes(df, signal_date, 21)
    vol = last_vols(df,  signal_date, 25)

    if c2 is None: return None
    c_now = float(c2[-1]); c_prev = float(c2[-2])
    if c_now <= 0 or c_prev <= 0: return None

    m1  = mom(c2,  1)
    m5  = mom(c6,  5)  if c6  is not None else 0
    m10 = mom(c11, 10) if c11 is not None else 0
    m20 = mom(c21, 20) if c21 is not None else 0

    vol_ratio = 1.0
    if vol is not None and len(vol) >= 20:
        vol_ratio = float(vol[-5:].mean()) / float(vol[-20:].mean())

    above_10ma = (c_now > float(c11.mean())) if c11 is not None else False
    above_20ma = (c_now > float(c21.mean())) if c21 is not None else False

    score = 0
    if   m1  >  3:  score += 3
    elif m1  >  1.5:score += 2
    elif m1  >  0.5:score += 1
    elif m1  < -1.5:score -= 2

    if   m5  >  6:  score += 3
    elif m5  >  3:  score += 2
    elif m5  >  1:  score += 1
    elif m5  < -4:  score -= 2

    if   m10 > 10:  score += 2
    elif m10 >  5:  score += 1

    if   m20 > 15:  score += 2
    elif m20 >  8:  score += 1
    elif m20 < -5:  score -= 1

    if above_10ma:  score += 1
    if above_20ma:  score += 1

    if   vol_ratio > 2.0: score += 2
    elif vol_ratio > 1.3: score += 1

    return {
        "score": score, "mom_1d": round(m1,2), "mom_5d": round(m5,2),
        "mom_10d": round(m10,2), "vol_ratio": round(vol_ratio,2),
        "price": round(c_now,2), "above_20ma": above_20ma,
    }


def gold_signal(macro_data, signal_date):
    gld = macro_data.get("GLD")
    if gld is None: return 0, {}

    c6  = last_closes(gld, signal_date, 6)
    c11 = last_closes(gld, signal_date, 11)
    c21 = last_closes(gld, signal_date, 21)

    if c6 is None: return 0, {}
    c_now = float(c6[-1])

    m5  = mom(c6,  5)
    m10 = mom(c11, 10) if c11 is not None else 0
    m20 = mom(c21, 20) if c21 is not None else 0
    above_10ma = (c_now > float(c11.mean())) if c11 is not None else False
    above_20ma = (c_now > float(c21.mean())) if c21 is not None else False

    score = 0
    if   m5  >  3:  score += 3
    elif m5  >  1:  score += 2
    elif m5  >  0:  score += 1
    elif m5  < -2:  score -= 1

    if   m10 >  5:  score += 2
    elif m10 >  2:  score += 1

    if   m20 > 10:  score += 2
    elif m20 >  4:  score += 1

    if above_10ma:  score += 1
    if above_20ma:  score += 1

    return score, {"mom_5d": round(m5,2), "mom_10d": round(m10,2),
                   "mom_20d": round(m20,2), "price": round(c_now,2),
                   "above_20ma": above_20ma}


# ── Position Exit Logic ───────────────────────────────────

def check_exit_pos(pos, cur_close, cur_open, sl_pct, tp_pct, trail_pct,
                   max_hold, hold_days, sig_score=None):
    entry = pos["entry_price"]
    peak  = pos["peak_price"]

    if cur_close > peak:
        pos["peak_price"] = cur_close

    if cur_close <= entry * (1 - sl_pct):
        return True, "STOP_LOSS", min(cur_open, entry*(1-sl_pct))
    if cur_close >= entry * (1 + tp_pct):
        return True, "TAKE_PROFIT", max(cur_open, entry*(1+tp_pct))
    if cur_close <= peak * (1 - trail_pct):
        return True, "TRAILING_STOP", cur_open
    if hold_days >= max_hold:
        return True, "MAX_HOLD", cur_open
    if sig_score is not None and sig_score < -2:
        return True, "SIGNAL_REVERSAL", cur_open

    return False, None, None


# ── Backtest Engine ───────────────────────────────────────

def run_backtest(stock_data, macro_data):
    date_sets = [set(df.loc[START_DATE:END_DATE].index.normalize())
                 for df in stock_data.values()]
    common_dates = sorted(set.intersection(*date_sets))

    capital      = INITIAL_CAPITAL
    stock_pos    = {}   # {ticker: pos_dict}
    gold_pos     = None # single gold position
    # Cash = capital variable (always available)

    trades       = []
    equity_curve = []
    daily_log    = []
    max_equity   = capital
    max_drawdown = 0.0

    print(f"  Trading days: {len(common_dates)}")
    print(f"  3 Buckets: Stocks (35%x2) | Gold (25%) | Cash (rest)\n")
    print(f"  {'Date':<12} {'Regime':<12} {'Stocks':^20} {'Gold':^10} {'Cash':^9} {'Total':>8}")
    print(f"  {'-'*72}")

    gld_df = macro_data.get("GLD")

    for i, trade_ts in enumerate(common_dates):
        trade_date  = trade_ts
        signal_date = common_dates[i-1] if i > 0 else trade_ts

        # ── Mark-to-market all open positions
        stock_value = 0
        for tk, pos in stock_pos.items():
            df = stock_data.get(tk)
            if df is not None and trade_date in df.index:
                stock_value += pos["shares"] * float(df.loc[trade_date,"Close"])

        gold_value = 0
        if gold_pos is not None and gld_df is not None and trade_date in gld_df.index:
            gold_value = gold_pos["shares"] * float(gld_df.loc[trade_date,"Close"])

        total_value = capital + stock_value + gold_value
        equity_curve.append({"date": str(trade_date.date()), "equity": round(total_value,2),
                              "cash": round(capital,2), "stocks": round(stock_value,2),
                              "gold": round(gold_value,2)})

        if total_value > max_equity: max_equity = total_value
        dd = (max_equity - total_value) / max_equity * 100
        if dd > max_drawdown: max_drawdown = dd

        if i == 0: continue

        # ── Macro regime
        regime, mac_score, mac_info = macro_regime(macro_data, signal_date)

        # ── Scan signals
        candidates = {}
        for tk in GROWTH_STOCKS:
            sig = stock_signal(tk, stock_data, macro_data, signal_date)
            if sig and sig["score"] >= MIN_STOCK_SCORE:
                candidates[tk] = sig

        g_score, g_info = gold_signal(macro_data, signal_date)

        # ── Allocation target
        alloc = decide_allocation(regime, mac_score, candidates, g_score)

        # ── Exit stock positions
        to_exit = []
        for tk, pos in stock_pos.items():
            df = stock_data.get(tk)
            if df is None or trade_date not in df.index: continue

            cur_close = float(df.loc[trade_date,"Close"])
            cur_open  = float(df.loc[trade_date,"Open"])
            sig = stock_signal(tk, stock_data, macro_data, signal_date)
            sig_score = sig["score"] if sig else None

            should_exit, reason, exit_price = check_exit_pos(
                pos, cur_close, cur_open,
                STOCK_SL_PCT, STOCK_TP_PCT, STOCK_TRAIL_PCT,
                MAX_HOLD_DAYS, pos["hold_days"], sig_score
            )

            # Exit stocks when regime turns very bearish
            if not should_exit and regime in ("RISK_OFF","CRISIS") and pos["hold_days"] >= 2:
                should_exit, reason, exit_price = True, "REGIME_EXIT", cur_open

            if should_exit:
                gross   = pos["shares"]*(exit_price-pos["entry_price"])
                comm    = pos["shares"]*exit_price*COMMISSION
                net_pnl = gross - comm - pos["entry_commission"]
                capital += pos["shares"]*exit_price - comm

                trades.append({
                    "asset":"stock:"+tk, "type":"STOCK",
                    "date_entry": str(pos["entry_date"].date()),
                    "date_exit":  str(trade_date.date()),
                    "hold_days":  (trade_date-pos["entry_date"]).days,
                    "entry": round(pos["entry_price"],2),
                    "exit":  round(exit_price,2),
                    "reason": reason,
                    "pnl_pct": round((exit_price-pos["entry_price"])/pos["entry_price"]*100,2),
                    "net_pnl": round(net_pnl,2),
                    "regime": regime,
                    "capital_after": round(capital,2),
                })
                to_exit.append(tk)
            else:
                pos["hold_days"] += 1

        for tk in to_exit:
            del stock_pos[tk]

        # ── Exit gold position
        if gold_pos is not None and gld_df is not None and trade_date in gld_df.index:
            g_close = float(gld_df.loc[trade_date,"Close"])
            g_open  = float(gld_df.loc[trade_date,"Open"])

            g_exit, g_reason, g_exit_p = check_exit_pos(
                gold_pos, g_close, g_open,
                GOLD_SL_PCT, GOLD_TP_PCT, GOLD_TRAIL_PCT,
                MAX_HOLD_DAYS*2, gold_pos["hold_days"]  # gold holds longer
            )

            # Reduce gold when strong stock signals appear + regime good
            if (not g_exit and regime in ("STRONG_ON","RISK_ON")
                    and len(candidates) >= MAX_STOCK_POS
                    and gold_pos["hold_days"] >= 3
                    and g_score < MIN_GOLD_SCORE):
                g_exit, g_reason, g_exit_p = True, "ROTATE_TO_STOCKS", g_open

            if g_exit:
                gross   = gold_pos["shares"]*(g_exit_p-gold_pos["entry_price"])
                comm    = gold_pos["shares"]*g_exit_p*COMMISSION
                net_pnl = gross - comm - gold_pos["entry_commission"]
                capital += gold_pos["shares"]*g_exit_p - comm

                trades.append({
                    "asset":"GLD", "type":"GOLD",
                    "date_entry": str(gold_pos["entry_date"].date()),
                    "date_exit":  str(trade_date.date()),
                    "hold_days":  (trade_date-gold_pos["entry_date"]).days,
                    "entry": round(gold_pos["entry_price"],2),
                    "exit":  round(g_exit_p,2),
                    "reason": g_reason,
                    "pnl_pct": round((g_exit_p-gold_pos["entry_price"])/gold_pos["entry_price"]*100,2),
                    "net_pnl": round(net_pnl,2),
                    "regime": regime,
                    "capital_after": round(capital,2),
                })
                gold_pos = None
            else:
                if gold_pos:
                    gold_pos["hold_days"] += 1
                    if g_close > gold_pos["peak_price"]:
                        gold_pos["peak_price"] = g_close

        # ── Enter new stock positions
        if i > 5 and regime not in ("RISK_OFF","CRISIS"):
            ranked = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)
            slots  = MAX_STOCK_POS - len(stock_pos)
            for tk, sig in ranked[:slots]:
                if tk in stock_pos: continue
                df = stock_data.get(tk)
                if df is None or trade_date not in df.index: continue
                entry_p = float(df.loc[trade_date,"Open"])
                if entry_p <= 0: continue

                pos_val = min(capital * STOCK_ALLOC_PCT, capital * 0.9)
                if pos_val < 50: continue

                shares   = pos_val / entry_p
                e_comm   = shares * entry_p * COMMISSION
                capital -= shares * entry_p + e_comm

                stock_pos[tk] = {
                    "entry_price": entry_p, "entry_date": trade_date,
                    "shares": shares, "peak_price": entry_p,
                    "hold_days": 0, "entry_commission": e_comm,
                    "signal_score": sig["score"],
                }

        # ── Enter/maintain gold position (runs alongside stocks)
        if (gold_pos is None and gld_df is not None
                and trade_date in gld_df.index
                and g_score >= MIN_GOLD_SCORE
                and alloc["gold"] > 0
                and capital > 200):

            g_open_p = float(gld_df.loc[trade_date,"Open"])
            if g_open_p > 0:
                gold_val = min(capital * alloc["gold"], capital * 0.8)
                g_shares = gold_val / g_open_p
                g_comm   = g_shares * g_open_p * COMMISSION
                capital -= g_shares * g_open_p + g_comm

                gold_pos = {
                    "entry_price": g_open_p, "entry_date": trade_date,
                    "shares": g_shares, "peak_price": g_open_p,
                    "hold_days": 0, "entry_commission": g_comm,
                }

        # ── Daily log
        sv = sum(pos["shares"]*float(stock_data[tk].loc[trade_date,"Close"])
                 for tk, pos in stock_pos.items()
                 if tk in stock_data and trade_date in stock_data[tk].index)
        gv = (gold_pos["shares"]*float(gld_df.loc[trade_date,"Close"])
              if gold_pos is not None and gld_df is not None and trade_date in gld_df.index else 0)
        tot = capital + sv + gv
        tickers_held = list(stock_pos.keys()) + (["GLD"] if gold_pos else [])

        daily_log.append({
            "date": str(trade_date.date()), "regime": regime, "macro_score": mac_score,
            "capital": round(capital,2), "stock_val": round(sv,2),
            "gold_val": round(gv,2), "total": round(tot,2),
            "holdings": tickers_held,
        })

        if i % 10 == 1 or i == len(common_dates)-1:
            held_str = ",".join(tickers_held) if tickers_held else "CASH"
            print(f"  {str(trade_date.date()):<12} {regime:<12} "
                  f"{held_str:<20} "
                  f"${gv:>7,.0f}    "
                  f"${capital:>7,.0f}  "
                  f"${tot:>8,.0f}")

    # ── Force close remaining
    last_date = common_dates[-1]

    for tk, pos in stock_pos.items():
        df = stock_data.get(tk)
        if df is None: continue
        lp = float(df.loc[last_date,"Close"]) if last_date in df.index else pos["entry_price"]
        gross   = pos["shares"]*(lp-pos["entry_price"])
        comm    = pos["shares"]*lp*COMMISSION
        net_pnl = gross-comm-pos["entry_commission"]
        capital += pos["shares"]*lp-comm
        trades.append({
            "asset":"stock:"+tk, "type":"STOCK",
            "date_entry": str(pos["entry_date"].date()), "date_exit": str(last_date.date()),
            "hold_days": (last_date-pos["entry_date"]).days,
            "entry": round(pos["entry_price"],2), "exit": round(lp,2),
            "reason": "END_OF_PERIOD",
            "pnl_pct": round((lp-pos["entry_price"])/pos["entry_price"]*100,2),
            "net_pnl": round(net_pnl,2), "regime":"N/A",
            "capital_after": round(capital,2),
        })

    if gold_pos is not None and gld_df is not None:
        lp = float(gld_df.loc[last_date,"Close"]) if last_date in gld_df.index else gold_pos["entry_price"]
        gross   = gold_pos["shares"]*(lp-gold_pos["entry_price"])
        comm    = gold_pos["shares"]*lp*COMMISSION
        net_pnl = gross-comm-gold_pos["entry_commission"]
        capital += gold_pos["shares"]*lp-comm
        trades.append({
            "asset":"GLD", "type":"GOLD",
            "date_entry": str(gold_pos["entry_date"].date()), "date_exit": str(last_date.date()),
            "hold_days": (last_date-gold_pos["entry_date"]).days,
            "entry": round(gold_pos["entry_price"],2), "exit": round(lp,2),
            "reason": "END_OF_PERIOD",
            "pnl_pct": round((lp-gold_pos["entry_price"])/gold_pos["entry_price"]*100,2),
            "net_pnl": round(net_pnl,2), "regime":"N/A",
            "capital_after": round(capital,2),
        })

    return {
        "trades": trades, "equity_curve": equity_curve,
        "daily_log": daily_log,
        "final_capital": round(capital,2),
        "max_drawdown":  round(max_drawdown,2),
    }


# ── Analytics ─────────────────────────────────────────────

def analyze(results):
    trades = results["trades"]
    if not trades: return {"error": "No trades"}

    df     = pd.DataFrame(trades)
    total  = len(df)
    wins   = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]

    win_rate      = len(wins)/total*100
    avg_win       = wins["net_pnl"].mean()   if len(wins)>0   else 0
    avg_loss      = losses["net_pnl"].mean() if len(losses)>0 else 0
    pf_d          = abs(losses["net_pnl"].sum())
    profit_factor = abs(wins["net_pnl"].sum())/pf_d if pf_d>0 else 99.0
    total_pnl     = results["final_capital"] - INITIAL_CAPITAL
    total_ret     = total_pnl/INITIAL_CAPITAL*100
    avg_hold      = df["hold_days"].mean()

    eq = pd.DataFrame(results["equity_curve"])
    eq["ret"] = eq["equity"].pct_change()
    sharpe = (eq["ret"].mean()/eq["ret"].std()*(252**0.5) if eq["ret"].std()>0 else 0)

    stock_df = df[df["type"]=="STOCK"]
    gold_df  = df[df["type"]=="GOLD"]

    # Time in each bucket
    dl = pd.DataFrame(results["daily_log"])
    avg_cash_pct   = (dl["capital"]   / dl["total"]*100).mean()
    avg_stock_pct  = (dl["stock_val"] / dl["total"]*100).mean()
    avg_gold_pct   = (dl["gold_val"]  / dl["total"]*100).mean()

    regime_c = dl["regime"].value_counts().to_dict() if "regime" in dl else {}

    asset_stats = (df.groupby("type")
                   .agg(trades=("net_pnl","count"),
                        total_pnl=("net_pnl","sum"),
                        win_rate=("net_pnl", lambda x:(x>0).mean()*100),
                        avg_pnl=("net_pnl","mean"),
                        avg_hold=("hold_days","mean"))
                   .reset_index().to_dict("records"))

    ticker_stats = (df.groupby("asset")
                    .agg(trades=("net_pnl","count"),
                         total_pnl=("net_pnl","sum"),
                         win_rate=("net_pnl",lambda x:(x>0).mean()*100),
                         avg_pnl=("net_pnl","mean"),
                         avg_hold=("hold_days","mean"))
                    .sort_values("total_pnl",ascending=False)
                    .reset_index().to_dict("records"))

    return {
        "total_trades":    total,
        "stock_trades":    len(stock_df),
        "gold_trades":     len(gold_df),
        "win_rate":        round(win_rate,1),
        "avg_win":         round(avg_win,2),
        "avg_loss":        round(avg_loss,2),
        "profit_factor":   round(profit_factor,2),
        "total_pnl":       round(total_pnl,2),
        "total_return":    round(total_ret,2),
        "final_capital":   results["final_capital"],
        "max_drawdown":    results["max_drawdown"],
        "sharpe_ratio":    round(sharpe,2),
        "avg_hold_days":   round(avg_hold,1),
        "stock_pnl":       round(stock_df["net_pnl"].sum(),2) if len(stock_df)>0 else 0,
        "gold_pnl":        round(gold_df["net_pnl"].sum(),2)  if len(gold_df)>0 else 0,
        "avg_stock_pct":   round(avg_stock_pct,1),
        "avg_gold_pct":    round(avg_gold_pct,1),
        "avg_cash_pct":    round(avg_cash_pct,1),
        "regime_counts":   regime_c,
        "exit_reasons":    df["exit_reason"].value_counts().to_dict() if "exit_reason" in df else df["reason"].value_counts().to_dict(),
        "best_trade":      df.loc[df["net_pnl"].idxmax()][["asset","date_entry","date_exit","pnl_pct","net_pnl","hold_days"]].to_dict(),
        "worst_trade":     df.loc[df["net_pnl"].idxmin()][["asset","date_entry","date_exit","pnl_pct","net_pnl","hold_days"]].to_dict(),
        "asset_stats":     asset_stats,
        "ticker_stats":    ticker_stats,
    }


# ── Report ────────────────────────────────────────────────

def print_report(s, results):
    SEP = "=" * 66
    print(f"\n{SEP}")
    print("  3-ASSET SWING TRADE  --  STOCKS + GOLD + CASH")
    print(f"  {START_DATE} to {END_DATE}  |  Start: ${INITIAL_CAPITAL:,.0f}")
    print(SEP)

    tag = "PROFIT" if s["total_return"] > 0 else "LOSS"
    print(f"\n  [{tag}] Total Return    : {s['total_return']:+.2f}%")
    print(f"  Final Capital   : ${s['final_capital']:,.2f}")
    print(f"  Net P&L         : ${s['total_pnl']:+,.2f}")

    print(f"\n  --- Bucket Breakdown ---")
    print(f"  Stocks  : ${s['stock_pnl']:+,.2f}  ({s['stock_trades']} trades)")
    print(f"  Gold    : ${s['gold_pnl']:+,.2f}  ({s['gold_trades']} trades)")
    print(f"  Cash    : always available as buffer")

    print(f"\n  --- Average Allocation (time-weighted) ---")
    bar_s = "#"*int(s['avg_stock_pct']/2)
    bar_g = "#"*int(s['avg_gold_pct']/2)
    bar_c = "#"*int(s['avg_cash_pct']/2)
    print(f"  Stocks {s['avg_stock_pct']:>5.1f}%  |{bar_s}")
    print(f"  Gold   {s['avg_gold_pct']:>5.1f}%  |{bar_g}")
    print(f"  Cash   {s['avg_cash_pct']:>5.1f}%  |{bar_c}")

    print(f"\n  --- Trade Stats ---")
    print(f"  Total Trades    : {s['total_trades']}")
    print(f"  Win Rate        : {s['win_rate']:.1f}%")
    print(f"  Avg Hold        : {s['avg_hold_days']:.1f} days")
    print(f"  Avg Win         : ${s['avg_win']:+.2f}")
    print(f"  Avg Loss        : ${s['avg_loss']:+.2f}")
    print(f"  Profit Factor   : {s['profit_factor']:.2f}x")
    print(f"  Max Drawdown    : -{s['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio    : {s['sharpe_ratio']:.2f}")

    print(f"\n  --- Macro Regime Days ---")
    total_days = sum(s["regime_counts"].values()) or 1
    order = ["STRONG_ON","RISK_ON","NEUTRAL","RISK_OFF","CRISIS"]
    icons = {"STRONG_ON":"[++]","RISK_ON":"[+] ","NEUTRAL":"[=] ","RISK_OFF":"[-] ","CRISIS":"[!!]"}
    for reg in order:
        cnt = s["regime_counts"].get(reg, 0)
        bar = "#"*int(cnt/total_days*30)
        print(f"  {icons.get(reg,'[ ]')} {reg:<12}: {cnt:3d}d  {bar}")

    print(f"\n  --- Exit Reasons ---")
    labels = {
        "TAKE_PROFIT":"[TP]  ","STOP_LOSS":"[SL]  ","TRAILING_STOP":"[TSL] ",
        "SIGNAL_REVERSAL":"[REV] ","REGIME_EXIT":"[REG] ","MAX_HOLD":"[MAX] ",
        "ROTATE_TO_STOCKS":"[ROT] ","END_OF_PERIOD":"[EOP] ",
    }
    for reason, count in s["exit_reasons"].items():
        pct = count/s["total_trades"]*100
        print(f"  {labels.get(reason,'[---] ')} {reason:<18}: {count:2d} ({pct:.0f}%)")

    bt = s["best_trade"]; wt = s["worst_trade"]
    print(f"\n  Best  : {bt['asset']:<12} {bt['date_entry']}->{bt['date_exit']} "
          f"({bt['hold_days']}d)  {bt['pnl_pct']:+.2f}%  ${bt['net_pnl']:+.2f}")
    print(f"  Worst : {wt['asset']:<12} {wt['date_entry']}->{wt['date_exit']} "
          f"({wt['hold_days']}d)  {wt['pnl_pct']:+.2f}%  ${wt['net_pnl']:+.2f}")

    print(f"\n  --- Per-Asset Type ---")
    for row in s["asset_stats"]:
        tag2 = "(+)" if row["total_pnl"]>0 else "(-)"
        print(f"  {tag2} {row['type']:<6}  trades={int(row['trades'])}  "
              f"win={row['win_rate']:.0f}%  hold={row['avg_hold']:.1f}d  "
              f"total=${row['total_pnl']:+.2f}  avg=${row['avg_pnl']:+.2f}")

    print(f"\n  --- Per-Ticker ---")
    print(f"  {'':3} {'Asset':<14} {'#':>3}  {'Win%':>5}  {'Hold':>5}  {'P&L':>9}")
    print(f"  {'-'*46}")
    for row in s["ticker_stats"]:
        tag2 = "(+)" if row["total_pnl"]>0 else "(-)"
        print(f"  {tag2} {row['asset']:<14} {int(row['trades']):>3}  "
              f"{row['win_rate']:>5.1f}%  {row['avg_hold']:>4.1f}d  "
              f"${row['total_pnl']:>8.2f}")

    # Equity curve with 3 asset layers
    eq   = pd.DataFrame(results["equity_curve"])
    tots = eq["equity"].values
    stkv = eq["stocks"].values
    gldv = eq["gold"].values
    cshv = eq["cash"].values

    pts = np.interp(np.linspace(0,len(tots)-1,55), range(len(tots)), tots)
    lo  = min(pts)*0.998; hi = max(pts)*1.002
    H   = 8

    print(f"\n  Equity Curve  (${INITIAL_CAPITAL:,.0f} -> ${tots[-1]:,.2f})")
    print(f"  Legend: # = total equity")
    for r in range(H,-1,-1):
        thresh = lo + (hi-lo)*r/H
        label  = f"  ${thresh:>7,.0f} |"
        line   = "".join("#" if v>=thresh else "." for v in pts)
        print(label+line)
    print(f"           {' '*8}+" + "-"*55)

    # Show avg allocation over time as bottom bar
    n = 55
    alloc_line = "           " + " "*9
    for j in range(n):
        idx = int(j * len(eq) / n)
        s_  = float(stkv[idx]); g_ = float(gldv[idx]); c_ = float(cshv[idx])
        tot_ = s_+g_+c_
        if   tot_ > 0 and s_/tot_ > 0.5: alloc_line += "S"
        elif tot_ > 0 and g_/tot_ > 0.3: alloc_line += "G"
        elif tot_ > 0 and s_/tot_ > 0.2 and g_/tot_ > 0.2: alloc_line += "M"
        else:                              alloc_line += "C"
    print(alloc_line + "   <- S=Stock G=Gold M=Mixed C=Cash")
    print(f"           {eq['date'].iloc[0]}{'':>35}{eq['date'].iloc[-1]}")

    print(f"\n{SEP}")
    print(f"  FINAL COMPARISON -- 3 Months | $5,000 Portfolio")
    print(f"  {'-'*50}")
    results_table = [
        ("Day Trade (1d hold)",    "-6.41%",  "$4,679",  "9.26%",  "-1.89"),
        ("Swing Trade (stocks)",   "+10.58%", "$5,529",  "4.83%",  "+2.71"),
        ("Swing + Gold only",      "-0.51%",  "$4,974",  "10.53%", "-0.03"),
        (f"3-Asset (this run)",    f"{s['total_return']:+.2f}%",
                                   f"${s['final_capital']:,.0f}",
                                   f"{s['max_drawdown']:.2f}%",
                                   f"{s['sharpe_ratio']:+.2f}"),
    ]
    print(f"  {'Strategy':<26} {'Return':>8}  {'Capital':>8}  {'MaxDD':>6}  {'Sharpe':>7}")
    print(f"  {'-'*62}")
    for row in results_table:
        print(f"  {row[0]:<26} {row[1]:>8}  {row[2]:>8}  {row[3]:>6}  {row[4]:>7}")

    print(f"\n  Full log saved -> backtest_3asset.json")
    print(SEP+"\n")


# ── Main ──────────────────────────────────────────────────

def main():
    print("\n"+"="*66)
    print("  3-Asset Swing Trade  |  Stocks + Gold + Cash  |  $5,000")
    print("="*66)

    ext = (datetime.strptime(START_DATE,"%Y-%m-%d")-timedelta(days=50)).strftime("%Y-%m-%d")

    print("\n[1/3] Loading stock data...")
    stock_data = load_all(GROWTH_STOCKS, ext, END_DATE)
    print("[2/3] Loading macro + gold data...")
    macro_data = load_all(MACRO_TICKERS, ext, END_DATE)

    print("[3/3] Running simulation...\n")
    results = run_backtest(stock_data, macro_data)

    stats = analyze(results)
    if "error" in stats:
        print(f"  No trades. Check data or lower thresholds.")
        return

    print_report(stats, results)

    with open("backtest_3asset.json","w") as f:
        json.dump({
            "config": {
                "initial_capital": INITIAL_CAPITAL,
                "start": START_DATE, "end": END_DATE,
                "stock_sl": STOCK_SL_PCT, "stock_tp": STOCK_TP_PCT,
                "stock_trail": STOCK_TRAIL_PCT,
                "gold_sl": GOLD_SL_PCT, "gold_tp": GOLD_TP_PCT,
                "gold_trail": GOLD_TRAIL_PCT,
                "max_hold_days": MAX_HOLD_DAYS,
            },
            "summary": stats,
            "trades": results["trades"],
            "equity_curve": results["equity_curve"],
            "daily_log": results["daily_log"],
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
