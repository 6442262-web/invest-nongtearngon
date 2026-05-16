# -*- coding: utf-8 -*-
"""
Swing Trade + Gold Rotation Backtest -- 3 Months
Portfolio : $5,000
Logic     : Buy growth stocks when signals strong
            Rotate to Gold (GLD) when no stock signals / macro bad
            Hold each position until SL / TP / Trailing Stop / Signal Exit
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

MAX_STOCK_POS     = 2       # max growth stock positions
POSITION_SIZE_PCT = 0.40    # 40% per stock position
GOLD_SIZE_PCT     = 0.80    # 80% of free cash into gold when rotating
STOP_LOSS_PCT     = 0.07    # -7% hard stop
TAKE_PROFIT_PCT   = 0.15    # +15% take profit
TRAILING_STOP_PCT = 0.05    # 5% trailing from peak
MAX_HOLD_DAYS     = 20
COMMISSION        = 0.001

# Signal thresholds
MIN_STOCK_SCORE   = 4       # min score to buy a stock
GOLD_TRIGGER_DAYS = 2       # days with no stock signal before moving to gold
GOLD_EXIT_SCORE   = 3       # if any stock reaches this score, exit gold

GROWTH_STOCKS = [
    "NVDA","MSFT","AAPL","META","GOOGL",
    "AMZN","TSLA","AVGO","NOW",
    "ADBE","CRM","PANW","DDOG","NET","ZS",
]

MACRO_TICKERS = ["DX-Y.NYB","^VIX","^TNX","HYG","QQQ","XLK","GLD","GC=F"]

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


# ── Macro Environment Scorer ──────────────────────────────

def macro_environment(macro_data, signal_date):
    """
    Returns:
      'RISK_ON'  : buy stocks
      'NEUTRAL'  : cautious, small positions
      'RISK_OFF' : buy gold
    """
    def last_c(tk, n):
        df = macro_data.get(tk)
        if df is None: return None
        sub = df.loc[:signal_date, "Close"].dropna()
        return sub.values[-n:] if len(sub) >= n else None

    score = 0
    details = {}

    # VIX
    c = last_c("^VIX", 2)
    if c is not None:
        vix = float(c[-1])
        details["vix"] = round(vix, 1)
        if   vix < 16: score += 2
        elif vix < 20: score += 1
        elif vix < 28: score -= 1
        else:          score -= 3   # extreme fear

    # HYG (credit spread proxy)
    c = last_c("HYG", 6)
    if c is not None and c[-6] > 0:
        hyg_5d = (c[-1] - c[-6]) / c[-6] * 100
        details["hyg_5d"] = round(hyg_5d, 2)
        if   hyg_5d >  0.5: score += 2   # credit improving
        elif hyg_5d >  0.0: score += 1
        elif hyg_5d < -0.5: score -= 2   # credit deteriorating

    # QQQ trend (5d)
    c = last_c("QQQ", 6)
    if c is not None and c[-6] > 0:
        qqq_5d = (c[-1] - c[-6]) / c[-6] * 100
        details["qqq_5d"] = round(qqq_5d, 2)
        if   qqq_5d >  2: score += 2
        elif qqq_5d >  0: score += 1
        elif qqq_5d < -3: score -= 2

    # DXY
    c = last_c("DX-Y.NYB", 2)
    if c is not None and c[-2] > 0:
        dxy_1d = (c[-1] - c[-2]) / c[-2] * 100
        details["dxy_1d"] = round(dxy_1d, 2)
        if dxy_1d >  0.5: score -= 1   # USD strong = headwind
        if dxy_1d < -0.5: score += 1   # USD weak = tailwind

    # Gold trend (gold rising = risk-off signal)
    c = last_c("GLD", 6)
    if c is not None and c[-6] > 0:
        gold_5d = (c[-1] - c[-6]) / c[-6] * 100
        details["gold_5d"] = round(gold_5d, 2)
        if gold_5d > 2:  score -= 1   # gold rally = risk-off pressure
        if gold_5d < -1: score += 1   # gold weak = risk-on

    details["macro_score"] = score

    if   score >= 3:  env = "RISK_ON"
    elif score >= 0:  env = "NEUTRAL"
    else:             env = "RISK_OFF"

    return env, details


# ── Stock Signal Scorer ───────────────────────────────────

def get_stock_score(tk, stock_data, macro_data, signal_date):
    df = stock_data.get(tk)
    if df is None: return None

    def lc(n):
        sub = df.loc[:signal_date, "Close"].dropna()
        return sub.values[-n:] if len(sub) >= n else None

    def lv(n):
        sub = df.loc[:signal_date, "Volume"].dropna()
        return sub.values[-n:] if len(sub) >= n else None

    c2  = lc(2);  c6 = lc(6)
    c11 = lc(11); c21= lc(21)
    vol = lv(25)

    if c2 is None: return None
    c_now, c_prev = float(c2[-1]), float(c2[-2])
    if c_now <= 0 or c_prev <= 0: return None

    mom_1d  = (c_now - c_prev)          / c_prev  * 100
    mom_5d  = ((c_now - float(c6[-6]))  / float(c6[-6])  * 100) if c6  is not None else 0
    mom_10d = ((c_now - float(c11[-11]))/ float(c11[-11])* 100) if c11 is not None else 0
    mom_20d = ((c_now - float(c21[-21]))/ float(c21[-21])* 100) if c21 is not None else 0

    vol_ratio = 1.0
    if vol is not None and len(vol) >= 20:
        vol_ratio = float(vol[-5:].mean()) / float(vol[-20:].mean())

    above_10ma = (c_now > float(c11.mean())) if c11 is not None else False
    above_20ma = (c_now > float(c21.mean())) if c21 is not None else False

    score = 0
    if   mom_1d  >  3:  score += 3
    elif mom_1d  >  1.5:score += 2
    elif mom_1d  >  0.5:score += 1
    elif mom_1d  < -1.5:score -= 2

    if   mom_5d  >  6:  score += 3
    elif mom_5d  >  3:  score += 2
    elif mom_5d  >  1:  score += 1
    elif mom_5d  < -4:  score -= 2

    if   mom_10d > 10:  score += 2
    elif mom_10d >  5:  score += 1

    if   mom_20d > 15:  score += 2
    elif mom_20d >  8:  score += 1
    elif mom_20d < -5:  score -= 1

    if above_10ma: score += 1
    if above_20ma: score += 1

    if   vol_ratio > 2.0: score += 2
    elif vol_ratio > 1.3: score += 1

    return {
        "score":     score,
        "mom_1d":    round(mom_1d, 2),
        "mom_5d":    round(mom_5d, 2),
        "mom_10d":   round(mom_10d, 2),
        "vol_ratio": round(vol_ratio, 2),
        "price":     round(c_now, 2),
        "above_20ma":above_20ma,
    }


# ── Gold Signal ───────────────────────────────────────────

def get_gold_score(macro_data, signal_date):
    def lc(tk, n):
        df = macro_data.get(tk)
        if df is None: return None
        sub = df.loc[:signal_date, "Close"].dropna()
        return sub.values[-n:] if len(sub) >= n else None

    score = 0
    c = lc("GLD", 21)
    if c is None: return 0, {}

    c_now = float(c[-1])
    mom_5d  = (c_now - float(c[-6])) / float(c[-6])  * 100 if len(c) >= 6  else 0
    mom_10d = (c_now - float(c[-11]))/ float(c[-11]) * 100 if len(c) >= 11 else 0
    mom_20d = (c_now - float(c[-21]))/ float(c[-21]) * 100 if len(c) >= 21 else 0
    above_10ma = c_now > float(c[-10:].mean())
    above_20ma = c_now > float(c.mean())

    if mom_5d  >  2: score += 2
    elif mom_5d>  0: score += 1
    elif mom_5d< -2: score -= 1

    if mom_10d >  4: score += 2
    elif mom_10d>  1:score += 1

    if above_10ma: score += 1
    if above_20ma: score += 1

    return score, {
        "mom_5d": round(mom_5d,2), "mom_10d": round(mom_10d,2),
        "mom_20d": round(mom_20d,2), "price": round(c_now,2)
    }


# ── Exit Check ────────────────────────────────────────────

def check_exit(pos, current_price, current_open, signal_score, hold_days):
    entry = pos["entry_price"]
    peak  = pos["peak_price"]

    if current_price > peak:
        pos["peak_price"] = current_price

    if current_price <= entry * (1 - STOP_LOSS_PCT):
        return True, "STOP_LOSS",   min(current_open, entry*(1-STOP_LOSS_PCT))
    if current_price >= entry * (1 + TAKE_PROFIT_PCT):
        return True, "TAKE_PROFIT", max(current_open, entry*(1+TAKE_PROFIT_PCT))
    if current_price <= peak * (1 - TRAILING_STOP_PCT):
        return True, "TRAILING_STOP", current_open
    if hold_days >= MAX_HOLD_DAYS:
        return True, "MAX_HOLD", current_open
    if signal_score is not None and signal_score < -1:
        return True, "SIGNAL_REVERSAL", current_open

    return False, None, None


# ── Backtest Engine ───────────────────────────────────────

def run_backtest(stock_data, macro_data):
    # Common dates
    date_sets = [set(df.loc[START_DATE:END_DATE].index.normalize())
                 for df in stock_data.values()]
    common_dates = sorted(set.intersection(*date_sets))

    capital      = INITIAL_CAPITAL
    positions    = {}  # {ticker: pos_dict}  includes "GOLD" slot
    gold_pos     = None
    no_signal_days = 0

    trades       = []
    equity_curve = []
    regime_log   = []   # track macro regime each day
    max_equity   = capital
    max_drawdown = 0.0

    print(f"  Trading days: {len(common_dates)}")
    print(f"  Stocks: SL={STOP_LOSS_PCT*100:.0f}% TP={TAKE_PROFIT_PCT*100:.0f}% Trail={TRAILING_STOP_PCT*100:.0f}% Max={MAX_HOLD_DAYS}d")
    print(f"  Gold: rotate in when {GOLD_TRIGGER_DAYS} days no stock signal or RISK_OFF\n")

    gld_df = macro_data.get("GLD")

    for i, trade_ts in enumerate(common_dates):
        trade_date  = trade_ts
        signal_date = common_dates[i-1] if i > 0 else trade_ts

        # ── Mark-to-market
        portfolio_value = capital
        for tk, pos in positions.items():
            df = stock_data.get(tk)
            if df is not None and trade_date in df.index:
                portfolio_value += pos["shares"] * float(df.loc[trade_date,"Close"])
        if gold_pos is not None and gld_df is not None and trade_date in gld_df.index:
            portfolio_value += gold_pos["shares"] * float(gld_df.loc[trade_date,"Close"])

        equity_curve.append({"date": str(trade_date.date()), "equity": round(portfolio_value,2)})
        if portfolio_value > max_equity: max_equity = portfolio_value
        dd = (max_equity - portfolio_value) / max_equity * 100
        if dd > max_drawdown: max_drawdown = dd

        if i == 0: continue

        # ── Macro regime
        regime, macro_details = macro_environment(macro_data, signal_date)
        regime_log.append({"date": str(trade_date.date()), "regime": regime, **macro_details})

        # ── Exit existing stock positions
        to_exit = []
        for tk, pos in positions.items():
            df = stock_data.get(tk)
            if df is None or trade_date not in df.index: continue

            cur_price = float(df.loc[trade_date,"Close"])
            cur_open  = float(df.loc[trade_date,"Open"])
            sig       = get_stock_score(tk, stock_data, macro_data, signal_date)
            sig_score = sig["score"] if sig else None

            should_exit, reason, exit_price = check_exit(
                pos, cur_price, cur_open, sig_score, pos["hold_days"]
            )
            # Also exit on RISK_OFF regime
            if regime == "RISK_OFF" and pos["hold_days"] >= 2:
                should_exit, reason, exit_price = True, "REGIME_EXIT", cur_open

            if should_exit:
                gross_pnl = pos["shares"] * (exit_price - pos["entry_price"])
                comm      = pos["shares"] * exit_price * COMMISSION
                net_pnl   = gross_pnl - comm - pos["entry_commission"]
                capital  += pos["shares"] * exit_price - comm

                trades.append({
                    "asset":        tk,
                    "asset_type":   "STOCK",
                    "date_entry":   str(pos["entry_date"].date()),
                    "date_exit":    str(trade_date.date()),
                    "hold_days":    (trade_date - pos["entry_date"]).days,
                    "entry_price":  round(pos["entry_price"],2),
                    "exit_price":   round(exit_price,2),
                    "exit_reason":  reason,
                    "pnl_pct":      round((exit_price-pos["entry_price"])/pos["entry_price"]*100,2),
                    "net_pnl":      round(net_pnl,2),
                    "macro_regime": regime,
                    "capital_after":round(capital,2),
                })
                to_exit.append(tk)
            else:
                pos["hold_days"] += 1

        for tk in to_exit:
            del positions[tk]

        # ── Exit gold position if stock signals return
        if gold_pos is not None and gld_df is not None and trade_date in gld_df.index:
            gld_open  = float(gld_df.loc[trade_date,"Open"])
            gld_close = float(gld_df.loc[trade_date,"Close"])
            gld_score, _ = get_gold_score(macro_data, signal_date)

            # Check gold SL/TP/Trail
            g_exit, g_reason, g_exit_price = check_exit(
                gold_pos, gld_close, gld_open, gld_score, gold_pos["hold_days"]
            )

            # Exit gold if regime turned risk-on AND stocks have signals
            if not g_exit and regime == "RISK_ON":
                best_stock_score = max(
                    (get_stock_score(tk, stock_data, macro_data, signal_date) or {}).get("score",0)
                    for tk in GROWTH_STOCKS
                )
                if best_stock_score >= GOLD_EXIT_SCORE:
                    g_exit, g_reason, g_exit_price = True, "ROTATE_TO_STOCKS", gld_open

            if g_exit:
                gross_pnl = gold_pos["shares"] * (g_exit_price - gold_pos["entry_price"])
                comm      = gold_pos["shares"] * g_exit_price * COMMISSION
                net_pnl   = gross_pnl - comm - gold_pos["entry_commission"]
                capital  += gold_pos["shares"] * g_exit_price - comm

                trades.append({
                    "asset":        "GLD",
                    "asset_type":   "GOLD",
                    "date_entry":   str(gold_pos["entry_date"].date()),
                    "date_exit":    str(trade_date.date()),
                    "hold_days":    (trade_date - gold_pos["entry_date"]).days,
                    "entry_price":  round(gold_pos["entry_price"],2),
                    "exit_price":   round(g_exit_price,2),
                    "exit_reason":  g_reason,
                    "pnl_pct":      round((g_exit_price-gold_pos["entry_price"])/gold_pos["entry_price"]*100,2),
                    "net_pnl":      round(net_pnl,2),
                    "macro_regime": regime,
                    "capital_after":round(capital,2),
                })
                gold_pos = None
            else:
                gold_pos["hold_days"] += 1
                if gld_close > gold_pos["peak_price"]:
                    gold_pos["peak_price"] = gld_close

        # ── Scan for new stock entries
        if i > 5 and len(positions) < MAX_STOCK_POS and regime != "RISK_OFF":
            candidates = {}
            for tk in GROWTH_STOCKS:
                if tk in positions: continue
                sig = get_stock_score(tk, stock_data, macro_data, signal_date)
                if sig and sig["score"] >= MIN_STOCK_SCORE and sig["vol_ratio"] >= 1.1:
                    candidates[tk] = sig

            if candidates:
                no_signal_days = 0
                ranked = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)
                slots  = MAX_STOCK_POS - len(positions)

                for tk, sig in ranked[:slots]:
                    df = stock_data.get(tk)
                    if df is None or trade_date not in df.index: continue
                    entry_price = float(df.loc[trade_date,"Open"])
                    if entry_price <= 0: continue

                    pos_val = min(capital * POSITION_SIZE_PCT, capital - 300)
                    if pos_val < 100: continue

                    shares   = pos_val / entry_price
                    e_comm   = shares * entry_price * COMMISSION
                    capital -= shares * entry_price + e_comm

                    positions[tk] = {
                        "entry_price": entry_price, "entry_date": trade_date,
                        "shares": shares, "peak_price": entry_price,
                        "hold_days": 0, "entry_commission": e_comm,
                        "signal_score": sig["score"],
                    }
            else:
                no_signal_days += 1
        else:
            if regime == "RISK_OFF":
                no_signal_days += 1

        # ── Rotate to Gold if no stock signal for N days or RISK_OFF
        should_buy_gold = (
            gold_pos is None
            and gld_df is not None
            and len(positions) == 0
            and capital > 500
            and (no_signal_days >= GOLD_TRIGGER_DAYS or regime == "RISK_OFF")
        )

        if should_buy_gold and trade_date in gld_df.index:
            gld_score, gld_details = get_gold_score(macro_data, signal_date)
            gld_open = float(gld_df.loc[trade_date,"Open"])

            if gld_open > 0 and gld_score >= 0:  # gold not in downtrend
                gold_size = capital * GOLD_SIZE_PCT
                g_shares  = gold_size / gld_open
                g_comm    = g_shares * gld_open * COMMISSION
                capital  -= g_shares * gld_open + g_comm

                gold_pos = {
                    "entry_price": gld_open, "entry_date": trade_date,
                    "shares": g_shares, "peak_price": gld_open,
                    "hold_days": 0, "entry_commission": g_comm,
                }
                print(f"  [{trade_date.date()}] ROTATE -> GOLD  @ ${gld_open:.2f}  "
                      f"(regime={regime}, no_signal={no_signal_days}d, gold_score={gld_score})")

        # Print stock entries
        for tk in list(positions.keys()):
            pos = positions[tk]
            if pos["hold_days"] == 0:
                print(f"  [{trade_date.date()}] BUY {tk:<5} @ ${pos['entry_price']:.2f}  "
                      f"score={pos['signal_score']}  regime={regime}")

    # ── Force close all at end
    last_date = common_dates[-1]
    for tk, pos in positions.items():
        df = stock_data.get(tk)
        if df is None: continue
        lp = float(df.loc[last_date,"Close"]) if last_date in df.index else pos["entry_price"]
        gross_pnl = pos["shares"]*(lp-pos["entry_price"])
        comm      = pos["shares"]*lp*COMMISSION
        net_pnl   = gross_pnl-comm-pos["entry_commission"]
        capital  += pos["shares"]*lp-comm
        trades.append({
            "asset": tk, "asset_type": "STOCK",
            "date_entry": str(pos["entry_date"].date()), "date_exit": str(last_date.date()),
            "hold_days": (last_date-pos["entry_date"]).days,
            "entry_price": round(pos["entry_price"],2), "exit_price": round(lp,2),
            "exit_reason": "END_OF_PERIOD",
            "pnl_pct": round((lp-pos["entry_price"])/pos["entry_price"]*100,2),
            "net_pnl": round(net_pnl,2), "macro_regime":"N/A", "capital_after":round(capital,2),
        })

    if gold_pos is not None and gld_df is not None:
        lp = float(gld_df.loc[last_date,"Close"]) if last_date in gld_df.index else gold_pos["entry_price"]
        gross_pnl = gold_pos["shares"]*(lp-gold_pos["entry_price"])
        comm      = gold_pos["shares"]*lp*COMMISSION
        net_pnl   = gross_pnl-comm-gold_pos["entry_commission"]
        capital  += gold_pos["shares"]*lp-comm
        trades.append({
            "asset": "GLD", "asset_type": "GOLD",
            "date_entry": str(gold_pos["entry_date"].date()), "date_exit": str(last_date.date()),
            "hold_days": (last_date-gold_pos["entry_date"]).days,
            "entry_price": round(gold_pos["entry_price"],2), "exit_price": round(lp,2),
            "exit_reason": "END_OF_PERIOD",
            "pnl_pct": round((lp-gold_pos["entry_price"])/gold_pos["entry_price"]*100,2),
            "net_pnl": round(net_pnl,2), "macro_regime":"N/A", "capital_after":round(capital,2),
        })

    return {
        "trades": trades, "equity_curve": equity_curve,
        "regime_log": regime_log,
        "final_capital": round(capital,2), "max_drawdown": round(max_drawdown,2),
    }


# ── Analytics ─────────────────────────────────────────────

def analyze(results):
    trades = results["trades"]
    if not trades: return {"error": "No trades"}

    df = pd.DataFrame(trades)
    total  = len(df)
    wins   = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]

    win_rate      = len(wins)/total*100
    avg_win       = wins["net_pnl"].mean()   if len(wins)   > 0 else 0
    avg_loss      = losses["net_pnl"].mean() if len(losses) > 0 else 0
    pf_d          = abs(losses["net_pnl"].sum())
    profit_factor = abs(wins["net_pnl"].sum())/pf_d if pf_d > 0 else 99.0
    total_pnl     = results["final_capital"] - INITIAL_CAPITAL
    total_ret     = total_pnl/INITIAL_CAPITAL*100
    avg_hold      = df["hold_days"].mean()

    eq = pd.DataFrame(results["equity_curve"])
    eq["ret"] = eq["equity"].pct_change()
    sharpe = (eq["ret"].mean()/eq["ret"].std()*(252**0.5) if eq["ret"].std()>0 else 0)

    # Split by asset type
    stock_df = df[df["asset_type"]=="STOCK"]
    gold_df  = df[df["asset_type"]=="GOLD"]

    regime_counts = {}
    for r in results["regime_log"]:
        reg = r.get("regime","?")
        regime_counts[reg] = regime_counts.get(reg,0)+1

    # Per-asset breakdown
    asset_stats = (df.groupby("asset")
                   .agg(trades=("net_pnl","count"),
                        total_pnl=("net_pnl","sum"),
                        win_rate=("net_pnl", lambda x:(x>0).mean()*100),
                        avg_pnl=("net_pnl","mean"),
                        avg_hold=("hold_days","mean"))
                   .sort_values("total_pnl",ascending=False)
                   .reset_index().to_dict("records"))

    return {
        "total_trades":  total,
        "stock_trades":  len(stock_df),
        "gold_trades":   len(gold_df),
        "win_rate":      round(win_rate,1),
        "avg_win":       round(avg_win,2),
        "avg_loss":      round(avg_loss,2),
        "profit_factor": round(profit_factor,2),
        "total_pnl":     round(total_pnl,2),
        "total_return":  round(total_ret,2),
        "final_capital": results["final_capital"],
        "max_drawdown":  results["max_drawdown"],
        "sharpe_ratio":  round(sharpe,2),
        "avg_hold_days": round(avg_hold,1),
        "stock_pnl":     round(stock_df["net_pnl"].sum(),2) if len(stock_df)>0 else 0,
        "gold_pnl":      round(gold_df["net_pnl"].sum(),2)  if len(gold_df)>0 else 0,
        "regime_counts": regime_counts,
        "exit_reasons":  df["exit_reason"].value_counts().to_dict(),
        "best_trade":    df.loc[df["net_pnl"].idxmax()][["asset","date_entry","date_exit","pnl_pct","net_pnl","hold_days"]].to_dict(),
        "worst_trade":   df.loc[df["net_pnl"].idxmin()][["asset","date_entry","date_exit","pnl_pct","net_pnl","hold_days"]].to_dict(),
        "asset_stats":   asset_stats,
    }


# ── Report ────────────────────────────────────────────────

def print_report(s, results):
    SEP = "=" * 66
    print(f"\n{SEP}")
    print("  SWING TRADE + GOLD ROTATION  --  3 MONTHS  |  $5,000")
    print(f"  {START_DATE} to {END_DATE}")
    print(SEP)

    tag = "PROFIT" if s["total_return"] > 0 else "LOSS"
    print(f"\n  [{tag}] Total Return    : {s['total_return']:+.2f}%")
    print(f"  Final Capital   : ${s['final_capital']:,.2f}   (start: ${INITIAL_CAPITAL:,.0f})")
    print(f"  Net P&L         : ${s['total_pnl']:+,.2f}")
    print(f"    Stock P&L     : ${s['stock_pnl']:+,.2f}  ({s['stock_trades']} trades)")
    print(f"    Gold  P&L     : ${s['gold_pnl']:+,.2f}  ({s['gold_trades']} trades)")

    print(f"\n  Total Trades    : {s['total_trades']}")
    print(f"  Win Rate        : {s['win_rate']:.1f}%")
    print(f"  Avg Hold        : {s['avg_hold_days']:.1f} days")
    print(f"  Avg Win         : ${s['avg_win']:+.2f}")
    print(f"  Avg Loss        : ${s['avg_loss']:+.2f}")
    print(f"  Profit Factor   : {s['profit_factor']:.2f}x")
    print(f"\n  Max Drawdown    : -{s['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio    : {s['sharpe_ratio']:.2f}")

    print(f"\n  Macro Regime Days:")
    total_days = sum(s["regime_counts"].values())
    for reg, cnt in s["regime_counts"].items():
        bar = "#"*int(cnt/total_days*30)
        print(f"    {reg:<12}: {cnt:3d}d {bar}")

    print(f"\n  Exit Breakdown:")
    labels = {
        "TAKE_PROFIT":"[TP]  ","STOP_LOSS":"[SL]  ","TRAILING_STOP":"[TSL] ",
        "SIGNAL_REVERSAL":"[REV] ","REGIME_EXIT":"[REG] ","MAX_HOLD":"[MAX] ",
        "ROTATE_TO_STOCKS":"[ROT] ","END_OF_PERIOD":"[EOP] ",
    }
    for reason, count in s["exit_reasons"].items():
        pct = count/s["total_trades"]*100
        lbl = labels.get(reason,"[---] ")
        print(f"    {lbl} {reason:<18}: {count:3d} ({pct:.1f}%)")

    bt = s["best_trade"]; wt = s["worst_trade"]
    print(f"\n  Best  : {bt['asset']:<5} {bt['date_entry']}->{bt['date_exit']} "
          f"({bt['hold_days']}d)  {bt['pnl_pct']:+.2f}%  ${bt['net_pnl']:+.2f}")
    print(f"  Worst : {wt['asset']:<5} {wt['date_entry']}->{wt['date_exit']} "
          f"({wt['hold_days']}d)  {wt['pnl_pct']:+.2f}%  ${wt['net_pnl']:+.2f}")

    print(f"\n  Per-Asset Performance:")
    print(f"  {'':3} {'Asset':<6} {'Type':<7} {'#':>3}  {'Win%':>5}  {'Hold':>5}  {'Total':>9}  {'Avg':>7}")
    print(f"  {'-'*54}")
    for row in s["asset_stats"]:
        tag2 = "(+)" if row["total_pnl"] > 0 else "(-)"
        atype = "GOLD " if row["asset"]=="GLD" else "STOCK"
        print(f"  {tag2} {row['asset']:<6} {atype} {int(row['trades']):>3}  "
              f"{row['win_rate']:>5.1f}%  "
              f"{row['avg_hold']:>4.1f}d  "
              f"${row['total_pnl']:>8.2f}  "
              f"${row['avg_pnl']:>6.2f}")

    # Equity curve
    eq_vals  = [e["equity"] for e in results["equity_curve"]]
    eq_dates = [e["date"]   for e in results["equity_curve"]]
    reg_vals = [r["regime"] for r in results["regime_log"]]
    pts  = np.interp(np.linspace(0,len(eq_vals)-1,55), range(len(eq_vals)), eq_vals)
    regs = [reg_vals[min(int(i*len(reg_vals)/55), len(reg_vals)-1)] for i in range(55)] if reg_vals else ["?"]*55
    lo, hi = min(pts), max(pts)
    H = 8
    print(f"\n  Equity Curve  (${INITIAL_CAPITAL:,.0f} -> ${eq_vals[-1]:,.2f})")
    for r in range(H, -1, -1):
        thresh = lo + (hi-lo)*r/H
        label  = f"  ${lo+(hi-lo)*r/H:>7,.0f} |"
        line   = "".join("#" if v>=thresh else "." for v in pts)
        print(label+line)
    print(f"           {' '*8}+"+"-"*55)

    # Regime bar below chart
    reg_line = "           " + " "*9
    for reg in regs:
        reg_line += {"RISK_ON":"R","NEUTRAL":"N","RISK_OFF":"G"}.get(reg,"?")
    print(reg_line + "   <- R=RiskOn N=Neutral G=GoldZone")
    print(f"           {eq_dates[0]}{'':>37}{eq_dates[-1]}")

    print(f"\n{SEP}")
    print(f"  COMPARISON  (same 3-month window)")
    print(f"  Day Trade      : -6.41%   ($4,679)")
    print(f"  Swing Trade    : +10.58%  ($5,529)")
    print(f"  Swing+Gold     : {s['total_return']:+.2f}%   (${s['final_capital']:,.0f})")
    print(f"\n  Max Drawdown   : {s['max_drawdown']:.2f}% (Swing was 4.83%)")
    print(f"  Sharpe Ratio   : {s['sharpe_ratio']:.2f} (Swing was 2.71)")
    print(f"\n  Gold acts as 'parking' when stocks have no signal.")
    print(f"  It cushions drawdown during RISK_OFF regimes.")
    print(f"  Full log -> backtest_swing_gold.json")
    print(SEP+"\n")


# ── Main ──────────────────────────────────────────────────

def main():
    print("\n"+"="*66)
    print("  Swing Trade + Gold Rotation  |  $5,000 Portfolio")
    print("  Stocks when signals strong, Gold when no signal / risk-off")
    print("="*66)

    ext_start = (datetime.strptime(START_DATE,"%Y-%m-%d")-timedelta(days=50)).strftime("%Y-%m-%d")

    print("\n[1/3] Loading stock data...")
    stock_data = load_all(GROWTH_STOCKS, ext_start, END_DATE)

    print("[2/3] Loading macro + gold data...")
    macro_data = load_all(MACRO_TICKERS, ext_start, END_DATE)

    print("[3/3] Running simulation...\n")
    results = run_backtest(stock_data, macro_data)

    stats = analyze(results)
    if "error" in stats:
        print("  No trades executed.")
        return

    print_report(stats, results)

    with open("backtest_swing_gold.json","w") as f:
        json.dump({
            "config": {
                "initial_capital": INITIAL_CAPITAL, "start": START_DATE, "end": END_DATE,
                "stop_loss_pct": STOP_LOSS_PCT, "take_profit_pct": TAKE_PROFIT_PCT,
                "trailing_stop_pct": TRAILING_STOP_PCT, "max_hold_days": MAX_HOLD_DAYS,
                "gold_trigger_days": GOLD_TRIGGER_DAYS,
            },
            "summary": stats,
            "trades": results["trades"],
            "equity_curve": results["equity_curve"],
            "regime_log": results["regime_log"],
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
