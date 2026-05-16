# -*- coding: utf-8 -*-
"""
Day Trading Backtest -- 3 Months
Portfolio: $5,000 | Strategy: Momentum + Volume Surge + Macro Filter
Data: yfinance OHLC daily (Buy @ Open, Sell @ Close or SL/TP)
"""

import os, json, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────
INITIAL_CAPITAL   = 5_000.0
START_DATE        = "2026-02-14"
END_DATE          = "2026-05-14"
MAX_POSITIONS     = 2
POSITION_SIZE_PCT = 0.45     # 45% of portfolio per position
STOP_LOSS_PCT     = 0.02     # -2% from open
TAKE_PROFIT_PCT   = 0.03     # +3% from open
COMMISSION        = 0.001    # 0.1% per side
MIN_VOLUME_RATIO  = 1.2
MIN_SIGNAL_SCORE  = 3

GROWTH_STOCKS = [
    "NVDA","MSFT","AAPL","META","GOOGL",
    "AMZN","TSLA","AVGO","ASML","NOW",
    "ADBE","CRM","PANW","MRVL","TTD",
    "AXON","DDOG","NET","ZS","SNOW",
]

# ── Data Loading ──────────────────────────────────────────

def load_all(tickers, start, end):
    print(f"  Downloading {len(tickers)} tickers ({start} to {end})...")
    raw = yf.download(tickers, start=start, end=end,
                      interval="1d", progress=False, auto_adjust=True, group_by="ticker")
    result = {}
    for tk in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[tk].copy()
            df = df.dropna(subset=["Close","Open","High","Low"])
            if len(df) > 15:
                result[tk] = df
        except Exception:
            pass
    print(f"  Loaded: {len(result)}/{len(tickers)} OK")
    return result


# ── Signal Engine ─────────────────────────────────────────

def compute_daily_signals(stock_data, macro_data, signal_date):
    """
    signal_date: the date we compute signals ON (using data UP TO this date)
    These signals drive next-day (trade_date) entries.
    """
    signals = {}

    # Helper: slice df up to signal_date (inclusive) and return last N closes
    def last_closes(df, n):
        sub = df.loc[:signal_date, "Close"].dropna()
        if len(sub) < n:
            return None
        return sub.values[-n:]

    def last_volumes(df, n):
        sub = df.loc[:signal_date, "Volume"].dropna()
        if len(sub) < n:
            return None
        return sub.values[-n:]

    # Macro: VIX filter
    vix_val  = None
    hyg_1d   = None
    dxy_1d   = None

    vix_df = macro_data.get("^VIX")
    if vix_df is not None:
        c = last_closes(vix_df, 2)
        if c is not None:
            vix_val = float(c[-1])

    hyg_df = macro_data.get("HYG")
    if hyg_df is not None:
        c = last_closes(hyg_df, 2)
        if c is not None and c[-2] > 0:
            hyg_1d = (c[-1] - c[-2]) / c[-2] * 100

    dxy_df = macro_data.get("DX-Y.NYB")
    if dxy_df is not None:
        c = last_closes(dxy_df, 2)
        if c is not None and c[-2] > 0:
            dxy_1d = (c[-1] - c[-2]) / c[-2] * 100

    macro_ok = (vix_val is None) or (vix_val <= 35)

    # QQQ benchmark for relative strength
    qqq_1d = None
    qqq_df = macro_data.get("QQQ")
    if qqq_df is not None:
        c = last_closes(qqq_df, 2)
        if c is not None and c[-2] > 0:
            qqq_1d = (c[-1] - c[-2]) / c[-2] * 100

    for tk in GROWTH_STOCKS:
        df = stock_data.get(tk)
        if df is None:
            continue

        c1d = last_closes(df, 2)
        c3d = last_closes(df, 4)
        c5d = last_closes(df, 6)
        vol  = last_volumes(df, 25)

        if c1d is None:
            continue

        c_now  = float(c1d[-1])
        c_prev = float(c1d[-2])
        if c_now <= 0 or c_prev <= 0:
            continue

        mom_1d = (c_now - c_prev) / c_prev * 100
        mom_3d = ((c_now - float(c3d[-4])) / float(c3d[-4]) * 100) if c3d is not None else 0
        mom_5d = ((c_now - float(c5d[-6])) / float(c5d[-6]) * 100) if c5d is not None else 0

        vol_ratio = 1.0
        if vol is not None and len(vol) >= 20:
            v5  = float(vol[-5:].mean())
            v20 = float(vol[-20:].mean())
            vol_ratio = v5 / v20 if v20 > 0 else 1.0

        score = 0

        # Momentum
        if   mom_1d > 2.0:  score += 3
        elif mom_1d > 1.0:  score += 2
        elif mom_1d > 0.3:  score += 1
        elif mom_1d < -2.0: score -= 2

        if   mom_3d > 4.0:  score += 2
        elif mom_3d > 1.5:  score += 1

        if   mom_5d > 6.0:  score += 2
        elif mom_5d > 2.5:  score += 1

        # Volume
        if   vol_ratio > 2.0: score += 2
        elif vol_ratio > 1.3: score += 1

        # Relative strength vs QQQ
        if qqq_1d is not None and mom_1d > qqq_1d:
            score += 1

        # Macro
        if hyg_1d and hyg_1d > 0.2:  score += 1
        if dxy_1d and dxy_1d > 0.5:  score -= 1
        if vix_val and vix_val > 25:  score -= 1

        signals[tk] = {
            "score":     score,
            "mom_1d":    round(mom_1d, 2),
            "mom_3d":    round(mom_3d, 2),
            "mom_5d":    round(mom_5d, 2),
            "vol_ratio": round(vol_ratio, 2),
            "macro_ok":  macro_ok,
        }

    return signals


# ── Trade Simulation ──────────────────────────────────────

def simulate_trade(df, trade_date):
    try:
        row = df.loc[trade_date]
        open_p  = float(row["Open"])
        high_p  = float(row["High"])
        low_p   = float(row["Low"])
        close_p = float(row["Close"])

        if open_p <= 0:
            return None

        stop_p   = open_p * (1 - STOP_LOSS_PCT)
        target_p = open_p * (1 + TAKE_PROFIT_PCT)

        if low_p  <= stop_p:
            exit_p = stop_p;   reason = "STOP_LOSS"
        elif high_p >= target_p:
            exit_p = target_p; reason = "TAKE_PROFIT"
        else:
            exit_p = close_p;  reason = "EOD_CLOSE"

        pnl_pct = (exit_p - open_p) / open_p * 100
        return {"open": open_p, "exit": round(exit_p,4),
                "reason": reason, "pnl_pct": round(pnl_pct,3)}
    except Exception:
        return None


# ── Backtest Engine ───────────────────────────────────────

def run_backtest(stock_data, macro_data):
    # Build sorted list of trading dates in the target window
    # Use the intersection of all stock dates in the window
    all_dfs_in_window = []
    for df in stock_data.values():
        sub = df.loc[START_DATE:END_DATE]
        all_dfs_in_window.append(set(sub.index.normalize()))

    common_dates = sorted(set.intersection(*all_dfs_in_window))

    capital   = INITIAL_CAPITAL
    trades    = []
    equity_curve = [{"date": str(common_dates[0].date()), "equity": capital}]
    max_equity   = capital
    max_drawdown = 0.0

    print(f"  Trading days in window: {len(common_dates)}")

    for i, trade_ts in enumerate(common_dates):
        trade_date = trade_ts  # Timestamp

        # Signal date = previous trading day
        if i == 0:
            equity_curve.append({"date": str(trade_date.date()), "equity": round(capital,2)})
            continue

        signal_ts   = common_dates[i - 1]
        signal_date = signal_ts

        signals = compute_daily_signals(stock_data, macro_data, signal_date)

        eligible = {
            tk: s for tk, s in signals.items()
            if s["score"] >= MIN_SIGNAL_SCORE
            and s["macro_ok"]
            and s["vol_ratio"] >= MIN_VOLUME_RATIO
        }
        ranked = sorted(eligible.items(), key=lambda x: x[1]["score"], reverse=True)
        picks  = ranked[:MAX_POSITIONS]

        for tk, sig in picks:
            df = stock_data.get(tk)
            if df is None or trade_date not in df.index:
                continue

            position_size = capital * POSITION_SIZE_PCT
            result = simulate_trade(df, trade_date)
            if result is None:
                continue

            shares      = position_size / result["open"]
            gross_pnl   = shares * result["open"] * (result["pnl_pct"] / 100)
            commission  = position_size * COMMISSION * 2
            net_pnl     = gross_pnl - commission
            capital    += net_pnl

            trades.append({
                "date":         str(trade_date.date()),
                "ticker":       tk,
                "score":        sig["score"],
                "mom_1d":       sig["mom_1d"],
                "vol_ratio":    sig["vol_ratio"],
                "open":         result["open"],
                "exit":         result["exit"],
                "exit_reason":  result["reason"],
                "pnl_pct":      result["pnl_pct"],
                "net_pnl":      round(net_pnl, 2),
                "capital_after":round(capital, 2),
            })

        if capital > max_equity:
            max_equity = capital
        dd = (max_equity - capital) / max_equity * 100
        if dd > max_drawdown:
            max_drawdown = dd

        equity_curve.append({"date": str(trade_date.date()), "equity": round(capital,2)})

    return {"trades": trades, "equity_curve": equity_curve,
            "final_capital": round(capital,2), "max_drawdown": round(max_drawdown,2)}


# ── Analytics ─────────────────────────────────────────────

def analyze(results):
    trades = results["trades"]
    if not trades:
        return {"error": "No trades executed"}

    df = pd.DataFrame(trades)
    total  = len(df)
    wins   = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]
    win_rate      = len(wins) / total * 100
    avg_win       = wins["net_pnl"].mean()   if len(wins)   > 0 else 0
    avg_loss      = losses["net_pnl"].mean() if len(losses) > 0 else 0
    pf_denom      = abs(losses["net_pnl"].sum())
    profit_factor = abs(wins["net_pnl"].sum()) / pf_denom if pf_denom > 0 else 99.0
    total_pnl     = results["final_capital"] - INITIAL_CAPITAL
    total_ret     = total_pnl / INITIAL_CAPITAL * 100

    eq = pd.DataFrame(results["equity_curve"])
    eq["ret"] = eq["equity"].pct_change()
    sharpe = (eq["ret"].mean() / eq["ret"].std() * (252**0.5)
              if eq["ret"].std() > 0 else 0)

    ticker_stats = (df.groupby("ticker")
                    .agg(trades=("net_pnl","count"),
                         total_pnl=("net_pnl","sum"),
                         win_rate=("net_pnl", lambda x: (x>0).mean()*100),
                         avg_pnl=("net_pnl","mean"))
                    .sort_values("total_pnl", ascending=False)
                    .reset_index().to_dict("records"))

    return {
        "total_trades":  total,
        "win_rate":      round(win_rate, 1),
        "avg_win":       round(avg_win, 2),
        "avg_loss":      round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "total_pnl":     round(total_pnl, 2),
        "total_return":  round(total_ret, 2),
        "final_capital": results["final_capital"],
        "max_drawdown":  results["max_drawdown"],
        "sharpe_ratio":  round(sharpe, 2),
        "exit_reasons":  df["exit_reason"].value_counts().to_dict(),
        "best_trade":    df.loc[df["net_pnl"].idxmax()][["date","ticker","pnl_pct","net_pnl"]].to_dict(),
        "worst_trade":   df.loc[df["net_pnl"].idxmin()][["date","ticker","pnl_pct","net_pnl"]].to_dict(),
        "ticker_stats":  ticker_stats,
    }


# ── Report ────────────────────────────────────────────────

def print_report(s, results):
    SEP = "=" * 62
    print(f"\n{SEP}")
    print("  BACKTEST RESULTS  --  3 MONTH DAY TRADING")
    print(f"  Portfolio: ${INITIAL_CAPITAL:,.0f}   {START_DATE} to {END_DATE}")
    print(SEP)

    ret_sign = "+" if s["total_return"] > 0 else ""
    print(f"\n  Total Return    : {ret_sign}{s['total_return']:.2f}%")
    print(f"  Final Capital   : ${s['final_capital']:,.2f}  (start: ${INITIAL_CAPITAL:,.0f})")
    print(f"  Net P&L         : ${s['total_pnl']:+,.2f}")
    print(f"\n  Total Trades    : {s['total_trades']}")
    print(f"  Win Rate        : {s['win_rate']:.1f}%")
    print(f"  Avg Win         : ${s['avg_win']:+.2f}")
    print(f"  Avg Loss        : ${s['avg_loss']:+.2f}")
    print(f"  Profit Factor   : {s['profit_factor']:.2f}x")
    print(f"\n  Max Drawdown    : -{s['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio    : {s['sharpe_ratio']:.2f}")

    print(f"\n  Exit Breakdown:")
    for reason, count in s["exit_reasons"].items():
        pct   = count / s["total_trades"] * 100
        label = {"TAKE_PROFIT":"[TP] ","STOP_LOSS":"[SL] ","EOD_CLOSE":"[EOD]"}.get(reason,"[ ]  ")
        print(f"    {label} {reason:<14}: {count:3d} ({pct:.1f}%)")

    bt = s["best_trade"];  wt = s["worst_trade"]
    print(f"\n  Best  Trade : {bt['ticker']} on {bt['date']}  {bt['pnl_pct']:+.2f}%  ${bt['net_pnl']:+.2f}")
    print(f"  Worst Trade : {wt['ticker']} on {wt['date']}  {wt['pnl_pct']:+.2f}%  ${wt['net_pnl']:+.2f}")

    print(f"\n  Per-Ticker Performance (all):")
    print(f"  {'Ticker':<8} {'Trades':>6}  {'Win%':>5}  {'Total P&L':>10}  {'Avg':>7}")
    print(f"  {'-'*46}")
    for row in s["ticker_stats"]:
        tag = "(+)" if row["total_pnl"] > 0 else "(-)"
        print(f"  {tag} {row['ticker']:<7} {int(row['trades']):>5}  "
              f"{row['win_rate']:>5.1f}%  "
              f"${row['total_pnl']:>9.2f}  "
              f"${row['avg_pnl']:>6.2f}")

    # ASCII equity curve
    eq_vals = [e["equity"] for e in results["equity_curve"]]
    pts = np.interp(np.linspace(0,len(eq_vals)-1,50), range(len(eq_vals)), eq_vals)
    lo, hi = min(pts), max(pts)
    H = 7
    print(f"\n  Equity Curve  (${INITIAL_CAPITAL:,.0f} -> ${eq_vals[-1]:,.2f})")
    for r in range(H, -1, -1):
        thresh = lo + (hi - lo) * r / H
        row_label = f"  ${lo+(hi-lo)*r/H:>8,.0f} |"
        line = "".join("#" if v >= thresh else " " for v in pts)
        print(row_label + line)
    print(f"           {' '*10}+" + "-"*50)
    print(f"           {START_DATE}{'':>30}{END_DATE}")

    print(f"\n{SEP}")
    print("  RISK ASSESSMENT")
    ok = s["total_return"] > 0
    print(f"  Return      : {'POSITIVE' if ok else 'NEGATIVE'} {s['total_return']:+.2f}%")
    print(f"  Drawdown    : {s['max_drawdown']:.2f}% {'<-- HIGH, reduce size!' if s['max_drawdown']>15 else 'OK'}")
    print(f"  Win Rate    : {s['win_rate']:.1f}% {'<-- low, check signals' if s['win_rate']<45 else 'OK'}")
    print(f"  Prof.Factor : {s['profit_factor']:.2f}x {'<-- marginal' if s['profit_factor']<1.5 else 'GOOD'}")
    print(f"  Sharpe      : {s['sharpe_ratio']:.2f} {'GOOD (>1)' if s['sharpe_ratio']>1 else 'below 1'}")
    print(f"\n  NOTE: Backtest uses daily OHLC data. Real day trading")
    print(f"  requires intraday execution. Slippage not fully modeled.")
    print(f"  Trade log -> backtest_trades.json")
    print(SEP + "\n")


# ── Main ──────────────────────────────────────────────────

def main():
    print("\n" + "="*62)
    print("  Day Trading Backtest  --  3 Months  |  $5,000 Portfolio")
    print("  Strategy: Momentum + Volume Surge + Macro Filter")
    print("="*62)

    ext_start = (datetime.strptime(START_DATE,"%Y-%m-%d") - timedelta(days=45)).strftime("%Y-%m-%d")

    print("\n[1/3] Loading stock data...")
    stock_data = load_all(GROWTH_STOCKS, ext_start, END_DATE)

    macro_tickers = ["DX-Y.NYB","^VIX","^TNX","HYG","QQQ"]
    print("[2/3] Loading macro data...")
    macro_data = load_all(macro_tickers, ext_start, END_DATE)

    print("[3/3] Running simulation...")
    results = run_backtest(stock_data, macro_data)

    stats = analyze(results)
    if "error" in stats:
        print(f"\n  No trades were executed.")
        print(f"  Try lowering MIN_SIGNAL_SCORE (currently {MIN_SIGNAL_SCORE})")
        print(f"  or MIN_VOLUME_RATIO (currently {MIN_VOLUME_RATIO})")
        return

    print_report(stats, results)

    with open("backtest_trades.json","w") as f:
        json.dump({
            "config": {
                "initial_capital": INITIAL_CAPITAL, "start": START_DATE, "end": END_DATE,
                "stop_loss_pct": STOP_LOSS_PCT, "take_profit_pct": TAKE_PROFIT_PCT,
                "max_positions": MAX_POSITIONS, "position_size_pct": POSITION_SIZE_PCT,
            },
            "summary": stats,
            "trades": results["trades"],
            "equity_curve": results["equity_curve"],
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
