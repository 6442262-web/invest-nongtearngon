# -*- coding: utf-8 -*-
"""
Swing Trade Backtest -- 3 Months
Portfolio: $5,000
Strategy : Momentum + Macro Filter
Hold     : Until SL / TP / Signal Reversal / Max Hold Days
Entry    : Buy @ Next-Day Open after signal fires
Exit     : Trailing stop or signal exit (sell @ Open of exit day)
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

MAX_POSITIONS     = 3          # hold up to 3 stocks at a time
POSITION_SIZE_PCT = 0.30       # 30% portfolio per position
STOP_LOSS_PCT     = 0.07       # -7% from entry (swing-style)
TAKE_PROFIT_PCT   = 0.15       # +15% take profit
TRAILING_STOP_PCT = 0.05       # 5% trailing stop from peak
MAX_HOLD_DAYS     = 20         # force exit after 20 trading days
COMMISSION        = 0.001      # 0.1% per side

MIN_SIGNAL_SCORE  = 4
MIN_VOLUME_RATIO  = 1.15

GROWTH_STOCKS = [
    "NVDA","MSFT","AAPL","META","GOOGL",
    "AMZN","TSLA","AVGO","NOW",
    "ADBE","CRM","PANW","DDOG","NET","ZS",
]

MACRO_TICKERS = ["DX-Y.NYB","^VIX","^TNX","HYG","QQQ","XLK"]

# ── Data Loading ──────────────────────────────────────────

def load_all(tickers, start, end):
    print(f"  Downloading {len(tickers)} tickers ({start} to {end})...")
    raw = yf.download(tickers, start=start, end=end,
                      interval="1d", progress=False, auto_adjust=True,
                      group_by="ticker")
    result = {}
    for tk in tickers:
        try:
            df = raw[tk].copy() if len(tickers) > 1 else raw.copy()
            df = df.dropna(subset=["Close","Open","High","Low"])
            if len(df) > 15:
                result[tk] = df
        except Exception:
            pass
    print(f"  Loaded: {len(result)}/{len(tickers)} OK")
    return result


# ── Signal Engine ─────────────────────────────────────────

def get_signal_score(tk, stock_data, macro_data, signal_date):
    df = stock_data.get(tk)
    if df is None:
        return None

    def last_closes(d, n):
        sub = d.loc[:signal_date, "Close"].dropna()
        return sub.values[-n:] if len(sub) >= n else None

    def last_vols(d, n):
        sub = d.loc[:signal_date, "Volume"].dropna()
        return sub.values[-n:] if len(sub) >= n else None

    c2  = last_closes(df, 2)
    c6  = last_closes(df, 6)
    c11 = last_closes(df, 11)
    c21 = last_closes(df, 21)
    vol = last_vols(df, 25)

    if c2 is None:
        return None

    c_now, c_prev = float(c2[-1]), float(c2[-2])
    if c_now <= 0 or c_prev <= 0:
        return None

    mom_1d = (c_now - c_prev) / c_prev * 100
    mom_5d = ((c_now - float(c6[-6])) / float(c6[-6]) * 100) if c6 is not None else 0
    mom_10d= ((c_now - float(c11[-11]))/ float(c11[-11])* 100) if c11 is not None else 0
    mom_20d= ((c_now - float(c21[-21]))/ float(c21[-21])* 100) if c21 is not None else 0

    vol_ratio = 1.0
    if vol is not None and len(vol) >= 20:
        vol_ratio = float(vol[-5:].mean()) / float(vol[-20:].mean())

    # Trend confirmation: price above 10d and 20d MA
    above_10ma = (c_now > float(c11.mean())) if c11 is not None else False
    above_20ma = (c_now > float(c21.mean())) if c21 is not None else False

    # Macro
    vix_val = None
    hyg_1d  = None
    dxy_1d  = None
    qqq_5d  = None
    xlk_5d  = None

    vix_df = macro_data.get("^VIX")
    if vix_df is not None:
        c = last_closes(vix_df, 2)
        if c is not None: vix_val = float(c[-1])

    hyg_df = macro_data.get("HYG")
    if hyg_df is not None:
        c = last_closes(hyg_df, 2)
        if c is not None and c[-2] > 0:
            hyg_1d = (c[-1]-c[-2])/c[-2]*100

    dxy_df = macro_data.get("DX-Y.NYB")
    if dxy_df is not None:
        c = last_closes(dxy_df, 2)
        if c is not None and c[-2] > 0:
            dxy_1d = (c[-1]-c[-2])/c[-2]*100

    qqq_df = macro_data.get("QQQ")
    if qqq_df is not None:
        c = last_closes(qqq_df, 6)
        if c is not None and c[-6] > 0:
            qqq_5d = (c[-1]-c[-6])/c[-6]*100

    xlk_df = macro_data.get("XLK")
    if xlk_df is not None:
        c = last_closes(xlk_df, 6)
        if c is not None and c[-6] > 0:
            xlk_5d = (c[-1]-c[-6])/c[-6]*100

    macro_ok = True
    if vix_val and vix_val > 30:
        macro_ok = False
    if qqq_5d and qqq_5d < -5:
        macro_ok = False  # market in downtrend

    score = 0

    # 1-day momentum
    if   mom_1d > 3.0:  score += 3
    elif mom_1d > 1.5:  score += 2
    elif mom_1d > 0.5:  score += 1
    elif mom_1d < -1.5: score -= 2

    # 5-day momentum (trend)
    if   mom_5d > 6.0:  score += 3
    elif mom_5d > 3.0:  score += 2
    elif mom_5d > 1.0:  score += 1
    elif mom_5d < -4.0: score -= 2

    # 10-day momentum
    if   mom_10d > 10: score += 2
    elif mom_10d > 5:  score += 1

    # 20-day trend
    if   mom_20d > 15: score += 2
    elif mom_20d > 8:  score += 1
    elif mom_20d < -5: score -= 1

    # Trend confirmation (above MAs)
    if above_10ma: score += 1
    if above_20ma: score += 1

    # Volume surge
    if   vol_ratio > 2.0: score += 2
    elif vol_ratio > 1.3: score += 1

    # Relative strength vs QQQ
    if qqq_5d and mom_5d > qqq_5d: score += 1

    # Tech sector leadership
    if xlk_5d and xlk_5d > 0: score += 1

    # Macro adjustments
    if hyg_1d and hyg_1d > 0.2:  score += 1  # risk-on
    if dxy_1d and dxy_1d > 0.8:  score -= 1  # strong USD
    if vix_val and vix_val > 22:  score -= 1  # elevated fear

    return {
        "score":     score,
        "mom_1d":    round(mom_1d, 2),
        "mom_5d":    round(mom_5d, 2),
        "mom_10d":   round(mom_10d, 2),
        "mom_20d":   round(mom_20d, 2),
        "vol_ratio": round(vol_ratio, 2),
        "macro_ok":  macro_ok,
        "vix":       round(vix_val, 1) if vix_val else None,
        "price":     round(c_now, 2),
    }


def should_exit_signal(tk, stock_data, macro_data, signal_date, entry_price, peak_price):
    """Check if we should exit an existing position based on signal reversal."""
    sig = get_signal_score(tk, stock_data, macro_data, signal_date)
    if sig is None:
        return False, "NO_DATA"

    # Hard stop loss
    current_price = sig["price"]
    if current_price <= entry_price * (1 - STOP_LOSS_PCT):
        return True, "STOP_LOSS"

    # Take profit
    if current_price >= entry_price * (1 + TAKE_PROFIT_PCT):
        return True, "TAKE_PROFIT"

    # Trailing stop from peak
    if peak_price > 0 and current_price <= peak_price * (1 - TRAILING_STOP_PCT):
        return True, "TRAILING_STOP"

    # Momentum reversal: 5d goes deeply negative
    if sig["mom_5d"] < -5 and sig["score"] < 0:
        return True, "SIGNAL_REVERSAL"

    # Macro deterioration
    if sig["vix"] and sig["vix"] > 32:
        return True, "MACRO_EXIT"

    return False, None


# ── Backtest Engine ───────────────────────────────────────

def run_backtest(stock_data, macro_data):
    # Common dates in target window
    date_sets = []
    for df in stock_data.values():
        sub = df.loc[START_DATE:END_DATE]
        date_sets.append(set(sub.index.normalize()))
    common_dates = sorted(set.intersection(*date_sets))

    capital      = INITIAL_CAPITAL
    positions    = {}   # {ticker: {entry_price, entry_date, shares, peak_price, hold_days}}
    trades       = []
    equity_curve = []
    max_equity   = capital
    max_drawdown = 0.0

    print(f"  Trading days: {len(common_dates)}")
    print(f"  Strategy: Swing Trade | SL={STOP_LOSS_PCT*100:.0f}% | TP={TAKE_PROFIT_PCT*100:.0f}% | Trail={TRAILING_STOP_PCT*100:.0f}% | Max Hold={MAX_HOLD_DAYS}d\n")

    for i, trade_ts in enumerate(common_dates):
        trade_date  = trade_ts
        signal_date = common_dates[i - 1] if i > 0 else trade_ts

        # ── Compute portfolio value (mark-to-market open positions)
        portfolio_value = capital
        for tk, pos in positions.items():
            df = stock_data.get(tk)
            if df is not None and trade_date in df.index:
                current_price = float(df.loc[trade_date, "Close"])
                portfolio_value += pos["shares"] * current_price

        equity_curve.append({"date": str(trade_date.date()), "equity": round(portfolio_value, 2)})

        if portfolio_value > max_equity:
            max_equity = portfolio_value
        dd = (max_equity - portfolio_value) / max_equity * 100
        if dd > max_drawdown:
            max_drawdown = dd

        if i == 0:
            continue

        # ── Check exits for existing positions
        to_exit = []
        for tk, pos in positions.items():
            df = stock_data.get(tk)
            if df is None or trade_date not in df.index:
                continue

            current_price = float(df.loc[trade_date, "Close"])
            open_price    = float(df.loc[trade_date, "Open"])

            # Update peak
            if current_price > pos["peak_price"]:
                pos["peak_price"] = current_price

            should_exit, reason = should_exit_signal(
                tk, stock_data, macro_data, signal_date,
                pos["entry_price"], pos["peak_price"]
            )

            hold_days = pos["hold_days"]
            if hold_days >= MAX_HOLD_DAYS:
                should_exit, reason = True, "MAX_HOLD"

            if should_exit:
                # Exit at open price of current day
                exit_price = open_price
                if reason == "STOP_LOSS":
                    exit_price = min(open_price, pos["entry_price"] * (1 - STOP_LOSS_PCT))
                elif reason == "TAKE_PROFIT":
                    exit_price = max(open_price, pos["entry_price"] * (1 + TAKE_PROFIT_PCT))

                gross_pnl  = pos["shares"] * (exit_price - pos["entry_price"])
                commission = pos["shares"] * exit_price * COMMISSION
                net_pnl    = gross_pnl - commission - pos["entry_commission"]
                capital   += pos["shares"] * exit_price - commission

                hold_actual = (trade_date - pos["entry_date"]).days

                trades.append({
                    "date_entry":   str(pos["entry_date"].date()),
                    "date_exit":    str(trade_date.date()),
                    "ticker":       tk,
                    "entry_price":  round(pos["entry_price"], 2),
                    "exit_price":   round(exit_price, 2),
                    "shares":       round(pos["shares"], 4),
                    "hold_days":    hold_actual,
                    "exit_reason":  reason,
                    "pnl_pct":      round((exit_price - pos["entry_price"]) / pos["entry_price"] * 100, 2),
                    "net_pnl":      round(net_pnl, 2),
                    "signal_score": pos["signal_score"],
                    "capital_after":round(capital, 2),
                })
                to_exit.append(tk)

            pos["hold_days"] += 1

        for tk in to_exit:
            del positions[tk]

        # ── Scan for new entries
        if len(positions) < MAX_POSITIONS and i > 5:
            candidates = {}
            for tk in GROWTH_STOCKS:
                if tk in positions:
                    continue
                sig = get_signal_score(tk, stock_data, macro_data, signal_date)
                if sig is None:
                    continue
                if (sig["score"] >= MIN_SIGNAL_SCORE
                        and sig["macro_ok"]
                        and sig["vol_ratio"] >= MIN_VOLUME_RATIO):
                    candidates[tk] = sig

            ranked = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)
            slots  = MAX_POSITIONS - len(positions)

            for tk, sig in ranked[:slots]:
                df = stock_data.get(tk)
                if df is None or trade_date not in df.index:
                    continue

                entry_price   = float(df.loc[trade_date, "Open"])
                if entry_price <= 0:
                    continue

                position_value = min(capital * POSITION_SIZE_PCT,
                                     capital - 200)  # keep min $200 cash buffer
                if position_value < 100:
                    continue

                shares         = position_value / entry_price
                entry_comm     = shares * entry_price * COMMISSION
                capital       -= (shares * entry_price + entry_comm)

                positions[tk] = {
                    "entry_price":      entry_price,
                    "entry_date":       trade_date,
                    "shares":           shares,
                    "peak_price":       entry_price,
                    "hold_days":        0,
                    "signal_score":     sig["score"],
                    "entry_commission": entry_comm,
                }

    # Force close remaining positions at last price
    last_date = common_dates[-1]
    for tk, pos in positions.items():
        df = stock_data.get(tk)
        if df is None:
            continue
        last_price = float(df.loc[last_date, "Close"]) if last_date in df.index else pos["entry_price"]
        gross_pnl  = pos["shares"] * (last_price - pos["entry_price"])
        commission = pos["shares"] * last_price * COMMISSION
        net_pnl    = gross_pnl - commission - pos["entry_commission"]
        capital   += pos["shares"] * last_price - commission

        hold_actual = (last_date - pos["entry_date"]).days
        trades.append({
            "date_entry":   str(pos["entry_date"].date()),
            "date_exit":    str(last_date.date()),
            "ticker":       tk,
            "entry_price":  round(pos["entry_price"], 2),
            "exit_price":   round(last_price, 2),
            "shares":       round(pos["shares"], 4),
            "hold_days":    hold_actual,
            "exit_reason":  "END_OF_PERIOD",
            "pnl_pct":      round((last_price - pos["entry_price"]) / pos["entry_price"] * 100, 2),
            "net_pnl":      round(net_pnl, 2),
            "signal_score": pos["signal_score"],
            "capital_after":round(capital, 2),
        })

    return {
        "trades":        trades,
        "equity_curve":  equity_curve,
        "final_capital": round(capital, 2),
        "max_drawdown":  round(max_drawdown, 2),
    }


# ── Analytics ─────────────────────────────────────────────

def analyze(results):
    trades = results["trades"]
    if not trades:
        return {"error": "No trades"}

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
    avg_hold      = df["hold_days"].mean()

    eq = pd.DataFrame(results["equity_curve"])
    eq["ret"] = eq["equity"].pct_change()
    sharpe = (eq["ret"].mean() / eq["ret"].std() * (252**0.5)
              if eq["ret"].std() > 0 else 0)

    ticker_stats = (df.groupby("ticker")
                    .agg(trades=("net_pnl","count"),
                         total_pnl=("net_pnl","sum"),
                         win_rate=("net_pnl", lambda x: (x>0).mean()*100),
                         avg_pnl=("net_pnl","mean"),
                         avg_hold=("hold_days","mean"))
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
        "avg_hold_days": round(avg_hold, 1),
        "exit_reasons":  df["exit_reason"].value_counts().to_dict(),
        "best_trade":    df.loc[df["net_pnl"].idxmax()][["date_entry","date_exit","ticker","pnl_pct","net_pnl","hold_days"]].to_dict(),
        "worst_trade":   df.loc[df["net_pnl"].idxmin()][["date_entry","date_exit","ticker","pnl_pct","net_pnl","hold_days"]].to_dict(),
        "ticker_stats":  ticker_stats,
    }


# ── Report ────────────────────────────────────────────────

def print_report(s, results):
    SEP = "=" * 64
    print(f"\n{SEP}")
    print("  SWING TRADE BACKTEST  --  3 MONTHS  |  $5,000 PORTFOLIO")
    print(f"  {START_DATE} to {END_DATE}")
    print(f"  SL={STOP_LOSS_PCT*100:.0f}%  TP={TAKE_PROFIT_PCT*100:.0f}%  Trail={TRAILING_STOP_PCT*100:.0f}%  MaxHold={MAX_HOLD_DAYS}d  Pos={int(POSITION_SIZE_PCT*100)}%")
    print(SEP)

    ret_tag = "PROFIT" if s["total_return"] > 0 else "LOSS"
    print(f"\n  [{ret_tag}] Total Return    : {s['total_return']:+.2f}%")
    print(f"  Final Capital   : ${s['final_capital']:,.2f}   (start: ${INITIAL_CAPITAL:,.0f})")
    print(f"  Net P&L         : ${s['total_pnl']:+,.2f}")
    print(f"\n  Total Trades    : {s['total_trades']}")
    print(f"  Win Rate        : {s['win_rate']:.1f}%")
    print(f"  Avg Hold        : {s['avg_hold_days']:.1f} days")
    print(f"  Avg Win         : ${s['avg_win']:+.2f}")
    print(f"  Avg Loss        : ${s['avg_loss']:+.2f}")
    print(f"  Profit Factor   : {s['profit_factor']:.2f}x")
    print(f"\n  Max Drawdown    : -{s['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio    : {s['sharpe_ratio']:.2f}")

    print(f"\n  Exit Breakdown:")
    labels = {"TAKE_PROFIT":"[TP]  ","STOP_LOSS":"[SL]  ",
              "TRAILING_STOP":"[TSL] ","SIGNAL_REVERSAL":"[REV] ",
              "MACRO_EXIT":"[MAC] ","MAX_HOLD":"[MAX] ","END_OF_PERIOD":"[EOP] "}
    for reason, count in s["exit_reasons"].items():
        pct = count / s["total_trades"] * 100
        lbl = labels.get(reason, "[---] ")
        print(f"    {lbl} {reason:<18}: {count:3d} ({pct:.1f}%)")

    bt = s["best_trade"]; wt = s["worst_trade"]
    print(f"\n  Best  : {bt['ticker']:<5} {bt['date_entry']}->{bt['date_exit']} "
          f"({bt['hold_days']}d)  {bt['pnl_pct']:+.2f}%  ${bt['net_pnl']:+.2f}")
    print(f"  Worst : {wt['ticker']:<5} {wt['date_entry']}->{wt['date_exit']} "
          f"({wt['hold_days']}d)  {wt['pnl_pct']:+.2f}%  ${wt['net_pnl']:+.2f}")

    print(f"\n  Per-Ticker Performance:")
    print(f"  {'':3} {'Ticker':<7} {'#':>4}  {'Win%':>5}  {'AvgHold':>7}  {'Total':>9}  {'Avg':>7}")
    print(f"  {'-'*52}")
    for row in s["ticker_stats"]:
        tag = "(+)" if row["total_pnl"] > 0 else "(-)"
        print(f"  {tag} {row['ticker']:<7} {int(row['trades']):>3}   "
              f"{row['win_rate']:>5.1f}%  "
              f"{row['avg_hold']:>6.1f}d  "
              f"${row['total_pnl']:>8.2f}  "
              f"${row['avg_pnl']:>6.2f}")

    # Equity curve
    eq_vals = [e["equity"] for e in results["equity_curve"]]
    eq_dates= [e["date"]   for e in results["equity_curve"]]
    pts  = np.interp(np.linspace(0,len(eq_vals)-1,55), range(len(eq_vals)), eq_vals)
    lo, hi = min(pts), max(pts)
    H = 8
    print(f"\n  Equity Curve  ($5,000 -> ${eq_vals[-1]:,.2f})")
    for r in range(H, -1, -1):
        thresh = lo + (hi - lo) * r / H
        label  = f"  ${lo+(hi-lo)*r/H:>7,.0f} |"
        line   = "".join("#" if v >= thresh else "." for v in pts)
        print(label + line)
    print(f"           {' '*8}+" + "-"*55)
    print(f"           {eq_dates[0]}{'':>37}{eq_dates[-1]}")

    print(f"\n{SEP}")
    print(f"  STRATEGY SUMMARY")
    ok = s["total_return"] > 0
    print(f"  Return      : {'POSITIVE' if ok else 'NEGATIVE'} {s['total_return']:+.2f}%")
    print(f"  vs Day Trade: Day trade was -6.41% | Swing is {s['total_return']:+.2f}%")
    print(f"  Drawdown    : {s['max_drawdown']:.2f}%  {'<< HIGH' if s['max_drawdown']>20 else 'manageable'}")
    print(f"  Profit Fac  : {s['profit_factor']:.2f}x  {'GOOD' if s['profit_factor']>1.5 else 'needs improvement'}")
    print(f"  Sharpe      : {s['sharpe_ratio']:.2f}  {'GOOD' if s['sharpe_ratio']>1 else 'moderate'}")
    print(f"\n  NOTE: Swing trade holds avg {s['avg_hold_days']:.0f} days -- much less noise")
    print(f"  than day trading. Trailing stop protects profits on big winners.")
    print(f"  Full log -> backtest_swing_trades.json")
    print(SEP + "\n")


# ── Main ──────────────────────────────────────────────────

def main():
    print("\n" + "="*64)
    print("  Swing Trade Backtest  --  3 Months  |  $5,000 Portfolio")
    print("  Hold: multi-day until SL / TP / Trail / Signal Reversal")
    print("="*64)

    ext_start = (datetime.strptime(START_DATE,"%Y-%m-%d") - timedelta(days=50)).strftime("%Y-%m-%d")

    print("\n[1/3] Loading stock data...")
    stock_data = load_all(GROWTH_STOCKS, ext_start, END_DATE)

    print("[2/3] Loading macro data...")
    macro_data = load_all(MACRO_TICKERS, ext_start, END_DATE)

    print("[3/3] Running simulation...")
    results = run_backtest(stock_data, macro_data)

    stats = analyze(results)
    if "error" in stats:
        print(f"  No trades. Lower MIN_SIGNAL_SCORE (now {MIN_SIGNAL_SCORE})")
        return

    print_report(stats, results)

    with open("backtest_swing_trades.json","w") as f:
        json.dump({
            "config": {
                "initial_capital": INITIAL_CAPITAL, "start": START_DATE, "end": END_DATE,
                "stop_loss_pct": STOP_LOSS_PCT, "take_profit_pct": TAKE_PROFIT_PCT,
                "trailing_stop_pct": TRAILING_STOP_PCT, "max_hold_days": MAX_HOLD_DAYS,
                "max_positions": MAX_POSITIONS, "position_size_pct": POSITION_SIZE_PCT,
            },
            "summary": stats,
            "trades": results["trades"],
            "equity_curve": results["equity_curve"],
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
