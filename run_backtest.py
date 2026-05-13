"""End-to-end backtest of the Phase 1 EMA-crossover strategy.

Pulls BTCUSDT 1h spot history (cached locally), generates signals, runs the
backtester, and prints a summary plus the trade log.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest import BacktestConfig, compute_metrics, run_backtest
from bybit_klines import load_or_fetch
from strategy import StrategyParams, generate_signals

DATA_PATH = Path(__file__).parent / "data" / "btcusdt_1h_spot.csv"


def main() -> None:
    df = load_or_fetch(
        DATA_PATH, symbol="BTCUSDT", interval="60", category="spot",
        lookback="540D", refresh=True,
    )
    print(f"Data: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    params = StrategyParams()
    config = BacktestConfig()
    print(f"Strategy: EMA({params.ema_fast})/EMA({params.ema_slow})  "
          f"RSI>{params.rsi_threshold:.0f}  "
          f"stop={params.atr_stop_mult}xATR({params.atr_length})")
    print(f"Config:   capital=${config.initial_capital:,.0f}  fee={config.fee_rate*100:.2f}%")
    print()

    signals = generate_signals(df, params)
    result = run_backtest(signals, params, config)
    m = compute_metrics(result)

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Final equity        ${m['final_equity']:>14,.2f}")
    print(f"  Total return        {m['total_return_pct']:>14.2f}%")
    print(f"  CAGR                {m['cagr_pct']:>14.2f}%")
    print(f"  Buy & hold return   {m['buy_hold_return_pct']:>14.2f}%")
    print(f"  Max drawdown        {m['max_drawdown_pct']:>14.2f}%")
    print(f"  Sharpe (annualized) {m['sharpe']:>14.2f}")
    print()
    print(f"  Trades              {int(m['num_trades']):>14d}")
    print(f"  Win rate            {m['win_rate_pct']:>14.2f}%")
    print(f"  Avg win             {m['avg_win_pct']:>14.2f}%")
    print(f"  Avg loss            {m['avg_loss_pct']:>14.2f}%")
    print(f"  Profit factor       {m['profit_factor']:>14.2f}")
    print(f"  Stop-loss exits     {int(m['stop_exits']):>14d}")
    print()

    trade_log = pd.DataFrame([
        {
            "entry":   t.entry_time.strftime("%Y-%m-%d %H:%M"),
            "exit":    t.exit_time.strftime("%Y-%m-%d %H:%M"),
            "entry_$": round(t.entry_price, 2),
            "exit_$":  round(t.exit_price, 2),
            "ret_%":   round(t.return_pct * 100, 2),
            "pnl_$":   round(t.pnl_usd, 2),
            "bars":    t.bars_held,
            "reason":  t.exit_reason,
        }
        for t in result.trades
    ])
    if not trade_log.empty:
        print("LAST 15 TRADES")
        print("-" * 60)
        print(trade_log.tail(15).to_string(index=False))

    out_path = Path(__file__).parent / "data" / "backtest_trades.csv"
    if not trade_log.empty:
        trade_log.to_csv(out_path, index=False)
        print(f"\nFull trade log -> {out_path}")


if __name__ == "__main__":
    main()
