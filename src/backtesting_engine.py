import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRANSACTION_COST = 0.0015


def trade_stats(signals: pd.Series, returns: pd.Series, equity: pd.Series, start_capital: float,
                periods: int, transaction_cost: float = TRANSACTION_COST):
    sig = signals.fillna(0)
    prev = sig.shift(1).fillna(0)
    side, prev_side = np.sign(sig), np.sign(prev)

    entry = (side != 0) & (side != prev_side)
    leave = (prev_side != 0) & (side != prev_side)
    trade_id = entry.cumsum().where(side != 0)

    fee_in = transaction_cost * pd.Series(
        np.where(entry, sig.abs(), np.where(side == prev_side, (sig - prev).abs(), 0.0)), index=sig.index
    )
    fee_in.iloc[-1] += abs(sig.iloc[-1]) * transaction_cost
    fee_out = transaction_cost * prev.abs() * leave

    eq_prev = equity.shift(1).fillna(start_capital)
    held = eq_prev * (sig * returns - fee_in)
    closed = -eq_prev * fee_out

    pnl = held.groupby(trade_id).sum().add(closed.groupby(trade_id.shift(1)).sum(), fill_value=0)
    days = trade_id.value_counts() * 365 / periods

    wins, losses = pnl[pnl > 0], pnl[pnl < 0]

    return {
        "gross_profit": wins.sum(),
        "gross_loss": losses.sum(),
        "total_trades": len(pnl),
        "win_rate": len(wins) / len(pnl) if len(pnl) else 0.0,
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "largest_win": wins.max() if len(wins) else 0.0,
        "largest_loss": losses.min() if len(losses) else 0.0,
        "avg_holding_days": days.mean() if len(days) else 0.0,
        "max_holding_days": days.max() if len(days) else 0.0,
    }


def bactester(df: pd.DataFrame, signals: pd.Series, start_capital: float, periods: int = 6*365,
              transaction_cost: float = TRANSACTION_COST):

    # df holds OHLC indexed by timestamp; signals are the position held DURING each bar (already lagged)

    returns = df.close.pct_change().fillna(0)

    dt = signals.index

    returns = returns.loc[dt]

    returns = returns.fillna(0)

    fees = (signals.fillna(0) - signals.shift(1).fillna(0)).abs() * transaction_cost

    fees.iloc[-1] += abs(signals.fillna(0).iloc[-1]) * transaction_cost

    actual_returns = signals.fillna(0) * returns - fees

    equity = start_capital * (1 + actual_returns).cumprod()

    net_profit = equity.iloc[-1] - start_capital

    std = actual_returns.std()
    sharpe = actual_returns.mean() / std * np.sqrt(periods) if std > 0 else 0.0

    downside = np.sqrt((np.minimum(actual_returns, 0) ** 2).mean())
    sortino = actual_returns.mean() / downside * np.sqrt(periods) if downside > 0 else 0.0

    max_dd = (equity / equity.cummax() - 1).min()
    n_trades = int((signals.fillna(0).diff().abs() > 0).sum())

    hold = pd.Series(returns.copy())
    hold.iloc[0] -= transaction_cost
    hold.iloc[-1] -= transaction_cost
    buy_hold_return =( np.prod(1+hold) - 1)

    return {
        "fees": fees,
        "equity": equity,
        "returns": actual_returns,
        "net_profit": net_profit,
        "total_return": equity.iloc[-1] / start_capital - 1,
        "buy_hold_return": buy_hold_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        **trade_stats(signals, returns, equity, start_capital, periods, transaction_cost),
    }


def compute_buy_hold_benchmark(df, periods: int = 6 * 365, start_capital=1.0, fee_rate=TRANSACTION_COST):
    result = bactester(df, pd.Series(1.0, index=df.index), start_capital, periods, fee_rate)

    net = result["returns"]
    result["n_trades"] = 2
    result["quarterly_returns"] = (1 + net).groupby(net.index.tz_localize(None).to_period('Q')).prod() - 1

    return result

