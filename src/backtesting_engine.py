import pandas as pd
import numpy as np

def bactester(df : pd.DataFrame , signals : pd.Series , start_capital : float, periods : int = 6*365):
    
    returns  = df.close.pct_change().fillna(0)
    
    dt = signals.index
    
    returns = returns.loc[dt]
    
    returns = returns.fillna(0)
    
    # df contains Open  High Low close indexed with timestamp
    # signals contains signals indexed with same timestamps, each row of signal represents signal 
    # generated on previous day for that day
    
    transaction_cost = 0.0015
    
    fees = (signals.fillna(0) - signals.shift(1).fillna(0)).abs()*transaction_cost
    
    fees.iloc[-1] += abs(signals.fillna(0).iloc[-1])*transaction_cost
    
    actual_returns = signals.fillna(0) * returns - fees
    
    equity = start_capital * (1 + actual_returns).cumprod()
        
    # we need to exit if we are in trade
   
    net_profit = equity.iloc[-1] - start_capital

    std = actual_returns.std()
    sharpe = actual_returns.mean() / std * np.sqrt(periods) if std > 0 else 0.0

    downside = np.sqrt((np.minimum(actual_returns, 0) ** 2).mean())
    sortino = actual_returns.mean() / downside * np.sqrt(periods) if downside > 0 else 0.0

    max_dd = (equity / equity.cummax() - 1).min()
    n_trades = int((signals.fillna(0).diff().abs() > 0).sum())

    return {
        "fees": fees,
        "equity": equity,
        "returns": actual_returns,
        "net_profit": net_profit,
        "total_return": equity.iloc[-1] / start_capital - 1,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
    }


def compute_buy_hold_benchmark(df, periods:int = 6 * 365,start_capital=1.0, fee_rate=0.0015):
    returns = df['close'].pct_change().fillna(0)

    net = returns.copy()
    net.iloc[0] -= fee_rate
    net.iloc[-1] -= fee_rate

    equity = start_capital * (1 + net).cumprod()
    total_return = equity.iloc[-1] / start_capital - 1

    std = net.std()
    sharpe = net.mean() / std * np.sqrt(periods) if std > 0 else 0.0

    downside = np.sqrt((np.minimum(net, 0) ** 2).mean())
    sortino = net.mean() / downside * np.sqrt(periods) if downside > 0 else 0.0

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    quarterly = (1 + net).groupby(net.index.to_period('Q')).prod() - 1


    return {
        "equity": equity,
        "returns": net,
        "net_profit": equity.iloc[-1] - start_capital,
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "n_trades": 2,
        "quarterly_returns": quarterly,
    }
