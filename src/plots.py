import matplotlib.pyplot as plt
import pandas as pd


def plot_price_volume(btc, eth):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    # 1. BTC Price
    axes[0, 0].plot(btc.index, btc['close'], color='#F7931A', linewidth=1.5, label='BTC Price')
    axes[0, 0].set_title('BTC-USD close Price (USD)')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend(loc='upper left')

    # 2. ETH Price
    axes[0, 1].plot(eth.index, eth['close'], color='#627EEA', linewidth=1.5, label='ETH Price')
    axes[0, 1].set_title('ETH-USD close Price (USD)')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].legend(loc='upper left')

    # 3. BTC volume
    axes[1, 0].bar(btc.index, btc['volume'], color='#F7931A', alpha=0.6, label='BTC volume')
    axes[1, 0].set_title('BTC-USD Trading volume')
    axes[1, 0].set_ylabel('volume')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 0].legend(loc='upper left')

    # 4. ETH volume
    axes[1, 1].bar(eth.index, eth['volume'], color='#627EEA', alpha=0.6, label='ETH volume')
    axes[1, 1].set_title('ETH-USD Trading volume')
    axes[1, 1].set_ylabel('volume')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_regimes(btc, eth):
    # Define market regime date ranges
    regimes = [
        {"label": "2021 Bull Run", "start": "2021-01-01", "end": "2021-11-30", "color": "green", "alpha": 0.12},
        {"label": "2022 Crash", "start": "2021-12-01", "end": "2022-12-31", "color": "red", "alpha": 0.12},
        {"label": "2023–2025 Recovery", "start": "2023-01-01", "end": "2025-10-31", "color": "blue", "alpha": 0.08}
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(btc.index, btc['close'], color='#F7931A', linewidth=1.5, label='BTC close Price')
    ax1.set_title('Bitcoin (BTC-USD): Market Regimes (2021 - 2025)')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2.plot(eth.index, eth['close'], color='#627EEA', linewidth=1.5, label='ETH close Price')
    ax2.set_title('Ethereum (ETH-USD): Market Regimes (2021 - 2025)')
    ax2.set_ylabel('Price ($)')
    ax2.grid(True, linestyle='--', alpha=0.4)

    for ax in [ax1, ax2]:
        for regime in regimes:
            ax.axvspan(
                xmin=pd.to_datetime(regime['start']),
                xmax=pd.to_datetime(regime['end']),
                color=regime['color'],
                alpha=regime['alpha'],
                label=regime['label']
            )

    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))
    ax1.legend(by_label1.values(), by_label1.keys(), loc='upper left')

    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    ax2.legend(by_label2.values(), by_label2.keys(), loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_equity(results: dict, strategies: list, assets, title: str = "{} equity curve", figsize=(10, 4)):
    for name in assets:
        fig, ax = plt.subplots(figsize=figsize)
        for strategy in strategies:
            ax.plot(results[(strategy, name)]["equity"], label=strategy)
        ax.set_title(title.format(name))
        ax.grid(True)
        ax.legend()
        plt.show()