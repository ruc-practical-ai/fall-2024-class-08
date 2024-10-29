import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D

_CANDLE_STICK_FIGURE_WIDTH = 14
_CANDLE_STICK_FIGURE_HEIGHT = 6


def plot_candlesticks(
    dates: np.ndarray,
    predicted_low: np.ndarray,
    predicted_high: np.ndarray,
    actual_low: np.ndarray,
    actual_high: np.ndarray,
) -> None:
    plt.figure(
        figsize=(_CANDLE_STICK_FIGURE_WIDTH, _CANDLE_STICK_FIGURE_HEIGHT)
    )
    for idx in range(len(dates)):
        plt.vlines(
            dates[idx],
            predicted_low[idx],
            predicted_high[idx],
            color="orange",
            linewidth=1,
            transform=Affine2D().translate(-0.1, 0) + plt.gca().transData,
        )
    for idx in range(len(dates)):
        plt.vlines(
            dates[idx],
            actual_low[idx],
            actual_high[idx],
            color="blue",
            linewidth=1,
            transform=Affine2D().translate(0.1, 0) + plt.gca().transData,
        )
    plt.title("Predicted (Orange) vs Actual (Blue) Low / High Prices")
    plt.grid(False)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=70, fontsize=8)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.show()
