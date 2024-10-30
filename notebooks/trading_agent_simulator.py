from dataclasses import dataclass, field

from sklearn.linear_model import LinearRegression


def compute_middle_value(y_low, y_high):
    return (y_high + y_low) / 2


@dataclass
class BasicTradingAlgorithmConfiguration:
    """A basic configuration for a trading algorithm."""

    low_model: LinearRegression
    high_model: LinearRegression
    buy_shares: int = 1
    sell_shares: int = 1
    delta_buy_threshold: float = 1
    delta_sell_threshold: float = -1


def return_default_trading_history_dict() -> dict:
    default_trading_history_dict: dict = {
        "Middle Share Value": [],
        "Cash Spent": [],
        "Cash Earned": [],
        "Shares Held": [],
        "Shares Bought": [],
        "Shares Sold": [],
        "Share Value Held": [],
        "Portfolio Value": [],
        "Total Earnings": [],
        "Percent Earnings": [],
    }
    return default_trading_history_dict


@dataclass
class TraderState:
    """The current state of the trading agent."""

    cash_spent: float = 0
    cash_earned: float = 0
    shares_held: int = 0
    trading_history: dict = field(
        default_factory=return_default_trading_history_dict
    )


def execute_trading_step(
    params, state, x_test_sample, y_low_today, y_high_today
):
    y_low_future_predicted = params.low_model.predict(
        x_test_sample.reshape(1, -1)
    )
    y_high_future_predicted = params.high_model.predict(
        x_test_sample.reshape(1, -1)
    )
    y_middle_future_predicted = compute_middle_value(
        y_high_future_predicted, y_low_future_predicted
    )
    y_middle_today = compute_middle_value(y_high_today, y_low_today)
    delta = y_middle_future_predicted - y_middle_today

    shares_bought = 0
    shares_sold = 0

    if delta > params.delta_buy_threshold:
        shares_bought = params.buy_shares
        state.shares_held += shares_bought
        state.cash_spent += shares_bought * y_middle_today

    if delta < params.delta_sell_threshold:
        if state.shares_held >= params.sell_shares:
            shares_sold = params.sell_shares
            state.shares_held -= shares_sold
        else:
            shares_sold = state.shares_held
            state.shares_held = 0
        state.cash_earned += shares_sold * y_middle_today

    share_value_held = state.shares_held * y_middle_today
    portfolio_value = share_value_held + state.cash_earned
    total_earnings = portfolio_value - state.cash_spent

    eps = 1e-7
    if portfolio_value > eps:
        percent_earnings = total_earnings / state.cash_spent * 100
    else:
        percent_earnings = 0

    state.trading_history["Middle Share Value"].append(y_middle_today)
    state.trading_history["Cash Spent"].append(state.cash_spent)
    state.trading_history["Cash Earned"].append(state.cash_earned)
    state.trading_history["Shares Held"].append(state.shares_held)
    state.trading_history["Shares Bought"].append(shares_bought)
    state.trading_history["Shares Sold"].append(shares_sold)
    state.trading_history["Share Value Held"].append(share_value_held)
    state.trading_history["Portfolio Value"].append(portfolio_value)
    state.trading_history["Total Earnings"].append(total_earnings)
    state.trading_history["Percent Earnings"].append(percent_earnings)


def run_basic_regressor_agent(params, state, x_test, y_low_test, y_high_test):
    for x_test_sample, y_low_today, y_high_today in zip(
        x_test[1:], y_low_test[:-1], y_high_test[:-1]
    ):
        execute_trading_step(
            params,
            state,
            x_test_sample,
            y_low_today,
            y_high_today,
        )
