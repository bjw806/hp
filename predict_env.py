import datetime
import glob
import os
import warnings
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import pandas as pd
from gym_trading_env.utils.history import History
from gym_trading_env.utils.portfolio import TargetPortfolio
from gymnasium import spaces

warnings.filterwarnings("error")


def basic_reward_function(history: History):
    roe = (
        history["portfolio_valuation", -1] / (history["entry_valuation", -1] + 1e-8) - 1
    )
    total_roe = (
        history["portfolio_valuation", -1] / history["portfolio_valuation", 0] - 1
    )
    reward = (
        (roe**2)
        if roe > 0
        else -(roe**2) + ((total_roe**2) if total_roe > 0 else -(total_roe**2)) * 0.1
    )
    return reward


def dynamic_feature_last_position_taken(history: History):
    return history["position", -1]


def dynamic_feature_real_position(history: History):
    return history["real_position", -1]


class DiscretedTradingEnv(gym.Env):
    metadata = {"render_modes": ["logs"]}

    def __init__(
        self,
        df: pd.DataFrame,
        dynamic_feature_functions: list = [
            # dynamic_feature_last_position_taken,
            # dynamic_feature_real_position,
        ],
        reward_function: Callable = basic_reward_function,
        leverage: int = 5,
        take_profit: float = 0.1,
        stop_loss: float = -0.1,
        window_size: int = None,
        trading_fees: float = 0.0001,
        borrow_interest_rate: float = 0.00003,
        portfolio_initial_value: int = 1000,
        max_episode_duration: str = "max",
        verbose: int = 1,
        name: str = "Stock",
        render_mode: str = "logs",
        force_clear: int = 0,
    ):
        # initial
        self.name = name
        self.verbose = verbose
        self.render_mode = render_mode
        self.log_metrics = []

        # trading
        self.positions = [-1, 0, 1]
        self.trading_fees = trading_fees
        self.borrow_interest_rate = borrow_interest_rate
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.force_clear = force_clear
        self.pc_counter = 0
        self.liquidation = False
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.leverage = leverage

        # env
        self.max_episode_duration = max_episode_duration
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function = reward_function
        self.window_size = window_size
        self._set_df(df)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=[4])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=[self.window_size, self._nb_features]
            if window_size
            else [self._nb_features],
            dtype=np.float32,
        )

    def _set_df(self, df):
        df = df.copy()
        self._features_columns = [col for col in df.columns if "feature" in col]
        self._info_columns = list(
            set(list(df.columns) + ["close"]) - set(self._features_columns)
        )
        self._nb_features = len(self._features_columns)
        self._nb_static_features = self._nb_features

        for i in range(len(self.dynamic_feature_functions)):
            df[f"dynamic_feature__{i}"] = 0
            self._features_columns.append(f"dynamic_feature__{i}")
            self._nb_features += 1

        self.df = df
        self._obs_array = np.array(self.df[self._features_columns], dtype=np.float32)
        self._info_array = np.array(self.df[self._info_columns])
        self._price_array = np.array(self.df["close"])

    def _get_ticker(self, delta=0):
        return self.df.iloc[self._idx + delta]

    def _get_price(self, delta=0):
        return self._price_array[self._idx + delta]

    def _get_position_roe(self, history):
        prev_position = history[-2]["position"]
        curr_position = self.historical_info["position", -1]
        index = 1
        index_limit = len(history)

        while index < index_limit and history["position", -index] == prev_position:
            index += 1

        if prev_position == curr_position:
            return 0
        else:
            return (
                history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
                - 1
            )

    def _get_obs(self):
        for i, dynamic_feature_function in enumerate(self.dynamic_feature_functions):
            self._obs_array[self._idx, self._nb_static_features + i] = (
                dynamic_feature_function(self.historical_info)
            )

        _step_index = (
            self._idx
            if self.window_size is None
            else np.arange(self._idx + 1 - self.window_size, self._idx + 1)
        )

        observation = self._obs_array[_step_index]
        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.pc_counter = 0
        self._step = 0
        self._position = 0
        self._limit_orders = {}

        self._idx = 0

        if self.window_size is not None:
            self._idx = self.window_size - 1
        if self.max_episode_duration != "max":
            self._idx = np.random.randint(
                low=self._idx, high=len(self.df) - self.max_episode_duration - self._idx
            )

        self._portfolio = TargetPortfolio(
            position=self._position,
            value=self.portfolio_initial_value,
            price=self._get_price(),
        )

        self.historical_info = History(max_size=len(self.df))
        self.historical_info.set(
            idx=self._idx,
            step=self._step,
            date=self.df.index.values[self._idx],
            # position_index=self.positions.index(self._position),
            position=self._position,
            real_position=self._position,
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation=self.portfolio_initial_value,
            # portfolio_distribution=self._portfolio.get_portfolio_distribution(),
            reward=0,
            #
            entry_price=self._get_price(),
            entry_valuation=self.portfolio_initial_value,
            PNL=0,
            ROE=0,
            pc_counter=0,
        )

        return self._get_obs(), self.historical_info[0]

    def render(self):
        pass

    def _trade(self, position, price=None):
        self._portfolio.trade_to_position(
            position,
            price=self._get_price() if price is None else price,
            trading_fees=self.trading_fees,
        )
        self._position = position
        return

    def _take_action(self, position):
        if position != self._position:
            self._trade(position)
            self.pc_counter += 1

    def _take_action_order_limit(self):
        if len(self._limit_orders) > 0:
            ticker = self._get_ticker()
            for position, params in self._limit_orders.items():
                if (
                    position != self._position
                    and params["limit"] <= ticker["high"]
                    and params["limit"] >= ticker["low"]
                ):
                    self._trade(position, price=params["limit"])
                    if not params["persistent"]:
                        del self._limit_orders[position]

    def add_limit_order(self, position, limit, persistent=False):
        self._limit_orders[position] = {"limit": limit, "persistent": persistent}

    def step(self, ohlc=None):
        lr_current_candle = np.array(
            self.df[
                [
                    "feature_open_lr",
                    "feature_high_lr",
                    "feature_low_lr",
                    "feature_log_returns",
                ]
            ],
            dtype=np.float32,
        )[self._idx + 1]

        raw_obs_candle = np.array(
            self.df[
                [
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            ],
            dtype=np.float32,
        )[self._idx]
        raw_predict_candle = np.exp(ohlc) * raw_obs_candle[3]

        _open = (ohlc[0] - lr_current_candle[0]) * 1
        _high = (ohlc[1] - lr_current_candle[1]) * 3
        _low = (ohlc[2] - lr_current_candle[2]) * 3
        _close = (ohlc[3] - lr_current_candle[3]) * 2
        _ohlc = np.array([_open, _high, _low, _close])

        # pos = self.positions[0]

        if (
            (ohlc[0] > ohlc[1] or ohlc[0] < ohlc[2])
            or (ohlc[3] > ohlc[1] or ohlc[3] < ohlc[2])
            or ohlc[1] < ohlc[2]
        ):
            pos = 0
            # print("Invalid prediction")
        else:
            long_expected_roe = (
                abs(raw_predict_candle[1] - raw_obs_candle[3])
                / raw_obs_candle[3]
                * self.leverage
            )
            short_expected_roe = (
                abs(raw_predict_candle[2] - raw_obs_candle[3])
                / raw_obs_candle[3]
                * self.leverage
            )     
            expected_roe = (
                abs(raw_predict_candle[0] - raw_obs_candle[3])
                / raw_obs_candle[3]
                * self.leverage
            )

            print(max(raw_predict_candle / raw_obs_candle - 1) * 100)

            # print(long_expected_roe, short_expected_roe, expected_roe)

            # if self._position != 0:
            #     if self.historical_info["ROE", -1] > self.take_profit:  # TF
            #         pos = 0
            #     elif self.historical_info["ROE", -1] < self.stop_loss:  # SL
            #         pos = 0
            #     else:
            #         pos = -1 if short_expected_roe > long_expected_roe else 1
            #         pos *= (
            #             self.leverage
            #             if max(long_expected_roe, short_expected_roe) > self.take_profit
            #             else self._position
            #         )
            # else:
            #     pos = -1 if short_expected_roe > long_expected_roe else 1
            #     pos *= (
            #         self.leverage
            #         if max(long_expected_roe, short_expected_roe) > self.take_profit
            #         else 0
            #     )

            if self._position != 0:
                if self.historical_info["ROE", -1] > self.take_profit:  # TF
                    pos = 0
                elif self.historical_info["ROE", -1] < self.stop_loss:  # SL
                    pos = 0
                else:
                    if (
                        raw_predict_candle[0] < raw_obs_candle[3]
                        and long_expected_roe > short_expected_roe
                        # and long_expected_roe > self.take_profit
                        # and expected_roe > self.take_profit
                        # and short_expected_roe < self.stop_loss
                    ):
                        pos = 1  # long
                    elif (
                        raw_predict_candle[0] > raw_obs_candle[3]
                        and long_expected_roe < short_expected_roe
                        # and short_expected_roe > self.take_profit
                        # and expected_roe > self.take_profit
                        # and long_expected_roe < self.stop_loss
                    ):
                        pos = -1  # short
                    else:
                        pos = self._position  # hold
            else:
                if (
                    raw_predict_candle[0] < raw_obs_candle[3]
                    and long_expected_roe > short_expected_roe
                    # and long_expected_roe > self.take_profit
                    # and expected_roe > self.take_profit
                    # and short_expected_roe < self.stop_loss
                ):
                    pos = 1  # long
                elif (
                    raw_predict_candle[0] > raw_obs_candle[3]
                    and long_expected_roe < short_expected_roe
                    # and short_expected_roe > self.take_profit
                    # and expected_roe > self.take_profit
                    # and long_expected_roe < self.stop_loss
                ):
                    pos = -1  # short
                else:
                    pos = 0

        is_position_changed = pos != self._position

        pos *= self.leverage
        self._take_action(pos)

        self._idx += 1
        self._step += 1

        self._take_action_order_limit()
        price = self._get_price()
        self._portfolio.update_interest(borrow_interest_rate=self.borrow_interest_rate)

        portfolio_value = self._portfolio.valorisation(price)

        done, truncated = False, False

        if portfolio_value <= 0:
            done = True
        if self._idx >= len(self.df) - 1:
            truncated = True
        if (
            isinstance(self.max_episode_duration, int)
            and self._step >= self.max_episode_duration - 1
        ):
            truncated = True

        entry_price = (
            self.historical_info["entry_price", -1]
            if not is_position_changed
            else price
        )
        entry_valuation = (
            self.historical_info["entry_valuation", -1]
            if not is_position_changed
            else portfolio_value
        )

        self.historical_info.add(
            idx=self._idx,
            step=self._step,
            date=self.df.index.values[self._idx],
            position=self._position,
            real_position=self._portfolio.real_position(price),
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation=portfolio_value,
            reward=-(abs(_ohlc)).sum(),
            entry_price=entry_price,
            entry_valuation=entry_valuation,
            PNL=portfolio_value - entry_valuation,
            ROE=(portfolio_value / entry_valuation) - 1,
            pc_counter=self.pc_counter,
        )

        if truncated:
            self.calculate_metrics()
            self.log()

        return (
            self._get_obs(),
            self.historical_info["reward", -1],
            done,
            truncated,
            self.historical_info[-1],
        )

    def add_metric(self, name, function):
        self.log_metrics.append({"name": name, "function": function})

    def calculate_metrics(self):
        self.results_metrics = {
            "Market Return": f"{100 * (self.historical_info['data_close', -1] / self.historical_info['data_close', 0] - 1):5.2f}%",
            "Portfolio Return": f"{100 * (self.historical_info['portfolio_valuation', -1] / self.historical_info['portfolio_valuation', 0] - 1):5.2f}%",
            "Position Changes": np.sum(np.diff(self.historical_info["position"]) != 0),
            "Portfolio Value": self.historical_info["portfolio_valuation", -1],
            "Episode Length": len(self.historical_info["position"]),
            "Episode Reward": sum(self.historical_info["reward"]),
        }

        for metric in self.log_metrics:
            self.results_metrics[metric["name"]] = metric["function"](
                self.historical_info
            )

    def get_metrics(self):
        return self.results_metrics

    def log(self):
        if self.verbose > 0:
            text = ""
            for key, value in self.results_metrics.items():
                text += f"{key} : {value}   |   "
            print(text)

    def save_for_render(self, dir="render_logs"):
        assert (
            "open" in self.df
            and "high" in self.df
            and "low" in self.df
            and "close" in self.df
        ), "Your DataFrame needs to contain columns : open, high, low, close to render !"
        columns = list(
            set(self.historical_info.columns)
            - set([f"date_{col}" for col in self._info_columns])
        )
        history_df = pd.DataFrame(self.historical_info[columns], columns=columns)
        history_df.set_index("date", inplace=True)
        history_df.sort_index(inplace=True)
        render_df = self.df.join(history_df, how="inner")

        if not os.path.exists(dir):
            os.makedirs(dir)
        render_df.to_pickle(
            f"{dir}/{self.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        )


class MultiDatasetDiscretedTradingEnv(DiscretedTradingEnv):
    def __init__(
        self,
        dataset_dir,
        *args,
        preprocess=lambda df: df,
        episodes_between_dataset_switch=1,
        btc_index=False,
        **kwargs,
    ):
        self.dataset_dir = dataset_dir
        self.preprocess = preprocess
        self.episodes_between_dataset_switch = episodes_between_dataset_switch
        self.dataset_pathes = glob.glob(self.dataset_dir)
        self.btc_index = btc_index

        if self.btc_index:
            for k, v in enumerate(self.dataset_pathes):
                if "binanceusdm-BTCDOMUSDT-15m.pkl" in v:
                    self.dataset_pathes.pop(k)

        self.dataset_nb_uses = np.zeros(shape=(len(self.dataset_pathes),))
        super().__init__(self.next_dataset(), *args, **kwargs)

    def next_dataset(self):
        self._episodes_on_this_dataset = 0
        # Find the indexes of the less explored dataset
        potential_dataset_pathes = np.where(
            self.dataset_nb_uses == self.dataset_nb_uses.min()
        )[0]
        # Pick one of them
        random_int = np.random.randint(potential_dataset_pathes.size)
        dataset_path = self.dataset_pathes[random_int]
        self.dataset_nb_uses[random_int] += 1  # Update nb use counts

        self.name = Path(dataset_path).name
        df = self.preprocess(pd.read_pickle(dataset_path))

        print(dataset_path)

        if self.btc_index:
            BTCUSDT_PATH = "/".join(
                dataset_path.split("/")[:-1] + ["binanceusdm-BTCDOMUSDT-15m.pkl"]
            )
            BTCUSDT = pd.read_pickle(BTCUSDT_PATH)
            BTCUSDT = pd.DataFrame(
                {"feature_btc_log_returns": np.log(BTCUSDT.close).diff().dropna()}
            )

            df = pd.concat([BTCUSDT, df], axis=1).fillna(0)

        return df

    def reset(self, seed=None, options=None):
        self._episodes_on_this_dataset += 1
        if self._episodes_on_this_dataset % self.episodes_between_dataset_switch == 0:
            self._set_df(self.next_dataset())
        if self.verbose > 1:
            print(f"Selected dataset {self.name} ...")
        return super().reset(seed, options)
