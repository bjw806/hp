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
    # episode_length = len(history)
    roe = history["ROE", -1] * 100
    pnl = history["PNL", -1]
    # (
    #     history["portfolio_valuation", -1] / (history["entry_valuation", -1]) - 1
    # )  # * math.sqrt(math.sqrt(3000 - episode_length))
    # total_roe = (
    #     history["portfolio_valuation", -1] / history["portfolio_valuation", 0] - 1
    # )  # * math.sqrt(math.sqrt(episode_length))
    # position = -(history["position"].mean() ** 2) * 0.1
    # record = -history["record"].sum() * 0.01
    # MDD = history["ROE"].min()

    reward = roe #+ total_roe + position + record
    return roe + pnl


# def basic_reward_function(history: History):
#     roe = (
#         history["portfolio_valuation", -1] / (history["entry_valuation", -1] + 1e-8) - 1
#     )
#     total_roe = (
#         history["portfolio_valuation", -1] / history["portfolio_valuation", 0] - 1
#     )
#     reward = (
#         (roe**2)
#         if roe > 0
#         else -(roe**2) + ((total_roe**2) if total_roe > 0 else -(total_roe**2)) * 0.1
#     )
#     return reward


# def basic_reward_function(history: History):
#     mean_return = np.mean(history["ROE"])
#     std_return = np.std(history["ROE"])

#     if std_return == 0:
#         return 0.0

#     risk_free_rate = 0
#     sharpe_ratio = (mean_return - risk_free_rate) / std_return

#     reward = sharpe_ratio ** 2 if sharpe_ratio > 0 else -(sharpe_ratio ** 2)

#     return reward


def dynamic_feature_last_position_taken(history: History):
    return history["position", -1]


def dynamic_feature_real_position(history: History):
    return history["real_position", -1]


class DiscretedTradingEnv(gym.Env):
    metadata = {"render_modes": ["logs"]}

    def __init__(
        self,
        df: pd.DataFrame,
        positions: list = [-1, 1],
        multiplier: list = [2, 5, 10],
        dynamic_feature_functions: list = [
            # dynamic_feature_last_position_taken,
            # dynamic_feature_real_position,
        ],
        reward_function: Callable = basic_reward_function,
        window_size: int = 30,
        trading_fees: float = 0.0001,
        borrow_interest_rate: float = 0.00003,
        portfolio_initial_value: int = 1000,
        # hold_threshold: float = 0,  # 0.5
        # close_threshold: float = 0,  # 0.5
        initial_position: int = "random",  # str or int
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
        self.positions = positions
        self.multiplier = multiplier
        self.trading_fees = trading_fees
        self.borrow_interest_rate = borrow_interest_rate
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.initial_position = initial_position
        self.force_clear = force_clear
        self.pc_counter = 0
        self.liquidation = False

        # env
        self.max_episode_duration = max_episode_duration
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function = reward_function
        self.window_size = window_size
        self._set_df(df)
        self.action_space = spaces.MultiDiscrete(
            [len(self.positions), len(self.multiplier)]
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=[self.window_size, self._nb_features],
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

        _step_index = np.arange(self._idx + 1 - self.window_size, self._idx + 1)
        observation = self._obs_array[_step_index]

        for i in range(observation.shape[1]):
            col = observation[:, i]
            min_val = np.min(col)
            max_val = np.max(col)

            observation[:, i] = (
                (col - min_val) / (max_val - min_val) if max_val - min_val != 0 else 0
            )

        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.pc_counter = 0
        self.liquidation = False
        self._step = 0
        self._position = (
            np.random.choice(self.positions) * np.random.choice(self.multiplier)
            if self.initial_position == "random"
            else self.initial_position
        )
        self._limit_orders = {}
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
            entry_valuation=self.portfolio_initial_value,
            PNL=0,
            ROE=0,
            pc_counter=0,
            record=0,
            liquidation=0,
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

    def step(self, position_index=None):
        pos = self.positions[position_index[0]] * self.multiplier[position_index[1]]

        temp_position = self._position
        self._take_action(pos)

        self._idx += 1
        self._step += 1

        self._take_action_order_limit()
        price = self._get_price()
        self._portfolio.update_interest(borrow_interest_rate=self.borrow_interest_rate)

        portfolio_value = self._portfolio.valorisation(price)
        # portfolio_distribution = self._portfolio.get_portfolio_distribution()

        done, truncated = False, False
        is_position_changed = self._position != temp_position

        if portfolio_value <= 100:  # liquidation at 5% of initial value
            done = True
            self.liquidation = True
            portfolio_value = 1
        if self._idx >= len(self.df) - 1:
            truncated = True
        if (
            isinstance(self.max_episode_duration, int)
            and self._step >= self.max_episode_duration - 1
        ):
            truncated = True

        record = 0
        prev_valuation = self.historical_info["entry_valuation", -1]

        if is_position_changed:
            if self._position == 0:  # close position
                entry_valuation = 0
                record = 1 if portfolio_value > prev_valuation else -1
            else:  # position taken
                entry_valuation = portfolio_value
        else:  # hold position
            entry_valuation = prev_valuation

        self.historical_info.add(
            idx=self._idx,
            step=self._step,
            date=self.df.index.values[self._idx],
            # position_index=position_index,
            position=self._position,
            real_position=self._portfolio.real_position(price),
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation=portfolio_value,
            # portfolio_distribution=portfolio_distribution,
            reward=0,
            entry_valuation=entry_valuation,
            PNL=0 if prev_valuation == 0 else (portfolio_value - prev_valuation),
            ROE=0 if prev_valuation == 0 else ((portfolio_value / prev_valuation) - 1),
            pc_counter=self.pc_counter,
            record=record,
            liquidation=-1 if self.liquidation else 0,
        )
        self.historical_info["reward", -1] = (
            -1e5 if self.liquidation else self.reward_function(self.historical_info)
        )

        # print(self.historical_info["pc_counter"])

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
                if "BTC" in v:
                    self.dataset_pathes.pop(k - 1)

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

        if self.btc_index:
            p = dataset_path.split("/")[:-1]

            BTCUSDT_PATH = "/".join(p + ["binanceusdm-BTCUSDT-15m.pkl"])
            BTCUSDT = pd.read_pickle(BTCUSDT_PATH)
            BTCUSDT = pd.DataFrame(
                {"feature_btc_log_returns": np.log(BTCUSDT.close).diff()}
            )

            BTCDOMUSDT_PATH = "/".join(p + ["binanceusdm-BTCDOMUSDT-15m.pkl"])
            BTCDOMUSDT = pd.read_pickle(BTCDOMUSDT_PATH)
            BTCDOMUSDT = pd.DataFrame(
                {"feature_btcdom_log_returns": np.log(BTCDOMUSDT.close).diff()}
            )

            df = pd.concat([BTCUSDT, BTCDOMUSDT, df], axis=1)

        return df.fillna(0)

    def reset(self, seed=None, options=None):
        self._episodes_on_this_dataset += 1
        if self._episodes_on_this_dataset % self.episodes_between_dataset_switch == 0:
            self._set_df(self.next_dataset())
        if self.verbose > 1:
            print(f"Selected dataset {self.name} ...")
        return super().reset(seed, options)
