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
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])


def basic_penalty_function(history: History):
    no_action_steps = 60 * 6

    if len(history) > no_action_steps:
        penalty = 0
        counter = 0
        # current_valuation = history["portfolio_valuation", -1]
        # previous_valuation = history["portfolio_valuation", -2] if len(history) > 1 else current_valuation

        # if current_valuation < previous_valuation:
        #     penalty -= (previous_valuation - current_valuation) * 0.01

        curr_position = history["position", -1]

        for i in range(no_action_steps):
            if curr_position != history["position", -(i + 1)]:
                counter += 1

        if counter < 10:
            penalty -= 10

        return penalty

    return 0


def reward_only_position_changed(history):
    if history["portfolio_valuation", -1] <= 0:
        return -1

    prev_position = history["position", -2]
    curr_position = history["position", -1]

    position_roe = history["portfolio_valuation", -1] / history["entry_valuation", -1] - 1
    total_roe = history["portfolio_valuation", -1] / history["portfolio_valuation", 0] - 1  # / sqrt(index)

    if prev_position == curr_position:
        return 0
    else:
        return position_roe + total_roe


def dynamic_feature_last_position_taken(history):
    return history["position", -1]


def dynamic_feature_real_position(history):
    return history["real_position", -1]


class DiscretedTradingEnv(gym.Env):
    metadata = {"render_modes": ["logs"]}

    def __init__(
        self,
        df: pd.DataFrame,
        positions: list = [0, 1],
        dynamic_feature_functions: list = [
            dynamic_feature_last_position_taken,
            # dynamic_feature_real_position,
        ],
        reward_function: Callable = reward_only_position_changed,
        penalty_function: Callable = basic_penalty_function,
        window_size: int = None,
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
        self.trading_fees = trading_fees
        self.borrow_interest_rate = borrow_interest_rate
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.initial_position = initial_position
        self.last_trade_position = 0
        self.last_trade_position_index = 0
        self.force_clear = force_clear

        # env
        self.max_episode_duration = max_episode_duration
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function = reward_function
        self.penalty_function = penalty_function
        self.window_size = window_size
        self._set_df(df)
        self.action_space = spaces.Discrete(len(positions))
        self.observation_space = spaces.Dict({
            "equity": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "total_ROE": spaces.Box(low=-1, high=np.inf, shape=(1,), dtype=np.float32),
            # "position": spaces.Box(
            #     low=-np.inf, high=np.inf, shape=(1,), dtype=np.int64
            # ),
            "PNL": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float32,
            ),
            "ROE": spaces.Box(low=-1, high=np.inf, shape=(1,), dtype=np.float32),
            "entry_price": spaces.Box(
                low=0,
                high=np.inf,
                shape=(1,),
                dtype=np.float32,
            ),
            "features": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=[self.window_size, self._nb_features] if window_size else [self._nb_features],
                dtype=np.float32,
            ),
        })

    def _set_df(self, df):
        df = df.copy()
        self._features_columns = [col for col in df.columns if "feature" in col]
        self._info_columns = list(set(list(df.columns) + ["close"]) - set(self._features_columns))
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
            return history["portfolio_valuation", -1] / history["portfolio_valuation", -2] - 1

    def _get_obs(self):
        for i, dynamic_feature_function in enumerate(self.dynamic_feature_functions):
            self._obs_array[self._idx, self._nb_static_features + i] = dynamic_feature_function(self.historical_info)

        _step_index = (
            self._idx if self.window_size is None else np.arange(self._idx + 1 - self.window_size, self._idx + 1)
        )

        observation = {
            "equity": np.float32(self.historical_info["portfolio_valuation", -1]),
            "total_ROE": np.float32(self.historical_info["portfolio_valuation", -1] / self.portfolio_initial_value - 1),
            # "position": np.int64(self.historical_info["position", -1]),
            "PNL": np.float32(self.historical_info["PNL", -1]),
            "ROE": np.float32(self.historical_info["ROE", -1]),
            "entry_price": np.float32(self.historical_info["entry_price", -1]),
            "features": self._obs_array[_step_index],
        }

        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self._step = 0
        self._position = (
            np.random.choice(self.positions) if self.initial_position == "random" else self.initial_position
        )
        self._limit_orders = {}

        self._idx = 0

        if self.window_size is not None:
            self._idx = self.window_size - 1
        if self.max_episode_duration != "max":
            self._idx = np.random.randint(low=self._idx, high=len(self.df) - self.max_episode_duration - self._idx)

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
            position_index=self.positions.index(self._position),
            position=self._position,
            real_position=self._position,
            last_position_changed_idx=0,
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation=self.portfolio_initial_value,
            portfolio_distribution=self._portfolio.get_portfolio_distribution(),
            reward=0,
            #
            entry_price=self._get_price(),
            entry_valuation=self.portfolio_initial_value,
            PNL=0,
            ROE=0,
            prev_position_valuation=self.portfolio_initial_value,
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
            self.last_trade_position_index = self._idx + 1

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
        pos = self.positions[position_index]
        is_position_changed = position_index is not None and position_index != self.last_trade_position
        is_position_cleared = False

        if self._idx - self.historical_info["last_position_changed_idx"][-1] > 60 and not is_position_changed:
            if pos == 0:
                ls = self.positions
                ls.remove(0)
                pos = np.random.choice(ls)
            else:
                pos = 0
            is_position_cleared = True

        if position_index is not None:
            self._take_action(pos)

        self._idx += 1
        self._step += 1

        self._take_action_order_limit()
        price = self._get_price()
        self._portfolio.update_interest(borrow_interest_rate=self.borrow_interest_rate)
        portfolio_value = self._portfolio.valorisation(price)
        portfolio_distribution = self._portfolio.get_portfolio_distribution()

        done, truncated = False, False

        if portfolio_value <= 0:
            done = True
        if self._idx >= len(self.df) - 1:
            truncated = True
        if isinstance(self.max_episode_duration, int) and self._step >= self.max_episode_duration - 1:
            truncated = True

        entry_price = self.historical_info["entry_price", -1] if not is_position_changed else price
        entry_valuation = self.historical_info["entry_valuation", -1] if not is_position_changed else portfolio_value
        prev_position_valuation = (
            self.historical_info["portfolio_valuation", -1] if not is_position_changed else portfolio_value
        )

        self.historical_info.add(
            idx=self._idx,
            step=self._step,
            date=self.df.index.values[self._idx],
            position_index=position_index,
            position=self._position,
            real_position=self._portfolio.real_position(price),
            last_position_changed_idx=(
                self.historical_info["last_position_changed_idx"][-1]
                if is_position_changed or is_position_cleared
                else self._idx
            ),
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation=portfolio_value,
            portfolio_distribution=portfolio_distribution,
            reward=0,
            entry_price=entry_price,
            entry_valuation=entry_valuation,
            PNL=0,
            ROE=0,
            prev_position_valuation=prev_position_valuation,
        )

        # if done:
        #     reward = -1 * abs(portfolio_value - entry_valuation)
        # else:
        #     reward = (
        #         self.reward_function(self.historical_info)
        #         if is_position_changed
        #         else self.penalty_function(self.historical_info)
        #     )

        reward = self.reward_function(self.historical_info)

        self.historical_info["reward", -1] = reward
        self.historical_info["PNL", -1] = (
            self.historical_info["portfolio_valuation", -1] - self.historical_info["entry_valuation", -1]
        )

        self.historical_info["ROE", -1] = (
            self.historical_info["portfolio_valuation", -1] / self.historical_info["entry_valuation", -1] - 1
        )

        if done or truncated:
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
            self.results_metrics[metric["name"]] = metric["function"](self.historical_info)

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
            "open" in self.df and "high" in self.df and "low" in self.df and "close" in self.df
        ), "Your DataFrame needs to contain columns : open, high, low, close to render !"
        columns = list(set(self.historical_info.columns) - set([f"date_{col}" for col in self._info_columns]))
        history_df = pd.DataFrame(self.historical_info[columns], columns=columns)
        history_df.set_index("date", inplace=True)
        history_df.sort_index(inplace=True)
        render_df = self.df.join(history_df, how="inner")

        if not os.path.exists(dir):
            os.makedirs(dir)
        render_df.to_pickle(f"{dir}/{self.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl")


class MultiDatasetDiscretedTradingEnv(DiscretedTradingEnv):
    def __init__(
        self,
        dataset_dir,
        *args,
        preprocess=lambda df: df,
        episodes_between_dataset_switch=1,
        **kwargs,
    ):
        self.dataset_dir = dataset_dir
        self.preprocess = preprocess
        self.episodes_between_dataset_switch = episodes_between_dataset_switch
        self.dataset_pathes = glob.glob(self.dataset_dir)
        self.dataset_nb_uses = np.zeros(shape=(len(self.dataset_pathes),))
        super().__init__(self.next_dataset(), *args, **kwargs)

    def next_dataset(self):
        self._episodes_on_this_dataset = 0
        # Find the indexes of the less explored dataset
        potential_dataset_pathes = np.where(self.dataset_nb_uses == self.dataset_nb_uses.min())[0]
        # Pick one of them
        random_int = np.random.randint(potential_dataset_pathes.size)
        dataset_path = self.dataset_pathes[random_int]
        self.dataset_nb_uses[random_int] += 1  # Update nb use counts

        self.name = Path(dataset_path).name
        return self.preprocess(pd.read_pickle(dataset_path))

    def reset(self, seed=None, options=None):
        self._episodes_on_this_dataset += 1
        if self._episodes_on_this_dataset % self.episodes_between_dataset_switch == 0:
            self._set_df(self.next_dataset())
        if self.verbose > 1:
            print(f"Selected dataset {self.name} ...")
        return super().reset(seed, options)
