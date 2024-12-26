import datetime
import glob
import math
from mimetypes import init
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
from sympy import li

warnings.filterwarnings("error")


def basic_reward_function(history: History):
    # episode_length = len(history)
    initial_valuation = history["entry_valuation", 0]
    pnl = history["realized_pnl", -1]
    # (
    #     history["portfolio_valuation", -1] / (history["entry_valuation", -1]) - 1
    # )  # * math.sqrt(math.sqrt(3000 - episode_length))

    # position = abs(history["position"].mean())
    # record = history["record"].sum()
    # r_flag = 1 if record > 0 else -1
    # MDD = history["ROE"].min()
    # unrealized_pnl = history["unrealized_pnl", -1] * 0.1

    reward = (
        0
        # pnl * 3
        # + position * 0.01
        # + math.sqrt(abs(record)) * r_flag * 0.01
    )

    # if pnl == 0:
    #     lifetime_pnl = history["realized_pnl"].sum()
    #     # lifetime_roe = (initial_valuation + lifetime_pnl) / initial_valuation - 1
    #     _lifetime_roe = (
    #         initial_valuation + lifetime_pnl  # + history["unrealized_pnl", -1]
    #     ) / initial_valuation - 1
    #     _flag = 1 if _lifetime_roe > 0 else -1
    #     reward += (_lifetime_roe**2) * _flag
    # reward += _lifetime_roe
    # else:
    #     roe = history["realized_roe", -1]  # * 100 # %
    #     total_roe = (history["portfolio_valuation", -1] / initial_valuation - 1) * 100
    #     reward += total_roe
    #     tr = history["portfolio_valuation", -1] / history["portfolio_valuation", 0]
    #     reward *= tr if pnl > 0 else 1
    #     # reward -= math.sqrt(position)
    #     # reward += math.sqrt(record)
    reward = pnl  # if pnl > 0 else (pnl*2)
    # _flag = 1 if roe > 0 else -1
    # reward += roe

    # if history["position", -2] < 0 and pnl > 0:
    #     reward *= 2

    return reward  # ((reward**2) if reward > 0 else -(reward**2)) * 0.1


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
        self.force_clear = force_clear
        self.pc_counter = 0
        self.liquidation = False
        self._position_idx = 1
        self.backup_value_counter = 0
        self.short_counter = 0
        self.long_counter = 0

        # env
        self._info = [
            "realized_total_ROE",
            "realized_ROE",
            "unrealized_total_ROE",
            "unrealized_ROE",
            "position",
            "record",
            "real_position",
        ]
        self.max_episode_duration = max_episode_duration
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function = reward_function
        self.window_size = window_size
        self._set_df(df)
        self.action_space = spaces.Discrete(len(self.positions))
        self.observation_space = spaces.Dict(
            {
                "features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=[self.window_size, self._nb_features],
                    dtype=np.float32,
                ),
                "infos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=[len(self._info)],
                    dtype=np.float32,
                ),
            }
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

        realized_total_pnl = self.historical_info["realized_pnl"].sum()
        infos = np.array(
            [
                realized_total_pnl / self.portfolio_initial_value
                - 1,  # realized_total_ROE
                (realized_total_pnl + self.historical_info["unrealized_pnl", -1])
                / self.portfolio_initial_value
                - 1,  # unrealized_total_ROE
                self.historical_info["realized_roe", -1],  # realized_ROE
                self.historical_info["unrealized_roe", -1],  # unrealized_ROE
                self._position_idx,  # position
                self.historical_info["record", -1],  # win or lose
                self.historical_info["real_position", -1] * 0.01,  # real position
                # <- too larger than other values (-50 ~ 50)
            ]
        )

        # infos[~np.isfinite(infos)] = -1

        obs = {
            "features": observation,
            "infos": infos,
        }

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.pc_counter = 0
        self.liquidation = False
        self.backup_value_counter = 0
        self.short_counter = 0
        self.long_counter = 0
        self._step = 0
        self._position = 0
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
            pc_counter=0,
            record=0,
            liquidation=0,
            unrealized_pnl=0,
            unrealized_roe=0,
            realized_pnl=0,
            realized_roe=0,
            hold_time=0,
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
        # if self._portfolio.fiat > 2000:
        #     floor = self._portfolio.fiat // 1000
        #     self._portfolio.fiat = self._portfolio.fiat % 1000 + 1000
        #     self.backup_value_counter += floor - 1
        # elif self._portfolio.fiat < 100 and self.backup_value_counter > 0:
        #     self._portfolio.fiat += 1000
        #     self.backup_value_counter -= 1

        self._position_idx = position_index[0]
        pos = self.positions[self._position_idx] * 2

        if pos > 0:
            self.long_counter += 1
        elif pos < 0:
            self.short_counter += 1

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
            # portfolio_value = 1
        if self._idx >= len(self.df) - 1:
            truncated = True
        if (
            isinstance(self.max_episode_duration, int)
            and self._step >= self.max_episode_duration - 1
        ):
            truncated = True

        record = 0
        prev_valuation = self.historical_info["entry_valuation", -1]
        realized_pnl = 0
        realized_roe = 0
        no_position_panelty = 0

        if is_position_changed:
            if self._position == 0:  # close position
                entry_valuation = 0
                realized_pnl = portfolio_value - prev_valuation
                realized_roe = (realized_pnl / prev_valuation) - 1
                record = 1 if portfolio_value > prev_valuation else -1
                hold_time = 0

            elif prev_valuation != 0:  # switch position
                entry_valuation = portfolio_value
                realized_pnl = portfolio_value - prev_valuation
                realized_roe = (realized_pnl / prev_valuation) - 1
                record = 1 if portfolio_value > prev_valuation else -1
                hold_time = 1

            else:  # open position
                entry_valuation = portfolio_value
                hold_time = 1

        else:  # hold position
            entry_valuation = prev_valuation
            # if self._position == 0:
            #     no_position_panelty -= 0.1
            # else:
            #     no_position_panelty += 0.1
            hold_time = self.historical_info["hold_time", -1] + 1

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
            pc_counter=self.pc_counter,
            record=record,
            liquidation=-1 if self.liquidation else 0,
            unrealized_pnl=0
            if prev_valuation == 0
            else (portfolio_value - prev_valuation),
            unrealized_roe=0
            if prev_valuation == 0
            else ((portfolio_value / prev_valuation) - 1),
            realized_pnl=realized_pnl,
            realized_roe=realized_roe,
            hold_time=hold_time,
        )

        if self.liquidation:
            if portfolio_value > prev_valuation:
                reward = portfolio_value
            else:
                reward = prev_valuation
            reward = -abs(reward)
        else:
            reward = self.reward_function(
                self.historical_info
            )  # + 0.1 + no_position_panelty
            # reward += self.backup_value_counter
            # norm =  abs(self.long_counter - self.short_counter) / (self.long_counter + self.short_counter)
            # reward -= norm * 0.1

        self.historical_info["reward", -1] = reward

        # print(self.historical_info["pc_counter"])

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
            self.dataset_pathes = [v for v in self.dataset_pathes if "BTC" not in v]

        self.dataset_nb_uses = np.zeros(shape=(len(self.dataset_pathes),))
        super().__init__(self.next_dataset(), *args, **kwargs)

    def next_dataset(self):
        self._episodes_on_this_dataset = 0
        # Find the indexes of the less explored dataset
        potential_dataset_pathes = np.where(
            self.dataset_nb_uses == self.dataset_nb_uses.min()
        )[0]
        # Pick one of them
        random_int = np.random.choice(potential_dataset_pathes)
        dataset_path = self.dataset_pathes[random_int]
        self.dataset_nb_uses[random_int] += 1  # Update nb use counts

        self.name = Path(dataset_path).name
        df = self.preprocess(pd.read_pickle(dataset_path))

        if self.btc_index:
            p = dataset_path.split("/")[:-1]

            BTCUSDT_PATH = "/".join(p + ["binanceusdm-BTCUSDT-5m.pkl"])
            BTCUSDT = pd.read_pickle(BTCUSDT_PATH)
            BTCUSDT = pd.DataFrame(
                {"feature_btc_log_returns": np.log(BTCUSDT.close).diff()}
            )

            BTCDOMUSDT_PATH = "/".join(p + ["binanceusdm-BTCDOMUSDT-5m.pkl"])
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
