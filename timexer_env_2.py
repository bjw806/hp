import datetime
import glob
import os
import warnings
import math
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
    reward = 0
    pnl = history["realized_pnl", -1]
    # roe = history["realized_roe", -1] * 100
    positions = history["position"]
    current_position = positions[-1]

    if pnl == 0:
        if current_position == 0:  # no position
            # 포지션이 0이 아닌 마지막 시점 찾기 (뒤에서부터 탐색)

            last_non_zero_position_idx = (
                len(positions) - 1 - (positions[::-1] != 0).argmax()
                if (positions != 0).any()
                else -1
            )

            # 마지막 포지션이 0이 아닌 시점 바로 다음부터 현재까지의 data_close 분산 계산
            data_close = history["data_close", last_non_zero_position_idx + 1 :]

            if len(data_close) > 12:  # 5m * 12 = 1h
                recent_data_close = (
                    np.diff(data_close) / data_close[:-1] * 100
                )  # % 변화율
                variance_penalty = recent_data_close.var(ddof=1)
                reward -= variance_penalty  # 분산을 패널티로 적용
                reward *= len(data_close)
                # reward *= 10

            # l = positions[positions > 0].sum()
            # s = positions[positions < 0].sum()
            # ratio = l / (l + s) if (l + s) != 0 else 0.5
            # reward -= abs(ratio - 0.5) * len(positions)

        else:  # hold/open position
            # reward += roe
            # unrealized_roe = history["unrealized_roe", -1] * 100
            # reward += unrealized_roe * 0.1
            unrealized_pnl = history["unrealized_pnl", -1]
            reward += unrealized_pnl * 0.1

            # if history["hold_time", -1] > 12:  # 1h
            #     reward -= math.sqrt(history["hold_time", -1])
    else:  # close position
        # cummulative_pnl = history["realized_pnl"].sum()
        # reward += cummulative_pnl  # if pnl > 0 else (pnl*2)
        # total_roe = cummulative_pnl / history["portfolio_valuation", 0] * 100
        # realized_roe = history["realized_roe", -1] * 100
        # reward += realized_roe
        # reward += total_roe

        reward += pnl
        # reward *= math.sqrt(history["hold_time", -2])

        # if history["hold_time", -2] == 1:
        #     hold_time = history["hold_time"]
        #     h = hold_time[hold_time == 1]
        #     # p = positions[positions != 0]
        #     reward -= math.sqrt(h.sum())

        # last_diff_position_idx = (
        #     (len(positions) - 1 - (positions[::-1] != current_position).argmax())
        #     if (positions != current_position).any()
        #     else -1
        # )

        # data_close = history["data_close", last_diff_position_idx + 1 :]

        # if len(data_close) > 1:
        #     recent_data_close = np.diff(data_close) / data_close[:-1]
        #     variance_bonus = (
        #         recent_data_close.var(ddof=1) if len(recent_data_close) > 12 else 0
        #     )
        #     reward += variance_bonus

        # position bias
        # pc = np.sum(np.diff(positions) != 0)

        # if pc > 1:
        #     spr = history["single_position_record"]
        #     position_balance = spr[spr != 0].mean()
        #     reward -= abs(position_balance) * 0.1

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

        # env
        self._info = [
            "realized_total_ROE",
            "realized_ROE",
            "unrealized_total_ROE",
            "unrealized_ROE",
            # "position",
            # "record",
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

        """
        for i in range(observation.shape[1]):
            col = observation[:, i]
            min_val = np.min(col)
            max_val = np.max(col)

            observation[:, i] = (
                (col - min_val) / (max_val - min_val) if max_val - min_val != 0 else 0
            )
        """

        # 1열과 2열 각각 정규화
        for i in range(2):
            col = observation[:, i]
            col_range = np.ptp(col)  # 최대값 - 최소값 계산
            observation[:, i] = np.where(
                col_range != 0, (col - np.min(col)) / col_range, 0
            )

        # 3열부터 7열까지 함께 정규화
        cols_to_normalize = slice(2, 7)
        cols = observation[:, cols_to_normalize]
        col_range = np.ptp(cols)  # 전체 범위 계산
        observation[:, cols_to_normalize] = np.where(
            col_range != 0, (cols - np.min(cols)) / col_range, 0
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
                # self._position,  # position
                # self.historical_info["record", -1],  # win or lose
                self.historical_info["position", -1] * 0.01,  # real position
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
        self._step = 0
        self._position = np.random.choice(self.positions)
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
            single_position_record=self._position,
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
        self._position_idx = position_index[0]
        pos = self.positions[self._position_idx]

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

        if portfolio_value <= self.portfolio_initial_value * 0.1:
            done = True
            self.liquidation = True
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
            if self._position == 0 or prev_valuation != 0:
                if self._position == 0:  # close position
                    entry_valuation = 0
                    hold_time = 0

                elif prev_valuation != 0:  # switch position
                    entry_valuation = portfolio_value
                    hold_time = 1

                realized_pnl = portfolio_value - prev_valuation
                realized_roe = realized_pnl / prev_valuation
                record = 1 if portfolio_value > prev_valuation else -1
            else:  # open position
                entry_valuation = portfolio_value
                hold_time = 1

        else:  # hold position
            entry_valuation = prev_valuation
            # if self._position == 0:
            #     no_position_panelty -= 0.1
            # else:
            #     no_position_panelty += 0.1

            hold_time = (
                self.historical_info["hold_time", -1] + 1 if self._position != 0 else 0
            )

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
            single_position_record=self._position if is_position_changed else 0,
        )

        if self.liquidation:
            # reward = max(
            #     portfolio_value,
            #     prev_valuation,
            #     # self.portfolio_initial_value,
            # )
            reward = self.portfolio_initial_value
            # reward *= math.sqrt(self._step)
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
            # self.historical_info["reward", -1] = portfolio_value - self.portfolio_initial_value
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
        rng = np.random.default_rng()
        random_int = rng.choice(potential_dataset_pathes)
        dataset_path = self.dataset_pathes[random_int]
        self.dataset_nb_uses[random_int] += 1  # Update nb use counts

        self.name = Path(dataset_path).name
        df = self.preprocess(pd.read_pickle(dataset_path))

        if self.btc_index:
            p = dataset_path.split("/")[:-1]

            BTCUSDT_PATH = "/".join(p + ["binanceusdm-BTCUSDT-5m.pkl"])
            BTCUSDT = pd.read_pickle(BTCUSDT_PATH)
            # BTCUSDT = pd.DataFrame(
            #     {"feature_btc_log_returns": np.log(BTCUSDT.close).diff()}
            # )
            BTCUSDT = pd.DataFrame({"feature_btc": BTCUSDT.close})

            BTCDOMUSDT_PATH = "/".join(p + ["binanceusdm-BTCDOMUSDT-5m.pkl"])
            BTCDOMUSDT = pd.read_pickle(BTCDOMUSDT_PATH)
            # BTCDOMUSDT = pd.DataFrame(
            #     {"feature_btcdom_log_returns": np.log(BTCDOMUSDT.close).diff()}
            # )
            BTCDOMUSDT = pd.DataFrame({"feature_btcdom": BTCDOMUSDT.close})

            df = pd.concat([BTCUSDT, BTCDOMUSDT, df], axis=1)

        return df.fillna(0)

    def reset(self, seed=None, options=None):
        self._episodes_on_this_dataset += 1
        if self._episodes_on_this_dataset % self.episodes_between_dataset_switch == 0:
            self._set_df(self.next_dataset())
        if self.verbose > 1:
            print(f"Selected dataset {self.name} ...")
        return super().reset(seed, options)
