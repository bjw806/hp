import datetime
import glob
import os
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gym_trading_env.utils.history import History
from gymnasium import spaces

warnings.filterwarnings("error")


def dynamic_feature_last_position_taken(history: History):
    return history["position", -1]


def dynamic_feature_real_position(history: History):
    return history["real_position", -1]


class DiscretedTradingEnv(gym.Env):
    metadata = {"render_modes": ["logs"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = None,
        dynamic_feature_functions: list = [
            # dynamic_feature_last_position_taken,
            # dynamic_feature_real_position,
        ],
        max_episode_duration: str = "max",
        verbose: int = 1,
        name: str = "Stock",
        render_mode: str = "logs",
    ):
        # initial
        self.name = name
        self.verbose = verbose
        self.render_mode = render_mode
        self.log_metrics = []

        # env
        self.max_episode_duration = max_episode_duration
        self.dynamic_feature_functions = dynamic_feature_functions
        self.window_size = window_size
        self._set_df(df)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=[3])
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
        self._idx = 0

        if self.window_size is not None:
            self._idx = self.window_size - 1
        if self.max_episode_duration != "max":
            self._idx = np.random.randint(
                low=self._idx, high=len(self.df) - self.max_episode_duration - self._idx
            )

        self.historical_info = History(max_size=len(self.df))
        self.historical_info.set(
            idx=self._idx,
            step=self._step,
            date=self.df.index.values[self._idx],
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            reward=0,
        )

        return self._get_obs(), self.historical_info[0]

    def render(self):
        pass

    def step(self, hlc=[None]):
        lr_current_candle = np.array(
            self.df[
                [
                    # "feature_open_lr",
                    "feature_high_lr",
                    "feature_low_lr",
                    "feature_log_returns",
                ]
            ],
            dtype=np.float32,
        )[self._idx + 1]

        # MSE = (raw_current_candle - raw_predict_candle)
        # MAPE = (np.abs((raw_current_candle - raw_predict_candle) / raw_current_candle)) * 100

        # if MAPE.mean() > 1:
        #     print(MAPE, MAPE.mean())
        # print(MAPE, MAPE.mean())
        # print(real_candle, predict_candle)

        penalty = 0

        # open
        if 0 > hlc[0]:
            penalty -= 1

        if 0 < hlc[1]:
            penalty -= 1

        # close
        if hlc[2] > hlc[0]:
            penalty -= 1

        if hlc[2] < hlc[1]:
            penalty -= 1

        # high / low
        if hlc[0] < hlc[1]:
            penalty -= 1

        self._idx += 1
        self._step += 1

        done, truncated = False, False

        if (self._idx >= len(self.df) - 1) or (
            isinstance(self.max_episode_duration, int)
            and self._step >= self.max_episode_duration - 1
        ):
            truncated = True

        # _high = (ohlc[1] - lr_current_candle[1])
        # _low = (ohlc[2] - lr_current_candle[2])
        # _close = (ohlc[3] - lr_current_candle[3])
        # _ohlc = np.array([_open, _high, _low, _close])
        # _w = abs(lr_current_candle[1] - lr_current_candle[2])

        self.historical_info.add(
            idx=self._idx,
            step=self._step,
            date=self.df.index.values[self._idx],
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            reward=-(abs(hlc - lr_current_candle).sum()),
            # -((abs(np.array(ohlc) - data) + 1) ** 2).sum() - 4
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
            "Episode Length": len(self.historical_info),
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
