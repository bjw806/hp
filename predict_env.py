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

    def step(self, hlc=None):
        """
                다음 캔들(feature) 정보 불러오기:
        lr_current_candle 변수를 통해 현재 스텝 다음 시점의 캔들 관련 feature들을 가져옵니다.
        이때 사용되는 feature로는 feature_high_lr, feature_low_lr, feature_log_returns, high, low, close 등이 있으며, 이들은 다음 시점(self._idx + 1)의 값을 참고합니다.

        현재 캔들(관측) 정보 불러오기:
        raw_obs_candle를 통해 현재 시점(self._idx)의 open, high, low, close 가격 정보를 가져옵니다.
        이 값들은 현재 스텝에서의 실제 시장 상황을 반영합니다.

        예측값(hlc)로부터 예상 가격 계산:
        입력으로 받은 hlc(high, low, close에 대한 비율 변화 log-scale 예측치 가정)를 바탕으로 raw_predict_candle를 계산합니다.
        hlc 값을 지수변환 후 현재 종가(raw_obs_candle[3])에 곱해 예측된 (예상) 고가, 저가, 종가를 추정합니다.

        예측값 검증:
        예측한 hlc가 유효한 구조를 가지고 있는지 검사합니다.

        예: hlc[0] > 0 이어야 고가가 현재 종가보다 높다거나, hlc[1] < 0 이어야 저가가 종가보다 낮는 등, 혹은 예측한 close가 예측 high와 low 사이에 존재하는지 등의 제약을 검증합니다.
        만약 이러한 조건이 어긋나면, 현재 스텝에서는 포지션을 잡지 않고 넘어갑니다(pos = 0).
        포지션 방향 결정(예측 기반 전략 로직):
        예측이 유효하다면, 롱과 숏 포지션 각각에 대한 기대 수익률(ROE)을 계산합니다.

        long_expected_roe: 예측된 고가를 기준으로 롱에 진입했을 때 기대되는 수익률
        short_expected_roe: 예측된 저가를 기준으로 숏에 진입했을 때 기대되는 수익률
        현재 가지고 있는 포지션(self._position)을 토대로 다음을 결정합니다.

        포지션이 이미 있을 경우:
        목표 이익 도달(take_profit) 여부나 손절(stop_loss) 조건을 먼저 체크해 포지션 청산 여부 결정
        청산하지 않는다면, 롱/숏 기대값 비교를 통해 포지션 유지, 변경 혹은 그대로 유지
        포지션이 없을 경우:
        롱이 유리한지, 숏이 유리한지 판단하고 그 방향으로 진입
        이 단계에서 최종적으로 pos 값(포지션 방향: -1, 0, +1)을 결정합니다.

        포지션 변경 여부 확인:
        is_position_changed = pos != self._position를 통해 이번 스텝에서 포지션이 바뀌었는지 확인합니다.
        이후 pos에 레버리지를 곱해 최종 포지션 크기를 결정하고 _take_action(pos)를 통해 실제 포지션을 업데이트합니다.

        환경 진행:
        self._idx와 self._step를 증가시켜 시뮬레이션을 다음 시점으로 진행합니다.

        주문 처리 후 가격 갱신:
        _take_action_order_limit()를 통해 지정가 주문 등의 처리 로직을 실행하고 _get_price()를 통해 현재 가격(또는 다음 가격)을 반영합니다.

        금리 비용 업데이트:
        self._portfolio.update_interest()를 통해 차입 금리(borrow_interest_rate)에 따른 이자 비용을 업데이트합니다.

        포트폴리오 평가액 계산:
        portfolio_value = self._portfolio.valorisation(price)를 통해 현재 포지션과 금리, 가격 변동을 반영한 포트폴리오 평가액을 계산합니다.

        시뮬레이션 종료 조건 체크:

        portfolio_value <= 0: 원금 전액 상실 → done = True
        self._idx가 데이터 마지막에 도달 → truncated = True
        self.max_episode_duration(최대 에피소드 길이)에 도달 → truncated = True
        이를 통해 에피소드 종료 조건을 판별합니다.

        진입 기준(Entry Valuation) 및 기록용 변수 설정:
        포지션 변경이 있었다면, 그 시점의 price와 portfolio_value를 진입 시점 기준으로 삼습니다(entry_price 및 entry_valuation 갱신).
        포지션 변경이 없었다면 이전 스텝의 entry_valuation을 유지합니다.

        수익률, PNL, ROE 계산 및 기록:

        PNL = portfolio_value - entry_valuation
        ROE = (portfolio_value / entry_valuation) - 1
        이를 계산한 뒤 historical_info.add()를 통해 이 스텝의 정보(가격, 포지션, PNL, ROE, reward 등)를 기록합니다.

        트러케이션 시 메트릭 계산 및 로깅:
        만약 이번 스텝에서 truncated = True라면, calculate_metrics()와 log()를 호출해 정리 작업을 수행합니다.

        결과 반환:

        관측값(obs) = _get_obs()
        보상(reward) = 이번 스텝의 portfolio_value - entry_valuation 값 (또는 해당 로직 상 reward)
        done, truncated, 현재 스텝에 대한 historical_info 항목을 반환합니다.
        정리하자면, step() 메서드는

        다음 시점의 특징 추출 →
        현재 시점 관측값 불러오기 →
        예측값 검증 및 포지션 결정 →
        포지션 업데이트 및 환경 진행 →
        가격, 금리, 포트폴리오 가치 업데이트 →
        종료 조건 확인 →
        결과 기록 및 반환
        """
        lr_current_candle = np.array(
            self.df[
                [
                    # "feature_open_lr",
                    "feature_high_lr",
                    "feature_low_lr",
                    "feature_log_returns",
                    "high",
                    "low",
                    "close",
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
        raw_predict_candle = np.exp(hlc) * raw_obs_candle[3]

        # _open = (ohlc[0] - lr_current_candle[0]) * 1
        # _high = (ohlc[1] - lr_current_candle[1]) * 3
        # _low = (ohlc[2] - lr_current_candle[2]) * 3
        # _close = (ohlc[3] - lr_current_candle[3]) * 2
        # _ohlc = np.array([_open, _high, _low, _close])

        # pos = self.positions[0]

        if (
            0 > hlc[0]
            or 0 < hlc[1]
            or hlc[2] > hlc[0]
            or hlc[2] < hlc[1]
            or hlc[0] < hlc[1]
        ):
            pos = 0
            # print("Invalid prediction")
        else:
            long_expected_roe = (
                abs(raw_predict_candle[0] - raw_obs_candle[3])
                / raw_obs_candle[3]
                * self.leverage
            )
            short_expected_roe = (
                abs(raw_predict_candle[1] - raw_obs_candle[3])
                / raw_obs_candle[3]
                * self.leverage
            )
            expected_roe = (
                abs(raw_obs_candle[3] - raw_obs_candle[3])
                / raw_obs_candle[3]
                * self.leverage
            )

            # a = max(raw_predict_candle / lr_current_candle[3:] - 1) * 100
            # if a > 2:
            #     print(a)

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
                        long_expected_roe > short_expected_roe
                        # and long_expected_roe > self.take_profit
                        # and expected_roe > self.take_profit
                        # and short_expected_roe < self.stop_loss
                        and raw_obs_candle[3] < raw_predict_candle[2]
                    ):
                        pos = 1 * self.leverage  # long
                    elif (
                        long_expected_roe < short_expected_roe
                        # and short_expected_roe > self.take_profit
                        # and expected_roe > self.take_profit
                        # and long_expected_roe < self.stop_loss
                        and raw_obs_candle[3] > raw_predict_candle[2]
                    ):
                        pos = -1 * self.leverage  # short
                    else:
                        pos = self._position  # hold
            else:
                if (
                    long_expected_roe > short_expected_roe
                    # and long_expected_roe > self.take_profit
                    # and expected_roe > self.take_profit
                    # and short_expected_roe < self.stop_loss
                    and raw_obs_candle[3] < raw_predict_candle[2]
                ):
                    pos = 1  # long
                elif (
                    long_expected_roe < short_expected_roe
                    # and short_expected_roe > self.take_profit
                    # and expected_roe > self.take_profit
                    # and long_expected_roe < self.stop_loss
                    and raw_obs_candle[3] > raw_predict_candle[2]
                ):
                    pos = -1  # short
                else:
                    pos = 0

                pos *= self.leverage

        temp_position = self._position
        self._take_action(pos)

        self._idx += 1
        self._step += 1

        self._take_action_order_limit()
        price = self._get_price()
        self._portfolio.update_interest(borrow_interest_rate=self.borrow_interest_rate)

        portfolio_value = self._portfolio.valorisation(price)

        done, truncated = False, False
        is_position_changed = pos != temp_position

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
            reward=portfolio_value - entry_valuation,
            # -(abs(hlc - lr_current_candle[:3]) * 10).sum(),
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
            False,
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

        # print(dataset_path)

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
