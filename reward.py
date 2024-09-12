import numpy as np


def reward_by_pnl(history):
    if history["portfolio_valuation", -1] <= 0:
        return -1

    prev_position = history["position", -2]
    curr_position = history["position", -1]

    if prev_position == curr_position:
        return 0
    else:
        return (history["portfolio_valuation", -1] - history["entry_valuation", -1]) / history["portfolio_valuation", 0]
    



def refined_pnl(history):
    total_roe = history["portfolio_valuation", -1] / history["portfolio_valuation", 0] - 1

    roe = (history["portfolio_valuation", -1] - history["entry_valuation", -1]) / history["portfolio_valuation", 0]
    if abs(roe + total_roe) > 1:
        return 1 if roe + total_roe > 0 else -1

    return roe + total_roe



def calculate_volatility(portfolio_valuations):
    """
    포트폴리오 가치의 리스트를 기반으로 변동성(표준편차)를 계산합니다.
    """
    if len(portfolio_valuations) < 2:
        return 0  # 데이터가 충분하지 않은 경우 변동성을 0으로 반환
    log_returns = np.diff(np.log(portfolio_valuations))
    volatility = np.std(log_returns)
    return volatility


def reward_function_volatility(history):
    """
    변경된 보상 함수: 동적으로 N을 계산하고, 손해가 났을 때 음수 보상을 부여합니다.
    """
    total_steps = history[-1]["step"] + 1  # 총 스텝 수
    N_ratio = 0.1  # 전체 데이터의 마지막 10%를 사용
    N = max(int(total_steps * N_ratio), 1)  # 적어도 한 스텝은 포함되도록

    # 최근 N개 스텝의 포트폴리오 가치 추출
    portfolio_valuations = [history[max(0, len(history) - N + i)]["portfolio_valuation"] for i in range(N)]

    # 변동성 계산
    volatility = calculate_volatility(portfolio_valuations)

    # 현재 스텝 정보 추출
    current_portfolio_valuation = history[-1]["portfolio_valuation"]
    initial_portfolio_valuation = history[0]["portfolio_valuation"]
    current_step = history[-1]["step"]

    # 수익률 계산
    total_roe = current_portfolio_valuation / initial_portfolio_valuation - 1

    # 장기 실행 보상 및 변동성에 기반한 보상 조정
    long_term_bonus = max(1, current_step / 100)
    stability_bonus = 1 / (volatility + 0.01)  # 분모가 0이 되지 않도록

    # 손실이 발생했을 경우 음수 보상 부여
    if total_roe < 0:
        reward = total_roe * stability_bonus * long_term_bonus
    else:
        reward = max(0, total_roe) * stability_bonus * long_term_bonus

    return reward
