import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import Shape, deterministic_model

from preprocess import only_sub_indicators, refined_pnl

set_seed(42)

gym.register(
    id="MultiDatasetDiscretedTradingEnv",
    entry_point="environment:MultiDatasetDiscretedTradingEnv",
    disable_env_checker=True,
)

env = gym.make_vec(
    "MultiDatasetDiscretedTradingEnv",
    vectorization_mode="sync",
    num_envs=4,
    wrappers=[FlattenObservation],
    dataset_dir="./data/train/day/**/**/*.pkl",
    preprocess=only_sub_indicators,
    reward_function=refined_pnl,
    positions=[-5, -2, 0, 2, 5],
    trading_fees=0.0001,
    borrow_interest_rate=0.0003,
    portfolio_initial_value=100,
    max_episode_duration="max",  # 24 * 60,
    verbose=2,
    window_size=120,
)


env = wrap_env(env, wrapper="gymnasium")

device = env.device
memory = RandomMemory(
    memory_size=4096,
    num_envs=env.num_envs,
    device=device,
    replacement=False,
)


models = {}
models["q_network"] = deterministic_model(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
    clip_actions=False,
    input_shape=Shape.OBSERVATIONS,
    hiddens=[64, 64],
    hidden_activation=["relu", "relu"],
    output_shape=Shape.ACTIONS,
    output_activation=None,
    output_scale=1.0,
)
models["target_q_network"] = deterministic_model(
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
    clip_actions=False,
    input_shape=Shape.OBSERVATIONS,
    hiddens=[64, 64],
    hidden_activation=["relu", "relu"],
    output_shape=Shape.ACTIONS,
    output_activation=None,
    output_scale=1.0,
)


cfg = DQN_DEFAULT_CONFIG.copy()
cfg["learning_starts"] = 100
cfg["exploration"]["final_epsilon"] = 0.04
cfg["exploration"]["timesteps"] = 1500
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/torch/CartPole"


agent = DQN(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)


cfg_trainer = {"timesteps": 50000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

trainer.train()
