"""Train DDPG for dynamic positioning using dp_env + custom reward.

Wiring follows reference/swimmer_ddpg.py.  Run from workspace root:

    python train_ddpg.py
"""
import copy
import datetime
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm import DDPG
from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorDeterministic, ContinuousCritic

from dp_env import VesselDPEnv, EnvConfig, ActionMaskWrapper
from my_reward import CustomRewardWrapper

# ── hyper-parameters (fixed) ────────────────────────────────────────────
SEED = 42
HIDDEN_SIZES = [256, 256]
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
EXPLORATION_NOISE = 0.05
START_TIMESTEPS = 10_000
EPOCH = 200
EPOCH_NUM_STEPS = 10_000       # 200 * 10_000 = 2,000,000 total
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 256
N_TRAIN_ENVS = 10
N_TEST_ENVS = 5
UPDATE_PER_STEP = 1
LOGDIR = "log/ddpg_dp"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_env(cfg):
    """Closure factory for SubprocVectorEnv (must be pickle-safe)."""
    c = copy.deepcopy(cfg)
    env = VesselDPEnv(c)
    env = ActionMaskWrapper(env, mode="legacy_4d_fixed_azimuth")
    env = CustomRewardWrapper(env)
    return env


def main():
    # ── config ──────────────────────────────────────────────────────────
    cfg = EnvConfig()
    print(f"Config: target=({cfg.task.target_north_m}, {cfg.task.target_east_m})"
          f" yaw={cfg.task.target_yaw_deg}°"
          f" radius=[{cfg.task.start_radius_min_m}, {cfg.task.start_radius_max_m}]m"
          f" timeout={cfg.task.timeout_s}s"
          f" current={cfg.ship.current_speed}m/s wind={cfg.ship.wind_speed}m/s")

    # ── vectorised envs ─────────────────────────────────────────────────
    train_envs = SubprocVectorEnv(
        [lambda: _make_env(cfg) for _ in range(N_TRAIN_ENVS)]
    )
    test_envs = SubprocVectorEnv(
        [lambda: _make_env(cfg) for _ in range(N_TEST_ENVS)]
    )
    env = _make_env(cfg)

    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    max_action = float(env.action_space.high[0])

    print(f"Obs shape: {state_shape}  Action shape: {action_shape}"
          f"  max_action: {max_action}")

    # ── seed ────────────────────────────────────────────────────────────
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_envs.seed(SEED)
    test_envs.seed(SEED + 1000)

    # ── actor ───────────────────────────────────────────────────────────
    net_a = Net(state_shape=state_shape, hidden_sizes=HIDDEN_SIZES)
    actor = ContinuousActorDeterministic(
        preprocess_net=net_a, action_shape=action_shape, max_action=max_action,
    ).to(DEVICE)
    actor_optim = AdamOptimizerFactory(lr=ACTOR_LR)

    # ── critic ──────────────────────────────────────────────────────────
    net_c = Net(
        state_shape=state_shape, action_shape=action_shape,
        hidden_sizes=HIDDEN_SIZES, concat=True,
    )
    critic = ContinuousCritic(preprocess_net=net_c).to(DEVICE)
    critic_optim = AdamOptimizerFactory(lr=CRITIC_LR)

    # ── policy + algorithm ──────────────────────────────────────────────
    policy = ContinuousDeterministicPolicy(
        actor=actor,
        exploration_noise=GaussianNoise(sigma=EXPLORATION_NOISE * max_action),
        action_space=env.action_space,
    )
    algorithm = DDPG(
        policy=policy,
        policy_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        tau=TAU,
        gamma=GAMMA,
    )

    # ── buffer + collectors ─────────────────────────────────────────────
    buffer = VectorReplayBuffer(BUFFER_SIZE, len(train_envs))
    train_collector = Collector[CollectStats](
        algorithm, train_envs, buffer, exploration_noise=True,
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)

    # ── random warmup ───────────────────────────────────────────────────
    print(f"Collecting {START_TIMESTEPS} random warmup steps ...")
    train_collector.reset()
    train_collector.collect(n_step=START_TIMESTEPS, random=True)
    print("Warmup done.")

    # ── logging ─────────────────────────────────────────────────────────
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_path = os.path.join(LOGDIR, f"seed{SEED}", now)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=1)

    def save_best_fn(pol):
        torch.save(pol.state_dict(), os.path.join(log_path, "policy.pth"))

    # ── train ───────────────────────────────────────────────────────────
    print(f"Training {EPOCH} epochs, {EPOCH_NUM_STEPS} steps/epoch"
          f"  → {EPOCH * EPOCH_NUM_STEPS:,} total steps  [ctrl-c to stop]\n")

    result = algorithm.run_training(
        OffPolicyTrainerParams(
            training_collector=train_collector,
            test_collector=test_collector,
            max_epochs=EPOCH,
            epoch_num_steps=EPOCH_NUM_STEPS,
            batch_size=BATCH_SIZE,
            update_step_num_gradient_steps_per_sample=UPDATE_PER_STEP,
            test_step_num_episodes=N_TEST_ENVS,
            logger=logger,
            save_best_fn=save_best_fn,
            test_in_training=False,
        )
    )

    print(f"\nDone.  Best policy: {log_path}/policy.pth")
    print(f"Result: {result}")
    print(f"TensorBoard: tensorboard --logdir {log_path}")


if __name__ == "__main__":
    main()
