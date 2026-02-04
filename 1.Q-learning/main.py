from Q_learning import QLearningAgent, evaluate_policy
from torch.utils.tensorboard import SummaryWriter  # used to plot training curve
from gymnasium.wrappers import TimeLimit
from datetime import datetime
import argparse
import csv
import gymnasium as gym
import numpy as np
import os
import shutil


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def build_env(env_name, max_episode_steps, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def make_discretizer(obs_space, bins, low, high):
    bins = np.asarray(bins, dtype=int)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if bins.ndim != 1:
        raise ValueError('bins must be a 1D list')
    if low.shape != high.shape or low.shape != bins.shape:
        raise ValueError('bins/low/high must have the same length')

    denom = np.where(high > low, high - low, 1.0)

    def discretize(obs):
        obs = np.asarray(obs, dtype=float)
        clipped = np.clip(obs, low, high)
        ratios = (clipped - low) / denom
        indices = (ratios * bins).astype(int)
        indices = np.minimum(indices, bins - 1)
        return np.ravel_multi_index(indices, bins)

    return discretize, int(np.prod(bins))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CliffWalking-v0', help='Gymnasium env name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max_train_steps', type=int, default=20000, help='Max training steps')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluate every N steps')
    parser.add_argument('--save_interval', type=int, default=0, help='Save every N steps (0 means only at end)')
    parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record training curve')
    parser.add_argument('--load_model', type=str2bool, default=False, help='Load model or not')
    parser.add_argument('--log_root', type=str, default='runs', help='Root dir for tensorboard logs')
    parser.add_argument('--run_name', type=str, default='', help='Optional run name for logs/models')
    parser.add_argument('--model_dir', type=str, default='model', help='Directory to save the Q table')
    parser.add_argument('--model_path', type=str, default='', help='Full path to save/load Q table')
    parser.add_argument('--lr', type=float, default=0.2, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon-greedy noise')
    parser.add_argument('--max_episode_steps', type=int, default=500, help='TimeLimit max steps (train)')
    parser.add_argument('--eval_max_episode_steps', type=int, default=100, help='TimeLimit max steps (eval)')
    parser.add_argument('--bins', type=int, nargs='*', default=None, help='Bins per dimension for Box obs')
    parser.add_argument('--low', type=float, nargs='*', default=None, help='Lower bounds for Box obs')
    parser.add_argument('--high', type=float, nargs='*', default=None, help='Upper bounds for Box obs')
    parser.add_argument('--eval_turns', type=int, default=1, help='Number of eval episodes to average')
    parser.add_argument('--record_video', type=str2bool, default=False, help='Record evaluation videos')
    parser.add_argument('--video_dir', type=str, default='videos', help='Directory to save videos')
    parser.add_argument('--video_episodes', type=int, default=1, help='Number of episodes to record')
    parser.add_argument('--video_prefix', type=str, default='', help='Prefix for video filenames')
    parser.add_argument('--csv_log', type=str, default='', help='CSV path to log eval scores')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    print(f"Random Seed: {args.seed}")

    ''' ↓↓↓ Build Env ↓↓↓ '''
    env = build_env(args.env, args.max_episode_steps)
    eval_env = build_env(args.env, args.eval_max_episode_steps)

    state_encoder = None
    if isinstance(env.observation_space, gym.spaces.Box):
        if args.bins is None or args.low is None or args.high is None:
            raise ValueError('bins/low/high are required for Box observation spaces')
        state_encoder, s_dim = make_discretizer(env.observation_space, args.bins, args.low, args.high)
    else:
        s_dim = env.observation_space.n

    ''' ↓↓↓ Use tensorboard to record training curves ↓↓↓ '''
    if args.write:
        timenow = str(datetime.now())[0:-7]
        timenow = ' ' + timenow[0:13] + '_' + timenow[14:16] + '_' + timenow[-2::]
        run_name = args.run_name if args.run_name else f'{args.env}{timenow}'
        writepath = os.path.join(args.log_root, run_name)
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    ''' ↓↓↓ Build Q-learning Agent ↓↓↓ '''
    os.makedirs(args.model_dir, exist_ok=True)
    if args.model_path:
        model_path = args.model_path
    elif args.run_name:
        safe_name = args.run_name.replace(' ', '_')
        model_path = os.path.join(args.model_dir, f'q_table_{safe_name}.npy')
    else:
        model_path = os.path.join(args.model_dir, 'q_table.npy')

    agent = QLearningAgent(
        s_dim=s_dim,
        a_dim=env.action_space.n,
        lr=args.lr,
        gamma=args.gamma,
        exp_noise=args.epsilon)
    if args.load_model:
        agent.restore(model_path)

    ''' ↓↓↓ Iterate and Train ↓↓↓ '''
    total_steps = 0
    seed = args.seed
    save_interval = args.save_interval if args.save_interval > 0 else args.max_train_steps
    if args.csv_log:
        os.makedirs(os.path.dirname(args.csv_log), exist_ok=True)
        if not os.path.exists(args.csv_log):
            with open(args.csv_log, "w", newline="", encoding="utf-8") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["time", "step", "episode_reward", "seed", "env"])
    while total_steps < args.max_train_steps:
        s, info = env.reset(seed=seed)
        seed += 1
        done = False

        while not done:
            if state_encoder is not None:
                s_id = state_encoder(s)
            else:
                s_id = s
            a = agent.select_action(s_id, deterministic=False)
            s_next, r, dw, tr, info = env.step(a)
            if state_encoder is not None:
                s_next_id = state_encoder(s_next)
            else:
                s_next_id = s_next
            agent.train(s_id, a, r, s_next_id, dw)

            done = (dw or tr)
            s = s_next

            total_steps += 1
            '''record & log'''
            if total_steps % args.eval_interval == 0:
                ep_r = evaluate_policy(eval_env, agent, state_encoder=state_encoder, turns=args.eval_turns)
                if args.write:
                    writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                print(f'EnvName:{args.env}, Seed:{seed}, Steps:{total_steps}, Episode reward:{ep_r}')
                if args.csv_log:
                    with open(args.csv_log, "a", newline="", encoding="utf-8") as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([datetime.now().isoformat(timespec="seconds"), total_steps, ep_r, seed, args.env])

            '''save model'''
            if total_steps % save_interval == 0:
                agent.save(model_path)

    if args.record_video:
        from gymnasium.wrappers import RecordVideo
        prefix = args.video_prefix or args.run_name or "q_learning"
        if os.path.exists(args.video_dir):
            shutil.rmtree(args.video_dir)
        video_env = build_env(args.env, args.eval_max_episode_steps, render_mode="rgb_array")
        if hasattr(video_env, "metadata") and video_env.metadata.get("render_fps") is None:
            video_env.metadata["render_fps"] = 30
        video_env = RecordVideo(
            video_env,
            video_folder=args.video_dir,
            name_prefix=prefix,
            episode_trigger=lambda episode_id: True,
            disable_logger=True,
        )
        evaluate_policy(video_env, agent, state_encoder=state_encoder, turns=args.video_episodes)
        video_env.close()

    env.close()
    eval_env.close()
    if args.write:
        writer.close()


if __name__ == '__main__':
    main()
