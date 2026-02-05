from utils import evaluate_policy, str2bool
from datetime import datetime
from DQN import DQN_agent
import gymnasium as gym
import os, shutil
import argparse
import torch
import csv


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')
parser.add_argument('--record_video', type=str2bool, default=False, help='Record evaluation videos')
parser.add_argument('--video_dir', type=str, default='videos', help='Directory to save videos')
parser.add_argument('--video_episodes', type=int, default=1, help='Number of episodes to record')
parser.add_argument('--video_prefix', type=str, default='', help='Prefix for video filenames')
parser.add_argument('--csv_log', type=str, default='', help='CSV path to log eval scores')
parser.add_argument('--eval_only', type=str2bool, default=False, help='Only evaluate a loaded model')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(1e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(50e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')
parser.add_argument('--eval_turns', type=int, default=3, help='Number of evaluation episodes')
parser.add_argument('--random_steps', type=int, default=int(3e3), help='steps for random policy to explore')
parser.add_argument('--buffer_size', type=int, default=int(1e6), help='replay buffer size')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lr_init', type=float, default=None, help='Initial learning rate')
parser.add_argument('--lr_end', type=float, default=None, help='Final learning rate')
parser.add_argument('--lr_decay_steps', type=int, default=0, help='Steps for linear LR decay (0 uses Max_train_steps)')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise (legacy)')
parser.add_argument('--exp_noise_init', type=float, default=None, help='Initial explore noise (linear schedule)')
parser.add_argument('--exp_noise_end', type=float, default=None, help='Final explore noise (linear schedule)')
parser.add_argument('--noise_decay_steps', type=int, default=0, help='Steps for linear explore noise decay (0 uses Max_train_steps)')
parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise (legacy)')
parser.add_argument('--Double', type=str2bool, default=True, help='Whether to use Double Q-learning')
parser.add_argument('--Duel', type=str2bool, default=True, help='Whether to use Duel networks')
opt = parser.parse_args()
if opt.lr_init is None:
    opt.lr_init = opt.lr
if opt.lr_end is None:
    opt.lr_end = opt.lr_init
if opt.lr_decay_steps <= 0:
    opt.lr_decay_steps = opt.Max_train_steps
if opt.exp_noise_init is None:
    opt.exp_noise_init = opt.exp_noise
if opt.exp_noise_end is None:
    opt.exp_noise_end = opt.exp_noise_init
if opt.noise_decay_steps <= 0:
    opt.noise_decay_steps = opt.Max_train_steps
opt.exp_noise = opt.exp_noise_init
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


def linear_schedule(t, t_max, v0, v1):
    fraction = min(float(t) / t_max, 1.0)
    return v0 + fraction * (v1 - v0)


def main():
    EnvName = ['CartPole-v1','LunarLander-v2']
    BriefEnvName = ['CPV1', 'LLdV2']
    env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps

    #Algorithm Setting
    if opt.Duel: algo_name = 'Duel'
    else: algo_name = ''
    if opt.Double: algo_name += 'DDQN'
    else: algo_name += 'DQN'

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Algorithm:',algo_name,'  Env:',BriefEnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,
          '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps, '\n')

    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}-{}_S{}_'.format(algo_name,BriefEnvName[opt.EnvIdex],opt.seed) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    if opt.csv_log:
        os.makedirs(os.path.dirname(opt.csv_log), exist_ok=True)
        if not os.path.exists(opt.csv_log):
            with open(opt.csv_log, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["time", "step", "score", "seed", "env", "algo"])

    #Build model and replay buffer
    if not os.path.exists('model'): os.mkdir('model')
    agent = DQN_agent(**vars(opt))
    if opt.Loadmodel: agent.load(algo_name,BriefEnvName[opt.EnvIdex],opt.ModelIdex)

    if opt.eval_only:
        step_mark = opt.ModelIdex * 1000 if opt.Loadmodel else 0
        score = evaluate_policy(eval_env, agent, turns=opt.eval_turns)
        print('EnvName:',BriefEnvName[opt.EnvIdex],'seed:',opt.seed,'steps:', step_mark,'score:', int(score))
        if opt.write:
            writer.add_scalar('ep_r', score, global_step=step_mark)
        if opt.csv_log:
            with open(opt.csv_log, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([datetime.now().isoformat(timespec="seconds"), step_mark, score, opt.seed, BriefEnvName[opt.EnvIdex], algo_name])
        if opt.record_video:
            from gymnasium.wrappers import RecordVideo
            prefix = opt.video_prefix or f"{algo_name}_{BriefEnvName[opt.EnvIdex]}_S{opt.seed}"
            if os.path.exists(opt.video_dir):
                shutil.rmtree(opt.video_dir)
            video_env = gym.make(EnvName[opt.EnvIdex], render_mode="rgb_array")
            if hasattr(video_env, "metadata") and video_env.metadata.get("render_fps") is None:
                video_env.metadata["render_fps"] = 30
            video_env = RecordVideo(
                video_env,
                video_folder=opt.video_dir,
                name_prefix=prefix,
                episode_trigger=lambda episode_id: True,
                disable_logger=True,
            )
            evaluate_policy(video_env, agent, turns=opt.video_episodes)
            video_env.close()
        env.close()
        eval_env.close()
        if opt.write:
            writer.close()
        return

    if opt.render:
        while True:
            score = evaluate_policy(env, agent, 1)
            print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed) # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False

            '''Interact & trian'''
            while not done:
                #e-greedy exploration
                if total_steps < opt.random_steps: a = env.action_space.sample()
                else: a = agent.select_action(s, deterministic=False)
                s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
                done = (dw or tr)

                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next

                '''Update'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                lr = None
                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every): agent.train()
                    lr = linear_schedule(total_steps, opt.lr_decay_steps, opt.lr_init, opt.lr_end)
                    for p in agent.q_net_optimizer.param_groups:
                        p['lr'] = lr
                    agent.exp_noise = linear_schedule(total_steps, opt.noise_decay_steps, opt.exp_noise_init, opt.exp_noise_end)

                '''Noise decay & Record & Log'''
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, turns = opt.eval_turns)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('noise', agent.exp_noise, global_step=total_steps)
                        if lr is not None:
                            writer.add_scalar('lr', lr, global_step=total_steps)
                    print('EnvName:',BriefEnvName[opt.EnvIdex],'seed:',opt.seed,'steps:', total_steps,'score:', int(score))
                    if opt.csv_log:
                        with open(opt.csv_log, "a", newline="", encoding="utf-8") as f:
                            w = csv.writer(f)
                            w.writerow([datetime.now().isoformat(timespec="seconds"), total_steps, score, opt.seed, BriefEnvName[opt.EnvIdex], algo_name])
                total_steps += 1

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    agent.save(algo_name,BriefEnvName[opt.EnvIdex],int(total_steps/1000))
        if opt.record_video:
            from gymnasium.wrappers import RecordVideo
            prefix = opt.video_prefix or f"{algo_name}_{BriefEnvName[opt.EnvIdex]}_S{opt.seed}"
            if os.path.exists(opt.video_dir):
                shutil.rmtree(opt.video_dir)
            video_env = gym.make(EnvName[opt.EnvIdex], render_mode="rgb_array")
            if hasattr(video_env, "metadata") and video_env.metadata.get("render_fps") is None:
                video_env.metadata["render_fps"] = 30
            video_env = RecordVideo(
                video_env,
                video_folder=opt.video_dir,
                name_prefix=prefix,
                episode_trigger=lambda episode_id: True,
                disable_logger=True,
            )
            evaluate_policy(video_env, agent, turns=opt.video_episodes)
            video_env.close()
    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()
