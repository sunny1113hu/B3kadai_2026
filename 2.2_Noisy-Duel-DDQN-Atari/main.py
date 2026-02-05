import numpy as np
import torch
from Agent import DeepQ_Agent, ReplayBuffer_torch, PrioritizedReplayBuffer_torch
import os, shutil
from datetime import datetime
import argparse
from utils import evaluate_policy, str2bool, LinearSchedule
from tianshou_wrappers import make_env_tianshou
from AtariNames import Name
import csv

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='running device of algorithm: cuda or cpu')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=900, help='which model to load')
parser.add_argument('--record_video', type=str2bool, default=False, help='Record evaluation videos')
parser.add_argument('--video_dir', type=str, default='videos', help='Directory to save videos')
parser.add_argument('--video_episodes', type=int, default=1, help='Number of episodes to record')
parser.add_argument('--video_prefix', type=str, default='', help='Prefix for video filenames')
parser.add_argument('--csv_log', type=str, default='', help='CSV path to log eval scores')
parser.add_argument('--eval_only', type=str2bool, default=False, help='Only evaluate a loaded model')

parser.add_argument('--Max_train_steps', type=int, default=int(1E6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1E5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')
parser.add_argument('--eval_turns', type=int, default=3, help='Number of evaluation episodes')
parser.add_argument('--random_steps', type=int, default=int(1e4), help='random steps before training, 5E4 in DQN Nature')
parser.add_argument('--buffersize', type=int, default=int(1e4), help='Size of the replay buffer')
parser.add_argument('--target_freq', type=int, default=int(1E3), help='frequency of target net updating')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (legacy)')
parser.add_argument('--lr_init', type=float, default=None, help='Initial learning rate')
parser.add_argument('--lr_end', type=float, default=None, help='Final learning rate')
parser.add_argument('--lr_decay_steps', type=int, default=0, help='Steps for linear LR decay (0 uses Max_train_steps)')
parser.add_argument('--batch_size', type=int, default=32, help='lenth of sliced trajectory')
parser.add_argument('--init_e', type=float, default=1.0, help='Initial e-greedy noise')
parser.add_argument('--anneal_frac', type=int, default=3e5, help='annealing fraction of e-greedy noise')
parser.add_argument('--final_e', type=float, default=0.02, help='Final e-greedy noise')
parser.add_argument('--noop_reset', type=str2bool, default=False, help='use NoopResetEnv or not')
parser.add_argument('--huber_loss', type=str2bool, default=True, help='True: use huber_loss; False:use mse_loss')
parser.add_argument('--fc_width', type=int, default=200, help='number of units in Fully Connected layer')
parser.add_argument('--PER', type=str2bool, default=False, help='use Prioritized Experience Replay')
parser.add_argument('--per_alpha', type=float, default=0.6, help='PER alpha')
parser.add_argument('--per_beta_init', type=float, default=0.4, help='PER beta initial')
parser.add_argument('--per_beta_gain_steps', type=int, default=int(3e5), help='steps for beta to reach 1.0')
parser.add_argument('--per_eps', type=float, default=1e-6, help='PER epsilon')
parser.add_argument('--per_replacement', type=str2bool, default=False, help='sample method for PER')

parser.add_argument('--EnvIdex', type=int, default=37, help='Index of the Env; 20=Enduro; 37=Pong')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--Double', type=str2bool, default=False, help="whether to use Double Q-learning")
parser.add_argument('--Duel', type=str2bool, default=False, help="whether to use Duel. Q-learning")
parser.add_argument('--Noisy', type=str2bool, default=False, help="whether to use NoisyNet")
opt = parser.parse_args()
if opt.lr_init is None:
    opt.lr_init = opt.lr
if opt.lr_end is None:
    opt.lr_end = opt.lr_init
if opt.lr_decay_steps <= 0:
    opt.lr_decay_steps = opt.Max_train_steps
opt.dvc = torch.device(opt.device)
opt.algo_name = ('Double-' if opt.Double else '') + ('Duel-' if opt.Duel else '') + ('Noisy-' if opt.Noisy else '') + 'DQN'
opt.EnvName = Name[opt.EnvIdex] + "NoFrameskip-v4"
opt.ExperimentName = opt.algo_name + '_' + opt.EnvName
print(opt)

def main():
    # Seed Everything
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build evaluation env
    render_mode = 'human' if opt.render else None
    eval_env = make_env_tianshou(opt.EnvName, noop_reset=opt.noop_reset, episode_life=False, clip_rewards=False, render_mode=render_mode)
    opt.action_dim = eval_env.action_space.n
    print('Algorithm:',opt.algo_name,'  Env:',opt.EnvName,'  Action_dim:',opt.action_dim,'  Seed:',opt.seed, '\n')

    #Build Agent
    if not os.path.exists('model'): os.mkdir('model')
    agent = DeepQ_Agent(opt)
    if opt.Loadmodel: agent.load(opt.ExperimentName,opt.ModelIdex)

    if opt.eval_only:
        if opt.write:
            from torch.utils.tensorboard import SummaryWriter
            timenow = str(datetime.now())[0:-7]
            timenow = ' ' + timenow[0:13] + '_' + timenow[14:16] + '_' + timenow[-2::]
            writepath = f'runs/{opt.ExperimentName}_S{opt.seed}' + timenow
            if os.path.exists(writepath): shutil.rmtree(writepath)
            writer = SummaryWriter(log_dir=writepath)
        if opt.csv_log:
            os.makedirs(os.path.dirname(opt.csv_log), exist_ok=True)
            if not os.path.exists(opt.csv_log):
                with open(opt.csv_log, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["time", "step", "score", "seed", "env", "algo"])

        step_mark = opt.ModelIdex * 1000 if opt.Loadmodel else 0
        score = evaluate_policy(eval_env, agent, seed=opt.seed, turns=opt.eval_turns)
        print(f"{opt.ExperimentName}, Seed:{opt.seed}, Step:{step_mark}, Score:{score}")
        if opt.write:
            writer.add_scalar('ep_r', score, global_step=step_mark)
        if opt.csv_log:
            with open(opt.csv_log, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([datetime.now().isoformat(timespec="seconds"), step_mark, score, opt.seed, opt.EnvName, opt.algo_name])

        if opt.record_video:
            from gymnasium.wrappers import RecordVideo
            prefix = opt.video_prefix or f"{opt.ExperimentName}_S{opt.seed}"
            if os.path.exists(opt.video_dir):
                shutil.rmtree(opt.video_dir)
            video_env = make_env_tianshou(
                opt.EnvName,
                noop_reset=opt.noop_reset,
                episode_life=False,
                clip_rewards=False,
                render_mode="rgb_array",
            )
            if hasattr(video_env, "metadata") and video_env.metadata.get("render_fps") is None:
                video_env.metadata["render_fps"] = 30
            video_env = RecordVideo(
                video_env,
                video_folder=opt.video_dir,
                name_prefix=prefix,
                episode_trigger=lambda episode_id: True,
                disable_logger=True,
            )
            evaluate_policy(video_env, agent, seed=opt.seed + 999, turns=opt.video_episodes)
            video_env.close()

        eval_env.close()
        if opt.write:
            writer.close()
        return

    if opt.render:
        while True:
            score = evaluate_policy(eval_env, agent, seed=opt.seed, turns=1)
            print(opt.ExperimentName, 'seed:', opt.seed, 'score:', score)
    else:
        if opt.write:
            from torch.utils.tensorboard import SummaryWriter
            timenow = str(datetime.now())[0:-7]
            timenow = ' ' + timenow[0:13] + '_' + timenow[14:16] + '_' + timenow[-2::]
            writepath = f'runs/{opt.ExperimentName}_S{opt.seed}' + timenow
            if os.path.exists(writepath): shutil.rmtree(writepath)
            writer = SummaryWriter(log_dir=writepath)
        if opt.csv_log:
            os.makedirs(os.path.dirname(opt.csv_log), exist_ok=True)
            if not os.path.exists(opt.csv_log):
                with open(opt.csv_log, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["time", "step", "score", "seed", "env", "algo"])

        # Build replay buffer and training env
        if opt.PER:
            buffer = PrioritizedReplayBuffer_torch(
                device=opt.dvc,
                max_size=opt.buffersize,
                alpha=opt.per_alpha,
                beta=opt.per_beta_init,
                eps=opt.per_eps,
                replacement=opt.per_replacement,
            )
        else:
            buffer = ReplayBuffer_torch(device=opt.dvc, max_size=opt.buffersize)
        env = make_env_tianshou(opt.EnvName, noop_reset = opt.noop_reset)

        #explore noise linearly annealed from 1.0 to 0.02 within anneal_frac steps
        schedualer = LinearSchedule(schedule_timesteps=opt.anneal_frac, final_p=opt.final_e, initial_p=opt.init_e)
        agent.exp_noise = opt.init_e
        lr_schedule = LinearSchedule(schedule_timesteps=opt.lr_decay_steps, final_p=opt.lr_end, initial_p=opt.lr_init)
        seed = opt.seed
        if opt.PER:
            beta_schedule = LinearSchedule(schedule_timesteps=opt.per_beta_gain_steps, final_p=1.0, initial_p=opt.per_beta_init)

        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=seed)
            seed += 1 # 每次reset都使用新的seed,防止overfitting
            done = False
            while not done:
                a = agent.select_action(s, evaluate=False)
                s_next, r, dw, tr, info = env.step(a) # dw(dead & win): terminated; tr: truncated
                buffer.add(s, a, r, s_next, dw)
                done = dw + tr
                s = s_next

                # train, e-decay, log, save
                if buffer.size >= opt.random_steps:
                    agent.train(buffer)
                    for p in agent.q_net_optimizer.param_groups:
                        p['lr'] = lr_schedule.value(total_steps)

                    '''record & log'''
                    if total_steps % opt.eval_interval == 0:
                        score = evaluate_policy(eval_env, agent, seed=seed+1, turns=opt.eval_turns) # 不与当前训练seed重合，更general
                        if opt.write:
                            writer.add_scalar('ep_r', score, global_step=total_steps)
                            writer.add_scalar('noise', agent.exp_noise, global_step=total_steps)
                        print(f"{opt.ExperimentName}, Seed:{opt.seed}, Step:{total_steps}, Score:{score}")
                        if opt.csv_log:
                            with open(opt.csv_log, "a", newline="", encoding="utf-8") as f:
                                w = csv.writer(f)
                                w.writerow([datetime.now().isoformat(timespec="seconds"), total_steps, score, opt.seed, opt.EnvName, opt.algo_name])
                        agent.exp_noise = schedualer.value(total_steps) # e-greedy decay
                        if opt.PER:
                            buffer.beta = beta_schedule.value(total_steps)

                    total_steps += 1
                    '''save model'''
                    if total_steps % opt.save_interval == 0:
                        agent.save(opt.ExperimentName,int(total_steps/1000))
        if opt.record_video:
            from gymnasium.wrappers import RecordVideo
            prefix = opt.video_prefix or f"{opt.ExperimentName}_S{opt.seed}"
            if os.path.exists(opt.video_dir):
                shutil.rmtree(opt.video_dir)
            video_env = make_env_tianshou(
                opt.EnvName,
                noop_reset=opt.noop_reset,
                episode_life=False,
                clip_rewards=False,
                render_mode="rgb_array",
            )
            if hasattr(video_env, "metadata") and video_env.metadata.get("render_fps") is None:
                video_env.metadata["render_fps"] = 30
            video_env = RecordVideo(
                video_env,
                video_folder=opt.video_dir,
                name_prefix=prefix,
                episode_trigger=lambda episode_id: True,
                disable_logger=True,
            )
            evaluate_policy(video_env, agent, seed=opt.seed + 999, turns=opt.video_episodes)
            video_env.close()

    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()
