import time
import numpy as np
import torch
import gymnasium as gym
from DQN import DQN_Agent
from LPRB import LightPriorReplayBuffer, device
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
from utils import evaluate_policy,str2bool, LinearSchedule
import csv

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=250, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(3e5), help='Max training steps')
parser.add_argument('--buffer_size', type=int, default=int(2e5), help='size of the replay buffer')
parser.add_argument('--save_interval', type=int, default=int(5e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(1e3), help='Model evaluating interval, in steps.')
parser.add_argument('--eval_turns', type=int, default=3, help='Number of evaluation episodes')
parser.add_argument('--warmup', type=int, default=int(3e3), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--lr_init', type=float, default=1.5e-4, help='Initial Learning rate')
parser.add_argument('--lr_end', type=float, default=6e-5, help='Final Learning rate')
parser.add_argument('--lr_decay_steps', type=float, default=int(3e5), help='Learning rate decay steps')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise_init', type=float, default=0.6, help='init explore noise')
parser.add_argument('--exp_noise_end', type=float, default=0.03, help='final explore noise')
parser.add_argument('--noise_decay_steps', type=int, default=int(1e5), help='decay steps of explore noise')
parser.add_argument('--DDQN', type=str2bool, default=True, help='True:DDQN; False:DQN')

parser.add_argument('--alpha', type=float, default=0.6, help='alpha for PER')
parser.add_argument('--beta_init', type=float, default=0.4, help='beta for PER')
parser.add_argument('--beta_gain_steps', type=int, default=int(3e5), help='steps of beta from beta_init to 1.0')
parser.add_argument('--replacement', type=str2bool, default=False, help='sample method')
parser.add_argument('--record_video', type=str2bool, default=False, help='Record evaluation videos')
parser.add_argument('--video_dir', type=str, default='videos', help='Directory to save videos')
parser.add_argument('--video_episodes', type=int, default=1, help='Number of episodes to record')
parser.add_argument('--video_prefix', type=str, default='', help='Prefix for video filenames')
parser.add_argument('--csv_log', type=str, default='', help='CSV path to log eval scores')
parser.add_argument('--eval_only', type=str2bool, default=False, help='Only evaluate a loaded model')
opt = parser.parse_args()
print(opt)


def main():
    EnvName = ['CartPole-v1','LunarLander-v2']
    BriefEnvName = ['CPV1', 'LLdV2']
    Env_With_DW = [True, True] #DW: Die or Win
    opt.env_with_dw = Env_With_DW[opt.EnvIdex]
    env = gym.make(EnvName[opt.EnvIdex])
    eval_env = gym.make(EnvName[opt.EnvIdex], render_mode = 'human' if opt.render else None)
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps

    #Use DDQN or DQN
    if opt.DDQN: algo_name = 'DDQN'
    else: algo_name = 'DQN'

    #Seed everything
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    print('Algorithm:',algo_name,'  Env:',BriefEnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,
          '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps, '\n')

    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/LightPrior{}_{}'.format(algo_name,BriefEnvName[opt.EnvIdex]) + timenow
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
    model = DQN_Agent(opt)
    if opt.Loadmodel: model.load(algo_name,BriefEnvName[opt.EnvIdex],opt.ModelIdex)
    buffer = LightPriorReplayBuffer(opt)

    exp_noise_scheduler = LinearSchedule(opt.noise_decay_steps, opt.exp_noise_init, opt.exp_noise_end)
    beta_scheduler = LinearSchedule(opt.beta_gain_steps, opt.beta_init, 1.0)
    lr_scheduler = LinearSchedule(opt.lr_decay_steps, opt.lr_init, opt.lr_end)

    if opt.eval_only:
        step_mark = opt.ModelIdex * 1000 if opt.Loadmodel else 0
        score = evaluate_policy(eval_env, model, turns=opt.eval_turns)
        print('EnvName:',BriefEnvName[opt.EnvIdex],'seed:',opt.seed,'steps:', step_mark,'score:', score)
        if opt.write:
            writer.add_scalar('ep_r', score, global_step=step_mark)
        if opt.csv_log:
            with open(opt.csv_log, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([datetime.now().isoformat(timespec="seconds"), step_mark, score, opt.seed, BriefEnvName[opt.EnvIdex], algo_name])
        if opt.record_video:
            from gymnasium.wrappers import RecordVideo
            prefix = opt.video_prefix or f"LightPrior{algo_name}_{BriefEnvName[opt.EnvIdex]}_S{opt.seed}"
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
            evaluate_policy(video_env, model, turns=opt.video_episodes)
            video_env.close()
        env.close()
        eval_env.close()
        if opt.write:
            writer.close()
        return

    if opt.render:
        score = evaluate_policy(eval_env, model, 20)
        print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset()
            a, q_a = model.select_action(s, deterministic=False)
            # ↑ cover s, a, q_a from last episode ↑

            while True:
                s_next, r, dw, tr, info = env.step(a) # dw: terminated; tr: truncated
                if r <= -100: r = -10  # good for LunarLander
                a_next, q_a_next = model.select_action(s_next, deterministic=False)

                # [s; a, q_a; r, dw, tr, s_next; a_next, q_a_next] have been all collected.
                priority = (torch.abs(r + (~dw)*opt.gamma*q_a_next - q_a) + 0.01)**opt.alpha #scalar
                buffer.add(s, a, r, dw, tr, priority)

                s, a, q_a = s_next, a_next, q_a_next

                '''update if its time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if total_steps >= opt.warmup and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        model.train(buffer)

                    # parameter annealing
                    model.exp_noise = exp_noise_scheduler.value(total_steps)
                    buffer.beta = beta_scheduler.value(total_steps)
                    for p in model.q_net_optimizer.param_groups: p['lr'] = lr_scheduler.value(total_steps)

                '''record & log'''
                if (total_steps) % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, model, turns=opt.eval_turns)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('noise', model.exp_noise, global_step=total_steps)
                        writer.add_scalar('beta', buffer.beta, global_step=total_steps)
                        # priorities = buffer.priorities[0: buffer.size].cpu()
                        # writer.add_scalar('p_max', priorities.max(), global_step=total_steps)
                        # writer.add_scalar('p_sum', priorities.sum(), global_step=total_steps)
                    print('EnvName:',BriefEnvName[opt.EnvIdex],'seed:',opt.seed,'steps:', total_steps,'score:', int(score))
                    if opt.csv_log:
                        with open(opt.csv_log, "a", newline="", encoding="utf-8") as f:
                            w = csv.writer(f)
                            w.writerow([datetime.now().isoformat(timespec="seconds"), total_steps, score, opt.seed, BriefEnvName[opt.EnvIdex], algo_name])

                total_steps += 1

                '''save model'''
                if (total_steps) % opt.save_interval == 0:
                    model.save(algo_name,BriefEnvName[opt.EnvIdex],int(total_steps/1000))

                if dw or tr: break

        if opt.record_video:
            from gymnasium.wrappers import RecordVideo
            prefix = opt.video_prefix or f"LightPrior{algo_name}_{BriefEnvName[opt.EnvIdex]}_S{opt.seed}"
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
            evaluate_policy(video_env, model, turns=opt.video_episodes)
            video_env.close()

    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()



