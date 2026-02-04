<div align=center>
<img src="https://github.com/XinJingHao/RL-Algorithms-by-Pytorch/blob/main/RL_PYTORCH.png" width=500 />
</div>

<div align=center>
PyTorchで主要な深層強化学習（DRL）アルゴリズムを統一的に実装したリポジトリ
</div>

<div align=center>
  <img src="https://img.shields.io/badge/Python-blue" />
  <img src="https://img.shields.io/badge/DRL-blueviolet" />
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/Pytorch-ff69b4" /></a>
  <a href="https://deepwiki.com/XinJingHao/DRL-Pytorch"><img src="https://img.shields.io/badge/DeepWiki-FF6347" /></a>
</div>

<br/>
<br/>

## 0. Star履歴

<div align="left">
<img width="70%" height="auto" src="https://api.star-history.com/svg?repos=XinJingHao/Deep-Reinforcement-Learning-Algorithms-with-Pytorch&type=Date">
</div>
<br/>

## 1. 依存関係
このリポジトリは以下の依存を前提としています（必要に応じて各フォルダのREADMEも確認してください）。
```python
gymnasium==0.29.1
numpy==1.26.1
pytorch==2.1.0

python==3.11.5
```

<br/>

## 2. 使い方（基本）
使用したいアルゴリズムのフォルダに入って `main.py` を実行します。
```bash
python main.py
```
より詳細な設定は各アルゴリズムフォルダの `README.md` を参照してください。

<br/>

## 3. アルゴリズム一覧（個別リポジトリ）
+ [1.Q-learning](https://github.com/XinJingHao/Q-learning)
+ [2.1Duel Double DQN](https://github.com/XinJingHao/Duel-Double-DQN-Pytorch)
+ [2.2Noisy Duel DDQN on Atari Game](https://github.com/XinJingHao/Noisy-Duel-DDQN-Atari-Pytorch)
+ [2.3Prioritized Experience Replay(PER) DQN/DDQN](https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch)
+ [2.4Categorical DQN (C51)](https://github.com/XinJingHao/C51-Categorical-DQN-Pytorch)
+ [2.5NoisyNet DQN](https://github.com/XinJingHao/NoisyNet-DQN-Pytorch)
+ [3.1Proximal Policy Optimization(PPO) for Discrete Action Space](https://github.com/XinJingHao/PPO-Discrete-Pytorch)
+ [3.2Proximal Policy Optimization(PPO) for Continuous Action Space](https://github.com/XinJingHao/PPO-Continuous-Pytorch)
+ [4.1Deep Deternimistic Policy Gradient(DDPG)](https://github.com/XinJingHao/DDPG-Pytorch)
+ [4.2Twin Delayed Deep Deterministic Policy Gradient(TD3)](https://github.com/XinJingHao/TD3-Pytorch)
+ [5.1Soft Actor Critic(SAC) for Discrete Action Space](https://github.com/XinJingHao/SAC-Discrete-Pytorch)
+ [5.2Soft Actor Critic(SAC) for Continuous Action Space](https://github.com/XinJingHao/SAC-Continuous-Pytorch)
+ [6.Actor-Sharer-Learner(ASL)](https://github.com/XinJingHao/Actor-Sharer-Learner)

<br/>

## 4. 実験管理（追加機能）
このプロジェクトには「複数手法を一括で実行し、結果をまとめる実験マネージャ」が追加されています。

- 実験スクリプト: `experiments/compare.py`
- 設定ファイル: `experiments/configs/*.json`
- 出力先: `experiments/outputs/<experiment_name>_<timestamp>/`

詳細は `experiments/README.md` を参照してください（日本語でまとめ済み）。

<br/>

## 5. 可視化と出力
- TensorBoardログ: `runs/` 配下
- 集計結果:
  - `summary.csv`
  - `summary.md`（日本語レポート）
  - `summary.json`
- グラフ:
  - `ep_r_comparison.png`（全体）
  - `ep_r_<EnvName>.png`（環境別）
- 動画（RecordVideo）:
  - `experiments/outputs/<experiment_name>_<timestamp>/videos/<run_id>_S<seed>/`

<br/>

## 6. Docker（GPU）
GPU対応のDocker環境が用意されています。
```bash
docker compose build
docker compose run --rm -it drl
```
コンテナ内で実験を実行してください。

<br/>

## 7. DRLの参考資料
### 7.1 シミュレーション環境
+ [gym](https://www.gymlibrary.dev/) / [gymnasium](https://gymnasium.farama.org/)
+ [Isaac Sim](https://developer.nvidia.com/isaac/sim#isaac-lab)
+ [Sparrow](https://github.com/XinJingHao/Sparrow-V2)
+ [ROS](https://www.ros.org/)
+ [Webots](https://cyberbotics.com/)
+ [Envpool](https://envpool.readthedocs.io/en/latest/index.html)
+ [Other Popular Envs](https://github.com/clvrai/awesome-rl-envs)

### 7.2 書籍
+ [《Reinforcement learning: An introduction》](https://books.google.com.sg/books?hl=zh-CN&lr=&id=uWV0DwAAQBAJ&oi=fnd&pg=PR7&dq=Reinforcement+Learning&ots=mivIu01Xp6&sig=zQ6jkZRxJop4fkAgScMgzULGlbY&redir_esc=y#v=onepage&q&f=false)--Richard S. Sutton
+ 《深度学习入门：基于Python的理论と実装》--斋藤康毅

### 7.3 オンライン講座
+ [RL Courses(bilibili)](https://www.bilibili.com/video/BV1UE411G78S?p=1&vd_source=df4b7370976f5ca5034cc18488eec368)--李宏毅(Hongyi Li)
+ [RL Courses(Youtube)](https://www.youtube.com/watch?v=z95ZYgPgXOY&list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_)--李宏毅(Hongyi Li)
+ [UCL Course on RL](https://www.davidsilver.uk/teaching/)--David Silver
+ [动手强化学习](https://hrl.boyuai.com/chapter/1/%E5%88%9D%E6%8E%A2%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0)--上海交通大学
+ [DRL Courses](https://github.com/wangshusen/DRL)--Shusen Wang

### 7.4 ブログ
+ [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
+ [Policy Gradient Theorem --Cangxi](https://zhuanlan.zhihu.com/p/491647161)
+ [Policy Gradient Algorithms --Lilian](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
+ [Theorem of PPO](https://zhuanlan.zhihu.com/p/563166533)
+ [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
+ [Prioritized Experience Replay](https://zhuanlan.zhihu.com/p/631171588)
+ [Soft Actor Critic](https://zhuanlan.zhihu.com/p/566722896)
+ [A (Long) Peek into Reinforcement Learning --Lilian](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)
+ [Introduction to TD3](https://zhuanlan.zhihu.com/p/409536699)

<br/>

## 8. 重要論文
DQN: [Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. nature, 2015, 518(7540): 529-533.](https://www.nature.com/articles/nature14236/?source=post_page)

Double DQN: [Van Hasselt H, Guez A, Silver D. Deep reinforcement learning with double q-learning[C]//Proceedings of the AAAI conference on artificial intelligence. 2016, 30(1).](https://ojs.aaai.org/index.php/AAAI/article/view/10295)

Duel DQN: [Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." International conference on machine learning. PMLR, 2016.](https://proceedings.mlr.press/v48/wangf16.pdf)

PER: [Schaul T, Quan J, Antonoglou I, et al. Prioritized experience replay[J]. arXiv preprint arXiv:1511.05952, 2015.](https://arxiv.org/abs/1511.05952)

C51: [Bellemare M G, Dabney W, Munos R. A distributional perspective on reinforcement learning[C]//International conference on machine learning. PMLR, 2017: 449-458.](https://proceedings.mlr.press/v70/bellemare17a/bellemare17a.pdf)

NoisyNet DQN: [Fortunato M, Azar M G, Piot B, et al. Noisy networks for exploration[J]. arXiv preprint arXiv:1706.10295, 2017.](https://arxiv.org/abs/1706.10295)

PPO: [Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms[J]. arXiv preprint arXiv:1707.06347, 2017.](https://arxiv.org/pdf/1707.06347.pdf)

DDPG: [Lillicrap T P, Hunt J J, Pritzel A, et al. Continuous control with deep reinforcement learning[J]. arXiv preprint arXiv:1509.02971, 2015.](https://arxiv.org/abs/1509.02971)

TD3: [Fujimoto S, Hoof H, Meger D. Addressing function approximation error in actor-critic methods[C]//International conference on machine learning. PMLR, 2018: 1587-1596.](https://proceedings.mlr.press/v80/fujimoto18a.html)

SAC: [Haarnoja T, Zhou A, Abbeel P, et al. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor[C]//International conference on machine learning. PMLR, 2018: 1861-1870.](https://proceedings.mlr.press/v80/haarnoja18b)

ASL: [Train a Real-world Local Path Planner in One Hour via Partially Decoupled Reinforcement Learning and Vectorized Diversity](https://arxiv.org/abs/2305.04180)

ColorDynamic: [Generalizable, Scalable, Real-time, End-to-end Local Planner for Unstructured and Dynamic Environments](https://arxiv.org/abs/2502.19892)

<br/>

## 9. Citation
```bash
@misc{DRL-Pytorch,
  author = {Jinghao Xin},
  title = {DRL-Pytorch},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/XinJingHao/DRL-Pytorch}},
}
```
