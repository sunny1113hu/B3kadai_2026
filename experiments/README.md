# Experiments（日本語）

このフォルダは「複数のアルゴリズムを一括で実行し、結果をまとめる」ための実験マネージャです。

## 1. クイックスタート
### CartPole: Q-learning / DQN / PER DQN
```bash
python experiments/compare.py --config experiments/configs/cartpole_compare_dqn3.json --mode all
```

### Atari: DQN / PER DQN（3ゲーム）
```bash
python experiments/compare.py --config experiments/configs/atari_compare_dqn_per_3games.json --mode all
```

## 2. 出力物
`experiments/outputs/<experiment_name>_<timestamp>/` に出力されます。
- `manifest.json`（実行内容）
- `summary.csv` / `summary.md` / `summary.json`
- `ep_r_comparison.png`（全体）
- `ep_r_<EnvName>.png`（環境別）
- `metrics/*.csv`（評価スコアの逐次ログ）
- `videos/<run_id>_S<seed>/`（動画）

## 2.5 ノートブックで再プロット
`experiments/notebooks/plot_metrics.ipynb` を用意しました。  
`STEP_BIN` を変えることで、**プロット点の間隔**（ステップ幅）を自由に調整できます。  
`SMOOTH_WINDOW` を増やすと移動平均で滑らかになります。

高解像度のCSVが欲しい場合は、実行前に `eval_interval` を小さくしてください。

### GUIモード
`ipywidgets` が入っている場合、ドロップダウンから run を選べます。  
全環境を1枚にまとめたい場合は **Merge ENVS** をオンにしてください。

依存の追加（未導入なら）:
```bash
uv add ipywidgets
uv sync
```

### 手動モード
GUIが使えない場合は、`OUTPUT_DIR` を直接指定して使えます。

## 3. 動画について
`record_video` を true にすると動画を保存します。
保存先:
```
experiments/outputs/<experiment_name>_<timestamp>/videos/<run_id>_S<seed>/
```

必要な依存:
```bash
uv add imageio imageio-ffmpeg moviepy
```
Linux では `ffmpeg` も必要です。

## 4. 設定ファイル（config）の編集
`experiments/configs/*.json` / `*.yaml` でパラメータを管理します。  
YAML ではコメントを書けるので、メモ用途にも便利です。

### 主要パラメータ
- `max_train_steps`: 学習ステップ
- `eval_interval`: 評価の間隔
- `eval_turns`: 評価エピソード数
- `random_steps`: 学習前のランダム探索
- `buffersize`: リプレイバッファ
- `target_freq`: ターゲット更新間隔
- `gamma`: 割引率
- `batch_size`: バッチサイズ
- `init_e` / `final_e` / `anneal_frac`: ε-greedyスケジュール
- `device`: `cuda` or `cpu`

### PER専用
- `PER`, `per_alpha`, `per_beta_init`, `per_beta_gain_steps`, `per_eps`, `per_replacement`

### 学習率スケジューラ（線形）
- `lr_init`: 初期学習率
- `lr_end`: 最終学習率
- `lr_decay_steps`: 線形減衰のステップ数

## 5. 実行例
```bash
python experiments/compare.py --config experiments/configs/atari_compare_dqn_per_3games.json --mode all
```

## 6. Docker (GPU)
GPU対応のDocker環境が用意されています。
```bash
docker compose build
docker compose run --rm -it drl
```

コンテナ内で実験を実行してください。

## 7. 報酬について
- 基本は **環境のデフォルト報酬**
- Atariは **学習時に報酬クリッピング（-1/0/+1）** を使用
- 評価時はクリップなし（スコアは本来の値）

必要ならクリップ有無を設定で切り替えるように変更できます。
