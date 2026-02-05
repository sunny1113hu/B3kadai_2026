# B3課題実験リポジトリ

本リポジトリは、「DRL-Pytorch」( https://github.com/XinJingHao/DRL-Pytorch.git ) にて実装されている強化学習フレームワークをベースに、B3課題用に改変したものです。

---

## 1. クローン

```bash
git clone https://github.com/sunny1113hu/B3kadai_2026.git
cd B3kadai_2026
```

---

## 1.5 自分のGitHubに公開する手順
クローンした後に **自分のGitHubにリポジトリを置きたい場合**の手順です。

### 1) GitHub で新規リポジトリを作成
- GitHub 上で **New repository**
- リポジトリ名を決める（例: `DRL-Pytorch`）
- **README / .gitignore は作成しない**（ローカルに既にあるため）

### 2) ローカルで remote を設定
```bash
git remote -v
```

`origin` が無ければ追加:
```bash
git remote add origin https://github.com/<あなたのユーザー名>/<リポジトリ名>.git
```

`origin` が既にある場合は差し替え:
```bash
git remote set-url origin https://github.com/<あなたのユーザー名>/<リポジトリ名>.git
```

### 3) コミット
```bash
git status
git add .
git commit -m "Initial commit"
```

### 4) プッシュ
```bash
git branch -M main
git push -u origin main
```

---

## 2. 環境構築（uv）

Python 3.11 前提です。`uv` を使って仮想環境と依存をまとめて管理します。

```bash
uv sync
```

- これで `.venv` が作成され、依存も解決されます
- 追加パッケージは `uv add <package>` でOK

---

## 3. Docker（GPU）

GPU 環境で実験を回す場合は Docker を使えます。

```bash
docker compose build
docker compose run --rm -it drl
```

- コンテナ内で `python experiments/compare.py ...` を実行してください
- 動画保存には `ffmpeg` が必要です（Dockerfile で導入済み）
- 依存追加後は **再ビルド**が必要です（例: YAML対応のPyYAML追加など）
  ```bash
  docker compose build --no-cache
  ```

---

## 4. 実験の実行

### CartPole: Q-learning / DQN / PER DQN

```bash
python experiments/compare.py --config experiments/configs/cartpole_compare_dqn3.yaml --mode all
```

### Atari: DQN / PER DQN（3ゲーム）

```bash
python experiments/compare.py --config experiments/configs/atari_compare_dqn_per_3games.yaml --mode all
```

出力先:

```
experiments/outputs/<experiment_name>_<timestamp>/
```

---

## 5. 出力物

`experiments/outputs/<experiment_name>_<timestamp>/` に以下が生成されます。

- `manifest.json`（実行内容）
- `summary.csv` / `summary.md` / `summary.json`
- `ep_r_comparison.png`（全体）
- `ep_r_<EnvName>.png`（環境別）
- `metrics/*.csv`（評価スコアの逐次ログ）
- `videos/<run_id>_S<seed>/`（動画）

---

## 6. 可視化ノートブック

`experiments/notebooks/plot_metrics.ipynb` を使うと、
**プロット点の間隔（STEP_BIN）や平滑化（SMOOTH_WINDOW）** を自由に調整できます。

### VS Code で開く場合（推奨）

1. `experiments/notebooks/plot_metrics.ipynb` を開く
2. カーネルに `.venv` を選択
3. 上から順にセルを実行

### GUIモード

- run をドロップダウンから選択
- `Merge ENVS` で全環境を1枚にまとめる

### 手動モード

`OUTPUT_DIR` を直接指定して描画します。

---

## 7. 実験設定の変更

実験条件は `experiments/configs/*.yaml` で管理しています。
YAML ではコメントを書けるため、メモを残したい場合に便利です。

よく使う項目:

- `max_train_steps`, `eval_interval`, `eval_turns`
- `random_steps`, `buffer_size`（CartPole DQN用）, `buffersize`（Atari用）, `target_freq`
- `gamma`, `batch_size`
- `lr_init`, `lr_end`, `lr_decay_steps`（学習率スケジューラ）
- `exp_noise_init`, `exp_noise_end`, `noise_decay_steps`（探索率スケジューラ）

### 学習率スケジューラの挙動（線形減衰）
- **CartPole DQN / PER**: `update_every` のタイミングで lr を更新  
- **Atari DQN / PER**: `random_steps` 以降、学習が走るたびに lr を更新  
（`lr_init` と `lr_end` を同じにすると固定学習率になります）

### 探索率スケジューラ（CartPole DQN / PER）
CartPole側は探索率も線形スケジュールで統一しています。  
`exp_noise_init` → `exp_noise_end` を `noise_decay_steps` で線形に減衰させます。

### 保存済みモデルで評価だけ行う（再実行）
学習せずに **保存済みモデルを評価だけ**したい場合は、各 run に以下を追加します。

```yaml
load_model: true
eval_only: true
model_index: 200   # steps/1000（DQN/Atari/PER用。例: 200000 steps -> 200）
```

Q-learning の場合は `model_path` を指定します。
```yaml
load_model: true
eval_only: true
model_path: model/q_table_q_learning_cartpole_S0.npy
```

評価専用に整理した YAML も用意しています:
- `experiments/configs/cartpole_eval_only.yaml`
- `experiments/configs/atari_eval_only.yaml`

---

## 8. 動画保存について

`record_video: true` で評価動画を保存します。

必要な依存:

```bash
uv add imageio imageio-ffmpeg moviepy
uv sync
```

---

## 9. 備考

- Atariは学習時に報酬クリッピング（-1/0/+1）
- 評価時はクリップなし（本来のスコア）
- AtariのROMが無い場合は、初回のみ以下を実行します:
  ```bash
  AutoROM --accept-license
  ```

必要に応じて設定で切り替え可能です。

---
