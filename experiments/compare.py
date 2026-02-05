import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import statistics
import shutil


DEFAULT_CONFIG = "experiments/configs/cartpole_compare.json"


RUNS_ROOT = {
    "q_learning": "1.Q-learning/runs",
    "dqn_cartpole": "2.1_Duel-Double-DQN/runs",
    "atari_dqn": "2.2_Noisy-Duel-DDQN-Atari/runs",
    "prioritized_dqn": "2.3 Prioritized-Experience-Replay-DDQN-DQN/LightPriorDQN_gym0.2x/runs",
}


def detect_config_format(path):
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    return "json"


def load_config(path):
    fmt = detect_config_format(path)
    with open(path, "r", encoding="utf-8") as f:
        if fmt == "yaml":
            try:
                import yaml
            except ImportError as exc:
                raise SystemExit("PyYAML is required for .yaml configs. Run: uv add pyyaml") from exc
            return yaml.safe_load(f)
        return json.load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def find_latest_dir(root, newer_than=None):
    root = Path(root)
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if newer_than is not None:
        candidates = [p for p in candidates if p.stat().st_mtime >= newer_than]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_latest_event_file(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    candidates = [p for p in run_dir.rglob("events.out.tfevents*") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_command(run_cfg, seed, python_exec):
    run_type = run_cfg["type"]
    if run_type == "q_learning":
        script = "main.py"
        args = [
            "--env", run_cfg.get("env", "CliffWalking-v0"),
            "--seed", str(seed),
            "--max_train_steps", str(run_cfg.get("max_train_steps", 20000)),
            "--eval_interval", str(run_cfg.get("eval_interval", 100)),
            "--save_interval", str(run_cfg.get("save_interval", 0)),
            "--write", str(run_cfg.get("write", True)),
            "--lr", str(run_cfg.get("lr", 0.2)),
            "--gamma", str(run_cfg.get("gamma", 0.9)),
            "--epsilon", str(run_cfg.get("epsilon", 0.1)),
            "--max_episode_steps", str(run_cfg.get("max_episode_steps", 500)),
            "--eval_max_episode_steps", str(run_cfg.get("eval_max_episode_steps", 100)),
            "--eval_turns", str(run_cfg.get("eval_turns", 1)),
        ]
        if "bins" in run_cfg:
            args += ["--bins"] + [str(x) for x in run_cfg["bins"]]
        if "low" in run_cfg:
            args += ["--low"] + [str(x) for x in run_cfg["low"]]
        if "high" in run_cfg:
            args += ["--high"] + [str(x) for x in run_cfg["high"]]
        if run_cfg.get("run_name"):
            args += ["--run_name", f"{run_cfg['run_name']}_S{seed}"]
        if run_cfg.get("log_root"):
            args += ["--log_root", run_cfg["log_root"]]
        if run_cfg.get("model_dir"):
            args += ["--model_dir", run_cfg["model_dir"]]
        if run_cfg.get("load_model"):
            args += ["--load_model", "True"]
        if run_cfg.get("model_path"):
            args += ["--model_path", run_cfg["model_path"]]
        if run_cfg.get("eval_only"):
            args += ["--eval_only", "True"]
        if run_cfg.get("csv_log"):
            args += ["--csv_log", run_cfg["csv_log"]]
        if run_cfg.get("record_video"):
            args += ["--record_video", "True"]
            if run_cfg.get("video_dir"):
                args += ["--video_dir", run_cfg["video_dir"]]
            if run_cfg.get("video_episodes") is not None:
                args += ["--video_episodes", str(run_cfg.get("video_episodes", 1))]
            if run_cfg.get("video_prefix"):
                args += ["--video_prefix", run_cfg["video_prefix"]]
        cmd = [python_exec, script] + args
        return cmd, "1.Q-learning"

    if run_type == "dqn_cartpole":
        script = "main.py"
        args = [
            "--EnvIdex", str(run_cfg.get("env_index", 0)),
            "--seed", str(seed),
            "--Max_train_steps", str(run_cfg.get("max_train_steps", int(1e6))),
            "--eval_interval", str(run_cfg.get("eval_interval", int(2e3))),
            "--eval_turns", str(run_cfg.get("eval_turns", 3)),
            "--save_interval", str(run_cfg.get("save_interval", int(50e3))),
            "--random_steps", str(run_cfg.get("random_steps", int(3e3))),
            "--update_every", str(run_cfg.get("update_every", 50)),
            "--gamma", str(run_cfg.get("gamma", 0.99)),
            "--net_width", str(run_cfg.get("net_width", 200)),
            "--lr", str(run_cfg.get("lr", 1e-4)),
            "--lr_init", str(run_cfg.get("lr_init", run_cfg.get("lr", 1e-4))),
            "--lr_end", str(run_cfg.get("lr_end", run_cfg.get("lr", 1e-4))),
            "--lr_decay_steps", str(run_cfg.get("lr_decay_steps", run_cfg.get("max_train_steps", int(1e6)))),
            "--batch_size", str(run_cfg.get("batch_size", 256)),
            "--exp_noise", str(run_cfg.get("exp_noise", 0.2)),
            "--noise_decay", str(run_cfg.get("noise_decay", 0.99)),
            "--Double", str(run_cfg.get("double", True)),
            "--Duel", str(run_cfg.get("duel", False)),
            "--write", str(run_cfg.get("write", True)),
        ]
        if "device" in run_cfg:
            args += ["--dvc", run_cfg["device"]]
        if run_cfg.get("load_model"):
            args += ["--Loadmodel", "True"]
        if run_cfg.get("model_index") is not None:
            args += ["--ModelIdex", str(run_cfg.get("model_index"))]
        if run_cfg.get("eval_only"):
            args += ["--eval_only", "True"]
        if run_cfg.get("csv_log"):
            args += ["--csv_log", run_cfg["csv_log"]]
        if run_cfg.get("record_video"):
            args += ["--record_video", "True"]
            if run_cfg.get("video_dir"):
                args += ["--video_dir", run_cfg["video_dir"]]
            if run_cfg.get("video_episodes") is not None:
                args += ["--video_episodes", str(run_cfg.get("video_episodes", 1))]
            if run_cfg.get("video_prefix"):
                args += ["--video_prefix", run_cfg["video_prefix"]]
        cmd = [python_exec, script] + args
        return cmd, "2.1_Duel-Double-DQN"

    if run_type == "atari_dqn":
        script = "main.py"
        args = [
            "--EnvIdex", str(run_cfg.get("env_index", 37)),
            "--seed", str(seed),
            "--Max_train_steps", str(run_cfg.get("max_train_steps", int(1e6))),
            "--eval_interval", str(run_cfg.get("eval_interval", int(5e3))),
            "--eval_turns", str(run_cfg.get("eval_turns", 3)),
            "--save_interval", str(run_cfg.get("save_interval", int(1e5))),
            "--random_steps", str(run_cfg.get("random_steps", int(1e4))),
            "--buffersize", str(run_cfg.get("buffersize", int(1e4))),
            "--target_freq", str(run_cfg.get("target_freq", int(1e3))),
            "--gamma", str(run_cfg.get("gamma", 0.99)),
            "--lr", str(run_cfg.get("lr", 1e-4)),
            "--batch_size", str(run_cfg.get("batch_size", 32)),
            "--init_e", str(run_cfg.get("init_e", 1.0)),
            "--anneal_frac", str(run_cfg.get("anneal_frac", int(3e5))),
            "--final_e", str(run_cfg.get("final_e", 0.02)),
            "--Double", str(run_cfg.get("double", False)),
            "--Duel", str(run_cfg.get("duel", False)),
            "--Noisy", str(run_cfg.get("noisy", False)),
            "--write", str(run_cfg.get("write", True)),
        ]
        if "device" in run_cfg:
            args += ["--device", run_cfg["device"]]
        if run_cfg.get("load_model"):
            args += ["--Loadmodel", "True"]
        if run_cfg.get("model_index") is not None:
            args += ["--ModelIdex", str(run_cfg.get("model_index"))]
        if run_cfg.get("eval_only"):
            args += ["--eval_only", "True"]
        if "noop_reset" in run_cfg:
            args += ["--noop_reset", str(run_cfg["noop_reset"])]
        if "huber_loss" in run_cfg:
            args += ["--huber_loss", str(run_cfg["huber_loss"])]
        if "lr_init" in run_cfg:
            args += ["--lr_init", str(run_cfg["lr_init"])]
        if "lr_end" in run_cfg:
            args += ["--lr_end", str(run_cfg["lr_end"])]
        if "lr_decay_steps" in run_cfg:
            args += ["--lr_decay_steps", str(run_cfg["lr_decay_steps"])]
        if "PER" in run_cfg:
            args += ["--PER", str(run_cfg["PER"])]
        if "per_alpha" in run_cfg:
            args += ["--per_alpha", str(run_cfg["per_alpha"])]
        if "per_beta_init" in run_cfg:
            args += ["--per_beta_init", str(run_cfg["per_beta_init"])]
        if "per_beta_gain_steps" in run_cfg:
            args += ["--per_beta_gain_steps", str(run_cfg["per_beta_gain_steps"])]
        if "per_eps" in run_cfg:
            args += ["--per_eps", str(run_cfg["per_eps"])]
        if "per_replacement" in run_cfg:
            args += ["--per_replacement", str(run_cfg["per_replacement"])]
        if run_cfg.get("csv_log"):
            args += ["--csv_log", run_cfg["csv_log"]]
        if run_cfg.get("record_video"):
            args += ["--record_video", "True"]
            if run_cfg.get("video_dir"):
                args += ["--video_dir", run_cfg["video_dir"]]
            if run_cfg.get("video_episodes") is not None:
                args += ["--video_episodes", str(run_cfg.get("video_episodes", 1))]
            if run_cfg.get("video_prefix"):
                args += ["--video_prefix", run_cfg["video_prefix"]]
        cmd = [python_exec, script] + args
        return cmd, "2.2_Noisy-Duel-DDQN-Atari"

    if run_type == "prioritized_dqn":
        script = "main.py"
        args = [
            "--EnvIdex", str(run_cfg.get("env_index", 0)),
            "--seed", str(seed),
            "--Max_train_steps", str(run_cfg.get("max_train_steps", int(3e5))),
            "--buffer_size", str(run_cfg.get("buffer_size", int(2e5))),
            "--save_interval", str(run_cfg.get("save_interval", int(5e4))),
            "--eval_interval", str(run_cfg.get("eval_interval", int(1e3))),
            "--eval_turns", str(run_cfg.get("eval_turns", 3)),
            "--warmup", str(run_cfg.get("warmup", int(3e3))),
            "--update_every", str(run_cfg.get("update_every", 50)),
            "--gamma", str(run_cfg.get("gamma", 0.99)),
            "--net_width", str(run_cfg.get("net_width", 256)),
            "--lr_init", str(run_cfg.get("lr_init", 1.5e-4)),
            "--lr_end", str(run_cfg.get("lr_end", 6e-5)),
            "--lr_decay_steps", str(run_cfg.get("lr_decay_steps", int(3e5))),
            "--batch_size", str(run_cfg.get("batch_size", 256)),
            "--exp_noise_init", str(run_cfg.get("exp_noise_init", 0.6)),
            "--exp_noise_end", str(run_cfg.get("exp_noise_end", 0.03)),
            "--noise_decay_steps", str(run_cfg.get("noise_decay_steps", int(1e5))),
            "--DDQN", str(run_cfg.get("ddqn", True)),
            "--alpha", str(run_cfg.get("alpha", 0.6)),
            "--beta_init", str(run_cfg.get("beta_init", 0.4)),
            "--beta_gain_steps", str(run_cfg.get("beta_gain_steps", int(3e5))),
            "--replacement", str(run_cfg.get("replacement", False)),
            "--write", str(run_cfg.get("write", True)),
        ]
        if run_cfg.get("load_model"):
            args += ["--Loadmodel", "True"]
        if run_cfg.get("model_index") is not None:
            args += ["--ModelIdex", str(run_cfg.get("model_index"))]
        if run_cfg.get("eval_only"):
            args += ["--eval_only", "True"]
        if run_cfg.get("csv_log"):
            args += ["--csv_log", run_cfg["csv_log"]]
        if run_cfg.get("record_video"):
            args += ["--record_video", "True"]
            if run_cfg.get("video_dir"):
                args += ["--video_dir", run_cfg["video_dir"]]
            if run_cfg.get("video_episodes") is not None:
                args += ["--video_episodes", str(run_cfg.get("video_episodes", 1))]
            if run_cfg.get("video_prefix"):
                args += ["--video_prefix", run_cfg["video_prefix"]]
        cmd = [python_exec, script] + args
        return cmd, "2.3 Prioritized-Experience-Replay-DDQN-DQN/LightPriorDQN_gym0.2x"

    raise ValueError(f"Unknown run type: {run_type}")


def run_command(cmd, cwd=None, workdir=None, dry_run=False):
    print(">>", " ".join(cmd))
    if dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if workdir is not None:
        cwd = workdir
    result = subprocess.run(cmd, cwd=cwd, env=env)
    return result.returncode


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def write_yaml(path, data):
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to write .yaml configs. Run: uv add pyyaml") from exc
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def summarize_run(run):
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as e:
        return {"error": f"tensorboard not available: {e}"}

    if not run.get("run_dir"):
        return {"error": "run_dir not found"}

    event_file = find_latest_event_file(run["run_dir"])
    if event_file is None:
        return {"error": "no event file"}

    acc = event_accumulator.EventAccumulator(str(event_file))
    acc.Reload()
    if "ep_r" not in acc.Tags().get("scalars", []):
        return {"error": "ep_r not found"}

    scalars = acc.Scalars("ep_r")
    values = [(s.step, float(s.value)) for s in scalars]
    best_step, best_val = max(values, key=lambda x: x[1])
    first_step, first_val = values[0]
    last_step, last_val = values[-1]
    return {
        "points": len(values),
        "first_step": first_step,
        "first_value": first_val,
        "best_step": best_step,
        "best_value": best_val,
        "last_step": last_step,
        "last_value": last_val,
        "event_file": str(event_file),
    }


def write_summary(output_dir, manifest):
    rows = []
    for run in manifest["runs"]:
        summary = summarize_run(run)
        row = {
            "id": run["id"],
            "type": run["type"],
            "seed": run["seed"],
            "env": run.get("env", ""),
            "run_dir": run.get("run_dir", ""),
            "video_dir": run.get("video_dir", ""),
            "video_prefix": run.get("video_prefix", ""),
            "csv_log": run.get("csv_log", ""),
            "cmd": " ".join(run.get("cmd", [])),
            "started_at": run.get("started_at", ""),
            "ended_at": run.get("ended_at", ""),
            "duration_sec": run.get("duration_sec", ""),
        }
        row.update(summary)
        rows.append(row)

    aggregates = build_aggregates(rows)
    param_table = build_param_table(manifest)

    csv_path = Path(output_dir) / "summary.csv"
    headers = [
        "id",
        "type",
        "seed",
        "env",
        "run_dir",
        "video_dir",
        "video_prefix",
        "csv_log",
        "started_at",
        "ended_at",
        "duration_sec",
        "points",
        "first_step",
        "first_value",
        "best_step",
        "best_value",
        "last_step",
        "last_value",
        "event_file",
        "cmd",
        "error",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")

    md_path = Path(output_dir) / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 実験レポート\n\n")
        f.write("## 概要\n")
        f.write(f"- 実験名: {manifest.get('experiment_name','')}\n")
        f.write(f"- 実行時刻: {manifest.get('timestamp','')}\n")
        f.write(f"- 出力先: {output_dir}\n")
        f.write(f"- 実行本数: {len(rows)}\n\n")

        highlights = build_highlights(aggregates)
        if highlights:
            f.write("## ハイライト\n")
            for item in highlights:
                f.write(
                    f"- {item['env']}: 最高平均スコアは **{item['id']}** "
                    f"(best={item['best_value_mean']:.2f} ± {item['best_value_std']:.2f}, n={item['count']})\n"
                )
            f.write("\n")

        f.write("## 集計（平均 ± 標準偏差）\n")
        f.write("| id | type | env | n | best | last | first |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for agg in aggregates:
            best = format_mean_std(agg["best_value_mean"], agg["best_value_std"])
            last = format_mean_std(agg["last_value_mean"], agg["last_value_std"])
            first = format_mean_std(agg["first_value_mean"], agg["first_value_std"])
            f.write(f"| {agg['id']} | {agg['type']} | {agg['env']} | {agg['count']} | {best} | {last} | {first} |\n")
        f.write("\n")

        f.write("## 個別結果（seed別）\n")
        f.write("| id | type | seed | env | duration(s) | points | first | best | last |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            if "error" in row and row["error"]:
                f.write(f"| {row['id']} | {row['type']} | {row['seed']} | {row.get('env','')} | - | - | - | - | - |\n")
            else:
                first = f"{row['first_step']}:{row['first_value']:.2f}"
                best = f"{row['best_step']}:{row['best_value']:.2f}"
                last = f"{row['last_step']}:{row['last_value']:.2f}"
                f.write(f"| {row['id']} | {row['type']} | {row['seed']} | {row.get('env','')} | {row.get('duration_sec','')} | {row['points']} | {first} | {best} | {last} |\n")
        f.write("\n")

        f.write("## 実行パラメータ（主要）\n")
        f.write("| id | env | key | value |\n")
        f.write("|---|---|---|---|\n")
        for entry in param_table:
            f.write(f"| {entry['id']} | {entry['env']} | {entry['key']} | {entry['value']} |\n")
        f.write("\n")

        f.write("## 詳細\n")
        for row in rows:
            f.write(f"- {row['id']} (seed {row['seed']}): {row.get('run_dir','')}\n")
            if row.get("error"):
                f.write(f"  - error: {row['error']}\n")
            else:
                f.write(f"  - best: step {row['best_step']} value {row['best_value']}\n")
                f.write(f"  - last: step {row['last_step']} value {row['last_value']}\n")
            if row.get("duration_sec"):
                f.write(f"  - duration_sec: {row['duration_sec']}\n")
            if row.get("video_dir"):
                f.write(f"  - video_dir: {row.get('video_dir','')}\n")
            if row.get("csv_log"):
                f.write(f"  - csv_log: {row.get('csv_log','')}\n")
            f.write(f"  - cmd: {row.get('cmd','')}\n")

    json_path = Path(output_dir) / "summary.json"
    json_payload = {
        "experiment_name": manifest.get("experiment_name", ""),
        "timestamp": manifest.get("timestamp", ""),
        "output_dir": str(output_dir),
        "runs": rows,
        "aggregates": aggregates,
        "params": param_table,
    }
    write_json(json_path, json_payload)

    return rows


def build_aggregates(rows):
    groups = {}
    for row in rows:
        if row.get("error"):
            continue
        key = (row.get("id", ""), row.get("type", ""), row.get("env", ""))
        groups.setdefault(key, []).append(row)

    aggregates = []
    for (run_id, run_type, env), items in groups.items():
        best_vals = [float(i["best_value"]) for i in items if i.get("best_value") is not None]
        last_vals = [float(i["last_value"]) for i in items if i.get("last_value") is not None]
        first_vals = [float(i["first_value"]) for i in items if i.get("first_value") is not None]
        best_steps = [int(i["best_step"]) for i in items if i.get("best_step") is not None]
        last_steps = [int(i["last_step"]) for i in items if i.get("last_step") is not None]

        aggregates.append({
            "id": run_id,
            "type": run_type,
            "env": env,
            "count": len(items),
            "best_value_mean": safe_mean(best_vals),
            "best_value_std": safe_std(best_vals),
            "last_value_mean": safe_mean(last_vals),
            "last_value_std": safe_std(last_vals),
            "first_value_mean": safe_mean(first_vals),
            "first_value_std": safe_std(first_vals),
            "best_step_mean": safe_mean(best_steps),
            "best_step_std": safe_std(best_steps),
            "last_step_mean": safe_mean(last_steps),
            "last_step_std": safe_std(last_steps),
        })

    aggregates.sort(key=lambda x: (x["id"], x["env"]))
    return aggregates


def safe_mean(values):
    if not values:
        return None
    return float(sum(values) / len(values))


def safe_std(values):
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.stdev(values))


def format_mean_std(mean, std):
    if mean is None:
        return "-"
    if std is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def build_highlights(aggregates):
    best_by_env = {}
    for agg in aggregates:
        env = agg.get("env", "")
        mean = agg.get("best_value_mean")
        if mean is None:
            continue
        if env not in best_by_env or mean > best_by_env[env]["best_value_mean"]:
            best_by_env[env] = agg
    return list(best_by_env.values())


def build_param_table(manifest):
    keys = [
        "EnvIdex",
        "EnvName",
        "gamma",
        "lr",
        "lr_init",
        "lr_end",
        "lr_decay_steps",
        "batch_size",
        "buffersize",
        "target_freq",
        "random_steps",
        "init_e",
        "anneal_frac",
        "final_e",
        "PER",
        "per_alpha",
        "per_beta_init",
        "per_beta_gain_steps",
    ]
    rows = []
    for run in manifest.get("runs", []):
        cmd = run.get("cmd", [])
        args = parse_cmd_args(cmd)
        env = run.get("env", "") or args.get("EnvName", "")
        for key in keys:
            if key in args:
                rows.append({
                    "id": run.get("id", ""),
                    "env": env,
                    "key": key,
                    "value": args.get(key, ""),
                })
    return rows


def parse_cmd_args(cmd):
    args = {}
    if not cmd:
        return args
    i = 0
    while i < len(cmd):
        token = cmd[i]
        if token.startswith("--"):
            key = token[2:]
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                args[key] = cmd[i + 1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            i += 1
    return args


def algo_key(run_id):
    if not run_id:
        return "other"
    rid = run_id.lower()
    if rid.startswith("per_"):
        return "per_dqn"
    if "prioritized" in rid:
        return "per_dqn"
    if "double" in rid:
        return "double_dqn"
    if "dqn" in rid:
        return "dqn"
    if "q_learning" in rid or "qlearning" in rid:
        return "q_learning"
    return run_id


def build_color_map(run_ids):
    palette = {
        "dqn": "#1f77b4",
        "double_dqn": "#ff7f0e",
        "per_dqn": "#2ca02c",
        "q_learning": "#d62728",
    }
    mapping = {}
    for rid in run_ids:
        key = algo_key(rid)
        if key not in mapping:
            mapping[key] = palette.get(key)
    return mapping


def plot_summary(output_dir, manifest, rows):
    try:
        import matplotlib.pyplot as plt
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        print("matplotlib or tensorboard not available; skip plotting.")
        return

    plt.figure(figsize=(8, 4))
    color_map = build_color_map([r.get("id", "") for r in manifest.get("runs", [])])
    all_values = []
    for run in manifest["runs"]:
        if not run.get("run_dir"):
            continue
        event_file = find_latest_event_file(run["run_dir"])
        if event_file is None:
            continue
        acc = event_accumulator.EventAccumulator(str(event_file))
        acc.Reload()
        if "ep_r" not in acc.Tags().get("scalars", []):
            continue
        scalars = acc.Scalars("ep_r")
        steps = [s.step for s in scalars]
        values = [s.value for s in scalars]
        label = f"{run['id']}_S{run['seed']}"
        color = color_map.get(algo_key(run.get("id", "")), None)
        plt.plot(steps, values, label=label, color=color)
        if values:
            all_values.extend(values)
            best_idx = max(range(len(values)), key=lambda i: values[i])
            plt.scatter(steps[best_idx], values[best_idx], marker="^", color=color, s=30, zorder=3)
            plt.scatter(steps[-1], values[-1], marker="o", color=color, s=25, zorder=3)
    plt.title("ep_r comparison")
    plt.xlabel("steps")
    plt.ylabel("episode reward")
    plt.grid(True, alpha=0.3)
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        rng = y_max - y_min
        pad = 1.0 if rng == 0 else rng * 0.05
        plt.ylim(y_min - pad, y_max + pad)
    plt.legend(fontsize=8)
    plot_path = Path(output_dir) / "ep_r_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)


def plot_by_env(output_dir, manifest):
    try:
        import matplotlib.pyplot as plt
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        print("matplotlib or tensorboard not available; skip plotting.")
        return

    runs_by_env = {}
    for run in manifest.get("runs", []):
        env = run.get("env", "") or "unknown"
        runs_by_env.setdefault(env, []).append(run)

    color_map = build_color_map([r.get("id", "") for r in manifest.get("runs", [])])
    for env, runs in runs_by_env.items():
        plt.figure(figsize=(8, 4))
        all_values = []
        for run in runs:
            if not run.get("run_dir"):
                continue
            event_file = find_latest_event_file(run["run_dir"])
            if event_file is None:
                continue
            acc = event_accumulator.EventAccumulator(str(event_file))
            acc.Reload()
            if "ep_r" not in acc.Tags().get("scalars", []):
                continue
            scalars = acc.Scalars("ep_r")
            steps = [s.step for s in scalars]
            values = [s.value for s in scalars]
            label = f"{run['id']}_S{run['seed']}"
            color = color_map.get(algo_key(run.get("id", "")), None)
            plt.plot(steps, values, label=label, color=color)
            if values:
                all_values.extend(values)
                best_idx = max(range(len(values)), key=lambda i: values[i])
                plt.scatter(steps[best_idx], values[best_idx], marker="^", color=color, s=30, zorder=3)
                plt.scatter(steps[-1], values[-1], marker="o", color=color, s=25, zorder=3)
        plt.title(f"ep_r comparison - {env}")
        plt.xlabel("steps")
        plt.ylabel("episode reward")
        plt.grid(True, alpha=0.3)
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            rng = y_max - y_min
            pad = 1.0 if rng == 0 else rng * 0.05
            plt.ylim(y_min - pad, y_max + pad)
        plt.legend(fontsize=8)
        safe_env = env.replace("/", "_")
        plot_path = Path(output_dir) / f"ep_r_{safe_env}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)


def find_latest_output(results_root):
    root = Path(results_root)
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to config JSON/YAML")
    parser.add_argument("--mode", type=str, default="all", choices=["run", "report", "all"])
    parser.add_argument("--output", type=str, default="", help="Use an existing output dir for report")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--no_plot", action="store_true", help="Skip generating plots")
    args = parser.parse_args()

    config_format = detect_config_format(args.config)
    config = load_config(args.config)
    experiment_name = config.get("experiment_name", "experiment")
    results_root = config.get("results_dir", "experiments/outputs")
    repo_root = Path(__file__).resolve().parent.parent

    python_exec = config.get("python", sys.executable)
    python_path = Path(python_exec).expanduser()
    if not python_path.is_absolute():
        python_path = repo_root / python_path
    python_path = python_path.absolute()
    if not python_path.exists():
        fallback = Path(sys.executable)
        print(f"Warning: python not found at {python_path}, fallback to {fallback}", file=sys.stderr)
        python_path = fallback
        if not python_path.exists():
            raise SystemExit(f"Python executable not found: {python_path}")
    python_exec = str(python_path)
    ensure_dir(results_root)

    if args.mode in ("run", "all"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(results_root) / f"{experiment_name}_{timestamp}"
        ensure_dir(output_dir)
        write_json(output_dir / "config.json", config)
        if config_format == "yaml":
            write_yaml(output_dir / "config.yaml", config)

        manifest = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "runs": [],
        }

        for run_cfg in config.get("runs", []):
            run_id = run_cfg["id"]
            run_type = run_cfg["type"]
            run_env = run_cfg.get("env", "")
            run_seeds = run_cfg.get("seeds", config.get("seeds", [0]))
            for seed in run_seeds:
                run_cfg_effective = dict(run_cfg)
                if run_cfg_effective.get("eval_only"):
                    run_cfg_effective["eval_only"] = bool(run_cfg_effective.get("eval_only"))
                    if not run_cfg_effective.get("load_model"):
                        run_cfg_effective["load_model"] = True
                    if run_type in ("dqn_cartpole", "atari_dqn", "prioritized_dqn"):
                        if run_cfg_effective.get("model_index") is None:
                            max_steps = run_cfg_effective.get("max_train_steps", 0)
                            if max_steps:
                                run_cfg_effective["model_index"] = int(max_steps // 1000)
                if run_cfg_effective.get("record_video"):
                    run_cfg_effective["record_video"] = bool(run_cfg_effective.get("record_video"))
                    if not run_cfg_effective.get("video_dir"):
                        run_cfg_effective["video_dir"] = str(
                            (Path(output_dir) / "videos" / f"{run_id}_S{seed}").resolve()
                        )
                    if not run_cfg_effective.get("video_prefix"):
                        run_cfg_effective["video_prefix"] = f"{run_id}_S{seed}"
                if not run_cfg_effective.get("csv_log"):
                    run_cfg_effective["csv_log"] = str(
                        (Path(output_dir) / "metrics" / f"{run_id}_S{seed}.csv").resolve()
                    )
                if run_cfg_effective.get("record_video"):
                    video_dir = Path(run_cfg_effective["video_dir"])
                    if video_dir.exists():
                        shutil.rmtree(video_dir)
                cmd, workdir = build_command(run_cfg_effective, seed, python_exec)
                runs_root = RUNS_ROOT.get(run_type)
                before = time.time()
                if run_type in RUNS_ROOT:
                    latest_before = find_latest_dir(RUNS_ROOT[run_type])
                    if latest_before:
                        before = max(before, latest_before.stat().st_mtime + 0.01)
                started_at = datetime.now().isoformat(timespec="seconds")
                t0 = time.time()
                ret = run_command(cmd, workdir=workdir, dry_run=args.dry_run)
                t1 = time.time()
                ended_at = datetime.now().isoformat(timespec="seconds")
                if ret != 0:
                    raise SystemExit(f"Command failed: {cmd}")

                run_dir = None
                if run_type in RUNS_ROOT:
                    run_dir_path = find_latest_dir(RUNS_ROOT[run_type], newer_than=before - 0.01)
                    if run_dir_path is not None:
                        run_dir = str(run_dir_path)

                manifest["runs"].append({
                    "id": run_id,
                    "type": run_type,
                    "seed": seed,
                    "env": run_env,
                    "cmd": cmd,
                    "run_dir": run_dir,
                    "video_dir": run_cfg_effective.get("video_dir", ""),
                    "video_prefix": run_cfg_effective.get("video_prefix", ""),
                    "csv_log": run_cfg_effective.get("csv_log", ""),
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "duration_sec": int(t1 - t0),
                })

        write_json(output_dir / "manifest.json", manifest)
    else:
        output_dir = Path(args.output) if args.output else find_latest_output(results_root)
        if output_dir is None:
            raise SystemExit("No output directory found. Run with --mode run first.")
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"manifest.json not found in {output_dir}")
        manifest = load_config(str(manifest_path))

    if args.mode in ("report", "all"):
        rows = write_summary(output_dir, manifest)
        if not args.no_plot:
            plot_summary(output_dir, manifest, rows)
            plot_by_env(output_dir, manifest)
        print(f"Summary written to: {output_dir}")


if __name__ == "__main__":
    main()
