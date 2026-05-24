from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REQUIRED_COLUMNS = [
    "battle_idx",
    "train_win_rate",
    "eval_win_rate",
    "q_table_size",
    "reward_total",
    "reward_damage",
    "reward_taken",
    "reward_ko",
    "reward_faint",
    "reward_status",
    "reward_time",
    "reward_potential",
]

REWARD_COMPONENTS = [
    "reward_damage",
    "reward_taken",
    "reward_ko",
    "reward_faint",
    "reward_status",
    "reward_time",
    "reward_potential",
]


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.35,
        }
    )


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"CSV tidak memiliki kolom wajib: {missing_str}")


def clean_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")


def plot_train_win_rate_with_ma(
    df: pd.DataFrame, output_dir: Path, ma_window: int, prefix: str
) -> None:
    plot_df = df.sort_values("battle_idx").copy()
    ma_col = f"ma_{ma_window}"
    plot_df[ma_col] = plot_df["train_win_rate"].rolling(window=ma_window, min_periods=1).mean()

    fig, ax = plt.subplots()
    sns.lineplot(
        data=plot_df,
        x="battle_idx",
        y="train_win_rate",
        color="#1f77b4",
        linewidth=1.8,
        label="Train Win Rate",
        ax=ax,
    )
    sns.lineplot(
        data=plot_df,
        x="battle_idx",
        y=ma_col,
        color="#d62728",
        linewidth=2.4,
        label=f"Moving Average ({ma_window})",
        ax=ax,
    )

    ax.set_title("Kurva Win Rate Training")
    ax.set_xlabel("Indeks Battle")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_train_win_rate_ma.png", dpi=300)
    plt.close(fig)


def plot_last_train_vs_eval_bar(df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    latest = df.sort_values("battle_idx").iloc[-1]
    data = pd.DataFrame(
        {
            "Metrik": ["Train Win Rate (Terakhir)", "Eval Win Rate (Terakhir)"],
            "Nilai": [latest["train_win_rate"], latest["eval_win_rate"]],
        }
    )

    fig, ax = plt.subplots()
    sns.barplot(data=data, x="Metrik", y="Nilai", palette=["#1f77b4", "#ff7f0e"], ax=ax)

    for idx, val in enumerate(data["Nilai"]):
        ax.text(idx, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_title("Perbandingan Win Rate Terakhir")
    ax.set_xlabel("")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_last_train_vs_eval_win_rate.png", dpi=300)
    plt.close(fig)


def plot_qtable_growth(df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    plot_df = df.sort_values("battle_idx")

    fig, ax = plt.subplots()
    sns.lineplot(
        data=plot_df,
        x="battle_idx",
        y="q_table_size",
        color="#2ca02c",
        linewidth=2.2,
        ax=ax,
    )

    ax.set_title("Pertumbuhan Jumlah State Q-Table")
    ax.set_xlabel("Indeks Battle")
    ax.set_ylabel("Jumlah State Q-Table")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_q_table_growth.png", dpi=300)
    plt.close(fig)


def plot_reward_components_stacked_area(df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    plot_df = df.sort_values("battle_idx")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.stackplot(
        plot_df["battle_idx"],
        *[plot_df[col] for col in REWARD_COMPONENTS],
        labels=[
            "Damage",
            "Taken",
            "KO",
            "Faint",
            "Status",
            "Time",
            "Potential",
        ],
        alpha=0.85,
    )

    ax.set_title("Komposisi Reward per Komponen")
    ax.set_xlabel("Indeks Battle")
    ax.set_ylabel("Nilai Reward")
    ax.legend(loc="upper left", ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_reward_components_stacked_area.png", dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualisasi log training Q-learning")
    parser.add_argument(
        "--input",
        required=True,
        help="Path ke file CSV log training atau PKL Q-table",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots",
        help="Folder output gambar (default: results/plots)",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=50,
        help="Window moving average untuk train_win_rate (default: 50)",
    )
    return parser


def load_qtable(input_path: Path) -> dict[Any, np.ndarray]:
    with input_path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError("Format PKL tidak sesuai: objek utama harus dict.")

    q_table: dict[Any, np.ndarray] = {}
    for key, value in data.items():
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 1:
            raise ValueError("Format PKL tidak sesuai: value Q harus array 1 dimensi.")
        q_table[key] = arr

    if not q_table:
        raise ValueError("Q-table kosong: tidak ada state yang bisa divisualisasikan.")

    return q_table


def plot_qvalue_distribution(q_table: dict[Any, np.ndarray], output_dir: Path, prefix: str) -> None:
    all_q_values = np.concatenate(list(q_table.values()))

    fig, ax = plt.subplots()
    sns.histplot(all_q_values, bins=40, kde=True, color="#1f77b4", ax=ax)
    ax.set_title("Distribusi Nilai Q")
    ax.set_xlabel("Nilai Q")
    ax.set_ylabel("Frekuensi")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_q_value_distribution.png", dpi=300)
    plt.close(fig)


def plot_best_action_distribution(q_table: dict[Any, np.ndarray], output_dir: Path, prefix: str) -> None:
    best_actions = [int(np.argmax(q_values)) for q_values in q_table.values() if q_values.size > 0]
    action_count = pd.Series(best_actions).value_counts().sort_index()

    action_df = pd.DataFrame(
        {
            "Aksi": [f"Move {idx}" for idx in action_count.index],
            "Jumlah State": action_count.values,
        }
    )

    fig, ax = plt.subplots()
    sns.barplot(
        data=action_df,
        x="Aksi",
        y="Jumlah State",
        hue="Aksi",
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title("Distribusi Aksi Terbaik per State")
    ax.set_xlabel("Aksi dengan Q Tertinggi")
    ax.set_ylabel("Jumlah State")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_best_action_distribution.png", dpi=300)
    plt.close(fig)


def plot_state_value_distribution(q_table: dict[Any, np.ndarray], output_dir: Path, prefix: str) -> None:
    state_values = np.array([float(np.max(q_values)) for q_values in q_table.values() if q_values.size > 0])
    state_df = pd.DataFrame({"State Value (max Q)": state_values})

    fig, ax = plt.subplots()
    sns.boxplot(data=state_df, y="State Value (max Q)", color="#ff7f0e", ax=ax)
    ax.set_title("Sebaran State Value")
    ax.set_ylabel("Nilai max Q per State")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_state_value_boxplot.png", dpi=300)
    plt.close(fig)


def plot_action_q_means(q_table: dict[Any, np.ndarray], output_dir: Path, prefix: str) -> None:
    max_actions = max(arr.size for arr in q_table.values())
    matrix = np.full((len(q_table), max_actions), np.nan)

    for idx, q_values in enumerate(q_table.values()):
        matrix[idx, : q_values.size] = q_values

    mean_q = np.nanmean(matrix, axis=0)
    action_idx = np.arange(max_actions)

    fig, ax = plt.subplots()
    sns.lineplot(x=action_idx, y=mean_q, marker="o", linewidth=2.2, color="#2ca02c", ax=ax)
    ax.set_title("Rata-rata Nilai Q per Index Aksi")
    ax.set_xlabel("Index Aksi")
    ax.set_ylabel("Rata-rata Q")
    ax.set_xticks(action_idx)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_mean_q_per_action.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        candidate_csv = sorted(Path(".").glob("**/*.csv"))
        candidate_csv = [p for p in candidate_csv if ".git" not in p.parts]
        candidate_pkl = sorted(Path(".").glob("**/*.pkl"))
        candidate_pkl = [p for p in candidate_pkl if ".git" not in p.parts]

        guidance = [
            f"File input tidak ditemukan: {input_path}",
            "",
            "Catatan: nilai 'path/to/training_log.csv' pada README adalah placeholder.",
        ]

        if candidate_csv:
            guidance.append("CSV yang terdeteksi di workspace:")
            guidance.extend([f"- {p.as_posix()}" for p in candidate_csv[:10]])
        if candidate_pkl:
            guidance.append("PKL yang terdeteksi di workspace:")
            guidance.extend([f"- {p.as_posix()}" for p in candidate_pkl[:10]])

        if candidate_csv or candidate_pkl:
            guidance.append("Gunakan salah satu path di atas untuk argumen --input.")
        else:
            guidance.append("Belum ada file CSV/PKL yang bisa divisualisasikan di workspace Anda.")
            guidance.append(
                "Buat dulu file log training (mis. results/training_log.csv), lalu jalankan:"
            )
            guidance.append(
                "python src/visualize_training.py --input results/training_log.csv"
            )

        raise FileNotFoundError("\n".join(guidance))
    if args.ma_window <= 0:
        raise ValueError("ma_window harus lebih besar dari 0")

    input_tag = clean_name(input_path.stem)
    if not input_tag:
        input_tag = "input"
    prefix = f"plot_{input_tag}"
    output_dir = output_dir / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path)
        validate_columns(df)

        plot_train_win_rate_with_ma(df, output_dir, args.ma_window, prefix)
        plot_last_train_vs_eval_bar(df, output_dir, prefix)
        plot_qtable_growth(df, output_dir, prefix)
        plot_reward_components_stacked_area(df, output_dir, prefix)

        print("Visualisasi mode CSV berhasil dibuat.")
        print(f"Input: {input_path}")
        print(f"Output folder: {output_dir.resolve()}")
        return

    if suffix == ".pkl":
        q_table = load_qtable(input_path)

        plot_qvalue_distribution(q_table, output_dir, prefix)
        plot_best_action_distribution(q_table, output_dir, prefix)
        plot_state_value_distribution(q_table, output_dir, prefix)
        plot_action_q_means(q_table, output_dir, prefix)

        print("Visualisasi mode PKL berhasil dibuat.")
        print(f"Input: {input_path}")
        print(f"Jumlah state: {len(q_table)}")
        print(f"Output folder: {output_dir.resolve()}")
        return

    raise ValueError(
        "Format file tidak didukung. Gunakan .csv (log training) atau .pkl (Q-table)."
    )

if __name__ == "__main__":
    main()
