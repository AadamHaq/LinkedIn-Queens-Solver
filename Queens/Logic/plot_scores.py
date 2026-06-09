import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import FuncFormatter

project_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=project_root / ".env")


def name_to_env_key(name):
    cleaned = name.replace(".", "").strip()
    return "ALIAS_" + cleaned.replace(" ", "_").upper()


def get_dow(day):
    dow_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    return dow_names[((day + 2) % 7)]  # Monday is day % 7 == 6 → index 6 = "Sat"


# Build alias dict directly from os.environ
alias_dict = {
    key: value for key, value in os.environ.items() if key.startswith("ALIAS_")
}


def prepare_csv(results):
    # Extract all Queens numbers from results
    day_numbers = []
    for _, text in results:
        match = re.search(r"Queens\s+#(\d+)", text)
        if match:
            day_numbers.append(int(match.group(1)))

    if not day_numbers:
        raise ValueError("No Queens numbers found in results.")

    # Use the highest day number (latest puzzle)
    latest_day = max(day_numbers)
    dow = get_dow(latest_day)

    row = {"Day": latest_day, "DoW": dow}

    for name, text in results:
        if f"Queens #{latest_day}" not in text:
            continue  # Skip if it's not from the latest puzzle

        alias_key = name_to_env_key(name)
        alias = alias_dict.get(alias_key)

        print(
            f"Processing name '{name}' → alias key '{alias_key}' → alias value '{alias}'"
        )

        if alias:
            score_match = re.search(r"(\d{1,2}:\d{2})", text)
            if score_match:
                row[alias] = score_match.group(1)

    print(f"Row obtained: {row}")
    return row


def backfill_previous_day(results, latest_day):
    previous_day = latest_day - 1
    csv_file = project_root / "scores.csv"

    if not csv_file.exists():
        return  # Nothing to backfill

    df = pd.read_csv(csv_file)
    if previous_day not in df["Day"].values:
        return  # No row to update

    prev_idx = df.index[df["Day"] == previous_day][0]
    modified = False

    for name, text in results:
        if f"Queens #{previous_day}" not in text:
            continue

        alias_key = name_to_env_key(name)
        alias = alias_dict.get(alias_key)

        if alias and alias in df.columns and pd.isna(df.at[prev_idx, alias]):
            score_match = re.search(r"(\d{1,2}:\d{2})", text)
            if score_match:
                df.at[prev_idx, alias] = score_match.group(1)
                print(f"[Backfill] Inserted {alias}'s score into Day {previous_day}")
                modified = True

    if modified:
        df.to_csv(csv_file, index=False)
        print(f"[Backfill] Updated Day {previous_day} in {csv_file.name}")


def write_csv(results):
    row = prepare_csv(results)

    # Path to Queens/scores.csv
    csv_file = project_root / "scores.csv"
    file_exists = csv_file.exists()

    # Define the column order explicitly:
    columns = ["Day", "DoW"] + list(alias_dict.values())

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()

        # Ensure all columns are present in row dict, fill missing with empty string
        row_filled = {col: row.get(col, "") for col in columns}

        writer.writerow(row_filled)

    print(f"Row written to {csv_file.name}: {row}")

    backfill_previous_day(results, row["Day"])


def plot_graph():
    # Load CSV
    df = pd.read_csv(project_root / "scores.csv")

    # Convert mm:ss to total seconds
    def time_to_seconds(t):
        if pd.isna(t):
            return None
        try:
            m, s = map(int, t.split(":"))
            return m * 60 + s
        except Exception:
            return None

    # Convert to seconds for plotting
    df_seconds = df.copy()
    for col in df.columns:
        if col not in ["Day", "DoW"]:
            df_seconds[col] = df[col].apply(time_to_seconds)

    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(16, 8))

    # Define distinct colors and markers for each player
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    markers = ["o", "s", "^", "D", "v", "p", "*"]

    # Plot each player with distinct styling (skip Backtracking)
    player_cols = [
        col for col in df_seconds.columns if col not in ["Day", "DoW", "Backtracking"]
    ]
    for idx, col in enumerate(player_cols):
        ax.plot(
            df_seconds["Day"],
            df_seconds[col],
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            label=col,
            linewidth=2.5,
            markersize=6,
            alpha=0.8,
        )

    # Improve x-axis readability
    x_ticks = df_seconds["Day"]
    x_labels = [
        f"{day}\n{dow}" for day, dow in zip(df_seconds["Day"], df_seconds["DoW"])
    ]

    # Show every 7th label to reduce crowding (weekly intervals)
    tick_spacing = 7
    ax.set_xticks(x_ticks[::tick_spacing])
    ax.set_xticklabels(x_labels[::tick_spacing], fontsize=10, rotation=0)

    # Set y-axis limit to 6 minutes for better visibility
    ax.set_ylim(0, 360)

    # Format y-axis ticks to mm:ss
    def format_mmss(x, _):
        if x < 0 or pd.isna(x):
            return ""
        m, s = divmod(int(x), 60)
        return f"{m}:{s:02}"

    ax.yaxis.set_major_formatter(FuncFormatter(format_mmss))

    # Increase y-axis tick frequency
    ax.set_yticks(range(0, 361, 30))  # Every 30 seconds

    # Labels and styling
    ax.set_xlabel("Day (Day of Week)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time (mm:ss)", fontsize=12, fontweight="bold")
    ax.set_title("Queens Scores Over Time", fontsize=16, fontweight="bold", pad=20)

    # Improve legend - place outside plot area
    ax.legend(
        title="Player",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=11,
        title_fontsize=12,
        framealpha=0.95,
        shadow=True,
        fancybox=True,
    )

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)  # Put grid behind data

    # Add subtle background color
    ax.set_facecolor("#f8f9fa")

    plt.tight_layout()

    # Save with high DPI for better quality
    plt.savefig("queens_scores_plot.png", dpi=300, bbox_inches="tight")
    print("Plot saved as queens_scores_plot.png")


def main():
    plot_graph()


if __name__ == "__main__":
    main()
