import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_training_results(file_path, game_name):
    """Performs exploratory data analysis for a given game training dataset using Seaborn & Matplotlib."""

    # Load dataset
    df = pd.read_csv(file_path)

    # Ensure steps are sorted for smooth trend analysis
    df = df.sort_values(by="step")

    # Set Seaborn style
    sns.set_theme(style="darkgrid")

    # Plot CPU Usage Over Training Steps
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df["step"], y=df["cpu_percent"], label="CPU Usage (%)", color="blue")
    plt.xlabel("Training Step")
    plt.ylabel("CPU Usage (%)")
    plt.title(f"CPU Usage Over Training Steps - {game_name} (450K Steps)")
    plt.legend()
    plt.show()

    # Plot GPU Usage Over Training Steps
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        x=df["step"], y=df["gpu_usage_percent"], label="GPU Usage (%)", color="red"
    )
    plt.xlabel("Training Step")
    plt.ylabel("GPU Usage (%)")
    plt.title(f"GPU Usage Over Training Steps - {game_name} (450K Steps)")
    plt.legend()
    plt.show()

    # Plot Replay Buffer Growth Over Time
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        x=df["step"],
        y=df["buffer_usage_percent"],
        label="Replay Buffer Usage (%)",
        color="green",
    )
    plt.xlabel("Training Step")
    plt.ylabel("Buffer Usage (%)")
    plt.title(f"Replay Buffer Growth Over Training Steps - {game_name} (450K Steps)")
    plt.legend()
    plt.show()

    print(f"✅ Analysis for {game_name} complete!\n")
