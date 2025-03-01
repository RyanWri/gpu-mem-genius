import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("example_dataset.csv")


def plot_episode_reward(df):
    plt.plot(df["episode_reward"], df.index, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Agent Performance Over Episodes")
    plt.show()
