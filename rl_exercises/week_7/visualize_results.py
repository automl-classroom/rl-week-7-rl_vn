import os

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from rliable import library as rly
from rliable import metrics, plot_utils
from scipy.interpolate import interp1d

results_dir = os.path.join(os.path.dirname(__file__), "results")

# Number of seeds
n_seeds = 5

# Algorithms to compare
algorithms = ["DQN", "RND_DQN"]

# Load DQN data
df_dqn_s0 = pd.read_csv(os.path.join(results_dir, "dqn_training_data_seed_0.csv"))
df_dqn_s1 = pd.read_csv(os.path.join(results_dir, "dqn_training_data_seed_1.csv"))
df_dqn_s2 = pd.read_csv(os.path.join(results_dir, "dqn_training_data_seed_2.csv"))
df_dqn_s3 = pd.read_csv(os.path.join(results_dir, "dqn_training_data_seed_3.csv"))
df_dqn_s4 = pd.read_csv(os.path.join(results_dir, "dqn_training_data_seed_4.csv"))

# Load RND_DQN data
df_rnd_s0 = pd.read_csv(os.path.join(results_dir, "rnd_dqn_training_data_seed_0.csv"))
df_rnd_s1 = pd.read_csv(os.path.join(results_dir, "rnd_dqn_training_data_seed_1.csv"))
df_rnd_s2 = pd.read_csv(os.path.join(results_dir, "rnd_dqn_training_data_seed_2.csv"))
df_rnd_s3 = pd.read_csv(os.path.join(results_dir, "rnd_dqn_training_data_seed_3.csv"))
df_rnd_s4 = pd.read_csv(os.path.join(results_dir, "rnd_dqn_training_data_seed_4.csv"))

# Add seed identifiers
df_dqn_s0["seed"] = 0
df_dqn_s1["seed"] = 1
df_dqn_s2["seed"] = 2
df_dqn_s3["seed"] = 3
df_dqn_s4["seed"] = 4
df_rnd_s0["seed"] = 0
df_rnd_s1["seed"] = 1
df_rnd_s2["seed"] = 2
df_rnd_s3["seed"] = 3
df_rnd_s4["seed"] = 4

# Combine dataframes
df_dqn = pd.concat(
    [df_dqn_s0, df_dqn_s1, df_dqn_s2, df_dqn_s3, df_dqn_s4], ignore_index=True
)
df_rnd = pd.concat(
    [df_rnd_s0, df_rnd_s1, df_rnd_s2, df_rnd_s3, df_rnd_s4], ignore_index=True
)

# Get a common set of frames (use seed 0's frames as reference)
frames_dqn = df_dqn_s0["frames_Epsilon"].to_numpy()
frames_rnd = df_rnd_s0["frames_RND"].to_numpy()

# Create a common frame grid
# Use the union of frames, sorted, and interpolate
all_frames = np.sort(np.unique(np.concatenate([frames_dqn, frames_rnd])))

# Initialize arrays to store rewards
dqn_rewards = np.zeros((n_seeds, len(all_frames)))
rnd_rewards = np.zeros((n_seeds, len(all_frames)))

# Interpolate rewards for each seed to all frames
for seed in range(n_seeds):
    # DQN
    df_seed = df_dqn[df_dqn["seed"] == seed]
    frames = df_seed["frames_Epsilon"].to_numpy()
    rewards = df_seed["AvgRewardEpsilon(10)"].to_numpy()
    interp_func = interp1d(
        frames, rewards, bounds_error=False, fill_value="extrapolate"
    )
    dqn_rewards[seed] = interp_func(all_frames)

    # RND_DQN
    df_seed = df_rnd[df_rnd["seed"] == seed]
    frames = df_seed["frames_RND"].to_numpy()
    rewards = df_seed["AvgRewardRND(10)"].to_numpy()
    interp_func = interp1d(
        frames, rewards, bounds_error=False, fill_value="extrapolate"
    )
    rnd_rewards[seed] = interp_func(all_frames)

# Combine into a dictionary for rliable
train_scores = {"DQN": dqn_rewards, "RND_DQN": rnd_rewards}


# Define IQM metric
def iqm(scores):
    return np.array(
        [
            metrics.aggregate_iqm(scores[:, eval_idx])
            for eval_idx in range(scores.shape[-1])
        ]
    )


# Compute IQM and confidence intervals
iqm_scores, iqm_cis = rly.get_interval_estimates(
    train_scores,
    iqm,
    reps=2000,
)

# Plot sample efficiency curve

# Set global Matplotlib parameters
plt.rcParams.update(
    {
        "font.size": 10,  # Base font size
        "axes.labelsize": 12,  # Axis labels
        "axes.titlesize": 14,  # Title
        "xtick.labelsize": 10,  # X-axis tick labels
        "ytick.labelsize": 10,  # Y-axis tick labels
        "legend.fontsize": 10,  # Legend font size
    }
)


plot_utils.plot_sample_efficiency_curve(
    all_frames,
    iqm_scores,
    iqm_cis,
    algorithms=algorithms,
    xlabel="Steps",
    ylabel="IQM Average Reward",
    colors=dict(zip(algorithms, sns.color_palette("colorblind"))),
)

# Customize marker sizes
for line in plt.gca().get_lines():
    line.set_markersize(2)


# Create legend patches
fake_patches = [
    mpatches.Patch(color=sns.color_palette("colorblind")[i], alpha=1)
    for i in range(len(algorithms))
]

plt.legend(
    fake_patches, algorithms, loc="upper left", fancybox=True, ncol=1, fontsize=10
)

plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(results_dir, "IQM_Sample_Efficiency_Curve.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"RLiable analysis saved to {plot_path}")

# Display the plot
plt.show()
