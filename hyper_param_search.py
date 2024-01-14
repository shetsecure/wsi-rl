import yaml
from utils import GymParams, TrainingParams
from itertools import product
from pathlib import Path

save_to_dir = Path("confs")
how_many_files_to_generate = 10
start_from = 20

# gym params possible values
accepted_patch_sizes = reversed([64, 128])
accepted_thumbnail_sizes = reversed([256, 512, 768, False])
accepted_max_eps_steps = [1_000]

# Train params possible_values
batch_size = [32, 64, 92, 128]
gamma = [0.9]
eps_start = [1]
eps_end = [0.01]
eps_decay = [0.001]
target_update = [10]
memory_size = reversed([1_500, 2_500, 5_000, 7_500])
lr = [0.001]
num_episodes = [5000]
saving_update = [100]

# ORDER MATTERS, DO NOT CHANGE
all_combos = list(
    product(
        accepted_patch_sizes,
        accepted_thumbnail_sizes,
        accepted_max_eps_steps,
        batch_size,
        gamma,
        eps_start,
        eps_end,
        eps_decay,
        target_update,
        memory_size,
        lr,
        num_episodes,
        saving_update,
    )
)

for i, combo in enumerate(all_combos):
    if i < start_from:
        continue

    if i + 1 == 10 + start_from:
        break

    gym_params = GymParams(*combo[:3])
    train_params = TrainingParams(*combo[3:])

    conf_name = (
        f"p_{gym_params.patch_size}_th_{gym_params.resize_thumbnail}"
        + f"_b{train_params.batch_size}_mem_{train_params.memory_size}"
        + f"_update_{train_params.target_update}.yaml"
    )

    gym_dict = gym_params._asdict()
    train_dict = train_params._asdict()

    config = {"gym": gym_dict, "train": train_dict}

    with open(save_to_dir / conf_name, "w") as conf_file:
        yaml.dump(
            config,
            conf_file,
            default_flow_style=False,
        )
