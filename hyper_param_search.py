import yaml
from utils import GymParams, TrainingParams
from itertools import product
from pathlib import Path

save_to_dir = Path("confs")
how_many_files_to_generate = 15
start_from = 0

# gym params possible values
accepted_patch_sizes = [128]
accepted_thumbnail_sizes = [512]
accepted_max_eps_steps = [1_000]

# Train params possible_values
batch_size = [128, 256, 512]
gamma = [0.9]
eps_start = [1]
eps_end = [0.01]
eps_decay = [0.001]
target_update = [10]
memory_size = [2_500, 5_000, 7_500]
lr = [1e-4]
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

print(len(all_combos))

for i, combo in enumerate(all_combos):
    if i < start_from:
        continue

    if i == how_many_files_to_generate + start_from:
        break

    gym_params = GymParams(*combo[:3])
    train_params = TrainingParams(*combo[3:])

    conf_name = (
        f"c{i}_p_{gym_params.patch_size}_th_{gym_params.resize_thumbnail}"
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
