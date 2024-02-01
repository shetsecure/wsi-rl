# import multiprocessing
# import gymnasium as gym
# import yaml
# import utils
# import torch
# import time
# import h5py


# def run_environment(queue, gym_params, process_id, num_games):
#     try:
#         device = torch.device("cpu")
#         env = gym.make(
#             "gym_envs/WSIWorldEnv-v1",
#             wsi_path="test.ndpi",
#             render_mode=None,
#             patch_size=gym_params.patch_size,
#             resize_thumbnail=gym_params.resize_thumbnail,
#             max_episode_steps=gym_params.max_episode_steps,
#         )

#         all_transitions = []

#         start_time = time.time()
#         for game in range(num_games):
#             observation, info = env.reset()
#             done, truncated = False, False
#             while not done and not truncated:
#                 action = env.action_space.sample()
#                 next_observation, reward, done, truncated, info = env.step(action)
#                 # Store transition (state, action, reward, next_state, done)
#                 all_transitions.append(
#                     (observation, action, reward, next_observation, done)
#                 )
#                 observation = next_observation
#         end_time = time.time()

#         queue.put(all_transitions)

#         print(
#             f"Process {process_id} completed {num_games} games in {end_time - start_time:.2f} seconds"
#         )
#     except Exception as e:
#         print(f"Error in process {process_id}: {e}")
#         queue.put([])


# def save_transitions(all_transitions, filename):
#     with h5py.File(filename, "w") as f:
#         for i, (state, action, reward, next_state, done) in enumerate(all_transitions):
#             grp = f.create_group(f"transition_{i}")
#             # Save each component of the state and next state
#             for key, value in state.items():
#                 grp.create_dataset(f"state/{key}", data=value)
#             for key, value in next_state.items():
#                 grp.create_dataset(f"next_state/{key}", data=value)
#             grp.create_dataset("action", data=action)
#             grp.create_dataset("reward", data=reward)
#             grp.create_dataset("done", data=done)


# if __name__ == "__main__":
#     config = yaml.load(open("config.yaml"), Loader=yaml.SafeLoader)
#     gym_params = utils.GymParams(**config["gym"])

#     num_processes = 4
#     num_games_per_process = 12 // num_processes

#     queue = multiprocessing.Queue()
#     processes = []

#     for i in range(num_processes):
#         process = multiprocessing.Process(
#             target=run_environment, args=(queue, gym_params, i, num_games_per_process)
#         )
#         processes.append(process)
#         process.start()

#     # Collect all transitions from the queue
#     print("Collecting all transitions ...")
#     start_time = time.time()

#     all_transitions = []
#     for process in processes:
#         process.join(timeout=100)

#         if process.is_alive():
#             print(
#                 f"Process {process.name} did not complete in time and will be terminated."
#             )
#             process.terminate()
#             process.join()

#         while not queue.empty():
#             all_transitions.extend(queue.get())

#     end_time = time.time()
#     # Save all transitions to a file
#     print(f"Collected all transitions in {end_time - start_time:.2f} seconds")

#     start_time = time.time()
#     save_transitions(all_transitions, "transitions.hdf5")
#     end_time = time.time()
#     print(f"Saved all transitions in {end_time - start_time:.2f} seconds")

from multiprocessing import Pool, Manager
import gymnasium as gym
import yaml
import utils
import numpy as np
import os
import h5py


# {
#     "b_rect": array([-0.04795208, 0.47526938, 0.23724574, -0.23852383], dtype=float32),
#     "birdeye_view": array([146], dtype=uint8),
#     "current_view": array([235], dtype=uint8),
#     "level": array([1, 0, 1, 0], dtype=uint8),
#     "p_coords": array([-0.941912, -0.28077626], dtype=float32),
# }
def save_transitions(all_transitions, filename):
    with h5py.File(filename, "w") as f:
        for i, transition in enumerate(all_transitions):
            grp = f.create_group(f"transition_{i}")
            grp.create_dataset("state", data=transition[0])
            grp.create_dataset("action", data=transition[1])
            grp.create_dataset("reward", data=transition[2])
            grp.create_dataset("next_state", data=transition[3])
            grp.create_dataset("done", data=transition[4])


def init_h5_file(filename, max_episode_steps, num_episodes, obs_space_sample):
    with h5py.File(filename, "w") as h5file:
        for key, space in obs_space_sample.items():
            dtype = space.dtype
            shape = (num_episodes, max_episode_steps) + space.shape
            h5file.create_dataset(
                key, shape, maxshape=(None, None) + space.shape, dtype=dtype
            )

        print(h5file.keys())

        # h5file.create_dataset(
        #     "actions",
        #     (num_episodes, max_episode_steps),
        #     maxshape=(None, None),
        #     dtype="i",
        # )
        # h5file.create_dataset(
        #     "rewards",
        #     (num_episodes, max_episode_steps),
        #     maxshape=(None, None),
        #     dtype="f",
        # )
        # h5file.create_dataset(
        #     "dones", (num_episodes, max_episode_steps), maxshape=(None, None), dtype="i"
        # )
        # h5file.create_dataset(
        #     "truncated",
        #     (num_episodes, max_episode_steps),
        #     maxshape=(None, None),
        #     dtype="i",
        # )


def play_episode_batch(gym_params, batch_size, lock, file_name, start_idx, counter):
    np.random.seed(os.getpid())
    env = gym.make(
        "gym_envs/WSIWorldEnv-v1",
        wsi_path="test.ndpi",
        render_mode=None,
        patch_size=gym_params.patch_size,
        resize_thumbnail=gym_params.resize_thumbnail,
        max_episode_steps=gym_params.max_episode_steps,
    )

    for batch_idx in range(batch_size):
        episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "dones": [],
            "truncated": [],
        }

        observation = env.reset()
        for _ in range(gym_params.max_episode_steps):
            action = env.action_space.sample()
            next_observation, reward, done, truncated, info = env.step(action)

            episode_data["observations"].append(observation)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["next_observations"].append(next_observation)
            episode_data["dones"].append(done)
            episode_data["truncated"].append(truncated)

            observation = next_observation
            if done or truncated:
                break

        with lock:
            with h5py.File(file_name, "a") as h5file:
                idx = start_idx + batch_idx
                for key in obs_space_sample.keys():
                    h5file[key][idx, : len(episode_data["observations"])] = [
                        obs[key] for obs in episode_data["observations"]
                    ]
                h5file["actions"][idx, : len(episode_data["actions"])] = episode_data[
                    "actions"
                ]
                h5file["rewards"][idx, : len(episode_data["rewards"])] = episode_data[
                    "rewards"
                ]
                h5file["dones"][idx, : len(episode_data["dones"])] = episode_data[
                    "dones"
                ]
                h5file["truncated"][
                    idx, : len(episode_data["truncated"])
                ] = episode_data["truncated"]

        env.close()
        with counter.get_lock():
            counter.value += 1
            print(f"Completed episodes: {counter.value} / {num_episodes}")


def init_pool(l, c):
    global lock, counter
    lock, counter = l, c


if __name__ == "__main__":
    config = yaml.load(open("config.yaml"), Loader=yaml.SafeLoader)
    gym_params = utils.GymParams(**config["gym"])

    num_episodes = 10
    num_processes = 10  # Adjust based on your system's capability
    batch_size = num_episodes // num_processes

    file_name = "replay_buffer.h5"
    env_sample = gym.make(
        "gym_envs/WSIWorldEnv-v1",
        wsi_path="test.ndpi",
        render_mode=None,
        patch_size=gym_params.patch_size,
        resize_thumbnail=gym_params.resize_thumbnail,
        max_episode_steps=gym_params.max_episode_steps,
    )
    obs_space_sample = dict(env_sample.observation_space.sample())

    # 'state', data=transition[0])
    #             grp.create_dataset('action', data=transition[1])
    #             grp.create_dataset('reward', data=transition[2])
    #             grp.create_dataset('next_state', data=transition[3])
    #             grp.create_dataset('done', data=transition[4]
    all_transitions = [
        [
            obs_space_sample,
            env_sample.action_space.sample(),
            1000,
            dict(env_sample.observation_space.sample()),
            False,
        ]
    ]
    env_sample.close()
    save_transitions(all_transitions, "test.hdf5")

    exit()
    init_h5_file(
        file_name, gym_params.max_episode_steps, num_episodes, obs_space_sample
    )
    manager = Manager()
    lock = manager.Lock()
    counter = manager.Value("i", 0)

    tasks = [
        (gym_params, batch_size, lock, file_name, i * batch_size, counter)
        for i in range(num_processes)
    ]

    with Pool(num_processes, initializer=init_pool, initargs=(lock, counter)) as pool:
        pool.starmap(play_episode_batch, tasks)

    print("All episodes completed.")
