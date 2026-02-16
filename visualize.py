import time

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from scene_visualizer import SceneVisualizer, pose7_to_T


def select_episode(meta):
    n = meta.total_episodes
    print(f"\nAvailable episodes ({n} total):")
    for i in range(n):
        ep = meta.episodes[i]
        length = int(ep["length"])
        task = ep["tasks"][0]
        print(f"  [{i}] {length} frames  task={task!r}")

    while True:
        try:
            choice = int(input(f"\nSelect episode [0-{n - 1}]: "))
            if 0 <= choice < n:
                return choice
        except (ValueError, EOFError):
            pass
        print(f"Please enter a number between 0 and {n - 1}.")


REPO_ID = "aprilrobotics/sample"


def main():
    meta = LeRobotDatasetMetadata(repo_id=REPO_ID)
    episode = select_episode(meta)

    dataset = LeRobotDataset(repo_id=REPO_ID, episodes=[episode])
    fps = meta.fps
    print(f"Loaded {len(dataset)} frames at {fps} FPS (episode {episode})")

    vis = SceneVisualizer()

    dt = 1.0 / fps
    print("Replaying... Press Ctrl+C to stop.")
    try:
        for i in range(len(dataset)):
            t0 = time.monotonic()
            item = dataset[i]

            state = item["observation.state"].numpy()
            T_world_cam = pose7_to_T(state[0:7])
            T_world_wrist = pose7_to_T(state[7:14])
            keypoints = state[14:].reshape(21, 3)

            head_img = item["observation.images.head_cam"].permute(1, 2, 0).numpy()
            wrist_img = item["observation.images.wrist_cam"].permute(1, 2, 0).numpy()

            if head_img.dtype != np.uint8:
                head_img = (head_img * 255).clip(0, 255).astype(np.uint8)
            if wrist_img.dtype != np.uint8:
                wrist_img = (wrist_img * 255).clip(0, 255).astype(np.uint8)

            vis.log_head_cam(T_world_cam, head_img)
            vis.log_wrist_cam(T_world_wrist, wrist_img)
            vis.log_hand_keypoints(keypoints)

            print(f"\rFrame {i + 1}/{len(dataset)}", end="")

            elapsed = time.monotonic() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
