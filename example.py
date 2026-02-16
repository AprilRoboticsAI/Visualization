import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

EPISODE_ID = 0
SAMPLE_ID = 10

dataset = LeRobotDataset(repo_id="aprilrobotics/sample", episodes=[EPISODE_ID])
frame = dataset[SAMPLE_ID]

# Camera images (H, W, 3)
head_img = frame["observation.images.head_cam"].numpy().transpose(1, 2, 0)
wrist_img = frame["observation.images.wrist_cam"].numpy().transpose(1, 2, 0)

# State: [head_pose(7), wrist_pose(7), hand_keypoints(63)]
#   pose = [x, y, z, qx, qy, qz, qw]
state = frame["observation.state"].numpy()
head_pose = state[0:7]  # position + quaternion (xyzw)
wrist_pose = state[7:14]  # position + quaternion (xyzw)

# 21 hand keypoints:
#   0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
keypoints = state[14:].reshape(21, 3)
