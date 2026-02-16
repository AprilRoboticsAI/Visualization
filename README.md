# Visualization

## Setup

```bash
git clone https://github.com/AprilRoboticsAI/Visualization.git
cd Visualization
```

Install [uv](https://docs.astral.sh/uv/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install dependencies:

```bash
uv sync
```

## Usage

```bash
python visualize.py
```

You'll be prompted to select an episode. The script then replays it in Rerun, showing:

- Head and wrist camera feeds
- Hand skeleton keypoints in 3D
- Camera and wrist coordinate frames

## Files

- `visualize.py` — dataset loading, episode selection, and replay loop
- `example.py` — basic data loading example
- `scene_visualizer.py` — `SceneVisualizer` class and transform utilities
