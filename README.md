# VLMgineer: Vision Language Models as Robotic Toolsmiths

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Project Page](https://vlmgineer.github.io/) | [arXiv](https://arxiv.org/abs/2507.12644)


VLMgineer uses Vision-Language Models (VLMs) to automatically design tools and generate action sequences for robotic manipulation tasks


## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/vlmgineer/vlmgineer.git
cd vlmgineer

# Create conda environment
conda env create -f environment.yaml
conda activate vlmgineer

# Install package
pip install -e .
```

### API Key Setup

Create and source your API key:

```bash
echo 'export GEMINI_API="your-api-key"' > api_key.sh
source api_key.sh
```

## Supported Models

Our original experiments were run on **`gemini-2.5-pro-preview`**, and we have also tested on **`gemini-2.5-pro`**.

> **Note:** `gemini-3-pro` is not supported at the time being due to its API call frequency limits, which are too restrictive for the parallel sampling required by VLMgineer.

## Quick Start

### Run a Single Task

```bash
python scripts/run_task.py --task bring_cube_closer
```

### Run Benchmark

```bash
python scripts/run_benchmark.py
```

## Project Structure

```
vlmgineer/
├── src/vlmgineer/          # Main package
│   ├── common/             # Shared utilities
│   ├── configs/            # Task configurations
│   ├── envs/               # PyBullet environments
│   ├── evaluator/          # Evaluation and metrics
│   ├── models/             # URDF models
│   ├── prompts/            # VLM prompts
│   ├── runners/            # Task and environment runners
│   ├── samplers/           # VLM sampling agents
│   └── baseline/           # Baseline comparisons
├── scripts/                # Entry point scripts
└── tests/                  # Unit tests
```

## Available Tasks

- `bring_cube_closer` - Move a cube to a target position
- `elevate_plate` - Lift a plate to a target height
- `clean_table_top` - Clear objects from a table
- `collect_and_elevate_spheres` - Gather and lift spheres
- `dislodge_cube` - Dislodge a cube from a constrained space
- `serve_turkey_legs` - Serve turkey legs to a target location
- `take_cookie_out_from_jar` - Extract a cookie from a jar
- `move_ball` - Move a ball to a target position
- `puck_to_goal` - Push a puck to a goal
- `retrieve_high_object` - Retrieve an object from a high position
- `take_one_book_out` - Take one book out from a shelf
- `lift_box` - Lift a box to a target height

## Configuration

Task configs are in `src/vlmgineer/configs/`. Example:

```yaml
task_name: bring_cube_closer
model_name: gemini-2.5-pro
n_agent: 20
n_tool_samples_batch_size: 10
n_action_samples_batch_size: 10
save_top_k: 5
n_evolution: 3
```

## Citation

If you use VLMgineer in your research, please cite:

```bibtex
@article{vlmgineer,
  title={VLMgineer: Vision Language Models as Robotic Toolsmiths},
  author={George Jiayuan Gao and Tianyu Li and Junyao Shi and Yihan Li and Zizhe Zhang and Nadia Figueroa and Dinesh Jayaraman},
  journal={arXiv preprint arXiv:2507.12644},
  year={2025}
}
```
