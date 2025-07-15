# AXH-SonarFish

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Azure ML](https://img.shields.io/badge/Azure-ML-0078D4?logo=microsoft-azure)](https://azure.microsoft.com/en-us/services/machine-learning/)
[![OpenCV](https://img.shields.io/badge/OpenCV-computer%20vision-5C3EE8?logo=opencv)](https://opencv.org/)

# Introduction 
The fish sonar tracks fish and floating debris in front of run-of-river powerplants and classifies the trajectories into either fish or object. Based on the tracked fish, a bypass in the run-of-river powerplant can be opened to allow the fish to travel downstream safely. The current version of the fishsonar is not set up for real time classification of trajectories but instead runs tracking, saves trajectories, and then classifies the trajectories in batch.

### Powerplant Stroppel Example Results
<p align="center">
  <img src="data/sample_tracking/stroppel.gif" alt="Fish Detection Demo" width="900">
</p>

# Environment Setup
Create a virtual environment with uv:
  ```bash
  uv venv --python 3.11 .venv
  ```
Install dependencies:
  ```bash
  source .venv/bin/activate
  uv pip install -r requirements.txt
  ```
Install `ffmpeg` for compression of the mp4 (this might take several minutes):
  ```bash
  # Ubuntu/Debian
  sudo apt install ffmpeg
  
  # macOS with Homebrew
  brew install ffmpeg
  ```
optional: Add .env file with the following contents in the top level folder:
  ```bash
    RESOURCE_GROUP=<your-azure-resource-group>
    WORKSPACE_NAME=<your-azure-ml-workspace>
    SUBSCRIPTION_ID=<your-azure-subscription-id>
  ```

# Running the algorithm

### Run on a sample video
You can now run the `algorithm/run_algorithm.py` file in debugging mode in VS Code or with:
```bash
python algorithm/run_algorithm.py --yaml_file settings/settings_stroppel.yaml
```
The processing oftentimes takes some seconds to start. Once the video pops up, it will take several seconds more of processing video material for tracks to show up. These tracks do not contain labels since tracking and classification are split into two different steps.

### Run on a batch of files and evaluating the output on Azure-ML
You can run the tracking and classification algorithms as Azure-ML pipelines with the [run_on_azure/](run_on_azure/launch_kalman_tracking_azure.ipynb) notebook. The notebook has different pipelines that either launch only tracking, tracking and classification, or tracking, classification, and labeling of videos with labeled trajectories. For classification you will need ground truth data for training the classification model and reference the correct path on your Azure ML workspace.

# Structure

<details>
<summary><b>Click to expand project structure</b></summary>

The codebase is structured into the following sections:
- **algorithm:** The fish detection algorithm, taking a video file as an input and giving .csv / visual / video 
  output for trajectories. 
- **analysis:** All code associated with the classification of trajectories into the categories fish and object.
- **continous_operation (deprecated):** Code pertaining to the setup, initialization and continuous operation of the 
  Raspberry Pi for tracking in Stroppel. This part is deprecated since for Lavey and all future sonar installations, a more professional setup for video capture and storage will be used.
- **run_on_azure:** definitions of Azure-ML pipelines and launch notebook to run tracking and classification on a large number of videos in a blob storage. We have run analysis on up to 80'000 videos, the equivalent of 1 year of continuous video material.

### Data Structure
The data structure is as follows:

```
data/
├── labels/
├── model_output/
└── raw/
    ├── videos/
    └── labels/
```

- **model_output**: Contains the output of the fish detection algorithm as video and csv. 
- **raw**: contains the raw video files.
  - **videos**: Contains the raw video files.
  - **labels**: Contains the labels video files. If a csv with labeled fish tracks exists in this directory, they are read and displayed in the output video. This way videos with labeled trajectories can be generated

</details>

# How to Contribute

<details>
<summary><b>Click to expand contribution guidelines</b></summary>

We welcome contributions to the Fish Sonar project! Here's how you can help:

### Getting Started
1. Fork the repository and create your feature branch from `main`
2. Set up the development environment following the [Environment Setup](#environment-setup) instructions
3. Install pre-commit hooks to ensure code quality:
   ```bash
   pre-commit install
   ```
4. Install dev requirements
    ```bash
    uv pip install -r requirements.dev.txt
    ```

### Running Tests
For now, tests have to be executed from the tests folder. This is due to the fact that the algorithm relies on relative paths, and this behaviour should be tested in the tests. Use this command to run tests:
  ```bash
  export PYTHONPATH=${PYTHONPATH}:$(pwd); cd tests; pytest
  ```

### Development Guidelines
- **Code Style**: We use [Black](https://github.com/psf/black) for Python formatting, [isort](https://github.com/PyCQA/isort) for import sorting, and [Flake8](https://flake8.pycqa.org/) for linting. These are automatically enforced through pre-commit hooks
- **Testing**: Add tests for new features in the [`tests/`](tests/) directory. Run tests with:
  ```bash
  export PYTHONPATH=${PYTHONPATH}:$(pwd); cd tests; pytest
  ```
- **Documentation**: Update the README and add docstrings to new functions and classes

### Making a Pull Request
1. Ensure your code passes all pre-commit checks
2. Write clear commit messages describing your changes
3. Create a pull request with a description of what you've done
4. Link any relevant issues in your PR description

### Reporting Issues
- Use GitHub Issues to report bugs or suggest features
- Include relevant information: Python version, error messages, and steps to reproduce

### Areas We Need Help
- Improving real-time classification capabilities
- Enhancing trajectory classification accuracy
- Documentation improvements
- Test coverage expansion
- Performance optimizations for video processing

</details>

# Creating Ground Truth Data
The easiest way to create ground truth data is to run the tracking on the videos you want to use to extract ground truth trajectories. You can run the tracking and select the dual output mode in the tracking settings yaml. This will save a single video with two versions of the sonar video next to eachother, the original and the video with tracks and ids displayed. Noting down the tracks of fish in a csv can then be joined onto the output track csv to have tracks with labels.

An alternative approach is the use of a video editing tool that allows to track a trajectory with your mouse pointer. We have found this approach more cumbersome than the one described above.

# Acknowledgements
Development of this project was financed by the Swiss Bundesamt für Umwelt BAFU. Further details can be found in the [notice file](NOTICE).