# Cleaning Robot Environment

This project implements a reinforcement learning environment for a cleaning robot using OpenAI's Gym framework. The environment simulates a grid where a robot can move, clean dirt, and create dirt under certain conditions.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/AISC10-SSO/cleaning_robot.git
    cd cleaning_robot_env
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -e .
    ```

## Usage

### Training the Agent

To train the agent, run the [train.py](http://_vscodecontentref_/10) script:
```sh
python cleaning_robot_env/train.py