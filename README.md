# Pacman Planning Robot

A robot planning simulation where Pacman estimates its position, maps the environment using a laser sensor, and navigates to a goal while avoiding ghosts in a randomly generated maze.

<img width="1366" alt="Visual" src="figures/visual.png">
<hr>

## Setup

1.  **Install `uv`** (if not already installed):
    ```bash
    pip install uv
    ```

2.  **Sync dependencies**:
    ```bash
    uv sync
    ```

3.  **Activate the environment**:
    -   On Windows:
        ```bash
        .venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

4.  **Run the simulation**:
    ```bash
    uv run main.py
    ```

## Configuration

You can adjust the simulation parameters, such as window size, map dimensions, and game mechanics, in **`constants.py`**.

Key parameters include:
-   `NUM_GHOSTS`: Number of ghosts chasing Pacman.
-   `PING_PERIOD`: How often ghosts update their target (Pacman's position).
-   `N_JOBS`: Number of parallel threads used for path planning.

## How It Works

The simulation combines several robotic planning concepts:

-   **Path Estimation (EST)**: Pacman uses an Expanding Space Tree (EST) algorithm (similar to RRT) to explore the free space and find a path to the goal. This runs in parallel for Pacman and all ghosts.
-   **Mapping**: The robot builds a map of the environment using a **Log-Odds Occupancy Grid**. A simulated laser sensor casts rays to detect obstacles, updating the probability of each grid cell being occupied or free.
-   **Sensing**: The `LaserSensor` class simulates a 2D LiDAR, generating point clouds from the environment map.
-   **Navigation**: Pacman continually replans its path based on the current map and the estimated positions of ghosts.

---

This was for Caltech's ME 133b class final project.
Other contributors: Julian Peres (class of 2024), Isabela Ceniceros (class of 2025)