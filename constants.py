COLORS = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0)
}


# Simulation Configuration
WIDTH = 800                    # Window width in pixels
HEIGHT = 800                   # Window height in pixels
MAZE_SIZE = 20                 # Size of the maze grid (e.g., 20x20)
RESOLUTION = (MAZE_SIZE + 1) / min(WIDTH, HEIGHT)

# Mapping Probabilities (Log Odds)
LFREE = -0.1                   # Log odds update for detecting free space
LOCCUPIED = 0.1                # Log odds update for detecting obstacles
LMAX = None                    # Maximum log odds value (optional)

# Path Planning Costs (positive: attract, negative: repel)
C_GOAL = 1                     # Cost weight for distance to goal
C_GHOST = -2                   # Cost weight (penalty) for proximity to ghosts
C_NEAR = 40                    # Cost weight to explore

# Simulation Parameters
SEED = 42                      # Random seed for map generation
RMIN = 10                      # Minimum laser range
RMAX = 100                     # Maximum laser range
SCAN_RESOLUTION = 30           # Number of points along each laser ray
HEADING_RESOLUTION = 120       # Number of rays in a full 360-degree scan

# Game Mechanics
PING_PERIOD = 10               # Frames between ghost position updates (pings)
CAUGHT_DISTANCE = 10           # Distance threshold for being caught

# Performance
N_JOBS = 6                     # Number of parallel jobs for processing