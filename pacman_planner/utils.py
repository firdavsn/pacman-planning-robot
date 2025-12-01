import numpy as np

from pacman_planner.constants import *

def bresenham(start: tuple[int, int], end: tuple[int, int]):
    """Return a list of all intermediate (integer) pixel coordinates from (start) to (end) coordinates (which could be non-integer)."""
    
    # Extract the coordinates
    (xs, ys) = start
    (xe, ye) = end

    # Move along ray (excluding endpoint).
    if (np.abs(xe-xs) >= np.abs(ye-ys)):
        return[(u, int(ys + (ye-ys)/(xe-xs) * (u+0.5-xs)))
                for u in range(int(xs), int(xe), int(np.sign(xe-xs)))]
    else:
        return[(int(xs + (xe-xs)/(ye-ys) * (v+0.5-ys)), v)
                for v in range(int(ys), int(ye), int(np.sign(ye-ys)))]
        
def grid_to_pixel(pos: tuple[int, int]) -> tuple[int, int]:
    """Converts grid coordinates to pixel coordinates."""
    
    return ((pos[1] + 0.5) / RESOLUTION, (pos[0] + 0.5) / RESOLUTION)

def pixel_to_grid(pos: tuple[int, int]) -> tuple[int, int]:
    """Converts pixel coordinates to grid coordinates."""
    
    return (pos[1] * RESOLUTION - 0.5, pos[0] * RESOLUTION - 0.5)
    
    # ((pos[1] + 0.5) / RESOLUTION, (pos[0] + 0.5) / RESOLUTION)

def get_round_grid(pos: tuple[int, int]) -> tuple[int, int]:
    """Converts pixel coordinates to grid coordinates."""
    
    return (round(pos[1] * RESOLUTION - 0.5), round(pos[0] * RESOLUTION - 0.5))
    
    # ((pos[1] + 0.5) / RESOLUTION, (pos[0] + 0.5) / RESOLUTION)

def calc_point_pos(distance: float, angle: float, robot_pos: tuple[int, int]) -> tuple[int, int]:
    """Calculates the position of a point away from robot.

    Args:
        distance (float): distace of point from robot
        angle (float): angle of point from robot
        robot_pos (tuple[int, int]): (x, y) position of robot in map
        
    Returns:
        tuple[int, int]: (x, y) position of point in map
    """
    
    x =  distance * np.cos(angle) + robot_pos[0]
    y = -distance * np.sin(angle) + robot_pos[1]
    return (round(x), round(y))

def euclidean(p1: tuple[int, int], p2: tuple[int, int]):
    """Calculates the euclidean distance between the sensor and a point."""
    
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def within_thresh(p1: tuple[int, int], p2: tuple[int, int], thresh: int):
    """Checks whether the distance between two points is within a threshold."""
    
    return euclidean(p1, p2) < thresh