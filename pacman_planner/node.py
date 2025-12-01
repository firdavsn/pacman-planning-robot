import numpy as np
from math     import pi, sin, cos, atan2, sqrt, ceil

from pacman_planner.buildmap import WIDTH, HEIGHT, MAZE_SIZE, RESOLUTION
from pacman_planner.utils    import grid_to_pixel, pixel_to_grid
from pacman_planner.constants import SCAN_RESOLUTION

class Node:
    def __init__(self, x, y, env, learn = False, pacman = True, distance_add = 0):
        self.parent = None

        self.x = x
        self.y = y

        self.env = env
        self.learn = learn
        self.pacman = pacman

        if learn:
            if pacman:
                self.walls = env.pacman_learned_walls
            else:
                self.walls = env.ghosts_learned_walls
        else:
            self.walls = env.walls
        
        self.distance_add = 0
        self.depth = 0

    def __repr__(self):
        return ("<Point %5.2f,%5.2f>" % (self.x, self.y))

    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                    self.y + alpha * (other.y - self.y),
                    self.env,
                    self.learn,
                    self.pacman)

    def coordinates(self):
        return (self.x, self.y)

    def distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2) + self.distance_add

    def inFreespace(self):
        if (self.x <= 0 or self.x >= WIDTH or
            self.y <= 0 or self.y >= HEIGHT):
            return False
        
        for dx in [-2, 0, 2]:
            for dy in [-2, 0, 2]:
                grid_point = pixel_to_grid((self.x + dx, self.y + dy))
                try:
                    if (self.walls[round(grid_point[0]), round(grid_point[1])] == 1):
                        return False
                except IndexError:
                    pass
        return True

    def connectsTo(self, other):
        for i in range(SCAN_RESOLUTION * 3):
            if not (self.intermediate(other, i/(SCAN_RESOLUTION * 3)).inFreespace()):
                return False
        return True

