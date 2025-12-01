import math 
import pygame as pg
import numpy as np
from mazelib import Maze
import time
from mazelib.generate.Prims import Prims
from mazelib.solve.BacktrackingSolver import BacktrackingSolver as BackTracker
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from mazelib.solve.BacktrackingSolver import BacktrackingSolver
import random

from pacman_planner.utils import grid_to_pixel, calc_point_pos, pixel_to_grid
from pacman_planner.constants import *
from pacman_planner.node import Node

class Environment:
    """The environment for the map and point cloud generated from a laser 
    scan.
    """
    
    def __init__(self, dims, seed=1, map_size=20, loops=1):
        """Initializes the map environment.

        Args:
            dims (tuple[int, int]): (width, height) of the map window
            seed (int, optional): seed for the random number generator. Defaults to 1.
            map_size (int, optional): size of the map. Defaults to 20.
        """
        
        # Initialize variables
        self.seed = seed
        self.map_size = map_size
        self.walls = None
        self.start = None
        self.goal = None
        self.ghost_pos = [None for _ in range(NUM_GHOSTS)]
        self.paths = [[] for _ in range(NUM_GHOSTS + 1)] # pacman, ghost1, ghost2, ...
        self.point_clouds = [[] for _ in range(NUM_GHOSTS + 1)]
        self.map_file = os.path.join(os.path.dirname(__file__), 'maps', f'map_{seed}_{map_size}.png')

        # Generate the maze and map image
        self.generate_maze(loops)
        self.generate_map_img()

        # Initialize pygame
        pg.init()
        pg.display.set_caption("map")
        self.map = pg.display.set_mode(dims, pg.RESIZABLE)
        self.original_map = pg.display.set_mode(dims, pg.RESIZABLE)

        # Load map image
        self.map_img = pg.image.load(self.map_file)
        self.map_img_arr = pg.surfarray.array3d(self.map_img)

        # Draw the map
        self.map.blit(self.map_img, (0, 0))

        # Draw goal
        pg.draw.circle(self.map, COLORS['green'], self.goal, 25)

        # Learning walls for EST
        self.pacman_learned_walls = np.zeros((np.size(self.walls, axis=0), np.size(self.walls, axis=1))) - 1
        self.ghosts_learned_walls = np.zeros((np.size(self.walls, axis=0), np.size(self.walls, axis=1))) - 1

        self.show_learned_maps()

    def generate_maze(self, loops):
        """Generates a maze and sets the walls, start, and goal."""
        
        # Set the seed
        Maze.set_seed(self.seed)

        # Create a maze object
        m = Maze()
        m.generator = Prims(self.map_size // 2, self.map_size // 2)
        m.generate()

        # Set the walls
        self.walls = m.grid
        rows  = np.size(self.walls, axis=0)
        cols  = np.size(self.walls, axis=1)
        
        # Generates start and end, if want them on outer wall its true
        m.generate_entrances(True, True)
        start = m.start
        goal = m.end

        # force walls and holes
        if loops != 0:
            self.create_maze_loops(loops)

        # Randomly select ghost positions
        indices = np.where(self.walls == 0)
        indices_list = list(zip(indices[0], indices[1]))

        for i in range(NUM_GHOSTS):
            random_index = random.choice(indices_list)
            self.ghost_pos[i] = grid_to_pixel(random_index)
            indices_list.remove(random_index)

        # Make sure the start and end not on outer wall
        if start[0] == 0:
            start = (start[0] + 1, start[1])
        elif start[0] == rows - 1:
            start = (start[0] - 1, start[1])

        if start[1] == 0:
            start = (start[0], start[1] + 1)
        elif start[1] == cols - 1:
            start = (start[0], start[1] - 1)

        if goal[0] == 0:
            goal = (goal[0] + 1, goal[1])
        elif goal[0] == rows - 1:
            goal = (goal[0] - 1, goal[1])

        if goal[1] == 0:
            goal = (goal[0], goal[1] + 1)
        elif goal[1] == cols - 1:
            goal = (goal[0], goal[1] - 1)

        self.start = grid_to_pixel(start)
        self.goal = grid_to_pixel(goal)

        #IF NEEDED, GENERALLY SHOULDNT BE USED
        # Solve the maze
        # m.solver = BackTracker()
        # m.solve()

        # # Get the path
        # # self.true_path = m.solutions[0]
        # self.true_path = []
        # for i in range(len(m.solutions[0])):
        #     self.true_path.append(grid_to_pixel(m.solutions[0][i]))

    def create_maze_loops(self, loops):
        np.random.seed(self.seed) # no seed doesn't shuffle for some reason

        central_points = []
        rows  = np.size(self.walls, axis=0)
        cols  = np.size(self.walls, axis=1)

        for r in range(2, rows - 1):
            for c in range(2, cols - 1):
                subgraph = self.walls[r - 1: r + 2, c - 1: c + 2]
                if np.array_equal(subgraph[1], [1, 1, 1]):
                    if subgraph[0, 1] == 0 and subgraph[2, 1] == 0:
                        central_points.append((r, c))
                elif np.array_equal(subgraph[:, 1], [1, 1, 1]):
                    if subgraph[1, 0] == 0 and subgraph[1, 2] == 0:
                        central_points.append((r, c))

        np.random.shuffle(central_points)
        loops = min(loops, len(central_points))

        remove = central_points[:loops]

        for points in remove:
            self.walls[points[0], points[1]] = 0


    def generate_map_img(self, display=False):
        """Generates a map image and saves it to a file."""
        
        # Create an image of the maze
        colormap = colors.ListedColormap(["white", "black"])
        plt.figure(0, figsize=(8, 8))
        plt.imshow(self.walls, cmap=colormap)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')
        plt.savefig(self.map_file)

        if not display:
            plt.close()
        else:
            plt.show()

    def show_learned_maps(self, display=True):
        """Shows the learned maps one above the other."""

        # setup figure 1 to dispaly what the pacman and ghosts have learned
        fig = plt.figure(1, figsize=(4, 8))
        plt.clf()

        colormap = colors.ListedColormap(["green", "white", "black"])

        ax1 = plt.subplot(2, 1, 1)
        img1 = ax1.imshow(self.pacman_learned_walls, cmap=colormap)
        plt.title('Pacman Learned Map')
        plt.axis('off')

        ax2 = plt.subplot(2, 1, 2)
        img2 = ax2.imshow(self.ghosts_learned_walls, cmap=colormap)
        plt.title('Ghosts Learned Map')
        plt.axis('off')

        plt.subplots_adjust(left=0, right=1, top=.95, bottom=0)

        img1.set_data(self.pacman_learned_walls)
        img2.set_data(self.ghosts_learned_walls)
        plt.draw()

        if display:
            plt.show(block=False)
        else:
            plt.close()

    
    def process_data(self, data: list[list[float, float, tuple[int, int]]], player: int = 0):
        """Processes the laser scan data and stores it in the point cloud.

        Args:
            data (list[list[float, float, tuple[int, int], bool]]): list of laser scan data
                data[i] = [distance, angle, robot_pos, is_obstacle]
            player: 0 for pacman, 1 for ghost1, 2 for ghost2
        """
        # Reset point cloud
        for point in self.point_clouds[player]:
            self.map.set_at(point, self.map_img_arr[point[0], point[1]])
        self.point_clouds[player] = []
        
        colors = [COLORS['yellow'], COLORS['green'], COLORS['blue']]

        for distance, angle, robot_pos, is_obstacle in data:
            # Calculate the position of the point
            point = calc_point_pos(distance, angle, robot_pos)
            
            if point not in self.point_clouds[player] and is_obstacle:
                self.point_clouds[player].append(point)

            if robot_pos not in self.paths[player]:
                self.paths[player].append(robot_pos)
                # if len(self.paths[player]) > 1:
                #     pg.draw.line(self.map, colors[player], self.paths[player][-2], self.paths[player][-1], 3)
                #     pg.draw.circle(self.map, COLORS['blue'], self.paths[player][-1], 5)
                #     pg.draw.circle(self.map, COLORS['white'], self.paths[player][-2], 5)

    def show(self, probs: np.ndarray = None, changes: np.ndarray = None, loc: tuple = None, player: int = 0):
        """Shows the map image with the point cloud."""

        to_update = []

        if loc is not None:
            player_grid = pixel_to_grid(loc)
            player_grid = (round(player_grid[0]), round(player_grid[1]))

            to_update.append(((player_grid[0], player_grid[1]), 0))

        # Draw the probs
        if probs is not None and changes is not None:
            for (i, j) in changes:
                if probs[i, j] > 0:
                    color = np.array([255, 255, 255]) * (1 - probs[i, j])
                    self.map.set_at((i, j), color)

        # Draw point cloud
        for point in self.point_clouds[player]:
            self.map.set_at(point, COLORS['red'])

            if loc is not None:
                p1, p2 = self.get_border_grid(point)

                # ingore edges cus they cause issues and arent necessary
                if p1 is None:
                    continue

                d1 = np.linalg.norm(np.array(p1) - np.array(player_grid))
                d2 = np.linalg.norm(np.array(p2) - np.array(player_grid))

                if d1 > d2:
                    to_update.append(((p1[0], p1[1]), 1))
                    to_update.append(((p2[0], p2[1]), 0))
                else:
                    to_update.append(((p1[0], p1[1]), 0))
                    to_update.append(((p2[0], p2[1]), 1))

        for (x, y), val in to_update:
            if player != 0:
                self.ghosts_learned_walls[x, y] = val
            else:
                self.pacman_learned_walls[x, y] = val
        
        colors = [COLORS['yellow'], COLORS['green'], COLORS['blue']]
        
        # Draw player path
        if len(self.paths[player]) > 1:
            for i in range(len(self.paths[player])-1):
                pg.draw.line(self.map, colors[player], self.paths[player][i], self.paths[player][i+1], 3)
                # pg.draw.circle(self.map, COLORS['blue'], self.paths[player][i+1], 5)
                # pg.draw.circle(self.map, COLORS['white'], self.paths[player][i], 5)

        # Draw goal
        pg.draw.circle(self.map, COLORS['green'], self.goal, 20)

        # Update the display
        pg.display.flip()
    
    def get_border_grid(self, loc):
        valid_pixels = [round(grid_to_pixel((0, i + .5))[0]) for i in range(MAZE_SIZE + 1)]
        
        vert_border = -1
        horiz_border = -1

        for i, val in enumerate(valid_pixels):
            if abs(loc[0] - val) <= 2:
                vert_border = i
            if abs(loc[1] - val) <= 2:
                horiz_border = i
        
        # should be on one
        if vert_border == -1 and horiz_border == -1:
            # print('this shouldn\'t happen')
            return None, None

        # if is an edge, ignore
        if vert_border != -1 and horiz_border != -1:
            return None, None

        if vert_border != -1:
            grid_point_x = round(pixel_to_grid(loc)[0])
            p1 = (grid_point_x, vert_border)
            p2 = (grid_point_x, vert_border + 1)
        else:
            grid_point_y = round(pixel_to_grid(loc)[1])
            p1 = (horiz_border, grid_point_y)
            p2 = (horiz_border + 1, grid_point_y)
        
        return [p1, p2]