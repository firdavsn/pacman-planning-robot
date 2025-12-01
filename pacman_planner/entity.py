import numpy as np
from scipy.spatial      import KDTree
from math               import pi, sin, cos, atan2, sqrt, ceil
import random

from pacman_planner.env import Environment
from pacman_planner.node import Node
from pacman_planner.constants import *
from pacman_planner.utils import euclidean, get_round_grid, pixel_to_grid, grid_to_pixel

class Entity:
    def __init__(self, pos: tuple[int, int], goal: tuple[int, int], env: Environment, speed: int = .1, pacman: bool = True):
        """Initializes the pacman object.

        Args:
            pos (tuple[int, int]): initial position of the pacman
            speed (int, optional): speed of the pacman. Defaults to 1.
        """

        self.pos = pos
        self.speed = speed
        self.direction = None
        self.env = env

        self.start = pos
        self.goal = goal

        self.index = 0

        self.intermediate_points = SCAN_RESOLUTION

        self.costs = np.ones((WIDTH, HEIGHT)) * 1e9
        self.path = [pos]
        self.target_pos = None

        self.learn = True
        self.pacman = pacman
        
        if pacman:
            self.ghost_pos = [None for _ in range(NUM_GHOSTS)]
        else:
            self.ghost_pos = None
        
        self.traversed = np.zeros((np.size(env.walls, axis=0), np.size(env.walls, axis=1)))

        self.count = 0
    
    def update_pos(self, path):
        if path is None and self.target_pos is not None:
            # Get angle to target_pos
            angle = atan2(self.target_pos[1] - self.pos[1], self.target_pos[0] - self.pos[0])
            # Move pacman
            new_pos = (int(self.pos[0] + self.speed * cos(angle)), int(self.pos[1] + self.speed * sin(angle)))
            
            self.pos = new_pos
            self.path.append(new_pos)

        if path is not None:
            try:
                start = Node(self.pos[0], self.pos[1], self.env, learn=self.learn, pacman=self.pacman)
                end = Node(path[self.index + 1][0], path[self.index + 1][1], self.env, learn=self.learn, pacman=self.pacman)
            except IndexError:
                self.index = 0
                return

            # if we can no longer go this way, restart
            if not start.connectsTo(end):
                self.count += 1
                self.est()
                self.index = 0
                return

            curr_path = np.array(end.coordinates()) - np.array(start.coordinates())
            if np.linalg.norm(curr_path) > self.speed:
                new_pos = tuple(self.speed * curr_path / np.linalg.norm(curr_path) + np.array(self.pos))
            else:
                new_pos = end.coordinates()
                self.index += 1

            self.pos = new_pos
            pix = get_round_grid(new_pos)
            self.traversed[pix[0], pix[1]] = 1

    def est(self):
        startnode = Node(self.pos[0], self.pos[1], self.env, learn=self.learn, pacman=self.pacman)
        goalnode = Node(self.goal[0], self.goal[1], self.env, learn=self.learn, pacman=self.pacman)

        # Start the tree with the startnode (set no parent just in case).
        startnode.parent = None
        tree = [startnode]

        # Function to attach a new node to an existing node: attach the
        # parent, add to the tree, and show in the figure.
        def addtotree(oldnode, newnode, val = False):
            newnode.parent = oldnode
            tree.append(newnode)

        this_round_traversed = np.zeros(self.traversed.shape)

        def search_traversed(prev_node, new_spot, change, dist):
            last = True
            new_spot_grid = get_round_grid(new_spot.coordinates())

            this_round_traversed[new_spot_grid[0], new_spot_grid[1]] = 1

            for dx, dy in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                if (-dx, -dy) == change: # dont go back
                    continue

                trial_loc = tuple(np.array(new_spot_grid) + np.array([dx, dy]))

                if self.traversed[trial_loc[0], trial_loc[1]] == 1: # we have been that way at some point
                    pixel = grid_to_pixel(trial_loc)
                    nextnode = Node(pixel[0], pixel[1], self.env, learn=self.learn, pacman=self.pacman, distance_add=10000)
                    addtotree(new_spot, nextnode, val=True)

                    search_traversed(new_spot, nextnode, (dx, dy), dist + 1)
                    last = False

            if last: # allow us to actually expand from this node
                new_spot.distance_add = dist * 38

        # Loop - keep growing the tree.
        while True:
            # Determine the local density by the number of nodes nearby.
            # KDTree uses the coordinates to compute the Euclidean distance.
            # It returns a NumPy array, same length as nodes in the tree.
            X = np.array([node.coordinates() for node in tree])
            kdtree  = KDTree(X)
            numnear = kdtree.query_ball_point(X, r=1.5*self.speed, return_length=True)

            # Directly determine the distances to the goal node.
            goal_distances = np.array([node.distance(goalnode) for node in tree])
            ghost_distances = np.zeros(len(X))
            if self.pacman and NUM_GHOSTS > 0 and self.ghost_pos[0] is not None:
                ghost_nodes = [Node(pos[0], pos[1], self.env, self.learn, False) for pos in self.ghost_pos]
                ghost_distances = []
                for node in tree:
                    avg = 0
                    for ghost_node in ghost_nodes:
                        avg += node.distance(ghost_node)
                    avg /= NUM_GHOSTS
                    ghost_distances.append(avg)
                ghost_distances = np.array(ghost_distances)


            depths = np.array([node.depth for node in tree])

            new_metric = np.array([C_NEAR * numnear[i] + C_GOAL * goal_distances[i] + C_GHOST * ghost_distances[i] + 20 * depths[i] for i in range(len(X))])
            index     = np.argmin(new_metric)
            grownode  = tree[index]

            # Check the incoming heading, potentially to bias the next node.
            if grownode.parent is None:
                heading = 0
            else:
                heading = atan2(grownode.y - grownode.parent.y,
                                grownode.x - grownode.parent.x)
            

            # Find something nearby: keep looping until the tree grows.
            while True:
                angle = np.random.normal(heading, np.pi/2)
                nextnode = Node(grownode.x + self.speed*np.cos(angle), grownode.y + self.speed*np.sin(angle), self.env, learn=self.learn, pacman=self.pacman)
                
                # Try to connect.
                if nextnode.inFreespace() and (nextnode.connectsTo(grownode)):
                    addtotree(grownode, nextnode)

                    grid_spot = get_round_grid(nextnode.coordinates())
                    if self.traversed[grid_spot[0], grid_spot[1]] == 1 and this_round_traversed[grid_spot[0], grid_spot[1]] == 0:
                        old_spot = get_round_grid(grownode.coordinates())
                        search_traversed(grownode, nextnode, tuple(np.array(grid_spot) - np.array(old_spot)), 0)

                    break

            nextnode.depth = grownode.depth + 1

            # Once grown, also check whether to connect to goal.
            if nextnode.connectsTo(goalnode): # and nextnode.distance(goalnode) <= self.speed: # nextnode.distance(goalnode) <= self.speed and 
                addtotree(nextnode, goalnode)
                break

            # Check whether we should abort - too many nodes.
            if (len(tree) >= 15000):
                print("Aborted with the tree having %d nodes" % len(tree))
                return None

        # Build the path.
        path = [goalnode]
        while path[0].parent is not None:
            path.insert(0, path[0].parent)
        self.PostProcess(path)

        # return path
        pos_path = []
        for node in path:
            pos_path.append((node.x, node.y))

        self.est_path = pos_path
        self.tree = path

    # Post process the path.
    def PostProcess(self, path):
        i = 0
        while (i < len(path)-2):
            if path[i].connectsTo(path[i+2]):
                path.pop(i+1)
            else:
                i = i+1
    