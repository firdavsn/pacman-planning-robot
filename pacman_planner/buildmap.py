import numpy as np

from pacman_planner.utils import *
from pacman_planner.constants import *

class Map:
    def __init__(self):
        # Create the log-odds-ratio grid.
        self.logoddsratio = np.zeros((HEIGHT, WIDTH))

        self.old_prob = None

    def get_probs(self):
        """Get probabilities from the logoddsratio."""
        
        # Convert the log odds ratio into probabilities (0...1).
        probs = 1 - 1/(1+np.exp(self.logoddsratio))
        probs = probs.T

        if LMAX is not None:
            probs[probs > LMAX] = LMAX

        if self.old_prob is None:
            self.old_prob = probs
            changes = [(i, j) for i in range(WIDTH) for j in range(HEIGHT)]
        else:
            changes = np.argwhere(np.abs(probs - self.old_prob) > 0.01)
            self.old_prob = probs
        
        return probs, changes
    
    def adjust(self, u: int, v: int, delta: float):
        """Adjust the log odds ratio value."""
        
        # Update only if legal.
        if (u>=0) and (u<WIDTH) and (v>=0) and (v<HEIGHT):
            self.logoddsratio[v,u] += delta
        else:
            print("Out of bounds (%d, %d)" % (u,v))
    
    def laserCB(self, data: list[list[float, float, tuple[int, int]]], rmin: float, rmax: float):
        """

        Args:
            list[list[float, float, tuple[int, int], bool]]: list of laser scan data
                data[i] = [r, theta, laser_pos, is_obstacle]
        """
        
        # If no scanned data, return
        if not data:
            return
        
        xc, yc = data[0][2]

        # Convert the laser position to pixel coordinates
        xs = xc
        ys = yc
        
        for r, theta, _, is_obstacle in data:
            if rmin < r:
                l_occ = LOCCUPIED
                if not is_obstacle or r >= rmax:
                    l_occ = 0
                
                # Calculate the endpoint of the laser
                x_r = xc + r * np.cos(theta)
                y_r = yc - r * np.sin(theta)
                
                # Convert the endpoint to pixel coordinates
                xe = x_r
                ye = y_r
                
                # Set points from laser to endpoint as free
                for (u, v) in bresenham((xs, ys), (xe, ye)):
                    self.adjust(u, v, LFREE)
                
                # Set the endpoint as occupied
                self.adjust(int(xe), int(ye), l_occ)