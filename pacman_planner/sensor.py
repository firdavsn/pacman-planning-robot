import pygame as pg
import math
import numpy as np

from pacman_planner.utils import euclidean

class LaserSensor:
    """The laser sensor class."""
    
    def __init__(self, map: pg.Surface, uncertainty: tuple[float, float], dims: tuple[int, int], rmin: float, rmax: float, scan_resolution: int = 100, heading_resolution: int = 60):
        """Initializes the laser sensor.
        
        Args:
            map (pg.Surface): map image
            uncertainty (tuple[float, float]): standard deviation of distance and angle
            dims (tuple[int, int]): (width, height) of the map window
            rmin (float, optional): minimum range of the laser.
            rmax (float, optional): maximum range of the laser.
        """
        
        self.map = map
        self.pos = (0, 0)
        self.w, self.h = dims
        
        self.rmin = 0 # temporarily set to 0 until fixed
        
        self.rmax = rmax
        self.sigma = np.array(uncertainty)
        self.scan_resolution = scan_resolution
        self.heading_resolution = heading_resolution
        
        self.sensed_obstacles = []
    
    def add_uncertainty(self, distance: float, angle: float, sigma) -> list[float, float]:
        """Adds uncertainty to the distance and angle measurements.
        
        Args:
            distance (float): distance measurement
            angle (float): angle measurement
            sigma (tuple[float, float]): standard deviation of distance and angle
        
        Returns:
            list[float, float]: distance and angle with added uncertainty
        """
        
        mean = np.array([distance, angle])
        covariance = np.diag(sigma**2)
        distance, angle = np.random.multivariate_normal(mean, covariance)
        distance = max(distance, 0)
        angle = max(angle, 0)
        return [distance, angle]
    
    def scan(self):
        """Scans the environment for obstacles and returns the data.
        
        Returns:
            list[list[float, float, tuple[int, int], bool]]: list of laser scan data
                data[i] = [distance, angle, robot_pos, is_obstacle]
                False if no obstacles are found
        """
        
        data = []
        
        # Iterate through all headings
        for theta in np.linspace(0, 2*math.pi, self.heading_resolution, False):
            target = (self.pos[0] + self.rmax * math.cos(theta), 
                      self.pos[1] - self.rmax * math.sin(theta))
            
            # Iterate through all points along the ray
            for i in range(self.scan_resolution):
                u = i/self.scan_resolution
                x = round(target[0] * u + self.pos[0] * (1-u))
                y = round(target[1] * u + self.pos[1] * (1-u))
                
                # Check if the point is within the map
                if 0 < x < self.w and 0 < y < self.h:
                    distance = euclidean((x, y), self.pos)
                    output = self.add_uncertainty(distance, theta, self.sigma)
                    output.append(self.pos)
                    
                    if distance > self.rmin:
                        # Check if the point is an obstacle
                        color = tuple(self.map[x, y])
                        if color == (0, 0, 0):

                            prev_u = (i - 1)/self.scan_resolution
                            prev_x = round(target[0] * prev_u + self.pos[0] * (1-prev_u))
                            prev_y = round(target[1] * prev_u + self.pos[1] * (1-prev_u))

                            central_point = np.array([prev_x, prev_y])
                            perc = .03
                            while tuple(self.map[int(central_point[0]), int(central_point[1])]) != (0, 0, 0):
                                central_point = np.round(central_point + perc * (np.array([x, y]) - central_point))
                                perc += .03

                            distance = euclidean((central_point[0], central_point[1]), self.pos)
                            output = self.add_uncertainty(distance, theta, self.sigma)
                            output.append(self.pos)

                            output.append(True)
                            data.append(output)
                            break
                        
                    if i == self.scan_resolution - 1:
                        output.append(False)
                        data.append(output)
                        break

        if len(data) > 0:
            return data
        else:
            return False