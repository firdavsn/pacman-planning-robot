import pygame as pg
import numpy as np
import time
from joblib import Parallel, delayed
import dotenv

import pacman_planner.sensor as sensor
import pacman_planner.env as env
import pacman_planner.buildmap as buildmap
from pacman_planner.constants import *
from pacman_planner.utils import within_thresh
from pacman_planner.entity import Entity

import warnings
warnings.filterwarnings("ignore")

# dotenv.load_dotenv('.emv')

def main():
    """
    Main simulation loop for the Pacman planning robot.
    
    Initializes the environment, sensors, and entities (Pacman and ghosts),
    then runs the simulation loop handling:
      - Path estimation (EST) using joblib for parallelism.
      - Sensor scanning and map updates.
      - Pygame event handling and rendering.
      - Collision detection and goal checking.
    """
    # --- Initialization ---
    env_  = env.Environment((WIDTH, HEIGHT), seed=SEED, map_size=MAZE_SIZE, loops=5)
    laser_ = sensor.LaserSensor(env_.map_img_arr, (0, 0), (WIDTH, HEIGHT), RMIN, RMAX, scan_resolution=SCAN_RESOLUTION, heading_resolution=HEADING_RESOLUTION)
    
    ghost_lasers = []
    for i in range(NUM_GHOSTS):
        ghost_lasers.append(sensor.LaserSensor(env_.map_img_arr, (0, 0), (WIDTH, HEIGHT), RMIN, RMAX, scan_resolution=SCAN_RESOLUTION, heading_resolution=HEADING_RESOLUTION))
    
    map_ = buildmap.Map()
    
    start, goal = env_.start, env_.goal
    robot_pos = start
    running = True

    probs = np.zeros((WIDTH, HEIGHT))
    changes = None
    
    # --- Create Entities ---
    pacman = Entity(robot_pos, goal, env_, 15, pacman=True)
    
    ghosts: list[Entity] = []
    for i in range(NUM_GHOSTS):
        ghosts.append(Entity(env_.ghost_pos[i], tuple(pacman.pos), env_, 15, pacman=False))
        
    # --- Main Loop ---
    # Use joblib to parallelize computationally intensive tasks (path estimation and sensor scanning)
    with Parallel(n_jobs=N_JOBS, prefer="threads") as parallel:
        # Initial path estimation for all entities
        parallel(delayed(e.est)() for e in [pacman] + ghosts)

        count = 0
        while running:
            sensor_on = True
            
            # 1. Event Handling
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    print("Quitting...")
                    running = False
                if pg.mouse.get_focused():
                    if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                        pos = pg.mouse.get_pos()
                        print(f'mouse: {pos}')
                        print(pacman.costs[pos[0], pos[1]])
    
            # 2. Win/Loss Conditions
            for ghost in ghosts:
                if within_thresh(ghost.pos, pacman.pos, CAUGHT_DISTANCE):
                    print('Pacman caught by ghost')
                    sensor_on = False
            
            if within_thresh(pacman.pos, goal, CAUGHT_DISTANCE):
                print('Pacman reached goal')
                sensor_on = False
        
            # 3. Periodic Re-planning (Ping)
            if count % PING_PERIOD == 0:
                # Re-instantiate ghosts to reset state/goal tracking
                ghosts: list[Entity] = []
                for i in range(NUM_GHOSTS):
                    ghosts.append(Entity(env_.ghost_pos[i], tuple(pacman.pos), env_, 15, pacman=False))
                
                # Re-run path estimation in parallel
                parallel(delayed(e.est)() for e in [pacman] + ghosts)
                
                # Update ghost positions known to Pacman
                for i, ghost in enumerate(ghosts):
                    pacman.ghost_pos[i] = tuple(ghost.pos)
                
            # 4. Sensor Updates & Movement
            if sensor_on:
                # Move entities along their estimated paths
                pacman.update_pos(pacman.est_path)
                for i, ghost in enumerate(ghosts):
                    ghost.update_pos(ghost.est_path)
                    env_.ghost_pos[i] = ghost.pos
    
                # Update laser positions
                laser_.pos = pacman.pos
                for i, ghost_laser in enumerate(ghost_lasers):
                    ghost_laser.pos = ghosts[i].pos
    
                if laser_.pos is None:
                    sensor_on = False
    
                # Parallelize laser scanning
                all_lasers = [laser_] + ghost_lasers
                all_data = parallel(delayed(l.scan)() for l in all_lasers)
                sensor_data = all_data[0]
                ghost_data = all_data[1:]
    
                # Process sensor data (update environment and probabilities)
                if sensor_data: env_.process_data(sensor_data)
                for i, data in enumerate(ghost_data):
                    if data: env_.process_data(data, player=i+1)
                
                map_.laserCB(sensor_data, RMIN, RMAX)
                for data in ghost_data:
                    map_.laserCB(data, RMIN, RMAX)
    
                probs, changes = map_.get_probs()
            
            # 5. Visualization
            env_.show(probs, changes, pacman.pos)
            for i, ghost in enumerate(ghosts):
                env_.show(None, None, ghost.pos, player=i+1)
    
            count += 1
            pg.display.update()
    
            # Show learned maps periodically
            if count % 3 == 0:
                env_.show_learned_maps()



    pg.quit()
    
if __name__ == "__main__":
    main()
