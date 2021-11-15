import random
import cv2
import numpy as np


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class Environment:
    def __init__(self,obstacles):
        self.obst = obstacles
        self.margin = 5
        #coordinates are in [x,y] format
        self.car_length = 10
        self.car_width = 6
        self.wheel_length = 3
        self.wheel_width = 1
        self.wheel_positions = np.array([[4,1],[4,-1],[-4,1],[-4,-1]])
        self.burning_state = np.zeros((112,112)) #Nothing
        self.set_array()
        self.random_fire()
        self.random_fire()
        # self.burning_state[5:10, 5:10] = 1 #Obst
        # self.burning_state[5:8, 5:8] = 2 #Burning
        # self.burning_state[12, :] = 2 #Ext
        
        self.color = np.array([0,0,255])/255
        self.wheel_color = np.array([20,20,20])/255

        self.car_struct = np.array([[+self.car_length/2, +self.car_width/2],
                                    [+self.car_length/2, -self.car_width/2],  
                                    [-self.car_length/2, -self.car_width/2],
                                    [-self.car_length/2, +self.car_width/2]], 
                                    np.int32)
        
        self.wheel_struct = np.array([[+self.wheel_length/2, +self.wheel_width/2],
                                      [+self.wheel_length/2, -self.wheel_width/2],  
                                      [-self.wheel_length/2, -self.wheel_width/2],
                                      [-self.wheel_length/2, +self.wheel_width/2]], 
                                      np.int32)

        #height and width
        self.background = np.ones((1000+20*self.margin,1000+20*self.margin,3))
        self.background[10:1000+20*self.margin:10,:] = np.array([200,200,200])/255
        self.background[:,10:1000+20*self.margin:10] = np.array([200,200,200])/255
        self.place_obstacles(obstacles)
    def random_fire(self):
        select = random.randint(0,len(self.obst)-1)
        self.burning_state[self.obst[select][0], self.obst[select][1]] = 2

    def fire_tick(self):
        print("fire tick")
        self.place_obstacles(self.obst)
        cp_burning_state = self.burning_state
        for x in range(len(cp_burning_state)):
            for y in range(len(cp_burning_state[x])):
                if self.burning_state[x][y] == 2:
                    for x2 in range(clamp(x - 2, 0, 109),clamp(x + 2, 0, 109)):
                        for y2 in range(clamp(y - 2, 0, 109),clamp(y + 2, 0, 109)):
                            if self.burning_state[x2][y2] == 1:
                                cp_burning_state[x2][y2] = 2


        self.burning_state = cp_burning_state

    def update_obstacle_colors(self):
        self.place_obstacles(self.obst)


    def place_obstacles(self, obs):
        obstacles = np.concatenate([np.array([[0,i] for i in range(100+2*self.margin)]),
                                    np.array([[100+2*self.margin-1,i] for i in range(100+2*self.margin)]),
                                    np.array([[i,0] for i in range(100+2*self.margin)]),
                                    np.array([[i,100+2*self.margin-1] for i in range(100+2*self.margin)]),
                                    obs + np.array([self.margin,self.margin])])*10
        for index, ob in enumerate(obstacles):
            if self.burning_state[int(ob[0]/10),int(ob[1]/10)] == 2: #Burning
                self.background[ob[1]:ob[1]+10,ob[0]:ob[0]+10]= np.array([50, 90, 255]) / 255
            elif self.burning_state[int(ob[0]/10),int(ob[1]/10)] == 3: #Extinguished
                self.background[ob[1]:ob[1]+10,ob[0]:ob[0]+10]= np.array([255, 50, 50]) / 255
            else:
                self.background[ob[1]:ob[1] + 10, ob[0]:ob[0] + 10] = 0  # Black
    
    def draw_path(self, path):
        path = np.array(path)*10
        color = np.random.randint(0,150,3)/255
        path = path.astype(int)
        for p in path:
            self.background[p[1]+10*self.margin:p[1]+10*self.margin+3,p[0]+10*self.margin:p[0]+10*self.margin+3]=color

    def rotate_car(self, pts, angle=0):
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        return ((R @ pts.T).T).astype(int)

    def render(self, x, y, psi, delta):
        # x,y in 100 coordinates
        x = int(10*x)
        y = int(10*y)
        # x,y in 1000 coordinates
        # adding car body
        rotated_struct = self.rotate_car(self.car_struct, angle=psi)
        rotated_struct += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        rendered = cv2.fillPoly(self.background.copy(), [rotated_struct], self.color)

        # adding wheel
        rotated_wheel_center = self.rotate_car(self.wheel_positions, angle=psi)
        for i,wheel in enumerate(rotated_wheel_center):
            
            if i <2:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=delta+psi)
            else:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=psi)
            rotated_wheel += np.array([x,y]) + wheel + np.array([10*self.margin,10*self.margin])
            rendered = cv2.fillPoly(rendered, [rotated_wheel], self.wheel_color)

        # gel
        # gel = np.vstack([np.random.randint(-50,-30,16),np.hstack([np.random.randint(-20,-10,8),np.random.randint(10,20,8)])]).T
        # gel = self.rotate_car(gel, angle=psi)
        # gel += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        # gel = np.vstack([gel,gel+[1,0],gel+[0,1],gel+[1,1]])
        # rendered[gel[:,1],gel[:,0]] = np.array([60,60,135])/255
        #
        # new_center = np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        # self.background = cv2.circle(self.background, (new_center[0],new_center[1]), 2, [255/255, 150/255, 100/255], -1)

        rendered = cv2.resize(np.flip(rendered, axis=0), (700,700))
        return rendered

    def set_array(self):
        for obst in self.obst:
            self.burning_state[obst[0]+5,obst[1]+5] = 1

    def get_next_fire(self):
        done = 0
        while done == 0:
            select = random.randint(0, len(self.obst)-1)
            x = self.obst[select][0]
            y = self.obst[select][1]
            if self.burning_state[x,y] == 2:
                for x2 in range(clamp(x - 1, 0, 109), clamp(x + 1, 0, 109)):
                    for y2 in range(clamp(y - 1, 0, 109), clamp(y + 1, 0, 109)):
                        if self.burning_state[x2][y2] == 0:
                            done == 1
                            return x2-5, y2-5
            if self.burning_state[x, y] == 2:
                for x2 in range(clamp(x - 2, 0, 109), clamp(x + 2, 0, 109)):
                    for y2 in range(clamp(y - 2, 0, 109), clamp(y + 2, 0, 109)):
                        if self.burning_state[x2][y2] == 0:
                            done == 1
                            return x2 - 5, y2 - 5

    def kill_fire(self, current_x, current_y):
        x = current_x
        y = current_y
        for x2 in range(clamp(x - 3, 0, 109), clamp(x + 3, 0, 109)):
            for y2 in range(clamp(y - 3, 0, 109), clamp(y + 3, 0, 109)):
                if self.burning_state[x2][y2] == 2:
                    self.burning_state[x2][y2] = 3


class Parking1:
    def __init__(self, car_pos):
        self.car_obstacle_hori = self.make_car_hori()
        self.walls = self.generate_feild(.20);
        self.obs = np.array(self.walls)
        self.cars = {1 : [[95,3]]}
        self.end = self.cars[car_pos][0]
        self.cars.pop(car_pos)

    def generate_obstacles(self):
        for i in self.cars.keys():
            for j in range(len(self.cars[i])):
                obstacle = self.car_obstacle_hori + self.cars[i]
                self.obs = np.append(self.obs, obstacle)
        return self.end, np.array(self.obs).reshape(-1,2)

    def make_car_hori(self):
        car_obstacle_x, car_obstacle_y = np.meshgrid(np.arange(-4,4), np.arange(-2,2))
        car_obstacle = np.dstack([car_obstacle_x, car_obstacle_y]).reshape(-1,2)
        # car_obstacle = np.array([[0,0],[0,-1],[0,1],[-1,-1],[-1,0],[-1,1],[1,-1],[1,0],[1,1]])
        return car_obstacle

    def generate_tetrominoes(self, list_of_points, curr_x, curr_y, x_max, y_max, depth=0):
        if len(list_of_points) == 4:  # Base Case we made 4 cells
            return 0

        if not (0 <= curr_y < y_max and 0 <= curr_x < x_max):  # Not in field - try next random direction
            return 1

        list_of_points.add((curr_x, curr_y))

        completion = 1

        while completion == 1:
            direction = random.randint(0, 3)
            if direction == 0:  # up
                completion = self.generate_tetrominoes(list_of_points, curr_x, curr_y + 1, x_max, y_max, depth + 1)
            elif direction == 1:  # right
                completion = self.generate_tetrominoes(list_of_points, curr_x + 1, curr_y, x_max, y_max, depth + 1)
            elif direction == 2:  # down
                completion = self.generate_tetrominoes(list_of_points, curr_x, curr_y - 1, x_max, y_max, depth + 1)
            else:  # left
                completion = self.generate_tetrominoes(list_of_points, curr_x - 1, curr_y, x_max, y_max, depth + 1)

    def generate_feild(self, p):
        # data = [[random.randint(a=0, b=1) for x in range(0, 3)],  # row 1
        grid_x_size = 110
        grid_y_size = 110

        field = np.zeros((grid_x_size, grid_y_size), dtype=int)

        required_obstacles = grid_x_size * grid_y_size * p
        current_obstacles = 0

        while (current_obstacles < required_obstacles):

            # randomly sample a point
            start_cell_x = random.randint(a=0, b=grid_x_size - 1)
            start_cell_y = random.randint(a=0, b=grid_y_size - 1)

            list_of_points = set()
            # recursively genearte 4 adjacent blocks that are within bounds
            self.generate_tetrominoes(list_of_points, start_cell_x, start_cell_y, grid_x_size, grid_y_size)

            # Add each point to the feild
            for point in list_of_points:
                field[point[0], point[1]] = 1

            # update the count of filled cells
            current_obstacles = np.count_nonzero(field)

        list_of_walls = [];

        for i in range(len(field)):
            for j in range(len(field[i])):
                if field[i][j] == 1 and (i > 20 or j < 90):
                    list_of_walls.append([i-5,j-5])

        return list_of_walls