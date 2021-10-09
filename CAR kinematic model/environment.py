import cv2
import numpy as np

class Environment:
    def __init__(self,obstacles):
        self.margin = 5
        #coordinates are in [x,y] format
        self.car_length = 80
        self.car_width = 40
        self.wheel_length = 15
        self.wheel_width = 7
        self.wheel_positions = np.array([[25,15],[25,-15],[-25,15],[-25,-15]])
        
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
                
    def place_obstacles(self, obs):
        obstacles = np.concatenate([np.array([[0,i] for i in range(100+2*self.margin)]),
                                    np.array([[100+2*self.margin-1,i] for i in range(100+2*self.margin)]),
                                    np.array([[i,0] for i in range(100+2*self.margin)]),
                                    np.array([[i,100+2*self.margin-1] for i in range(100+2*self.margin)]),
                                    obs + np.array([self.margin,self.margin])])*10
        for ob in obstacles:
            self.background[ob[1]:ob[1]+10,ob[0]:ob[0]+10]=0
    
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
        gel = np.vstack([np.random.randint(-50,-30,16),np.hstack([np.random.randint(-20,-10,8),np.random.randint(10,20,8)])]).T
        gel = self.rotate_car(gel, angle=psi)
        gel += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        gel = np.vstack([gel,gel+[1,0],gel+[0,1],gel+[1,1]])
        rendered[gel[:,1],gel[:,0]] = np.array([60,60,135])/255

        new_center = np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        self.background = cv2.circle(self.background, (new_center[0],new_center[1]), 2, [255/255, 150/255, 100/255], -1)

        rendered = cv2.resize(np.flip(rendered, axis=0), (700,700))
        return rendered


class Parking1:
    def __init__(self, car_pos):
        self.car_obstacle_hori = self.make_car_hori()
        self.car_obstacle_vert = self.make_car_vert()
        self.car_obstacle_left = self.make_car_left()
        self.car_obstacle_right = self.make_car_right()
        self.walls = [[20,i] for i in range(25, 105)]+\
                     [[20,i] for i in range(-5, 10)] +\
                     [[60,i] for i in range(25,80) ]+\
                     [[100,i] for i in range(25,80) ]
        # self.walls = [0,100]
        self.obs = np.array(self.walls)
        self.cars = {1 : [[95,45]],2 : [[27,35]],3 : [[27,47]],4 : [[27,59]],5 : [[27,71]]
                    ,6 : [[53,35]],7 : [[53,47]],8 : [[53,59]],9 : [[53,71]]
                     ,10 : [[30,100]],11 : [[45,100]],12 : [[60,100]],13 : [[75,100]],14 : [[90,100]],
                     15 : [[68,33]],16 : [[68,41]],17 : [[68,49]],18 : [[68,57]],19 : [[68,65]],20 : [[68,73]],
                     21 : [[95,33]],22 : [[95,59]],23 : [[95,73]]}
        self.car_type = ["self(blank)", "left", "left", "left", "left",
                         "right", "right", "right", "right",
                         "hori", "hori", "hori", "hori","hori", "hori", "hori", "hori","hori", "hori", "hori",
                         "vert", "vert", "vert"]
        self.end = self.cars[car_pos][0]
        self.cars.pop(car_pos)

    def generate_obstacles(self):
        for i in self.cars.keys():
            for j in range(len(self.cars[i])):
                if self.car_type[i-1] == "hori":
                    obstacle = self.car_obstacle_hori + self.cars[i]
                elif self.car_type[i-1] == "vert":
                    obstacle = self.car_obstacle_vert + self.cars[i]
                elif self.car_type[i-1] == "left":
                    obstacle = self.car_obstacle_left + self.cars[i]
                elif self.car_type[i-1] == "right":
                    obstacle = self.car_obstacle_right + self.cars[i]
                self.obs = np.append(self.obs, obstacle)
        return self.end, np.array(self.obs).reshape(-1,2)

    def make_car_vert(self):
        car_obstacle_x, car_obstacle_y = np.meshgrid(np.arange(-2,2), np.arange(-4,4))
        car_obstacle = np.dstack([car_obstacle_x, car_obstacle_y]).reshape(-1,2)
        # car_obstacle = np.array([[0,0],[0,-1],[0,1],[-1,-1],[-1,0],[-1,1],[1,-1],[1,0],[1,1]])
        return car_obstacle

    def make_car_hori(self):
        car_obstacle_x, car_obstacle_y = np.meshgrid(np.arange(-4,4), np.arange(-2,2))
        car_obstacle = np.dstack([car_obstacle_x, car_obstacle_y]).reshape(-1,2)
        # car_obstacle = np.array([[0,0],[0,-1],[0,1],[-1,-1],[-1,0],[-1,1],[1,-1],[1,0],[1,1]])
        return car_obstacle

    def make_car_right(self):
        # car_obstacle = np.array([[1,4],[0,3],[1,3],[2,3],[-1,2],[0,2],[1,2],[2,2],[3,2],[-2,1],[-1,1],[0,1],[1,1],[2,1],[3,1],[4,1],[-2,1],[-1,0],[0,0],[1,0],[2,0],[3,0],[-3,0],[-2,-1],[-1,-1],[0,-1],[1,-1],[2,-1],[-4,-1],[-3,-1],[-2,1],[-1,-2],[0,-2],[1,-2],[-3,-2],[-4,-2],[-5,-2],[-2,-3],[-1,-3],[0,-3],[-3,-3],[-4,-3],[-3,-4],[-2,-4],[-1,-4],[-2,-5]])
        car_obstacle = np.array(
            [[1, 4], [0, 3], [1, 3], [2, 3], [-1, 2], [0, 2], [1, 2], [2, 2], [3, 2], [-2, 1], [-1, 1], [0, 1], [1, 1],
             [2, 1], [3, 1], [4, 1], [-2, 1], [-1, 0], [0, 0], [1, 0], [2, 0], [3, 0], [-3, 0], [-2, 0], [-2, -1], [-1, -1],
             [0, -1], [1, -1], [2, -1], [-4, -1], [-3, -1], [-2, -2], [-1, -2], [0, -2], [1, -2], [-3, -2], [-4, -2],
             [-5, -2], [-2, -3], [-1, -3], [0, -3], [-3, -3], [-4, -3], [-3, -4], [-2, -4], [-1, -4], [-2, -5]])
        return car_obstacle

    def make_car_left(self):
        car_obstacle = np.array(
            [[-1, 4], [0, 3], [-1, 3], [-2, 3], [1, 2], [0, 2], [-1, 2], [-2, 2], [-3, 2], [2, 1], [1, 1], [0, 1], [-1, 1],
             [-2, 1], [-3, 1], [-4, 1], [2, 1], [1, 0], [0, 0], [-1, 0], [-2, 0], [-3, 0], [3, 0], [2, 0], [2, -1],
             [1, -1],
             [0, -1], [-1, -1], [-2, -1], [4, -1], [3, -1], [2, -2], [1, -2], [0, -2], [-1, -2], [3, -2], [4, -2],
             [5, -2], [2, -3], [1, -3], [0, -3], [3, -3], [4, -3], [3, -4], [2, -4], [1, -4], [2, -5]])
        return car_obstacle