import cv2
import numpy as np

class Environment:
    def __init__(self,obstacles):
        self.margin = 5
        #coordinates are in [x,y] format
        self.car_length = 85
        self.car_width = 40
        self.trailer_length = 55
        self.trailer_width = 40
        self.trailer_con_length = 100
        self.trailer_con_width = 7
        self.wheel_length = 15
        self.wheel_width = 7
        self.wheel_positions = np.array([[60,15],[60,-17.5],[0,15],[0,-17.5]])
        self.trailer_wheel_positions = np.array([[0, 17.5], [0, -17.5]])
        
        self.color = np.array([0,0,255])/255
        self.wheel_color = np.array([20,20,20])/255

        self.car_struct = np.array([[+self.car_length-10, +self.car_width/2],
                                    [+self.car_length-10, -self.car_width/2],
                                    [-10, -self.car_width/2],
                                    [-10, +self.car_width/2]],
                                    np.int32)

        self.trailer_struct = np.array([[+self.trailer_length/2+10, +self.trailer_width/2],
                                    [+self.trailer_length/2+10, -self.trailer_width/2],
                                    [-self.trailer_length/2+10, -self.trailer_width/2],
                                    [-self.trailer_length/2+10, +self.trailer_width/2]],
                                    np.int32)

        self.trailer_con_struct = np.array([[+self.trailer_con_length/2+50, +self.trailer_con_width/2],
                                    [+self.trailer_con_length/2+50, -self.trailer_con_width/2],
                                    [-self.trailer_con_length/2+50, -self.trailer_con_width/2],
                                    [-self.trailer_con_length/2+50, +self.trailer_con_width/2]],
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

    def render(self, x, y, psi, psi2, delta):
        # x,y in 100 coordinates
        x = int(10*x)
        y = int(10*y)

        d1_in_px = 100
        trailer_x = x-(d1_in_px*np.cos(-psi2)).astype(int)
        trailer_y = y+(d1_in_px*np.sin(-psi2)).astype(int)
        # x,y in 1000 coordinates
        # adding car body
        rotated_struct = self.rotate_car(self.car_struct, angle=psi)
        rotated_struct += np.array([x,y]) + np.array([10*self.margin,10*self.margin])
        rendered = cv2.fillPoly(self.background.copy(), [rotated_struct], self.color)

        #adding trailer

        rotated_struct = self.rotate_car(self.trailer_struct, angle=psi2)
        rotated_struct += np.array([trailer_x,trailer_y]) + np.array([10*self.margin,10*self.margin])
        rendered = cv2.fillPoly(rendered, [rotated_struct], self.color)

        rotated_struct = self.rotate_car(self.trailer_con_struct, angle=psi2)
        rotated_struct += np.array([trailer_x,trailer_y]) + np.array([10*self.margin,10*self.margin])
        rendered = cv2.fillPoly(rendered, [rotated_struct], np.array([90,90,90])/255)

        # adding wheel
        rotated_wheel_center = self.rotate_car(self.wheel_positions, angle=psi)
        for i,wheel in enumerate(rotated_wheel_center):
            
            if i <2:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=delta+psi)
            else:
                rotated_wheel = self.rotate_car(self.wheel_struct, angle=psi)
            rotated_wheel += np.array([x,y]) + wheel + np.array([10*self.margin,10*self.margin])
            rendered = cv2.fillPoly(rendered, [rotated_wheel], self.wheel_color)

        # adding wheel
        rotated_wheel_center = self.rotate_car(self.trailer_wheel_positions, angle=psi2)
        for i, wheel in enumerate(rotated_wheel_center):
            rotated_wheel = self.rotate_car(self.wheel_struct, angle=psi2)
            rotated_wheel += np.array([trailer_x, trailer_y]) + wheel + np.array([10 * self.margin, 10 * self.margin])
            rendered = cv2.fillPoly(rendered, [rotated_wheel], self.wheel_color)

        # gel
        gel = np.vstack([np.random.randint(-20,-10,16),np.hstack([np.random.randint(-20,-10,8),np.random.randint(10,20,8)])]).T
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
        self.walls = [[60,i] for i in range(-10, 100)] +\
                     [[i,55] for i in range(-10, 100)] +\
                     [[i, 90] for i in range(20, 40)] +\
                     [[i,80] for i in range(20, 40)] + \
                     [[20, i] for i in range(80, 90)] +\
                     [[40, i] for i in range(80, 91)]
        self.obs = np.array(self.walls)
        self.cars = {1 : [[21,60]],2 : [[5,60]],3 : [[40,60]]}
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