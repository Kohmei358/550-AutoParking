import time

import cv2
import numpy as np
from time import sleep
import argparse

from environment import Environment, Parking1
from pathplanning import PathPlanning, ParkPathPlanning, interpolate_path
from control import Car_Dynamics, MPC_Controller, Linear_MPC_Controller
from utils import angle_of_line, make_square, DataLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_start', type=int, default=3, help='X of start')
    parser.add_argument('--y_start', type=int, default=100, help='Y of start')
    parser.add_argument('--psi_start', type=int, default=0, help='psi of start')
    parser.add_argument('--x_end', type=int, default=10, help='X of end')
    parser.add_argument('--y_end', type=int, default=10, help='Y of end')
    parser.add_argument('--parking', type=int, default=1, help='park position in parking1 out of 24')

    args = parser.parse_args()
    logger = DataLogger()

    ########################## default variables ################################################
    start = np.array([args.x_start, args.y_start])
    end   = np.array([args.x_end, args.y_end])
    #############################################################################################

    # environment margin  : 5
    # pathplanning margin : 5

    ########################## defining obstacles ###############################################
    parking1 = Parking1(args.parking)
    end, obs = parking1.generate_obstacles()

    # add squares
    # square1 = make_square(10,65,20)
    # square2 = make_square(15,30,20)
    # square3 = make_square(50,50,10)
    # obs = np.vstack([obs,square1,square2,square3])

    # Rahneshan logo
    # start = np.array([50,5])
    # end = np.array([35,67])
    # rah = np.flip(cv2.imread('READ_ME/rahneshan_obstacle.png',0), axis=0)
    # obs = np.vstack([np.where(rah<100)[1],np.where(rah<100)[0]]).T

    # new_obs = np.array([[78,78],[79,79],[78,79]])
    # obs = np.vstack([obs,new_obs])
    #############################################################################################

    ########################### initialization ##################################################
    env = Environment(obs)
    my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=2.8, dt=0.2)
    MPC_HORIZON = 5
    controller = MPC_Controller()
    # controller = Linear_MPC_Controller()

    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    cv2.imshow('environment', res)
    key = cv2.waitKey(1)
    #############################################################################################

    ############################# path planning #################################################
    park_path_planner = ParkPathPlanning(obs)
    path_planner = PathPlanning(obs)

    # print('planning park scenario ...')
    # # new_end, park_path, ensure_path1, ensure_path2 = park_path_planner.generate_park_scenario(int(start[0]),int(start[1]),int(end[0]),int(end[1]))
    #
    # print('routing to destination ...')
    # path = path_planner.plan_path(int(start[0]),int(start[1]),int(end[0]),int(end[1]))
    # # path = np.vstack([path, ensure_path1])
    #
    # print('interpolating ...')
    # interpolated_path = interpolate_path(path, sample_rate=2)
    # # interpolated_park_path = interpolate_path(park_path, sample_rate=2)
    #
    # # interpolated_path = path
    # # interpolated_park_path = park_path
    # # interpolated_park_path = np.vstack([ensure_path1[::-1], interpolated_park_path, ensure_path2[::-1]])
    #
    # env.draw_path(interpolated_path)
    # # env.draw_path(interpolated_park_path)
    #
    # final_path = np.vstack([interpolated_path])

    #############################################################################################

    ################################## control ##################################################
    print('driving to destination ...')

    current_x = start[0]
    current_y = start[1]

    next_x = start[0]+5
    next_y = start[1]-8

    next_fire_tick_time = time.time() + 5
    while True:
        counter = 0
        done = 0
        while done == 0:
            next_x, next_y = env.get_next_fire()
            path = path_planner.plan_path(int(current_x), int(current_y), int(next_x), int(next_y))
            print(str(current_x)+", "+str(current_y)+","+str(next_x)+", "+str(next_y))
            cv2.imshow('environment', res)
            env.update_obstacle_colors()
            if time.time() > next_fire_tick_time:
                counter = counter + 1
                if counter % 6 == 0:
                    env.random_fire()
                env.fire_tick()
                next_fire_tick_time = time.time() + 5
            if len(path) > 2:
                print(len(path))
                done = 1
        try:
            interpolated_path = interpolate_path(path, sample_rate=2)
            # interpolated_path = path
            env.draw_path(interpolated_path)
            for i, point in enumerate(interpolated_path):

                acc, delta = controller.optimize(my_car, interpolated_path[i:i + MPC_HORIZON])
                my_car.update_state(my_car.move(acc, delta))
                res = env.render(my_car.x, my_car.y, my_car.psi, delta)
                logger.log(point, my_car, acc, delta)
                cv2.imshow('environment', res)
                key = cv2.waitKey(1)
                if key == ord('s'):
                    cv2.imwrite('res.png', res * 255)

                if time.time() > next_fire_tick_time:
                    counter = counter + 1
                    if counter % 3 == 0:
                        env.random_fire()
                    env.fire_tick()
                    next_fire_tick_time = time.time() + 5

                env.update_obstacle_colors()

            current_x = next_x
            current_y = next_y

            env.kill_fire(current_x+5, current_y+5)
            next_x, next_y = env.get_next_fire()

        except Exception as e:
            print(e)
            print("Trying new fire")
            cv2.imshow('environment', res)
            env.update_obstacle_colors()
            if time.time() > next_fire_tick_time:
                counter = counter + 1
                if counter % 3 == 0:
                    env.random_fire()
                env.fire_tick()
                next_fire_tick_time = time.time() + 5

        next_x, next_y = env.get_next_fire()

    # zeroing car steer
    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    logger.save_data()
    cv2.imshow('environment', res)
    key = cv2.waitKey()
    #############################################################################################

    cv2.destroyAllWindows()

