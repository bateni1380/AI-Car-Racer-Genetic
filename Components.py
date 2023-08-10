from Maps import map1_walls, map1_checkpoints
import pygame
import sys
import math
from typing import Tuple, List
import numpy as np
from CarNet import CarNet
import torch

class ScreenPars:
    def __init__(self, screen, screen_dim, camera_position, dest_camera_position):
        self.screen = screen
        self.screen_dim = screen_dim
        self.camera_position = camera_position
        self.dest_camera_position = dest_camera_position
    
    CAMERA_SPEED = 0.08
    def move_camera(self):
        cp1 = self.camera_position.to_np()
        cp2 = self.dest_camera_position.to_np()
        self.camera_position.update_from_np(cp1 + ScreenPars.CAMERA_SPEED*(cp2-cp1))


class Point:
    POINT_RADIUS = 0

    def __init__(self, x: int, y: int, color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.color = color

    def distance(self, other_point):
        return np.linalg.norm(self.to_np()-other_point.to_np())

    def draw(self, screen_pars: ScreenPars):
        p_pos = Point(self.x - screen_pars.camera_position.x + screen_pars.screen_dim[0]/2,
                      self.y - screen_pars.camera_position.y + screen_pars.screen_dim[1]/2)
        pygame.draw.circle(screen_pars.screen, self.color,
                           (p_pos.x, p_pos.y), Point.POINT_RADIUS)

    def draw_cross(self, screen_pars: ScreenPars):
        p_pos = Point(self.x - screen_pars.camera_position.x + screen_pars.screen_dim[0]/2,
                      self.y - screen_pars.camera_position.y + screen_pars.screen_dim[1]/2)
        pygame.draw.line(screen_pars.screen, self.color, (p_pos.x -
                         10, p_pos.y - 10), (p_pos.x + 10, p_pos.y + 10), 2)
        pygame.draw.line(screen_pars.screen, self.color, (p_pos.x +
                         10, p_pos.y - 10), (p_pos.x - 10, p_pos.y + 10), 2)

    def to_np(self):
        return np.array([self.x, self.y])

    def update_from_np(self, array):
        self.x, self.y = array[0], array[1]
    
    def copy(self):
        return Point(float(self.x), float(self.y))


class Line:
    def __init__(self, p1: Point, p2: Point, color=(0, 0, 255), diameter=5):
        self.p1 = p1
        self.p2 = p2
        self.color = color
        self.diameter = diameter

    def draw(self, screen_pars: ScreenPars):
        p1_pos = Point(self.p1.x - screen_pars.camera_position.x + screen_pars.screen_dim[0]/2,
                       self.p1.y - screen_pars.camera_position.y + screen_pars.screen_dim[1]/2)
        p2_pos = Point(self.p2.x - screen_pars.camera_position.x + screen_pars.screen_dim[0]/2,
                       self.p2.y - screen_pars.camera_position.y + screen_pars.screen_dim[1]/2)
        pygame.draw.line(screen_pars.screen, self.color, (p1_pos.x,
                         p1_pos.y), (p2_pos.x, p2_pos.y), self.diameter)

    def intersects(self, other_line):
        x1, y1, x2, y2 = self.p1.x, self.p1.y, self.p2.x, self.p2.y
        x3, y3, x4, y4 = other_line.p1.x, other_line.p1.y, other_line.p2.x, other_line.p2.y

        denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denominator == 0:
            return None  # Lines are parallel or coincident

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            intersection_x = x1 + ua * (x2 - x1)
            intersection_y = y1 + ua * (y2 - y1)
            return Point(intersection_x, intersection_y)
        else:
            return None  # Lines do not intersect


class Car:
    class Actions:
        FRICTION = 0
        PRESS_GAS_PEDAL = 1
        PRESS_BRAKE_PEDAL = 2
        TURN_STEER = 3

    FRICTION_ACCELERATION = 0.02
    GAS_ACCELERATION = 0.08
    BRAKE_ACCELERATION = 0.1
    MAX_SPEED = 15
    MIN_STEER_SPEED = 3
    MAX_ROTATION_SPEED = 3.5

    SENSORS_DIRECTIONS = [-50, -25, 0, 25, 50]
    SENSORS_COUNT = len(SENSORS_DIRECTIONS)
    SENSORS_VIEW_DEPTH = 200

    def __init__(self, center: Point, speed: float, size: List, rotation_speed: float, heading: float, color=(255, 0, 0)):
        self.__initial_values = [center.copy(), float(speed), float(rotation_speed), float(heading), color]
        self.center = center
        self.speed = speed
        self.size = size
        self.rotation_speed = rotation_speed
        self.heading = heading
        self.color = color
        self.stage = 0
        self.score = 1
        self.frames_in_checkpoint = 0
        self.frames_alive = 0
        self.ai_steer = None
        self.sensors_values = [None for i in range(Car.SENSORS_COUNT)]
        self.sensors_hit_points = [None for i in range(Car.SENSORS_COUNT)]
        self.alive = True

    def reset_place(self, walls, checkpoints, strategy):
        self.center = self.__initial_values[0].copy()
        self.speed = self.__initial_values[1]
        self.rotation_speed = self.__initial_values[2]
        self.heading = self.__initial_values[3]
        self.color = self.__initial_values[4]
        self.alive = True
        self.update_sensors(walls)
        self.stage = 0
        self.score = 0
        self.update_stage(checkpoints)
        self.update_ai(strategy)

    def update_stage(self, checkpoints):
        stage = 0
        min_distance = self.center.distance(checkpoints[stage])
        for i in range(len(checkpoints)):
            distance = self.center.distance(checkpoints[i])
            if distance < min_distance:
                stage, min_distance = i, distance
        checkpoints[stage].color = (0, 255, 0)
        if stage > self.stage:
            self.frames_in_checkpoint = 0
            self.stage = stage
            self.score = self.frames_alive+1  
        else:
            self.frames_in_checkpoint += 1
            if self.frames_in_checkpoint > 100:
                self.alive = False

    def update_sensors(self, walls: List[Line]):
        for i in range(Car.SENSORS_COUNT):
            sight_point = Point(0, 0)
            sight_point.x = self.center.x + Car.SENSORS_VIEW_DEPTH * math.cos(math.radians(
                self.heading+Car.SENSORS_DIRECTIONS[i]))
            sight_point.y = self.center.y + -Car.SENSORS_VIEW_DEPTH * math.sin(math.radians(
                self.heading+Car.SENSORS_DIRECTIONS[i]))
            sight_line = Line(self.center, sight_point, (255, 255, 255), 2)

            min_hit_dist = None
            min_hit_point = None
            for wall in walls:
                hit_point = wall.intersects(sight_line)
                if hit_point is not None:
                    hit_dist = self.center.distance(hit_point)
                    if min_hit_dist is None or hit_dist < min_hit_dist:
                        min_hit_dist = hit_dist
                        min_hit_point = hit_point

            if min_hit_point is not None:
                self.sensors_values[i] = min_hit_dist
                self.sensors_hit_points[i] = min_hit_point
            else:
                self.sensors_values[i] = Car.SENSORS_VIEW_DEPTH
                self.sensors_hit_points[i] = sight_point
                

    def do_action(self, action, pedal_pressure=0):
        if action == Car.Actions.FRICTION:
            self.speed -= Car.FRICTION_ACCELERATION * self.speed
        if action == Car.Actions.PRESS_GAS_PEDAL:
            self.speed += pedal_pressure * Car.GAS_ACCELERATION * \
                (Car.MAX_SPEED - self.speed)
        if action == Car.Actions.PRESS_BRAKE_PEDAL:
            self.speed -= pedal_pressure * Car.BRAKE_ACCELERATION * self.speed
        if action == Car.Actions.TURN_STEER:
            self.rotation_speed = pedal_pressure * Car.MAX_ROTATION_SPEED

    def init_ai(self):
        model = CarNet(Car.SENSORS_COUNT, 5, 3)
        self.ai_steer = model

    def update_ai(self, strategy):
        self.ai_steer.update_from_list(strategy)

    def do_action_ai(self):
        sensors_values_pre = [float(i)/float(Car.MAX_SPEED) for i in self.sensors_values]
        pedal_values = self.ai_steer(torch.tensor(sensors_values_pre)).tolist()
        self.do_action(Car.Actions.PRESS_GAS_PEDAL, pedal_pressure=pedal_values[0])
        self.do_action(Car.Actions.PRESS_BRAKE_PEDAL, pedal_pressure=pedal_values[0]/1.5)  
        self.do_action(Car.Actions.TURN_STEER, pedal_pressure=(pedal_values[1]-0.5)*2)

    def can_move(self, walls):
        if not self.alive:
            return False
        new_car = Car(self.center.copy(), float(self.speed), self.size.copy(), float(
            self.rotation_speed), float(self.heading))
        new_car.move()
        for wall in walls:
            if new_car.collide(wall):
                del new_car
                self.alive = False
                self.color = (100,100,100)
                if wall.p1.x == 0 and wall.p1.y == 100 and wall.p2.x == 0 and wall.p2.y == 0:
                    self.color = (0,255,0)
                return False
        del new_car
        return True

    def move(self):
        self.frames_alive += 1
        if self.speed > Car.MIN_STEER_SPEED:
            self.heading += self.rotation_speed
        self.center.x += self.speed * math.cos(math.radians(self.heading))
        self.center.y += -self.speed * math.sin(math.radians(self.heading))

    def draw(self, walls: List[Line], screen_pars: ScreenPars):
        c_pos = Point(self.center.x - screen_pars.camera_position.x +
                      screen_pars.screen_dim[0]/2, self.center.y - screen_pars.camera_position.y + screen_pars.screen_dim[1]/2)
        rotated_car = pygame.Surface(
            (self.size[0], self.size[1]), pygame.SRCALPHA)
        pygame.draw.rect(rotated_car, self.color,
                         (0, 0, self.size[0], self.size[1]))
        rotated_car = pygame.transform.rotate(rotated_car, self.heading)
        screen_pars.screen.blit(rotated_car, (c_pos.x - rotated_car.get_width() /
                                              2, c_pos.y - rotated_car.get_height() / 2))

        font = pygame.font.Font(None, 28)
        text_surface = font.render(str(int(self.stage/0.6))+'%', True, (255, 255, 255))
        text_pos = (c_pos.x - text_surface.get_width() / 2,
                    c_pos.y - text_surface.get_height() / 2)
        screen_pars.screen.blit(text_surface, text_pos)

        for hit_points in self.sensors_hit_points:
            hit_points.color = (100, 100, 100)
            hit_points.draw_cross(screen_pars)


    def collide(self, line: Line):
        rotated_car = pygame.Surface(
            (self.size[0], self.size[1]), pygame.SRCALPHA)
        pygame.draw.rect(rotated_car, self.color,
                         (0, 0, self.size[0], self.size[1]))
        rotated_car = pygame.transform.rotate(rotated_car, self.heading)
        car_center = pygame.math.Vector2(self.center.x, self.center.y)
        car_corners = [
            pygame.math.Vector2(-self.size[0] / 2, -self.size[1] /
                                2).rotate(-self.heading) + car_center,
            pygame.math.Vector2(
                self.size[0] / 2, -self.size[1] / 2).rotate(-self.heading) + car_center,
            pygame.math.Vector2(
                self.size[0] / 2, self.size[1] / 2).rotate(-self.heading) + car_center,
            pygame.math.Vector2(-self.size[0] / 2, self.size[1] /
                                2).rotate(-self.heading) + car_center
        ]
        car_corners = [Point(p.x, p.y) for p in car_corners]

        for i in range(len(car_corners)):
            if Line(car_corners[i], car_corners[(i + 1) % len(car_corners)]).intersects(line) is not None:
                return True
        return False


def draw_all(cars, walls, checkpoints, screen_pars, itteration):
    font = pygame.font.Font(None, 28)
    text_surface = font.render(str(itteration)+'th attempt!', True, (255, 255, 255))
    text_pos = (380, 20)
    screen_pars.screen.blit(text_surface, text_pos)
    for car in cars:
        car.draw(walls, screen_pars)
    for wall in walls:
        wall.draw(screen_pars)
    for checkpoint in checkpoints:
        checkpoint.draw(screen_pars)


