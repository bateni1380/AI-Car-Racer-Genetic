# AI-Car-Racer-Genetic
This is a car racing game written by python and pygame along with some controllers for each car which can be controlled by user or an AI!
![](https://github.com/bateni1380/AI-Car-Racer-Genetic/blob/main/Capture.PNG)

# Code Overview
```python
# Classes and methods of Components.py
class ScreenPars:
    def __init__(self, screen, screen_dim, camera_position, dest_camera_position)
    def move_camera(self) # It moves the camera as much as CAMERA_SPEED from camera_position to dest_camera_position

class Point:
    def __init__(self, x, y, color)
    def distance(self, other_point)
    def draw(self, screen_pars)
    def draw_cross(self, screen_pars)
    def to_np(self)
    def update_from_np(self, array)
    def copy(self)

class Line:
    def __init__(self, p1: Point, p2: Point, color, diameter)
    def draw(self, screen_pars)
    def intersects(self, other_line)

class Car:
    def __init__(self, center: Point, speed, size, rotation_speed, heading, color)
    def reset_place(self, walls, checkpoints, strategy)
    def update_stage(self, checkpoints)
    def update_sensors(self, walls: List[Line])
    def do_action(self, action, pedal_pressure)
    def init_ai(self)
    def update_ai(self, strategy)
    def do_action_ai(self)
    def can_move(self, walls)
    def move(self)
    def draw(self, walls: List[Line], screen_pars)
    def collide(self, wall: Line)



```

Video Link : https://github.com/bateni1380/AI-Car-Racer-Genetic/blob/main/demo.mp4
