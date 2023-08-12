# AI-Car-Racer-Genetic
This is a car racing game written by python and pygame along with some controllers for each car which can be controlled by user or an AI!


# Code Overview

[![Video Thumbnail](https://github.com/bateni1380/AI-Car-Racer-Genetic/blob/main/Capture.PNG)](https://github.com/bateni1380/AI-Car-Racer-Genetic/blob/main/demo.mp4)

An screenshot of the game!
## Summery of main.py
1. Imports: The code imports necessary modules and classes from different files for the game simulation and genetic algorithms.

2. Game Elements: Walls and checkpoints for the game are defined based on predefined map data.

3. Initializing Cars: A set number of cars are initialized on the screen. Each car starts with certain properties like position, speed, size, heading, and rotation speed. The cars nueral networks with random weights and biases are also being initialized here.

4. Genetic Model Initialization: The genetic algorithm model is set up by defining two functions: one for crossover (mixing genetic information of two genes) and one for mutation (making small changes to a gene).

5. Initializing Pygame: The Pygame library is initialized, and the screen dimensions and title and camera details are set in an ScreenPars object.

6. Game Loop: The main game loop starts, where the game events are handled. Inside the loop:

   - The behaviors of the cars are updated, including their actions and responses.

   - For each car, its movement, stage update, and sensor data are managed.

   - If all cars are "dead" (unable to move), the genetic algorithm is used to evolve the population. The best-performing genes are selected for the next generation, and some crossover and mutation are applied to generate new genes.

    - The screen is refreshed, showing the game's current state.


## Classes and methods of Components.py
```python
class ScreenPars:
    def __init__(self, screen, screen_dim, camera_position, dest_camera_position):
    def move_camera(self):  # Moves the camera gradually from camera_position to dest_camera_position

class Point:
    def __init__(self, x, y, color): 
    def distance(self, other_point):  # Calculates Euclidean distance to another point
    def draw(self, screen_pars):  # Draws the point on the screen
    def draw_cross(self, screen_pars):  # Draws a cross centered at the point
    def to_np(self):  # Converts the point to a NumPy array
    def update_from_np(self, array):  # Updates point coordinates from a NumPy array
    def copy(self):  # Creates a copy of the point

class Line:
    def __init__(self, p1: Point, p2: Point, color, diameter):
    def draw(self, screen_pars):  # Draws the line on the screen
    def intersects(self, other_line):  # Checks if this line intersects with another line

class Car:
    def __init__(self, center: Point, speed, size, rotation_speed, heading, color):
    def reset_place(self, walls, checkpoints, strategy):  # Resets car's position and AI parameters based on provided parameters
    def update_stage(self, checkpoints):  # Updates car's progress based on passed checkpoints
    def update_sensors(self, walls: List[Line]):  # Updates car's sensor data based on surrounding walls
    def do_action(self, action, pedal_pressure):  # Updates car speed based on given action (rotate steer, press gas pedal, press broke pedal)
    def init_ai(self):  # Initializes the car's nueral network
    def update_ai(self, strategy):  # Updates the AI's nueral network weights
    def do_action_ai(self):  # Executes an AI-driven action (using the nueral net)
    def can_move(self, walls):  # Checks if the car can move without colliding with walls
    def move(self):  # Moves the car based on its current state and speed
    def draw(self, screen_pars):  # Draws the car on the screen
    def collide(self, wall: Line):  # Checks collision with a wall
```

## Classes and methods of GeneticModel.py
```python
class Gene:
    def __init__(self, values: list, objective_val: float):  # Initializes a gene with values and an objective value
    def copy(self):  # Creates a copy of the gene

class GeneticAlgorithmModel:
    def __init__(self, metrics=[]):  # Initializes a genetic algorithm model with optional metrics (what to print each itteration)
    def compile(self, crossover_fun, mutation_fun, initial_population_fun, crossover_coeff, mutation_coeff):  # Sets up the model some necessary functions 
    def __choose_weighted(self, k):  # Chooses genes from the population using weighted probability based on their objectives
    def __choose_best(self, k):  # Chooses the best-performing genes from the population
    def extend(self, genes):  # Extends the population with provided genes
    def remove(self, gene):  # Removes a gene from the population
    def __print_on_epoch(self, epoch, metrics):  # Prints information about the current epoch based on metrics
    def step(self):  # Advances the genetic algorithm by one iteration

```

## Classes and methods of CarNet.py
```python
class CarNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):  # Initializes a neural network model for a car
    def forward(self, x):  # Implements the forward pass of the neural network
    def to_list(self):  # Concatinates the weights of all layers to a single list (to send it to genetic model)
    def update_from_list(self, l):  # Updates the model's weights from a provided list
```


Video Link : https://github.com/bateni1380/AI-Car-Racer-Genetic/blob/main/demo.mp4
