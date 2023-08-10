import Maps
from Components import *
from GeneticModel import Gene, GeneticAlgorithmModel
import math

# Game Elements
walls = [Line(*(Point(p[0], -p[1]) for p in w)) for w in map1_walls]
checkpoints = [Point(p[0], -p[1]) for p in map1_checkpoints]

# Initialize Cars
cars_generation_count = 15
cars = [Car(center=Point(40, -50), speed=0, size=[50, 30],  heading=0.0, rotation_speed=0) for i in range(cars_generation_count)]
for car in cars:
    car.init_ai()
    car.update_stage(checkpoints)
    car.update_sensors(walls) 
    n = len(car.ai_steer.to_list())
    car.reset_place(walls, checkpoints, np.random.normal(0, 5, n).tolist())

# Initialize Genetic Model
def crossover(parent1, parent2):
    # Gets two genes and returns two genes
    if len(parent1.values) != len(parent2.values):
        raise Exception('Gene sizes are not equal!!')
    child1_values, child2_values = [], []
    random_bools = np.random.choice([True, False], size=len(parent1.values))
    for i in range(len(parent1.values)):
        if random_bools[i]:
            child1_values.append(parent1.values[i])
            child2_values.append(parent2.values[i])
        else:
            child1_values.append(parent2.values[i])
            child2_values.append(parent1.values[i])
    average_objective = (parent1.objective_val + parent2.objective_val)/2
    return Gene(child1_values, average_objective), Gene(child2_values, average_objective)

def mutation(gene):
    # Gets a gene and returns a gene
    gene_size = len(gene.values)
    m_indices = np.random.choice(range(gene_size), 20, replace=False)
    gene_values = gene.values.copy()
    for i in m_indices:
        gene_values[i] = float(gene_values[i] + np.random.normal(0, 5, 1)[0])
    return Gene(gene_values, gene.objective_val/2)

def objective_func(stage, score):
    return math.exp(stage-score/300)

def initial_population():
    genes = [Gene(car.ai_steer.to_list(), objective_func(car.stage, car.score)) for car in cars]
    return genes


genetic_model = GeneticAlgorithmModel(metrics=['best_objective_val'])# ,'mutates','crossovers', ...
genetic_model.compile(crossover, mutation, initial_population, crossover_coeff=0.1, mutation_coeff=0.5)

# Initialize Pygame
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 1366, 768
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Car Game")
screen_pars = ScreenPars(screen, (SCREEN_WIDTH, SCREEN_HEIGHT), car.center.copy(), car.center.copy())

# Game loop
clock = pygame.time.Clock()
frame_num = 0
itteration = 1
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    max_stage = 0
    dead_cars = 0
    for car in cars:
        car.rotation_speed = 0
        car.do_action(Car.Actions.FRICTION)
        car.do_action_ai()
        if car.can_move(walls):
            car.move()
            if car.stage > max_stage:
                screen_pars.dest_camera_position = car.center.copy()
                max_stage = car.stage
        else:
            dead_cars += 1
        car.update_stage(checkpoints)
        car.update_sensors(walls) 
    
    if dead_cars == cars_generation_count:
        itteration += 1
        genetic_model.population = [Gene(car.ai_steer.to_list(), objective_func(car.stage, car.score)) for car in cars]
        genetic_model.step()
        for i in range(cars_generation_count):
            cars[i].reset_place(walls, checkpoints, genetic_model.population[i].values)

    screen_pars.move_camera()
    screen.fill((0, 0, 0))
    draw_all(cars, walls, checkpoints, screen_pars, itteration)
    pygame.display.flip()
    # Limit frame rate
    clock.tick(60)
    
