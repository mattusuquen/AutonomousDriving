import pygame
from car import Car
from road import Road
import math
#Initialize Pygame
pygame.init()

#Display Settings
FPS = 60
WIDTH, HEIGHT = 1000, 1000
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Car Simulator')

#Colors
BACKGROUND_COLOR = (100, 225, 100)

running = True
clock = pygame.time.Clock()

#Car and Road objects
road = Road(window)
car = Car(window,road)

#Run once at the beginning of simulation
def Start():
    road.Generate((0,0)) # Generate inital road points
    angle = road.Recenter() # Reposition road so car initialized on the road
    car.SetRotation(angle) # Adjust car orientation

    car_size, car_angle = car.GetSize(), math.radians(car.GetAngle())
    X = road.GetSensorData(car_size,car_angle)
    X[len(X)-1][len(X[0])-1] = car_angle
    f = open("initial_data.txt", "w")
    for x in X: 
        f.write('\t'.join(str(i) for i in x))
        f.write('\n')
    
#Run every frame update
def Update():
    #Set simulation FPS
    clock.tick(FPS)

    # Reset screen background
    window.fill(BACKGROUND_COLOR)

    #Update for car movement based on user input
    car.Move()
    car_pos, car_size, car_angle = car.GetPosition(), car.GetSize(), math.radians(car.GetAngle())
    #print(car_angle)
    #Generate road dynamically based on car position
    road.Generate(car_pos)
    X = road.GetSensorData(car_size,car_angle)
    #print(X)
    #Render the road and car on to the window
    road.Render()
    car.Render()
    print(car.Reward())
    # Update display
    pygame.display.update()


if __name__ == "__main__":

    Start() # Run initialization

    # Simulation loop
    while running:
        # Stop main loop when user closes window
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        # Otherwise, update the window accordingly
        Update()

    # Quit pygame when simulation is ended
    pygame.quit()