import pygame
from car import Car
from road import Road

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
car = Car(window)
road = Road(window)

#Run once at the beginning of simulation
def Start():
    road.Generate((0,0)) # Generate inital road points
    road.Recenter() # Reposition road so car initialized on the road
    
#Run every frame update
def Update():
    #Set simulation FPS
    clock.tick(FPS)

    # Reset screen background
    window.fill(BACKGROUND_COLOR)

    #Update for car movement based on user input
    car.Move()
    car_pos = car.GetPosition()
    #Generate road dynamically based on car position
    road.Generate(car_pos)

    #Render the road and car on to the window
    road.Render()
    car.Render()

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