import pygame
from car import Car
from road import Road
import math
import torch
#Initialize Pygame
pygame.init()

#Display Settings
FPS = 60
WIDTH, HEIGHT = 1000, 1000
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Driving Simulator')

#Colors
BACKGROUND_COLOR = (100, 225, 100)

running = True
clock = pygame.time.Clock()

#Car and Road objects
road = Road(window)
car = Car(window,road)

# Network locations
acceleration_path = 'models/acceleration_network.pth'
turn_path = 'models/turn_network.pth'
value_path = 'models/value_network.pth'

<<<<<<< HEAD
font = pygame.font.SysFont('Arial', 32)

def RenderSpeedometer():
=======
def RenderSpeedometer():
    font = pygame.font.SysFont('Arial', 32)
>>>>>>> 8b84eab937deecec6fc53e0e922f3612d8bfd2df
    speed_text = font.render('Speed: '+str(round(-car.GetSpeed(),2)), True, (255, 255, 255))
    angle_text = font.render('Angle: '+str(round(car.GetAngle(),2)), True, (255, 255, 255))
    window.blit(speed_text,(5,0))
    window.blit(angle_text,(5,40))

<<<<<<< HEAD
def RenderSimulationCount():
    count_text = font.render('Simulation: '+str(car.simulation_count())+'/1000', True, (255, 255, 255))
    window.blit(count_text,(5,80))

def Render():
    # Render the road and car on to the window
    road.Render()
    car.Render()
    # Render sensor
    road.RenderSensor(car.GetSensorPts())
    # Print car speed and angle
    RenderSpeedometer()
    # Print simulation number
    RenderSimulationCount()

def Save():
    # Save models
    torch.save(acceleration_network.state_dict(), acceleration_path)
    torch.save(turn_network.state_dict(), turn_path)
    torch.save(value_network.state_dict(), value_path)

    # Save trajectory data
    car.SaveData()
=======
>>>>>>> 8b84eab937deecec6fc53e0e922f3612d8bfd2df

#Run once at the beginning of simulation
def Start(): car.Reset()

#Run every frame update
def Update():
    #Set simulation FPS
    clock.tick(FPS)

    # Reset screen background
    window.fill(BACKGROUND_COLOR)

    #Update for car movement based on user input
    car.Run()

    #Generate road dynamically based on car position
    car_pos = car.GetPosition()
    road.Generate(car_pos)

    #Render the road and car on to the window
<<<<<<< HEAD
    Render()
    
=======
    road.Render()
    car.Render()

    # Print car speed and angle
    RenderSpeedometer()

    #road.RenderSensor(car.GetSensorPts())
    #print(car.Reward())
>>>>>>> 8b84eab937deecec6fc53e0e922f3612d8bfd2df
    # Update display
    pygame.display.update()


if __name__ == "__main__":

    Start() # Run initialization

    # Simulation loop
    while running:
        # Stop main loop when user closes window
        for event in pygame.event.get():
            acceleration_network, turn_network, value_network = car.GetNetworks()
            if event.type == pygame.QUIT: running = False
        # If 1000 simulations ran, exit
        if car.simulation_count() >= 1000: running = False

        # Otherwise, update the window accordingly
        Update()

<<<<<<< HEAD
    # On quit, save simulation data
    Save()
=======
    # Save models
    torch.save(acceleration_network.state_dict(), acceleration_path)
    torch.save(turn_network.state_dict(), turn_path)
    torch.save(value_network.state_dict(), value_path)

    # Save trajectory data
    car.SaveData()
>>>>>>> 8b84eab937deecec6fc53e0e922f3612d8bfd2df

    # Quit pygame when simulation is ended
    pygame.quit()