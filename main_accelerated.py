import pygame
from car import Car
from road import Road
import math
import torch
import os
from datetime import datetime
from config import num_of_simulations
os.environ["SDL_VIDEODRIVER"] = "dummy"

#Initialize Pygame
pygame.init()

#Display Settings
FPS = 60
WIDTH, HEIGHT = 1000, 1000
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Driving Simulator')

# Hide display
os.environ["SDL_VIDEODRIVER"] = "dummy"

running = True

#Car and Road objects
road = Road(window)
car = Car(window,road)

# Network locations
acceleration_path = 'models/acceleration_network.pth'
turn_path = 'models/turn_network.pth'
value_path = 'models/value_network.pth'

def Save():
    # Get networks from car
    acceleration_network, turn_network, value_network = car.GetNetworks()

    # Save models
    torch.save(acceleration_network.state_dict(), acceleration_path)
    torch.save(turn_network.state_dict(), turn_path)
    torch.save(value_network.state_dict(), value_path)

    # Save trajectory data
    car.SaveData()

#Run once at the beginning of simulation
def Start(): car.Reset()

#Run every frame update
def Update():
    #Update for car movement based on user input
    car.Run()

    #Generate road dynamically based on car position
    car_pos = car.GetPosition()
    road.Generate(car_pos)


if __name__ == "__main__":

    print('Starting simulation.')
    start_time = datetime.now()
    Start() # Run initialization

    # Simulation loop
    while running:
        # Stop main loop when user closes window
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        # If 1000 simulations ran, exit
        if car.simulation_count > num_of_simulations: break
        # Otherwise, update the window accordingly
        Update()

    # On quit, save simulation data
    Save()

    # Print time taken to complete data collection
    time_elapsed = abs(datetime.now()-start_time).total_seconds()
    minutes = int(time_elapsed // 60)
    seconds = round(time_elapsed % 60)
    print('Time elapsed:', minutes, 'minutes', seconds, 'seconds')
    print('Simulation Terminated')

    # Quit pygame when simulation is ended
    pygame.quit()