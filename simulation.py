import os
import pygame
from car import Car
from road import Road
import math
import torch
import multiprocessing
import os
from datetime import datetime
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Simulation Settings
FPS = 60
WIDTH, HEIGHT = 1000, 1000
BACKGROUND_COLOR = (100, 225, 100)

# Network locations
acceleration_path = 'models/acceleration_network.pth'
turn_path = 'models/turn_network.pth'
value_path = 'models/value_network.pth'

def run_simulation(simulation_number):
    
    pygame.init()

    # Setup display and clock for each simulation
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Driving Simulator {simulation_number}')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 32)

    # Initialize Road and Car objects
    road = Road(window)
    car = Car(window, road,simulation_number)

    def RenderSpeedometer():
        speed_text = font.render('Speed: '+str(round(-car.GetSpeed(), 2)), True, (255, 255, 255))
        angle_text = font.render('Angle: '+str(round(car.GetAngle(), 2)), True, (255, 255, 255))
        window.blit(speed_text, (5, 0))
        window.blit(angle_text, (5, 40))

    def RenderSimulationCount():
        count_text = font.render('Simulation: '+str(car.simulation_count())+'/1000', True, (255, 255, 255))
        window.blit(count_text, (5, 80))

    def Render():
        road.Render()
        car.Render()
        road.RenderSensor(car.GetSensorPts())
        RenderSpeedometer()
        RenderSimulationCount()

    def Save():
        torch.save(car.GetNetworks()[0].state_dict(), acceleration_path)
        torch.save(car.GetNetworks()[1].state_dict(), turn_path)
        torch.save(car.GetNetworks()[2].state_dict(), value_path)
        car.SaveData()

    def Start(): car.Reset()

    def Update():
        # Run car's movement and update road
        car.Run()
        car_pos = car.GetPosition()
        road.Generate(car_pos)

    # Start the simulation
    Start()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        if car.simulation_count() >= 999: running = False
        else:
            # Perform updates and render
            Update()
            Render()
            pygame.display.update()
            clock.tick(FPS)

    # Save data when simulation ends
    Save()
    print(f'Simulation {simulation_number} Terminated')
    pygame.quit()

def start_simulations(num_simulations):
    print('Starting',num_simulations,'simulations.')
    print('Start time:',datetime.now().time())
    processes = []
    for i in range(num_simulations):
        process = multiprocessing.Process(target=run_simulation, args=(i+1,))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()
    print('End time:',datetime.now().time())
    print('All simulations completed.')

if __name__ == "__main__":
    num_simulations = 1  # Set the number of simulations to run concurrently
    start_simulations(num_simulations)
