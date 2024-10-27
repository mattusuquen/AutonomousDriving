import pygame
from car import Car
from road import Road
import math
import torch
import threading

# Initialize Pygame
pygame.init()

# Display Settings
FPS = 60
WIDTH, HEIGHT = 1000, 1000
BACKGROUND_COLOR = (100, 225, 100)

# Network locations
acceleration_path = 'models/acceleration_network.pth'
turn_path = 'models/turn_network.pth'
value_path = 'models/value_network.pth'

# Font
font = pygame.font.SysFont('Arial', 32)

class Simulation(threading.Thread):
    def __init__(self, simulation_id):
        threading.Thread.__init__(self)
        self.simulation_id = simulation_id
        self.running = True
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f'Driving Simulator {self.simulation_id}')

        # Car and Road objects
        self.road = Road(self.window)
        self.car = Car(self.window, self.road)
        self.clock = pygame.time.Clock()

    def RenderSpeedometer(self):
        speed_text = font.render('Speed: '+str(round(-self.car.GetSpeed(), 2)), True, (255, 255, 255))
        angle_text = font.render('Angle: '+str(round(self.car.GetAngle(), 2)), True, (255, 255, 255))
        self.window.blit(speed_text, (5, 0))
        self.window.blit(angle_text, (5, 40))

    def RenderSimulationCount(self):
        count_text = font.render('Simulation: '+str(self.car.simulation_count())+'/1000', True, (255, 255, 255))
        self.window.blit(count_text, (5, 80))

    def Render(self):
        # Render the road and car on the window
        self.road.Render()
        self.car.Render()
        # Render sensor
        self.road.RenderSensor(self.car.GetSensorPts())
        # Print car speed and angle
        self.RenderSpeedometer()
        # Print simulation number
        self.RenderSimulationCount()

    def Save(self):
        # Save models
        torch.save(self.car.GetNetworks()[0].state_dict(), acceleration_path)
        torch.save(self.car.GetNetworks()[1].state_dict(), turn_path)
        torch.save(self.car.GetNetworks()[2].state_dict(), value_path)
        # Save trajectory data
        self.car.SaveData()

    def Start(self): self.car.Reset()

    def Update(self):
        # Set simulation FPS
        self.clock.tick(FPS)

        # Reset screen background
        self.window.fill(BACKGROUND_COLOR)

        # Update for car movement based on user input
        self.car.Run()

        # Generate road dynamically based on car position
        car_pos = self.car.GetPosition()
        self.road.Generate(car_pos)

        # Render the road and car on the window
        self.Render()

        # Update display
        pygame.display.update()

    def run(self):
        self.Start()  # Run initialization
        while self.running:
            # Stop main loop when user closes window or simulation limit is reached
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
            if self.car.simulation_count() >= 1000:
                self.running = False
            # Otherwise, update the window accordingly
            self.Update()

        # On quit, save simulation data
        self.Save()
        pygame.quit()

if __name__ == "__main__":
    n = 5  # Number of simulations to run in parallel
    simulations = [Simulation(i) for i in range(n)]

    # Start all simulations
    for sim in simulations: sim.start()

    # Wait for all simulations to finish
    for sim in simulations: sim.join()

    print("All simulations completed.")
