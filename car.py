import pygame
import math
from sensor import Sensor
from PolicyNetwork import PolicyNetwork
from ValueNetwork import ValueNetwork
import numpy
import torch

class Car:

    def __init__(self,window,env):
        # Set simulation window
        self.window = window
        self.WIDTH, self.HEIGHT = window.get_size()

        # Enviornment
        self.env = env

        # Sensor parameters
        self.num_of_sensors = 50
        self.sensor = Sensor(window, env, self.num_of_sensors)
        # Car parameters
        self.car_x, self.car_y = 0,0
        self.car_angle = 0  
        self.car_speed = 0
        self.max_speed = 10
        self.acceleration = 0.1
        self.brake_deceleration = 0.2
        self.friction = 0.05
        self.turn_speed = 2
        self.car_size = (50,100)

        # Networks
        input_size = 4*self.num_of_sensors+2
        self.accel_policy = PolicyNetwork(input_size)
        self.turn_policy = PolicyNetwork(input_size)
        self.value_network = ValueNetwork(input_size)

        # Find and set car image
        self.car_image = pygame.image.load('car.png')

        # Rescale car image
        self.car_image = pygame.transform.scale(self.car_image, self.car_size)

    def GetSensorData(self): return self.sensor.GetSensorData(self.car_size,self.car_angle,[self.car_x,self.car_y])
    def GetSensorPts(self): return self.sensor.GetSensorPts()

    # Set car angle
    def SetRotation(self, angle): self.car_angle = angle
    
    def ClipRotation(self):
        '''
        Ensure car angle range stays within (-180,180)
        '''
        
        if self.car_angle > 180: self.car_angle -= 360
        if self.car_angle < -180: self.car_angle += 360

    def Run(self):
        state = self.GetSensorData()
        state.append(self.acceleration)
        state.append(self.turn_speed)
        state = numpy.array(state)
        action = [self.accel_policy(torch.from_numpy(state).float())[0],self.turn_policy(torch.from_numpy(state).float())[0]]
        #self.acceleration = action[0]
        #self.turn_speed = action[1]
        reward = numpy.array(self.Reward())
        trajectory = [state,action,reward]

    def Move(self):
        '''
        Handle manual user input for car movement
        '''

        # Get keys being pressed by user
        keys = pygame.key.get_pressed()

        # Handle acceleration and deceleration
        if keys[pygame.K_UP]:self.car_speed -= self.acceleration
        if keys[pygame.K_DOWN]:self.car_speed += self.brake_deceleration

        # Only turn if car is moving (forward or backward)
        if (keys[pygame.K_DOWN] or keys[pygame.K_UP]) and keys[pygame.K_LEFT]: self.car_angle += self.turn_speed
        if (keys[pygame.K_DOWN] or keys[pygame.K_UP]) and keys[pygame.K_RIGHT]: self.car_angle -= self.turn_speed
        
        # Clip car angle to (-180,180)
        self.ClipRotation()

        # Limit speed
        if self.car_speed > self.max_speed: self.car_speed = self.max_speed
        if self.car_speed < -self.max_speed // 2: self.car_speed = -self.max_speed // 2

        # Apply friction to slow the car down when no keys are pressed
        if self.car_speed > 0: self.car_speed -= self.friction
        elif self.car_speed < 0: self.car_speed += self.friction
        if abs(self.car_speed) < self.friction: self.car_speed = 0  

        # Update car position
        self.car_x -= self.car_speed * math.sin(math.radians(self.car_angle))
        self.car_y += self.car_speed * math.cos(math.radians(self.car_angle))
    
    # Return the position of the car
    def GetPosition(self): return (self.car_x,self.car_y)
    
    def GetSize(self): return self.car_size
    
    def GetAngle(self): return self.car_angle
    
    def distance_from_center(self):
        road_center = self.env.get_center_pt()
        return self.car_x

    def Reward(self): 
        road_width = self.env.get_road_width()
        offset = self.distance_from_center()
        return 1 - 2 * offset / road_width if offset < road_width / 2 else 0

    def Render(self):
        '''
        Render car on to window
        '''

        car_width,car_height = self.car_size

        # Rotate image based on angle
        rotated_image = pygame.transform.rotate(self.car_image, self.car_angle)
        new_rect = rotated_image.get_rect(center=self.car_image.get_rect(topleft=(self.WIDTH // 2-(car_width // 2), self.HEIGHT // 2-(car_height // 2))).center)

        # Display image onto rectangle
        self.window.blit(rotated_image, new_rect.topleft)