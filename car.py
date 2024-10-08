import pygame
import math
from sensor import Sensor
from PolicyNetwork import PolicyNetwork
from ValueNetwork import ValueNetwork
import numpy
import torch
import os
import pandas as pd

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
        if os.path.exists("acceleration_network.pth"): self.accel_policy.load_state_dict(torch.load("acceleration_network.pth"))
        if os.path.exists("turn_network.pth"): self.turn_policy.load_state_dict(torch.load("turn_network.pth"))
        if os.path.exists("value_network.pth"): self.value_network.load_state_dict(torch.load("value_network.pth"))

        # Find and set car image
        self.car_image = pygame.image.load('car.png')

        # Rescale car image
        self.car_image = pygame.transform.scale(self.car_image, self.car_size)

        # Dataframe
        self.columns = ['state_'+str(i) for i in range(self.num_of_sensors*4+2)]
        self.columns += ['action_1','action_2','reward']
        self.trajectories = pd.DataFrame(columns=self.columns)

    def GetNetworks(self): return self.accel_policy, self.turn_policy, self.value_network

    # Return data from sensors
    def GetSensorData(self): return self.sensor.GetSensorData(self.car_size,self.car_angle,[self.car_x,self.car_y])

    def GetSensorPts(self): return self.sensor.GetSensorPts()

    # Return the position of the car
    def GetPosition(self): return (self.car_x,self.car_y)
    
    # Return the size of the car
    def GetSize(self): return self.car_size
    
    # Return car speed
    def GetSpeed(self): return self.car_speed

    # Return car angle
    def GetAngle(self): return self.car_angle

    # Set car angle
    def SetRotation(self, angle): self.car_angle = angle
    
    def ClipRotation(self):
        '''
        Ensure car angle range stays within (-180,180)
        '''
        
        if self.car_angle > 180: self.car_angle -= 360
        if self.car_angle < -180: self.car_angle += 360
    
    def distance_from_center(self):
        road_center = self.env.get_center_pt()-self.WIDTH/2
        return abs(road_center)

    # Reward function
    def Reward(self):
        min_reward = 0.0001
        offset = self.distance_from_center()
        road_width = self.env.get_road_width()/2
        return (1-min_reward)*((road_width-offset)/road_width)+min_reward if offset < road_width else min_reward

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
    
    def Run(self):
        '''
        Handle car movement via output of the policy network
        Store trajectory data for training
        '''
        state = []
        # Read data from sensor
        state = self.GetSensorData()
        # Add acceleration and turn_speed to current state
        state.append(self.acceleration)
        state.append(self.turn_speed)
        
        state = numpy.array(state)              # Convert state to numpy array
        state = torch.from_numpy(state).float() # Convert state to tensor

        # Find action based on current policy
        action = [self.accel_policy(state)[0],self.turn_policy(state)[0]]

        self.acceleration = action[0].detach().numpy()   # Set acceleration to network output
        self.turn_speed = action[1].detach().numpy()     # Set turn speed to network output
        
        # Calculate and store award at given state
        reward = self.Reward()
        
        # Move car based on accerlation and turn_speed
        self.Move()

        # Store trajectory (state, action, reward)
        trajectory = state.tolist()
        trajectory.append(self.acceleration)
        trajectory.append(self.turn_speed)
        trajectory.append(reward)
        trajectory = pd.DataFrame([trajectory],columns=self.columns)
        self.trajectories = pd.concat([self.trajectories,trajectory],ignore_index=True)
        self.trajectories.to_csv('trajectories.csv')

    def Move(self):
        '''
        Update car location based on acceleration and turn speed
        '''
        self.car_angle += self.turn_speed
        self.car_speed -= self.acceleration
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

    def MoveManually(self):
        '''
        Handle manual user input for car movement
        '''

        # Get keys being pressed by user
        keys = pygame.key.get_pressed()

        # Handle acceleration and deceleration
        if keys[pygame.K_UP]: self.car_speed -= self.acceleration
        if keys[pygame.K_DOWN]: self.car_speed += self.brake_deceleration

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