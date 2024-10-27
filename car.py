import pygame
import math
from sensor import Sensor
from PolicyNetwork import PolicyNetwork
from ValueNetwork import ValueNetwork
import numpy as np
import torch
import os
import pandas as pd
import random

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
        self.accel_policy = PolicyNetwork(input_size,mean_range=1,stdev_coeff=0.1)
        self.turn_policy = PolicyNetwork(input_size,mean_range=1,stdev_coeff=0.1)
        self.value_network = ValueNetwork(input_size)

        # Load networks if exists
        acceleration_path = 'models/acceleration_network.pth'
        turn_path = 'models/turn_network.pth'
        value_path = 'models/value_network.pth'
        if os.path.exists(acceleration_path): self.accel_policy.load_state_dict(torch.load(acceleration_path))
        if os.path.exists(turn_path): self.turn_policy.load_state_dict(torch.load(turn_path))
        if os.path.exists(value_path): self.value_network.load_state_dict(torch.load(value_path))

        self.resetTimer = 0
        self.resetTimeLimit = 100

        # Data .csv location
        self.trajectories_path = 'data/trajectories.csv'
        # Dataframe
        self.columns = ['state_'+str(i+1) for i in range(self.num_of_sensors*4+2)] + ['action_1','action_2','reward']
        self.trajectories = []

        # Find and set car image
        car_img_path = 'images/car.png'
        self.car_image = pygame.image.load(car_img_path)

        # Rescale car image
        self.car_image = pygame.transform.scale(self.car_image, self.car_size)

<<<<<<< HEAD
    def simulation_count(self): return len(self.trajectories)//self.resetTimeLimit
=======

>>>>>>> 8b84eab937deecec6fc53e0e922f3612d8bfd2df
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
    
    def Reset(self):
        self.car_x = 0
        self.car_y = 0
        self.env.Generate((0,0),seed=random.uniform(0,1))
        angle = self.env.Recenter() # Reposition road so car initialized on the road
        self.SetRotation(angle) # Adjust car orientation
        self.resetTimer = 0

    def ClampRotation(self):
        '''
        Ensure car angle range stays within (-180,180)
        '''
        self.car_angle = (self.car_angle + 180) % 360 - 180
    
    def distance_from_center(self):
        '''
        return the number of pixels the far center is to the road center
        '''
        road_center = self.env.get_center_pt()-self.WIDTH/2
        return abs(road_center)

    
    def Reward(self):
        '''
        Reward function
        '''
        # Minimum possible reward
        min_reward = 1e-6

        # Road attributes
        offset = self.distance_from_center()
        road_width = self.env.get_road_width()/2

        # Distance travelled
        dist_from_start = math.sqrt((self.car_x)**2 + (self.car_y)**2)

        # Return calculated reward
        if offset < road_width: return (1 - min_reward) * ((road_width - offset) / road_width) + min_reward + dist_from_start
        if self.car_speed < 1: return min_reward
        return min_reward

    def Render(self):
        '''
        Render car on to window
        '''
        # Car size
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
        
        state_array = np.array([state])                      # Convert state to numpy array
        state_tensor = torch.from_numpy(state_array).float() # Convert state to tensor

        # Find mean and standard deviation based on current policy
        accel_mean, accel_stdev = self.accel_policy(state_tensor).detach().numpy()[0]
        turn_mean, turn_stdev = self.turn_policy(state_tensor).detach().numpy()[0]

        # Update acceleration and turn speed
        self.acceleration = np.random.normal(accel_mean, accel_stdev) # Get acceleration based on normal distribution
        self.turn_speed = np.random.normal(turn_mean, turn_stdev)     # Get turn speed based on normal distribution
        
        # Calculate and store award at given state
        reward = self.Reward()
        
        # Move car based on accerlation and turn_speed
        # Updating the state space
        self.Move()

        # Store trajectory (state, action, reward)
        trajectory = state+[self.acceleration]+[self.turn_speed]+[reward]
        self.StoreTrajectory(trajectory)
        
        # Simulation reset scheduling
        if self.resetTimer >= self.resetTimeLimit: self.Reset()
        else: self.resetTimer += 1

<<<<<<< HEAD
    def StoreTrajectory(self,trajectory): 
        if self.sensor.off_road: # If car off road end simulation
            self.trajectories += [trajectory]*(self.resetTimeLimit-self.resetTimer)
            self.resetTimer = self.resetTimeLimit
        else: self.trajectories.append(trajectory)

    def SaveData(self):
        '''
        Save trajectory data in a data frame
        Save data frame in csv file
        '''
=======
    def StoreTrajectory(self,trajectory): self.trajectories.append(trajectory)

    def SaveData(self):
>>>>>>> 8b84eab937deecec6fc53e0e922f3612d8bfd2df
        dataframe = pd.DataFrame(self.trajectories,columns=self.columns)
        dataframe.to_csv(self.trajectories_path)


    def Move(self):
        '''
        Update car location based on acceleration and turn speed
        '''
        if self.sensor.off_road: 
            self.car_speed = 0
            return
        # Update angle and speed
        if self.car_speed != 0: self.car_angle += self.turn_speed
        self.car_speed -= self.acceleration

        # Clamp car angle to (-180,180)
        self.ClampRotation()

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
        
        self.Move()