import pygame
import math

class Car:

    def __init__(self,window):
        # Set simulation window
        self.window = window
        self.WIDTH, self.HEIGHT = window.get_size()

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

        # Find and set car image
        self.car_image = pygame.image.load('car.png')

        # Rescale car image
        self.car_image = pygame.transform.scale(self.car_image, self.car_size)


    # Set car angle
    def SetRotation(self, angle): self.car_angle = angle
    
    def ClipRotation(self):
        '''
        Ensure car angle range stays within (-180,180)
        '''
        
        if self.car_angle > 180: self.car_angle -= 360
        if self.car_angle < -180: self.car_angle += 360

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