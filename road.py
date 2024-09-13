import random
import pygame
import noise
import math
class Road:
    
    def __init__(self,window):
        # Set simulation window
        self.window = window

        # Road parameters
        self.points = []
        self.road_width = 120
        self.segment_height = 5
        self.alignment = 0

        # Noise parameters
        self.scale = 0.00085 # How stretched out is the road

        # Seed for rendomized road generation
        self.seed = random.uniform(0,1)
        
        # Colors
        self.ROAD_COLOR = (50, 50, 50)

    def Recenter(self):
        '''
        Recenter at beginning of the simulation such that car is centered in the middle of the road

        return car angle to realign car orientation
        '''

        if not self.points: raise Exception('Road points have not been generated')

        WIDTH,HEIGHT = self.window.get_size()

        i = len(self.points)//2

        self.alignment = WIDTH//2-self.points[i][0]
        
        x1,x2 = self.points[i][0], self.points[i - 1][0]
        y1,y2 = self.points[i][1], self.points[i - 1][1]

        m = (y2 - y1) / (x2 - x1)
        angle = math.degrees(-math.atan(m))

        return angle + 90 if angle < 0 else angle - 90
    
    def Generate(self,offset):
        '''
        Generate road points based on the car's position
        '''
        WIDTH,HEIGHT = self.window.get_size()

        offset_x,offset_y = offset
        
        # Clear all previous points
        self.points = []

        # Update points based on car locations
        for y in range(0, HEIGHT, self.segment_height):
            x_center = noise.pnoise1((offset_y + y + self.seed) * self.scale + self.seed) * WIDTH / 4 + WIDTH / 2
            self.points.append((x_center+offset_x+self.alignment, y))

    
    def Render(self):
        '''
        Render road on to window based on current road points
        Draw polygons based on the location of the points and width of the road
        '''

        if not self.points: raise Exception('Road points have not been generated')

        for i in range(len(self.points) - 1):
            pygame.draw.polygon(self.window, self.ROAD_COLOR, [
                (self.points[i][0] - self.road_width / 2, self.points[i][1]),
                (self.points[i][0] + self.road_width / 2, self.points[i][1]),
                (self.points[i + 1][0] + self.road_width / 2, self.points[i + 1][1]),
                (self.points[i + 1][0] - self.road_width / 2, self.points[i + 1][1])
            ])

        for i in range(len(self.points) - 1):
            pygame.draw.line(self.window, (255,255,255),
                (self.points[i][0] - self.road_width / 2, self.points[i][1]),
                (self.points[i + 1][0] - self.road_width / 2, self.points[i + 1][1]), 
                width=3)
        
        for i in range(len(self.points) - 1):
            pygame.draw.line(self.window, (255,255,255),
                (self.points[i + 1][0] + self.road_width / 2, self.points[i + 1][1]),
                (self.points[i][0] + self.road_width / 2, self.points[i][1]), 
                width=3)
        

