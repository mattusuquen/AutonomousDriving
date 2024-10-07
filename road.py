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
        self.segment_height = 1
        self.alignment = 0

        # Noise parameters
        self.scale = 0.00085 # How stretched out is the road

        # Seed for rendomized road generation
        self.seed = random.uniform(0,1)
        
        # Colors
        self.ROAD_COLOR = (50, 50, 50)
    

    # Return road segement height
    def get_segment_height(self): return self.segment_height

    # Return road width
    def get_road_width(self): return self.road_width

    # Return road points
    def get_points(self): return self.points

    def get_center_pt(self): 
        _,HEIGHT = self.window.get_size()
        return (self.points[HEIGHT // 2][0] + self.points[HEIGHT // 2][1]) / 2

    def Recenter(self):
        '''
        Recenter at beginning of the simulation such that car is centered in the middle of the road

        return car angle to realign car orientation
        '''

        if not self.points: raise Exception('Road points have not been generated')

        WIDTH = self.window.get_size()[0]

        i = len(self.points)//2

        self.alignment = WIDTH//2-self.points[i][0]
        
        x1,x2 = self.points[i][0], self.points[i - 1][0]
        y1,y2 = self.points[i][1], self.points[i - 1][1]
        # Calculate slope inline with current orientation
        m = (y2 - y1) / (x2 - x1)
        # Recalculate angle
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

    
    def RenderSensor(self,sensor_pts):
        '''
        Render sensors projected from the four corners of the car
        '''

        for i in range(0,len(sensor_pts)-1,2):
            p1,p2 = sensor_pts[i],sensor_pts[i+1]
            pygame.draw.line(self.window, (0,0,0), p1, p2)

    def Render(self):
        '''
        Render road on to window based on current road points
        Draw polygons based on the location of the points and width of the road
        '''

        # Check if road has been generated
        # If not throw error message
        if not self.points: raise Exception('Road points have not been generated')

        # Render polygons for the road
        for i in range(len(self.points) - 1):
            pygame.draw.polygon(self.window, self.ROAD_COLOR, [
                (self.points[i][0] - self.road_width / 2, self.points[i][1]),
                (self.points[i][0] + self.road_width / 2, self.points[i][1]),
                (self.points[i + 1][0] + self.road_width / 2, self.points[i + 1][1]),
                (self.points[i + 1][0] - self.road_width / 2, self.points[i + 1][1])
            ])

        # Render lines for the road borders
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
            
