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

        self.sensor_pts = []
    def get_segment_height(self): return self.segment_height
    def get_road_width(self): return self.road_width
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

    
    def GetSensorData(self,car_size, car_angle):

        WIDTH,HEIGHT = self.window.get_size()

        center_point = len(self.points) // 2
        car_width, car_height = car_size
        self.sensor_pts = []

        # Center of the car
        car_center_x = WIDTH // 2
        car_center_y = HEIGHT // 2
        half_car_width, half_car_height = car_width//2-10, car_height//2-10

        # Car angle in radians
        theta = -car_angle

        # Define the four corner points relative to the center of the car
        top_left_pt = (-half_car_width, -half_car_height)
        top_right_pt = (half_car_width, -half_car_height)
        bot_right_pt = (half_car_width, half_car_height)
        bot_left_pt = (-half_car_width, half_car_height)

        # Function to rotate a point around the center by a given angle
        def rotate_point(point, angle):
            x, y = point
            x_rotated = x * math.cos(angle) - y * math.sin(angle)
            y_rotated = x * math.sin(angle) + y * math.cos(angle)
            return x_rotated, y_rotated

        # Rotate each corner point
        rotated_top_left_pt = rotate_point(top_left_pt, theta)
        rotated_top_right_pt = rotate_point(top_right_pt, theta)
        rotated_bot_right_pt = rotate_point(bot_right_pt, theta)
        rotated_bot_left_pt = rotate_point(bot_left_pt, theta)

        # Translate rotated points back to the center of the car
        top_left_pt = (rotated_top_left_pt[0] + car_center_x, rotated_top_left_pt[1] + car_center_y)
        top_right_pt = (rotated_top_right_pt[0] + car_center_x, rotated_top_right_pt[1] + car_center_y)
        bot_right_pt = (rotated_bot_right_pt[0] + car_center_x, rotated_bot_right_pt[1] + car_center_y)
        bot_left_pt = (rotated_bot_left_pt[0] + car_center_x, rotated_bot_left_pt[1] + car_center_y)

        def flip(p1,p2,p3,p4): return p2,p1,p4,p3

        #flip modifier
        p = 1

        #If backward flip sensor behavior
        if abs(car_angle)%(2*math.pi)+math.pi/2 >= math.pi: 
            top_left_pt,top_right_pt,bot_right_pt,bot_left_pt = flip(top_left_pt,top_right_pt,bot_right_pt,bot_left_pt)
            p = -1

        def Distance(p1,p2):
            x1, y1 = p1
            x2, y2 = p2
            return math.sqrt((x1-x2)**2+(y1-y2)**2)

        num_of_sensors = 50

        data = [[0 for _ in range(num_of_sensors+2)] for _ in range(6)]
        
        for z in range(num_of_sensors):
            #TOP RIGHT
            self.sensor_pts.append(top_right_pt)
            y = top_right_pt[1] - car_center_y
            y = int(round(y/self.segment_height))
            road_point = (self.points[center_point+y][0] + self.road_width / 2, self.points[center_point+y][1])
            d = Distance(top_right_pt, road_point)
            y -= p*int(d*math.sin(car_angle))
            road_point =  (self.points[center_point+y-z*p][0] + self.road_width / 2, self.points[center_point+y-z*p][1])
            self.sensor_pts.append((self.points[center_point+y-z*p][0] + self.road_width / 2, self.points[center_point+y-z*p][1]))

            data[(len(self.sensor_pts)//2)%4][z] = Distance(top_right_pt,road_point)

            #TOP LEFT
            self.sensor_pts.append(top_left_pt)
            _,y = top_left_pt
            y -= car_center_y
            y = int(round(y/self.segment_height))
            road_point = (self.points[center_point+y][0] - self.road_width / 2, self.points[center_point+y][1])
            d = Distance(top_left_pt, road_point)
            y += p*int(d*math.sin(car_angle))
            road_point = (self.points[center_point+y-z*p][0] - self.road_width / 2, self.points[center_point+y-z*p][1])
            self.sensor_pts.append(road_point)

            data[(len(self.sensor_pts)//2)%4][z] = Distance(top_left_pt,road_point)

            #BOTTOM RIGHT
            self.sensor_pts.append(bot_right_pt)
            _,y = bot_right_pt
            y -= car_center_y
            y = int(round(y/self.segment_height))
            road_point = (self.points[center_point+y][0] + self.road_width / 2, self.points[center_point+y][1])
            d = Distance(bot_right_pt, road_point)
            y -= p*int(d*math.sin(car_angle))
            road_point = (self.points[center_point+y+z*p][0] + self.road_width / 2, self.points[center_point+y+z*p][1])
            self.sensor_pts.append(road_point)

            data[(len(self.sensor_pts)//2)%4][z] = Distance(bot_right_pt,road_point)

            #BOTTOM LEFT
            self.sensor_pts.append(bot_left_pt)
            _,y = bot_left_pt
            y -= car_center_y
            y = int(round(y/self.segment_height))
            road_point = (self.points[center_point+y][0] - self.road_width / 2, self.points[center_point+y][1])
            d = Distance(bot_left_pt, road_point)
            y += p*int(d*math.sin(car_angle))
            road_point = (self.points[center_point+y+z*p][0] - self.road_width / 2, self.points[center_point+y+z*p][1])
            self.sensor_pts.append(road_point)

            data[(len(self.sensor_pts)//2)%4][z] = Distance(bot_left_pt,road_point)

        return data

    def RenderSensor(self,sensor_pts):
        for i in range(0,len(sensor_pts)-1,2):
            p1,p2 = sensor_pts[i],sensor_pts[i+1]
            pygame.draw.line(self.window, (0,0,0), p1, p2)

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
        self.RenderSensor(self.sensor_pts)
