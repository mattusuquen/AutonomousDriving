import math
from road import Road
class Sensor:
    def __init__(self, window, env):
        self.env = env
        self.sensor_pts = []
        self.window = window


    # Function to rotate a point around the center by a given angle
    def rotate_point(self, point, angle):
        x, y = point
        x_rotated = x * math.cos(angle) - y * math.sin(angle)
        y_rotated = x * math.sin(angle) + y * math.cos(angle)
        return x_rotated, y_rotated
    
    def Distance(self,p1,p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x1-x2)**2+(y1-y2)**2)
    
    def GetSensorPts(self): return self.sensor_pts

    def GetSensorData(self,car_size, car_angle):
        
        points = self.env.get_points()
        segment_height = self.env.get_segment_height()
        road_width = self.env.get_road_width()

        WIDTH,HEIGHT = self.window.get_size()

        center_point = len(points) // 2
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


        # Rotate each corner point
        rotated_top_left_pt = self.rotate_point(top_left_pt, theta)
        rotated_top_right_pt = self.rotate_point(top_right_pt, theta)
        rotated_bot_right_pt = self.rotate_point(bot_right_pt, theta)
        rotated_bot_left_pt = self.rotate_point(bot_left_pt, theta)

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


        num_of_sensors = 50

        data = [[0 for _ in range(num_of_sensors+2)] for _ in range(6)]
        
        for z in range(num_of_sensors):
            #TOP RIGHT
            self.sensor_pts.append(top_right_pt)
            y = top_right_pt[1] - car_center_y
            y = int(round(y/segment_height))
            road_point = (points[center_point+y][0] + road_width / 2, points[center_point+y][1])
            d = self.Distance(top_right_pt, road_point)
            y -= p*int(d*math.sin(car_angle))
            road_point =  (points[center_point+y-z*p][0] + road_width / 2, points[center_point+y-z*p][1])
            self.sensor_pts.append((points[center_point+y-z*p][0] + road_width / 2, points[center_point+y-z*p][1]))

            data[(len(self.sensor_pts)//2)%4][z] = self.Distance(top_right_pt,road_point)

            #TOP LEFT
            self.sensor_pts.append(top_left_pt)
            _,y = top_left_pt
            y -= car_center_y
            y = int(round(y/segment_height))
            road_point = (points[center_point+y][0] - road_width / 2, points[center_point+y][1])
            d = self.Distance(top_left_pt, road_point)
            y += p*int(d*math.sin(car_angle))
            road_point = (points[center_point+y-z*p][0] - road_width / 2, points[center_point+y-z*p][1])
            self.sensor_pts.append(road_point)

            data[(len(self.sensor_pts)//2)%4][z] = self.Distance(top_left_pt,road_point)

            #BOTTOM RIGHT
            self.sensor_pts.append(bot_right_pt)
            _,y = bot_right_pt
            y -= car_center_y
            y = int(round(y/segment_height))
            road_point = (points[center_point+y][0] + road_width / 2, points[center_point+y][1])
            d = self.Distance(bot_right_pt, road_point)
            y -= p*int(d*math.sin(car_angle))
            road_point = (points[center_point+y+z*p][0] + road_width / 2, points[center_point+y+z*p][1])
            self.sensor_pts.append(road_point)

            data[(len(self.sensor_pts)//2)%4][z] = self.Distance(bot_right_pt,road_point)

            #BOTTOM LEFT
            self.sensor_pts.append(bot_left_pt)
            _,y = bot_left_pt
            y -= car_center_y
            y = int(round(y/segment_height))
            road_point = (points[center_point+y][0] - road_width / 2, points[center_point+y][1])
            d = self.Distance(bot_left_pt, road_point)
            y += p*int(d*math.sin(car_angle))
            road_point = (points[center_point+y+z*p][0] - road_width / 2, points[center_point+y+z*p][1])
            self.sensor_pts.append(road_point)

            data[(len(self.sensor_pts)//2)%4][z] = self.Distance(bot_left_pt,road_point)


        return data