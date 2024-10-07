import math
from road import Road
class Sensor:
    def __init__(self, window, env, num_of_sensors):
        self.env = env          # Reference to enviornment for queries
        self.sensor_pts = []    # Store sensor points
        self.window = window

        # Number of lines produced by sensor at each corner
        self.num_of_sensors = num_of_sensors

    # Function to rotate a point around the center by a given angle
    def rotate_point(self, point, angle):
        x, y = point
        x_rotated = x * math.cos(angle) - y * math.sin(angle) # Recalculate new x point based on current angle
        y_rotated = x * math.sin(angle) + y * math.cos(angle) # Recalculate new y point based on current angle
        return x_rotated, y_rotated
    
    # Return distance between two points
    def Distance(self,p1,p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x1-x2)**2+(y1-y2)**2)
    
    # Return sensor points
    def GetSensorPts(self): return self.sensor_pts

    def Generate(self,car_size, car_angle, car_pos):
        '''
        Generate sensor points based on current orientation of the car
        '''

        # Car parameters
        car_angle = math.radians(car_angle)
        self.env.Generate(car_pos)

        # Road parameters
        points = self.env.get_points()
        segment_height = self.env.get_segment_height()
        road_width = self.env.get_road_width()

        WIDTH,HEIGHT = self.window.get_size()

        center_point = HEIGHT // 2
        car_width, car_height = car_size
        # Reset current sensor orientation
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

        for z in range(self.num_of_sensors):
            #TOP RIGHT SENSOR
            self.sensor_pts.append(top_right_pt)
            y = top_right_pt[1] - car_center_y
            y = int(round(y/segment_height))
            road_point = (points[center_point+y][0] + road_width / 2, points[center_point+y][1])
            d = self.Distance(top_right_pt, road_point)
            y -= p*int(d*math.sin(car_angle))
            road_point =  (points[center_point+y-z*p][0] + road_width / 2, points[center_point+y-z*p][1])
            self.sensor_pts.append((points[center_point+y-z*p][0] + road_width / 2, points[center_point+y-z*p][1]))

            #TOP LEFT SENSOR
            self.sensor_pts.append(top_left_pt)
            _,y = top_left_pt
            y -= car_center_y
            y = int(round(y/segment_height))
            road_point = (points[center_point+y][0] - road_width / 2, points[center_point+y][1])
            d = self.Distance(top_left_pt, road_point)
            y += p*int(d*math.sin(car_angle))
            road_point = (points[center_point+y-z*p][0] - road_width / 2, points[center_point+y-z*p][1])
            self.sensor_pts.append(road_point)

            #BOTTOM RIGHT SENSOR
            self.sensor_pts.append(bot_right_pt)
            _,y = bot_right_pt
            y -= car_center_y
            y = int(round(y/segment_height))
            road_point = (points[center_point+y][0] + road_width / 2, points[center_point+y][1])
            d = self.Distance(bot_right_pt, road_point)
            y -= p*int(d*math.sin(car_angle))
            road_point = (points[center_point+y+z*p][0] + road_width / 2, points[center_point+y+z*p][1])
            self.sensor_pts.append(road_point)

            #BOTTOM LEFT SENSOR
            self.sensor_pts.append(bot_left_pt)
            _,y = bot_left_pt
            y -= car_center_y
            y = int(round(y/segment_height))
            road_point = (points[center_point+y][0] - road_width / 2, points[center_point+y][1])
            d = self.Distance(bot_left_pt, road_point)
            y += p*int(d*math.sin(car_angle))
            road_point = (points[center_point+y+z*p][0] - road_width / 2, points[center_point+y+z*p][1])
            self.sensor_pts.append(road_point)


    def GetSensorData(self,car_size, car_angle, car_pos):
        '''
        Return the distances generated by the sensors
        '''

        #Generate sensors
        self.Generate(car_size, car_angle, car_pos)

        # Array to store sensor data
        data = []

        # For each sensor line calculate the distance of line and store in data array
        for i in range(0,len(self.sensor_pts)-1,2):
            pt1,pt2 = self.sensor_pts[0],self.sensor_pts[1]
            data.append(self.Distance(pt1,pt2))

        #self.env.RenderSensor(self.sensor_pts)

        return data