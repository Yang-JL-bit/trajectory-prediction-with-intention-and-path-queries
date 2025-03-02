'''
Author: Yang Jialong
Date: 2024-11-06 20:00:09
LastEditors: Please set LastEditors
LastEditTime: 2025-02-25 16:48:21
Description: constant
PS: 该部分代码引用了 https://github.com/RobertKrajewski/highD-dataset.git
'''
# TRACK FILE
BBOX = "bbox"
FRAMES = "frames"
FRAME = "frame"
TRACK_ID = "id"
X = "x"
Y = "y"
WIDTH = "width"
HEIGHT = "height"
X_VELOCITY = "xVelocity"
Y_VELOCITY = "yVelocity"
X_ACCELERATION = "xAcceleration"
Y_ACCELERATION = "yAcceleration"
FRONT_SIGHT_DISTANCE = "frontSightDistance"
BACK_SIGHT_DISTANCE = "backSightDistance"
DHW = "dhw"
THW = "thw"
TTC = "ttc"
PRECEDING_X_VELOCITY = "precedingXVelocity"
PRECEDING_ID = "precedingId"
FOLLOWING_ID = "followingId"
LEFT_PRECEDING_ID = "leftPrecedingId"
LEFT_ALONGSIDE_ID = "leftAlongsideId"
LEFT_FOLLOWING_ID = "leftFollowingId"
RIGHT_PRECEDING_ID = "rightPrecedingId"
RIGHT_ALONGSIDE_ID = "rightAlongsideId"
RIGHT_FOLLOWING_ID = "rightFollowingId"
LANE_ID = "laneId"

# STATIC FILE
INITIAL_FRAME = "initialFrame"
FINAL_FRAME = "finalFrame"
NUM_FRAMES = "numFrames"
CLASS = "class"
DRIVING_DIRECTION = "drivingDirection"
TRAVELED_DISTANCE = "traveledDistance"
MIN_X_VELOCITY = "minXVelocity"
MAX_X_VELOCITY = "maxXVelocity"
MEAN_X_VELOCITY = "meanXVelocity"
MIN_DHW = "minDHW"
MIN_THW = "minTHW"
MIN_TTC = "minTTC"
NUMBER_LANE_CHANGES = "numLaneChanges"

# VIDEO META
ID = "id"
FRAME_RATE = "frameRate"
LOCATION_ID = "locationId"
SPEED_LIMIT = "speedLimit"
MONTH = "month"
WEEKDAY = "weekDay"
START_TIME = "startTime"
DURATION = "duration"
TOTAL_DRIVEN_DISTANCE = "totalDrivenDistance"
TOTAL_DRIVEN_TIME = "totalDrivenTime"
N_VEHICLES = "numVehicles"
N_CARS = "numCars"
N_TRUCKS = "numTrucks"
UPPER_LANE_MARKINGS = "upperLaneMarkings"
LOWER_LANE_MARKINGS = "lowerLaneMarkings"

SAMPLE_RATE = 25


#轨迹预测采样
MIN_LANE_CHANGE_TIME = 1
MAX_LANE_CHANGE_TIME = 12
LANE_CHANGE_SAMPLR_STEP = 0.5

RAW_DATA_DIR = r"C:\\Users\\Public\\yjl\\master_project\\code\\highD\\data\\"   ## 原始数据所在文件夹
PROCESSED_DATA_DIR = r"E:\\master project\dataset\\highD\\processed_data_0102_3s\\" ## 处理后的数据所在文件夹
DATASET_DIR = r"C:\\Users\\Public\\yjl\\master_project\\code\\highD\\"  ## 数据集所在文件夹
