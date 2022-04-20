ROBOT_RADIUS = 10
OPTIMAL_GRAZING_FRONT_WIDTH = 15  # might increase
MIN_GRAZING_FRONT_WIDTH = 5  # meters
# obstacles with a lower width won't be considered during path planning
MIN_OBSTACLE_WIDTH = OPTIMAL_GRAZING_FRONT_WIDTH - MIN_GRAZING_FRONT_WIDTH
SNAP_ACTIVATE = False
SNAP_TOLERANCE = 10
SNAP_USE_CONVEX_HULL = True

BOUNDARY_COLOR = "blue"
BOUNDARY_WIDTH = 2
INTRA_PATH_COLOR = "silver"
INTRA_PATH_WIDTH = 1
INTRA_PATH_MARKER_SIZE = 12

INTER_PATH_COLOR = "dimgray"
INTER_PATH_WIDTH = 2
ACTIVATE_APPROX_POLYGON = True
APPROX_POLYGON_TOLERANCE = 1

RATE_AREA_LIMIT = .99  # .95
DISTANCE_STEP = 10
DISTANCE_STEP_INTERPOLATE = 1
INTER_FRONT_DISTANCE = .2

#SMOOTH_ANGLE = 28
SMOOTH_ANGLE_MAX = 60
SMOOTH_ANGLE_MIN = 20

ROBOT_BETA_ANGLE = 30

AREAL_SPEED = .316