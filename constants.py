import numpy as np

# radius of the scene, used for filtering points by the distance from track center
MIN_RADIUS_OF_SCENE = 50

# max distance between two lanes' points to decide if they are connected, while traversing lanes
CONNECTION_RADIUS = 1.0

# max angle difference between two lanes' points to decide if they are connected, while traversing lanes
CONNECTION_DIR = 0.2

# max distance from first found projection we wanna go for while traversing lanes
MAX_DISTANCE_ALONG_GRAPH = 100.

# closure distance between our lane point and start point from another lane for making a decision, they are connected
TEE_CLOSE_DIST = 0.05

# indices
X, Y = 0, 1

# estimation of parabola z = a + b*t + c*t^2, where z = x, y.
# For t = -2, -1, 0 we have equation X * coeff = measured_values. So coeff = inv(X) * measured_values
# Here is inv(X) matrix:
ANGLE_ESTIMATION_MATRIX = np.array([[0.0,  0.0,  1.0],
                                    [0.5, -2.0,  1.5],
                                    [0.5, -1.0,  0.5]])

# If we have found coeff ⬆️, we can now find the coords of the next point (t = 1), by [1, 1, 1] * coeff
NEXT_POINT_STRING = np.ones((1, 3))
