from enum import Enum
import numpy as np


class DatasetPart(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

THEMECOLORS = {
    'background': '#0e0e30',
    'dark-grey': '#353535',
    'medium-grey': '#454545',
    'light-grey': '#555555',
    'blue': '#0079cc',
    'dark-blue': '#0065bb',
    'light-blue': '#569cd6',
    'magenta': '#69217a',
    'light-magenta': '#da71d6',
    'green': '#24c93e',
    'dark-green': '#257057',
    'white': '#d0d0d0',
    'black': '#101010',
    'yellow': '#8b6023',
    'red': '#86372f',
}

MASK_VALUE = -100500
vehicle_contour = np.array([[-2.0, 1.7, 2.0,  2.0,  1.7, -2.0, -2.0],
                            [0.8,  0.8, 0.5, -0.5, -0.8, -0.8,  0.8]])



bus_contour = np.array([[-6.0, 6.0,  6.0, -6.0, -6.0],
                        [1.3,  1.3, -1.3, -1.3,  1.3]])

ped_contour = np.array([[4.8, 7.7, 9.3, 9.3, 8.7, 8.4, 8.1, 8.1, 8.3, 8.5, 8.9, 9.5, 10.1, 10.7, 11.1, 11.3, 11.5, 11.5, 11.3, 11.2, 11., 10.5, 10., 9.3, 9.3, 9.7, 11.3, 13.9, 13.4, 10.4, 9.8, 9.3, 13.1, 11.7,  8.5,  7.1,  5.1,  3.9,  5.6,  6.1, 7.5, 6.1, 5.6, 4.3, 4.8],
                        [6.2, 3.8, 3.8, 3.3, 3.0, 2.6, 1.9, 1.4, 0.9, 0.6, 0.2, 0.0,  0.0,  0.2,  0.6,  0.9,  1.4,  1.9,  2.3,  2.6, 2.8,  3.2, 3.3, 3.3, 3.8, 4.0,  7.2,  8.6,  9.7,  8.2, 7.2, 9.5, 15.8, 16.7, 11.6, 14.1, 16.6, 15.9, 13.2, 12.2, 6.0, 7.0, 9.9, 9.8, 6.2]])
ped_contour[1, :] = -ped_contour[1, :]
ped_contour[0, :] = ped_contour[0, :] - 8.9
ped_contour[1, :] = ped_contour[1, :] + 16.7/2
ped_contour = np.multiply(ped_contour, 1/16.7*1.5)

moto_contour = np.array([[-5.0, -3.0, -2.0, 2.0, 3.0, 5.0,  5.0,  3.0,  2.0, -2.0, -3.0, -5.0, -5.0],
                         [ 0.5,  0.5,  2.0, 2.0, 0.5, 0.5, -0.5, -0.5, -2.0, -2.0, -0.5, -0.5,  0.5]])
moto_contour = np.multiply(0.4, moto_contour)

other_contour = np.array([[-6.0, -3.0, 3.0, 6.0,  3.0, -3.0, -6.0],
                          [ 0.0,  5.0, 5.0, 0.0, -5.0, -5.0,  0.0]])
other_contour = np.multiply(0.2, other_contour)


class WaymoTrackCategory(Enum):
    SDC: int = 0
    TRACK_TO_PREDICT: int = 1
    UNSCORED: int = 2

def scale_object(length, width, object_type=0):
    # waymo types
    # Unset=0, Vehicle=1, Pedestrian=2, Cyclist=3, Other=4
    contour = {
        0: vehicle_contour,
        1: vehicle_contour,
        2: ped_contour,
        3: moto_contour,
        4: other_contour,
    }[object_type]

    contour_scale_x = max(contour[0, :]) - min(contour[0, :])
    contour_scale_y = max(contour[1, :]) - min(contour[1, :])
    scaled_contour = np.vstack((
        contour[0, :] * (length / contour_scale_x),
        contour[1, :] * (width / contour_scale_y)
    ))
    return scaled_contour

none_vector = np.array([None, None], ndmin=2)
SIGNIFICANT_SCALE_DELTA = 20.

### NOT USED ###
# scale_coefficient = 0.08
#
# contour_x = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.4, 1.6,  1.9,  2.3,  2.6,  3.0,  3.5,  4.0,  4.5,  5.0,  5.5,  6.0,  7.0,  8.3, 42.5, 44.5, 48.0, 50.2, 50.5, 51.0, 52.0, 52.5, 53.0, 53.5, 54.0, 54.5, 55.0, 55.1, 55.2]
# contour_y = [0.0, 4.3, 5.8, 6.9, 7.2, 7.6, 8.0, 8.4, 8.9, 9.2, 9.5, 9.7, 10.1, 10.5, 10.8, 11.1, 11.4, 11.6, 11.8, 11.9, 12.1, 12.2, 12.5, 12.6, 12.4, 12.6, 12.4, 12.2, 12.1, 11.9, 11.4, 11.1, 10.7, 10.1,  9.0,  7.3,  3.5,  2.3,  0.0]
# contour_x.extend(contour_x[::-1])
# contour_x = [-1 * j + 27.6 for j in contour_x]
# contour_y.extend([-1 * j for j in contour_y[::-1]])
# vehicle_contour = np.array([contour_x, contour_y]) * scale_coefficient
#
# back_glass_x = [48.5, 48.5, 48.4, 48.3, 48.0, 48.9, 49.9, 50.8, 51.7, 52.6, 53.1, 53.2, 53.2]
# back_glass_y = [ 0.0,  4.3,  5.2,  6.3,  8.0,  8.2, 8.5,   8.8,  9.3,  6.4,  4.2,  2.6,  0.0]
# back_glass_x.extend(back_glass_x[::-1])
# back_glass_y.extend([-1 * j for j in back_glass_y[::-1]])
#
# front_glass_x = [13.9, 14.0, 14.2, 14.5, 14.8, 15.2, 15.5, 16.0, 18.7, 20.1, 24.1, 23.9, 23.8]
# front_glass_y = [ 0.0,  2.3,  3.8,  5.5,  6.8,  8.4,  9.3, 10.6, 10.0,  9.6,  8.4,  6.5,  0.0]
# front_glass_x.extend(front_glass_x[::-1])
# front_glass_y.extend([-1 * j for j in front_glass_y[::-1]])
#
# side_glass_x = [18.1, 23.9, 25.0, 25.7, 26.6, 42.3, 43.8, 44.4, 44.6, 44.8, 45.2, 45.2, 45.1, 44.8, 44.3, 18.1]
# side_glass_y = [11.2,  9.3,  9.0,  8.9,  8.8,  8.8,  8.9,  9.0,  9.1,  9.3, 10.4, 10.8, 10.9, 11.1, 11.2, 11.2]
# # side_glass_x.extend(side_glass_x[::-1])
# # side_glass_y.extend([-1 * j for j in side_glass_y[::-1]])
#
# vehicle_glass_x = [*front_glass_x, *back_glass_x]
# vehicle_glass_y = [*front_glass_y, *back_glass_y]
# vehicle_glass_x = [-1 * j + 27.6 for j in vehicle_glass_x]
#
# vehicle_glass_x = np.array(vehicle_glass_x) * scale_coefficient
# vehicle_glass_y = np.array(vehicle_glass_y) * scale_coefficient
# vehicle_glass = np.vstack((vehicle_glass_x, vehicle_glass_y))