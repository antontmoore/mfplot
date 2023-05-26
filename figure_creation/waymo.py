from .figure_creator import FigureCreatorBaseClass
from typing import List
from .shared import THEMECOLORS
from .shared import DatasetPart
from .shared import none_vector
from .shared import scale_object
from .shared import WaymoTrackCategory
from pathlib import Path
# import math
# import os
# import uuid
# import time
#
# from matplotlib import cm
# import matplotlib.animation as animation
# import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
# # from IPython.display import HTML
# import itertools
import tensorflow as tf

from google.protobuf import text_format


# [0, 19].
# LaneCenter-Freeway = 1,
# LaneCenter-SurfaceStreet = 2,
# LaneCenter-BikeLane = 3,

# RoadLine-BrokenSingleWhite = 6,
# RoadLine-SolidSingleWhite = 7,
# RoadLine-SolidDoubleWhite = 8,

# RoadLine-BrokenSingleYellow = 9,
# RoadLine-BrokenDoubleYellow = 10,
# Roadline-SolidSingleYellow = 11,
# Roadline-SolidDoubleYellow=12,
# RoadLine-PassingDoubleYellow = 13,

# RoadEdgeBoundary = 15,
# RoadEdgeMedian = 16,
# StopSign = 17,
# Crosswalk = 18,
# SpeedBump = 19,
# other values are unknown types and should not be present.
ROADCOLORBYTYPE = {
    1: THEMECOLORS['dark-grey'],
    2: THEMECOLORS['dark-grey'],
    3: THEMECOLORS['dark-grey'],
    6: THEMECOLORS['light-grey'],
    7: THEMECOLORS['light-grey'],
    8: THEMECOLORS['light-grey'],
    9: THEMECOLORS['yellow'],
    10: THEMECOLORS['yellow'],
    11: THEMECOLORS['yellow'],
    12: THEMECOLORS['yellow'],
    13: THEMECOLORS['yellow'],
    15: THEMECOLORS['dark-green'],
    16: THEMECOLORS['light-grey'],
    17: THEMECOLORS['blue'],
    18: THEMECOLORS['magenta'],
    19: THEMECOLORS['red']

}
ROADDASHBYTYPE = {
    1: 'dot',
    2: 'dot',
    3: 'dot',
    6: 'dash',
    7: 'solid',
    8: 'solid',
    9: 'dash',
    10: 'dash',
    11: 'solid',
    12: 'solid',
    13: 'solid',
    15: 'solid',
    16: 'solid',
    17: 'solid',
    18: 'solid',
    19: 'solid'
}

ROADFILLBYTYPE = {
    1: 'none',
    2: 'none',
    3: 'none',
    6: 'none',
    7: 'none',
    8: 'none',
    9: 'none',
    10: 'none',
    11: 'none',
    12: 'none',
    13: 'none',
    15: 'none',
    16: 'toself',
    17: 'toself',
    18: 'toself',
    19: 'toself'
}

color_by_category = {
                WaymoTrackCategory.SDC:              THEMECOLORS["green"],
                WaymoTrackCategory.TRACK_TO_PREDICT: THEMECOLORS["blue"],
                WaymoTrackCategory.UNSCORED:         THEMECOLORS["white"],
            }

visualize_trajectory_for_tracks = [WaymoTrackCategory.SDC,
                                   WaymoTrackCategory.TRACK_TO_PREDICT]

class WaymoFigureCreator(FigureCreatorBaseClass):
    def __init__(self, dataset_part=DatasetPart.VAL, show_trajectory=False, show_legend=False):
        self.scenario = None
        self.static_map = None
        self.dataset_part = dataset_part
        self.show_trajectory = show_trajectory
        self.trajectories_trace_indices = [0, 0]
        self.show_legend = show_legend
        self.scale_variant = 2

        self.data_dir_path = {DatasetPart.TRAIN: 'data/argoverse/train/',
                              DatasetPart.VAL:   'data/argoverse/val/',
                              DatasetPart.TEST:  'data/argoverse/test/'}

        self.static_data = []
        self.subdirs = {
            dataset_part: [path for path in Path(data_dir).iterdir() if path.is_dir()]
            for (dataset_part, data_dir) in self.data_dir_path.items()
        }

        self.number_of_scenes = len(self.subdirs[dataset_part])

        self.significant_scale_range_x = []
        self.significant_scale_range_y = []

        self.current_scene_id = 1
        self.current_scene = self.generate_figure(self.current_scene_id)

        self.cached_scene = {dataset_part: self.current_scene}
        self.cached_scene_id = {dataset_part: self.current_scene_id}

        # TODO: Move shared to the base class



    def make_static_data(self) -> List:
        print('Here we are')
        roadgraph_xyz = self.decoded_example['roadgraph_samples/xyz'].numpy()
        roadgraph_dir = self.decoded_example['roadgraph_samples/dir'].numpy()
        roadgraph_type = self.decoded_example['roadgraph_samples/type'].numpy()

        newxyz, newtype = self.simplify_road_graph(roadgraph_xyz, roadgraph_type, roadgraph_dir)
        # newxyz, newtype = roadgraph_xyz, roadgraph_type


        # for rtype in [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]:
        for rtype in [17]:
            this_type_xyz = newxyz[np.where(newtype == rtype)[0], :]
            if this_type_xyz.shape[0] == 0:
                # no contours of this type
                continue

            if rtype == 18 or rtype == 19:
                # crosswalk or speed bump
                if this_type_xyz.shape[0] % 4 != 0:
                    print("Couldn't separate crosswalks / speed bumps by four points")
                else:
                    obstacles = np.split(this_type_xyz, this_type_xyz.shape[0] // 4)
                    ob_xyz = obstacles[0]
                    for obidx, ob in enumerate(obstacles):
                        ob_xyz = np.vstack((
                            ob_xyz,
                            ob_xyz[-4, np.newaxis, :],
                            np.array([None, None, None]),
                        ))
                        if obidx < len(obstacles)-1:
                            ob_xyz = np.vstack((
                                ob_xyz,
                                obstacles[obidx+1],
                            ))
                this_type_xyz = ob_xyz

            print("type: ", rtype, "\t  amount: ", this_type_xyz.shape[0])
            xydata = none_vector
            if rtype == 17:
                stop_contour = scale_object(length=1, width=1, object_type=5).T
                for j in range(this_type_xyz.shape[0]):
                    xydata = np.vstack((xydata, np.add(stop_contour, this_type_xyz[j, :2])))


            else:
                xdata = this_type_xyz[:, 0]
                ydata = this_type_xyz[:, 1]

            this_type_trace = {
                "x": xdata.tolist(),
                "y": ydata.tolist(),
                "line": {
                    "width": 1,
                    "color": ROADCOLORBYTYPE[rtype],
                    "dash": ROADDASHBYTYPE[rtype],
                },
                "hoverinfo": 'none',
                "mode": 'lines',
                "fill": ROADFILLBYTYPE[rtype],
                "showlegend": False
            }
            self.static_data.append(this_type_trace)

        return self.static_data

    def make_dynamic_data(self, static_data: List) -> List:

        # object states
        past_state_x = self.decoded_example['state/past/x'].numpy()
        past_state_y = self.decoded_example['state/past/y'].numpy()
        past_state_yaw = self.decoded_example['state/past/bbox_yaw'].numpy()
        past_state_length = self.decoded_example['state/past/length'].numpy()
        past_state_width = self.decoded_example['state/past/width'].numpy()

        current_state_x = self.decoded_example['state/current/x'].numpy()
        current_state_y = self.decoded_example['state/current/y'].numpy()
        current_state_yaw = self.decoded_example['state/current/bbox_yaw'].numpy()
        current_state_length = self.decoded_example['state/current/length'].numpy()
        current_state_width = self.decoded_example['state/current/width'].numpy()

        future_state_x = self.decoded_example['state/future/x'].numpy()
        future_state_y = self.decoded_example['state/future/y'].numpy()
        future_state_yaw = self.decoded_example['state/future/bbox_yaw'].numpy()
        future_state_length = self.decoded_example['state/future/length'].numpy()
        future_state_width = self.decoded_example['state/future/width'].numpy()

        tracks_to_predict = self.decoded_example['state/tracks_to_predict'].numpy()
        is_sdc = self.decoded_example['state/is_sdc'].numpy()
        track_type = self.decoded_example['state/type'].numpy()

        all_state_x = np.hstack((past_state_x, current_state_x, future_state_x))
        all_state_y = np.hstack((past_state_y, current_state_y, future_state_y))
        all_state_yaw = np.hstack((past_state_yaw, current_state_yaw, future_state_yaw))
        all_state_length = np.hstack((past_state_length, current_state_length, future_state_length))
        all_state_width = np.hstack((past_state_width, current_state_width, future_state_width))

        states = np.concatenate((all_state_x[:, :, np.newaxis],
                                 all_state_y[:, :, np.newaxis],
                                 all_state_yaw[:, :, np.newaxis],
                                 all_state_length[:, :, np.newaxis],
                                 all_state_width[:, :, np.newaxis],), axis=2)

        del past_state_x, past_state_y, past_state_yaw, past_state_length, past_state_width
        del current_state_x, current_state_y, current_state_yaw, current_state_length, current_state_width
        del future_state_x, future_state_y, future_state_yaw, future_state_length, future_state_width

        # traffic light
        tl_past_state = self.decoded_example['traffic_light_state/past/state'].numpy()
        tl_past_x = self.decoded_example['traffic_light_state/past/x'].numpy()
        tl_past_y = self.decoded_example['traffic_light_state/past/y'].numpy()

        tl_current_state = self.decoded_example['traffic_light_state/current/state'].numpy()
        tl_current_x = self.decoded_example['traffic_light_state/current/x'].numpy()
        tl_current_y = self.decoded_example['traffic_light_state/current/y'].numpy()

        tl_future_state = self.decoded_example['traffic_light_state/future/state'].numpy()
        tl_future_x = self.decoded_example['traffic_light_state/future/x'].numpy()
        tl_future_y = self.decoded_example['traffic_light_state/future/y'].numpy()

        all_tl_state = np.vstack((tl_past_state, tl_current_state, tl_future_state)).T
        all_tl_x = np.vstack((tl_past_x, tl_current_x, tl_future_x)).T
        all_tl_y = np.vstack((tl_past_y, tl_current_y, tl_future_y)).T

        max_timestamp = states.shape[1]
        number_of_objects = is_sdc[is_sdc >= 0].shape[0]
        number_of_traffic_lights = all_tl_state.shape[0]

        track_category = [None] * number_of_objects
        for track_idx in range(number_of_objects):
            if is_sdc[track_idx] == 1:
                track_category[track_idx] = WaymoTrackCategory.SDC
            elif tracks_to_predict[track_idx] == 1:
                track_category[track_idx] = WaymoTrackCategory.TRACK_TO_PREDICT
            else:
                track_category[track_idx] = WaymoTrackCategory.UNSCORED

        frames = []
        for ts in range(max_timestamp):

            # objects
            objects_data = []
            for track_idx in range(number_of_objects):
                current_state = states[track_idx, ts, :]
                if current_state[0] == -1:
                    xdata, ydata = [], []
                else:
                    object_contour = scale_object(current_state[3], current_state[4], track_type[track_idx])
                    if track_type[track_idx] != 2:
                        # not pedestrian
                        rot_m = np.array([[np.cos(current_state[2]), -np.sin(current_state[2])],
                                          [np.sin(current_state[2]), np.cos(current_state[2])]])
                        object_contour = rot_m @ object_contour
                    xdata = object_contour[0, :] + current_state[0]
                    ydata = object_contour[1, :] + current_state[1]

                object_trace = {
                    "x": xdata,
                    "y": ydata,
                    "line": {
                        "width": 1,
                        "color": color_by_category[track_category[track_idx]],
                    },
                    "hoverinfo": 'none',
                    "mode": 'lines',
                    "fill": 'toself',
                    "showlegend": False
                }
                objects_data.append(object_trace)

            # traffic lights
            tl_now = all_tl_state[:, ts]

            traffic_lights_trace = {
                "x": all_tl_x[tl_now > 0, ts],
                "y": all_tl_y[tl_now > 0, ts],
                "line": {
                    "width": 1,
                    "color": THEMECOLORS["red"],
                },
                "hoverinfo": 'none',
                "mode": 'markers',
                "fill": 'none',
                "showlegend": False
            }

            # for tl_idx in range(number_of_traffic_lights):

            frame = {"data": [*self.static_data, *objects_data, traffic_lights_trace],
                     "name": str(ts)}
            frames.append(frame)
        return frames

    def read_scene_data(self, scene_id):
        FILENAME = '/Users/antontmur/projects/mfplot/data/waymo/val/uncompressed-tf_example-validation-validation_tfexample.tfrecord-00000-of-00150'
        FILENAME = '/Users/antontmur/projects/mfplot/data/waymo/train/uncompressed-tf_example-training-training_tfexample.tfrecord-00000-of-01000'
        # FILENAME = '/Users/antontmur/projects/mfplot/data/waymo/train/uncompressed-tf_example-training-training_tfexample.tfrecord-00001-of-01000'

        # Example field definition
        roadgraph_features = {
            'roadgraph_samples/dir':
                tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
            'roadgraph_samples/id':
                tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
            'roadgraph_samples/type':
                tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
            'roadgraph_samples/valid':
                tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
            'roadgraph_samples/xyz':
                tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
        }

        # Features of other agents.
        state_features = {
            'state/id':
                tf.io.FixedLenFeature([128], tf.float32, default_value=None),
            'state/type':
                tf.io.FixedLenFeature([128], tf.float32, default_value=None),
            'state/is_sdc':
                tf.io.FixedLenFeature([128], tf.int64, default_value=None),
            'state/tracks_to_predict':
                tf.io.FixedLenFeature([128], tf.int64, default_value=None),
            'state/current/bbox_yaw':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/height':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/length':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/timestamp_micros':
                tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
            'state/current/valid':
                tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
            'state/current/vel_yaw':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/velocity_x':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/velocity_y':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/width':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/x':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/y':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/z':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/future/bbox_yaw':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/height':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/length':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/timestamp_micros':
                tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
            'state/future/valid':
                tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
            'state/future/vel_yaw':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/velocity_x':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/velocity_y':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/width':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/x':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/y':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/z':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/past/bbox_yaw':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/height':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/length':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/timestamp_micros':
                tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
            'state/past/valid':
                tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
            'state/past/vel_yaw':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/velocity_x':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/velocity_y':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/width':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/x':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/y':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/z':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        }

        traffic_light_features = {
            'traffic_light_state/current/state':
                tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
            'traffic_light_state/current/valid':
                tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
            'traffic_light_state/current/x':
                tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
            'traffic_light_state/current/y':
                tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
            'traffic_light_state/current/z':
                tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
            'traffic_light_state/past/state':
                tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
            'traffic_light_state/past/valid':
                tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
            'traffic_light_state/past/x':
                tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
            'traffic_light_state/past/y':
                tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
            'traffic_light_state/past/z':
                tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
            'traffic_light_state/future/state':
                tf.io.FixedLenFeature([80, 16], tf.int64, default_value=None),
            'traffic_light_state/future/x':
                tf.io.FixedLenFeature([80, 16], tf.float32, default_value=None),
            'traffic_light_state/future/y':
                tf.io.FixedLenFeature([80, 16], tf.float32, default_value=None),
        }

        features_description = {}
        features_description.update(roadgraph_features)
        features_description.update(state_features)
        features_description.update(traffic_light_features)

        dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
        data = next(dataset.as_numpy_iterator())
        parsed = tf.io.parse_single_example(data, features_description)

        self.decoded_example = parsed
        return None

    @staticmethod
    def simplify_road_graph(rxyz, rtype, rdir):

        indices = np.where(rtype > 0)[0]
        print("Input length: ", indices.shape)

        newxyz, newtype = rxyz[0, np.newaxis, :], rtype[0, np.newaxis, :]
        current_added = True
        for j in range(indices.shape[0] - 1):
            current, prev = indices[j + 1], indices[j]
            if (rtype[current] == rtype[prev] and
               np.linalg.norm(rdir[current, :2] - rdir[prev, :2]) < 0.0001):
                current_added = False
                continue

            if np.linalg.norm(rxyz[current, :] - rxyz[prev, :]) > 2 and rtype[current] < 17:
                newxyz = np.vstack((
                    newxyz,
                    np.array([[None, None, None]])
                ))
                newtype = np.vstack((
                    newtype,
                    rtype[current, np.newaxis, :]
                ))

            newxyz = np.vstack((
                newxyz,
                rxyz[current, np.newaxis, :]
            ))
            newtype = np.vstack((
                newtype,
                rtype[current, np.newaxis, :]
            ))
            current_added = True

        # adding the last one if not added
        if not current_added:
            newxyz = np.vstack((
                newxyz,
                rxyz[indices[-1], np.newaxis, :]
            ))
            newtype = np.vstack((
                newtype,
                rtype[indices[-1], np.newaxis, :]
            ))

        print("Output length: ", len(newtype))

        return newxyz, newtype



