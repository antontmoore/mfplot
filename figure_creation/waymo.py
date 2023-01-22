from .figure_creator import FigureCreatorBaseClass
from typing import List
from .shared import THEMECOLORS
from .shared import DatasetPart
from .shared import none_vector
from pathlib import Path
# import math
# import os
# import uuid
# import time
#
# from matplotlib import cm
# import matplotlib.animation as animation
# import matplotlib.pyplot as plt
#
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
    17: THEMECOLORS['red'],
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

ROADFLLBYTYPE = {
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



    def make_static_data(self) -> List:
        print('Here we are')
        roadgraph_xyz = self.decoded_example['roadgraph_samples/xyz'].numpy()
        roadgraph_dir = self.decoded_example['roadgraph_samples/dir'].numpy()
        roadgraph_type = self.decoded_example['roadgraph_samples/type'].numpy()

        newxyz, newtype = self.simplify_road_graph(roadgraph_xyz, roadgraph_type, roadgraph_dir)
        # newxyz, newtype = roadgraph_xyz, roadgraph_type


        for rtype in [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]:
        # for rtype in [19]:
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
            this_type_trace = {
                "x": this_type_xyz[:, 0].tolist(),
                "y": this_type_xyz[:, 1].tolist(),
                "line": {
                    "width": 1,
                    "color": ROADCOLORBYTYPE[rtype],
                    "dash": ROADDASHBYTYPE[rtype],
                },
                "hoverinfo": 'none',
                "mode": 'lines',
                "fill": ROADFLLBYTYPE[rtype],
                "showlegend": False
            }
            self.static_data.append(this_type_trace)
        return self.static_data

    def make_dynamic_data(self, static_data: List) -> List:
        frames = []
        for ts in range(3):
            frame = {"data": self.static_data,
                     "name": str(ts)}
            frames.append(frame)
        return frames

    def read_scene_data(self, scene_id):
        FILENAME = '/Users/antontmur/projects/mfplot/data/waymo/val/uncompressed-tf_example-validation-validation_tfexample.tfrecord-00000-of-00150'

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







def visualize_all_agents_smooth(
    decoded_example,
    size_pixels=1000,
):
  """Visualizes all agent predicted trajectories in a serie of images.

  Args:
    decoded_example: Dictionary containing agent info about all modeled agents.
    size_pixels: The size in pixels of the output image.

  Returns:
    T of [H, W, 3] uint8 np.arrays of the drawn matplotlib's figure canvas.
  """
  # [num_agents, num_past_steps, 2] float32.
  past_states = tf.stack(
      [decoded_example['state/past/x'], decoded_example['state/past/y']],
      -1).numpy()
  past_states_mask = decoded_example['state/past/valid'].numpy() > 0.0

  # [num_agents, 1, 2] float32.
  current_states = tf.stack(
      [decoded_example['state/current/x'], decoded_example['state/current/y']],
      -1).numpy()
  current_states_mask = decoded_example['state/current/valid'].numpy() > 0.0

  # [num_agents, num_future_steps, 2] float32.
  future_states = tf.stack(
      [decoded_example['state/future/x'], decoded_example['state/future/y']],
      -1).numpy()
  future_states_mask = decoded_example['state/future/valid'].numpy() > 0.0

  # [num_points, 3] float32.
  roadgraph_xyz = decoded_example['roadgraph_samples/xyz'].numpy()

  num_agents, num_past_steps, _ = past_states.shape
  num_future_steps = future_states.shape[1]

  # color_map = get_colormap(num_agents)
  #
  # # [num_agens, num_past_steps + 1 + num_future_steps, depth] float32.
  # all_states = np.concatenate([past_states, current_states, future_states], 1)
  #
  # # [num_agens, num_past_steps + 1 + num_future_steps] float32.
  # all_states_mask = np.concatenate(
  #     [past_states_mask, current_states_mask, future_states_mask], 1)
  #
  # center_y, center_x, width = get_viewport(all_states, all_states_mask)
  #
  # images = []
  #
  # # Generate images from past time steps.
  # for i, (s, m) in enumerate(
  #     zip(
  #         np.split(past_states, num_past_steps, 1),
  #         np.split(past_states_mask, num_past_steps, 1))):
  #   im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
  #                           'past: %d' % (num_past_steps - i), center_y,
  #                           center_x, width, color_map, size_pixels)
  #   images.append(im)
  #
  # # Generate one image for the current time step.
  # s = current_states
  # m = current_states_mask
  #
  # im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz, 'current', center_y,
  #                         center_x, width, color_map, size_pixels)
  # images.append(im)
  #
  # # Generate images from future time steps.
  # for i, (s, m) in enumerate(
  #     zip(
  #         np.split(future_states, num_future_steps, 1),
  #         np.split(future_states_mask, num_future_steps, 1))):
  #   im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
  #                           'future: %d' % (i + 1), center_y, center_x, width,
  #                           color_map, size_pixels)
  #   images.append(im)
  #
  # return images
  return None

# images = visualize_all_agents_smooth(parsed)
ay = 1


