import numpy as np
import tensorflow as tf
from data_model.scene import Scene, Tracks, RoadMarkup
import warnings
import pickle

NUM_MAP_SAMPLES = 30000
PAST_AND_CURRENT = slice(0, 11)
FUTURE = slice(11, 91)


class WaymoConvertor:
    def __init__(self,
                 simplify_graph=False,
                 only_for_calc=False,
                 add_z_coordinate=False,
                 plot_result=False,
                 ):

        self.simplify_graph = simplify_graph
        self.only_for_calc = only_for_calc
        self.add_z_coordinate = add_z_coordinate
        self.plot_result = plot_result

        self.features_description = self.collect_features_description()

        self.coords_slice = slice(0, 3) if self.add_z_coordinate else slice(0, 2)
        self.coords_slice_tuple = (0, 3) if self.add_z_coordinate else (0, 2)
        self.none_vector = np.array([None, None, None]) if self.add_z_coordinate else np.array([None, None])

        self.parsed_tf_data = None

    @staticmethod
    def collect_features_description():

        # Example field definition
        roadgraph_features = {
            'roadgraph_samples/dir':
                tf.io.FixedLenFeature([NUM_MAP_SAMPLES, 3], tf.float32, default_value=None),
            'roadgraph_samples/id':
                tf.io.FixedLenFeature([NUM_MAP_SAMPLES, 1], tf.int64, default_value=None),
            'roadgraph_samples/type':
                tf.io.FixedLenFeature([NUM_MAP_SAMPLES, 1], tf.int64, default_value=None),
            'roadgraph_samples/valid':
                tf.io.FixedLenFeature([NUM_MAP_SAMPLES, 1], tf.int64, default_value=None),
            'roadgraph_samples/xyz':
                tf.io.FixedLenFeature([NUM_MAP_SAMPLES, 3], tf.float32, default_value=None),
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
            'state/objects_of_interest':
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
        features_description['scenario/id'] = tf.io.FixedLenFeature([1], tf.string, default_value=None)
        return features_description

    def read_and_convert_dataset_part(self, read_from, save_to=None):
        path2filename = str(read_from.absolute())
        dataset = tf.data.TFRecordDataset(path2filename, compression_type='')
        total_number_of_examples = sum(1 for data in dataset)

        example_number = -1
        for data in dataset:
            self.parsed_tf_data = tf.io.parse_single_example(data, self.features_description)
            scene_converted = self.convert()
            example_number += 1
            yield scene_converted, example_number, total_number_of_examples

            # print(f"    scenario_id: \t{scene_converted.scene_id}")
            # if save_to:
            #     path2save = save_to / scene_converted.scene_id
            #     path2save = str(path2save.absolute())
            #     pickle.dump(scene_converted, open(path2save, 'wb'))


    def convert(self):

        (
            lane_center,
            bike_lane_center,
            road_markup,
            border,
            stopsign,
            crosswalk,
            speedbump
        ) = self.get_road_and_lanes()
        scene = Scene()
        scene.lanes_centerline = lane_center
        # scene.lanes_borderline = lane_border
        scene.road_border = border
        scene.road_markup = road_markup
        scene.crosswalk = crosswalk
        scene.tracks = self.get_tracks()
        scene.scene_id = str(self.parsed_tf_data['scenario/id'].numpy()[0])[2:-1]

        # if self.plot_result:
        #     plt.style.use('dark_background')
        #     plt.plot(lane_center[:, 0], lane_center[:, 1], 'g', linewidth=0.5)
        #     plt.plot(border[:, 0], border[:, 1], 'c', linewidth=0.5)
        #     crosswalk_for_plot = self.separate_obstacles(crosswalk)
        #     plt.plot(crosswalk_for_plot[:, 0], crosswalk_for_plot[:, 1], 'm-', linewidth=0.5)
        #     speedbump_for_plot = self.separate_obstacles(speedbump)
        #     plt.plot(speedbump_for_plot[:, 0], speedbump_for_plot[:, 1], 'm-', linewidth=0.5)

        return scene

    def get_road_and_lanes(self):

        data = self.parsed_tf_data
        # roads and lanes
        roadgraph_xyz = data['roadgraph_samples/xyz'].numpy()
        roadgraph_dir = data['roadgraph_samples/dir'].numpy()
        roadgraph_type = data['roadgraph_samples/type'].numpy()
        # roadgraph_id = data['roadgraph_samples/id'].numpy()

        # remove waste -1
        indices = np.where(roadgraph_type > 0)[0]
        roadgraph_xyz = roadgraph_xyz[indices, :]
        roadgraph_dir = roadgraph_dir[indices, :]
        roadgraph_type = roadgraph_type[indices, :]

        if self.simplify_graph:
            roadgraph_xyz, roadgraph_type = self.simplify_road_graph(
                roadgraph_xyz,
                roadgraph_dir,
                roadgraph_type
            )

        coords_slice = slice(0, 3) if self.add_z_coordinate else slice(0, 2)
        coords_slice_tuple = (0, 3) if self.add_z_coordinate else (0, 2)
        start_zeros = np.zeros(coords_slice_tuple)

        border, lane_center = start_zeros, start_zeros
        white_broken_single, white_solid_single, white_solid_double = start_zeros, start_zeros, start_zeros
        yellow_solid_single, yellow_solid_double = start_zeros, start_zeros
        yellow_broken_single, yellow_broken_double, yellow_passing_double = start_zeros, start_zeros, start_zeros

        crosswalk, speedbump, stopsign = np.empty_like(border), np.empty_like(border), np.empty_like(border)
        bike_lane_center, white_markup_broken = np.empty_like(border), np.empty_like(border)

        if self.only_for_calc:
            road_types_to_add = [1, 2, 15, 16, 18, 19]
        else:
            road_types_to_add = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]

        for rtype in road_types_to_add:
            if rtype in [1, 2]:
                # LaneCenter-Freeway = 1, LaneCenter-SurfaceStreet = 2
                lane_center = np.vstack((lane_center,
                                         roadgraph_xyz[np.where(roadgraph_type == rtype)[0], coords_slice]))

            elif rtype == 3:
                # LaneCenter-BikeLane = 3,
                bike_lane_center = roadgraph_xyz[np.where(roadgraph_type == rtype)[0], coords_slice]

            elif rtype == 6:
                # RoadLine-BrokenSingleWhite = 6
                white_broken_single = roadgraph_xyz[np.where(roadgraph_type == rtype)[0], coords_slice]

            elif rtype == 7:
                # RoadLine-SolidSingleWhite = 7
                white_solid_single = roadgraph_xyz[np.where(roadgraph_type == rtype)[0], coords_slice]

            elif rtype == 8:
                # RoadLine - SolidDoubleWhite = 8
                white_solid_double = roadgraph_xyz[np.where(roadgraph_type == rtype)[0], coords_slice]

            elif rtype == 9:
                # RoadLine-BrokenSingleYellow = 9
                yellow_broken_single = roadgraph_xyz[np.where(roadgraph_type == rtype)[0], coords_slice]

            elif rtype == 10:
                # RoadLine-BrokenDoubleYellow = 10
                yellow_broken_double = roadgraph_xyz[np.where(roadgraph_type == rtype)[0], coords_slice]

            elif rtype == 11:
                # Roadline-SolidSingleYellow = 11,  RoadLine-PassingDoubleYellow = 13
                yellow_solid_single = np.vstack((yellow_solid_single,
                                                 roadgraph_xyz[np.where(roadgraph_type == rtype)[0], coords_slice]))

            elif rtype == 12:
                # Roadline - SolidDoubleYellow = 12
                yellow_solid_double = roadgraph_xyz[np.where(roadgraph_type == rtype)[0], coords_slice]

            elif rtype == 13:
                # RoadLine - PassingDoubleYellow = 13
                yellow_passing_double = roadgraph_xyz[np.where(roadgraph_type == rtype)[0], coords_slice]

            elif rtype in [15, 16]:
                # RoadEdgeBoundary = 15, RoadEdgeMedian = 16
                border = np.vstack((border, roadgraph_xyz[np.where(roadgraph_type == rtype)[0], coords_slice]))

            elif rtype == 17:
                # StopSign = 17
                stopsign = roadgraph_xyz[np.where(roadgraph_type == rtype)[0], coords_slice]

            elif rtype == 18:
                # Crosswalk = 18
                crosswalk = roadgraph_xyz[np.where(roadgraph_type == rtype)[0], coords_slice]

            elif rtype == 19:
                # SpeedBump = 19
                speedbump = roadgraph_xyz[np.where(roadgraph_type == rtype)[0], coords_slice]

            else:
                warnings.warn(f'Unknown type of road graph! road_type = {rtype}')

        road_markup = RoadMarkup()
        road_markup.white_broken_single = white_broken_single
        road_markup.white_solid_single = white_solid_single
        road_markup.white_solid_double = white_solid_double
        road_markup.yellow_broken_single = yellow_broken_single
        road_markup.yellow_broken_double = yellow_broken_double
        road_markup.yellow_solid_single = yellow_solid_single
        road_markup.yellow_solid_double = yellow_solid_double
        road_markup.yellow_passing_double = yellow_passing_double


        return (
            lane_center,
            bike_lane_center,
            road_markup,
            border,
            stopsign,
            crosswalk,
            speedbump
        )

    def get_tracks(self):
        data = self.parsed_tf_data
        # track type
        track_type = data['state/type'].numpy().astype(int)
        # 4 -> 5 (because in waymo dataset "unset" is 4)
        track_type[track_type == 4] = 5

        # track category
        # 1 - sdc/focal_track, 2 - tracks_to_predict/scored_track,
        # 3 - objects_of_interest/unscored_track, 4 - other fragment
        tracks_to_predict = data['state/tracks_to_predict'].numpy().astype(bool)
        objects_of_interest = data['state/objects_of_interest'].numpy().astype(bool)
        is_sdc = data['state/is_sdc'].numpy().astype(bool)
        track_category = np.ones_like(track_type, dtype=int) * 4
        track_category[tracks_to_predict] = 2
        track_category[objects_of_interest] = 3
        track_category[is_sdc] = 1

        # track validity
        valid_track = np.hstack((
            data['state/past/valid'].numpy(),
            data['state/current/valid'].numpy(),
            data['state/future/valid'].numpy()
        ))
        have_at_least_one_point = np.sum(valid_track, axis=1) > 0

        # positions, velocities, sizes, yaw
        position_x = np.hstack((
            data['state/past/x'].numpy(),
            data['state/current/x'].numpy(),
            data['state/future/x'].numpy()
        ))
        # position_x = position_x[have_at_least_one_point, :]

        position_y = np.hstack((
            data['state/past/y'].numpy(),
            data['state/current/y'].numpy(),
            data['state/future/y'].numpy()
        ))

        velocity_x = np.hstack((
            data['state/past/velocity_x'].numpy(),
            data['state/current/velocity_x'].numpy(),
            data['state/future/velocity_x'].numpy()
        ))

        velocity_y = np.hstack((
            data['state/past/velocity_y'].numpy(),
            data['state/current/velocity_y'].numpy(),
            data['state/future/velocity_y'].numpy()
        ))

        length = np.hstack((
            data['state/past/length'].numpy(),
            data['state/current/length'].numpy(),
            data['state/future/length'].numpy()
        ))

        width = np.hstack((
            data['state/past/width'].numpy(),
            data['state/current/width'].numpy(),
            data['state/future/width'].numpy()
        ))

        if self.add_z_coordinate:
            position_z = np.hstack((
                data['state/past/z'].numpy(),
                data['state/current/z'].numpy(),
                data['state/future/z'].numpy()
            ))

            height = np.hstack((
                data['state/past/height'].numpy(),
                data['state/current/height'].numpy(),
                data['state/future/height'].numpy()
            ))

        yaw = np.hstack((
            data['state/past/bbox_yaw'].numpy(),
            data['state/current/bbox_yaw'].numpy(),
            data['state/future/bbox_yaw'].numpy()
        ))

        tracks_to_scene = Tracks()
        tracks_to_scene.valid = valid_track[have_at_least_one_point, PAST_AND_CURRENT]
        tracks_to_scene.future_valid = valid_track[have_at_least_one_point, FUTURE]

        # features: position(x, y), velocity(x, y), yaw, length, width
        tracks_to_scene.features = np.concatenate((
            position_x[have_at_least_one_point, PAST_AND_CURRENT, np.newaxis],
            position_y[have_at_least_one_point, PAST_AND_CURRENT, np.newaxis],
            velocity_x[have_at_least_one_point, PAST_AND_CURRENT, np.newaxis],
            velocity_y[have_at_least_one_point, PAST_AND_CURRENT, np.newaxis],
            yaw[have_at_least_one_point, PAST_AND_CURRENT, np.newaxis],
            length[have_at_least_one_point, PAST_AND_CURRENT, np.newaxis],
            width[have_at_least_one_point, PAST_AND_CURRENT, np.newaxis]
        ), axis=2)

        if self.add_z_coordinate:
            tracks_to_scene.future_features = np.concatenate((
                position_x[have_at_least_one_point, FUTURE, np.newaxis],
                position_y[have_at_least_one_point, FUTURE, np.newaxis],
                position_z[have_at_least_one_point, FUTURE, np.newaxis],
                velocity_x[have_at_least_one_point, FUTURE, np.newaxis],
                velocity_y[have_at_least_one_point, FUTURE, np.newaxis],
                yaw[have_at_least_one_point, FUTURE, np.newaxis],
                length[have_at_least_one_point, FUTURE, np.newaxis],
                width[have_at_least_one_point, FUTURE, np.newaxis],
                height[have_at_least_one_point, FUTURE, np.newaxis],
            ), axis=2)
        else:
            tracks_to_scene.future_features = np.concatenate((
                position_x[have_at_least_one_point, FUTURE, np.newaxis],
                position_y[have_at_least_one_point, FUTURE, np.newaxis],
                velocity_x[have_at_least_one_point, FUTURE, np.newaxis],
                velocity_y[have_at_least_one_point, FUTURE, np.newaxis],
                yaw[have_at_least_one_point, FUTURE, np.newaxis],
                length[have_at_least_one_point, FUTURE, np.newaxis],
                width[have_at_least_one_point, FUTURE, np.newaxis]
            ), axis=2)

        tracks_to_scene.type_and_category = np.concatenate((
            track_type[have_at_least_one_point, np.newaxis],
            track_category[have_at_least_one_point, np.newaxis]
        ), axis=1)

        return tracks_to_scene

    def separate_obstacles(self, xyz):
        if xyz.shape[0] == 0:
            return xyz
        none_vector = np.array([None, None, None]) if self.add_z_coordinate else np.array([None, None])
        ob_xyz = none_vector
        if xyz.shape[0] % 4 != 0:
            print("Couldn't separate crosswalks / speed bumps by four points")
        else:
            obstacles = np.split(xyz, xyz.shape[0] // 4)
            ob_xyz = obstacles[0]
            for obidx, ob in enumerate(obstacles):
                ob_xyz = np.vstack((
                    ob_xyz,
                    ob_xyz[-4, np.newaxis, :],
                    none_vector
                ))
                if obidx < len(obstacles) - 1:
                    ob_xyz = np.vstack((
                        ob_xyz,
                        obstacles[obidx + 1],
                    ))
        return ob_xyz

    @staticmethod
    def simplify_road_graph(rxyz, rdir, rtype):

        indices = np.where(rtype > 0)[0]
        # print("Input length: ", indices.shape)

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

        # print("Output length: ", len(newtype))

        return newxyz, newtype


if __name__ == "__main__":
    wc = WaymoConvertor()
    FILENAME = '/Users/antontmur/projects/mfplot/data/waymo/val/' + \
               'uncompressed-tf_example-validation-validation_tfexample.tfrecord-00000-of-00150'
    wc.read_scene_data(FILENAME)
    scene_converted = wc.convert()

    # import pickle
    # path2save = '/Users/antontmur/projects/grumpy/data_model/scene_converted1.pkl'
    # pickle.dump(scene_converted, open(path2save, 'wb'))
    # new_scene = pickle.load(open(path2save, 'rb'))

# Now not used, to be used in future:
# 1. speedbumps are not added in the scene now
