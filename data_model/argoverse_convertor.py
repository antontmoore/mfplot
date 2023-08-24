import numpy as np
from pathlib import Path
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ObjectType, TrackCategory
from av2.map.map_api import ArgoverseStaticMap
from .scene import Scene
from .scene import Tracks

MASK_VALUE = -100500


class ArgoverseConvertor:
    def __init__(self,
                 simplify_graph=True,
                 only_for_calc=False,
                 add_z_coordinate=False,
                 plot_result=False,
                 ):

        self.simplify_graph = simplify_graph
        self.only_for_calc = only_for_calc
        self.add_z_coordinate = add_z_coordinate
        self.plot_result = plot_result

        self.coords_slice = slice(0, 3) if self.add_z_coordinate else slice(0, 2)
        self.coords_slice_tuple = (0, 3) if self.add_z_coordinate else (0, 2)
        self.none_vector = np.array([None, None, None]) if self.add_z_coordinate else np.array([None, None])

        self.static_map = None
        self.scenario = None

    def read_scene_data(self, scene_path):
        scene_hash = scene_path.split('/')[-1]
        static_map_path = scene_path + f"/log_map_archive_{scene_hash}.json"
        scenario_path = scene_path + f"/scenario_{scene_hash}.parquet"
        self.static_map = ArgoverseStaticMap.from_json(Path(static_map_path))
        self.scenario = scenario_serialization.load_argoverse_scenario_parquet(
            Path(scenario_path))

    def convert(self):

        scene = Scene()
        scene.road_border = self._get_road_border()
        scene.lanes_borderline, scene.lanes_centerline = self._get_lanes()
        scene.crosswalk = self._get_crosswalks()
        scene.tracks = self._get_tracks()

        return scene

    def _get_road_border(self):
        border = self.none_vector
        for drive_area in self.static_map.vector_drivable_areas.values():
            border = np.vstack((border,
                                drive_area.xyz[:, self.coords_slice],
                                self.none_vector))
        border = border[1:-1, :].astype(float)
        return border

    def _get_lanes(self):
        # already_added_points = set()
        lanes_borderline = self.none_vector
        for lane_segment in self.static_map.vector_lane_segments.values():
            points = np.vstack((
                lane_segment.left_lane_boundary.xyz[:, self.coords_slice],
                self.none_vector,
                lane_segment.right_lane_boundary.xyz[:, self.coords_slice],
            ))

            lanes_borderline = np.vstack((lanes_borderline, points))
            # for j in range(points.shape[0]):
            #     point = points[j, :]
            #     tuple_point = tuple(point)
            #     if tuple_point not in already_added_points or \
            #        np.all(point == none_vector):
            #         lanes_borderline = np.vstack((lanes_borderline, point))
            #         already_added_points.add(tuple_point)

            lanes_borderline = np.vstack((lanes_borderline, self.none_vector))
        lanes_borderline = lanes_borderline[1:].astype(float)
        return lanes_borderline, None

    def _get_crosswalks(self):
        crosswalks = self.none_vector
        for ped_crossing in self.static_map.vector_pedestrian_crossings.values():
            crosswalks = np.vstack((
                crosswalks,
                ped_crossing.edge1.xyz[:, self.coords_slice],
                ped_crossing.edge2.xyz[:, self.coords_slice]
            ))
        crosswalks = crosswalks[1:, :]
        return crosswalks

    def _get_tracks(self):
        # features: position(x, y), velocity(x, y), yaw, length, width
        # shape = (num_of_tracks, timesteps, feature_dimension)

        object_type_map = {
            ObjectType.VEHICLE: 1,              # vehicle
            ObjectType.PEDESTRIAN: 2,           # pedestrian
            ObjectType.MOTORCYCLIST: 3,         # cyclist/bicycle/motorcyclist
            ObjectType.CYCLIST: 3,              # cyclist/bicycle/motorcyclist
            ObjectType.RIDERLESS_BICYCLE: 5,    # other
            ObjectType.BUS: 4,                  # bus
            ObjectType.STATIC: 5,               # other
            ObjectType.BACKGROUND: 5,           # other
            ObjectType.CONSTRUCTION: 5,         # other
            ObjectType.UNKNOWN: 5,              # other
        }
        track_category_map = {
            TrackCategory.FOCAL_TRACK: 1,
            TrackCategory.SCORED_TRACK: 2,
            TrackCategory.UNSCORED_TRACK: 3,
            TrackCategory.TRACK_FRAGMENT: 4,
        }
        default_dims_by_type = {
            1: np.array([4.0, 1.8], ndmin=2),
            2: np.array([1.0, 1.0], ndmin=2),
            3: np.array([2.5, 1.1], ndmin=2),
            4: np.array([12., 2.6], ndmin=2),
            5: np.array([2.0, 2.0], ndmin=2),
        }

        max_timestamp = max([object_state.timestep
                             for track in self.scenario.tracks for object_state in track.object_states])
        number_of_tracks = len(self.scenario.tracks)

        type_and_category = np.empty_like(self.none_vector)
        for track_idx, track in enumerate(self.scenario.tracks):
            track_type_and_category = np.array(
                [
                    object_type_map[track.object_type],
                    track_category_map[track.category]
                ]
            )
            type_and_category = np.vstack((type_and_category, track_type_and_category))
        type_and_category = type_and_category[1:, :].astype(int)

        coords = np.ones((number_of_tracks, max_timestamp + 1, 2)) * MASK_VALUE
        velocities = np.ones((number_of_tracks, max_timestamp + 1, 2)) * MASK_VALUE
        valids = np.zeros((number_of_tracks, max_timestamp + 1)).astype(bool)
        yaws = np.ones((number_of_tracks, max_timestamp + 1)) * MASK_VALUE
        dimensions = np.zeros((number_of_tracks, max_timestamp + 1, 2))
        for track_idx, track in enumerate(self.scenario.tracks):
            for obj_state in track.object_states:
                coords[track_idx, obj_state.timestep, :] = obj_state.position
                velocities[track_idx, obj_state.timestep, :] = obj_state.velocity
                yaws[track_idx, obj_state.timestep] = obj_state.heading
                valids[track_idx, obj_state.timestep] = True
            default_dims = default_dims_by_type[type_and_category[track_idx, 0]]
            dimensions[track_idx, :, :] = default_dims

        # features: position(x, y), velocity(x, y), yaw, length, width
        zero_feature = np.zeros((number_of_tracks, max_timestamp + 1, 1))
        if self.add_z_coordinate:
            track_features = np.concatenate((
                coords, zero_feature,  # x, y, z
                velocities, zero_feature,  # vx, vy, vz
                yaws[:, :, np.newaxis],  # yaw
                dimensions  # length, width
            ), axis=2)
        else:
            track_features = np.concatenate((
                coords, zero_feature,  # x, y, z
                velocities, zero_feature,  # vx, vy, vz
                yaws[:, :, np.newaxis],  # yaw
                dimensions  # length, width
            ), axis=2)

        tracks = Tracks()
        tracks.type_and_category = type_and_category
        tracks.features = track_features
        tracks.valid = valids

        return tracks


if __name__ == "__main__":
    ac = ArgoverseConvertor()
    scene_id = '07f32ef1-2a19-4695-b0ea-59fd890ca813'
    data_dir_path = '/Users/antontmur/projects/mfplot/data/argoverse/val/'
    static_data_file = data_dir_path + scene_id + f"/log_map_archive_{scene_id}.json"
    ac.read_scene_data(scene_path=data_dir_path + scene_id)
    scene_converted = ac.convert()

# Now not used, to be used in future:
# 1. lane_segment has it's type: VEHICLE, BUS, BIKE
# 2. lane_segment has predecessors and successors
# 3. lane_segment has left_neighbour_id and right_neighbour_id
# 4. lane_segment boundaries have it's marktypes: DASHED_WHITE, DASHED_YELLOW, etc.
