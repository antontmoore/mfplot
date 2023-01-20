from dash_bootstrap_templates import load_figure_template
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ObjectType, TrackCategory
from av2.map.map_api import ArgoverseStaticMap
from .shared import vehicle_contour, ped_contour, moto_contour, bus_contour, other_contour
from .shared import THEMECOLORS
from .shared import DatasetPart
from .shared import none_vector
from .shared import SIGNIFICANT_SCALE_DELTA, MASK_VALUE
from .figure_creator import FigureCreatorBaseClass
from pathlib import Path
import numpy as np

load_figure_template(['darkly'])
template = 'darkly'

contour_by_type = {
                    ObjectType.VEHICLE: vehicle_contour,
                    ObjectType.PEDESTRIAN: ped_contour,
                    ObjectType.MOTORCYCLIST: moto_contour,
                    ObjectType.CYCLIST: moto_contour,
                    ObjectType.RIDERLESS_BICYCLE: moto_contour,
                    ObjectType.BUS: bus_contour,
                    ObjectType.STATIC: other_contour,
                    ObjectType.BACKGROUND: other_contour,
                    ObjectType.CONSTRUCTION: other_contour,
                    ObjectType.UNKNOWN: other_contour,
                }

color_by_category = {
                TrackCategory.FOCAL_TRACK: THEMECOLORS["green"],
                TrackCategory.SCORED_TRACK: THEMECOLORS["blue"],
                TrackCategory.UNSCORED_TRACK: THEMECOLORS["white"],
                TrackCategory.TRACK_FRAGMENT: THEMECOLORS["light-grey"]
            }

visualize_trajectory_for_tracks = [TrackCategory.FOCAL_TRACK,
                                   TrackCategory.SCORED_TRACK,
                                   TrackCategory.UNSCORED_TRACK]

track_category_text_by_category = \
    {
        tc: ' '.join(str(tc).split('.')[1].lower().split('_'))
        for tc in [TrackCategory.TRACK_FRAGMENT,
                   TrackCategory.UNSCORED_TRACK,
                   TrackCategory.SCORED_TRACK,
                   TrackCategory.FOCAL_TRACK]
    }


class ArgoverseFigureCreator(FigureCreatorBaseClass):

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

    def make_static_data(self):

        # generate dicts for drivable area traces
        drivable_area_coords = none_vector
        for drive_area in self.static_map.vector_drivable_areas.values():
            drivable_area_coords = np.vstack((drivable_area_coords,
                                              drive_area.xyz[:, :2],
                                              none_vector))

        drivable_area_traces = {
                "x": drivable_area_coords[:, 0].tolist(),
                "y": drivable_area_coords[:, 1].tolist(),
                "line": {
                    "width": 1,
                    "color": THEMECOLORS['dark-grey']
                },
                "hoverinfo": 'none',
                "mode": 'lines',
                "fill": 'toself',
                "showlegend": False
            }

        # generate dicts for lanes traces
        lanes_x, lanes_y = [], []
        for lane_segment in self.static_map.vector_lane_segments.values():
            lanes_x.extend([
                *list(lane_segment.left_lane_boundary.xyz[:, 0]), None,
                *list(lane_segment.right_lane_boundary.xyz[:, 0]), None
            ])
            lanes_y.extend([
                *list(lane_segment.left_lane_boundary.xyz[:, 1]), None,
                *list(lane_segment.right_lane_boundary.xyz[:, 1]), None
            ])
        lanes_traces = {
            "x": lanes_x,
            "y": lanes_y,
            "line": {
                "width": 0.5,
                "color": THEMECOLORS['light-grey']
            },
            "hoverinfo": 'none',
            "mode": 'lines',
            "showlegend": False,
            "fill": 'none'
        }

        # generate dicts for pedestrian crosses
        # single dict for every crossing because of filling
        ped_cross_traces = []
        for ped_xing in self.static_map.vector_pedestrian_crossings.values():
            ped_cross = np.vstack((
                ped_xing.edge1.xyz[:, :2],
                ped_xing.edge2.xyz[::-1, :2],
                ped_xing.edge1.xyz[0, :2],
            ))
            ped_cross_traces.append(
                {
                    "x": ped_cross[:, 0].tolist(),
                    "y": ped_cross[:, 1].tolist(),
                    "line": {
                        "width": 0,
                        "color": THEMECOLORS['medium-grey']
                    },
                    "hoverinfo": 'none',
                    "mode": 'lines',
                    "showlegend": False,
                    "fill": 'toself'
                }
            )

        return [drivable_area_traces, lanes_traces, *ped_cross_traces]

    def make_dynamic_data(self, static_data):

        # calculating the max of timestamps
        max_timestamp = max([object_state.timestep
                             for track in self.scenario.tracks for object_state in track.object_states])
        number_of_tracks = len(self.scenario.tracks)

        # forming arrays - preparing data
        coords = np.ones((number_of_tracks, max_timestamp + 1, 2)) * MASK_VALUE
        headings = np.zeros((number_of_tracks, max_timestamp + 1))
        track_colors = []
        track_types = []
        track_categories = []
        significant_tracks_ids = []

        for track_idx, track in enumerate(self.scenario.tracks):
            for objs in track.object_states:
                coords[track_idx, objs.timestep, :] = objs.position
                headings[track_idx, objs.timestep] = objs.heading

            if (track.object_type is ObjectType.PEDESTRIAN and
                    track.category not in [TrackCategory.FOCAL_TRACK,
                                           TrackCategory.SCORED_TRACK,
                                           TrackCategory.UNSCORED_TRACK]):
                track_color = THEMECOLORS["magenta"]
            else:
                track_color = color_by_category[track.category]

            track_colors.append(track_color)
            track_types.append(track.object_type)
            track_categories.append(track.category)

            if track.category in [TrackCategory.FOCAL_TRACK, TrackCategory.SCORED_TRACK, TrackCategory.UNSCORED_TRACK]:
                significant_tracks_ids.append(track_idx)

        # calculating scale for significant tracks only
        sign_x_coords = np.ma.masked_less(coords[significant_tracks_ids, :, 0], MASK_VALUE + 1).compressed()
        sign_y_coords = np.ma.masked_less(coords[significant_tracks_ids, :, 1], MASK_VALUE + 1).compressed()
        sign_scale_min_x, sign_scale_max_x = min(sign_x_coords), max(sign_x_coords)
        sign_scale_min_y, sign_scale_max_y = min(sign_y_coords), max(sign_y_coords)

        x_range = sign_scale_max_x - sign_scale_min_x
        y_range = sign_scale_max_y - sign_scale_min_y
        if y_range > x_range:
            x_center = (sign_scale_min_x + sign_scale_max_x) / 2
            self.significant_scale_range_x = [x_center - y_range / 2 - SIGNIFICANT_SCALE_DELTA,
                                              x_center + y_range / 2 + SIGNIFICANT_SCALE_DELTA]
            self.significant_scale_range_y = [sign_scale_min_y - SIGNIFICANT_SCALE_DELTA,
                                              sign_scale_max_y + SIGNIFICANT_SCALE_DELTA]

        else:
            y_center = (sign_scale_min_y + sign_scale_max_y) / 2
            self.significant_scale_range_x = [sign_scale_min_x - SIGNIFICANT_SCALE_DELTA,
                                              sign_scale_max_x + SIGNIFICANT_SCALE_DELTA]
            self.significant_scale_range_y = [y_center - x_range / 2 - SIGNIFICANT_SCALE_DELTA,
                                              y_center + x_range / 2 + SIGNIFICANT_SCALE_DELTA]

        # frames generation
        frames = []

        for ts in range(max_timestamp + 1):

            vehicles_data = []
            others_data = []
            trajectories = []
            for track_idx, track in enumerate(self.scenario.tracks):
                track_color, track_type = track_colors[track_idx], track_types[track_idx]

                # current coordinate and trace calculation
                track_current_coords = coords[track_idx, ts, :]
                track_zero_to_current = coords[track_idx, :ts+1, :]
                track_zero_to_current_masked = np.ma.masked_less(track_zero_to_current, MASK_VALUE + 1)
                trace = track_zero_to_current_masked.compressed().reshape((-1, 2))

                # flag - no points to plot in this track
                empty_track = abs(track_current_coords[0] - MASK_VALUE) < 1

                # contour calculation
                object_contour = contour_by_type[track_type]
                # rotate every object besides pedestrians
                if track_type is not ObjectType.PEDESTRIAN:
                    # rotation matrix
                    heading = headings[track_idx, ts]
                    rot_m = np.array([[np.cos(heading), -np.sin(heading)],
                                      [np.sin(heading), np.cos(heading)]])
                    object_contour = rot_m @ object_contour

                if track_type == ObjectType.VEHICLE:
                    if empty_track:
                        xdata, ydata = [], []
                    else:
                        rotated_vehicle = np.add(object_contour, trace[-1, :, np.newaxis])
                        xdata = rotated_vehicle[0, :].tolist()
                        ydata = rotated_vehicle[1, :].tolist()

                    vehicles_data.append(
                        {
                            "x": xdata,
                            "y": ydata,
                            "line": {
                                "width": 1,
                                "color": track_color
                            },
                            "hoverinfo": 'text',
                            "mode": 'lines',
                            "showlegend": self.show_legend,
                            "fill": 'toself',
                            "text": 'vehicle (' + track_category_text_by_category[track.category] + ")",
                        }
                    )

                else:
                    xdata = [] if empty_track else np.add(track_current_coords[0], object_contour[0, :]).tolist()
                    ydata = [] if empty_track else np.add(track_current_coords[1], object_contour[1, :]).tolist()
                    others_data.append(
                        {"x": xdata,
                         "y": ydata,
                         "hoverinfo": 'text',
                         "mode": 'lines',
                         "line": {
                             "width": 0.5,
                             "color": track_color
                         },
                         "text": str(track_type)[11:].lower() + ' (' +
                            track_category_text_by_category[track.category] + ")",
                         "fill": 'toself',
                         "showlegend": self.show_legend,
                         "name": str(track_type).lower()[11:] + " " + str(track_idx),
                         }
                    )

                if track.category in visualize_trajectory_for_tracks:
                    trajectory = {
                        "x": trace[:, 0],
                        "y": trace[:, 1],
                        "line":
                            {"width": 1,
                             "color": track_color,
                             },
                        "hoverinfo": 'none',
                        "mode": 'lines',
                        "showlegend": False,
                        "fill": 'none',
                        "visible": self.show_trajectory
                    }
                    trajectories.append(trajectory)

            frame = {"data": [*static_data, *vehicles_data, *others_data, *trajectories],
                     "name": str(ts)}

            first_traj_idx = len(static_data) + len(vehicles_data) + len(others_data)
            self.trajectories_trace_indices = [first_traj_idx, first_traj_idx+len(trajectories)]
            frames.append(frame)

        return frames

    def read_scene_data(self, scene_id):
        scene_path = str(
            self.subdirs[self.dataset_part][scene_id - 1]
        )
        scene_hash = scene_path.split('/')[-1]
        static_map_path = scene_path + f"/log_map_archive_{scene_hash}.json"
        scenario_path = scene_path + f"/scenario_{scene_hash}.parquet"

        self.scenario = scenario_serialization.load_argoverse_scenario_parquet(
            Path(scenario_path))
        self.static_map = ArgoverseStaticMap.from_json(Path(static_map_path))
