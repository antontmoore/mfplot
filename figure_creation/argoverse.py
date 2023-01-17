from dash_bootstrap_templates import load_figure_template
import plotly.graph_objects as go
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ObjectType, TrackCategory
from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path
import numpy as np

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
    'white': '#d0d0d0',
    'black': '#101010',
}
load_figure_template(['darkly'])
template = 'darkly'

MASK_VALUE = -100500
vehicle_contour = np.array([[-2.0, 1.7, 2.0,  2.0,  1.7, -2.0, -2.0],
                            [0.8,  0.8, 0.5, -0.5, -0.8, -0.8,  0.8]])

bus_contour = np.array([[-6.0, 6.0,  6.0, -6.0, -6.0],
                        [1.3,  1.3, -1.3, -1.3,  1.3]])

ped_contour = np.array([[4.8, 7.7, 9.3, 9.3, 8.7, 8.4, 8.1, 8.1, 8.3, 8.5, 8.9, 9.5, 10.1, 10.7, 11.1, 11.3, 11.5, 11.5, 11.3, 11.2, 11., 10.5, 10., 9.3, 9.3, 9.7, 11.3, 13.9, 13.4, 10.4, 9.8, 9.3, 13.1, 11.7,  8.5,  7.1,  5.1,  3.9,  5.6,  6.1, 7.5, 6.1, 5.6, 4.3, 4.8],
                        [6.2, 3.8, 3.8, 3.3, 3.0, 2.6, 1.9, 1.4, 0.9, 0.6, 0.2, 0.0,  0.0,  0.2,  0.6,  0.9,  1.4,  1.9,  2.3,  2.6, 2.8,  3.2, 3.3, 3.3, 3.8, 4.0,  7.2,  8.6,  9.7,  8.2, 7.2, 9.5, 15.8, 16.7, 11.6, 14.1, 16.6, 15.9, 13.2, 12.2, 6.0, 7.0, 9.9, 9.8, 6.2]])
ped_contour[1, :] = -ped_contour[1, :]
ped_contour = np.multiply(ped_contour, 1/16.7*1.5)

moto_contour = np.array([[-5.0, -3.0, -2.0, 2.0, 3.0, 5.0,  5.0,  3.0,  2.0, -2.0, -3.0, -5.0, -5.0],
                         [ 0.5,  0.5,  2.0, 2.0, 0.5, 0.5, -0.5, -0.5, -2.0, -2.0, -0.5, -0.5,  0.5]])
moto_contour = np.multiply(0.4, moto_contour)

other_contour = np.array([[-6.0, -3.0, 3.0, 6.0,  3.0, -3.0, -6.0],
                          [ 0.0,  5.0, 5.0, 0.0, -5.0, -5.0,  0.0]])
other_contour = np.multiply(0.2, other_contour)

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

none_vector = np.array([None, None], ndmin=2)

track_category_text_by_category = \
    {
        tc: ' '.join(str(tc).split('.')[1].lower().split('_'))
        for tc in [TrackCategory.TRACK_FRAGMENT,
                   TrackCategory.UNSCORED_TRACK,
                   TrackCategory.SCORED_TRACK,
                   TrackCategory.FOCAL_TRACK]
    }

SIGNIFICANT_SCALE_DELTA = 20.

class ArgoverseFigureCreator:

    def __init__(self, show_trajectory=False, show_legend=False):
        self.scenario = None
        self.static_map = None
        self.show_trajectory = show_trajectory
        self.trajectories_trace_indices = [0, 0]
        self.show_legend = show_legend
        self.scale_variant = 2

        self.data_dir_path = 'data/val/'
        subdirs = [path for path in Path(self.data_dir_path).iterdir() if path.is_dir()]

        self.number_of_scenes = len(subdirs)
        self.dirname_by_id = dict(zip(
            range(1, self.number_of_scenes+1),
            [subdir.name for subdir in subdirs]
        ))

        self.significant_scale_range_x = []
        self.significant_scale_range_y = []

        self.current_scene_id = 1
        self.current_scene = self.generate_figure(
            self.dirname_by_id[self.current_scene_id]
        )

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
        max_timestamp = max([object_state.timestep for track in self.scenario.tracks for object_state in track.object_states])
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

            track_colors.append(
                color_by_category[track.category]
                    if track.object_type is not ObjectType.PEDESTRIAN
                    else THEMECOLORS["magenta"]
            )
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
                        rotated_vehicle = np.add(rot_m @ vehicle_contour, trace[-1, :, np.newaxis])
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

                    if track.category in visualize_trajectory_for_tracks:

                        vehicle_trajectory = {
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
                        trajectories.append(vehicle_trajectory)

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
                         "text": str(track_type)[11:].lower() + ' (' \
                                 + track_category_text_by_category[track.category] + ")",
                         "fill": 'toself',
                         "showlegend": self.show_legend,
                         "name": str(track_type).lower()[11:] + " " + str(track_idx),
                         }
                    )

            frame = {"data": [*static_data, *vehicles_data, *others_data, *trajectories],
                     "name": str(ts)}

            first_traj_idx = len(static_data) + len(vehicles_data) + len(others_data)
            self.trajectories_trace_indices = [first_traj_idx, first_traj_idx+len(trajectories)]
            frames.append(frame)

        return frames

    def generate_figure(self, scene_id):

        scene_path = self.data_dir_path + scene_id
        static_map_path = scene_path + f"/log_map_archive_{scene_id}.json"
        scenario_path = scene_path + f"/scenario_{scene_id}.parquet"


        self.scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        self.static_map = ArgoverseStaticMap.from_json(Path(static_map_path))

        fig_dict = {
            "data": [],
            "layout": {},
            "frames": []
        }


        static_plot_data = self.make_static_data()

        sliders_dict = {
            "active": 0,
            "activebgcolor": THEMECOLORS['light-blue'],
            "bordercolor": THEMECOLORS['medium-grey'],
            "bgcolor": THEMECOLORS['blue'],
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "visible": False,
            },
            "transition": {"duration": 0, "easing": "linear"},
            "pad": {"b": 10, "t": 10},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": []
        }
        frames = self.make_dynamic_data(static_data=static_plot_data)

        fig_dict["data"] = frames[0]["data"]
        fig_dict["frames"] = frames
        for frame_num in range(len(frames)):
            slider_step = {"args": [
                [frame_num],
                {"frame": {"duration": 0, "redraw": True},
                 "mode": "immediate",
                 "transition": {"duration": 0}}
            ],
                "label": frame_num,
                "method": "animate"}
            sliders_dict["steps"].append(slider_step)

        fig_dict["layout"]["updatemenus"] = [
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 0,
                                                  "redraw": True},
                                        "fromcurrent": False,
                                        "transition": {"duration": 0,
                                                       "easing": "quadratic-in-out"}
                                        }],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
                "bordercolor": THEMECOLORS['blue'],
                "bgcolor": THEMECOLORS['dark-grey'],
            }
        ]
        fig_dict["layout"]["sliders"] = [sliders_dict]

        fig_dict["layout"]["xaxis"] = {"showgrid": False, "zeroline": False, "showticklabels": False}

        fig_dict["layout"]["yaxis"] = {"showgrid": False, "zeroline": False, "showticklabels": False,
                                       "scaleanchor": "x", "scaleratio": 1}
        if self.scale_variant == 2:
            fig_dict["layout"]["xaxis"]["range"] = self.significant_scale_range_x
            fig_dict["layout"]["yaxis"]["range"] = self.significant_scale_range_y
        fig_dict["layout"]["margin"] = dict(l=0, r=0, b=0, t=0)

        fig = go.Figure(fig_dict)
        return fig

    def get_current_scene(self):
        return self.current_scene

    def get_next_scene(self):

        if self.current_scene_id == self.number_of_scenes:
            print(f"Try to get the scene {self.current_figure_id+1}, but we have only {self.number_of_scenes} scenes.")
        else:
            self.current_scene_id += 1
            self.current_scene = self.generate_figure(
                self.dirname_by_id[self.current_scene_id]
            )

        return self.current_scene, self.current_scene_id

    def get_previous_scene(self):

        if self.current_scene_id == 1:
            print(f"Try to get the scene 0, but we start from 1.")
        else:
            self.current_scene_id -= 1
            self.current_scene = self.generate_figure(
                self.dirname_by_id[self.current_scene_id]
            )

        return self.current_scene, self.current_scene_id

    def get_scene_by_id(self, scene_id):
        if scene_id <= 0:
            print(f"Try to get the scene 0, but we start from 1.")
        elif scene_id > self.number_of_scenes:
            print(f"Try to get the scene {scene_id}, but we have only {self.number_of_scenes} scenes.")
        else:
            self.current_scene_id = scene_id
            self.current_scene = self.generate_figure(
                self.dirname_by_id[scene_id]
            )
        return self.current_scene, self.current_scene_id

    def change_visibility_of_trajectories(self, new_visibility_value):
        print("changing visibility of trajectories")
        print(f"Before: {self.show_trajectory}")
        fig = self.current_scene
        if not (new_visibility_value == self.show_trajectory):
            for trace_idx in range(self.trajectories_trace_indices[0], self.trajectories_trace_indices[1]):
                fig.data[trace_idx]["visible"] = not self.show_trajectory

                for frame in fig.frames:
                    frame.data[trace_idx]["visible"] = not self.show_trajectory

            self.show_trajectory = not self.show_trajectory
        print(f"After: {self.show_trajectory}")
        return self.current_scene, self.current_scene_id

    def change_scene_scale(self, scale_variant):
        fig = self.current_scene
        if scale_variant == 1:                # as is
            fig.update_layout(yaxis={"autorange": True, "fixedrange": False})
        elif scale_variant == 2:              # significant
            fig.layout.xaxis.range = self.significant_scale_range_x
            fig.layout.yaxis.range = self.significant_scale_range_y
            fig.update_layout(yaxis={"autorange": False, "fixedrange": True})
        self.scale_variant = scale_variant
        return fig, self.current_scene_id


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
