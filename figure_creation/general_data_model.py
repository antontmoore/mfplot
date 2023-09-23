from dash_bootstrap_templates import load_figure_template
from .shared import THEMECOLORS
from .shared import DatasetPart
from .shared import scale_object
from .shared import SIGNIFICANT_SCALE_DELTA
from .shared import none_vector
from .shared import tl_contour
from .figure_creator import FigureCreatorBaseClass
from pathlib import Path
from plotly.graph_objects import Figure
import numpy as np
import pickle
import os

load_figure_template(['darkly'])
template = 'darkly'

color_by_category = {
                1: THEMECOLORS["green"],
                2: THEMECOLORS["blue"],
                3: THEMECOLORS["white"],
                4: THEMECOLORS["light-grey"]
            }

visualize_trajectory_for_tracks = [1,  # sdc/focal_track
                                   2,  # tracks_to_predict/scored_track
                                   3]  # objects_of_interest/unscored_track

track_category_text_by_category = \
    {
        1: 'sdc/focal_track',
        2: 'tracks_to_predict/scored_track',
        3: 'objects_of_interest/unscored_track',
        4: 'other fragment'
    }

tl_color_by_state = {
    -1: THEMECOLORS['light-grey'],
    0: THEMECOLORS['light-grey'],
    1: THEMECOLORS['red'],
    2: THEMECOLORS['yellow'],
    3: THEMECOLORS['green'],
    4: THEMECOLORS['red'],
    5: THEMECOLORS['yellow'],
    6: THEMECOLORS['green'],
    7: THEMECOLORS['red'],
    8: THEMECOLORS['yellow'],
}


class GeneralFigureCreator(FigureCreatorBaseClass):

    def __init__(self, dataset_part=DatasetPart.TRAIN, show_trajectory=False, show_legend=False):
        self.deserealized_scene = None
        self.dataset_part = dataset_part
        self.show_trajectory = show_trajectory
        self.trajectories_trace_indices = [0, 0]
        self.show_legend = show_legend
        self.scale_variant = 2

        self.data_dir_path = {DatasetPart.TRAIN: 'data/waymo_converted/training/',
                              DatasetPart.VAL:   'data/waymo_converted/validation/',
                              DatasetPart.TEST:  'data/waymo_converted/test/'}

        self.folders_by_datasetpart = {
            dataset_part: [path for path in Path(data_dir).iterdir() if path.is_dir()]
            for (dataset_part, data_dir) in self.data_dir_path.items()
        }

        self.scene_files = os.listdir(self.folders_by_datasetpart[dataset_part][0])

        self.number_of_scenes = len(self.scene_files)

        self.significant_scale_range_x = []
        self.significant_scale_range_y = []

        self.current_scene_id = 0
        self.current_scene = self.generate_figure(self.current_scene_id)

        self.cached_scene = {dataset_part: self.current_scene}
        self.cached_scene_id = {dataset_part: self.current_scene_id}

    def generate_figure(self, scene_id: int) -> Figure:

        self.read_scene_data(scene_id)

        fig_dict = {
            "data": [],
            "layout": {},
            "frames": []
        }

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

        static_plot_data = self.make_static_data()
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
                                                       "easing": "quadratic-in-out"},
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

        fig_dict["layout"]["xaxis"] = {"showgrid": False, "zeroline": False, "showticklabels": True}

        fig_dict["layout"]["yaxis"] = {"showgrid": False, "zeroline": False, "showticklabels": True,
                                       "scaleanchor": "x", "scaleratio": 1}

        # TODO: don't forget
        # if self.scale_variant == 2:
        #     fig_dict["layout"]["xaxis"]["range"] = self.significant_scale_range_x
        #     fig_dict["layout"]["yaxis"]["range"] = self.significant_scale_range_y
        fig_dict["layout"]["margin"] = dict(l=0, r=0, b=0, t=0)

        fig = Figure(fig_dict)
        return fig

    def make_static_data(self):

        def make_one_trace(trace_array, color, dash='solid'):
            return {
                "x": trace_array[:, 0].tolist(),
                "y": trace_array[:, 1].tolist(),
                "line": {
                    "width": 0.5,
                    "color": THEMECOLORS[color],
                    "dash": dash,
                },
                "hoverinfo": 'none',
                "mode": 'lines',
                "showlegend": False,
                "fill": 'none',
            }

        # drivable area traces
        road_border = self.split_trace(self.deserealized_scene.road_border)
        drivable_area_traces = make_one_trace(road_border, 'dark-green')

        # lanes centers
        lanes_centerline = self.split_trace(self.deserealized_scene.lanes_centerline)
        lanes_traces = make_one_trace(lanes_centerline, 'dark-magenta')

        # white markup
        white_broken_single = self.split_trace(self.deserealized_scene.road_markup.white_broken_single)
        white_broken_single_traces = make_one_trace(white_broken_single, 'light-grey', 'dash')

        white_solid_single = self.split_trace(self.deserealized_scene.road_markup.white_solid_single)
        white_solid_single_traces = make_one_trace(white_solid_single, 'light-grey')

        white_solid_double = self.split_trace(self.deserealized_scene.road_markup.white_solid_double)
        white_solid_double_traces = make_one_trace(white_solid_double, 'light-grey')

        # yellow markup
        yellow_broken_single = self.split_trace(self.deserealized_scene.road_markup.yellow_broken_single)
        yellow_broken_single_traces = make_one_trace(yellow_broken_single, 'dark-orange', 'dash')

        yellow_broken_double = self.split_trace(self.deserealized_scene.road_markup.yellow_broken_double)
        yellow_broken_double_traces = make_one_trace(yellow_broken_double, 'dark-orange', 'dash')

        yellow_solid_single = self.split_trace(self.deserealized_scene.road_markup.yellow_solid_single)
        yellow_solid_single_traces = make_one_trace(yellow_solid_single, 'dark-orange')

        yellow_solid_double = self.split_trace(self.deserealized_scene.road_markup.yellow_solid_double)
        yellow_solid_double_traces = make_one_trace(yellow_solid_double, 'dark-orange')

        yellow_passing_double = self.split_trace(self.deserealized_scene.road_markup.yellow_passing_double)
        yellow_passing_double_traces = make_one_trace(yellow_passing_double, 'dark-orange', 'dash')

        # crosswalks
        crosswalk = self.split_trace(self.deserealized_scene.crosswalk)
        ped_cross_traces = {
            "x": crosswalk[:, 0].tolist(),
            "y": crosswalk[:, 1].tolist(),
            "line": {
                "width": 0,
                "color": THEMECOLORS['medium-grey']
            },
            "hoverinfo": 'none',
            "mode": 'lines',
            "showlegend": False,
            "fill": 'toself'
        }

        return [drivable_area_traces, lanes_traces,
                white_broken_single_traces, white_solid_single_traces, white_solid_double_traces,
                yellow_broken_single_traces, yellow_broken_double_traces,
                yellow_solid_single_traces, yellow_solid_double_traces, yellow_passing_double_traces,
                ped_cross_traces]

    def make_dynamic_data(self, static_data):

        # calculating the max of timestamps
        max_timestamp = self.deserealized_scene.tracks.features.shape[1] + \
            self.deserealized_scene.tracks.future_features.shape[1] - 1

        tracks = self.deserealized_scene.tracks
        coords = np.concatenate((tracks.features[:, :, :2], tracks.future_features[:, :, :2]), axis=1)
        headings = np.concatenate((tracks.features[:, :, 4], tracks.future_features[:, :, 4]), axis=1)
        dimensions = np.concatenate((tracks.features[:, :, 5:], tracks.future_features[:, :, 5:]), axis=1)
        valid = np.concatenate((tracks.valid, tracks.future_valid), axis=1)

        if self.deserealized_scene.traffic_lights:
            tl_states = np.vstack((self.deserealized_scene.traffic_lights.states,
                                   self.deserealized_scene.traffic_lights.future_states))

        track_colors = [
            THEMECOLORS["magenta"] if (tracks.type_and_category[idx, 0] == 2 and
                                       tracks.type_and_category[idx, 1] not in visualize_trajectory_for_tracks)
            else color_by_category[tracks.type_and_category[idx, 1]]
            for idx in range(tracks.features.shape[0])
        ]

        significant_tracks_ids = []
        for idx in range(tracks.features.shape[0]):
            if tracks.type_and_category[idx, 1] in visualize_trajectory_for_tracks:
                significant_tracks_ids.append(idx)

        # calculating scale for significant tracks only
        sign_x_coords = coords[:, :, 0][valid > 0]
        sign_y_coords = coords[:, :, 1][valid > 0]
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
            for track_idx in range(tracks.features.shape[0]):
                track_color = track_colors[track_idx]
                track_type, track_category = tuple(tracks.type_and_category[track_idx, :])

                # current coordinate and trace calculation
                track_current_coords = coords[track_idx, ts, :]
                track_zero_to_current = coords[track_idx, :ts+1, :]
                valid_zero_to_current = valid[track_idx, :ts+1]
                track_zero_to_current_masked = track_zero_to_current[valid_zero_to_current > 0]
                trace = track_zero_to_current_masked.reshape((-1, 2))

                # flag - no points to plot in this track
                empty_track = valid[track_idx, ts] < 0.5

                # contour calculation
                object_contour = scale_object(
                    dimensions[track_idx, ts, 0],
                    dimensions[track_idx, ts, 1],
                    track_type
                )
                # rotate every object besides pedestrians
                if track_type != 2:
                    # rotation matrix
                    heading = headings[track_idx, ts]
                    rot_m = np.array([[np.cos(heading), -np.sin(heading)],
                                      [np.sin(heading), np.cos(heading)]])
                    object_contour = rot_m @ object_contour

                if track_type == 1:  # vehicle
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
                            "text": 'vehicle (' + track_category_text_by_category[track_category] + ")",
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
                            track_category_text_by_category[track_category] + ")",
                         "fill": 'toself',
                         "showlegend": self.show_legend,
                         "name": str(track_type).lower()[11:] + " " + str(track_idx),
                         }
                    )

                if track_category in visualize_trajectory_for_tracks:
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

            tl_data = []
            if self.deserealized_scene.traffic_lights:
                traffic_lights = self.deserealized_scene.traffic_lights
                for tl_index in range(traffic_lights.coordinates.shape[0]):
                    cs = traffic_lights.directions[tl_index, :]
                    rot_matrix = np.array([[cs[0], -cs[1]], [cs[1], cs[0]]])
                    contour = rot_matrix @ tl_contour + traffic_lights.coordinates[tl_index:tl_index+1, :].T
                    traffic_light = {
                        "x": contour[0, :],
                        "y": contour[1, :],
                        "line": {
                            "width": 1,
                            "color": tl_color_by_state[tl_states[ts, tl_index]]
                        },
                        "hoverinfo": 'none',
                        "mode": 'lines',
                        "showlegend": self.show_legend,
                        "fill": 'toself',
                    }
                    tl_data.append(traffic_light)


            frame = {"data": [*static_data, *vehicles_data, *others_data, *tl_data, *trajectories],
                     "name": str(ts)}

            first_traj_idx = len(static_data) + len(vehicles_data) + len(others_data) + len(tl_data)
            self.trajectories_trace_indices = [first_traj_idx, first_traj_idx+len(trajectories)]
            frames.append(frame)

        return frames

    def read_scene_data(self, scene_id):
        scene_filename = self.scene_files[scene_id]
        p = self.folders_by_datasetpart[self.dataset_part][0] / scene_filename
        path2read = str(p.resolve())

        with open(path2read, 'rb') as opened_file:
            self.deserealized_scene = pickle.load(opened_file)

    @staticmethod
    def split_trace(trace):
        split_indices = []
        for j in range(1, trace.shape[0]-1):
            if abs(trace[j, 0] - trace[j + 1, 0]) > 2. or \
               abs(trace[j, 1] - trace[j + 1, 1]) > 2.:
                split_indices.append(j)

        result_trace = none_vector
        ind_before = 0
        for ind in split_indices:
            result_trace = np.vstack((result_trace, trace[ind_before:ind + 1, :], none_vector))
            ind_before = ind + 1

        result_trace = np.vstack((result_trace, trace[ind_before:, :]))

        return result_trace

    @staticmethod
    def separate_obstacles(xyz):
        if xyz.shape[0] == 0:
            return xyz
        none_vector = np.array([None, None, None]) if xyz.shape[1] == 3 else np.array([None, None])

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


if __name__ == "__main__":
    fc = GeneralFigureCreator()
    fc.read_scene_data(0)
