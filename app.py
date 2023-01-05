from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import pandas as pd
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ObjectType, TrackCategory
from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path
import numpy as np
from collections import defaultdict
from copy import deepcopy

import json
import os

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


scale_coefficient = 0.08

contour_x = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.4, 1.6,  1.9,  2.3,  2.6,  3.0,  3.5,  4.0,  4.5,  5.0,  5.5,  6.0,  7.0,  8.3, 42.5, 44.5, 48.0, 50.2, 50.5, 51.0, 52.0, 52.5, 53.0, 53.5, 54.0, 54.5, 55.0, 55.1, 55.2]
contour_y = [0.0, 4.3, 5.8, 6.9, 7.2, 7.6, 8.0, 8.4, 8.9, 9.2, 9.5, 9.7, 10.1, 10.5, 10.8, 11.1, 11.4, 11.6, 11.8, 11.9, 12.1, 12.2, 12.5, 12.6, 12.4, 12.6, 12.4, 12.2, 12.1, 11.9, 11.4, 11.1, 10.7, 10.1,  9.0,  7.3,  3.5,  2.3,  0.0]
contour_x.extend(contour_x[::-1])
contour_x = [-1 * j + 27.6 for j in contour_x]
contour_y.extend([-1 * j for j in contour_y[::-1]])
vehicle_contour = np.array([contour_x, contour_y]) * scale_coefficient


back_glass_x = [48.5, 48.5, 48.4, 48.3, 48.0, 48.9, 49.9, 50.8, 51.7, 52.6, 53.1, 53.2, 53.2]
back_glass_y = [ 0.0,  4.3,  5.2,  6.3,  8.0,  8.2, 8.5,   8.8,  9.3,  6.4,  4.2,  2.6,  0.0]
back_glass_x.extend(back_glass_x[::-1])
back_glass_y.extend([-1 * j for j in back_glass_y[::-1]])

front_glass_x = [13.9, 14.0, 14.2, 14.5, 14.8, 15.2, 15.5, 16.0, 18.7, 20.1, 24.1, 23.9, 23.8]
front_glass_y = [ 0.0,  2.3,  3.8,  5.5,  6.8,  8.4,  9.3, 10.6, 10.0,  9.6,  8.4,  6.5,  0.0]
front_glass_x.extend(front_glass_x[::-1])
front_glass_y.extend([-1 * j for j in front_glass_y[::-1]])

side_glass_x = [18.1, 23.9, 25.0, 25.7, 26.6, 42.3, 43.8, 44.4, 44.6, 44.8, 45.2, 45.2, 45.1, 44.8, 44.3, 18.1]
side_glass_y = [11.2,  9.3,  9.0,  8.9,  8.8,  8.8,  8.9,  9.0,  9.1,  9.3, 10.4, 10.8, 10.9, 11.1, 11.2, 11.2]
# side_glass_x.extend(side_glass_x[::-1])
# side_glass_y.extend([-1 * j for j in side_glass_y[::-1]])


vehicle_glass_x = [*front_glass_x, *back_glass_x]
vehicle_glass_y = [*front_glass_y, *back_glass_y]
vehicle_glass_x = [-1 * j + 27.6 for j in vehicle_glass_x]

vehicle_glass_x = np.array(vehicle_glass_x) * scale_coefficient
vehicle_glass_y = np.array(vehicle_glass_y) * scale_coefficient
vehicle_glass = np.vstack((vehicle_glass_x, vehicle_glass_y))


vehicle_contour = np.array([[-1.5, 1.0, 1.5, 1.5, 1.0, -1.5, -1.5],
                            [0.8, 0.8, 0.5, -0.5, -0.8, -0.8, 0.8]])
vehicle_glass = np.zeros((2, 1))

none_vector = np.array([None, None], ndmin=2)

class FigureCreator:

    def __init__(self):
        self.scenario = None
        self.static_map = None

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
        for track_idx, track in enumerate(self.scenario.tracks):
            for objs in track.object_states:
                coords[track_idx, objs.timestep, :] = objs.position
                headings[track_idx, objs.timestep] = objs.heading

            track_color = {
                TrackCategory.FOCAL_TRACK: THEMECOLORS["green"],
                TrackCategory.SCORED_TRACK: THEMECOLORS["blue"],
                TrackCategory.UNSCORED_TRACK: THEMECOLORS["white"],
                TrackCategory.TRACK_FRAGMENT: THEMECOLORS["light-grey"]
            }[track.category]

            track_colors.append(track_color)
            track_types.append(track.object_type)


        # frames generation
        frames = []

        for ts in range(max_timestamp + 1):
            trajectories_x, trajectories_y = [], []

            contours = defaultdict(list)
            vehicles_data, vehicle_glass_data = [], []
            others_data = []
            for track_idx, track in enumerate(self.scenario.tracks):
                track_color, track_type = track_colors[track_idx], track_types[track_idx]
                # track_color = THEMECOLORS["green"] if track_idx == 0 else THEMECOLORS["blue"]
                track_current_coords = coords[track_idx, ts, :]

                track_current_coords_masked = np.ma.masked_less(track_current_coords, MASK_VALUE + 1)

                # no points to plot in this track
                empty_track = abs(track_current_coords[0] - MASK_VALUE) < 1

                trace = track_current_coords_masked.compressed().reshape((-1, 2))

                # trajectories_x.extend(trace[:, 0])
                # trajectories_y.extend(trace[:, 1])
                # trajectories_x.append(None)
                # trajectories_y.append(None)

                # rotation matrix
                heading = headings[track_idx, ts]
                rot_m = np.array([[np.cos(heading), -np.sin(heading)],
                                  [np.sin(heading), np.cos(heading)]])

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
                            "hoverinfo": 'none',
                            "mode": 'lines',
                            "showlegend": False,
                            "fill": 'toself',
                            "name": "tracktupac "+str(track_idx),
                        }

                    )
                    # rotated_glass = np.add(rot_m @ vehicle_glass, trace[-1, :, np.newaxis])
                    # contours.append(rotated_vehicle)
                    # vehicle_glasses.append(rotated_glass)

                    # contours[track_color].append(
                    #     np.hstack((
                    #         rotated_vehicle,
                    #         none_vector.T
                    #     ))
                    # )


                    # vehicle_glass_data.append(
                    #     {"x": rotated_glass[0],
                    #      "y": rotated_glass[1],
                    #      "line":
                    #          {"width": 1,
                    #           "color": THEMECOLORS["black"],
                    #           },
                    #      "hoverinfo": 'none',
                    #      "mode": 'none',
                    #      "showlegend": False,
                    #      "fill": 'toself',
                    #      "fillcolor": THEMECOLORS["black"],
                    #      }
                    # )
                else:
                    # others_data.append(
                    #     {"x": [trace[-1, 0]],
                    #      "y": [trace[-1, 1]],
                    #      "hoverinfo": 'none',
                    #      "mode": 'markers',
                    #      "marker": {
                    #          "size": 12,
                    #          "line": {
                    #              "width": 2,
                    #              "color": THEMECOLORS["magenta"]
                    #          }
                    #      },
                    #      "fill": 'none',
                    #      "showlegend": False,
                    #      }
                    # )
                    # others.append(trace[-1, :])
                    # color = THEMECOLORS["magenta"]


                    a = 1

            trajectories_data = \
                {"x": trajectories_x,
                 "y": trajectories_y,
                 "line":
                     {"width": 1,
                      "color": THEMECOLORS["blue"],
                      },
                 "hoverinfo": 'none',
                 "mode": 'lines',
                 "showlegend": False,
                 "fill": 'none'
                 }

            # for color in [THEMECOLORS["green"], THEMECOLORS["blue"]]:
            #     if len(contours[color]) == 0:
            #         print("Empty")
            #     contour_data = np.hstack(contours[color])
            #
            # # for color, contour_list in contours.items():
            # #     contour_data = np.hstack(contour_list)
            #     vehicles_data.append(
            #         {"x": contour_data[0, :],
            #          "y": contour_data[1, :],
            #          "line": {
            #              "width": 1,
            #              "color": color
            #          },
            #          "hoverinfo": 'none',
            #          "mode": 'lines',
            #          "showlegend": False,
            #          "fill": 'toself'
            #          }
            #     )
            frame = {"data": [*static_data, *vehicles_data],
                     "name": str(ts)}

            if ts == 0:
                frame0 = deepcopy(frame)
            frames.append(frame)
            # if len(others) > 0:
            #     other_data = \
            #         {"x": others[0][0],
            #          "y": others[0][1],
            #          "hoverinfo": 'none',
            #          "mode": 'markers',
            #          "marker": {
            #              "size": 12,
            #              "line": {
            #                  "width": 2,
            #                  "color": THEMECOLORS["magenta"]
            #              }
            #          },
            #          "fill": 'none',
            #          "showlegend": False,
            #          }
            #     # frame["data"].append(other_data)


            a = 1


        return frames, frame0["data"]

    def generate_figure(self, scene_id):

        scene_path = 'data/val/' + scene_id
        static_map_path = scene_path + f"/log_map_archive_{scene_id}.json"
        scenario_path = scene_path + f"/scenario_{scene_id}.parquet"


        self.scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        self.static_map = ArgoverseStaticMap.from_json(Path(static_map_path))

        fig_dict = {
            "data": [],
            "layout": {},
            "frames": []
        }
        fig_dict["layout"]["xaxis"] = {"showgrid": False, "zeroline": False, "showticklabels": False}

        fig_dict["layout"]["yaxis"] = {"showgrid": False, "zeroline": False, "showticklabels": False,
                                       "scaleanchor": "x", "scaleratio": 1}
        fig_dict["layout"]["margin"] = dict(l=0, r=0, b=0, t=0)

        static_plot_data = self.make_static_data()
        # drive_area_traces.append(lanes_traces)






        sliders_dict = {
            "active": 0,
            "activebgcolor": THEMECOLORS['light-blue'],
            "bordercolor": THEMECOLORS['medium-grey'],
            "bgcolor": THEMECOLORS['blue'],
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
            #     "font": {"size": 20},
            #     "prefix": "Step:",
                "visible": False,
            #     "xanchor": "right"
            },
            "transition": {"duration": 0, "easing": "linear"},
            "pad": {"b": 10, "t": 10},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": []
        }
        frames, frame0_data = self.make_dynamic_data(static_data=static_plot_data)

        fig_dict["data"] = frame0_data
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
                        "args": [None, {"frame": {"duration": 0, "redraw": True},
                                        "fromcurrent": True, "transition": {"duration": 0,
                                                                            "easing": "quadratic-in-out"}}],
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
                "pad": {"r": 10, "t": 20},
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
        fig = go.Figure(fig_dict)


        return fig


fc = FigureCreator()

fig = fc.generate_figure(scene_id='0a0ef009-9d44-4399-99e6-50004d345f34')




# dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
dbc_css = dbc.themes.DARKLY
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
# app = Dash(__name__)


app.layout = dbc.Container(children=[
    html.H1(children='Structure'),
    html.Div(children='''
        Web application plotting.
    '''),

    dcc.Graph(
        id='example-graph',
        style={'width': '90vh', 'height': '90vh'},
        figure=fig,
    ),
], class_name='dbc')

if __name__ == '__main__':
    app.run_server(debug=True)