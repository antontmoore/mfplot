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

import json
import os

THEMECOLORS = {
    'background': '#0e0e30',
    'dark-grey': '#353536',
    'medium-grey': '#4c4c4c',
    'light-grey': '#505050',
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




class FigureCreator:

    def __init__(self):
        self.scenario = None
        self.static_map = None

    def make_static_data(self):

        # generate dicts for drivable area traces
        drivable_area_traces = []
        for drive_area in self.static_map.vector_drivable_areas.values():
            drivable_area_trace = {
                "x": drive_area.xyz[:, 0],
                "y": drive_area.xyz[:, 1],
                "line": dict(width=1, color=THEMECOLORS['dark-grey']),
                "hoverinfo": 'none',
                "mode": 'lines',
                "fill": 'toself',
                "showlegend": False
            }
            drivable_area_traces.append(drivable_area_trace)

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
                "width": 1,
                "color": THEMECOLORS['light-grey'],
            },
            "hoverinfo": 'none',
            "mode": 'lines',
            "showlegend": False
        }

        return drivable_area_traces + [lanes_traces]

    def make_dynamic_data(self, static_data):

        # vehicle_contour = np.array([[-1.5, 1.0, 1.5, 1.5, 1.0, -1.5],
        #                             [0.8, 0.8, 0.5, -0.5, -0.8, -0.8]])


        # calculating the max of timestamps
        max_timestamp = max([object_state.timestep for track in self.scenario.tracks for object_state in track.object_states])
        number_of_tracks = len(self.scenario.tracks)

        coords = -np.ones((number_of_tracks, max_timestamp + 1, 2))
        headings = np.zeros((number_of_tracks, max_timestamp + 1))
        for track_idx, track in enumerate(self.scenario.tracks):
            # print(track.category)
            for objs in track.object_states:
                coords[track_idx, objs.timestep, :] = objs.position
                headings[track_idx, objs.timestep] = objs.heading

        frames = []

        for ts in range(max_timestamp + 1):
            trajectories_x, trajectories_y = [], []
            vehicles_x, vehicles_y = [], []
            vehicle_glasses_x, vehicle_glasses_y = [], []
            other_x, other_y = [], []
            for track_idx, track in enumerate(self.scenario.tracks):
                track_coords = coords[track_idx, ts, :]

                track_coords_masked = np.ma.masked_less(track_coords, 0)
                if 0 in track_coords_masked.shape:
                    # no points to plot in this track
                    continue

                trace = track_coords_masked.compressed().reshape((-1, 2))

                trajectories_x.extend(trace[:, 0])
                trajectories_y.extend(trace[:, 1])
                trajectories_x.append(None)
                trajectories_y.append(None)

                if trace.shape[0] > 0:
                    heading = headings[track_idx, ts]
                    rot_m = np.array([[np.cos(heading), np.sin(heading)],
                                      [-np.sin(heading), np.cos(heading)]])
                    if track.object_type == ObjectType.VEHICLE and \
                        track.category in [TrackCategory.FOCAL_TRACK, TrackCategory.SCORED_TRACK]:
                        rotated_vehicle = rot_m @ vehicle_contour
                        #
                        vehicles_x.extend(np.add(rotated_vehicle[0, :], trace[-1, 0]))
                        vehicles_y.extend(np.add(rotated_vehicle[1, :], trace[-1, 1]))
                        vehicles_x.append(None)
                        vehicles_y.append(None)

                        rotated_glasses = rot_m @ np.vstack((vehicle_glass_x, vehicle_glass_y))
                        vehicle_glasses_x.extend(np.add(rotated_glasses[0, :], trace[-1, 0]))
                        vehicle_glasses_y.extend(np.add(rotated_glasses[1, :], trace[-1, 1]))
                        vehicle_glasses_x.append(None)
                        vehicle_glasses_y.append(None)
                    else:
                        other_x.append(trace[-1, 0])
                        other_y.append(trace[-1, 1])

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

            vehicles_data = \
                {"x": vehicles_x,
                 "y": vehicles_y,
                 "line":
                     {"width": 1,
                      "color": THEMECOLORS["blue"],
                      },
                 "hoverinfo": 'none',
                 "mode": 'lines',
                 "showlegend": False,
                 "fill": 'toself'
                 }

            vehicle_glass_data = \
                {"x": vehicle_glasses_x,
                 "y": vehicle_glasses_y,
                 "line":
                     {"width": 1,
                      "color": THEMECOLORS["black"],
                      },
                 "hoverinfo": 'none',
                 "mode": 'none',
                 "showlegend": False,
                 "fill": 'toself',
                 "fillcolor": THEMECOLORS["black"],
                 }

            other_data = \
                {"x": other_x,
                 "y": other_y,
                 "hoverinfo": 'none',
                 "mode": 'markers',
                 "marker": {
                     "size": 12,
                     "line": {
                         "width": 2,
                         "color": THEMECOLORS["magenta"]
                     }
                 },
                 "fill": 'none',
                 "showlegend": False,
                 }

            frame = {"data": [vehicles_data, vehicle_glass_data, other_data],
                     "name": str(ts)}
            frames.append(frame)
            a = 1


        return frames

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

        fig_dict["data"] = static_plot_data




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
        frames = self.make_dynamic_data(static_plot_data)
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