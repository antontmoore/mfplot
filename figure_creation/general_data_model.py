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
from plotly.graph_objects import Figure
import numpy as np
import pickle
import os

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
'motion_prediction_data_v_0_1'

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
        # frames = self.make_dynamic_data(static_data=static_plot_data)
        frames = []

        # fig_dict["data"] = frames[0]["data"]
        fig_dict["data"] = static_plot_data
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
        #TODO: don't forget
        # if self.scale_variant == 2:
        #     fig_dict["layout"]["xaxis"]["range"] = self.significant_scale_range_x
        #     fig_dict["layout"]["yaxis"]["range"] = self.significant_scale_range_y
        fig_dict["layout"]["margin"] = dict(l=0, r=0, b=0, t=0)

        fig = Figure(fig_dict)
        return fig

    def make_static_data(self):

        # generate dicts for drivable area traces
        drivable_area_traces = {
                "x": self.deserealized_scene.road_border[:, 0].tolist(),
                "y": self.deserealized_scene.road_border[:, 1].tolist(),
                "line": {
                    "width": 1,
                    "color": THEMECOLORS['dark-green']
                },
                "hoverinfo": 'none',
                "mode": 'lines',
                "fill": 'none',
                "showlegend": False
            }

        rb = self.deserealized_scene.road_border

        # generate dicts for lanes traces
        lanes_traces = {
            "x": self.deserealized_scene.lanes_borderline[:, 0].tolist(),
            "y": self.deserealized_scene.lanes_borderline[:, 1].tolist(),
            "line": {
                "width": 0.5,
                "color": THEMECOLORS['light-grey']
            },
            "hoverinfo": 'none',
            "mode": 'lines',
            # "marker": {'color': 'LightSkyBlue', 'size': 2},
            "showlegend": False,
            "fill": 'none'
        }

        # generate dicts for pedestrian crosses
        # single dict for every crossing because of filling
        # self.deserealized_scene.crosswalk
        ped_cross_traces = {
            "x": self.deserealized_scene.crosswalk[:, 0].tolist(),
            "y": self.deserealized_scene.crosswalk[:, 1].tolist(),
            "line": {
                "width": 0,
                "color": THEMECOLORS['medium-grey']
            },
            "hoverinfo": 'none',
            "mode": 'lines',
            "showlegend": False,
            "fill": 'toself'
        }

        return [drivable_area_traces, lanes_traces, ped_cross_traces]


    def read_scene_data(self, scene_id):
        scene_filename = self.scene_files[scene_id]
        p = self.folders_by_datasetpart[self.dataset_part][0] / scene_filename
        path2read = str(p.resolve())

        # opened_file = open(path2read, 'rb')
        with open(path2read, 'rb') as opened_file:
            self.deserealized_scene = pickle.load(opened_file)

if __name__ == "__main__":
    fc = GeneralFigureCreator()
    fc.read_scene_data(0)
