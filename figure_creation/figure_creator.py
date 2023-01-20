from abc import ABC
from typing import List
from plotly.graph_objects import Figure
from .shared import DatasetPart
from .shared import THEMECOLORS


class FigureCreatorBaseClass(ABC):
    def make_static_data(self) -> List:
        pass

    def make_dynamic_data(self, static_data: List) -> List:
        pass

    def read_scene_data(self, scene_id):
        pass

    def get_current_scene(self) -> Figure:
        return self.current_scene

    def generate_figure(self, scene_id: int) -> Figure:

        self.read_scene_data(scene_id)

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

        fig_dict["layout"]["xaxis"] = {"showgrid": False, "zeroline": False, "showticklabels": False}

        fig_dict["layout"]["yaxis"] = {"showgrid": False, "zeroline": False, "showticklabels": False,
                                       "scaleanchor": "x", "scaleratio": 1}
        if self.scale_variant == 2:
            fig_dict["layout"]["xaxis"]["range"] = self.significant_scale_range_x
            fig_dict["layout"]["yaxis"]["range"] = self.significant_scale_range_y
        fig_dict["layout"]["margin"] = dict(l=0, r=0, b=0, t=0)

        fig = Figure(fig_dict)
        return fig

    def change_datapart(self, dataset_part):
        dataset_part_enum = {'train': DatasetPart.TRAIN,
                             'val': DatasetPart.VAL,
                             'test': DatasetPart.TEST}[dataset_part]

        self.cached_scene[self.dataset_part] = self.current_scene
        self.cached_scene_id[self.dataset_part] = self.current_scene_id

        self.dataset_part = dataset_part_enum
        self.number_of_scenes = len(self.subdirs[dataset_part_enum])
        if dataset_part_enum in self.cached_scene:
            self.current_scene = self.cached_scene[dataset_part_enum]
            self.current_scene_id = self.cached_scene_id[dataset_part_enum]
        else:
            self.current_scene_id = 1
            self.current_scene = self.generate_figure(self.current_scene_id)
        return self.current_scene, self.current_scene_id

    def get_next_scene(self):

        if self.current_scene_id == self.number_of_scenes:
            print(f"Try to get the scene {self.current_figure_id+1}, but we have only {self.number_of_scenes} scenes.")
        else:
            self.current_scene_id += 1
            self.current_scene = self.generate_figure(self.current_scene_id)

        return self.current_scene, self.current_scene_id

    def get_previous_scene(self):

        if self.current_scene_id == 1:
            print(f"Try to get the scene 0, but we start from 1.")
        else:
            self.current_scene_id -= 1
            self.current_scene = self.generate_figure(self.current_scene_id)

        return self.current_scene, self.current_scene_id

    def get_scene_by_id(self, scene_id):
        if scene_id <= 0:
            print(f"Try to get the scene 0, but we start from 1.")
        elif scene_id > self.number_of_scenes:
            print(f"Try to get the scene {scene_id}, but we have only {self.number_of_scenes} scenes.")
        else:
            self.current_scene_id = scene_id
            self.current_scene = self.generate_figure(scene_id)
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
