import numpy.typing as npt
from typing import Union


class Tracks:
    # track type
    # 0 - unset, 1 - vehicle, 2 -pedestrian, 3 - cyclist/bicycle/motorcyclist, 4 - bus, 5 - other

    # track category
    # 1 - sdc/focal_track, 2 - tracks_to_predict/scored_track,
    # 3 - objects_of_interest/unscored_track, 4 - other fragment
    type_and_category: npt.NDArray

    # features: position(x, y), velocity(x, y), yaw, length, width
    # shape = (num_of_tracks, timesteps, feature_dimension)
    features: npt.NDArray
    valid: npt.NDArray

    future_features: npt.NDArray
    future_valid: npt.NDArray


class RoadMarkup:
    white_broken_single: Union[npt.NDArray, None]
    white_solid_single: Union[npt.NDArray, None]
    white_solid_double: Union[npt.NDArray, None]
    yellow_broken_single: Union[npt.NDArray, None]
    yellow_broken_double: Union[npt.NDArray, None]
    yellow_solid_single: Union[npt.NDArray, None]
    yellow_solid_double: Union[npt.NDArray, None]
    yellow_passing_double: Union[npt.NDArray, None]


class Scene:
    scene_id: str
    lanes_centerline: Union[npt.NDArray, None]
    road_markup: RoadMarkup
    road_border: npt.NDArray
    crosswalk: npt.NDArray
    tracks: Tracks
