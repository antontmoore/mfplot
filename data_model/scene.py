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


class TrafficLight:
    # coordinates: x, y
    # shape: (num_of_traffic_lights, 2)
    coordinates: Union[npt.NDArray, None]

    # directions: cos(theta), sin(theta)
    # shape: (num_of_traffic_lights, 2)
    directions: Union[npt.NDArray, None]

    # shape: (timesteps, num_of_traffic_lights)
    states: Union[npt.NDArray, None]
    future_states: Union[npt.NDArray, None]


class Lanes:
    # shape: (total_number_of_points, 2)
    centerlines: Union[npt.NDArray, None]

    # shape: (total_number_of_points,)
    ids: Union[npt.NDArray, None]

class Scene:
    scene_id: str
    lanes: Union[Lanes, None]
    road_markup: RoadMarkup
    road_border: npt.NDArray
    crosswalk: npt.NDArray
    traffic_lights: Union[TrafficLight, None]
    tracks: Tracks
