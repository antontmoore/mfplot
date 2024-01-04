import numpy as np
from data_model.scene import Lanes
from scipy.spatial import KDTree
from constants import MIN_RADIUS_OF_SCENE
from constants import CONNECTION_RADIUS
from constants import TEE_CLOSE_DIST


def filter_lane_points(lanes, track):
    """
        Function for filtering of lanes data structure.
        It gets lanes structure with two fields (ids, centerlines) and filter it leaving only the points
        close to the middle of full_track.

        :param lanes: see data_model.scene.Lanes
        :param track: 2D-polyline. shape: [number_of_points, 2]
        :return: lanes filtered.
    """

    track_center = np.mean(track, axis=0)
    min_x, max_x = np.min(track[:, 0]), np.max(track[:, 0])
    min_y, max_y = np.min(track[:, 1]), np.max(track[:, 1])
    scene_radius = max(max_x - min_x, max_y - min_y, MIN_RADIUS_OF_SCENE)
    lanes_filtered = Lanes()
    idxs_filtered = np.any(np.square(lanes.centerlines[:, None] - track_center).
                           sum(axis=2) <= scene_radius ** 2, axis=1)
    lanes_filtered.centerlines = lanes.centerlines[idxs_filtered]
    lanes_filtered.ids = lanes.ids[idxs_filtered]
    return lanes_filtered


def split_stuck_lanes(lanes):
    """
       Function splits lane in two different if it contain two separate parts.
       It gives new lane_id for part of the lane.

       :param lanes: see data_model.scene.Lanes
       :return: renewed lanes structure
    """

    unique_ids = np.unique(lanes.ids)
    new_lane_id = int(np.max(unique_ids) + 1)

    j_start_from = 0
    for j in range(1, lanes.centerlines.shape[0]-1):
        if lanes.ids[j] != lanes.ids[j-1]:
            j_start_from = j

        if lanes.ids[j] != lanes.ids[j+1]:
            continue

        if (
            lanes.ids[j] == lanes.ids[j-1] and
            np.linalg.norm(lanes.centerlines[j, :] - lanes.centerlines[j - 1, :]) > 2 * CONNECTION_RADIUS
        ):
            lanes.ids[j_start_from:j] = new_lane_id
            new_lane_id += 1
            j_start_from = j

    return lanes


def split_lanes_with_tee_in_the_middle(lanes):
    """
        Function splits lane if it has the start of another lane somewhere in the middle,
        very close to the point of lane.

        :param lanes: see data_model.scene.Lanes
        :return: renewed lanes structure
    """

    kdt = KDTree(data=lanes.centerlines)
    unique_ids = np.unique(lanes.ids)
    new_lane_id = int(np.max(unique_ids) + 1)

    j_start_from = 0
    for j in range(1, lanes.centerlines.shape[0]-1):
        if lanes.ids[j] != lanes.ids[j-1]:
            j_start_from = j

        if lanes.ids[j] != lanes.ids[j+1]:
            continue

        close_point_indices = kdt.query_ball_point(lanes.centerlines[j, :], TEE_CLOSE_DIST)
        close_point_indices.remove(j)

        if len(close_point_indices) > 0:
            # we have someone very close to the middle of the lane point
            for close_point_idx in close_point_indices:
                if lanes.ids[close_point_idx] != lanes.ids[close_point_idx-1]:
                    lanes.ids[j_start_from: j] = new_lane_id
                    new_lane_id += 1
                    j_start_from = j

    return lanes


def change_duplicate_ids_in_lanes(lanes):
    """
        Function finds different lanes with the same id and change id for one of them.
        Different lanes are detected as lanes with the same id, but points are not followed by each other.

        :param lanes: see data_model.scene.Lanes
        :return: renewed lanes structure
    """
    unique_ids = np.unique(lanes.ids)
    new_id = int(np.max(unique_ids) + 1)
    have_already_met = set()
    prev_id = -1
    changes = []  # tuples (index_from, index_to, new_id)
    j = 1
    while j < lanes.ids.shape[0]:
        this_id = int(lanes.ids[j])
        if this_id != prev_id:
            if this_id not in have_already_met:
                have_already_met.add(this_id)
                while j < lanes.ids.shape[0] and lanes.ids[j] == this_id:
                    j += 1

            else:
                change_tuple = (j,)
                while j < lanes.ids.shape[0] and lanes.ids[j] == this_id:
                    j += 1
                change_tuple += (j, new_id)
                new_id += 1
                changes.append(change_tuple)
            prev_id = this_id

    for change in changes:
        lanes.ids[change[0]: change[1]] = change[2]

    return lanes


def remove_duplicate_points_in_lane(lanes):
    """
        Function removes duplicates from the lane, only if we have two same points going one-by-one.

        :param lanes: see data_model.scene.Lanes
        :return: renewed lanes structure
    """

    mask = np.ones((lanes.ids.shape[0],), dtype=bool)
    for j in range(1, lanes.centerlines.shape[0]):
        if lanes.ids[j] == lanes.ids[j-1]:
            if np.linalg.norm(lanes.centerlines[j, :] - lanes.centerlines[j-1, :]) < 0.001:
                mask[j] = False

    lanes.ids = lanes.ids[mask]
    lanes.centerlines = lanes.centerlines[mask, :]

    return lanes


def normalized(a, axis=-1, order=2):
    """
        Normalize vector to [-1, 1].

        :param a: original array
        :param axis: axis to normalize through
        :param order: order of norm
        :return: normalized array
    """

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
