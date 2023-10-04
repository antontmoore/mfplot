from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from data_model.scene import Lanes

dir_path = '/Users/antontmur/projects/mfplot/data/waymo_converted/training/part00000'
p = Path(dir_path)

MIN_RADIUS_OF_SCENE = 50
NEW_LANE_DIST = 2.
CONNECTION_RADIUS = 1.5
CONNECTION_DIR = 0.2
MAX_DISTANCE_ALONG_GRAPH = 100.
X, Y = 0, 1


def filter_lane_points(lanes, track_center):
    track_center = np.mean(full_track, axis=0)
    min_x, max_x = np.min(full_track[:, 0]), np.max(full_track[:, 0])
    min_y, max_y = np.min(full_track[:, 1]), np.max(full_track[:, 1])
    scene_radius = max(max_x - min_x, max_y - min_y, MIN_RADIUS_OF_SCENE)
    lanes_filtered = Lanes()
    idxs_filtered = np.any(np.square(lanes.centerlines[:, None] - track_center).sum(axis=2) <= scene_radius ** 2, axis=1)
    lanes_filtered.centerlines = lanes.centerlines[idxs_filtered]
    lanes_filtered.ids = lanes.ids[idxs_filtered]
    return lanes_filtered


def are_connected(lane_from, lane_to):
    if np.linalg.norm(lane_from[-1, :] - lane_to[0, :]) < CONNECTION_RADIUS:
        angle_from = np.arctan2(lane_from[-1, Y] - lane_from[-2, Y], lane_from[-1, X] - lane_from[-2, X])
        angle_to = np.arctan2(lane_to[1, Y] - lane_to[0, Y], lane_to[1, X] - lane_to[0, X])
        if np.abs(np.abs(angle_from - angle_to)) < CONNECTION_DIR:
            return True
    return False


def create_lane_graph(lanes):
    lanes_list = []
    prev_point = lanes.centerlines[0, :]
    prev_lane_id = lanes.ids[0]
    start_idx = 0
    for j in range(1, lanes.centerlines.shape[0]):
        cur_point = lanes.centerlines[j, :]
        cur_lane_id = lanes.ids[j]
        if np.linalg.norm(cur_point - prev_point) > NEW_LANE_DIST or \
           cur_lane_id != prev_lane_id:
            if j - start_idx == 1:
                pass
                # lanes_list[-1] = np.vstack((lanes_list[-1], lanes[j, :]))
            else:
                lanes_list.append(lanes.centerlines[start_idx:j, :])
            start_idx = j

        prev_point, prev_lane_id = cur_point, cur_lane_id

    if start_idx < lanes.centerlines.shape[0]-1:
        lanes_list.append(lanes.centerlines[start_idx:, :])

    #  dictionary of connections key = lane_id, value = set_of_neighbours
    conn_dict = {j: set() for j in range(len(lanes_list))}

    # fill the dict
    for i in range(len(lanes_list)):
        for j in range(len(lanes_list)):
            if are_connected(lanes_list[i], lanes_list[j]):
                # conn_dict[j].add(i)
                conn_dict[i].add(j)

    # calculate the lengths of lanes
    lanes_len = []
    for lane in lanes_list:

        lane_points_num = lane.shape[0]
        lane_length = \
            np.sum(
                np.linalg.norm(
                    lane[1:lane_points_num, :] - lane[:lane_points_num-1, :], axis=1
                )
            )

        lanes_len.append(lane_length)

    return lanes_list, lanes_len, conn_dict


def find_closest_lane(lanes_list, past_traj):
    closure_metrics = np.zeros((len(lanes_list),))
    last_point = past_traj[-1, :]
    last_point_direction = past_traj[-1, :] - past_traj[-2, :]
    last_point_direction /= max(0.00001, np.linalg.norm(last_point_direction))
    for lane_idx, lane in enumerate(lanes_list):
        dist_squared = np.square(lane[:, None] - last_point).sum(axis=2)
        min_idx = np.argmin(dist_squared)
        if min_idx == 0:
            lane_angle = np.arctan2(
                lane[1, Y] - lane[0, Y],
                lane[1, X] - lane[0, X]
            )
        else:
            lane_angle = np.arctan2(
                lane[min_idx, Y] - lane[min_idx-1, Y],
                lane[min_idx, X] - lane[min_idx-1, X]
            )

        collinearity = np.dot(
            np.array([np.cos(lane_angle), np.sin(lane_angle)]),
            last_point_direction
        )
        closure_metrics[lane_idx] = dist_squared[min_idx] - 100 * collinearity**2

    closest_lane_index = np.argmin(closure_metrics)

    return closest_lane_index


def find_connected_lanes(lanes_list, lanes_len, conn_dict, closest_lane_idx):

    connected_lanes = [closest_lane_idx]
    to_visit = [(closest_lane_idx, 0)]
    visited = set()
    while len(to_visit) > 0:
        lane_idx, current_distance = to_visit.pop()

        for neighbour_idx in conn_dict[lane_idx]:
            if neighbour_idx not in visited and current_distance < MAX_DISTANCE_ALONG_GRAPH:
                connected_lanes.append(neighbour_idx)
                to_visit.append((neighbour_idx, current_distance + lanes_len[neighbour_idx]))
        visited.add(lane_idx)

    return connected_lanes


for filepath in p.iterdir():

    with open(filepath, 'rb') as opened_file:
        scene = pickle.load(opened_file)
        tracks = scene.tracks
        lanes = scene.lanes
        for t_index in range(tracks.features.shape[0]):
            if not (tracks.type_and_category[t_index, 1] in [1, 2]):
                # leave only sdc and track_to_predict
                continue

            if tracks.type_and_category[t_index, 0] == 2:
                # skip pedestrians
                continue

            valid_indicies = np.where(tracks.valid[t_index, :] > 0)[0]
            future_valid_indicies = np.where(tracks.future_valid[t_index, :] > 0)[0]
            full_track = np.vstack((
                tracks.features[t_index, valid_indicies, :2],
                tracks.future_features[t_index, future_valid_indicies, :2],
            ))
            print(f"filepath = {filepath}, t_index = {t_index}")

            lanes_filtered = filter_lane_points(lanes, full_track)
            lanes_list, lanes_len, conn_dict = create_lane_graph(lanes_filtered)
            closest_lane_idx = find_closest_lane(lanes_list=lanes_list,
                                                 past_traj=tracks.features[t_index, valid_indicies, :2])
            connected_lanes = find_connected_lanes(lanes_list, lanes_len, conn_dict, closest_lane_idx)

            colors = 'bgrcm'
            c_ind = 0
            for lane_idx, lane in enumerate(lanes_list):
                if lane_idx in connected_lanes:
                    plt.plot(lane[:, 0], lane[:, 1], 'k')
                # else:
                color = colors[c_ind]
                c_ind = (c_ind + 1) % 5
                plt.plot(lane[:, 0], lane[:, 1], color,  linewidth=0.5)
                text_coord = np.mean(lane, axis=0)
                # plt.text(text_coord[0], text_coord[1], str(lane_idx), fontsize=6, color=color)
                # plt.plot(lane[-1, 0], lane[-1, 1], color+'o', linewidth=0.5)
            plt.plot(full_track[:, 0], full_track[:, 1], 'b:', linewidth=1.5)
            plt.plot(full_track[10, 0], full_track[10, 1], 'go')
            plt.plot(full_track[-1, 0], full_track[-1, 1], 'ro')

            plt.title(str(filepath).split('/')[-1] + ' _ t_index = ' + str(t_index))

            plt.show()
            a = 1

