from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from constants import X, Y
from constants import CONNECTION_RADIUS
from constants import CONNECTION_DIR
from constants import MAX_DISTANCE_ALONG_GRAPH
from constants import NEW_LANE_DIST
from constants import ANGLE_ESTIMATION_MATRIX
from constants import NEXT_POINT_STRING
from preprocessing.functions import normalized
from preprocessing.functions import filter_lane_points
from preprocessing.functions import remove_duplicate_points_in_lane
from preprocessing.functions import split_stuck_lanes
from preprocessing.functions import split_lanes_with_tee_in_the_middle

dir_path = '/Users/antontmur/projects/mfplot/data/waymo_converted/training/part00000'
p = Path(dir_path)


def create_kd_tree(lanes):
    kdt = KDTree(data=lanes.centerlines)
    # lane_id_by_coord = {tuple(lanes.centerlines[j, :]) : lanes.ids[j] for j in range(lanes.ids.shape[0])}

    lane_id_by_coord, lane_coords_by_id = {}, {}
    prev_id = lanes.ids[0]
    start_j = 0
    for j in range(1, lanes.ids.shape[0]):
        coord = tuple(lanes.centerlines[j, :])
        lane_id = int(lanes.ids[j])

        # form dictionary lane_id_by_coord
        if coord not in lane_id_by_coord:
            lane_id_by_coord[coord] = set()
        lane_id_by_coord[coord].add(lane_id)

        # form dictionary lane_coords_by_id
        if lanes.ids[j] != prev_id or j == lanes.ids.shape[0]-1:
            lane_points_coords = lanes.centerlines[start_j: j, :]
            lane_coords_by_id[int(prev_id)] = lane_points_coords

            start_j = j
            prev_id = lanes.ids[j]

    return kdt, lane_id_by_coord, lane_coords_by_id


def find_closest_lane(past_traj, kdt, lanes):
    last_coordinate = past_traj[-1, :]
    close_points_inds = kdt.query_ball_point(tuple(last_coordinate), 10)
    if len(close_points_inds) < 1:
        close_points_inds = kdt.query_ball_point(tuple(last_coordinate), 30)
    close_points_inds = np.array(close_points_inds)
    close_points_inds = close_points_inds[close_points_inds > 1]

    dist = np.linalg.norm(lanes.centerlines[close_points_inds] - last_coordinate, axis=1)[:, np.newaxis]
    vector_lane_point = lanes.centerlines[close_points_inds] - lanes.centerlines[close_points_inds - 1]

    vector_traj = past_traj[-1, :] - past_traj[-2, :]

    vector_lane_point = normalized(vector_lane_point)
    vector_traj = normalized(vector_traj)
    collinearity = vector_lane_point @ vector_traj.T

    closure_metrics = dist ** 2 - 30 * collinearity ** 2

    closest_index = close_points_inds[np.argmin(closure_metrics)]
    closest_point = lanes.centerlines[closest_index]
    closest_lane_id = int(lanes.ids[closest_index])

    return closest_point, closest_lane_id


def find_connected_points(start_point, start_lane_id, kdt, lane_id_by_coord, lane_coords_by_id):

    def get_neighbours(lane_id, dist):
        neighbours = []

        lane_coords = lane_coords_by_id[lane_id]
        if lane_coords.shape[0] < 2:
            return neighbours

        # last point
        angle_from = np.arctan2(lane_coords[-1, Y] - lane_coords[-2, Y], lane_coords[-1, X] - lane_coords[-2, X])

        # close to current point
        point_to_indices = kdt.query_ball_point(lane_coords[-1, :], CONNECTION_RADIUS)
        points_to = [(point_to_idx, angle_from) for point_to_idx in point_to_indices]

        # close to point that had to be the next after lane
        if lane_coords.shape[0] >= 3:
            # estimation of parabola z = a + b*t + c*t^2, where z = x, y.
            parabola_coeff = ANGLE_ESTIMATION_MATRIX @ lane_coords[-3:, :]

            # estimation of the next point coordinate
            next_point = (NEXT_POINT_STRING @ parabola_coeff)[0]

            # and angle
            angle_from = np.arctan2(next_point[Y] - lane_coords[-1, Y], next_point[X] - lane_coords[-1, X])

            # take points from that region
            point_to_indices = kdt.query_ball_point(next_point, CONNECTION_RADIUS)
            points_to.extend(
                [(point_to_idx, angle_from) for point_to_idx in point_to_indices]
            )

        for point_to_idx, angle_from in points_to:

            if point_to_idx > 0 and lanes_filtered.ids[point_to_idx] == lanes_filtered.ids[point_to_idx - 1]:
                # only from the start of lane
                continue

            if (point_to_idx < lanes_filtered.ids.shape[0]-1 and
                lanes_filtered.ids[point_to_idx] != lanes_filtered.ids[point_to_idx + 1]):
                # only lanes with more than one point
                continue

            point_to_lane = int(lanes_filtered.ids[point_to_idx])
            point_to_coord = lanes_filtered.centerlines[point_to_idx]
            after_point_to_coord = lanes_filtered.centerlines[point_to_idx + 1]

            angle_to = np.arctan2(
                after_point_to_coord[Y] - point_to_coord[Y],
                after_point_to_coord[X] - point_to_coord[X]
            )
            if np.abs(angle_from - angle_to) < CONNECTION_DIR:
                neighb_lane_ids = lane_id_by_coord[tuple(point_to_coord)]

                distance = dist + np.linalg.norm(point_to_coord - lane_coords[-1, :])
                neighbours.append((point_to_coord, point_to_lane, distance))

        return neighbours

    connection_type = np.zeros((0, 1))
    connected_points = np.zeros((0, 2))
    connected_lanes_ids = set()
    connected_points = np.vstack((connected_points, start_point))
    current_lane_id = start_lane_id
    to_visit = [(start_point, current_lane_id, 0)]      # (point, lane_id, distance)
    visited = set()
    while len(to_visit) > 0:
        current_point, current_lane_id, current_distance = to_visit.pop()
        lane_coords = lane_coords_by_id[current_lane_id]

        # go through lane to find current point
        j_start = int(np.where(abs(lane_coords - current_point) < 0.0001)[0][0])

        # add all points from this lane, unless we are out of max_distance
        j = j_start + 1
        while j < lane_coords.shape[0] and current_distance < MAX_DISTANCE_ALONG_GRAPH:
            current_distance += np.linalg.norm(lane_coords[j, :] - lane_coords[j-1, :])
            j += 1
        connected_points = np.vstack((connected_points, lane_coords[j_start: j + 1, :]))

        if current_distance > MAX_DISTANCE_ALONG_GRAPH:
            continue

        if lane_coords.shape[0] > 1:
            neighbours = get_neighbours(current_lane_id, current_distance)
        else:
            neighbours = []

        for neighbour in neighbours:
            if tuple(neighbour[0]) not in visited:
                to_visit.append(neighbour)
        visited.add(tuple(current_point))

    return connected_points, connection_type


def create_lane_graph(lanes):
    def are_connected(lane_from, lane_to):
        if np.linalg.norm(lane_from[-1, :] - lane_to[0, :]) < CONNECTION_RADIUS:
            angle_from = np.arctan2(lane_from[-1, Y] - lane_from[-2, Y], lane_from[-1, X] - lane_from[-2, X])
            angle_to = np.arctan2(lane_to[1, Y] - lane_to[0, Y], lane_to[1, X] - lane_to[0, X])
            if np.abs(np.abs(angle_from - angle_to)) < CONNECTION_DIR:
                return True
        return False
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


def find_closest_lane_old(lanes_list, past_traj):
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
            lanes_filtered = remove_duplicate_points_in_lane(lanes_filtered)
            lanes_filtered = split_stuck_lanes(lanes_filtered)
            lanes_filtered = split_lanes_with_tee_in_the_middle(lanes_filtered)

            kdt, lane_id_by_coord, lane_coords_by_id = create_kd_tree(lanes_filtered)



            closest_point, closest_lane_id = find_closest_lane(past_traj=tracks.features[t_index, valid_indicies, :2],
                                                               kdt=kdt,
                                                               lanes=lanes_filtered)

            connected_points, _ = find_connected_points(closest_point, closest_lane_id, kdt, lane_id_by_coord, lane_coords_by_id)
            plt.plot(connected_points[:, 0], connected_points[:, 1], 'oc')




            lanes_list, lanes_len, conn_dict = create_lane_graph(lanes_filtered)
            closest_lane_idx = find_closest_lane_old(lanes_list=lanes_list,
                                                     past_traj=tracks.features[t_index, valid_indicies, :2])
            connected_lanes = find_connected_lanes(lanes_list, lanes_len, conn_dict, closest_lane_idx)
            # left_neighbours, right_neighbours = find_neighbour_lanes(kdt)

            colors = 'cbgmr'
            c_ind = 0
            for lane_idx, lane in enumerate(lanes_list):
                if lane_idx in connected_lanes:
                    plt.plot(lane[:, 0], lane[:, 1], '--k')
                # else:
                color = colors[c_ind]
                c_ind = (c_ind + 1) % 5
                plt.plot(lane[:, 0], lane[:, 1], color+'-',  linewidth=0.5)
                text_coord = np.mean(lane, axis=0)
                lane_id = '---'
                if lane.shape[0] > 1:
                    lset = lane_id_by_coord[tuple(lane[1, :])]
                    if bool(lset):
                        lane_id = list(lset)[0]
                plt.text(text_coord[0], text_coord[1], lane_id, fontsize=6, color=color)
                # plt.plot(lane[-1, 0], lane[-1, 1], color+'o', linewidth=0.5)
            plt.plot(full_track[:, 0], full_track[:, 1], 'b:', linewidth=1.5)
            plt.plot(full_track[10, 0], full_track[10, 1], 'go')
            plt.plot(full_track[-1, 0], full_track[-1, 1], 'ro')

            plt.title(str(filepath).split('/')[-1] + ' _ t_index = ' + str(t_index))

            plt.show()
            a = 1

