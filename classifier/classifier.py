from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from constants import X, Y
from constants import CONNECTION_RADIUS
from constants import CONNECTION_DIR
from constants import MIN_PAST_TRAJ_DIFFERENCE
from constants import MAX_DISTANCE_ALONG_GRAPH
from constants import ANGLE_ESTIMATION_MATRIX
from constants import NEXT_POINT_STRING
from preprocessing.functions import normalized
from preprocessing.functions import filter_lane_points
from preprocessing.functions import remove_duplicate_points_in_lane
from preprocessing.functions import change_duplicate_ids_in_lanes
from preprocessing.functions import split_stuck_lanes
from preprocessing.functions import split_lanes_with_tee_in_the_middle
from preprocessing.functions import relu
from math import pi


class TrajClassifier:
    def __init__(self):
        self.filepath = ''
        self.scene = None
        self.number_of_tracks = 0

    def load_scene(self, filepath):
        self.filepath = filepath
        with open(filepath, 'rb') as opened_file:
            self.scene = pickle.load(opened_file)
            self.number_of_tracks = self.scene.tracks.features.shape[0]

    def classify_track(self, track_idx, do_plot=False):
        if not (self.scene.tracks.type_and_category[track_idx, 1] in [1, 2]):
            # leave only sdc and track_to_predict
            return 'unscored track', None

        if self.scene.tracks.type_and_category[track_idx, 0] == 2:
            # skip pedestrians
            return 'pedestrian', None

        # if str(self.filepath).split("/")[-1][:5] != '8622b' or track_idx != 3:
        #     return '', None

        valid_indicies = np.where(self.scene.tracks.valid[track_idx, :] > 0)[0]
        future_valid_indicies = np.where(self.scene.tracks.future_valid[track_idx, :] > 0)[0]
        full_track = np.vstack((
            self.scene.tracks.features[track_idx, valid_indicies, :2],
            self.scene.tracks.future_features[track_idx, future_valid_indicies, :2],
        ))
        print(f"filepath = {self.filepath}, t_index = {track_idx}")

        # filter and preprpocess lanes data
        lanes_filtered = filter_lane_points(self.scene.lanes, full_track)
        lanes_filtered = remove_duplicate_points_in_lane(lanes_filtered)
        lanes_filtered = change_duplicate_ids_in_lanes(lanes_filtered)
        lanes_filtered = split_stuck_lanes(lanes_filtered)
        lanes_filtered = split_lanes_with_tee_in_the_middle(lanes_filtered)

        kdt, lane_id_by_coord, lane_coords_by_id = self.create_kd_tree_and_dictionaries(lanes=lanes_filtered)

        closest_point, closest_lane_id = self.find_closest_lane(
            past_traj=self.scene.tracks.features[track_idx, valid_indicies, :2],
            kdt=kdt,
            lanes=lanes_filtered
        )

        connected_points, connected_lanes = self.find_connected_points(
            closest_point,
            closest_lane_id,
            kdt,
            lane_coords_by_id,
            lane_id_by_coord,
            lanes_filtered)


        valid_indicies = np.where(self.scene.tracks.future_valid[track_idx, :] > 0)[0]
        future_traj = self.scene.tracks.future_features[track_idx, valid_indicies, :2]
        future_closest_point, future_closest_lane_id = self.find_closest_lane(
            past_traj=self.scene.tracks.future_features[track_idx, valid_indicies, :2],
            kdt=kdt,
            lanes=lanes_filtered
        )

        if int(future_closest_lane_id) in connected_lanes:
            label = 'lane keeping'
        else:
            label = 'something else'

        plot_ax = None
        if do_plot:
            plot_ax = self.make_plot(lane_coords_by_id, lane_id_by_coord,
                                     connected_points, full_track, track_idx, label)
        return label, plot_ax

    def make_plot(self, lane_coords_by_id, lane_id_by_coord, connected_points, full_track, track_idx, label):

        f, ax = plt.subplots()
        plt.plot(connected_points[:, 0], connected_points[:, 1], 'oc')

        colors = 'cbgmr'
        c_ind = 0
        for lane_id, lane in lane_coords_by_id.items():

            color = colors[c_ind]
            c_ind = (c_ind + 1) % 5
            plt.plot(lane[:, 0], lane[:, 1], color + '-', linewidth=0.5)
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

        label_for_plot = '+++++' if label == 'lane keeping' else ''
        plt.title(str(self.filepath).split('/')[-1] + ' _ t_index = ' + str(track_idx) + label_for_plot)

        return ax

    @staticmethod
    def create_kd_tree_and_dictionaries(lanes):
        kdt = KDTree(data=lanes.centerlines)

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
            if lanes.ids[j] != prev_id:
                lane_points_coords = lanes.centerlines[start_j: j, :]
                lane_coords_by_id[int(prev_id)] = lane_points_coords

                start_j = j
                prev_id = lanes.ids[j]
            elif j == lanes.ids.shape[0] - 1:
                lane_points_coords = lanes.centerlines[start_j: j + 1, :]
                lane_coords_by_id[int(prev_id)] = lane_points_coords

        return kdt, lane_id_by_coord, lane_coords_by_id

    @staticmethod
    def find_closest_lane(past_traj, kdt, lanes):
        last_coordinate = past_traj[-1, :]
        ball_size = 20
        close_points_inds = []
        while len(close_points_inds) == 0:
            close_points_inds = kdt.query_ball_point(tuple(last_coordinate), ball_size)
            ball_size *= 1.1
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

        if np.linalg.norm(past_traj[0, :] - past_traj[-1, :]) < MIN_PAST_TRAJ_DIFFERENCE:
            # staying at the same point
            collinearity = np.zeros_like(collinearity)

        closure_metrics = dist ** 2 + 1e6 * relu(-collinearity) ** 2

        closest_index = close_points_inds[np.argmin(closure_metrics)]
        closest_point = lanes.centerlines[closest_index]
        closest_lane_id = int(lanes.ids[closest_index])

        return closest_point, closest_lane_id

    @staticmethod
    def find_connected_points(start_point,
                              start_lane_id,
                              kdt,
                              lane_coords_by_id,
                              lane_id_by_coord,
                              lanes_filtered):

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

                if point_to_idx == lanes_filtered.ids.shape[0] - 1:
                    # skip last alone point
                    continue

                if (lanes_filtered.ids[point_to_idx] != lanes_filtered.ids[point_to_idx + 1]):
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
                    distance = dist + np.linalg.norm(point_to_coord - lane_coords[-1, :])
                    neighbours.append((point_to_coord, point_to_lane, distance))

            return neighbours

        connected_points = np.zeros((0, 2))
        connected_points = np.vstack((connected_points, start_point))
        current_lane_id = start_lane_id
        to_visit = [(start_point, current_lane_id, 0)]  # (point, lane_id, distance)
        visited = set()
        while len(to_visit) > 0:
            current_point, current_lane_id, current_distance = to_visit.pop()
            lane_coords = lane_coords_by_id[current_lane_id]

            # go through lane to find current point
            j_start = int(np.where(abs(lane_coords - current_point) < 0.0001)[0][0])

            # add all points from this lane, unless we are out of max_distance
            j = j_start + 1
            while j < lane_coords.shape[0] and current_distance < MAX_DISTANCE_ALONG_GRAPH:
                current_distance += np.linalg.norm(lane_coords[j, :] - lane_coords[j - 1, :])
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

        connected_lanes = set()
        for conn_point in connected_points:
            connected_lanes.add(list(lane_id_by_coord[tuple(conn_point)])[0])
        return connected_points, connected_lanes



if __name__ == "__main__":
    dir_path = '/Users/antontmur/projects/mfplot/data/waymo_converted/training/part00000'
    p = Path(dir_path)

    traj_classifier = TrajClassifier()
    lane_keeper_count, other_count = 0, 0
    for file_path in p.iterdir():
        traj_classifier.load_scene(file_path)

        for track_index in range(traj_classifier.number_of_tracks):
            label, track_plot = traj_classifier.classify_track(track_index, True)
            if label == 'lane keeping':
                lane_keeper_count += 1
            elif label == 'something else':
                other_count += 1

            if track_plot:
                plt.show()

    print(f"lane_keeper_count = {lane_keeper_count}, other = {other_count}")
