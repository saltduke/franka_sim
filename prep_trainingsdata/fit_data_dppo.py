import read_data
import numpy as np
import json
import rerun as rr
import os
import io
import shutil
# import vizualize_data
import matplotlib.pyplot as plt


def display_trajectory(xyz_data, title):
    """
    Displays a 3D trajectory plot from XYZ data.

    Args:
        xyz_data (numpy.ndarray): A 2D numpy array of shape (N, 3), where N is the number of points,
            and each row represents the X, Y, and Z coordinates of a point.
        title (str, optional): The title of the plot. Defaults to "Trajectory Plot".
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz_data[:, 0], xyz_data[:, 1], xyz_data[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)


if __name__ == "__main__":
    # Specify the path to your HDF5 file
    img = False
    f = open('traj_aut.json')
    files = json.load(f)

    obs_keys = {'pose': [0, 1, 2, 3, 4, 5, 6], 'velocity': [0, 1, 2], 'gripper_positions': 0,
                'compensated_base_force': [0, 1, 2]}
    fr = 20
    actions = np.zeros([1, 7])
    states = np.zeros([1, 16])
    # traj_lengths = np.zeros(100, dtype=int)
    # rr.init("3D Pose Visualization", spawn=True)
    file_idx = 0
    newdata = {}
    newdata_num = []
    traj_lengths = []
    time = {}
    time_num = []
    interval_k = 0
    idxStart = 0
    idxEnd = 0
    for file in files:
        try:

            data = read_data.load_h5_file("/srv/asl/REASEMBLE/data/" + file['file'])
            ts_min = np.inf
            diff_old = np.inf
            for k, v in data["timestamps"].items():
                if v[0] < ts_min:
                    ts_min = v[0]
            for k in obs_keys.keys():
                data["timestamps"][k] = data["timestamps"][k]  # - ts_min
                diff_new = np.searchsorted(data["timestamps"][k], file['end'], side="left") - np.searchsorted(
                    data["timestamps"][k], file['start'], side="left")
                if diff_new < diff_old:
                    idxStart = np.searchsorted(data["timestamps"][k], file['start'], side="left")
                    idxEnd = np.searchsorted(data["timestamps"][k], file['end'], side="left")
                    interval_k = k
                    diff_old = diff_new
            for k, v in data["timestamps"].items():
                data["timestamps"][k] = data["timestamps"][k] - ts_min
            traj_length = int(diff_old / fr)
            traj_lengths.append(traj_length)
            interval = data["timestamps"][interval_k][idxStart:idxEnd]
            for k, v in obs_keys.items():
                idxes = np.searchsorted(data["timestamps"][k], interval, side="left")
                newdata[k] = data['robot_state'][k][idxes][:, v]
                time[k] = data["timestamps"][k][idxes]  # anschauen
            action_traj = np.zeros([traj_length, 7])
            states_traj = np.zeros([traj_length, 13])
            action_first = np.zeros([1, 7])  # because action needs to be difference of previous postion
            first_pose_idx = np.searchsorted(data["timestamps"]['pose'], file['start'] - ts_min, side="left") - fr
            first_grip_idx = np.searchsorted(data["timestamps"]['gripper_positions'], file['start'] - ts_min,
                                             side="left") - fr
            action_first[0, 0:6] = data['robot_state']['pose'][first_pose_idx][0:6]
            action_first[0, 6] = data['robot_state']['gripper_positions'][first_grip_idx][0]
            for data_key in obs_keys.keys():
                for idx, i in enumerate(range(0, len(time[data_key]) - fr, fr)):
                    # rr.set_time_seconds("timestamp", time[data_key][i])
                    if type(newdata[data_key][i]) == np.float64:
                        action_traj[idx][6] = newdata[data_key][i]
                        states_traj[idx][6] = newdata[data_key][i]
                        if i + fr < len(time[data_key]) and states_traj.shape[1] == 13 and newdata[data_key][i + fr] - \
                                newdata[data_key][i] >= 0.01:
                            final_pose = newdata['pose'][i][0:3] * np.ones([traj_length, 3])
                            states_traj = np.hstack([states_traj, final_pose])
                        # rr.log(f"Traj{file_idx}:{data_key}/{name}", rr.Scalar(newdata[data_key][i]))
                    elif len(newdata[data_key][i]) == 3:
                        for _id, name in enumerate(["x", "y", "z"]):
                            if data_key == "compensated_base_force":
                                states_traj[idx][6 + _id] = newdata[data_key][i][_id]
                            elif data_key == "velocity":
                                states_traj[idx][9 + _id] = newdata[data_key][i][_id]
                            # rr.log(f"Traj{file_idx}:{data_key}/{name}", rr.Scalar(newdata[data_key][i][_id]))
                    elif len(newdata[data_key][i]) == 6:
                        for _id, name in enumerate(["x", "y", "z", "qx", "qy", "qz"]):
                            action_traj[idx][_id] = newdata[data_key][i][_id]
                            states_traj[idx][_id] = newdata[data_key][i][_id]
                            # rr.log(f"Traj{file_idx}:{data_key}/{name}", rr.Scalar(newdata[data_key][i][_id]))
                            # rr.log("scene/points", rr.Points3D(positions=points, colors=colors))
                    # elif len(newdata[data_key][i]) == 7 and "joint" in name:
                    # for _id in range(7):
                    # rr.log(f"Traj{file_idx}:{data_key}/joint{_id}", rr.Scalar(newdata[data_key][i][_id]))
            states = np.vstack((states, states_traj))
            action_traj = np.vstack([action_traj, np.zeros((1, 7))]) - np.vstack([action_first, action_traj])
            actions = np.vstack((actions, np.delete(action_traj, -1, 0)))
            # display_trajectory(newdata["pose"],file["file"])
            # newdata_num.append(newdata)
            # time_num.append(time)
            file_idx = file_idx + 1
        except Exception as e:
            traj_lengths.pop(-1)
            print(f"Error reading file {file['file']}: {e}")
            # Consider logging the error
    actions = np.delete(actions, 0, 0)
    states = np.delete(states, 0, 0)
    traj_lengths = np.array(traj_lengths)
    # plt.show()
    print("States: " + str(states.shape))
    print("Action: " + str(actions.shape))
    print("Trajectories count: " + str(traj_lengths.shape) + " sum of length" + str(np.sum(traj_lengths)))
    np.savez('train', actions=actions, states=states, traj_lengths=traj_lengths)

    # import pdb
    # pdb.set_trace()