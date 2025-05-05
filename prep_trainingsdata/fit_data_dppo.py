import read_data
import numpy as np
import json
import rerun as rr
import os
import io
import shutil
#import vizualize_data
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
    f =open('traj_aut.json')
    files = json.load(f)

    obs_keys = {'pose':[0,1,2,3,4,5], 'velocity':[0,1,2], 'gripper_positions':0, 'compensated_base_force':[0,1,2]}
    actions = np.zeros([100,4])
    states = np.zeros([100,23])
    traj_lengths = np.zeros(100)
    rr.init("3D Pose Visualization", spawn=True)
    file_idx = 0
    newdata = {}
    newdata_num = []
    time = {}
    time_num = []
    interval_k = 0
    idxStart = 0
    idxEnd = 0
    for file in files:
        try:
            data = read_data.load_h5_file("/home/uhhnie/robotics_project/data_prep/data/" + file['file'])
            ts_min = np.inf
            diff_old = np.inf
            for k, v in data["timestamps"].items():
                if v[0] < ts_min:
                    ts_min = v[0]
            for k in obs_keys.keys():
                data["timestamps"][k] = data["timestamps"][k] #- ts_min
                diff_new = np.searchsorted(data["timestamps"][k], file['end'], side="left") - np.searchsorted(data["timestamps"][k], file['start'], side="left")
                if diff_new < diff_old:
                    idxStart = np.searchsorted(data["timestamps"][k], file['start'], side="left")
                    idxEnd = np.searchsorted(data["timestamps"][k], file['end'], side="left")
                    interval_k = k
                    diff_old = diff_new
            for k, v in data["timestamps"].items():
                data["timestamps"][k] = data["timestamps"][k] - ts_min
            traj_lengths[file_idx]= diff_old
            interval = data["timestamps"][interval_k][idxStart:idxEnd]
            for k,v in obs_keys.items():
                idxes = np.searchsorted(data["timestamps"][k],interval,side="left")
                newdata[k] = data['robot_state'][k][idxes][:,v]
                time[k] = data["timestamps"][k][idxes]
            for data_key in obs_keys.keys():
                for i in range(0, len(time[data_key]), 10):
                    rr.set_time_seconds("timestamp", time[data_key][i])
                    if type(newdata[data_key][i]) == np.float64:
                        rr.log(f"Traj{file_idx}:{data_key}/{name}", rr.Scalar(newdata[data_key][i]))
                    elif len(newdata[data_key][i]) == 3:
                        for _id, name in enumerate(["x", "y", "z"]):
                            rr.log(f"Traj{file_idx}:{data_key}/{name}", rr.Scalar(newdata[data_key][i][_id]))
                    elif len(newdata[data_key][i]) == 7 and "joint" not in name:
                        for _id, name in enumerate(["x", "y", "z", "qx", "qy", "qz", "qw"]):
                            rr.log(f"Traj{file_idx}:{data_key}/{name}", rr.Scalar(newdata[data_key][i][_id]))
                            #rr.log("scene/points", rr.Points3D(positions=points, colors=colors))
                    elif len(newdata[data_key][i]) == 6:
                        for _id, name in enumerate(["x", "y", "z", "qx", "qy", "qz"]):
                            rr.log(f"Traj{file_idx}:{data_key}/{name}", rr.Scalar(newdata[data_key][i][_id]))
                    elif len(newdata[data_key][i]) == 7 and "joint" in name:
                        for _id in range(7):
                            rr.log(f"Traj{file_idx}:{data_key}/joint{_id}", rr.Scalar(newdata[data_key][i][_id]))
            display_trajectory(newdata["pose"],file["file"])
            #newdata_num.append(newdata)
            #time_num.append(time)
            file_idx = file_idx + 1
        except Exception as e:
            print(f"Error reading file: {e}")
             # Consider logging the error
    plt.show()
    np.savez('train', actions=actions, states=states, traj_lengths=traj_lengths)
    import pdb
    pdb.set_trace()