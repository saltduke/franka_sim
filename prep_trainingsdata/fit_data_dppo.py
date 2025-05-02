import read_data
import numpy as np
import json
import os
import io
import shutil
import vizualize_data
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
    f =open('traj.json')
    files = json.load(f)

    obs_keys = {'pose':[0,1,2], 'velocity':[0,1,2], 'gripper_positions':0, 'compensated_base_force':[0,1,2]}
    actions = np.zeros([100,4])
    states = np.zeros([100,23])
    traj_lengths = np.zeros(100)

    i = 0
    ts_min = np.inf
    diff_old = np.inf
    newdata = {}
    newdata_num = []
    time = {}
    time_num = []
    interval_k = 0
    idxStart = 0
    idxEnd = 0
    for file in files:
        try:
            data = read_data.load_h5_file("/home/cgreiml/Documents/extract_data_for_training/vis_lib/data/" + file['file'])
            ts_min = np.inf
            for k, v in data["timestamps"].items():
                if v[0] < ts_min:
                    ts_min = v[0]

            for k, v in data["timestamps"].items():
                data["timestamps"][k] = data["timestamps"][k] - ts_min
            for k in obs_keys.keys():
                data["timestamps"][k] = data["timestamps"][k] #- ts_min
                diff_new = np.searchsorted(data["timestamps"][k], file['end'], side="left") - np.searchsorted(data["timestamps"][k], file['start'], side="left")
                if diff_new < diff_old:
                    idxStart = np.searchsorted(data["timestamps"][k], file['start'], side="left")
                    idxEnd = np.searchsorted(data["timestamps"][k], file['end'], side="left")
                    interval_k = k
                    diff_old = diff_new
            traj_lengths[i]= diff_old
            interval = data["timestamps"][interval_k][idxStart:idxEnd]
            for k,v in obs_keys.items():
                idxes = np.searchsorted(data["timestamps"][k],interval,side="left")
                newdata[k] = data['robot_state'][k][idxes][:,v]
                time[k] = data["timestamps"][k][idxes]
            display_trajectory(newdata["pose"],"Trajectory" + str(i))
            #newdata_num.append(newdata)
            #time_num.append(time)
            i = i + 1
        except Exception as e:
            print(f"Error reading file: {e}")
             # Consider logging the error
    plt.show()
    np.savez('train', actions=actions, states=states, traj_lengths=traj_lengths)
    import pdb
    pdb.set_trace()