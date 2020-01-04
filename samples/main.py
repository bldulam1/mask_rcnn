import os
import io
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
# import custom_mrcnn

ROOT_DIR = '/home/bldulam/Downloads/2011_09_26_drive_0001_extract'
IMAGE_DIR = os.path.join(ROOT_DIR, 'image_03/data')
LIDAR_DIR = os.path.join(ROOT_DIR, 'velodyne_points/data')


def get_files(directory):
    return [os.path.join(directory, file_name) for file_name in os.listdir(directory)]


lidar_frames = get_files(LIDAR_DIR)


def project_lidar_to_camera(file_name):

    f = open(file_name, "r")
    df = pd.read_csv(io.StringIO('X Y Z I\n' + f.read()),
                     delim_whitespace=True)
    df['Depth'] = (df.X*df.X + df.Y*df.Y)**(1/2)
    df['R'] = (df.X*df.X + df.Y*df.Y + df.Z*df.Z)**(1/2)
    def azimuth(row):
        return math.degrees(math.atan(row['Z'] / row['R']))
    df['Azimuth'] = df.apply(azimuth, axis=1)
    plt.scatter(df.X, df.Y, df.Z)
    plt.show()
    # for f_line in f_lines:
    #     x, y, z, _ = f_line.split(" ")
    #     x, y, z = int(1000*float(x)), int(1000*float(y)), int(1000*float(z))
    #     xs, ys, zs = np.append(xs, x), np.append(ys, y), np.append(zs, z)

    #     print(xs, ys, zs)


project_lidar_to_camera(lidar_frames[0])
