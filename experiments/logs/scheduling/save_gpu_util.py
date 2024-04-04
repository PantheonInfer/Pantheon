import argparse

import numpy as np

import workload_pb2
from matplotlib import pyplot as plt
from matplotlib import ticker
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--log', type=str, default='obj_sign.log')
    parser.add_argument('--log', type=str, default='robot_234.log')
    # parser.add_argument('--log', type=str, default='obj_sign_sound_2345_45_ours.log')
    args = parser.parse_args()

    with open(args.log, 'r') as f:
        log = f.readlines()
    num_log = len(log)
    log_i = 0
    while log_i < num_log:
        line = log[log_i]
        log_i += 1
        split_line = line.strip().split()
        if split_line[4] == '[START]':
            start_time_stamp = int(split_line[-1])
            break
    log_i = 0
    trace = [[], []]
    while log_i < num_log:
        line = log[log_i]
        split_line = line.strip().split()
        if len(split_line) > 4 and split_line[4] == '[GPU]':
            time_stamp = (int(split_line[5]) - start_time_stamp) / 1000. / 1000.
            utilization = float(split_line[6])
            trace[0].append(time_stamp)
            trace[1].append(utilization)
        log_i += 1
    trace = np.array(trace)
    plt.plot(trace[0][::10], trace[1][::10])
    # start, end = 0, 30
    # file_name = 'util_2.txt'
    # start, end = 30, 60
    # file_name = 'util_3.txt'
    # start, end = 60, 90
    # file_name = 'util_4.txt'
    x = trace[0][trace[0] > start]
    y = trace[1][trace[0] > start]
    y = y[x <= end]
    x = x[x <= end]
    print(np.mean(y))
    print(len(y))
    plt.plot(x, y)
    print(len(y))
    plt.show()
    # np.savetxt('trace.txt', trace)
    np.savetxt(file_name, y[2:942])


