import argparse

import numpy as np

import workload_pb2
from matplotlib import pyplot as plt
from matplotlib import ticker
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--log', type=str, default='obj_sign.log')
    parser.add_argument('--log', type=str, default='indoor_smart_traffic_20be.log')

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
            power = float(split_line[7])
            trace[0].append(time_stamp)
            trace[1].append(power)
        log_i += 1
    trace = np.array(trace)
    subsampled_trace = np.array([trace[0][::10], trace[1][::10]])
    print(subsampled_trace[1].max())
    np.savetxt('trace.txt', subsampled_trace)
    plt.plot(subsampled_trace[0], subsampled_trace[1])
    plt.show()




