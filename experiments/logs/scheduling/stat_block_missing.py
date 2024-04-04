import argparse

import numpy as np

import workload_pb2
from matplotlib import pyplot as plt
from matplotlib import ticker
from tqdm import tqdm
import statsmodels.api as sm



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='indoor_smart_traffic.log')
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

    num = []

    while log_i < num_log:
        line = log[log_i]
        split_line = line.strip().split()
        if len(split_line) > 4 and split_line[4] == '[SCHE:START]':
            count = 0
            while True:
                log_i += 1
                next_line = log[log_i]
                split_next_line = next_line.strip().split()
                if split_next_line[4] == '[EXEC:BLOCK]' or split_next_line[4] == '[EXEC:EXIT]':
                    count += 1
                elif split_next_line[4] == '[SCHE]':
                    num.append(count)

                    break
        log_i += 1

    ecdf = sm.distributions.ECDF(num)
    x = np.linspace(min(num), max(num))
    y = ecdf(x)
    plt.plot(x, y)
    plt.show()
    np.savetxt('stat.txt', np.array(num))




