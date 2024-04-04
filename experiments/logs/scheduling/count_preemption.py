import argparse

import numpy as np

import workload_pb2
from matplotlib import pyplot as plt
from matplotlib import ticker
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='robot.log')
    args = parser.parse_args()

    with open(args.log, 'r') as f:
        log = f.readlines()
    num_log = len(log)
    log_i = 0
    preempted = 0
    count = 0
    while log_i < num_log:
        line = log[log_i]
        split_line = line.strip().split()
        log_i += 1
        if len(split_line) > 4 and split_line[4] == '[EXEC:BLOCK]':
            current = int(split_line[8])
            while log_i < num_log:
                line = log[log_i]
                split_line = line.strip().split()
                log_i += 1
                if len(split_line) > 4 and split_line[4] == '[EXEC:BLOCK]':
                    idx = int(split_line[8])
                    if current != idx:
                        preempted += 1
                        log_i -= 1
                        break
                if len(split_line) > 4 and split_line[4] == '[EXEC:EXIT]':
                    idx = int(split_line[10])
                    count += 1
                    if current != idx:
                        preempted += 1
                        break
                    else:
                        break

    print(f'{preempted}/{count}={preempted/count}')

