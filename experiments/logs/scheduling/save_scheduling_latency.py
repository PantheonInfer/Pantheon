import argparse

import numpy as np

import workload_pb2
from matplotlib import pyplot as plt
from matplotlib import ticker
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='robot_poisson_25.log')
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

    trace = [[], []]

    num_tasks = []
    while log_i < num_log:
        line = log[log_i]
        split_line = line.strip().split()
        if len(split_line) > 4 and split_line[4] == '[SCHE]':
            time_stamp = (int(split_line[5]) - start_time_stamp) / 1000. / 1000.
            latency = int(split_line[6])

            trace[0].append(time_stamp)
            trace[1].append(latency)
            num_tasks.append(int(split_line[7]))
        log_i += 1
    trace = np.array(trace)
    # easy_trace = np.array([trace[0][easy_case == 1], trace[1][easy_case == 1]])
    # no_easy_trace = np.array([trace[0][easy_case == 0], trace[1][easy_case == 0]])
    # x = easy_trace[0][easy_trace[0] >= 150]
    # y = easy_trace[1][easy_trace[0] >= 150]
    # y = y[x <= 250]
    # x = x[x <= 250]
    # # plt.scatter(x, y)
    #
    # np.savetxt('trace1.txt', np.array([x, y]))
    #
    # x = no_easy_trace[0][no_easy_trace[0] >= 150]
    # y = no_easy_trace[1][no_easy_trace[0] >= 150]
    # y = y[x <= 250]
    # x = x[x <= 250]
    # # plt.scatter(x, y)
    #
    # np.savetxt('trace2.txt', np.array([x, y]))

    # plt.scatter(trace[0][easy_case == 0], trace[1][easy_case == 0], label='sort only')
    # plt.scatter(trace[0][easy_case == 1], trace[1][easy_case == 1], label='full')
    # # print(np.mean(trace[1][easy_case == 0]))
    # # print(np.mean(trace[1][easy_case == 1]))
    # plt.legend()


    num_tasks = np.array(num_tasks)

    print('max num:', max(num_tasks))
    print('min num:', min(num_tasks))
    # plt.scatter(trace[0][num_tasks == 5], trace[1][num_tasks == 5], c='m')
    # plt.scatter(trace[0][num_tasks == 4], trace[1][num_tasks == 4], c='k')
    # plt.scatter(trace[0][num_tasks == 3], trace[1][num_tasks == 3], c='b')
    # plt.scatter(trace[0][num_tasks <= 3], trace[1][num_tasks <= 3], label='<=3')
    # plt.scatter(trace[0][num_tasks > 3], trace[1][num_tasks > 3], label='>3')
    # plt.legend()
    #
    # plt.xlim(0, 30)
    # plt.ylim(0, 500)
    # plt.show()
    idx = np.where((num_tasks >= 1) & (num_tasks <= 4))[0]
    easy_trace = np.array([trace[0][idx], trace[1][idx]])
    x = easy_trace[0][easy_trace[0] >= 0]
    y = easy_trace[1][easy_trace[0] >= 0]
    y = y[x <= 30]
    x = x[x <= 30]
    plt.scatter(x, y)
    np.savetxt('trace1.txt', np.array([x, y]))


    idx = np.where((num_tasks >= 5) & (num_tasks <= 8))[0]
    no_easy_trace = np.array([trace[0][idx], trace[1][idx]])
    x = no_easy_trace[0][no_easy_trace[0] >= 0]
    y = no_easy_trace[1][no_easy_trace[0] >= 0]
    y = y[x <= 30]
    x = x[x <= 30]
    plt.scatter(x, y)

    np.savetxt('trace2.txt', np.array([x, y]))

    idx = np.where((num_tasks >= 9))[0]
    no_easy_trace = np.array([trace[0][idx], trace[1][idx]])
    x = no_easy_trace[0][no_easy_trace[0] >= 0]
    y = no_easy_trace[1][no_easy_trace[0] >= 0]
    y = y[x <= 30]
    x = x[x <= 30]
    plt.scatter(x, y)

    np.savetxt('trace3.txt', np.array([x, y]))

    plt.show()




