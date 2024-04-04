import argparse

import numpy as np

import workload_pb2
from matplotlib import pyplot as plt
from matplotlib import ticker
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='obj_sign_sound_2345_45_ours_wo_pr.log')
    parser.add_argument('--workloads', type=str, default='../../settings/workload/obj_sign_sound_2345.bin')
    parser.add_argument('--trace_id', type=int, default=2)
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

    workloads = workload_pb2.Workloads()
    with open(args.workloads, 'rb') as f:
        workloads.ParseFromString(f.read())

    eval_info = []
    ids = []
    for i, w in tqdm(enumerate(workloads.workload)):
        eval_info.append({'id': w.id, 'model': w.model_name, 'target fps': 1 / ((w.deadline - w.release) / 1000. / 1000.),
                          'target response': (w.deadline - w.release) / 1000. , 'response_time': 0,
                          'release': w.release, 'deadline': w.deadline, 'fps': 0, 'accuracy': 0.})
        if w.id not in ids:
            ids.append(w.id)
    ids.sort()

    while log_i < num_log:
        line = log[log_i]
        split_line = line.strip().split()
        if len(split_line) > 4 and split_line[4] == '[EXEC:EXIT]':
            idx = int(split_line[10])
            time_stamp = int(split_line[7]) - start_time_stamp
            release_time_stamp = eval_info[idx]['release']
            delay = (time_stamp - release_time_stamp) / 1000. / 1000.
            # delay = int(split_line[7]) / 1000. / 1000.
            fps = 1 / delay
            eval_info[idx]['response_time'] = delay * 1000.
            eval_info[idx]['fps'] = fps
            eval_info[idx]['accuracy'] = float(split_line[-1]) * 100
        log_i += 1
    # for i, w in tqdm(enumerate(workloads.workload)):
    #     if w.id not in eval_info:
    #         eval_info[w.id] = {'model': w.model_name, 'deadline': (w.deadline - w.release) / 1000.,
    #                            'release': [w.release], 'latency': [], 'drop': [], 'accuracy': []}
    #     else:
    #         eval_info[w.id]['release'].append(w.release)
    #     mark = False
    #     temp = log_i
    #     while log_i < num_log:
    #         line = log[log_i]
    #         log_i += 1
    #         split_line = line.strip().split()
    #         if split_line[4] == '[EXEC:EXIT]' and int(split_line[7]) == i:
    #             mark = True
    #             time_stamp = int(split_line[5]) - start_time_stamp
    #             eval_info[w.id]['latency'].append((time_stamp - w.release) / 1000.)
    #             eval_info[w.id]['accuracy'].append(float(split_line[-1]) * 100)
    #             break
    #     if mark == False:
    #         log_i = temp
    #         eval_info[w.id]['latency'].append(0)
    #         eval_info[w.id]['accuracy'].append(0)
    #         eval_info[w.id]['drop'].append(len(eval_info[w.id]['accuracy']) - 1)
    release_records = []
    response_records = []
    fps_records = []
    accuracy_records = []
    target_fps = []
    target_response = []
    dmr = 0
    num_jobs = 0
    for i in ids:
        release_record = [eval_info[j]['release']  / 1000. / 1000. for j in range(len(eval_info)) if eval_info[j]['id'] == i]
        fps_record = [eval_info[j]['fps'] for j in range(len(eval_info)) if eval_info[j]['id'] == i]
        response_record = [eval_info[j]['response_time'] for j in range(len(eval_info)) if eval_info[j]['id'] == i]
        accuracy_record = [eval_info[j]['accuracy'] for j in range(len(eval_info)) if eval_info[j]['id'] == i]
        zipped = zip(release_record, fps_record, accuracy_record)
        sorted_zip = sorted(zipped, key=lambda x:x[0])
        release_record, fps_record, accuracy_record = zip(*sorted_zip)
        release_records.append(release_record)
        fps_records.append(fps_record)
        accuracy_records.append(accuracy_record)
        response_records.append(response_record)
        for info in eval_info:
            if info['id'] == i:
                target_fps.append(info['target fps'])
                target_response.append(info['target response'])
                break
        dmr += sum(np.array(response_record) > target_response[i])
        num_jobs += len(response_record)
    print(dmr/num_jobs)
    start_time = [0, 75, 150, 225]
    end_time = [75, 150, 225, 300]
    for s, e in zip(start_time, end_time):
        total = 0
        miss = 0
        for i in ids:
            x = np.array(release_records[i])
            y = np.array(response_records[i])
            y1 = np.array(accuracy_records[i])

            y = y[x >= s]
            y1 = y1[x >= s]
            x = x[x >= s]


            y = y[x < e]
            y1 = y1[x < e]
            x = x[x < e]

            y1[y > target_response[i]] = 0

            miss += sum(y > target_response[i])
            total += len(y)
            if len(y) > 1:
                print(f'{i} - {len(y)} - {np.mean(y1)}')
        rate = miss / total if total != 0 else 0
        print(f'[{s}, {e}] - {miss} / {total} - {rate}')
        print('----------------------------------------------------')


    # fig = plt.figure(figsize=(12, 12))
    # for i in ids:
    #     ax1 = fig.add_subplot(len(ids), 2, i * 2 + 1)
    #     ax1.plot(release_records[i], response_records[i], linewidth=1)
    #     # ax1.scatter(eval_info[i]['drop'], [0] * len(eval_info[i]['drop']), color='red', marker='x')
    #     ax1.axhline(y=target_response[i], color='r', linestyle='--')
    #     ax1.set_xlabel('Time')
    #     ax1.set_ylabel('Response Time (ms)')
    #     ax1.set_ylim(0, int(target_response[i] * 1.2))
    #     # ax1.set_ylim(0, target_fps[i] * 1.5)
    #     ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    #     ax2 = fig.add_subplot(len(ids), 2, 2 * (i + 1))
    #     # ax2.plot(eval_info[i]['release'], eval_info[i]['accuracy'])
    #     ax2.plot(release_records[i], accuracy_records[i], linewidth=1)
    #     ax2.set_xlabel('Time')
    #     ax2.set_ylabel('Accuracy')
    #     ax2.set_ylim(85, 100)
    #     ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    #
    # fig.tight_layout()
    # plt.show()
    #
    # i = args.trace_id
    # trace = np.array([release_records[i], response_records[i], accuracy_records[i]])
    # print(trace.shape)
    # np.savetxt('trace.txt', trace)
    # release_record_i = np.array(release_records[i])
    # response_record_i = np.array(response_records[i])
    # accuracy_record_i = np.array(accuracy_records[i])
    # accuracy_record_i[response_record_i > target_response[i]] = 0
    # print(np.sum((response_record_i[release_record_i <= 150] > target_response[i])) / len(response_record_i[release_record_i <= 150]))
    # print(np.sum((response_record_i[release_record_i > 150] > target_response[i])) / len(response_record_i[release_record_i > 150]))
    # print(np.mean(accuracy_record_i[release_record_i <= 150]))
    # print(np.mean(accuracy_record_i[release_record_i > 150]))

