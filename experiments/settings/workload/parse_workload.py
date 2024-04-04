import json
import argparse
import math
import os.path
import random

import matplotlib.pyplot as plt
import workload_pb2

def parse_one(workload: dict, id):

    assert workload['deadline'] <= workload['period']
    current_time = (workload['start'] // 33333) * 33333
    # current_time = workload['start']
    job_sequence = []

    i = 0
    while current_time < workload['end']:

        random_start_delay = random.randint(-1000, 1000)
        if current_time + random_start_delay < 0:
            random_start_delay = 0
        job_sequence.append({
            'model_name': workload['model_name'],
            'release': current_time + random_start_delay,
            'deadline': current_time + workload['deadline'] + random_start_delay,
            'shape': workload['shape'],
            'id': id
        })
        current_time += workload['period']
        i += 1
    return job_sequence


def merge_all(job_sequences):
    merged_job_sequence = []
    num_jobs = sum([len(seq) for seq in job_sequences])
    current_idx = [0 for i in range(len(job_sequences))]
    for i in range(num_jobs):
        earliest = math.inf
        earliest_idx = -1
        for j in range(len(job_sequences)):
            if current_idx[j] < len(job_sequences[j]):
                if job_sequences[j][current_idx[j]]['release'] < earliest:
                    earliest_idx = j
                    earliest = job_sequences[j][current_idx[j]]['release']
        merged_job_sequence.append(job_sequences[earliest_idx][current_idx[earliest_idx]])
        current_idx[earliest_idx] += 1
    return merged_job_sequence


def viz(job_sequences):
    fig, ax = plt.subplots(figsize=(12, 2))
    for i, seq in enumerate(job_sequences):
        for job in seq:
            ax.barh(i, width=job['deadline'] - job['release'], left=job['release'], color="blue", edgecolor='black')
    ax.set_yticks(range(len(job_sequences)))
    ax.set_yticklabels([f'Job {i}: {job_sequences[i][0]["model_name"]}' for i in range(len(job_sequences))])
    plt.show()


if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='robot_3.json')
    # parser.add_argument('--config', type=str, default='uav_new.json')
    parser.add_argument('--config', type=str, default='robot_new.json')
    # parser.add_argument('--config', type=str, default='robot_new.json')
    parser.add_argument('--viz', default=False)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    workloads_config = config['workloads']
    workloads = []
    # for i, wc in enumerate(workloads_config):
    #     workloads.append(parse_one(wc, i))
    #

    #
    # merged_workloads = merge_all(workloads)
    viz_workloads = []
    for i, wc in enumerate(workloads_config):
        viz_workloads.append(parse_one(wc, i))
        workloads += viz_workloads[-1]

    if args.viz:
        viz(viz_workloads)

    merged_workloads = sorted(workloads, key=lambda x:x['release'])

    workloads_pb = workload_pb2.Workloads()
    for w in merged_workloads:
        w_pb = workload_pb2.Workload()
        w_pb.model_name = w['model_name']
        w_pb.release = w['release']
        w_pb.deadline = w['deadline']
        w_pb.shape[:] = w['shape']
        w_pb.id = w['id']
        workloads_pb.workload.append(w_pb)

    serialized_workloads = workloads_pb.SerializeToString()
    output_file = '.'.join(args.config.split('.')[:-1]) + '.bin'
    with open(output_file, 'wb') as f:
        f.write(serialized_workloads)
