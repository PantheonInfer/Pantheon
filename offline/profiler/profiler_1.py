import torch
from torch import nn
from pytorch_lightning.metrics import Accuracy
import importlib
import os
import pandas as pd
from prettytable import PrettyTable
import socket
import struct
from .mAP import eval_mAP, eval_mAP_VOC, eval_mAP_FDDB, eval_mAP_Fire


class Profiler:
    def __init__(self, args):
        self.args = args
        self.num = len(os.listdir(self.args.weights))
        self.blocks = self._load_modules()
        self.records = dict()

    def _load_modules(self):
        blocks = []
        # branches = []
        for i in range(self.num):
            block = torch.jit.load(os.path.join(self.args.weights, 'block_{:02d}.pth'.format(i))).cuda()
            # branch = torch.jit.load(os.path.join(self.args.weights, 'branch_{:02d}.pth'.format(i))).cuda()
            block.eval()
            # branch.eval()
            blocks.append(block)
            # branches.append(branch)
        # return blocks, branches
        return blocks

    def profile_latency(self):
        configs = importlib.import_module(f'dnn.{self.args.task}.configs')
        self.records['block latency'] = []
        # self.records['branch latency'] = []
        # 先确定每个block和branch的输入大小
        block_input_size = []
        branch_input_size = []
        dummy_input = torch.randn([1] + configs.INPUT_SHAPE).cuda()
        for i in range(self.num):
            block_input_size.append(list(dummy_input.shape))
            dummy_input = self.blocks[i](dummy_input)
            branch_input_size.append(list(dummy_input.shape))

        # 连接到device上
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.args.host, self.args.port))
        server_socket.listen(1)
        print(f'Waiting for connection on {self.args.host}:{self.args.port}')
        client_socket, client_address = server_socket.accept()
        print(f'Connected by {client_address[0]}:{client_address[1]}')

        # 发送模型文件和输入大小，接受device测量的latency
        for i in range(self.num):
            print(f'Measuring latency of block {i}')
            input_size = block_input_size[i]
            client_socket.sendall(struct.pack('i', len(input_size)))
            for j in input_size:
                client_socket.sendall(struct.pack('i', j))

            file_path = os.path.join(self.args.weights, 'block_{:02d}.pth'.format(i))
            file_size = os.path.getsize(file_path)
            # print(file_size // 1024, 'KB')
            client_socket.sendall(struct.pack('i', file_size))
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(self.args.buffer)
                    if not chunk:
                        break
                    client_socket.sendall(chunk)

            latency = struct.unpack('d', client_socket.recv(1024))[0]

            # print(f'execution time of block {i} is {latency} ms')
            self.records['block latency'].append(latency)

    def save(self):
        df = pd.DataFrame(self.records)
        save = os.path.join(self.args.weights, 'profile.csv')
        df.to_csv(save, index=False)
        print(f"Records are saved at {save}")

    def print(self):
        table = PrettyTable()
        table.field_names = list(self.records.keys())
        for i in range(len(self.records[table.field_names[0]])):
            row = [self.records[key][i] for key in self.records.keys()]
            table.add_row(row)
        table.float_format = '.4'
        print(table)