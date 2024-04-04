import sys
import os
sys.path.append('../..')
from third_party.nni.common.graph_utils import build_module_graph


class ComputationGraph:
    def __init__(self, model, dummy_input):
        self.module_graph = build_module_graph(model, dummy_input)
        self.name_to_node = self.module_graph.name_to_node

    def exist_node(self, node_name):
        return node_name in self.name_to_node

    def get_start_nodes(self):
        start_nodes = []
        for node_name in self.name_to_node:
            if len(self.get_pre_nodes(node_name)) == 0:
                start_nodes.append(node_name)
        return start_nodes

    def get_end_nodes(self):
        end_nodes = []
        for node_name in self.name_to_node:
            if len(self.get_next_nodes(node_name)) == 0:
                end_nodes.append(node_name)
        return end_nodes

    def get_next_nodes(self, node_name):
        return self.module_graph.find_successors(node_name)

    def get_pre_nodes(self, node_name):
        return self.module_graph.find_predecessors(node_name)

    def find_all_paths(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if not self.exist_node(start):
            return []
        paths = []
        for node in self.get_next_nodes(start):
            if node not in path:
                new_paths = self.find_all_paths(node, end, path)
                for p in new_paths:
                    paths.append(p)
        return paths

    def get_auxiliary(self, node_name):
        return self.name_to_node[node_name].auxiliary

    def get_op_type(self, node_name):
        return self.name_to_node[node_name].op_type

    def get_type(self, node_name):
        return self.name_to_node[node_name].type

    def is_conv_fc(self, node_name):
        return self.get_op_type(node_name) in ['Conv2d', 'Linear']

    def size(self):
        return len(self.name_to_node)


if __name__ == '__main__':
    import torch
    from torchvision.models.resnet import resnet18, resnet50
    from torchvision.models.inception import inception_v3

    model = resnet50()
    block_manager = ComputationGraph(model, torch.randn((1, 3, 224, 224)))
    paths = block_manager.find_all_paths(block_manager.get_start_nodes()[0], block_manager.get_end_nodes()[0])
    for p in paths:
        print(p)
    print(len(paths))