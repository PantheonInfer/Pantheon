import torch
from torch import nn
import numpy as np
import copy
import importlib
from .block_extraction.block import Block
from .block_extraction.computation_graph import ComputationGraph
# from .block_extraction.graph_partition import graph_partition
from .block_extraction.graph_partition_no_merge import graph_partition


class EEN(nn.Module):
    def __init__(self, model, dummy_input, task, **kwargs):
        super(EEN, self).__init__()
        self.task = task
        graph = ComputationGraph(model, dummy_input)
        logic_blocks = graph_partition(graph, start_node=kwargs['start_node'] if 'start_node' in kwargs else None,
                                      end_node=kwargs['end_node'] if 'end_node' in kwargs else None)
        self.blocks = [Block(model, graph, nodes_in_block) for nodes_in_block in logic_blocks]

        # Make sure that the divided model has the same output with original model
        model.eval()
        original_out = model(dummy_input)
        new_out = copy.deepcopy(dummy_input)
        for i, block in enumerate(self.blocks):
            block.eval()
            new_out = block(new_out)
        for i in range(len(new_out)):
            assert torch.all(new_out[i] == original_out[i])

        self.branches = self.insert_branches(dummy_input, **kwargs)
        # 最后一个block也是一个exit branch
        self.branches.append(self.blocks[-1])
        self.blocks.pop(-1)

        self.blocks = nn.ModuleList(self.blocks)
        self.branches = nn.ModuleList(self.branches)

        self.exit_idx = -1

        self.active_branches = np.ones(len(self.branches), dtype=bool)

    def insert_branches(self, dummy_input, **kwargs):
        branch_lib = importlib.import_module(f'een.branch.{self.task}')
        intermediate = copy.deepcopy(dummy_input)
        branches = []
        i = 0
        for block in self.blocks[:-2]:
            block.eval()
            intermediate = block(intermediate)
            i += 1
            branches.append(branch_lib.build_exit_branch(intermediate.shape[-1], intermediate.shape[1], num_classes=kwargs['num_classes']))

        return branches

    def deactivate_branch(self, branch_index):
        self.active_branches[branch_index] = False

    def activate_branch(self, branch_index):
        self.active_branches[branch_index] = True

    def activate_all_branch(self):
        self.active_branches = np.ones(len(self.branches), dtype=bool)

    def forward(self, x):
        if self.exit_idx == -1:
            outputs = []
            for i, block in enumerate(self.blocks):
                x = block(x)
                if self.active_branches[i] == True:
                    outputs.append(self.branches[i](x))

            return outputs
        else:
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i == self.exit_idx:
                    return self.branches[i](x)

    def set_exit_idx(self, idx):
        assert idx < len(self.branches)
        self.exit_idx = idx

    def save(self, path):
        for i, block in enumerate(self.blocks):
            block_path = '{}_block_{}.pt'.format(path, i)
            torch.save(block.state_dict(), block_path)
        for i, branch in enumerate(self.branches):
            branch_path = '{}_branch_{}.pt'.format(path, i)
            torch.save(branch.state_dict(), branch_path)

    def load(self, path):
        for i, block in enumerate(self.blocks):
            block_path = '{}_block_{}.pt'.format(path, i)
            self.blocks[i].load_state_dict(torch.load(block_path))
        for i, branch in enumerate(self.branches):
            branch_path = '{}_branch_{}.pt'.format(path, i)
            self.branches[i].load_state_dict(torch.load(branch_path))

    def freeze_backbone(self):
        for param in self.blocks.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.blocks.parameters():
            param.requires_grad = True

