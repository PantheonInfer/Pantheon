import warnings

import torch
from torch.nn import Module, ModuleDict
import copy

MULTI_OUTPUT_TYPE = ['prim::TupleUnpack']


class Block(Module):
    def __init__(self, original_model, graph, nodes_in_block):
        super(Block, self).__init__()
        self.nodes_in_block = nodes_in_block
        self.graph = graph

        # model = copy.deepcopy(original_model)
        model = original_model
        defined_module_name = []

        for node_name in self.nodes_in_block:
            if self.graph.get_type(node_name) != 'module':
                continue

            module = get_module(model, node_name)
            # if self.graph.get_op_type(node_name) != 'ReLU':
            #     module = get_module(model, node_name)
            # else:
            #     module = torch.nn.ReLU(inplace=False)

            if not module:
                node_name = '.'.join(node_name.split('.')[0:-1])
                module = get_module(model, node_name)
                if module is None:
                    # 报错 没有在模型中找到这样的块
                    print('ERROR: can not find nn.Module named {}'.format(node_name))

            module_name_segment = node_name.split('.')
            if len(module_name_segment) == 1:
                exec('self.{} = module'.format(node_name))
            else:
                cur_dict = None
                cur_module_name = ''
                for i, name_segment in enumerate(module_name_segment[:-1]):
                    cur_module_name = '.'.join(
                        [cur_module_name, name_segment]) if cur_module_name != '' else name_segment
                    if cur_module_name in defined_module_name:
                        if i == 0:
                            cur_dict = eval('self.{}'.format(name_segment))
                        else:
                            cur_dict = cur_dict[name_segment]
                    elif i == 0:
                        exec('self.{} = ModuleDict()'.format(name_segment))
                        defined_module_name.append(name_segment)
                        cur_dict = eval('self.{}'.format(name_segment))
                    else:
                        cur_dict.update({name_segment: ModuleDict()})
                        cur_dict = cur_dict[name_segment]
                        defined_module_name.append(cur_module_name)
                cur_dict.update({module_name_segment[-1]: module})
                cur_module_name = ''
                cur_dict = None

        self.reliance = {}
        for i, node in enumerate(self.nodes_in_block):
            pre_nodes = self.graph.get_pre_nodes(node)
            pre_tensors = []
            pre_nodes_in_block = list(set(pre_nodes) & set(self.nodes_in_block))
            if len(pre_nodes_in_block) != len(pre_nodes) or len(pre_nodes) == 0: # 有pre_node不在当前block中或者没有pre_node
                pre_tensors.append('x') # rely on input tensor
            if len(pre_nodes_in_block) != 0:
                for pre_node in pre_nodes_in_block:
                    assert self.graph.get_op_type(node) not in MULTI_OUTPUT_TYPE
                    # pre_tensor = '{}.{}'.format(pre_node, self.graph.get_next_nodes(pre_node).index(node)) if self.graph.get_op_type(node) in MULTI_OUTPUT_TYPE else pre_node
                    pre_tensor = pre_node
                    pre_tensors.append(pre_tensor)
            self.reliance.update({node: pre_tensors})
        # print(self.reliance)

        # 拓扑排序得到一个可行的执行顺序保证所有layer执行前它需要的输入都准备好了
        indegrees = {node: 0 for node in self.nodes_in_block}
        for node in self.nodes_in_block:
            next_nodes = self.graph.get_next_nodes(node)
            for next_node in next_nodes:
                if next_node in self.nodes_in_block:
                    indegrees[next_node] += 1

        self.sorted_nodes_in_block = []
        zero_indegree_nodes = [node for node in self.nodes_in_block if indegrees[node] == 0]
        while zero_indegree_nodes:
            node = zero_indegree_nodes.pop()
            self.sorted_nodes_in_block.append(node)
            next_nodes = self.graph.get_next_nodes(node)
            for next_node in next_nodes:
                if next_node in self.nodes_in_block:
                    indegrees[next_node] -= 1
                    if indegrees[next_node] == 0:
                        zero_indegree_nodes.append(next_node)

        assert len(self.sorted_nodes_in_block) == len(self.nodes_in_block)

    def forward(self, x):
        output_tensors = {'x': x}
        for node in self.sorted_nodes_in_block:
            module = self.get_module_by_name(node) if self.graph.get_type(node) == 'module' else None
            if self.graph.get_type(node) == 'module':
                # 假定所有module都只有一个输入
                output_tensor = module(output_tensors[self.reliance[node][0]])
            elif self.graph.get_type(node) == 'func':
                op_type = self.graph.get_op_type(node)
                input_tensors = [output_tensors[pre_node] for pre_node in self.reliance[node]]
                input_list = [pre_node for pre_node in self.reliance[node]]
                if op_type in ['aten::add', 'aten::add_']:
                    output_tensor = self._handle_add(input_tensors)
                elif op_type == 'aten::flatten':
                    output_tensor = self._handle_flatten(input_tensors, node)
                elif op_type == 'aten::view':
                    output_tensor = self._handle_view(input_tensors, node)
                elif op_type == 'aten::adaptive_avg_pool2d':
                    output_tensor = self._handle_adaptive_avg_pool2d(input_tensors, node)
                elif op_type in ['aten::size', 'aten::Int']:
                    output_tensor = None
                elif op_type == 'aten::reshape':
                    output_tensor = self._handle_reshape(input_tensors, node)
                elif op_type == 'aten::permute':
                    output_tensor = self._handle_permute(input_tensors, node)
                elif op_type == 'aten::contiguous':
                    output_tensor = self._handle_contiguous(input_tensors)
                elif op_type in ['aten::relu', 'aten::relu_']:
                    output_tensor = self._handle_relu(input_tensors)
                elif op_type == 'aten::cat':
                    output_tensor = self._handle_cat(input_tensors, input_list, node)
                elif op_type == 'aten::mean':
                    output_tensor = self._handle_mean(input_tensors, node)
                else:
                    print(node, op_type)
                    raise NotImplementedError
            else:
                raise NotImplementedError
            output_tensors.update({node: output_tensor})
        network_end_nodes = self.graph.get_end_nodes()
        if network_end_nodes[0] in output_tensors: # 是网络的最后一个block
            out = [output_tensors[end_node] for end_node in network_end_nodes]
            if len(out) == 1:
                return out[0]
            else:
                return out
        return output_tensor

    def get_module_by_name(self, name):
        module_name_segment = name.split('.')
        if len(module_name_segment) == 1:
            return eval('self.{}'.format(name))
        else:
            module_dict = eval('self.{}'.format(module_name_segment[0]))
            if not isinstance(module_dict, ModuleDict):
                return module_dict
            for name_segment in module_name_segment[1:]:
                module = module_dict[name_segment]
                if not isinstance(module, ModuleDict):
                    return module
                module_dict = module

    def _handle_add(self, tensor_list):
        # print('add tensor_list len: {}'.format(len(tensor_list)))
        z = tensor_list[0]
        for x in tensor_list[1:]:
            z = z + x
        return z

    def _handle_flatten(self, tensor_list, node):
        # node中存有flatten需要的辅助信息
        x = tensor_list[0]
        auxiliary = self.graph.get_auxiliary(node)
        in_shape = auxiliary.get('in_shape')
        out_shape = auxiliary.get('out_shape')
        for i, num in enumerate(in_shape):
            if i == len(out_shape):
                start_dim = i - 1
                multi_result = out_shape[i-1]
                break
            if out_shape[i] != num:
                start_dim = i
                multi_result = out_shape[i]
                break
        multi = 1
        end_dim = start_dim
        for i in range(start_dim, len(in_shape)):
            # multi = multi*in_shape[i]
            multi *= in_shape[i]
            if multi == multi_result and len(in_shape)-(end_dim-start_dim) == len(out_shape):
                break
            # end_dim = end_dim + 1
            end_dim += 1
        # print('start dim: {}, end dim: {}'.format(start_dim, end_dim))
        return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)

    def _handle_view(self, tensor_list, node):
        for tensor in tensor_list:
            if tensor is not None:
                x = tensor
                break
        auxiliary = self.graph.get_auxiliary(node)
        out_shape = auxiliary.get('out_shape')
        real_batch_size = list(x.size())[0]
        out_shape[0] = real_batch_size
        return x.view(out_shape)

    def _handle_reshape(self, tensor_list, node):
        x = tensor_list[0]
        auxiliary = self.graph.get_auxiliary(node)
        return x.reshape([x.shape[0]] + auxiliary['out_shape'])

    def _handle_adaptive_avg_pool2d(self, tensor_list, node):
        x = tensor_list[0]
        auxiliary = self.graph.get_auxiliary(node)
        return torch.nn.functional.adaptive_avg_pool2d(x, auxiliary['out_shape'][2:])

    def _handle_permute(self, tensor_list, node):
        # auxiliary = self.graph.get_auxiliary(node)
        # in_shape = auxiliary.get('in_shape')
        # out_shape = auxiliary.get('out_shape')
        warnings.warn('Note that there may be a potential issue with the permute operation, as the ordering of dimensions is hard-coded.')
        x = tensor_list[0]
        return x.permute(0, 2, 3, 1)

    def _handle_contiguous(self, tensor_list):
        x = tensor_list[0]
        return x.contiguous()

    def _handle_relu(self, tensor_list):
        x = tensor_list[0]
        return torch.nn.functional.relu(x)

    def _handle_cat(self, tensor_list, input_list, node):
        # input_list是tensor_list中的顺序
        auxiliary = self.graph.get_auxiliary(node)
        cat_dim = auxiliary.get('cat_dim')
        in_order = auxiliary.get('in_order')  # cat的顺序

        # 建立名字与输入tensor的对应关系
        name_to_tensor = {}
        for i, name in enumerate(input_list):
            name_to_tensor.update({name: tensor_list[i]})
            # 若in_order中有input_list中没有的名字 则重新对应
        assert len(in_order) == len(input_list)
        # module_name_list = []
        # for module_name in in_order:
        #     if not module_name.startswith('.prim::TupleUnpack'):
        #         module_name_list.append(module_name)
        # new_input_list = []
        # for i, name in enumerate(input_list):
        #     if name not in module_name_list:
        #         new_input_list.append(name)
        # new_in_order = []
        # ii = 0
        # for module_name in in_order:
        #     if module_name in module_name_list:
        #         new_in_order.append(module_name)
        #     else:
        #         new_in_order.append(self._graph.order_to_node.get(new_input_list[ii][0]).get_name())
        #         ii = ii + 1

        # 根据in_order的顺序将tensor_list放入cat_list中
        cat_list = []
        # for module_name in new_in_order:
        for module_name in in_order:
            cat_list.append(name_to_tensor.get(module_name))
        return torch.cat(cat_list, dim=cat_dim)

    def _handle_mean(self, tensor_list, node):
        auxiliary = self.graph.get_auxiliary(node)
        in_shape = auxiliary['in_shape']
        out_shape = auxiliary['out_shape']
        warnings.warn(
            'Note that there may be a potential issue with the mean operation, as the dimensions to reduce is hard-coded.')
        return tensor_list[0].mean([2, 3])


def get_module(model, module_name):
    for name, module in model.named_modules():
        if name == module_name:
            return module
    return None


if __name__ == '__main__':
    import torch
    from torchvision.models.resnet import resnet18
    from torchvision.models.vgg import vgg16
    from torchvision.models.inception import inception_v3
    from computation_graph import ComputationGraph
    from graph_partition import graph_partition

    model = resnet18()
    # model = vgg16()

    graph = ComputationGraph(model, torch.randn((1, 3, 224, 224)))
    logical_blocks = graph_partition(graph)
    blocks = []
    for lb in logical_blocks:
        b = Block(model, graph, lb)
        blocks.append(b)

    x1 = torch.randn((1, 3, 224, 224))
    x2 = copy.deepcopy(x1)
    y1 = model(x1)

    for b in blocks:
        x2 = b(x2)
        # print(x2)

    assert torch.all(y1 == x2)

