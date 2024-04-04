from collections import deque
from .computation_graph import ComputationGraph


def graph_partition(graph, start_node=None, end_node=None):
    start_nodes = graph.get_start_nodes()
    end_nodes = graph.get_end_nodes()
    # for name, node in graph.name_to_node.items():
    #     print(name, graph.get_op_type(name))
    # TODO: 只支持只有一个输入和一个输出的网络结构，以后如果需要支持更多的结构再实现指定start和end node
    assert len(start_nodes) == 1
    if end_node == None:
        assert len(end_nodes) == 1
        paths = graph.find_all_paths(start_nodes[0], end_nodes[0])
    else:
        paths = graph.find_all_paths(start_nodes[0], end_node) # 如果网络有多个end nodes则需要指定end node

    # core nodes是所有路径的交点，只有在core nodes处将网络分割才有可能保证划分完的网络blocks构成一个单链的结构
    core_nodes = intersection(paths)
    # 在core nodes中有多个next nodes的nodes就是split nodes
    # split_nodes = graph_split_nodes_search(graph, core_nodes)
    split_nodes = core_nodes[:]
    # 以split nodes为划分点进行blocks划分，split nodes被分给它们的上一个块，所以从每个split node往前找
    blocks = []
    # 需要包括输出层
    # if len(end_nodes) == 1:
    #     split_nodes += [end_nodes[0]]
    # else:
    #     split_nodes += ['fake_end_node']
    for split_node in split_nodes:
        block = []
        block_search(graph, split_node, split_nodes, block)
        if 'fake_end_node' in block:
            block.remove('fake_end_node')
        blocks.append(block)

    # 为了尽可能划分的更小如果存在block有多个conv/fc，且其中有conv/fc是core node就从以这些是core node的cong/fc再进行划分
    # mark = True
    # while mark:
    #     mark = False
    #     for i, block in enumerate(blocks):
    #         block_split_node = block_split_node_search(graph, block, core_nodes)
    #         if block_split_node != None:
    #             blocks.pop(i)
    #             blocks += split_block(graph, block, block_split_node)
    #             mark = True
    #             break
    # 按照原来网络中的顺序排序
    blocks = sort_blocks(graph, blocks)
    return blocks


def sort_blocks(graph, blocks):
    sorted_blocks = []
    block = get_start_block(graph, blocks)
    while block != None:
        sorted_blocks.append(block)
        block = get_next_block(graph, block, blocks)
    assert len(sorted_blocks) == len(blocks)
    return sorted_blocks


def get_next_block(graph, block, blocks):
    for node in block:
        next_nodes = graph.get_next_nodes(node)
        next_nodes_not_in_block = list(set(next_nodes) - set(block))
        for next_node in next_nodes_not_in_block:
            for b in blocks:
                if next_node in b:
                    return b
    return None


def get_start_block(graph, blocks):
    for block in blocks:
        for node in block:
            if len(graph.get_pre_nodes(node)) == 0:
                return block


def split_block(graph, block, block_split_node):
    splited_blocks = [[]]
    queue = deque([block_split_node])
    visited = set()
    while queue:
        node = queue.popleft()
        splited_blocks[0].append(node)
        if node in visited:
            continue
        visited.add(node)
        next_nodes = graph.get_next_nodes(node)
        for next_node in next_nodes:
            if next_node not in visited and next_node in block:
                queue.append(next_node)
    splited_blocks.append(list(set(block) - set(splited_blocks[0])))
    return splited_blocks


def block_split_node_search(graph, block, core_nodes):
    # 如果这个block中有conv/fc是core nodes且它的祖先节点有别的conv/fc层那么它可以作为这个block的一个split node
    # 每次查找只返回一个，无论有几个
    for cur_node in block:
        if graph.is_conv_fc(cur_node) and cur_node in core_nodes:
            # bfs
            queue = deque([cur_node])
            visited = set()
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                pre_nodes = graph.get_pre_nodes(node)
                for pre_node in pre_nodes:
                    if pre_node not in visited and pre_node in block:
                        queue.append(pre_node)
                        if graph.is_conv_fc(pre_node):
                            return cur_node
    return None


def block_search(graph, current_node, split_nodes, block=[]):
    if current_node in split_nodes and block != []:
        pass
    else:
        if current_node not in block:
            block.append(current_node)
            if current_node != 'fake_end_node':
                next_nodes = graph.get_pre_nodes(current_node)
            else:
                next_nodes = graph.get_end_nodes()
            for node in next_nodes:
                block = block_search(graph, node, split_nodes, block)
    return block


def graph_split_nodes_search(graph, core_nodes):
    split_nodes = []
    for node in core_nodes:
        next_nodes = graph.get_next_nodes(node)
        if len(next_nodes) > 1:
            split_nodes.append(node)
    return split_nodes


def intersection(lists):
    if len(lists) == 1:
        return lists[0]
    out = set(lists[0])
    for l in lists[1:]:
        out = out & set(l)
    return list(out)


if __name__ == '__main__':
    import torch
    # from torchvision.models.resnet import resnet18
    from torchvision.models.vgg import vgg16
    # from torchvision.models.inception import inception_v3
    # from torchvision.models.mobilenet import mobilenet_v2

    # model = resnet18()
    model = vgg16()
    # model =mobilenet_v2()
    graph = ComputationGraph(model, torch.randn((1, 3, 224, 224)))
    # for node_name in graph.name_to_node.keys():
    #     print(node_name, graph.get_op_type(node_name))
    blocks = graph_partition(graph)
    for block in blocks:
        print('========================================')
        # print(block)
        for node in block:
            print(node, graph.get_op_type(node))
