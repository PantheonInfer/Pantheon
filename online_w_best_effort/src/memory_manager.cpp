#include "memory_manager.h"

MemoryManager::MemoryManager() {
    tensors.max_load_factor(0.5);
    tensors.rehash(503);
}

bool MemoryManager::fetch(NNJob job, torch::jit::IValue &tensor) {
    std::lock_guard<std::mutex> lock(mtx);
    // 待执行的block输入的id应该减1
    // 比如第0个block的输入id是-1
    if (tensors.find({job.id, job.current_block_id - 1}) != tensors.end()) {
        tensor = tensors[{job.id, job.current_block_id - 1}].toTensor().clone();
        return true;
    }
    else {
        return false;
    }
}

bool MemoryManager::fetch(NNJob job, int block_id, torch::jit::IValue &tensor) {
    std::lock_guard<std::mutex> lock(mtx);
    if (tensors.find({job.id, block_id}) != tensors.end()) {
        tensor = tensors[{job.id, block_id}].toTensor().clone();
        return true;
    }
    else {
        return false;
    }
}

bool MemoryManager::check(NNJob &job) {
    // 检查是否需要执行当前的block
    std::lock_guard<std::mutex> lock(mtx);
    bool mark = true;
    for (int i = job.current_block_id; i < MRM[job.model_name].get_num_blocks(); i++) {
        auto it = tensors.find({job.id, i});
        if (it != tensors.end()) {
            job.current_block_id = i + 1;
            mark = false;
        }
    }
    return mark;
}

bool MemoryManager::empty(NNJob job) {
    std::lock_guard<std::mutex> lock(mtx);
    bool mark = true;
    for (int i = 0; i < MRM[job.model_name].get_num_blocks(); i++) {
        auto it = tensors.find({job.id, i});
        if (it != tensors.end()) {
            mark = false;
            break;
        }
    }
    return mark;
}

void MemoryManager::erase(NNJob job) {
    std::lock_guard<std::mutex> lock(mtx);
    for (int i = -1; i < MRM[job.model_name].get_num_blocks(); i++) {
        auto it = tensors.find({job.id, i});
        if (it != tensors.end()) {
            tensors.erase(it);
        }
    }
}

void MemoryManager::erase(NNJob job, int block_id) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!MRM[job.model_name].is_key_block(block_id)) {
        tensors.erase({job.id, block_id});
    }
}

void MemoryManager::add(NNJob &job, torch::jit::IValue tensor) {
    std::lock_guard<std::mutex> lock(mtx);
    if ((job.current_block_id == -1) || (tensors.find({job.id, job.current_block_id - 1}) != tensors.end())) {
        tensors[{job.id, job.current_block_id}] = tensor;
        job.current_block_id += 1;
    }
}