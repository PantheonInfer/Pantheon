#include <mutex>
#include "model_repository_manager.h"
#include "tensor_pool.h"
#include "nnjob.h"
#include "utils.h"

#ifndef _H_MEMORY_MANAGER_
#define _H_MEMORY_MANAGER_

class MemoryManager {
private:
    TensorPool tensors;
    std::mutex mtx;
public:
    MemoryManager();
    bool fetch(NNJob job, torch::jit::IValue &tensor);
    bool fetch(NNJob job, int block_id, torch::jit::IValue &tensor);
    bool check(NNJob &job);
    bool empty(NNJob job);
    void erase(NNJob job);
    void erase(NNJob job, int block_id);
    void add(NNJob &job, torch::jit::IValue tensor);
};

#endif