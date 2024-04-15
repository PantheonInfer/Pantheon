#include <thread>
#include <string>
#include <vector>
#include <mutex>
#include <algorithm>
#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "model_repository_manager.h"
#include "utils.h"
#include "memory_manager.h"
#include "nnjob.h"

#ifndef _H_RUNTIME_
#define _H_RUNTIME_

class Runtime {
private:
    unsigned int last_job_id;
    std::vector<NNJob> scheduling_queue;
    std::vector<NNJob> executing_queue;
    MemoryManager memory_manager;

    bool running, high_priority_worker_ready, low_priority_worker_ready;

    std::thread high_priority_worker_thread, low_priority_worker_thread;

    std::mutex thread_ready_mtx, executing_queue_mtx;
    std::condition_variable executing_queue_cond, thread_ready_cond;
public:
    Runtime();
    void exec(std::string model_name, long long release, long long deadline, torch::jit::IValue input);
    void scheduler(NNJob job);
    void high_priority_worker();
    void low_priority_worker();
    void start();
    void stop();
};

#endif