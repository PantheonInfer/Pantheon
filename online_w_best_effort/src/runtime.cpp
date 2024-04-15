#include "runtime.h" 

Runtime::Runtime() {
    last_job_id = 0;
    running = false;
    high_priority_worker_ready = false;
    low_priority_worker_ready = false;
}

void Runtime::exec(std::string model_name, long long release, long long deadline, torch::jit::IValue input) {
    int exit_id = MRM[model_name].get_num_exits() - 1;
    int last_block_id = MRM[model_name].get_last_block_id(exit_id);
    NNJob job(last_job_id, model_name, release, deadline, exit_id, last_block_id);
    memory_manager.add(job, input);
    last_job_id = (last_job_id + 1) % UINT_MAX;
    scheduler(job);
}

void Runtime::scheduler(NNJob job) {

    std::unique_lock<std::mutex> executing_queue_lock(executing_queue_mtx);
    scheduling_queue = executing_queue;
    executing_queue_lock.unlock();

    scheduling_queue.push_back(job);
    
    auto start = NOW();
    spdlog::info("[SCHE:START] {}", start);

    std::stable_sort(scheduling_queue.begin(), scheduling_queue.end());
    // reset
    for (int i = 0; i < scheduling_queue.size(); i++) {
        scheduling_queue[i].exit_id = MRM[scheduling_queue[i].model_name].get_num_exits() - 1;
        scheduling_queue[i].last_block_id = MRM[scheduling_queue[i].model_name].get_last_block_id(scheduling_queue[i].exit_id);
    }

    int skipped = -1;
        
    while (true) {
            
        bool solved = true;
        int missing = scheduling_queue.size() - 1;
            
        auto time = NOW();
        for (int i = 0; i < scheduling_queue.size(); i++) {
            std::string model_name = scheduling_queue[i].model_name;
            int current_block_id = scheduling_queue[i].current_block_id;
            int exit_id = scheduling_queue[i].exit_id;
            time += MRM[model_name].get_submodel_latency(current_block_id, exit_id);
            long long deadline = scheduling_queue[i].deadline;
            if (time > deadline && i > skipped) {
                missing = i;
                solved = false;
                break;
            }
        }

        float min_accuracy_drop = 100;
        int min_accuracy_drop_id = -1;
        if (!solved) {
            for (int i = 0; i <= missing; i++) {
                if (scheduling_queue[i].exit_id - 1 >= 0) {
                    std::string model_name = scheduling_queue[i].model_name;
                    int num_exits = MRM[model_name].get_num_exits();
                    int exit_id = scheduling_queue[i].exit_id;

                    float accuracy_drop = MRM[model_name].get_accuracy(num_exits - 1) - MRM[model_name].get_accuracy(exit_id - 1);
                    if (accuracy_drop < min_accuracy_drop) {
                        min_accuracy_drop = accuracy_drop;
                        min_accuracy_drop_id = i;
                    }
                }
            }
        }
        else {
            break;
        }
            
        if (min_accuracy_drop_id != -1) {
            scheduling_queue[min_accuracy_drop_id].exit_id -= 1;
            std::string model_name = scheduling_queue[min_accuracy_drop_id].model_name;
            scheduling_queue[min_accuracy_drop_id].last_block_id = MRM[model_name].get_last_block_id(scheduling_queue[min_accuracy_drop_id].exit_id);
        }
        else {
            skipped = missing;
        }
    }
    auto end = NOW();
    spdlog::info("[SCHE] {} {} {}", end, end - start, scheduling_queue.size());
    executing_queue_lock.lock();
    executing_queue.assign(scheduling_queue.begin(), scheduling_queue.end());
    executing_queue_lock.unlock();
    executing_queue_cond.notify_all();
}

void Runtime::high_priority_worker() {
    // set_cpu_affinity(2);
    c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(true);
    {
        c10::cuda::CUDAStreamGuard guard(stream);
        torch::InferenceMode inference_guard(true);
        {
            std::lock_guard thread_ready_lock(thread_ready_mtx);
            MRM.warmup_all();
            high_priority_worker_ready = true;
        }
        thread_ready_cond.notify_one();
        spdlog::info("[EXEC:START] HIGH_PRIORITY {} ", NOW());

        while (running || !executing_queue.empty()) {
            std::unique_lock<std::mutex> executing_queue_lock(executing_queue_mtx);
            while (executing_queue.empty()) {
                spdlog::info("[EXEC:SLEEP] {} ", NOW());
                executing_queue_cond.wait(executing_queue_lock);
                spdlog::info("[EXEC:AWAKE] {} ", NOW());
            }
            NNJob job = executing_queue[0];
            executing_queue_lock.unlock();
            auto start = NOW();
            spdlog::info("[EXEC:BEGIN] {} {}", start, executing_queue.size());
            if (job.deadline <= start) {
                // alread exceed deadline
                memory_manager.erase(job);
                executing_queue_lock.lock();
                executing_queue.erase(std::remove_if(executing_queue.begin(), executing_queue.end(), [&](NNJob e) { return e.id == job.id; }), executing_queue.end());
                executing_queue_lock.unlock();

                auto end = job.deadline + 1000000;
                auto release = job.release;
                auto job_id = job.id;
                auto exit_id = job.exit_id;
                auto acc = MRM[job.model_name].get_accuracy(job.exit_id);
                spdlog::info("[EXEC:EXIT] DROP {} {} {} {} {} {} {}", release, end, end - release, 0, job_id, exit_id, acc);
            }
            else if (job.current_block_id <= job.last_block_id) {
                if (memory_manager.check(job)) {
                    torch::jit::IValue input;
                    if (memory_manager.fetch(job, input)) {
                        torch::jit::IValue output = MRM[job.model_name].get_module(false, job.current_block_id).forward({input});
                        memory_manager.add(job, output);
                        auto end = NOW();
                        // MRM[job.model_name].update_latency(job.current_block_id, false, end - start);
                        spdlog::info("[EXEC:BLOCK] {} HIGH_PRIORITY {} {} {}", end, end - start, job.id, job.current_block_id - 1);
                        // block_id指向未执行的block，block_id - 1指向它的输入，block_id - 2指向上一个block的输入
                        memory_manager.erase(job, job.current_block_id - 2);
                            
                    }
                    else {
                        job.current_block_id += 1;
                        if (memory_manager.empty(job)) {
                            executing_queue_lock.lock();
                            executing_queue.erase(std::remove_if(executing_queue.begin(), executing_queue.end(), [&](NNJob e) { return e.id == job.id; }), executing_queue.end());
                            executing_queue_lock.unlock();
                        }
                    }
                }
                executing_queue_lock.lock();
                for (int i = 0; i < executing_queue.size(); i++){
                    if (executing_queue[i].id == job.id) {
                        executing_queue[i].current_block_id = (executing_queue[i].current_block_id < job.current_block_id) ? job.current_block_id : executing_queue[i].current_block_id;
                        break;
                    }
                }
                executing_queue_lock.unlock();
            }
            else {
                auto start = NOW();
                torch::jit::IValue input;
                if (memory_manager.fetch(job, job.last_block_id, input)) {
                    torch::jit::IValue output = MRM[job.model_name].get_module(true, job.exit_id).forward({input});
                    auto end = NOW();
                    // MRM[job.model_name].update_latency(job.exit_id, true, end - start);
                    spdlog::info("[EXEC:EXIT] HIGH_PRIORITY {} {} {} {} {} {} {}", job.release, end, end - job.release, end - start, job.id, MRM[job.model_name].get_last_block_id(job.exit_id), MRM[job.model_name].get_accuracy(job.exit_id));
                }

                memory_manager.erase(job);

                executing_queue_lock.lock();
                executing_queue.erase(std::remove_if(executing_queue.begin(), executing_queue.end(), [&](NNJob e) { return e.id == job.id; }), executing_queue.end());
                executing_queue_lock.unlock();
            }
        }
    }
    spdlog::info("[EXEC:STOP] HIGH_PRIORITY {} ", NOW());
}

void Runtime::start() {
    running = true;
    // set_cpu_affinity(0);

    high_priority_worker_thread = std::thread(&Runtime::high_priority_worker, this);

    {
        std::unique_lock<std::mutex> thread_ready_lock(thread_ready_mtx);
        thread_ready_cond.wait(thread_ready_lock, [&] { return high_priority_worker_ready;});
    }
    std::cout << "All threads are ready" << std::endl;
}

void Runtime::stop() {
    running = false;
    high_priority_worker_thread.join();
}