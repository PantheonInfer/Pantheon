#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <chrono>
#include <vector>
#include <thread>
#include <mutex>
#include <filesystem>
#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "model_repository_manager.h"
#include "memory_manager.h"
#include "runtime.h"
#include "utils.h"
#include "workload.pb.h"

std::atomic<bool> monitor_running(false);
std::atomic<bool> best_effort_running(false);
std::mutex best_effort_start_mtx;
std::condition_variable best_effort_start_cond;

float get_gpu_util() {
    std::ifstream gpuUtilFile("/sys/devices/gpu.0/load");
    if (!gpuUtilFile) {
        std::cerr << "Failed to open GPU utilization file." << std::endl;
        exit(1);
    }
    std::string gpuUtilStr;
    if (!std::getline(gpuUtilFile, gpuUtilStr)) {
        std::cerr << "Failed to read GPU utilization." << std::endl;
        exit(1);
    }
    return std::stof(gpuUtilStr) / 10.0f;
}

void monitor() {
    while (monitor_running) {
        auto time = NOW();
        std::ifstream currFile("/sys/bus/i2c/drivers/ina3221/7-0040/hwmon/hwmon5/curr1_input");
        std::ifstream voltFile("/sys/bus/i2c/drivers/ina3221/7-0040/hwmon/hwmon5/in1_input");
        if (!currFile) {
            std::cerr << "Failed to open Current file." << std::endl;
            exit(1);
        }
        if (!voltFile) {
            std::cerr << "Failed to open Voltage file." << std::endl;
            exit(1);
        }
        std::string currStr;
        if (!std::getline(currFile, currStr)) {
            std::cerr << "Failed to read Current." << std::endl;
            exit(1);
        }
        std::string voltStr;
        if (!std::getline(voltFile, voltStr)) {
            std::cerr << "Failed to read Current." << std::endl;
            exit(1);
        }
        float gpuUtil = get_gpu_util();
        float curr = std::stof(currStr) / 1000.0f;
        float volt = std::stof(voltStr) / 1000.0f;
        spdlog::info("[GPU] {} {} {}", time, gpuUtil, curr * volt);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

void best_effort_runtime(int fps) {
    c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(false);
    {
        c10::cuda::CUDAStreamGuard guard(stream);
        torch::InferenceMode inference_guard(true);
        torch::jit::IValue dummy_input = torch::randn({1, 3, 224, 224}).to(torch::kCUDA);
        torch::jit::script::Module model = torch::jit::load("../../../../experiments/alexnet.pb");
        model.to(torch::kCUDA);
        model.eval();

        std::cout << "Warming up best effor task" << std::endl;
        for (int i = 0; i < 500; i++) {
            model.forward({dummy_input});
            AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        std::cout << "Best effort task ready" << std::endl;
        {
            std::unique_lock<std::mutex> best_effort_start_lock(best_effort_start_mtx);
            best_effort_start_cond.wait(best_effort_start_lock, [&] { return best_effort_running == true;});
        }
        std::cout << "Best effort task start" << std::endl;

        long long period = 1000000. / fps;
        while (best_effort_running) {
            auto begin = NOW();
            model.forward({dummy_input});
            AT_CUDA_CHECK(cudaStreamSynchronize(stream));
            auto end = NOW();
            SLEEP(end + (period - (end - begin))); // release task in xx FPS
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5 && argc != 4) {
        std::cerr << "Usage: ./runtime <path-to-model-repository> <path-to-workload-config> <release-frequency-of-BE> [path-to-log]" << std::endl;
        return -1;
    }
    std::filesystem::path path = argv[1];

    MRM.init(path);
    // MRM.warmup_all();
    
    Workloads workloads;
    std::ifstream input(argv[2], std::ios::binary);

    if (!workloads.ParseFromIstream(&input)) {
        std::cerr << "Failed to parse workloads." << std::endl;
        return -1;
    }

    std::thread monitor_t;
    
    if (argc == 5) {
        std::filesystem::path log_path = argv[4];

        if (std::filesystem::exists(log_path)) {
            std::filesystem::remove(log_path);
        }
        // auto logger = spdlog::basic_logger_mt("logger", log_path.string()); 
        auto logger = spdlog::basic_logger_mt<spdlog::async_factory>("logger", log_path.string());
        spdlog::set_default_logger(logger); 
        
        monitor_running = true;
        monitor_t = std::thread(monitor);
        monitor_t.detach();
    }
    else if (argc == 4) {
        auto logger = spdlog::stdout_color_mt("logger");
        spdlog::set_default_logger(logger); 
    }
    // spdlog::flush_every(std::chrono::seconds(1));

    std::thread best_effort_t(best_effort_runtime, std::stoi(argv[3]));
    best_effort_t.detach();

    Runtime runtime;
    runtime.start();

    std::this_thread::sleep_for(std::chrono::seconds(60));
    
    {
        std::lock_guard best_effort_start_lock(best_effort_start_mtx);
        best_effort_running = true;
    }
    best_effort_start_cond.notify_one();
    
    const Workload* p_w;

    auto start = NOW();
    spdlog::info("[START] {}", start);
    for (int i = 0; i < workloads.workload_size(); i++) {
        p_w = &workloads.workload(i);
        long long expected_release = start + p_w->release();
        SLEEP(expected_release);
        long long deadline = start + p_w->deadline();
        long long release = NOW();
        runtime.exec(p_w->model_name(), expected_release, deadline, MRM[p_w->model_name()].dummy_input);
    }
    runtime.stop();
    best_effort_running = false;

    std::this_thread::sleep_for(std::chrono::seconds(60));
    monitor_running = false;

    return 0;
}