#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <chrono>
#include <vector>
#include <thread>
#include <filesystem>
#include <torch/torch.h>
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

void monitor() {
    while (monitor_running) {
        auto time = NOW();
        std::ifstream gpuUtilFile("/sys/devices/gpu.0/load");
        std::ifstream currFile("/sys/bus/i2c/drivers/ina3221/7-0040/hwmon/hwmon5/curr1_input");
        std::ifstream voltFile("/sys/bus/i2c/drivers/ina3221/7-0040/hwmon/hwmon5/in1_input");
        if (!gpuUtilFile) {
            std::cerr << "Failed to open GPU utilization file." << std::endl;
            exit(1);
        }
        if (!currFile) {
            std::cerr << "Failed to open Current file." << std::endl;
            exit(1);
        }
        if (!voltFile) {
            std::cerr << "Failed to open Voltage file." << std::endl;
            exit(1);
        }
        std::string gpuUtilStr;
        if (!std::getline(gpuUtilFile, gpuUtilStr)) {
            std::cerr << "Failed to read GPU utilization." << std::endl;
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
        float gpuUtil = std::stof(gpuUtilStr) / 10.0f;
        float curr = std::stof(currStr) / 1000.0f;
        float volt = std::stof(voltStr) / 1000.0f;
        spdlog::info("[GPU] {} {} {}", time, gpuUtil, curr * volt);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5 && argc != 4 && argc != 3) {
        std::cerr << "Usage: ./runtime <path-to-model-repository> <path-to-workload-config> [path-to-log] [monitor-gpu-or-not]" << std::endl;
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
    
    if (argc == 4 || argc == 5) {
        std::filesystem::path log_path = argv[3];

        if (std::filesystem::exists(log_path)) {
            std::filesystem::remove(log_path);
        }
        // auto logger = spdlog::basic_logger_mt("logger", log_path.string()); 
        auto logger = spdlog::basic_logger_mt<spdlog::async_factory>("logger", log_path.string());
        spdlog::set_default_logger(logger); 
        
        if (argc == 5) {
            monitor_running = true;
            monitor_t = std::thread(monitor);
            monitor_t.detach();
        }
    }
    else if (argc == 3) {
        auto logger = spdlog::stdout_color_mt("logger");
        spdlog::set_default_logger(logger); 
    }
    // spdlog::flush_every(std::chrono::seconds(1));
    
    
    Runtime runtime;
    runtime.start();

    if (argc == 5) {
        std::this_thread::sleep_for(std::chrono::seconds(120));
    }

    const Workload* p_w;

    auto start = NOW();
    spdlog::info("[START] {}", start);
    for (int i = 0; i < workloads.workload_size(); i++) {
        p_w = &workloads.workload(i);
        long long expected_release = start + p_w->release();
        SLEEP(expected_release);
        long long deadline = start + p_w->deadline();
        long long release = NOW();
        // std::cout << "Expected release: " << expected_release << " Actual release: " << release << " Deadline: " << deadline << std::endl;
        // spdlog::info("[RELEASE] {} {} {}", expected_release, before_sleep, release);
        runtime.exec(p_w->model_name(), expected_release, deadline, MRM[p_w->model_name()].dummy_input);
    }
    runtime.stop();

    if (argc == 5) {
        std::this_thread::sleep_for(std::chrono::seconds(120));
        monitor_running = false;
    }
    return 0;
}