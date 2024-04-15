#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <google/protobuf/text_format.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "model_config.pb.h"
#include "utils.h"

#ifndef _H_MODEL_REPOSITORY_MANAGER_
#define _H_MODEL_REPOSITORY_MANAGER_

#define WARMUP_ROUND 2000

typedef struct {
    torch::jit::script::Module _module;
    long long latency;
} BlockModule;

typedef struct {
    torch::jit::script::Module _module;
    int previous_block_id;
    long long latency;
    float accuracy;
} ExitModule;

class Model {
private:
    std::vector<int64_t> dims;
    std::vector<int> exit_position;
    std::vector<std::vector<long long>> submodel_latency; // 从不同block开始在在不同出口结束的cost，形状是exit x (block + 1)
                                                          // +1是表示一个block都还没执行
    // long long** submodel_latency; 
    std::vector<BlockModule> block_modules;
    std::vector<ExitModule> exit_modules;
    std::vector<bool> key_block; // 有exit的block，用在执行时判断中间结果需不需要保留在tensor pool中
public:
    torch::jit::IValue dummy_input;
    std::string name;
    Model(std::filesystem::path path);
    ~Model();
    void warmup(int rounds);
    torch::jit::script::Module& get_module(bool is_exit, int id);
    int get_num_exits();
    int get_num_blocks();
    int get_last_block_id(int exit_id);
    // TODO: 
    long long get_submodel_latency(int current_block_id, int exit_id);
    float get_accuracy(int exit_id);
    bool is_key_block(int block_id);
};

class ModelRepositoryManager {
private:
    std::unordered_map<std::string, Model> models;
public:
    void init(std::filesystem::path path);
    void warmup_all();
    Model& operator[] (std::string);
};

// Global ModelRepositoryManager
extern ModelRepositoryManager MRM;
#endif