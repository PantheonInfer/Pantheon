#include "model_repository_manager.h"

Model::Model(std::filesystem::path path) {
    std::cout << "Loading model from: " << path.c_str() << std::endl;
    std::filesystem::path config_file = "config.pbtxt";
    std::filesystem::path config_path = path / config_file;
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cerr << "Faild to open file: " << config_path << std::endl;
        exit(-1);
    }
    std::string pbtxt;
    std::string buffer;
    while (std::getline(file, buffer)) {
        pbtxt += buffer + "\n";
    }
    ModelConfig model_config;
    if (!google::protobuf::TextFormat::ParseFromString(pbtxt, &model_config)) {
        std::cerr << "Fail to parse model config: " << config_path << std::endl;
        exit(-1);
    }
    
    name = model_config.name();
    for (const int64_t dim : model_config.dims()) {
        dims.push_back(dim);
    }

    for (const ModelBlockProfile bp : model_config.block_profile()) {
        std::ostringstream oss;
        oss << "model_files/block_" << std::setw(2) << std::setfill('0') << bp.id() << ".pth";
        std::filesystem::path block_file = oss.str();
        std::filesystem::path block_path = path / block_file;
        BlockModule bm = {torch::jit::load(block_path), bp.latency()};
        bm._module.to(torch::kCUDA);
        bm._module.eval();
        block_modules.push_back(bm);
    }

    for (const ModelBlockProfile ep : model_config.exit_profile()) {
        std::ostringstream oss;
        oss << "model_files/branch_" << std::setw(2) << std::setfill('0') << ep.id() << ".pth";
        std::filesystem::path exit_file = oss.str();
        std::filesystem::path exit_path = path / exit_file;
        ExitModule em = {torch::jit::load(exit_path), ep.id(), ep.latency(), ep.accuracy()};
        em._module.to(torch::kCUDA);
        em._module.eval();
        exit_modules.push_back(em);
        exit_position.push_back(ep.id());
    }

    dummy_input = torch::randn(dims).to(torch::kCUDA);

    // 计算submodel latency
    submodel_latency = std::vector<std::vector<long long>>(exit_modules.size(), std::vector<long long>(block_modules.size() + 1, 0));
    
    for (int i = 0; i < exit_modules.size(); i++) {
        for (int j = -1; j <= exit_modules[i].previous_block_id; j++) {
            long long l = exit_modules[i].latency;
            for (int k = j + 1; k <= exit_modules[i].previous_block_id; k++) {
                l += block_modules[k].latency;
            }
            submodel_latency[i][j + 1] = l;
        }
    }

    key_block = std::vector<bool>(get_num_blocks(), 0);
    for (auto em : exit_modules) {
        key_block[em.previous_block_id] = 1;
    }
}

Model::~Model() {

}

void Model::warmup(int rounds) {
    std::cout << "Warming up: " << name << std::endl;
    // torch::jit::IValue dummy_input = torch::randn(dims).to(torch::kCUDA);
    {
        torch::InferenceMode guard(true);
        for (int r = 0; r < rounds; r++) {
            auto start = NOW();
            torch::jit::IValue input = dummy_input;
            for (int i = 0; i < block_modules.size(); i++) {
                input = block_modules[i]._module.forward({input});
                for (int j = 0; j < exit_modules.size(); j++) {
                    if (exit_position[j] == i) {
                        exit_modules[j]._module.forward({input});
                        break;
                    }
                }
            }
            auto end = NOW();
            // std::cout << "Warmup round " << r << " takes " << (end - start) / 1000. << "ms" << std::endl; 
        }
    }
}

torch::jit::script::Module& Model::get_module(bool is_exit, int id) {
    if (is_exit) {
        return exit_modules[id]._module;
    }
    else {
        return block_modules[id]._module;
    }
}

int Model::get_num_exits() {
    return exit_modules.size();
}

int Model::get_num_blocks() {
    return block_modules.size();
}

int Model::get_last_block_id(int exit_id) {
    return exit_modules[exit_id].previous_block_id;
}

long long Model::get_submodel_latency(int current_block_id, int exit_id) {
    return submodel_latency[exit_id][current_block_id];
}

float Model::get_accuracy(int exit_id) {
    return exit_modules[exit_id].accuracy;
}

bool Model::is_key_block(int block_id) {
    return key_block[block_id];
}

ModelRepositoryManager MRM;

void ModelRepositoryManager::init(std::filesystem::path path) {
    std::cout << "Initializing model repository from: " << path.c_str() << std::endl;
    for (const auto &entry : std::filesystem::directory_iterator(path)) {
        Model model(entry.path());
        // model.warmup(WARMUP_ROUND);
        // std::cout << "Warming: Warmup is necessary" << std::endl;
        models.emplace(model.name, model);
    }
}

void ModelRepositoryManager::warmup_all() {
    for (auto it = models.begin(); it != models.end(); it++) {
        it->second.warmup(WARMUP_ROUND);
    }
}

Model& ModelRepositoryManager::operator[](std::string name) {
    if (!models.count(name)) {
        // std::cerr << "Model: " << name << " does not exist" << std::endl;
        spdlog::info("Model: {} does not exist", name);
        abort();
    }
    return models.at(name);
}