#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <chrono>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>


#define BUFFER_SIZE 1024 * 1024
#define MODEL_FILE_NAME "recv_model.pth"
#define WARMUP 1000
#define REPEAT 1000


int main(int argc, const char* argv[]) {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // 创建套接字
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    // 连接服务器
    struct sockaddr_in servaddr;
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(9999);
    inet_pton(AF_INET, "192.168.137.1", &servaddr.sin_addr);
    connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr));
    int round = 0;
    while (true) {
        // 接收模型输入的大小
        int size_of_input_size;
        recv(sockfd, &size_of_input_size, sizeof(size_of_input_size), 0);
        std::vector<int> input_size(size_of_input_size);
        for (int i = 0; i < size_of_input_size; i++) {
            int int_data;
            recv(sockfd, &int_data, sizeof(int_data), 0);
            input_size[i] = int_data;
        }

        std::cout << "Input size: [ ";
        for (int i : input_size) {
            std::cout << i << " ";
        }
        std::cout << "]" << std::endl;
        
        // 接收模型
        int size_of_file;
        recv(sockfd, &size_of_file, sizeof(size_of_file), 0);

        std::ofstream fout(MODEL_FILE_NAME, std::ios::binary);
        char buffer[BUFFER_SIZE];
        float total_recv_size = 0;
        while (true)
        {
            int recv_size = recv(sockfd, buffer, sizeof(buffer), 0);
            if (recv_size < 0) {
                std::cerr << "recv error" << std::endl;
                break;
            }
            fout.write(buffer, recv_size);
            total_recv_size += recv_size;
            if (total_recv_size >= size_of_file) break;
        }
        fout.close();
        std::cout << "Received model: " << total_recv_size / 1024 << " KB" << std::endl;
        // 测量模型的latency
        // 载入模型
        torch::NoGradGuard no_grad;
        torch::jit::script::Module model;
        model = torch::jit::load(MODEL_FILE_NAME);
        model.to(at::kCUDA);
        model.eval();
        
        // 输入tensor

        std::vector<int64_t> shape; 
        std::transform(input_size.begin(), input_size.end(), std::back_inserter(shape), [](int x) { return static_cast<int64_t>(x); });
        std::vector<torch::jit::IValue> inputs;
        torch::jit::IValue input_t = torch::rand(shape).to(at::kCUDA);
        inputs.push_back(input_t);
        // warmup

        for (int i = 0; i < WARMUP; i++) {
            auto outputs = model.forward(inputs);
        }

        double execution_time = 0;
        double worst_case_execution_time = 0;
        double average_execution_time = 0;
        std::chrono::high_resolution_clock::time_point start_point;
        std::chrono::high_resolution_clock::time_point stop_point;
        for (int i = 0; i < REPEAT; i++) {
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));

            start_point = std::chrono::high_resolution_clock::now();
            auto outputs = model.forward(inputs);
            
            C10_CUDA_CHECK(cudaStreamSynchronize(stream)); // CUDA异步执行，准确测量执行时间需要同步

            stop_point = std::chrono::high_resolution_clock::now();
            
            execution_time = std::chrono::duration_cast<std::chrono::microseconds>(stop_point - start_point).count() / 1000.0;
            average_execution_time += execution_time;
            if (execution_time > worst_case_execution_time) {
                worst_case_execution_time = execution_time;
            }
        }
        average_execution_time /= REPEAT;
        std::cout << "avg: " << average_execution_time << " ms, wcet: " << worst_case_execution_time << " ms" << std::endl;

        int double_data_bin_size = sizeof(double);
        char double_data_bin[double_data_bin_size];
        // std::memcpy(double_data_bin, &worst_case_execution_time, double_data_bin_size);
        std::memcpy(double_data_bin, &average_execution_time, double_data_bin_size);
        send(sockfd, double_data_bin, double_data_bin_size, 0);
        round++;
    }
    
    return 0;
}