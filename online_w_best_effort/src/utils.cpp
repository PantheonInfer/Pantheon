#include "utils.h"

void set_cpu_affinity(int cpu_id) {
    pthread_t thread_id = pthread_self();

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    CPU_SET(cpu_id, &cpuset);

    // 设置线程的 CPU 亲和性
    int result = pthread_setaffinity_np(thread_id, sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        std::cerr << "cannot set cpu affinity" << std::endl;
    }
}