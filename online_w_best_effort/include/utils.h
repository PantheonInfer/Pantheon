#include <iostream>
#include <chrono>
#include <pthread.h>
#include <thread>

#ifndef _H_UTILS_
#define _H_UTILS_

#define NOW() std::chrono::duration_cast< std::chrono::microseconds >(std::chrono::system_clock::now().time_since_epoch()).count()

#define SLEEP(x) std::this_thread::sleep_until(std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds>(std::chrono::microseconds(x)))

void set_cpu_affinity(int cpu_id);

#endif