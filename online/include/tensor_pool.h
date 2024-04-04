#include <unordered_map>
#include <utility>
#include <torch/script.h>
#include <torch/torch.h>

#ifndef _H_TENSOR_POOL_
#define _H_TENSOR_POOL_

struct hash_key {
    unsigned int id;
    int block_idx;
    hash_key(unsigned int i, int b) : id(i), block_idx(b) {}
};

struct hash_func {
    size_t operator()(const hash_key& key) const {
        return std::hash<unsigned int>()(key.id) ^ std::hash<int>()(key.block_idx);
    }
};

struct hash_equal {
    bool operator()(const hash_key& lhs, const hash_key& rhs) const {
        return lhs.id == rhs.id && lhs.block_idx == rhs.block_idx;
    }
};

typedef std::unordered_map<hash_key, torch::jit::IValue, hash_func, hash_equal> TensorPool;

#endif