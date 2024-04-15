#ifndef _H_NNJOB_
#define _H_NNJOB_

struct NNJob{
    unsigned int id;
    std::string model_name;
    int current_block_id; // 指等待执行的block
    int last_block_id;
    int exit_id;
    long long release;
    long long deadline;

    NNJob() : id(0), model_name("none"), current_block_id(-1), last_block_id(-1), exit_id(-1), release(0), deadline(0) {}
    NNJob(unsigned i, std::string m, long long r, long long d, int e, int l) : id(i), model_name(m), current_block_id(-1), release(r), deadline(d), exit_id(e), last_block_id(l) {}

    bool operator<(const NNJob& other) const {
        return deadline < other.deadline;
    }
};

#endif