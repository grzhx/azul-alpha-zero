#include "azul/engine.hpp"
#include "azul/c_api.h"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

// Measures training transport too: observations, dense masks, policy sampling,
// checked steps, result buffers and reset of completed episodes.
int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1],nullptr,10) : 8192;
    const unsigned threads = argc > 2 ? unsigned(std::strtoul(argv[2],nullptr,10)) : 1;
    const unsigned ticks = argc > 3 ? unsigned(std::strtoul(argv[3],nullptr,10)) : 1000;
    if (!n || !ticks || n > 1048576 || threads > 1024) return 1;
    const auto deleter = [](AzulBatch* p) { azul_batch_destroy(p); };
    std::unique_ptr<AzulBatch,decltype(deleter)> b(azul_batch_create(n,threads,42), deleter);
    if (!b) return 1;
    std::vector<float> observations(n*azul::observation_size);
    std::vector<std::uint8_t> masks(n*azul::action_count);
    std::vector<std::uint16_t> actions(n);
    std::vector<AzulStepResult> results(n);
    azul::Rng policy{42};
    std::uint64_t episodes = 0, checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (unsigned t = 0; t < ticks; ++t) {
        if (azul_batch_observe(b.get(), observations.data(), masks.data(),nullptr)) return 1;
        for (std::size_t i = 0; i < n; ++i) {
            azul::ActionList list;
            for (int a = 0; a < azul::action_count; ++a) if (masks[i*azul::action_count+a]) list.values[list.size++] = azul::Action(a);
            actions[i] = list.values[policy.bounded(list.size)];
        }
        if (azul_batch_step(b.get(),actions.data(),results.data())) return 1;
        for (std::size_t i = 0; i < n; ++i) {
            if (results[i].invalid_action) return 1;
            if (results[i].terminated) {
                ++episodes;
                checksum += results[i].scores[0] + 257U*results[i].scores[1];
                if (azul_batch_reset_at(b.get(),i,azul::episode_seed(42, n+episodes),unsigned(episodes&1))) return 1;
            }
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << "batch,threads,ticks,steps,seconds,steps_per_s,completed,checksum\n" << n << ',' << threads << ',' << ticks << ','
              << n*ticks << ',' << std::fixed << std::setprecision(6) << seconds << ',' << n*ticks/seconds << ',' << episodes << ',' << checksum << '\n';
}
