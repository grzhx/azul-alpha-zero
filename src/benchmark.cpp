#include "azul/engine.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

struct alignas(64) Stats {
    std::uint64_t games = 0, steps = 0, rounds = 0, checksum = 0;
    std::uint64_t wins0 = 0, wins1 = 0, draws = 0, truncated = 0;
};
int main(int argc, char** argv) {
    std::uint64_t games = 100000, seed = 42, max_steps = 4096;
    unsigned threads = 1;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "azul_bench [--games N] [--threads N] [--seed N] [--max-steps N]\n";
                return 0;
            }
            if (i + 1 == argc) throw std::invalid_argument("missing value");
            const std::string value = argv[++i];
            if (value.empty() || value[0] == '-') throw std::invalid_argument("negative/empty value");
            std::size_t parsed = 0;
            const auto n = std::stoull(value, &parsed);
            if (parsed != value.size()) throw std::invalid_argument("invalid integer");
            if (arg == "--games") games = n;
            else if (arg == "--seed") seed = n;
            else if (arg == "--max-steps") max_steps = n;
            else if (arg == "--threads" && n <= 1024) threads = static_cast<unsigned>(n);
            else throw std::invalid_argument("unknown option or excessive thread count");
        }
        if (!games || !threads || !max_steps) throw std::invalid_argument("counts must be positive");
        threads = static_cast<unsigned>(std::min<std::uint64_t>(threads, games));
        std::vector<Stats> stats(threads);
        std::vector<std::thread> workers;
        workers.reserve(threads);
        const auto start = std::chrono::steady_clock::now();
        try {
            for (unsigned t = 0; t < threads; ++t) workers.emplace_back([&, t] {
                auto& st = stats[t];
                for (std::uint64_t id = t; id < games; id += threads) {
                    auto s = azul::initial_state(azul::episode_seed(seed, id), static_cast<std::uint8_t>(id & 1));
                    azul::Rng policy{azul::episode_seed(seed ^ UINT64_C(0xd1b54a32d192ed03), id)};
                    std::uint64_t digest = azul::episode_seed(seed, id);
                    while (s.phase != azul::Phase::terminal && s.ply < max_steps) {
                        const auto actions = azul::legal_actions(s);
                        const auto a = actions.values[policy.bounded(actions.size)];
                        azul::step_unchecked(s, a);
                        digest = (digest ^ a) * UINT64_C(0x100000001b3);
                    }
                    ++st.games; st.steps += s.ply; st.rounds += s.round;
                    const int w = azul::winner(s);
                    st.wins0 += w == 0; st.wins1 += w == 1; st.draws += w == -1; st.truncated += w == -2;
                    digest ^= std::uint64_t{s.players[0].score} << 32;
                    digest ^= std::uint64_t{s.players[1].score} << 48;
                    st.checksum ^= digest;
                }
            });
        } catch (...) {
            for (auto& w : workers) w.join();
            throw;
        }
        for (auto& w : workers) w.join();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        Stats total;
        for (const auto& st : stats) {
            total.games += st.games; total.steps += st.steps; total.rounds += st.rounds;
            total.wins0 += st.wins0; total.wins1 += st.wins1; total.draws += st.draws;
            total.truncated += st.truncated; total.checksum ^= st.checksum;
        }
        std::cout << "threads,games,steps,seconds,games_per_s,steps_per_s,avg_rounds,wins0,wins1,draws,truncated,state_bytes,checksum\n"
                  << threads << ',' << total.games << ',' << total.steps << ',' << std::fixed << std::setprecision(6)
                  << seconds << ',' << total.games / seconds << ',' << total.steps / seconds << ','
                  << double(total.rounds) / double(total.games) << ',' << total.wins0 << ',' << total.wins1 << ','
                  << total.draws << ',' << total.truncated << ',' << sizeof(azul::State) << ',' << total.checksum << '\n';
        return total.truncated ? 2 : 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
