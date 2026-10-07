#include "azul/engine.hpp"
#include "azul/c_api.h"
#include "reference.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

static std::uint64_t checks = 0;
#define CHECK(expr) do { ++checks; if (!(expr)) { std::cerr << __FILE__ << ':' << __LINE__ << ": " #expr "\n"; std::exit(1); } } while (false)
using namespace azul;

// Reconstruct a conserved bag for hand-written legal decision fixtures.
static void balance(State& s) {
    s.bag.fill(20); s.bag_size = 0; s.tiles_remaining = 0;
    for (int c = 0; c < 5; ++c) {
        int used = s.discard[c];
        for (auto& src : s.sources) { used += src[c]; s.tiles_remaining += src[c]; }
        for (const auto& p : s.players) {
            used += p.floor_tiles[c] + std::popcount(p.wall & color_mask(c));
            for (int r = 0; r < 5; ++r) if (p.pattern_color[r] == c) used += p.pattern_count[r];
        }
        CHECK(used <= 20);
        s.bag[c] = static_cast<std::uint8_t>(20 - used); s.bag_size += s.bag[c];
    }
    s.phase = Phase::draft;
}
static void setup_and_rng() {
    Rng r{0};
    CHECK(r.next64() == UINT64_C(0xe220a8397b1dcdaf));
    CHECK(r.next64() == UINT64_C(0x6e789e6aa1b965f4));
    int histogram[5]{};
    for (int i = 0; i < 100000; ++i) ++histogram[r.bounded(5)];
    for (int n : histogram) CHECK(n > 19000 && n < 21000);
    for (int i = 0; i < 1000; ++i) {
        const auto s = initial_state(i, static_cast<std::uint8_t>(i & 1));
        CHECK(validate(s)); CHECK(s.bag_size == 80); CHECK(s.round == 1);
        CHECK(s.current == (i & 1)); CHECK(s.token_available);
        for (int f = 0; f < 5; ++f) { int n = 0; for (auto k : s.sources[f]) n += k; CHECK(n == 4); }
        CHECK(s == initial_state(i, static_cast<std::uint8_t>(i & 1)));
    }
    for (int a = 0; a < action_count; ++a) CHECK(encode(source_of(Action(a)), color_of(Action(a)), destination_of(Action(a))) == a);
}
static void drafting() {
    State s;
    s.sources[0] = {2,1,1,0,0}; s.sources[1][4] = 1;
    balance(s); CHECK(validate(s));
    CHECK(step(s, encode(0,0,0)));
    CHECK(s.players[0].pattern_count[0] == 1); CHECK(s.players[0].floor_tiles[0] == 1);
    CHECK(s.sources[5][1] == 1 && s.sources[5][2] == 1); CHECK(s.current == 1);
    CHECK(step(s, encode(5,1,4)));
    CHECK(s.next_start == 1 && !s.token_available); CHECK(s.players[1].floor_count == 1);
    const auto before = s;
    CHECK(!step(s, 65535)); CHECK(s == before);
    CHECK(!step(s, encode(0,0,0))); CHECK(s == before);
    CHECK(!legal(s, encode(5,2,0))); // full pattern row
    CHECK(step(s, encode(5,2,5))); CHECK(validate(s));

    State mismatch;
    mismatch.sources[5][0] = 2;
    mismatch.players[0].pattern_color[2] = 1; mismatch.players[0].pattern_count[2] = 1;
    mismatch.players[0].wall = cell(3,3);
    balance(mismatch);
    CHECK(!legal(mismatch, encode(5,0,2))); CHECK(!legal(mismatch, encode(5,0,3)));
    CHECK(legal(mismatch, encode(5,0,5)));

    State full;
    full.sources[5][0] = 3; full.sources[0][1] = 1;
    full.players[0].floor_count = 7; full.players[0].floor_tiles[2] = 7;
    balance(full); CHECK(step(full, encode(5,0,5)));
    CHECK(full.next_start == 0 && !full.token_available);
    CHECK(full.players[0].floor_count == 7 && full.discard[0] == 3); CHECK(validate(full));

    State token_first;
    token_first.sources[5][0] = 3; token_first.sources[0][1] = 1;
    token_first.players[0].floor_count = 6; token_first.players[0].floor_tiles[2] = 6;
    balance(token_first); CHECK(step(token_first, encode(5,0,5)));
    CHECK(token_first.players[0].floor_count == 7 && token_first.players[0].floor_tiles[0] == 0);
    CHECK(token_first.discard[0] == 3); CHECK(validate(token_first));

    State untouched_token;
    untouched_token.sources[0][0] = 4;
    balance(untouched_token); CHECK(apply(untouched_token, encode(0,0,5)));
    CHECK(untouched_token.phase == Phase::chance && untouched_token.current == 0);
    CHECK(untouched_token.token_available); // token alone is never a player action
    CHECK(legal_actions(untouched_token).size == 0); CHECK(validate(untouched_token));

    State consecutive;
    consecutive.current = 1; consecutive.sources[5][0] = 1;
    balance(consecutive); CHECK(step(consecutive, encode(5,0,4)));
    CHECK(consecutive.round == 1 && consecutive.current == 1); // same actor after round boundary
    CHECK(consecutive.players[1].pattern_count[4] == 1); CHECK(validate(consecutive));
}
static void scoring() {
    // Exhaustively compare the lookup implementation with contiguous runs.
    for (int row = 0; row < 5; ++row) for (int col = 0; col < 5; ++col)
        for (unsigned h = 0; h < 32; ++h) for (unsigned v = 0; v < 32; ++v) {
            Player p;
            for (int x = 0; x < 5; ++x) if (x != col && (h & (1U << x))) p.wall |= cell(row,x);
            for (int y = 0; y < 5; ++y) if (y != row && (v & (1U << y))) p.wall |= cell(y,col);
            int nh = 1, nv = 1;
            for (int x = col-1; x >= 0 && (h & (1U << x)); --x) ++nh;
            for (int x = col+1; x < 5 && (h & (1U << x)); ++x) ++nh;
            for (int y = row-1; y >= 0 && (v & (1U << y)); --y) ++nv;
            for (int y = row+1; y < 5 && (v & (1U << y)); ++y) ++nv;
            CHECK(place_and_score(p,row,color_at(row,col)) == ((nh == 1 && nv == 1) ? 1 : (nh > 1 ? nh : 0) + (nv > 1 ? nv : 0)));
        }
    CHECK(end_bonus((1U << 25) - 1) == 95);
    CHECK(end_bonus(row_mask(0)) == 2); CHECK(end_bonus(column_mask(0)) == 7);
    CHECK(end_bonus(color_mask(0)) == 10);
    const int penalty[8] = {0,1,2,4,6,8,11,14};
    for (int n = 0; n < 8; ++n) {
        State s; s.players[0].score = 20;
        s.players[0].floor_count = std::uint8_t(n); s.players[0].floor_tiles[0] = std::uint8_t(n);
        settle_round(s); CHECK(s.players[0].score == 20-penalty[n]); CHECK(s.discard[0] == n);
    }
    State order;
    order.players[0].pattern_color = {0,4,2,5,5};
    order.players[0].pattern_count = {1,2,1,0,0};
    order.players[0].floor_count = 2; order.players[0].floor_tiles[3] = 2;
    settle_round(order);
    CHECK(order.players[0].score == 1); // top-to-bottom: 1 + 2, then subtract 2
    CHECK(order.players[0].pattern_count[2] == 1 && order.players[0].pattern_color[2] == 2);
    CHECK(order.discard[4] == 1 && order.discard[3] == 2);
    State clamp; clamp.players[0].floor_count = 7; settle_round(clamp); CHECK(clamp.players[0].score == 0);
}
static void ending_and_bag() {
    State s;
    s.players[0].wall = row_mask(0) & ~cell(0,0);
    s.players[0].pattern_color[0] = 0; s.players[0].pattern_count[0] = 1;
    s.players[1].pattern_color[0] = 2; s.players[1].pattern_count[0] = 1;
    s.sources[0][3] = 1;
    balance(s); s.current = 1;
    const auto rng_before = s.rng;
    CHECK(step(s,encode(0,3,5))); CHECK(s.phase == Phase::terminal);
    CHECK(s.players[0].score == 7); CHECK(s.players[1].wall == cell(0,2));
    CHECK(s.players[1].score == 0); CHECK(s.rng == rng_before); CHECK(validate(s));
    CHECK(winner(s) == 0 && terminal_reward(s,0) == 1 && terminal_reward(s,1) == -1);
    const auto ended = s; CHECK(!step(s,0)); CHECK(!deal(s)); CHECK(s == ended);
    s.players[1].score = s.players[0].score; CHECK(winner(s) == 0);
    s.players[1].wall = row_mask(1); CHECK(winner(s) == -1);
    s.players[1].wall |= row_mask(2); CHECK(winner(s) == 1);

    State refill; refill.bag[0] = 2; refill.bag_size = 2; refill.discard[1] = 20;
    CHECK(deal(refill)); CHECK(refill.sources[0][0] == 2 && refill.sources[0][1] == 2);
    CHECK(refill.bag_size == 2 && refill.bag[1] == 2 && refill.discard[1] == 0);
    State exact; exact.bag[0] = 20; exact.bag_size = 20; exact.discard[1] = 20;
    CHECK(deal(exact)); CHECK(exact.bag_size == 0 && exact.discard[1] == 20); // no early refill
    State scarce; scarce.bag[0] = 2; scarce.bag_size = 2; scarce.discard[1] = 1;
    CHECK(deal(scarce)); CHECK(scarce.tiles_remaining == 3 && scarce.sources[0][0] == 2 && scarce.sources[0][1] == 1);

    // All-floor policies may continue forever. The engine must not invent a draw.
    auto forever = initial_state(131);
    for (int k = 0; k < 2000; ++k) {
        const auto list = legal_actions(forever);
        auto chosen = list.values[0];
        for (auto a : list) if (destination_of(a) == azul::floor) { chosen = a; break; }
        CHECK(step(forever, chosen)); CHECK(validate(forever));
    }
    CHECK(forever.phase == Phase::draft && winner(forever) == -2);
    CHECK(forever.players[0].wall == 0 && forever.players[1].wall == 0);
}
static void differential_games() {
    std::uint64_t total_steps = 0;
    for (std::uint64_t id = 0; id < 3000; ++id) {
        auto s = initial_state(episode_seed(123,id), std::uint8_t(id & 1));
        Rng policy{episode_seed(987,id)};
        while (s.phase != Phase::terminal && s.ply < 4096) {
            CHECK(validate(s));
            const auto list = legal_actions(s); CHECK(list.size > 0);
            std::array<std::uint8_t,action_count> mask{}; action_mask(s,mask.data());
            std::array<std::uint8_t,action_count> seen{};
            for (auto a : list) { CHECK(a < action_count); CHECK(!seen[a]); seen[a] = 1; }
            for (int a = 0; a < action_count; ++a) {
                CHECK(legal(s,Action(a)) == reference::legal(s,unsigned(a)));
                CHECK(mask[a] == seen[a] && bool(mask[a]) == legal(s,Action(a)));
            }
            std::array<float,observation_size> obs{}, hidden{};
            observe(s,s.current,obs.data());
            auto clone = s; clone.rng.next64(); observe(clone,clone.current,hidden.data());
            CHECK(obs == hidden);
            const auto a = list.values[policy.bounded(list.size)];
            auto ref = s; reference::apply(ref,a);
            CHECK(apply(s,a)); CHECK(s == ref); CHECK(validate(s));
            if (s.phase == Phase::chance) {
                auto copy = s; CHECK(deal(s)); CHECK(deal(copy)); CHECK(s == copy);
            }
            ++total_steps;
        }
        CHECK(s.phase == Phase::terminal); CHECK(validate(s));
    }
    std::cout << "Differential games: 3000, decisions: " << total_steps << '\n';
}
static void batch_api() {
    CHECK(azul_abi_version() == 1); static_assert(sizeof(AzulStepResult) == 12);
    CHECK(azul_batch_create(0,1,42) == nullptr);
    CHECK(azul_batch_reset(nullptr,0) == -1);
    constexpr std::size_t n = 129;
    auto* serial = azul_batch_create(n,1,42); auto* parallel = azul_batch_create(n,4,42);
    CHECK(serial && parallel); CHECK(azul_batch_size(serial) == n);
    std::vector<float> obs(n*observation_size), obs2(obs.size());
    std::vector<std::uint8_t> masks(n*action_count), masks2(masks.size());
    std::vector<std::uint8_t> players(n), players2(n);
    std::vector<std::uint16_t> actions(n);
    std::vector<AzulStepResult> r1(n),r2(n);
    Rng policy{998};
    for (int iteration = 0; iteration < 300; ++iteration) {
        CHECK(azul_batch_observe(serial,obs.data(),masks.data(),players.data()) == 0);
        CHECK(azul_batch_observe(parallel,obs2.data(),masks2.data(),players2.data()) == 0);
        CHECK(obs == obs2 && masks == masks2 && players == players2);
        for (std::size_t i = 0; i < n; ++i) {
            ActionList list;
            for (int a = 0; a < action_count; ++a) if (masks[i*action_count+a]) list.values[list.size++] = Action(a);
            actions[i] = list.size ? list.values[policy.bounded(list.size)] : 65535;
            if (iteration % 31 == 0) actions[i] = 65535;
        }
        CHECK(azul_batch_step(serial,actions.data(),r1.data()) == 0);
        CHECK(azul_batch_step(parallel,actions.data(),r2.data()) == 0);
        CHECK(std::memcmp(r1.data(),r2.data(),n*sizeof(AzulStepResult)) == 0);
        for (std::size_t i = 0; i < n; ++i) {
            State a,b;
            CHECK(azul_batch_snapshot(serial,i,&a,sizeof(a)) == 0);
            CHECK(azul_batch_snapshot(parallel,i,&b,sizeof(b)) == 0);
            CHECK(a == b && validate(a));
            if (r1[i].terminated) {
                CHECK(azul_batch_reset_at(serial,i,episode_seed(iteration,i),unsigned(i&1)) == 0);
                CHECK(azul_batch_reset_at(parallel,i,episode_seed(iteration,i),unsigned(i&1)) == 0);
            }
        }
    }
    State saved; CHECK(azul_snapshot_size() == sizeof(State));
    CHECK(azul_batch_snapshot(serial,0,&saved,sizeof(saved)) == 0);
    CHECK(azul_batch_restore(parallel,0,&saved,sizeof(saved)) == 0);
    State invalid = saved; invalid.bag[0] = 255;
    CHECK(azul_batch_restore(parallel,0,&invalid,sizeof(invalid)) == -1);
    State unchanged; CHECK(azul_batch_snapshot(parallel,0,&unchanged,sizeof(unchanged)) == 0);
    CHECK(unchanged == saved);
    CHECK(azul_batch_restore(parallel,0,&saved,sizeof(saved)-1) == -1);
    State public_source = initial_state(717, 1);
    const auto public_action = legal_actions(public_source).values[0];
    apply_unchecked(public_source, public_action);
    std::array<std::uint8_t, 30> imported_sources{};
    std::array<std::uint32_t, 2> imported_walls{};
    std::array<std::uint16_t, 2> imported_scores{};
    std::array<std::uint8_t, 10> imported_colors{}, imported_counts{}, imported_floor{};
    std::array<std::uint8_t, 2> imported_floor_counts{};
    for (int source = 0; source < 6; ++source) for (int color = 0; color < 5; ++color)
        imported_sources[source * 5 + color] = public_source.sources[source][color];
    for (int player = 0; player < 2; ++player) {
        imported_walls[player] = public_source.players[player].wall;
        imported_scores[player] = public_source.players[player].score;
        imported_floor_counts[player] = public_source.players[player].floor_count;
        for (int i = 0; i < 5; ++i) {
            imported_colors[player * 5 + i] = public_source.players[player].pattern_color[i];
            imported_counts[player * 5 + i] = public_source.players[player].pattern_count[i];
            imported_floor[player * 5 + i] = public_source.players[player].floor_tiles[i];
        }
    }
    CHECK(azul_batch_import_public(serial, 0, imported_sources.data(), imported_walls.data(),
        imported_scores.data(), imported_colors.data(), imported_counts.data(), imported_floor.data(),
        imported_floor_counts.data(), public_source.bag.data(), public_source.discard.data(),
        public_source.round, public_source.current, public_source.next_start, public_source.token_available) == 0);
    State imported_state; CHECK(azul_batch_snapshot(serial, 0, &imported_state, sizeof(imported_state)) == 0);
    CHECK(imported_state.players == public_source.players && imported_state.sources == public_source.sources);
    CHECK(imported_state.bag == public_source.bag && imported_state.discard == public_source.discard);
    CHECK(imported_state.current == public_source.current && imported_state.next_start == public_source.next_start);
    CHECK(imported_state.round == public_source.round && validate(imported_state));
    imported_floor_counts[0] = 8;
    CHECK(azul_batch_import_public(serial, 0, imported_sources.data(), imported_walls.data(),
        imported_scores.data(), imported_colors.data(), imported_counts.data(), imported_floor.data(),
        imported_floor_counts.data(), public_source.bag.data(), public_source.discard.data(),
        public_source.round, public_source.current, public_source.next_start, public_source.token_available) == -1);
    State after_invalid_import; CHECK(azul_batch_snapshot(serial, 0, &after_invalid_import, sizeof(after_invalid_import)) == 0);
    CHECK(after_invalid_import == imported_state);
    CHECK(azul_batch_reset_at(serial,n,0,0) == -1); CHECK(azul_batch_reset_at(serial,0,0,2) == -1);
    azul_batch_destroy(serial); azul_batch_destroy(parallel);
}
int main() {
    setup_and_rng(); drafting(); scoring(); ending_and_bag(); differential_games(); batch_api();
    std::cout << "All tests passed; checks=" << checks << ", sizeof(State)=" << sizeof(State) << '\n';
}
