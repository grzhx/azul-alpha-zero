#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <type_traits>

namespace azul {
inline constexpr int colors = 5, factories = 5, center = 5, floor = 5;
inline constexpr int action_count = 180, observation_size = 172;
inline constexpr std::uint8_t empty = 5;
using Action = std::uint16_t;
using Counts = std::array<std::uint8_t, 5>;
enum class Phase : std::uint8_t { draft, chance, terminal };

// SplitMix64; no global generator, no STL distribution implementation dependence.
struct Rng {
    std::uint64_t state = 0;
    std::uint64_t next64() noexcept {
        auto z = (state += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }
    std::uint32_t bounded(std::uint32_t n) noexcept {
        assert(n != 0);
        auto x = static_cast<std::uint32_t>(next64() >> 32);
        auto m = std::uint64_t{x} * n;
        auto low = static_cast<std::uint32_t>(m);
        if (low < n) {
            const auto threshold = (std::uint32_t{0} - n) % n;
            while (low < threshold) {
                x = static_cast<std::uint32_t>(next64() >> 32);
                m = std::uint64_t{x} * n;
                low = static_cast<std::uint32_t>(m);
            }
        }
        return static_cast<std::uint32_t>(m >> 32);
    }
    bool operator==(const Rng&) const = default;
};
inline std::uint64_t episode_seed(std::uint64_t base, std::uint64_t id) noexcept {
    Rng r{base + UINT64_C(0x9e3779b97f4a7c15) * id};
    return r.next64();
}

struct Player {
    std::uint32_t wall = 0; // bit(row * 5 + column)
    std::uint16_t score = 0;
    Counts pattern_color{empty, empty, empty, empty, empty};
    Counts pattern_count{};
    Counts floor_tiles{}; // occupied colored floor spaces, excluding overflow
    std::uint8_t floor_count = 0; // includes marker if it fitted
    bool operator==(const Player&) const = default;
};
struct State {
    Rng rng{}; // simulation private; never exposed in policy observations
    std::array<Player, 2> players{};
    std::array<Counts, 6> sources{};
    Counts bag{}, discard{};
    std::uint32_t round = 0, ply = 0;
    std::uint8_t bag_size = 0, tiles_remaining = 0;
    std::uint8_t current = 0, next_start = 0, token_available = 1;
    Phase phase = Phase::chance;
    bool operator==(const State&) const = default;
};
static_assert(std::is_trivially_copyable_v<State>);
static_assert(sizeof(State) <= 128);
struct ActionList {
    std::array<Action, action_count> values; // deliberately not zero-filled
    std::uint16_t size = 0;
    const Action* begin() const noexcept { return values.data(); }
    const Action* end() const noexcept { return values.data() + size; }
};
constexpr Action encode(int source, int color, int destination) noexcept {
    return static_cast<Action>((source * 5 + color) * 6 + destination);
}
constexpr int source_of(Action a) noexcept { return a / 30; }
constexpr int color_of(Action a) noexcept { return (a / 6) % 5; }
constexpr int destination_of(Action a) noexcept { return a % 6; }
constexpr int column_of(int row, int color) noexcept { return (row + color) % 5; }
constexpr int color_at(int row, int column) noexcept { return (column + 5 - row) % 5; }
constexpr std::uint32_t cell(int row, int column) noexcept { return 1U << (row * 5 + column); }
constexpr std::uint32_t row_mask(int row) noexcept { return 31U << (row * 5); }
constexpr std::uint32_t column_mask(int col) noexcept { return 0x108421U << col; }
constexpr std::uint32_t color_mask(int color) noexcept {
    std::uint32_t mask = 0;
    for (int row = 0; row < 5; ++row) mask |= cell(row, column_of(row, color));
    return mask;
}
inline int complete_rows(std::uint32_t wall) noexcept {
    int n = 0;
    for (int r = 0; r < 5; ++r) n += (wall & row_mask(r)) == row_mask(r);
    return n;
}
inline int end_bonus(std::uint32_t wall) noexcept {
    int score = 2 * complete_rows(wall);
    for (int c = 0; c < 5; ++c) {
        score += 7 * ((wall & column_mask(c)) == column_mask(c));
        score += 10 * ((wall & color_mask(c)) == color_mask(c));
    }
    return score;
}
// 160-byte compile-time table for contiguous line scoring, including the new tile.
inline constexpr auto runs = [] {
    std::array<std::array<std::uint8_t, 5>, 32> table{};
    for (int mask = 0; mask < 32; ++mask) for (int pos = 0; pos < 5; ++pos) {
        int n = 1;
        for (int i = pos - 1; i >= 0 && (mask & (1 << i)); --i) ++n;
        for (int i = pos + 1; i < 5 && (mask & (1 << i)); ++i) ++n;
        table[mask][pos] = static_cast<std::uint8_t>(n);
    }
    return table;
}();
inline int place_and_score(Player& p, int row, int color) noexcept {
    const int col = column_of(row, color);
    p.wall |= cell(row, col);
    const auto horizontal = (p.wall >> (row * 5)) & 31U;
    unsigned vertical = 0;
    for (int r = 0; r < 5; ++r) vertical |= ((p.wall >> (r * 5 + col)) & 1U) << r;
    const int h = runs[horizontal][col], v = runs[vertical][row];
    return (h == 1 && v == 1) ? 1 : (h > 1 ? h : 0) + (v > 1 ? v : 0);
}
inline bool accepts(const Player& p, int row, int color) noexcept {
    return p.pattern_count[row] < row + 1 &&
        (p.pattern_count[row] == 0 || p.pattern_color[row] == color) &&
        !(p.wall & cell(row, column_of(row, color)));
}
inline bool legal(const State& s, Action a) noexcept {
    if (s.phase != Phase::draft || a >= action_count) return false;
    const int src = source_of(a), c = color_of(a), dst = destination_of(a);
    return s.sources[src][c] != 0 && (dst == floor || accepts(s.players[s.current], dst, c));
}
inline ActionList legal_actions(const State& s) noexcept {
    ActionList list;
    if (s.phase != Phase::draft) return list;
    // Compute destination compatibility once per color, reuse across six sources.
    std::array<std::uint8_t, 5> destinations{};
    for (int c = 0; c < 5; ++c) {
        unsigned mask = 1U << floor;
        for (int r = 0; r < 5; ++r) mask |= unsigned(accepts(s.players[s.current], r, c)) << r;
        destinations[c] = static_cast<std::uint8_t>(mask);
    }
    for (int src = 0; src < 6; ++src) for (int c = 0; c < 5; ++c) if (s.sources[src][c]) {
        unsigned mask = destinations[c];
        while (mask) {
            const int dst = std::countr_zero(mask);
            list.values[list.size++] = encode(src, c, dst);
            mask &= mask - 1;
        }
    }
    return list;
}
inline void action_mask(const State& s, std::uint8_t* out) noexcept {
    std::fill_n(out, action_count, std::uint8_t{0});
    for (auto a : legal_actions(s)) out[a] = 1;
}
inline void add_floor(State& s, Player& p, int color, int count) noexcept {
    const int kept = std::min(count, 7 - int(p.floor_count));
    p.floor_tiles[color] += static_cast<std::uint8_t>(kept);
    p.floor_count += static_cast<std::uint8_t>(kept);
    s.discard[color] += static_cast<std::uint8_t>(count - kept);
}
inline void settle_round(State& s) noexcept {
    constexpr int penalties[8] = {0, 1, 2, 4, 6, 8, 11, 14};
    for (auto& p : s.players) {
        int gain = 0;
        for (int row = 0; row < 5; ++row) if (p.pattern_count[row] == row + 1) {
            const int c = p.pattern_color[row];
            gain += place_and_score(p, row, c);
            s.discard[c] += static_cast<std::uint8_t>(row);
            p.pattern_count[row] = 0;
            p.pattern_color[row] = empty;
        }
        p.score = static_cast<std::uint16_t>(std::max(0, int(p.score) + gain - penalties[p.floor_count]));
        for (int c = 0; c < 5; ++c) {
            s.discard[c] += p.floor_tiles[c];
            p.floor_tiles[c] = 0;
        }
        p.floor_count = 0;
    }
    if (complete_rows(s.players[0].wall) || complete_rows(s.players[1].wall)) {
        for (auto& p : s.players) p.score += static_cast<std::uint16_t>(end_bonus(p.wall));
        s.phase = Phase::terminal;
    } else {
        s.current = s.next_start;
        s.phase = Phase::chance;
    }
}
// Requires legal(s,a). Stops at an explicit chance node after round settlement.
inline void apply_unchecked(State& s, Action a) noexcept {
    assert(legal(s, a));
    const int src = source_of(a), c = color_of(a), dst = destination_of(a);
    auto& p = s.players[s.current];
    const int count = s.sources[src][c];
    s.sources[src][c] = 0;
    s.tiles_remaining -= static_cast<std::uint8_t>(count);
    if (src != center) {
        for (int color = 0; color < 5; ++color) {
            s.sources[center][color] += s.sources[src][color];
            s.sources[src][color] = 0;
        }
    } else if (s.token_available) {
        s.token_available = 0;
        s.next_start = s.current;
        if (p.floor_count < 7) ++p.floor_count;
    }
    if (dst == floor) add_floor(s, p, c, count);
    else {
        const int kept = std::min(count, dst + 1 - int(p.pattern_count[dst]));
        p.pattern_color[dst] = static_cast<std::uint8_t>(c);
        p.pattern_count[dst] += static_cast<std::uint8_t>(kept);
        add_floor(s, p, c, count - kept);
    }
    ++s.ply;
    s.current ^= 1;
    if (s.tiles_remaining == 0) settle_round(s);
}
inline bool apply(State& s, Action a) noexcept {
    if (!legal(s, a)) return false;
    apply_unchecked(s, a);
    return true;
}
// Weighted sampling without replacement is distribution-equivalent to a shuffled bag.
inline bool deal(State& s) noexcept {
    if (s.phase != Phase::chance) return false;
    s.sources = {};
    s.tiles_remaining = 0;
    s.token_available = 1;
    s.current = s.next_start;
    ++s.round;
    for (int f = 0; f < 5; ++f) for (int j = 0; j < 4; ++j) {
        if (!s.bag_size) {
            s.bag = s.discard;
            s.discard = {};
            for (auto n : s.bag) s.bag_size += n;
            if (!s.bag_size) { s.phase = Phase::draft; return true; }
        }
        int draw = static_cast<int>(s.rng.bounded(s.bag_size));
        int c = 0;
        while (draw >= s.bag[c]) draw -= s.bag[c++];
        --s.bag[c]; --s.bag_size;
        ++s.sources[f][c]; ++s.tiles_remaining;
    }
    s.phase = Phase::draft;
    return true;
}
inline State initial_state(std::uint64_t seed, std::uint8_t starting_player = 0) noexcept {
    assert(starting_player < 2);
    State s;
    s.rng.state = seed;
    s.bag.fill(20); s.bag_size = 100;
    s.next_start = starting_player;
    deal(s);
    return s;
}
inline void step_unchecked(State& s, Action a) noexcept {
    apply_unchecked(s, a);
    if (s.phase == Phase::chance) deal(s);
}
inline bool step(State& s, Action a) noexcept {
    if (!legal(s, a)) return false;
    step_unchecked(s, a);
    return true;
}
// -2 unfinished, -1 shared victory, 0/1 winner; do not use score difference for ties.
inline int winner(const State& s) noexcept {
    if (s.phase != Phase::terminal) return -2;
    const auto& a = s.players[0]; const auto& b = s.players[1];
    if (a.score != b.score) return a.score > b.score ? 0 : 1;
    const int ar = complete_rows(a.wall), br = complete_rows(b.wall);
    return ar == br ? -1 : (ar > br ? 0 : 1);
}
inline float terminal_reward(const State& s, unsigned perspective) noexcept {
    const int w = winner(s);
    return w < 0 ? 0.0F : (w == int(perspective) ? 1.0F : -1.0F);
}
// Caller-owned 172 floats; only public state/history counts, never rng or future draws.
inline void observe(const State& s, unsigned perspective, float* out) noexcept {
    assert(perspective < 2);
    auto* o = out;
    for (const auto& source : s.sources) for (auto n : source) *o++ = n / 20.0F;
    for (unsigned i = 0; i < 2; ++i) {
        const auto& p = s.players[perspective ^ i];
        for (int bit = 0; bit < 25; ++bit) *o++ = float((p.wall >> bit) & 1U);
        for (int r = 0; r < 5; ++r) for (int c = 0; c < 5; ++c)
            *o++ = float(p.pattern_count[r] && p.pattern_color[r] == c);
        for (int r = 0; r < 5; ++r) *o++ = float(p.pattern_count[r]) / float(r + 1);
        for (auto n : p.floor_tiles) *o++ = n / 7.0F;
        *o++ = p.floor_count / 7.0F;
        *o++ = p.score / 200.0F;
    }
    for (auto n : s.bag) *o++ = n / 20.0F;
    for (auto n : s.discard) *o++ = n / 20.0F;
    *o++ = float(s.token_available);
    *o++ = float(s.next_start == perspective);
    *o++ = float(s.current == perspective);
    for (int p = 0; p < 3; ++p) *o++ = float(int(s.phase) == p);
    *o++ = float(s.round) / 10.0F;
    *o++ = s.bag_size / 100.0F;
    assert(o == out + observation_size);
}
// Expensive diagnostic. Call in tests/debugging, not the training hot path.
inline bool validate(const State& s) noexcept {
    if (s.current > 1 || s.next_start > 1 || s.token_available > 1 || int(s.phase) > 2) return false;
    std::array<int, 5> total{};
    int bag_n = 0, available = 0, floor_markers = 0;
    for (int c = 0; c < 5; ++c) {
        total[c] = s.bag[c] + s.discard[c]; bag_n += s.bag[c];
        for (const auto& src : s.sources) { total[c] += src[c]; available += src[c]; }
    }
    for (int f = 0; f < 5; ++f) {
        int n = 0; for (auto x : s.sources[f]) n += x;
        if (n > 4) return false;
    }
    for (unsigned i = 0; i < 2; ++i) {
        const auto& p = s.players[i];
        if (p.wall >> 25 || p.floor_count > 7) return false;
        int floor_n = 0;
        for (int c = 0; c < 5; ++c) {
            total[c] += p.floor_tiles[c] + std::popcount(p.wall & color_mask(c));
            floor_n += p.floor_tiles[c];
        }
        const int marker = int(p.floor_count) - floor_n;
        if (marker < 0 || marker > 1 || (marker && (s.token_available || i != s.next_start))) return false;
        floor_markers += marker;
        for (int r = 0; r < 5; ++r) {
            const int c = p.pattern_color[r], n = p.pattern_count[r];
            if (n > r + 1 || (n == 0 ? c != empty : c >= 5)) return false;
            if (n) { if (p.wall & cell(r, column_of(r, c))) return false; total[c] += n; }
        }
    }
    if (floor_markers > 1 || bag_n != s.bag_size || available != s.tiles_remaining) return false;
    for (auto n : total) if (n != 20) return false;
    if (s.phase != Phase::draft && available) return false;
    if (s.phase == Phase::draft && !available) return false;
    const bool ended = complete_rows(s.players[0].wall) || complete_rows(s.players[1].wall);
    return ended == (s.phase == Phase::terminal);
}
} // namespace azul
