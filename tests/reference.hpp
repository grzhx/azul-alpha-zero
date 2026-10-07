#pragma once
#include "azul/engine.hpp"
#include <algorithm>

// Deliberately slow independent rules implementation. Shares only the storage
// format; no production legality, wall scoring, settlement or drawing helpers.
namespace reference {
inline bool legal(const azul::State& s, unsigned a) {
    if (a >= 180 || s.phase != azul::Phase::draft) return false;
    const unsigned src = a / 30, c = (a % 30) / 6, dst = a % 6;
    if (!s.sources[src][c]) return false;
    if (dst == 5) return true;
    const auto& p = s.players[s.current];
    if (p.pattern_count[dst] >= dst + 1) return false;
    if (p.pattern_count[dst] && p.pattern_color[dst] != c) return false;
    return ((p.wall >> (dst * 5 + (c + dst) % 5)) & 1U) == 0;
}
inline void apply(azul::State& s, unsigned a) {
    const int src = int(a / 30), c = int((a % 30) / 6), dst = int(a % 6);
    const int actor = s.current;
    auto& p = s.players[actor];
    const int n = s.sources[src][c];
    s.sources[src][c] = 0;
    if (src < 5) for (int k = 0; k < 5; ++k) {
        s.sources[5][k] += s.sources[src][k]; s.sources[src][k] = 0;
    }
    else if (s.token_available) {
        s.token_available = 0; s.next_start = static_cast<std::uint8_t>(actor);
        if (p.floor_count != 7) ++p.floor_count;
    }
    for (int k = 0; k < n; ++k) {
        if (dst < 5 && p.pattern_count[dst] < dst + 1) {
            p.pattern_color[dst] = static_cast<std::uint8_t>(c); ++p.pattern_count[dst];
        } else if (p.floor_count < 7) { ++p.floor_count; ++p.floor_tiles[c]; }
        else ++s.discard[c];
    }
    s.tiles_remaining = 0;
    for (const auto& source : s.sources) for (auto k : source) s.tiles_remaining += k;
    ++s.ply; s.current = static_cast<std::uint8_t>(1 - actor);
    if (s.tiles_remaining) return;
    bool end = false;
    for (auto& player : s.players) {
        bool grid[5][5]{};
        for (int r = 0; r < 5; ++r) for (int col = 0; col < 5; ++col)
            grid[r][col] = (player.wall >> (r * 5 + col)) & 1U;
        for (int r = 0; r < 5; ++r) if (player.pattern_count[r] == r + 1) {
            const int color = player.pattern_color[r], col = (r + color) % 5;
            grid[r][col] = true;
            int h = 1, v = 1;
            for (int x = col - 1; x >= 0 && grid[r][x]; --x) ++h;
            for (int x = col + 1; x < 5 && grid[r][x]; ++x) ++h;
            for (int y = r - 1; y >= 0 && grid[y][col]; --y) ++v;
            for (int y = r + 1; y < 5 && grid[y][col]; ++y) ++v;
            int points = 0;
            if (h > 1) points += h;
            if (v > 1) points += v;
            if (!points) points = 1;
            player.score += static_cast<std::uint16_t>(points);
            s.discard[color] += static_cast<std::uint8_t>(r);
            player.pattern_color[r] = 5; player.pattern_count[r] = 0;
        }
        int penalty = 0;
        const int costs[7] = {1,1,2,2,2,3,3};
        for (int k = 0; k < player.floor_count; ++k) penalty += costs[k];
        player.score = static_cast<std::uint16_t>(std::max(0, int(player.score) - penalty));
        for (int k = 0; k < 5; ++k) { s.discard[k] += player.floor_tiles[k]; player.floor_tiles[k] = 0; }
        player.floor_count = 0; player.wall = 0;
        for (int r = 0; r < 5; ++r) {
            int count = 0;
            for (int col = 0; col < 5; ++col) if (grid[r][col]) {
                player.wall |= 1U << (r * 5 + col); ++count;
            }
            if (count == 5) end = true;
        }
    }
    if (end) {
        s.phase = azul::Phase::terminal;
        for (auto& player : s.players) {
            int rows[5]{}, columns[5]{}, color_counts[5]{};
            for (int r = 0; r < 5; ++r) for (int col = 0; col < 5; ++col)
                if ((player.wall >> (r * 5 + col)) & 1U) {
                    ++rows[r]; ++columns[col]; ++color_counts[(col + 5 - r) % 5];
                }
            for (int i = 0; i < 5; ++i)
                player.score += static_cast<std::uint16_t>((rows[i] == 5 ? 2 : 0) +
                    (columns[i] == 5 ? 7 : 0) + (color_counts[i] == 5 ? 10 : 0));
        }
    } else { s.phase = azul::Phase::chance; s.current = s.next_start; }
}
}
