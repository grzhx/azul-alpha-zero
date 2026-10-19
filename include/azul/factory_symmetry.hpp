#pragma once
#include <array>
#include <cstdint>
#include <tuple>

namespace azul_symmetry {
// Only factories are interchangeable. Center, colors, player boards and RNG
// are untouched. Stable ties keep equal factories' external action mapping.
template<class State>
inline auto canonicalize_factories(State& state) noexcept {
    constexpr int count = int(std::tuple_size_v<decltype(state.sources)>) - 1;
    std::array<std::uint8_t, count> order{}; // canonical slot -> original slot
    std::array<std::uint32_t, count> keys{};
    for(int i=0;i<count;++i) {
        order[i]=std::uint8_t(i);
        std::uint32_t key=0;
        // Factory counts are <=4. Five 5-bit fields preserve lexicographic order.
        for(auto value:state.sources[i]) key=(key<<5)|value;
        keys[i]=key;
        const auto source=state.sources[i];const auto original=order[i];
        int j=i;
        while(j>0 && keys[j-1]>key) {
            keys[j]=keys[j-1];order[j]=order[j-1];state.sources[j]=state.sources[j-1];--j;
        }
        keys[j]=key;order[j]=original;state.sources[j]=source;
    }
    return order;
}

template<std::size_t Count>
inline std::uint16_t external_action(std::uint16_t action,
                                     const std::array<std::uint8_t,Count>& order) noexcept {
    const auto source=action/30;
    return source<Count?std::uint16_t(order[source]*30+action%30):action;
}
}
