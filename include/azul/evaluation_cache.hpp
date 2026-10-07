#pragma once
#include "search.hpp"

namespace azul {
// Worker-local direct-mapped cache: bounded memory, no shared locks. Exact
// semantic equality verifies hashes; factory permutation is undone on logits.
class EvaluationCache {
    struct Entry {
        State state{};
        std::array<float,action_count> logits{};
        float value=0;
        std::uint64_t hash=0,version=0;
        bool valid=false;
    };
    std::vector<Entry> entries;
    static State canonical(State s,std::array<int,5>& order,bool symmetric) {
        s.rng.state=0;s.ply=0;
        for(int i=0;i<5;++i) order[i]=i;
        if(symmetric) {
            std::sort(order.begin(),order.end(),[&](int a,int b) {return s.sources[a]<s.sources[b];});
            const auto sources=s.sources;
            for(int i=0;i<5;++i) s.sources[i]=sources[order[i]];
        }
        return s;
    }
public:
    struct Key {State state;std::array<int,5> order;std::uint64_t hash;};
    std::uint64_t hits=0,misses=0;
    bool symmetric=false;
    explicit EvaluationCache(std::size_t capacity=0):entries(capacity) {}
    bool get(const State& input,std::uint64_t version,float* logits,float& value,Key* prepared=nullptr) {
        if(entries.empty()) return false;
        std::array<int,5> order;const auto s=canonical(input,order,symmetric);const auto h=public_hash(s);
        if(prepared) *prepared={s,order,h};
        const auto& e=entries[h%entries.size()];
        if(!e.valid || e.hash!=h || e.version!=version || !(e.state==s)) {++misses;return false;}
        for(int i=0;i<5;++i) std::copy_n(e.logits.data()+i*30,30,logits+order[i]*30);
        std::copy_n(e.logits.data()+150,30,logits+150);value=e.value;++hits;return true;
    }
    void put(const State& input,std::uint64_t version,const float* logits,float value) {
        if(entries.empty()) return;
        std::array<int,5> order;const auto s=canonical(input,order,symmetric);const auto h=public_hash(s);
        put_prepared({s,order,h},version,logits,value);
    }
    void put_prepared(const Key& key,std::uint64_t version,const float* logits,float value) {
        if(entries.empty()) return;
        auto& e=entries[key.hash%entries.size()];e.state=key.state;e.hash=key.hash;e.version=version;e.valid=true;e.value=value;
        for(int i=0;i<5;++i) std::copy_n(logits+key.order[i]*30,30,e.logits.data()+i*30);
        std::copy_n(logits+150,30,e.logits.data()+150);
    }
};
}
