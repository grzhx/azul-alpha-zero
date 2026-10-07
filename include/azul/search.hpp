#pragma once
// Sequential Halving scheduling and completed-Q / deterministic-selection equations
// are adapted from DeepMind Mctx (Apache-2.0). See THIRD_PARTY_NOTICES.md.
// Chance-node integration and the Azul arena/request protocol are project code.
#include "engine.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace azul {
struct SearchConfig {
    int simulations = 128, candidates = 16, max_depth = 96;
    int chance_initial = 2, chance_cap = 64;
    float gumbel_scale = 1.0F, value_scale = 0.1F, maxvisit_init = 50.0F;
};
struct SearchTuning {
    int q_mode = 0; // 0 original, 1 range floor, 2 fixed [-1,1]
    float q_floor = 0.25F, chance_coefficient = 1.0F, chance_exponent = 0.5F;
    float chance_sensitivity = 0.0F, afterstate_prior = 0.0F;
    bool subtree_reuse = false;
};
inline bool valid_config(const SearchConfig& c) noexcept {
    return c.simulations >= 2 && c.simulations <= 4096 && c.candidates >= 1 &&
        c.candidates <= action_count && c.candidates <= c.simulations &&
        c.max_depth >= 1 && c.max_depth <= 512 && c.chance_initial >= 1 &&
        c.chance_cap >= c.chance_initial && c.chance_cap <= 256 &&
        std::isfinite(c.gumbel_scale) && c.gumbel_scale >= 0 && c.gumbel_scale <= 1000 &&
        std::isfinite(c.value_scale) && c.value_scale > 0 && c.value_scale <= 1000 &&
        std::isfinite(c.maxvisit_init) && c.maxvisit_init >= 0 && c.maxvisit_init <= 1000000;
}
// Visit-level Sequential Halving schedule (Full Gumbel). Not PUCT + root noise.
inline std::vector<int> considered_visits(int candidates, int budget) {
    std::vector<int> out; out.reserve(budget);
    if (candidates <= 1) { for (int n=0; n<budget; ++n) out.push_back(n); return out; }
    const int rounds = int(std::ceil(std::log2(candidates)));
    std::vector<int> counts(candidates, 0);
    int active = candidates;
    while (int(out.size()) < budget) {
        const int extra = std::max(1, budget / (rounds * active));
        for (int k=0; k<extra && int(out.size())<budget; ++k)
            for (int i=0; i<active && int(out.size())<budget; ++i) out.push_back(counts[i]++);
        active = std::max(2, active/2);
    }
    return out;
}
inline float unit_open_from_bits(std::uint64_t bits) noexcept {
    // The largest double midpoint rounds to 1.0 in float. Keep log(-log(u))
    // finite even at that endpoint; all other existing samples are unchanged.
    return std::min(float((double(bits >> 40) + 0.5) / 16777216.0), 0x1.fffffep-1F);
}
inline float unit_open(Rng& r) noexcept {
    return unit_open_from_bits(r.next64());
}
// Hash semantic public fields, never padding or simulator RNG. Includes scores.
inline std::uint64_t public_hash(const State& s) noexcept {
    std::uint64_t h = UINT64_C(0xcbf29ce484222325);
    const auto mix=[&](std::uint64_t n) {h=(h^n)*UINT64_C(0x100000001b3);};
    for(const auto& p:s.players) {
        mix(p.wall);mix(p.score);
        for(auto x:p.pattern_color) mix(x);
        for(auto x:p.pattern_count) mix(x);
        for(auto x:p.floor_tiles) mix(x);
        mix(p.floor_count);
    }
    for(const auto& src:s.sources) for(auto x:src) mix(x);
    for(auto x:s.bag) mix(x);
    for(auto x:s.discard) mix(x);
    mix(s.round);mix(s.bag_size);mix(s.tiles_remaining);mix(s.current);
    mix(s.next_start);mix(s.token_available);mix(unsigned(s.phase));
    return h;
}
struct ChanceEdge { int child=-1; unsigned multiplicity=0; };
struct SearchEdge {
    int child = -1;
    unsigned visits = 0;
    float sum0 = 0, logit = 0, prior = 0;
    Action action = 0;
};
struct SearchNode {
    State state;
    std::size_t first = 0;
    int count = 0, capacity = 0;
    unsigned visits = 0, samples = 0;
    float sum0 = 0, raw0 = 0;
    bool expanded = false;
    std::uint64_t sample_key = 0;
    float mean0() const noexcept { return visits ? sum0/float(visits) : raw0; }
};
struct SearchStats {
    std::uint64_t simulations = 0, evaluations = 0, chance_draws = 0;
    std::uint64_t chance_reuses = 0, chance_duplicates = 0, nodes = 0;
    std::uint64_t reused_nodes=0, afterstate_evaluations=0;
};
// An independent tree per root. Arenas and path buffers retain capacity across moves.
// All values stored in absolute player-0 perspective; convert only at decisions.
class SearchTree {
public:
    SearchConfig config;
    SearchTuning tuning;
    SearchStats stats;
    std::vector<SearchNode> nodes;
    std::vector<SearchEdge> edges;
    std::vector<ChanceEdge> chance_edges;
    int pending = -1, completed = 0;

    explicit SearchTree(SearchConfig c = {}) : config(c) {
        if (!valid_config(c)) throw std::invalid_argument("invalid search config");
        nodes.reserve(std::size_t(c.simulations)*2+1);
        edges.reserve(std::size_t(c.simulations)*48);
        path_nodes.reserve(c.max_depth*2+2); path_edges.reserve(c.max_depth*2+2);
    }
    void reset(State s, std::uint64_t seed) {
        nodes.clear(); edges.clear(); chance_edges.clear(); path_nodes.clear(); path_edges.clear();
        stats = {}; pending = -1; completed = 0; seed_ = seed;
        disabled_=false;root_base.fill(0);
        select_rng.state = episode_seed(seed, 1);
        s.rng.state = 0; // impossible for search to peek at the real episode RNG
        add_node(s);
        root_gumbel.fill(0);
        Rng noise{episode_seed(seed, 2)};
        for (auto& x : root_gumbel)
            x = config.gumbel_scale == 0 ? 0 : -config.gumbel_scale*std::log(-std::log(unit_open(noise)));
        if (s.phase == Phase::terminal) completed = config.simulations;
        else if (s.phase != Phase::draft) throw std::invalid_argument("root must be a decision");
    }
    void deactivate() {disabled_=true;completed=config.simulations;pending=-1;}
    void begin(State s,std::uint64_t seed,std::uint64_t version) {
        s.rng.state=0;
        int found=-1;
        if(tuning.subtree_reuse && version==version_ && !disabled_ && s.phase==Phase::draft) {
            for(int i=0;i<int(nodes.size());++i) if(nodes[i].state==s && nodes[i].expanded) {found=i;break;}
        }
        if(found<0) {reset(s,seed);version_=version;return;}
        scratch_nodes.clear();scratch_edges.clear();scratch_chance.clear();
        remap.assign(nodes.size(),-1);old_ids.clear();old_ids.push_back(found);remap[found]=0;
        scratch_nodes.push_back(nodes[found]);
        for(std::size_t j=0;j<old_ids.size();++j) {
            const auto original=nodes[old_ids[j]];
            auto copy=original;
            const bool chance=original.state.phase==Phase::chance;
            copy.first=chance?scratch_chance.size():scratch_edges.size();copy.capacity=copy.count;
            for(int k=0;k<original.count;++k) {
                int child=chance?chance_edges[original.first+k].child:edges[original.first+k].child;
                if(child>=0 && remap[child]<0) {remap[child]=int(scratch_nodes.size());old_ids.push_back(child);scratch_nodes.push_back(nodes[child]);}
                if(chance) {auto e=chance_edges[original.first+k];e.child=child<0?-1:remap[child];scratch_chance.push_back(e);}
                else {auto e=edges[original.first+k];e.child=child<0?-1:remap[child];scratch_edges.push_back(e);}
            }
            scratch_nodes[j]=copy;
        }
        nodes.swap(scratch_nodes);edges.swap(scratch_edges);chance_edges.swap(scratch_chance);
        stats={};stats.reused_nodes=nodes.size();stats.nodes=nodes.size();pending=-1;completed=0;
        path_nodes.clear();path_edges.clear();seed_=seed;disabled_=false;version_=version;
        select_rng.state=episode_seed(seed,1);Rng noise{episode_seed(seed,2)};
        for(auto& x:root_gumbel) x=config.gumbel_scale==0?0:-config.gumbel_scale*std::log(-std::log(unit_open(noise)));
        root_base.fill(0);
        for(int i=0;i<nodes[0].count;++i) root_base[i]=edges[nodes[0].first+i].visits;
        ensure_schedule();
    }
    bool done() const noexcept { return !nodes.empty() && completed == config.simulations; }
    // Advances until one network request or completion. Repeated request is idempotent.
    bool request(float* observation) {
        if (pending >= 0) { observe(nodes[pending].state,nodes[pending].state.current,observation); return true; }
        while (!done()) {
            path_nodes.clear(); path_edges.clear();
            int index = 0, depth = 0;
            while (true) {
                path_nodes.push_back(index);
                const auto phase = nodes[index].state.phase;
                if (phase == Phase::terminal) { backup(terminal_reward(nodes[index].state,0)); break; }
                if (phase == Phase::chance) {
                    if(tuning.afterstate_prior>0 && !nodes[index].expanded) {
                        pending=index;observe(nodes[index].state,nodes[index].state.current,observation);return true;
                    }
                    const auto e = chance_edge(index);
                    path_edges.push_back(e); index = chance_edges[e].child;
                    continue;
                }
                if (!nodes[index].expanded) {
                    pending = index;
                    observe(nodes[index].state,nodes[index].state.current,observation);
                    return true;
                }
                if (depth >= config.max_depth) { backup(nodes[index].raw0); break; }
                const auto edge = choose(index);
                path_edges.push_back(edge);
                if (edges[edge].child < 0) {
                    State next = nodes[index].state;
                    apply_unchecked(next,edges[edge].action);
                    edges[edge].child = add_node(next);
                }
                index = edges[edge].child; ++depth;
            }
        }
        return false;
    }
    // logits[180], scalar P(win)-P(loss) from requested state's current-player view.
    void submit(const float* logits, float value, bool policy_validated=false) {
        if (pending < 0 || !std::isfinite(value) || value < -1.00001F || value > 1.00001F)
            throw std::invalid_argument("invalid network response");
        const int index = pending;
        if(nodes[index].state.phase==Phase::chance) {
            auto& n=nodes[index];n.raw0=n.state.current?-value:value;n.expanded=true;
            pending=-1;++stats.evaluations;++stats.afterstate_evaluations;
            return; // Resume the same simulation; its actual chance sample still must be searched.
        }
        const auto list = legal_actions(nodes[index].state);
        float maximum = -std::numeric_limits<float>::infinity();
        for (auto a : list) {
            if (!policy_validated && !std::isfinite(logits[a])) throw std::invalid_argument("nonfinite policy");
            maximum = std::max(maximum,logits[a]);
        }
        auto& node = nodes[index];
        node.first = edges.size(); node.count = list.size;
        float normalizer = 0;
        for (auto a : list) {
            SearchEdge e; e.action = a; e.logit = float(std::max(-1e9,double(logits[a])-double(maximum)));
            e.prior = std::max(std::exp(e.logit),std::numeric_limits<float>::min());
            normalizer += e.prior; edges.push_back(e);
        }
        for (std::size_t e=node.first; e<node.first+node.count; ++e) edges[e].prior /= normalizer;
        node.raw0 = node.state.current ? -value : value; node.expanded = true;
        pending = -1; ++stats.evaluations;
        if (index == 0) {
            ensure_schedule();
            // Root evaluation is not one of the allocated action simulations.
            path_nodes.clear(); path_edges.clear();
        } else backup(node.raw0);
    }
    void result(float* policy, Action& action, float& root_value) const {
        if (!done()) throw std::logic_error("search unfinished");
        std::fill_n(policy, action_count, 0.0F);
        if(disabled_) {action=65535;root_value=0;return;}
        if (nodes[0].state.phase == Phase::terminal) {
            action = 65535; root_value = terminal_reward(nodes[0].state,nodes[0].state.current); return;
        }
        std::array<float,action_count> q{}, weights{};
        transformed(0,q.data()); improved(0,q.data(),weights.data());
        const auto& root = nodes[0]; unsigned most = 0;
        for (int i=0;i<root.count;++i) most = std::max(most,edges[root.first+i].visits-root_base[i]);
        float best = -std::numeric_limits<float>::infinity(); action = 65535;
        for (int i=0;i<root.count;++i) {
            const auto& e = edges[root.first+i]; policy[e.action] = weights[i];
            const float score = e.logit + root_gumbel[e.action] + q[i];
            if (e.visits-root_base[i] == most && score > best) { best=score; action=e.action; }
        }
        root_value = root.mean0() * (root.state.current ? -1.0F : 1.0F);
    }
    // Exposed for equation-level tests and search diagnostics.
    void transformed(int index, float* out) const {
        const auto& n = nodes[index]; const float sign = n.state.current ? -1.0F : 1.0F;
        unsigned total = 0, most = 0; double p_sum = 0, weighted = 0;
        for (int i=0;i<n.count;++i) {
            const auto& e = edges[n.first+i];
            total += e.visits; most = std::max(most,e.visits-(index==0?root_base[i]:0));
            if (e.visits) { p_sum += e.prior; weighted += double(e.prior)*edge_value0(e)*sign; }
        }
        const float mixed = float((n.raw0*sign + total*(p_sum ? weighted/p_sum : 0)) / (total+1.0));
        float low = 1e30F, high = -1e30F;
        for (int i=0;i<n.count;++i) {
            const auto& e = edges[n.first+i];
            out[i] = e.visits ? edge_value0(e)*sign : mixed;
            low = std::min(low,out[i]); high = std::max(high,out[i]);
        }
        const float denominator=tuning.q_mode==2?2.0F:std::max(high-low,tuning.q_mode==1?tuning.q_floor:1e-8F);
        const float scale = (config.maxvisit_init+float(most))*config.value_scale/denominator;
        for (int i=0;i<n.count;++i) out[i] = (out[i]-low)*scale;
    }
private:
    bool disabled_=false;
    std::uint64_t version_=0;
    std::array<unsigned,action_count> root_base{};
    std::vector<SearchNode> scratch_nodes;
    std::vector<SearchEdge> scratch_edges;
    std::vector<ChanceEdge> scratch_chance;
    std::vector<int> remap,old_ids;
    int schedule_candidates_=-1,schedule_budget_=-1;
    void ensure_schedule() {
        const int candidates=std::min(config.candidates,nodes[0].count);
        if(candidates!=schedule_candidates_ || schedule_budget_!=config.simulations) {
            schedule=considered_visits(candidates,config.simulations);
            schedule_candidates_=candidates;schedule_budget_=config.simulations;
        }
    }
    std::uint64_t seed_ = 0;
    Rng select_rng;
    std::array<float,action_count> root_gumbel{};
    std::vector<int> schedule, path_nodes;
    std::vector<std::size_t> path_edges;
    int add_node(const State& s) {
        SearchNode n; n.state=s; n.state.rng.state=0;
        if (s.phase == Phase::chance) {
            n.first=chance_edges.size();n.capacity=std::min(config.chance_cap,4);
            chance_edges.resize(chance_edges.size()+n.capacity);
            n.sample_key=episode_seed(seed_^public_hash(s),0);
        }
        nodes.push_back(n); stats.nodes=nodes.size(); return int(nodes.size()-1);
    }
    float edge_value0(const SearchEdge& e) const noexcept {
        // Use the current weighted chance estimate, not a stale average of old pools.
        if (e.child >= 0 && nodes[e.child].state.phase == Phase::chance) return nodes[e.child].mean0();
        return e.sum0/float(e.visits);
    }
    void improved(int index,const float* q,float* out) const {
        const auto& n=nodes[index]; float maximum=-1e30F,total=0;
        for(int i=0;i<n.count;++i) maximum=std::max(maximum,edges[n.first+i].logit+q[i]);
        for(int i=0;i<n.count;++i) { out[i]=std::exp(edges[n.first+i].logit+q[i]-maximum); total+=out[i]; }
        for(int i=0;i<n.count;++i) out[i]/=total;
    }
    std::size_t choose(int index) const {
        const auto& n=nodes[index]; std::array<float,action_count> q{},weights{};
        transformed(index,q.data());
        if(index) improved(index,q.data(),weights.data());
        unsigned total=0;
        for(int i=0;i<n.count;++i) total+=edges[n.first+i].visits;
        float best=-std::numeric_limits<float>::infinity(); std::size_t chosen=n.first;
        for(int i=0;i<n.count;++i) {
            const auto& e=edges[n.first+i]; float score;
            if(!index) {
                if(int(e.visits-root_base[i])!=schedule[completed]) continue;
                score=e.logit+root_gumbel[e.action]+q[i];
            } else score=weights[i]-float(e.visits)/float(total+1);
            if(score>best) {best=score;chosen=n.first+i;}
        }
        return chosen;
    }
    std::size_t chance_edge(int index) {
        float variance=0,mean=0;
        const auto& n=nodes[index];
        if(tuning.chance_sensitivity>0) {
            for(int i=0;i<n.count;++i) {const auto& e=chance_edges[n.first+i];const float v=nodes[e.child].mean0();mean+=e.multiplicity*v;variance+=e.multiplicity*v*v;}
            if(n.samples) {mean/=n.samples;variance=std::max(0.0F,variance/n.samples-mean*mean);}
        }
        const double power=tuning.chance_exponent==0.5F?std::sqrt(double(n.visits+1)):std::pow(double(n.visits+1),tuning.chance_exponent);
        const double growth=tuning.chance_coefficient*power*(1.0+tuning.chance_sensitivity*std::sqrt(variance));
        const unsigned target=unsigned(std::min(double(config.chance_cap),std::max(double(config.chance_initial),std::ceil(growth))));
        auto first=nodes[index].first;
        if(nodes[index].samples<target) {
            State next=nodes[index].state;
            next.rng.state=episode_seed(nodes[index].sample_key,nodes[index].samples);
            deal(next); next.rng.state=0;
            ++nodes[index].samples; ++stats.chance_draws;
            for(int i=0;i<nodes[index].count;++i) if(nodes[chance_edges[first+i].child].state==next) {
                ++chance_edges[first+i].multiplicity; ++stats.chance_duplicates; return first+i;
            }
            if(nodes[index].count==nodes[index].capacity) {
                const int capacity=std::min(config.chance_cap,std::max(4,nodes[index].capacity*2));
                const auto new_first=chance_edges.size();chance_edges.resize(new_first+capacity);
                std::copy_n(chance_edges.begin()+first,nodes[index].count,chance_edges.begin()+new_first);
                nodes[index].first=first=new_first;nodes[index].capacity=capacity;
            }
            const auto edge=first+nodes[index].count++;
            chance_edges[edge].child=add_node(next); chance_edges[edge].multiplicity=1; return edge;
        }
        ++stats.chance_reuses;
        unsigned draw=select_rng.bounded(nodes[index].samples);
        for(int i=0;i<nodes[index].count;++i) {
            const auto e=first+i;
            if(draw<chance_edges[e].multiplicity) return e;
            draw-=chance_edges[e].multiplicity;
        }
        throw std::logic_error("chance sample accounting");
    }
    void backup(float value0) {
        for(int k=int(path_nodes.size())-1;k>=0;--k) {
            auto& n=nodes[path_nodes[k]];
            ++n.visits;
            if(n.state.phase==Phase::chance) {
                const double prior=n.expanded?tuning.afterstate_prior:0;
                double weighted=prior*n.raw0;
                for(int i=0;i<n.count;++i) {
                    const auto& e=chance_edges[n.first+i];
                    weighted+=double(e.multiplicity)*nodes[e.child].mean0();
                }
                value0=float(weighted/(n.samples+prior)); n.sum0=value0*float(n.visits);
            } else n.sum0+=value0;
            if(k>0 && nodes[path_nodes[k-1]].state.phase!=Phase::chance) { auto& e=edges[path_edges[k-1]]; ++e.visits; e.sum0+=value0; }
        }
        ++completed; ++stats.simulations;
    }
};
} // namespace azul
