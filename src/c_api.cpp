#include "azul/c_api.h"
#include "azul/engine.hpp"
#include "azul/search.hpp"
#include "azul/search3.hpp"
#include "azul/evaluation_cache.hpp"
#include <memory>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

static bool finite_policy(const float* p) noexcept {
#if defined(_M_X64) || defined(__x86_64__)
    const auto limit=_mm_set1_ps(std::numeric_limits<float>::max());
    const auto sign=_mm_set1_ps(-0.0F);
    auto valid=_mm_cmpeq_ps(_mm_setzero_ps(),_mm_setzero_ps());
    for(int i=0;i<180;i+=4) valid=_mm_and_ps(valid,_mm_cmple_ps(_mm_andnot_ps(sign,_mm_loadu_ps(p+i)),limit));
    return _mm_movemask_ps(valid)==15;
#else
    for(int i=0;i<180;++i) if(!std::isfinite(p[i])) return false;
    return true;
#endif
}
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

// One persistent pool per batch. Each worker owns a contiguous disjoint range.
// Only the dispatch boundary synchronizes; there is no per-game/per-action lock.
class Pool {
public:
    using Job = void (*)(void*, std::size_t, std::size_t) noexcept;
    Pool(std::size_t size, unsigned count) : size_(size), count_(count) {
        try {
            for (unsigned i = 1; i < count_; ++i) workers_.emplace_back([this, i] { worker(i); });
        } catch (...) {
            stop();
            throw;
        }
    }
    ~Pool() { stop(); }
    void run(Job job, void* context) {
        if (count_ == 1) { job(context, 0, size_); return; }
        {
            std::lock_guard lock(mutex_);
            job_ = job; context_ = context; pending_ = count_ - 1; ++epoch_;
        }
        start_.notify_all();
        job(context, 0, boundary(1));
        std::unique_lock lock(mutex_);
        done_.wait(lock, [this] { return pending_ == 0; });
    }
private:
    std::size_t boundary(unsigned i) const noexcept {
        return (size_ / count_) * i + std::min<std::size_t>(size_ % count_, i);
    }
    void worker(unsigned i) noexcept {
        std::uint64_t seen = 0;
        for (;;) {
            std::unique_lock lock(mutex_);
            start_.wait(lock, [&] { return stopping_ || epoch_ != seen; });
            if (stopping_) return;
            seen = epoch_; const auto job = job_; void* ctx = context_;
            lock.unlock();
            job(ctx, boundary(i), boundary(i + 1));
            lock.lock();
            if (--pending_ == 0) done_.notify_one();
        }
    }
    void stop() noexcept {
        { std::lock_guard lock(mutex_); stopping_ = true; }
        start_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();
    }
    std::size_t size_;
    unsigned count_, pending_ = 0;
    std::uint64_t epoch_ = 0;
    bool stopping_ = false;
    Job job_ = nullptr;
    void* context_ = nullptr;
    std::mutex mutex_;
    std::condition_variable start_, done_;
    std::vector<std::thread> workers_;
};
struct AzulBatch {
    std::vector<azul::State> states;
    Pool pool;
    AzulBatch(std::size_t n, unsigned threads) : states(n), pool(n, threads) {}
};
extern "C" {
int azul_abi_version(void) { return AZUL_ABI_VERSION; }
AzulBatch* azul_batch_create(size_t n, unsigned threads, uint64_t base_seed) {
    if (n == 0 || n > std::numeric_limits<std::size_t>::max() / sizeof(azul::State)) return nullptr;
    if (!threads) threads = std::max(1U, std::thread::hardware_concurrency());
    threads = static_cast<unsigned>(std::min<std::size_t>(threads, n));
    try {
        auto* b = new AzulBatch(n, threads);
        if (azul_batch_reset(b, base_seed)) { delete b; return nullptr; }
        return b;
    } catch (...) { return nullptr; }
}
void azul_batch_destroy(AzulBatch* b) { delete b; }
size_t azul_batch_size(const AzulBatch* b) { return b ? b->states.size() : 0; }
int azul_batch_reset(AzulBatch* b, uint64_t seed) {
    if (!b) return -1;
    struct Context { AzulBatch* b; uint64_t seed; } ctx{b, seed};
    try {
        b->pool.run([](void* ptr, size_t first, size_t last) noexcept {
            const auto& c = *static_cast<Context*>(ptr);
            for (auto i = first; i < last; ++i)
                c.b->states[i] = azul::initial_state(azul::episode_seed(c.seed, i), static_cast<uint8_t>(i & 1));
        }, &ctx);
        return 0;
    } catch (...) { return -1; }
}
int azul_batch_reset_at(AzulBatch* b, size_t index, uint64_t seed, unsigned starting_player) {
    if (!b || index >= b->states.size() || starting_player > 1) return -1;
    b->states[index] = azul::initial_state(seed, static_cast<uint8_t>(starting_player));
    return 0;
}
int azul_batch_observe(AzulBatch* b, float* observations, uint8_t* masks, uint8_t* players) {
    if (!b) return -1;
    struct Context { AzulBatch* b; float* o; uint8_t* m; uint8_t* p; } ctx{b, observations, masks, players};
    try {
        b->pool.run([](void* ptr, size_t first, size_t last) noexcept {
            const auto& c = *static_cast<Context*>(ptr);
            for (auto i = first; i < last; ++i) {
                const auto& s = c.b->states[i];
                if (c.o) azul::observe(s, s.current, c.o + i * azul::observation_size);
                if (c.m) azul::action_mask(s, c.m + i * azul::action_count);
                if (c.p) c.p[i] = s.current;
            }
        }, &ctx);
        return 0;
    } catch (...) { return -1; }
}
int azul_batch_step(AzulBatch* b, const uint16_t* actions, AzulStepResult* results) {
    if (!b || !actions) return -1;
    struct Context { AzulBatch* b; const uint16_t* a; AzulStepResult* r; } ctx{b, actions, results};
    try {
        b->pool.run([](void* ptr, size_t first, size_t last) noexcept {
            const auto& c = *static_cast<Context*>(ptr);
            for (auto i = first; i < last; ++i) {
                auto& s = c.b->states[i];
                const auto actor = s.current;
                const auto round = s.round;
                const bool valid = azul::step(s, c.a[i]);
                if (c.r) {
                    auto& r = c.r[i];
                    r = {};
                    r.actor = actor;
                    r.terminated = s.phase == azul::Phase::terminal;
                    r.round_finished = valid && (s.round != round || r.terminated);
                    r.invalid_action = !valid;
                    r.reward = valid ? azul::terminal_reward(s, actor) : 0.0F;
                    r.scores[0] = s.players[0].score; r.scores[1] = s.players[1].score;
                }
            }
        }, &ctx);
        return 0;
    } catch (...) { return -1; }
}
size_t azul_snapshot_size(void) { return sizeof(azul::State); }
int azul_batch_snapshot(AzulBatch* b, size_t i, void* buffer, size_t bytes) {
    if (!b || i >= b->states.size() || !buffer || bytes != sizeof(azul::State)) return -1;
    std::memcpy(buffer, &b->states[i], bytes);
    return 0;
}
int azul_batch_restore(AzulBatch* b, size_t i, const void* buffer, size_t bytes) {
    if (!b || i >= b->states.size() || !buffer || bytes != sizeof(azul::State)) return -1;
    azul::State s;
    std::memcpy(&s, buffer, bytes);
    // Batch API deals automatically and must not accept a chance node.
    if (!azul::validate(s) || s.phase == azul::Phase::chance) return -1;
    b->states[i] = s;
    return 0;
}
}

struct AzulSearch {
    std::vector<azul::SearchTree> trees;
    std::vector<uint8_t> ready, errors;
    Pool pool;
    std::vector<std::unique_ptr<azul::EvaluationCache>> caches;
    std::vector<azul::EvaluationCache::Key> pending_keys;
    unsigned workers;
    int cache_capacity=0;
    bool symmetric_cache=false,optimized=false;
    uint64_t model_version=0;
    bool begun = false, failed = false;
    AzulSearch(size_t n, unsigned threads, azul::SearchConfig c) : ready(n), errors(n), pool(n,threads), caches(n),pending_keys(n),workers(threads) {
        trees.reserve(n);
        for(size_t i=0;i<n;++i) trees.emplace_back(c);
    }
    bool ok() {
        for(auto e:errors) if(e) failed=true;
        return !failed;
    }
};
extern "C" {
int azul_search_abi_version(void) { return 1; }
AzulSearch* azul_search_create(size_t n,unsigned threads,const AzulSearchConfig* cfg) {
    if(!cfg || !n || n>65536) return nullptr;
    azul::SearchConfig c{cfg->simulations,cfg->candidates,cfg->max_depth,cfg->chance_initial,
        cfg->chance_cap,cfg->gumbel_scale,cfg->value_scale,cfg->maxvisit_init};
    if(!azul::valid_config(c)) return nullptr;
    if(!threads) threads=std::max(1U,std::thread::hardware_concurrency());
    threads=unsigned(std::min<size_t>(threads,n));
    try { return new AzulSearch(n,threads,c); } catch(...) { return nullptr; }
}
void azul_search_destroy(AzulSearch* s) { delete s; }
int azul_search_begin(AzulSearch* s,const AzulBatch* b,uint64_t seed) {
    if(!s || !b || s->trees.size()!=b->states.size()) return -1;
    struct Ctx { AzulSearch* s; const AzulBatch* b; uint64_t seed; } c{s,b,seed};
    s->failed=false; std::fill(s->errors.begin(),s->errors.end(),uint8_t{0});
    try {
        s->pool.run([](void* ptr,size_t a,size_t z) noexcept {
            auto& c=*static_cast<Ctx*>(ptr);
            for(auto i=a;i<z;++i) try {
                c.s->trees[i].reset(c.b->states[i],azul::episode_seed(c.seed,i)); c.s->ready[i]=0;
            } catch(...) { c.s->errors[i]=1; }
        },&c);
        s->begun=s->ok(); return s->begun?0:-1;
    } catch(...) { s->failed=true;return -1; }
}
int azul_search_request(AzulSearch* s,float* observations,uint8_t* ready) {
    if(!s || !s->begun || s->failed || !observations || !ready) return -1;
    struct Ctx { AzulSearch* s; float* o; uint8_t* r; } c{s,observations,ready};
    try {
        s->pool.run([](void* ptr,size_t a,size_t z) noexcept {
            auto& c=*static_cast<Ctx*>(ptr);
            if(c.s->optimized && !c.s->caches[a]) {
                try {c.s->caches[a]=std::make_unique<azul::EvaluationCache>(c.s->cache_capacity/c.s->workers);
                    c.s->caches[a]->symmetric=c.s->symmetric_cache;}
                catch(...) {c.s->errors[a]=1;return;}
            }
            for(auto i=a;i<z;++i) try {
                auto* out=c.o+i*azul::observation_size;
                auto& tree=c.s->trees[i];
                bool requested=tree.request(out);
                if(c.s->optimized) {
                    std::array<float,180> logits;float value;
                    auto& cache=*c.s->caches[a];
                    while(requested && cache.get(tree.nodes[tree.pending].state,c.s->model_version,logits.data(),value,&c.s->pending_keys[i])) {
                        tree.submit(logits.data(),value,true);requested=tree.request(out);
                    }
                }
                c.r[i]=c.s->ready[i]=requested;
                if(!c.r[i]) std::fill_n(out,azul::observation_size,0.0F);
            } catch(...) {c.s->errors[i]=1;}
        },&c);
        if(!s->ok()) return -1;
        int n=0;for(auto r:s->ready) n+=r;return n;
    } catch(...) {s->failed=true;return -1;}
}
int azul_search_submit(AzulSearch* s,const float* logits,const float* values) {
    if(!s || !s->begun || s->failed || !logits || !values) return -1;
    // Validate whole batch before mutation, including responses for pending leaves only.
    bool any=false;
    for(size_t i=0;i<s->trees.size();++i) if(s->ready[i]) {
        any=true;
        if(!std::isfinite(values[i]) || std::abs(values[i])>1.00001F) return -1;
        if(!finite_policy(logits+i*azul::action_count)) return -1;
    }
    if(!any) return -1;
    struct Ctx { AzulSearch* s; const float* p; const float* v; } c{s,logits,values};
    try {
        s->pool.run([](void* ptr,size_t a,size_t z) noexcept {
            auto& c=*static_cast<Ctx*>(ptr);
            for(auto i=a;i<z;++i) if(c.s->ready[i]) try {
                if(c.s->optimized) c.s->caches[a]->put_prepared(c.s->pending_keys[i],c.s->model_version,c.p+i*azul::action_count,c.v[i]);
                c.s->trees[i].submit(c.p+i*azul::action_count,c.v[i],true);c.s->ready[i]=0;
            } catch(...) {c.s->errors[i]=1;}
        },&c);
        return s->ok()?0:-1;
    } catch(...) {s->failed=true;return -1;}
}
int azul_search_results(AzulSearch* s,float* policies,uint16_t* actions,float* values,AzulSearchStats* stats) {
    if(!s || !s->begun || s->failed || !policies || !actions || !values) return -1;
    for(const auto& t:s->trees) if(!t.done()) return -1;
    struct Ctx { AzulSearch* s; float* p; uint16_t* a; float* v; } c{s,policies,actions,values};
    try {
        s->pool.run([](void* ptr,size_t a,size_t z) noexcept {
            auto& c=*static_cast<Ctx*>(ptr);
            for(auto i=a;i<z;++i) try {c.s->trees[i].result(c.p+i*azul::action_count,c.a[i],c.v[i]);}
                catch(...) {c.s->errors[i]=1;}
        },&c);
        if(stats) {
            *stats={};
            for(const auto& t:s->trees) {
                stats->simulations+=t.stats.simulations; stats->evaluations+=t.stats.evaluations;
                stats->chance_draws+=t.stats.chance_draws; stats->chance_reuses+=t.stats.chance_reuses;
                stats->chance_duplicates+=t.stats.chance_duplicates; stats->nodes+=t.stats.nodes;
            }
        }
        return s->ok()?0:-1;
    } catch(...) {s->failed=true;return -1;}
}
}

extern "C" {
int azul_search_configure(AzulSearch* s,const AzulSearchTuning* p) {
    if(!s || !p || p->q_mode<0 || p->q_mode>2 || p->cache_capacity<0 || p->cache_capacity>1048576 ||
        !std::isfinite(p->q_floor) || p->q_floor<=0 || !std::isfinite(p->chance_coefficient) || p->chance_coefficient<=0 ||
        !std::isfinite(p->chance_exponent) || p->chance_exponent<0 || p->chance_exponent>1 ||
        !std::isfinite(p->chance_sensitivity) || p->chance_sensitivity<0 || p->chance_sensitivity>10 ||
        !std::isfinite(p->afterstate_prior) || p->afterstate_prior<0 || p->afterstate_prior>16) return -1;
    for(auto& tree:s->trees) tree.tuning={p->q_mode,p->q_floor,p->chance_coefficient,p->chance_exponent,
        p->chance_sensitivity,p->afterstate_prior,p->subtree_reuse!=0};
    if(s->cache_capacity!=p->cache_capacity || s->symmetric_cache!=(p->symmetric_cache!=0))
        for(auto& cache:s->caches) cache.reset();
    s->cache_capacity=p->cache_capacity;s->symmetric_cache=p->symmetric_cache!=0;s->optimized=true;
    return 0;
}
int azul_search_begin_range(AzulSearch* s,const AzulBatch* b,size_t offset,uint64_t seed,uint64_t version,const uint8_t* active) {
    if(!s || !b || offset>b->states.size() || s->trees.size()>b->states.size()-offset) return -1;
    struct Ctx {AzulSearch* s;const AzulBatch* b;size_t offset;uint64_t seed,version;const uint8_t* active;} c{s,b,offset,seed,version,active};
    s->failed=false;s->model_version=version;std::fill(s->errors.begin(),s->errors.end(),uint8_t{0});
    for(auto& cache:s->caches) if(cache) cache->hits=cache->misses=0;
    try {
        s->pool.run([](void* ptr,size_t a,size_t z) noexcept {
            auto& c=*static_cast<Ctx*>(ptr);
            for(auto i=a;i<z;++i) try {
                c.s->trees[i].begin(c.b->states[c.offset+i],azul::episode_seed(c.seed,c.offset+i),c.version);
                if(c.active && !c.active[c.offset+i]) c.s->trees[i].deactivate();
                c.s->ready[i]=0;
            } catch(...) {c.s->errors[i]=1;}
        },&c);
        s->begun=s->ok();return s->begun?0:-1;
    }catch(...) {s->failed=true;return -1;}
}
int azul_search_diagnostics(AzulSearch* s,uint64_t* out) {
    if(!s || !out) return -1;
    std::fill_n(out,4,UINT64_C(0));
    for(const auto& cache:s->caches) if(cache) {out[0]+=cache->hits;out[1]+=cache->misses;}
    for(const auto& tree:s->trees) {out[2]+=tree.stats.reused_nodes;out[3]+=tree.stats.afterstate_evaluations;}
    return 0;
}
int azul_batch_step_afterstates(AzulBatch* b,const uint16_t* actions,AzulStepResult* results,float* obs,uint8_t* flags,uint8_t* perspectives) {
    if(!b || !actions || !results || !obs || !flags || !perspectives) return -1;
    struct Ctx {AzulBatch* b;const uint16_t* a;AzulStepResult* r;float* o;uint8_t* f;uint8_t* p;} c{b,actions,results,obs,flags,perspectives};
    try {
        b->pool.run([](void* ptr,size_t a,size_t z) noexcept {
            auto& c=*static_cast<Ctx*>(ptr);
            for(auto i=a;i<z;++i) {
                auto& s=c.b->states[i];auto& r=c.r[i];r={};r.actor=s.current;c.f[i]=0;c.p[i]=s.current;
                const bool valid=azul::apply(s,c.a[i]);
                r.invalid_action=!valid;r.terminated=s.phase==azul::Phase::terminal;
                r.round_finished=valid && s.phase!=azul::Phase::draft;
                r.reward=valid?azul::terminal_reward(s,r.actor):0;
                r.scores[0]=s.players[0].score;r.scores[1]=s.players[1].score;
                if(valid && s.phase==azul::Phase::chance) {
                    c.f[i]=1;c.p[i]=s.current;azul::observe(s,s.current,c.o+i*azul::observation_size);azul::deal(s);
                }
            }
        },&c);return 0;
    }catch(...) {return -1;}
}
int azul_batch_manual_deal(AzulBatch* b,const uint8_t* colors) {
    if(!b || !colors || b->states.size()!=1) return -1;
    auto s=b->states[0];
    if(s.phase!=azul::Phase::chance) return -1;
    s.sources={};s.tiles_remaining=0;
    for(int i=0;i<20;++i) {
        if(!s.bag_size) {s.bag=s.discard;s.discard={};for(auto n:s.bag)s.bag_size+=n;}
        const int c=colors[i];
        if(!s.bag_size) {if(c!=255)return -1;continue;}
        if(c>=5 || !s.bag[c]) return -1;
        --s.bag[c];--s.bag_size;++s.sources[i/4][c];++s.tiles_remaining;
    }
    s.token_available=1;s.current=s.next_start;++s.round;s.phase=azul::Phase::draft;
    b->states[0]=s;
    return 0;
}
int azul_manual_reset(AzulBatch* b,uint64_t seed) {
    if(!b || b->states.size()!=1)return -1;
    azul::State s;s.rng.state=seed;s.bag.fill(20);s.bag_size=100;b->states[0]=s;return 0;
}
int azul_manual_step(AzulBatch* b,uint16_t action,AzulStepResult* r) {
    if(!b || b->states.size()!=1 || !r)return -1;
    auto& s=b->states[0];*r={};r->actor=s.current;
    const bool valid=azul::apply(s,action);r->invalid_action=!valid;
    r->terminated=s.phase==azul::Phase::terminal;r->round_finished=valid && s.phase!=azul::Phase::draft;
    r->reward=valid?azul::terminal_reward(s,r->actor):0;
    r->scores[0]=s.players[0].score;r->scores[1]=s.players[1].score;return 0;
}
int azul_batch_import_public(AzulBatch* b,size_t index,const uint8_t* sources,
    const uint32_t* walls,const uint16_t* scores,const uint8_t* pattern_colors,
    const uint8_t* pattern_counts,const uint8_t* floor_tiles,const uint8_t* floor_counts,
    const uint8_t* bag,const uint8_t* discard,uint32_t round,uint8_t current,
    uint8_t next_start,uint8_t token_available) {
    if(!b || index>=b->states.size() || !sources || !walls || !scores || !pattern_colors ||
       !pattern_counts || !floor_tiles || !floor_counts || !bag || !discard ||
       current>1 || next_start>1 || token_available>1) return -1;
    azul::State s;
    s.rng.state=azul::episode_seed(UINT64_C(0x243f6a8885a308d3),index);
    s.round=round;s.current=current;s.next_start=next_start;
    s.token_available=token_available;s.phase=azul::Phase::draft;
    for(int source=0;source<6;++source) for(int color=0;color<5;++color) {
        const auto n=sources[source*5+color];
        s.sources[source][color]=n;s.tiles_remaining+=n;
    }
    for(int player=0;player<2;++player) {
        auto& p=s.players[player];p.wall=walls[player];p.score=scores[player];
        p.floor_count=floor_counts[player];
        for(int color=0;color<5;++color)p.floor_tiles[color]=floor_tiles[player*5+color];
        for(int row=0;row<5;++row) {
            p.pattern_color[row]=pattern_colors[player*5+row];
            p.pattern_count[row]=pattern_counts[player*5+row];
        }
    }
    for(int color=0;color<5;++color) {
        s.bag[color]=bag[color];s.discard[color]=discard[color];s.bag_size+=bag[color];
    }
    if(!azul::validate(s))return -1;
    b->states[index]=s;return 0;
}
}

#include "azul/engine3.hpp"
struct Azul3Pool { size_t n; unsigned threads; template<class F> void run(F&& f) { if(threads<=1||n<threads*2){f(0,n);return;} std::vector<std::thread> ts; ts.reserve(threads-1); size_t q=n/threads, rem=n%threads, first=0; for(unsigned t=0;t<threads-1;++t){size_t last=first+q+(t<rem); ts.emplace_back([&,first,last]{f(first,last);}); first=last;} f(first,n); for(auto& t:ts)t.join(); } };
struct Azul3Batch { std::vector<azul3::State> states; Azul3Pool pool; Azul3Batch(size_t n,unsigned t):states(n),pool{n,t}{} };
extern "C" {
int azul3_abi_version(void){return AZUL3_ABI_VERSION;}
Azul3Batch* azul3_batch_create(size_t n,unsigned threads,uint64_t seed){if(!n||n>SIZE_MAX/sizeof(azul3::State))return nullptr;if(!threads)threads=std::max(1U,std::thread::hardware_concurrency());threads=std::min<unsigned>(threads,(unsigned)n);try{auto*b=new Azul3Batch(n,threads);if(azul3_batch_reset(b,seed)){delete b;return nullptr;}return b;}catch(...){return nullptr;}}
void azul3_batch_destroy(Azul3Batch*b){delete b;}
size_t azul3_batch_size(const Azul3Batch*b){return b?b->states.size():0;}
int azul3_batch_reset(Azul3Batch*b,uint64_t seed){if(!b)return -1;try{b->pool.run([&](size_t a,size_t z){for(size_t i=a;i<z;++i)b->states[i]=azul3::initial_state(azul3::episode_seed(seed,i),static_cast<uint8_t>(i%3));});return 0;}catch(...){return -1;}}
int azul3_batch_reset_at(Azul3Batch*b,size_t i,uint64_t seed,unsigned start){if(!b||i>=b->states.size()||start>2)return -1;b->states[i]=azul3::initial_state(seed,(uint8_t)start);return 0;}
int azul3_batch_observe(Azul3Batch*b,float*o,uint8_t*m,uint8_t*p){if(!b)return -1;try{b->pool.run([&](size_t a,size_t z){for(size_t i=a;i<z;++i){auto&s=b->states[i];if(o)azul3::observe(s,s.current,o+i*azul3::observation_size);if(m)azul3::action_mask(s,m+i*azul3::action_count);if(p)p[i]=s.current;}});return 0;}catch(...){return -1;}}
int azul3_batch_step(Azul3Batch*b,const uint16_t*actions,Azul3StepResult*r){if(!b||!actions)return -1;try{b->pool.run([&](size_t a,size_t z){for(size_t i=a;i<z;++i){auto&s=b->states[i];auto actor=s.current;auto round=s.round;bool valid=azul3::step(s,actions[i]);if(r){auto&x=r[i];x={};x.actor=actor;x.terminated=s.phase==azul3::Phase::terminal;x.round_finished=valid&&(s.round!=round||x.terminated);x.invalid_action=!valid;x.reward=valid?azul3::terminal_reward(s,actor):0;for(int p=0;p<3;++p)x.scores[p]=s.players[p].score;}}});return 0;}catch(...){return -1;}}
int azul3_batch_import_public(Azul3Batch*b,size_t i,const uint8_t*src,const uint32_t*walls,const uint16_t*scores,
 const uint8_t*pc,const uint8_t*pn,const uint8_t*ft,const uint8_t*fc,const uint8_t*bag,const uint8_t*discard,
 uint32_t round,uint8_t current,uint8_t next,uint8_t token){
 if(!b||i>=b->states.size()||!src||!walls||!scores||!pc||!pn||!ft||!fc||!bag||!discard||current>2||next>2||token>1)return -1;
 try { azul3::State s{}; s.phase=azul3::Phase::draft; s.round=round; s.current=current; s.next_start=next; s.token_available=token;
  s.rng.state=azul3::episode_seed(UINT64_C(0x9e3779b97f4a7c15),i);
  int bag_size=0, tiles_remaining=0;
  for(int c=0;c<5;++c){if(bag[c]>20||discard[c]>20)return -1;s.bag[c]=bag[c];s.discard[c]=discard[c];bag_size+=bag[c];}
  for(int f=0;f<8;++f) for(int c=0;c<5;++c){if(src[f*5+c]>20)return -1;s.sources[f][c]=src[f*5+c];tiles_remaining+=src[f*5+c];}
  if(bag_size>100||tiles_remaining>100)return -1;
  s.bag_size=static_cast<uint8_t>(bag_size);s.tiles_remaining=static_cast<uint8_t>(tiles_remaining);
  for(int p=0;p<3;++p){auto &q=s.players[p];q.wall=walls[p];q.score=scores[p];q.floor_count=fc[p];for(int c=0;c<5;++c)q.floor_tiles[c]=ft[p*5+c];for(int r=0;r<5;++r){q.pattern_color[r]=pc[p*5+r];q.pattern_count[r]=pn[p*5+r];}}
  if(!azul3::validate(s)) return -1;
  b->states[i]=s; return 0;
 } catch(...) { return -1; }
}
}

static bool finite_policy3(const float* p) noexcept { for(int i=0;i<240;++i) if(!std::isfinite(p[i])) return false; return true; }
struct Azul3Search {
    std::vector<azul3::SearchTree3> trees;
    std::vector<uint8_t> ready, errors;
    Pool pool;

    unsigned workers;
    int cache_capacity=0;
    bool symmetric_cache=false,optimized=false;
    uint64_t model_version=0;
    bool begun = false, failed = false;
    Azul3Search(size_t n, unsigned threads, azul3::SearchConfig c) : ready(n), errors(n), pool(n,threads),workers(threads) {
        trees.reserve(n);
        for(size_t i=0;i<n;++i) trees.emplace_back(c);
    }
    bool ok() {
        for(auto e:errors) if(e) failed=true;
        return !failed;
    }
};
extern "C" {
int azul3_search_abi_version(void) { return 1; }
Azul3Search* azul3_search_create(size_t n,unsigned threads,const AzulSearchConfig* cfg) {
    if(!cfg || !n || n>65536) return nullptr;
    azul3::SearchConfig c{cfg->simulations,cfg->candidates,cfg->max_depth,cfg->chance_initial,
        cfg->chance_cap,cfg->gumbel_scale,cfg->value_scale,cfg->maxvisit_init};
    if(!azul3::valid_config(c)) return nullptr;
    if(!threads) threads=std::max(1U,std::thread::hardware_concurrency());
    threads=unsigned(std::min<size_t>(threads,n));
    try { return new Azul3Search(n,threads,c); } catch(...) { return nullptr; }
}
void azul3_search_destroy(Azul3Search* s) { delete s; }
int azul3_search_begin(Azul3Search* s,const Azul3Batch* b,uint64_t seed) {
    if(!s || !b || s->trees.size()!=b->states.size()) return -1;
    struct Ctx { Azul3Search* s; const Azul3Batch* b; uint64_t seed; } c{s,b,seed};
    s->failed=false; std::fill(s->errors.begin(),s->errors.end(),uint8_t{0});
    try {
        s->pool.run([](void* ptr,size_t a,size_t z) noexcept {
            auto& c=*static_cast<Ctx*>(ptr);
            for(auto i=a;i<z;++i) try {
                c.s->trees[i].reset(c.b->states[i],azul3::episode_seed(c.seed,i)); c.s->ready[i]=0;
            } catch(...) { c.s->errors[i]=1; }
        },&c);
        s->begun=s->ok(); return s->begun?0:-1;
    } catch(...) { s->failed=true;return -1; }
}
int azul3_search_request(Azul3Search* s,float* observations,uint8_t* ready) {
    if(!s || !s->begun || s->failed || !observations || !ready) return -1;
    struct Ctx { Azul3Search* s; float* o; uint8_t* r; } c{s,observations,ready};
    try {
        s->pool.run([](void* ptr,size_t a,size_t z) noexcept {
            auto& c=*static_cast<Ctx*>(ptr);
            for(auto i=a;i<z;++i) try {
                auto* out=c.o+i*azul3::observation_size;
                auto& tree=c.s->trees[i];
                bool requested=tree.request(out);
                c.r[i]=c.s->ready[i]=requested;
                if(!c.r[i]) std::fill_n(out,azul3::observation_size,0.0F);
            } catch(...) {c.s->errors[i]=1;}
        },&c);
        if(!s->ok()) return -1;
        int n=0;for(auto r:s->ready) n+=r;return n;
    } catch(...) {s->failed=true;return -1;}
}
int azul3_search_submit(Azul3Search* s,const float* logits,const float* values) {
    if(!s || !s->begun || s->failed || !logits || !values) return -1;
    // Validate whole batch before mutation, including responses for pending leaves only.
    bool any=false;
    for(size_t i=0;i<s->trees.size();++i) if(s->ready[i]) {
        any=true;
        for(int p=0;p<3;++p) if(!std::isfinite(values[i*3+p]) || std::abs(values[i*3+p])>1.00001F) return -1;
        if(!finite_policy3(logits+i*azul3::action_count)) return -1;
    }
    if(!any) return -1;
    struct Ctx { Azul3Search* s; const float* p; const float* v; } c{s,logits,values};
    try {
        s->pool.run([](void* ptr,size_t a,size_t z) noexcept {
            auto& c=*static_cast<Ctx*>(ptr);
            for(auto i=a;i<z;++i) if(c.s->ready[i]) try {
                c.s->trees[i].submit(c.p+i*azul3::action_count,c.v+i*3,true);c.s->ready[i]=0;
            } catch(...) {c.s->errors[i]=1;}
        },&c);
        return s->ok()?0:-1;
    } catch(...) {s->failed=true;return -1;}
}
int azul3_search_results(Azul3Search* s,float* policies,uint16_t* actions,float* values,Azul3SearchStats* stats) {
    if(!s || !s->begun || s->failed || !policies || !actions || !values) return -1;
    for(const auto& t:s->trees) if(!t.done()) return -1;
    struct Ctx { Azul3Search* s; float* p; uint16_t* a; float* v; } c{s,policies,actions,values};
    try {
        s->pool.run([](void* ptr,size_t a,size_t z) noexcept {
            auto& c=*static_cast<Ctx*>(ptr);
            for(auto i=a;i<z;++i) try {c.s->trees[i].result(c.p+i*azul3::action_count,c.a[i],c.v[i]);}
                catch(...) {c.s->errors[i]=1;}
        },&c);
        if(stats) {
            *stats={};
            for(const auto& t:s->trees) {
                stats->simulations+=t.stats.simulations; stats->evaluations+=t.stats.evaluations;
                stats->chance_draws+=t.stats.chance_draws; stats->chance_reuses+=t.stats.chance_reuses;
                stats->chance_duplicates+=t.stats.chance_duplicates; stats->nodes+=t.stats.nodes;
            }
        }
        return s->ok()?0:-1;
    } catch(...) {s->failed=true;return -1;}
}
}
