#include "azul/search.hpp"
#include "azul/c_api.h"
#include "azul/evaluation_cache.hpp"
#include "azul/search3.hpp"
#include <iostream>
#include <cstdlib>
#include <numeric>

#define CHECK(x) do { if(!(x)) {std::cerr<<__LINE__<<": " #x "\n";std::exit(1);} } while(false)
using azul::State;
using azul::SearchTree;
void complete(SearchTree& tree, float prediction=0) {
    std::array<float,172> obs{};
    std::array<float,180> logits{};
    while(tree.request(obs.data())) {
        for(int i=0;i<180;++i) logits[i]=float(i%7)*0.01F;
        tree.submit(logits.data(),prediction);
    }
}
void conserved(State& s) {
    s.bag_size=0;s.tiles_remaining=0;
    for(int c=0;c<5;++c) {
        int used=s.discard[c];
        for(auto& src:s.sources) {used+=src[c];s.tiles_remaining+=src[c];}
        for(auto& p:s.players) {
            used+=std::popcount(p.wall & azul::color_mask(c))+p.floor_tiles[c];
            for(int r=0;r<5;++r) if(p.pattern_color[r]==c) used+=p.pattern_count[r];
        }
        s.bag[c]=std::uint8_t(20-used);s.bag_size+=s.bag[c];
    }
    s.phase=azul::Phase::draft;
    CHECK(azul::validate(s));
}
int main() {
    // Regression: the highest 24-bit midpoint used to round to float(1),
    // producing infinite Gumbel noise. Cover both ends directly, not by luck.
    for(auto bits : {UINT64_C(0), UINT64_C(1), UINT64_MAX-UINT64_C(1), UINT64_MAX}) {
        const float u=azul::unit_open_from_bits(bits);
        CHECK(u>0.0F && u<1.0F);
        CHECK(std::isfinite(-std::log(-std::log(u))));
    }
    auto seq=azul::considered_visits(16,128);
    CHECK(seq.size()==128);
    for(int i=0;i<16;++i) CHECK(seq[i]==0 && seq[i+16]==1);
    auto one=azul::considered_visits(1,8);
    for(int i=0;i<8;++i) CHECK(one[i]==i);
    // Hand-computed mixed-value completion, not ordinary visit-count targets.
    {
        State simple;simple.sources[0][0]=1;simple.tiles_remaining=1;simple.phase=azul::Phase::draft;
        SearchTree formula;formula.reset(simple,42);
        std::array<float,172> obs{};std::array<float,180> logits{},q{};
        CHECK(formula.request(obs.data()));formula.submit(logits.data(),0.2F);
        CHECK(formula.nodes[0].count==6);
        formula.edges[0].visits=2;formula.edges[0].sum0=1.2F;
        formula.edges[1].visits=1;formula.edges[1].sum0=-0.3F;
        formula.transformed(0,q.data());
        CHECK(std::abs(q[0]-5.2F)<1e-5F && std::abs(q[1])<1e-5F);
        for(int i=2;i<6;++i) CHECK(std::abs(q[i]-(0.1625F+0.3F)/0.9F*5.2F)<1e-5F);
    }
    azul::SearchConfig cfg;cfg.gumbel_scale=0;cfg.simulations=128;
    // Forced winning last draft, tested for BOTH absolute players.
    for(int actor=0;actor<2;++actor) {
        State s;s.current=std::uint8_t(actor);s.sources[5][0]=1;
        s.players[actor].wall=azul::row_mask(0)&~azul::cell(0,0);s.players[actor].score=6;
        auto& opponent=s.players[1-actor];
        opponent.wall=azul::row_mask(0)&~azul::cell(0,1);opponent.score=4;
        opponent.pattern_color[0]=1;opponent.pattern_count[0]=1;
        conserved(s);
        SearchTree t(cfg);t.reset(s,42);complete(t);
        std::array<float,180> p{};azul::Action a;float value;
        t.result(p.data(),a,value);
        CHECK(a==azul::encode(5,0,0)); CHECK(p[a]>0.99F);
        CHECK(t.stats.simulations==128 && t.nodes[0].visits==128);
        auto after=s;CHECK(azul::step(after,a)); CHECK(azul::winner(after)==actor);
    }
    // Hidden episode RNG cannot affect search; root and request buffers are deterministic.
    State s=azul::initial_state(21);auto hidden=s;hidden.rng.state^=UINT64_C(0x123456789);
    SearchTree a(cfg),b(cfg);a.reset(s,887);b.reset(hidden,887);complete(a);complete(b);
    std::array<float,180> pa{},pb{};azul::Action aa,ab;float va,vb;
    a.result(pa.data(),aa,va);b.result(pb.data(),ab,vb);
    CHECK(pa==pb && aa==ab && va==vb); CHECK(azul::legal(s,aa));
    CHECK(std::abs(std::accumulate(pa.begin(),pa.end(),0.0F)-1)<1e-5F);
    for(int i=0;i<180;++i) if(!azul::legal(s,azul::Action(i))) CHECK(pa[i]==0);
    // Crossing the round creates reusable chance outcomes and probability-weighted values.
    State cross;cross.sources[5][0]=1;conserved(cross);
    cfg.simulations=256;cfg.chance_cap=8;
    SearchTree t(cfg);t.reset(cross,19);complete(t,0.25F);
    CHECK(t.stats.chance_draws>0 && t.stats.chance_reuses>0);
    for(auto& node:t.nodes) if(node.state.phase==azul::Phase::chance) {
        CHECK(node.samples<=8); unsigned count=0;double expected=0;
        for(int k=0;k<node.count;++k) {
            const auto& e=t.chance_edges[node.first+k];count+=e.multiplicity;
            expected+=e.multiplicity*t.nodes[e.child].mean0();
            CHECK(azul::validate(t.nodes[e.child].state));
        }
        CHECK(count==node.samples);
        CHECK(std::abs(node.mean0()-expected/count)<1e-5);
    }
    // Duplicate draws count as probability mass, not extra distinct outcomes.
    State monochrome;monochrome.sources[5][0]=1;monochrome.tiles_remaining=1;
    monochrome.bag[1]=20;monochrome.bag_size=20;monochrome.phase=azul::Phase::draft;
    SearchTree duplicates(cfg);duplicates.reset(monochrome,42);complete(duplicates);
    CHECK(duplicates.stats.chance_duplicates>0 && duplicates.stats.chance_reuses>0);
    // Analytic chance expectation: only two colors, choose four out of five.
    // Factories are unlabeled: with two blue among five tiles split 4+1,
    // E[max factory blue count]/4 = (0.4*1 + 0.6*2)/4 = 0.4.
    double expected_sum=0;
    for(unsigned seed=0;seed<1000;++seed) {
        State toy;toy.sources[5][4]=1;toy.tiles_remaining=1;
        toy.players[0].wall=azul::color_mask(4); // only the floor action is legal
        toy.bag={2,3,0,0,0};toy.bag_size=5;toy.phase=azul::Phase::draft;
        azul::SearchConfig small;small.simulations=16;small.candidates=1;small.max_depth=1;small.chance_cap=8;
        SearchTree exact(small);exact.reset(toy,seed);
        std::array<float,172> o{};std::array<float,180> l{};
        while(exact.request(o.data())) {
            float maximum=0;
            for(int factory=0;factory<5;++factory) maximum=std::max(maximum,o[factory*5]);
            exact.submit(l.data(),maximum*5.0F);
        }
        std::array<float,180> policy{};azul::Action act;float v;
        exact.result(policy.data(),act,v);
        CHECK(act==azul::encode(5,4,5));
        expected_sum+=exact.nodes[1].mean0();
    }
    CHECK(std::abs(expected_sum/1000.0-0.4)<0.025);
    // Rejected network response leaves the pending request intact.
    SearchTree pending(cfg);pending.reset(s,42);std::array<float,172> obs{};std::array<float,180> logits{};
    CHECK(pending.request(obs.data()));
    bool rejected=false;try {pending.submit(logits.data(),std::numeric_limits<float>::quiet_NaN());}catch(...) {rejected=true;}
    CHECK(rejected && pending.pending==0);pending.submit(logits.data(),0);complete(pending);
    // Versioned canonical evaluation cache preserves factory/action correspondence.
    {
        auto original=azul::initial_state(42);auto permuted=original;
        std::swap(permuted.sources[0],permuted.sources[4]);
        azul::EvaluationCache cache(128);cache.symmetric=true;
        std::array<float,180> input{},output{};
        for(int i=0;i<180;++i)input[i]=float(i);
        cache.put(original,7,input.data(),0.25F);float v=0;
        CHECK(cache.get(permuted,7,output.data(),v));CHECK(v==0.25F);
        for(int i=0;i<30;++i) CHECK(output[i]==input[120+i] && output[120+i]==input[i]);
        CHECK(!cache.get(permuted,8,output.data(),v));
    }
    // Re-rooting preserves child information while allocating a fresh Gumbel budget.
    {
        azul::SearchConfig config;config.simulations=32;config.candidates=8;
        SearchTree reuse(config);reuse.tuning.subtree_reuse=true;
        auto root=azul::initial_state(98);reuse.begin(root,9,1);complete(reuse);
        std::array<float,180> p{};azul::Action act;float v;
        reuse.result(p.data(),act,v);CHECK(azul::step(root,act));
        reuse.begin(root,10,1);CHECK(reuse.stats.reused_nodes>0);complete(reuse);
        CHECK(reuse.stats.simulations==32);reuse.result(p.data(),act,v);CHECK(azul::legal(root,act));
        reuse.begin(root,11,2);CHECK(reuse.stats.reused_nodes==0);
    }
    // Afterstate prior augments rather than replaces real chance sampling.
    {
        SearchTree auxiliary(cfg);auxiliary.tuning.afterstate_prior=0.5F;
        auxiliary.tuning.q_mode=1;auxiliary.tuning.q_floor=0.25F;
        auxiliary.tuning.chance_coefficient=2;auxiliary.tuning.chance_sensitivity=1;
        auxiliary.reset(cross,51);complete(auxiliary,0.2F);
        CHECK(auxiliary.stats.afterstate_evaluations>0 && auxiliary.stats.chance_draws>0);
        CHECK(auxiliary.stats.simulations==unsigned(cfg.simulations));
        for(const auto& node:auxiliary.nodes) if(node.state.phase==azul::Phase::chance && node.visits) {
            double expected=0.5*node.raw0;
            for(int k=0;k<node.count;++k) {auto e=auxiliary.chance_edges[node.first+k];expected+=e.multiplicity*auxiliary.nodes[e.child].mean0();}
            CHECK(std::abs(node.mean0()-expected/(node.samples+0.5))<1e-5);
        }
    }
    // Equivalent factory order has the same tree/prior/noise; only external
    // action slots move. Center remains fixed, including repeated factories.
    {
        auto original=azul::initial_state(711);auto permuted=original;
        std::swap(permuted.sources[0],permuted.sources[4]);
        SearchTree x(cfg),y(cfg);x.reset(original,42);y.reset(permuted,42);
        complete(x);complete(y);
        CHECK(x.nodes[0].state==y.nodes[0].state);
        x.result(pa.data(),aa,va);y.result(pb.data(),ab,vb);
        for(int i=0;i<30;++i) CHECK(pa[i]==pb[120+i] && pa[120+i]==pb[i]);
        for(int i=30;i<120;++i) CHECK(pa[i]==pb[i]);
        for(int i=150;i<180;++i) CHECK(pa[i]==pb[i]);
        CHECK(azul::legal(original,aa) && azul::legal(permuted,ab));
    }
    // Re-root a sampled deal under a different REAL factory ordering. Both
    // child lookup and policy/action projection must survive the permutation.
    {
        SearchTree reuse(cfg);reuse.tuning.subtree_reuse=true;
        reuse.begin(cross,14,1);complete(reuse);
        State actual;bool found=false;
        for(const auto& n:reuse.nodes) if(n.expanded && n.state.phase==azul::Phase::draft && n.state.round>cross.round) {
            actual=n.state;found=true;break;
        }
        CHECK(found);
        std::swap(actual.sources[0],actual.sources[4]);actual.rng.state=999;
        reuse.begin(actual,15,1);CHECK(reuse.stats.reused_nodes>0);complete(reuse);
        reuse.result(pa.data(),aa,va);CHECK(azul::legal(actual,aa));
        for(int i=0;i<180;++i) if(!azul::legal(actual,azul::Action(i))) CHECK(pa[i]==0);
    }
    // Draw FIRST, then canonicalize. Original ordered deals with the one yellow
    // in any of five factories must share exactly one child, accumulating mass.
    {
        State toy;toy.sources[5][4]=1;toy.tiles_remaining=1;
        toy.players[0].wall=azul::color_mask(4);
        toy.bag={19,1,0,0,0};toy.bag_size=20;toy.phase=azul::Phase::draft;
        azul::SearchConfig fixed;fixed.simulations=16;fixed.candidates=1;fixed.max_depth=1;
        fixed.chance_initial=8;fixed.chance_cap=8;
        SearchTree dedup(fixed);dedup.reset(toy,29);complete(dedup);
        const auto& chance=dedup.nodes[1];CHECK(chance.samples==8 && chance.count==1);
        CHECK(dedup.chance_edges[chance.first].multiplicity==8);
        CHECK(dedup.stats.chance_duplicates==7);
        auto canonical=dedup.nodes[dedup.chance_edges[chance.first].child].state;
        std::array<int,5> yellow_positions{};
        for(unsigned sample=0;sample<8;++sample) {
            auto draw=chance.state;draw.rng.state=azul::episode_seed(chance.sample_key,sample);
            azul::deal(draw);draw.rng.state=0;
            for(int f=0;f<5;++f) if(draw.sources[f][1]) ++yellow_positions[f];
            azul_symmetry::canonicalize_factories(draw);CHECK(draw==canonical);
        }
        CHECK(std::count_if(yellow_positions.begin(),yellow_positions.end(),[](int n){return n>0;})>1);
    }
    // Find a three-sample X,X,Y pool and check the 2:1 expectation explicitly.
    {
        bool verified=false;
        for(unsigned seed=0;seed<100 && !verified;++seed) {
            State toy;toy.sources[5][4]=1;toy.tiles_remaining=1;
            toy.players[0].wall=azul::color_mask(4);
            toy.bag={18,2,0,0,0};toy.bag_size=20;toy.phase=azul::Phase::draft;
            azul::SearchConfig fixed;fixed.simulations=3;fixed.candidates=1;fixed.max_depth=1;
            fixed.chance_initial=3;fixed.chance_cap=3;
            SearchTree weighted(fixed);weighted.reset(toy,seed);
            std::array<float,172> o{};std::array<float,180> l{};
            while(weighted.request(o.data())) {
                float maximum=0;for(int f=0;f<5;++f) maximum=std::max(maximum,o[f*5+1]);
                weighted.submit(l.data(),maximum*10.0F);
            }
            const auto& chance=weighted.nodes[1];
            if(chance.count!=2) continue;
            unsigned mass=0;double expectation=0,uniform=0;
            for(int i=0;i<2;++i) {auto e=weighted.chance_edges[chance.first+i];
                mass+=e.multiplicity;expectation+=e.multiplicity*weighted.nodes[e.child].mean0();
                uniform+=weighted.nodes[e.child].mean0();}
            CHECK(mass==3);CHECK(std::abs(chance.mean0()-expectation/3)<1e-5);
            CHECK(std::abs(chance.mean0()-uniform/2)>0.01);verified=true;
        }
        CHECK(verified);
    }
    // Max-N uses the same symmetry while keeping three absolute utilities.
    {
        azul3::SearchConfig config;config.simulations=16;config.gumbel_scale=0;
        auto original=azul3::initial_state(98);auto permuted=original;
        std::swap(permuted.sources[0],permuted.sources[6]);
        azul3::SearchTree3 x(config),y(config);x.tuning.subtree_reuse=true;y.tuning.subtree_reuse=true;
        x.begin(original,42,1);y.begin(permuted,42,1);
        std::array<float,244> obs3{};std::array<float,240> logits3{},px{},py{};
        const std::array<float,3> utility{.2F,-.1F,-.1F};
        for(auto tree:{&x,&y}) while(tree->request(obs3.data())) tree->submit(logits3.data(),utility.data());
        azul3::Action ax,ay;float vx,vy;x.result(px.data(),ax,vx);y.result(py.data(),ay,vy);
        CHECK(azul3::legal(original,ax) && azul3::legal(permuted,ay));
        for(int i=0;i<30;++i) CHECK(px[i]==py[180+i] && px[180+i]==py[i]);
        for(int i=30;i<180;++i) CHECK(px[i]==py[i]);
        for(int i=210;i<240;++i) CHECK(px[i]==py[i]);
        auto real=y.nodes[0].state;std::swap(real.sources[1],real.sources[5]);
        y.begin(real,43,1);CHECK(y.stats.reused_nodes>0);
    }
    std::cout<<"Gumbel search tests passed: schedule, perspectives, canonical factories, action maps, chance masses and rerooting\n";
}
