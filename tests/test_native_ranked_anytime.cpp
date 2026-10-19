#include "azul/search.hpp"
#include "azul/search3.hpp"
#include <array>
#include <iostream>
#include <stdexcept>

#define CHECK(x) do { if(!(x))throw std::runtime_error(#x); } while(false)

void run(azul::SearchTree& t) {
    std::array<float,172> observation{};std::array<float,180> logits{};
    while(t.request(observation.data()))t.submit(logits.data(),.2F);
}
void run(azul3::SearchTree3& t) {
    std::array<float,244> observation{};std::array<float,240> logits{};
    std::array<float,3> utilities{.3F,0,-.3F};
    while(t.request(observation.data()))t.submit(logits.data(),utilities.data());
}

int main() {
#ifdef AZUL_RANKED_ANYTIME_DLL
    static_assert(sizeof(azul::VisitCount)==8 && sizeof(azul::ValueSum)==8);
    static_assert(sizeof(azul3::VisitCount)==8 && sizeof(azul3::ValueSum)==8);
    {
        azul::SearchNode n;n.visits=(UINT64_C(1)<<32)+123;n.sum0=double(n.visits)*.5;
        CHECK(std::abs(n.mean0()-.5F)<1e-7F);
        azul3::SearchNode n3;n3.visits=n.visits;n3.sum[0]=double(n3.visits)*.25;
        CHECK(std::abs(n3.mean()[0]-.25F)<1e-7F);
        azul::ValueSum sum=16777216.;sum+=1.;CHECK(sum==16777217.);
    }
#endif
    {
        azul::SearchConfig c;c.simulations=256;c.gumbel_scale=0;
        azul::SearchTree t(c);t.tuning.subtree_reuse=true;
        t.begin(azul::initial_state(12,0),3,1,true);run(t);
        auto visits=t.nodes[0].visits;auto size=t.nodes.size();
        t.continue_epoch(512);CHECK(t.nodes.size()==size);run(t);
        CHECK(t.stats.simulations==768);CHECK(t.nodes[0].visits==visits+512);
        azul::State descendant;bool found=false;
        for(std::size_t i=1;i<t.nodes.size();++i)if(t.nodes[i].expanded && t.nodes[i].state.phase==azul::Phase::draft){descendant=t.nodes[i].state;found=true;break;}
        CHECK(found);descendant.ply=0;
        t.begin(descendant,4,1,true);CHECK(t.stats.reused_nodes>0);run(t);
        auto wrong=descendant;wrong.players[0].score+=1;
        t.begin(wrong,5,1,true);CHECK(t.stats.reused_nodes==0);run(t);
        auto changed_model=wrong;t.begin(changed_model,6,2,true);CHECK(t.stats.reused_nodes==0);
        bool rejected=false;try{t.continue_epoch(32);}catch(const std::logic_error&){rejected=true;}
        CHECK(rejected);
    }
    {
        azul3::SearchConfig c;c.simulations=256;c.gumbel_scale=0;
        azul3::SearchTree3 t(c);t.tuning.subtree_reuse=true;
        t.begin(azul3::initial_state(12,0),3,1,true);run(t);
        auto size=t.nodes.size();t.continue_epoch(512);CHECK(t.nodes.size()==size);run(t);
        CHECK(t.stats.simulations==768);
        azul3::State descendant;bool found=false;
        for(std::size_t i=1;i<t.nodes.size();++i)if(t.nodes[i].expanded && t.nodes[i].state.phase==azul3::Phase::draft && t.nodes[i].state.current!=0){descendant=t.nodes[i].state;found=true;break;}
        CHECK(found);descendant.ply=0;
        t.begin(descendant,4,1,true);CHECK(t.stats.reused_nodes>0);
        const auto player=descendant.current;run(t);
        std::array<float,240> policy{};azul3::Action action;float value;
        t.result(policy.data(),action,value);
        CHECK(std::abs(value-t.nodes[0].mean()[player])<1e-6F);
        // Different hidden bag/discard composition can never hit the old root.
        auto different=descendant;different.bag[0]^=1;
        t.begin(different,5,1,true);CHECK(t.stats.reused_nodes==0);
    }
    std::cout<<"Ranked 2P/3P: one-tree epochs, strict public re-rooting and model invalidation passed\n";
}
