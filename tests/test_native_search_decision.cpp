// Crafted completed-tree results: low-visit optimistic Q must not become play.
#include "azul/search.hpp"
#include "azul/search3.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>

int main() {
    {
        azul::SearchConfig c; c.simulations=32; c.gumbel_scale=0;
        azul::SearchTree tree(c); tree.reset(azul::initial_state(1,0),2);
        std::array<float,172> obs{}; std::array<float,180> logits{},policy{};
        tree.request(obs.data()); tree.submit(logits.data(),0);
        auto& root=tree.nodes[0];
        auto& a=tree.edges[root.first]; auto& b=tree.edges[root.first+1];
        a.visits=31; a.sum0=-.9F*31; b.visits=1; b.sum0=.3F;
        tree.completed=32;
        azul::Action action; float value; tree.result(policy.data(),action,value);
        auto original=azul::initial_state(1,0);
        const auto expected=azul_symmetry::external_action(a.action,
            azul_symmetry::canonicalize_factories(original));
        if(action!=expected || action==std::distance(policy.begin(),std::max_element(policy.begin(),policy.end())))
            throw std::runtime_error("2P native final action/policy mismatch regression");
    }
    {
        azul3::SearchConfig c; c.simulations=32; c.gumbel_scale=0;
        azul3::SearchTree3 tree(c); tree.reset(azul3::initial_state(1,0),2);
        std::array<float,244> obs{}; std::array<float,240> logits{},policy{};
        const std::array<float,3> values{};
        tree.request(obs.data()); tree.submit(logits.data(),values.data());
        auto& root=tree.nodes[0];
        auto& a=tree.edges[root.first]; auto& b=tree.edges[root.first+1];
        a.visits=31; a.sum[0]=-.9F*31; b.visits=1; b.sum[0]=.3F;
        tree.completed=32;
        azul3::Action action; float value; tree.result(policy.data(),action,value);
        auto original=azul3::initial_state(1,0);
        const auto expected=azul_symmetry::external_action(a.action,
            azul_symmetry::canonicalize_factories(original));
        if(action!=expected || action==std::distance(policy.begin(),std::max_element(policy.begin(),policy.end())))
            throw std::runtime_error("3P native final action/policy mismatch regression");
    }
    std::cout << "2P/3P final Sequential Halving actions differ from policy argmax as expected\n";
}
