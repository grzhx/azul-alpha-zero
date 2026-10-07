#include "azul/engine3.hpp"
#include <cassert>
#include <iostream>
int main(){using namespace azul3; auto s=initial_state(123,0); assert(s.phase==Phase::draft&&s.current==0); int total=0; for(int f=0;f<factories;++f)for(auto n:s.sources[f])total+=n; assert(total==28); for(int ply=0;ply<10000&&s.phase!=Phase::terminal;++ply){assert(validate(s)); auto a=legal_actions(s); assert(a.size); auto pick=a.values[(ply*17)%a.size]; auto actor=s.current; assert(step(s,pick)); assert(actor<3); } assert(s.phase==Phase::terminal); assert(validate(s)); std::cout<<"3P engine tests passed; round="<<s.round<<" scores="<<s.players[0].score<<","<<s.players[1].score<<","<<s.players[2].score<<"\n";}
