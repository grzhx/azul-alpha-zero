import torch
from torch import nn
from torch.nn import functional as F
from .model import ResidualBlock
class PolicyValue3(nn.Module):
    """Three-player network: 7 factory slots + center, 3 rotated player views, max-n utility head."""
    def __init__(self,width=256,blocks=4):
        super().__init__(); self.architecture={"width":width,"blocks":blocks,"kind":"equivariant3"}; self.factory_width=max(16,width//4)
        self.factory_encoder=nn.Sequential(nn.Linear(5,self.factory_width),nn.SiLU(),nn.Linear(self.factory_width,self.factory_width),nn.SiLU())
        self.input=nn.Linear(209+self.factory_width,width); self.context=nn.Linear(width,self.factory_width)
        self.factory_policy=nn.Sequential(nn.Linear(2*self.factory_width,self.factory_width),nn.SiLU(),nn.Linear(self.factory_width,30))
        self.afterstate=nn.Linear(width,3); self.blocks=nn.Sequential(*(ResidualBlock(width) for _ in range(blocks))); self.norm=nn.LayerNorm(width)
        self.policy=nn.Linear(width,30); self.value=nn.Linear(width,3)
    def forward(self,x):
        f=self.factory_encoder(x[:,:35].reshape(-1,7,5)); h=self.norm(self.blocks(F.silu(self.input(torch.cat((x[:,35:],f.mean(1)),-1)))))
        ctx=self.context(h)[:,None,:].expand(-1,7,-1); pol=torch.cat((self.factory_policy(torch.cat((f,ctx),-1)).reshape(-1,210),self.policy(h)),-1)
        # 3 utilities in rotated absolute-seat order are bounded for stable max-n backup.
        return pol,torch.tanh(self.value(h))
    @staticmethod
    def from_1800(path,device="cpu"):
        from .model import PolicyValue
        ck=torch.load(path,map_location="cpu",weights_only=False); old=PolicyValue(**ck["architecture"]);old.load_state_dict(ck["model"])
        m=PolicyValue3(width=ck["architecture"]["width"],blocks=ck["architecture"]["blocks"])
        sd=m.state_dict(); osd=old.state_dict()
        for k in sd:
            if k in sd and k in osd and sd[k].shape==osd[k].shape:sd[k].copy_(osd[k])
        # Map old tail [center, two players, misc] into [center, three players, misc].
        sd["input.weight"].zero_(); sd["input.bias"].copy_(osd["input.bias"])
        ow=osd["input.weight"]; nw=sd["input.weight"]
        nw[:,:5]=ow[:,:5]; nw[:,5:67]=ow[:,5:67]; nw[:,67:129]=ow[:,67:129]; nw[:,191:209]=ow[:,129:147]
        # Third player starts as the mean of the two old opponent slots.
        nw[:,129:191]=(ow[:,5:67]+ow[:,67:129])/2
        sd["value.weight"].zero_(); sd["value.bias"].zero_()
        # A neutral utility prior; 1800 trunk/policy are retained and value learns from 3P outcomes.
        m.load_state_dict(sd); return m.to(device)
def utility_targets(scores):
    # Scores are absolute 3-seat totals. Tied players receive the average rank,
    # matching the native terminal Max-N utility (unique ranks are 1, 0, -1).
    s=scores.float(); greater=(s.unsqueeze(1)>s.unsqueeze(2)).sum(-1).float(); equal=(s.unsqueeze(1)==s.unsqueeze(2)).sum(-1).float(); return 1.0-(greater+(equal-1.0)/2.0)
def loss_for_batch(model,obs,targets,masks,utilities):
    logits,val=model(obs); logits=logits.float().masked_fill(~masks.bool(),-1e9); pl=-(targets.float()*F.log_softmax(logits,-1)).sum(-1).mean(); vl=F.mse_loss(val.float(),utilities.float()); return pl+vl,pl,vl
