import sys,ctypes as C
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from azul import Batch
from azul_ai.web_game import Match,validate_config,situation_from_wdl
root=Path(__file__).resolve().parents[1]
for actual,expected in ((situation_from_wdl([.7,.2,.1],0,0),(.7,.2,.1)),
                        (situation_from_wdl([.7,.2,.1],1,0),(.1,.2,.7))):
    assert all(abs(actual[k]-v)<1e-9 for k,v in zip(('human','draw','ai'),expected))
with Batch(1,library=root/'build-play/azul.dll') as b:
    b.manual_reset(42)
    b.manual_deal([0]*20)
    for _ in range(5):
        b.observe();b.manual_step(next(a for a,v in enumerate(b.masks) if v and a%6==5))
    before=b.snapshot(0)
    try:b.manual_deal([0]*20)
    except ValueError:pass
    assert b.snapshot(0)==before
    b.manual_deal([1]*20)
    for color in (2,3,4):
        for _ in range(5):
            b.observe();b.manual_step(next(a for a,v in enumerate(b.masks) if v and a%6==5))
        b.manual_deal([color]*20)
    for _ in range(5):
        b.observe();b.manual_step(next(a for a,v in enumerate(b.masks) if v and a%6==5))
    b.manual_deal([0]*20) # exhausted bag refills from discard
checkpoint=root/'runs/optimized/initial.pt'
m=Match(checkpoint,'optimized/initial.pt',0,42,validate_config({'device':'cpu','simulations':8,'candidates':4}),True)
assert m.state()['waiting_deal'] and m.state()['round']==0
first=m.state()['situation'];second=m.state()['situation']
assert first==second and abs(sum(first[x] for x in ('human','draw','ai'))-1)<1e-6
m.deal([i//4 for i in range(20)])
assert not m.state()['waiting_deal'] and m.state()['round']==1
m.human_step(m.state()['legal'][0]);m.undo()
assert m.moves==0
m.close()
print('Manual deal: first round, round pause, invalid atomicity, refill, web Match and undo passed')

if '--http' in sys.argv:
    import urllib.request,json,re,time
    base='http://127.0.0.1:8766'
    html=urllib.request.urlopen(base).read().decode()
    token=re.search('name="azul-token" content="([^"]+)',html).group(1)
    def state():return json.load(urllib.request.urlopen(base+'/api/state'))
    def post(op,data):
        data['revision']=state()['revision']
        req=urllib.request.Request(base+'/api/'+op,json.dumps(data).encode(),{'Content-Type':'application/json','X-Azul-Token':token})
        urllib.request.urlopen(req).close()
        for _ in range(300):
            s=state()
            if not s['busy']:
                assert not s['error'],s['error']
                return s
            time.sleep(.05)
        raise AssertionError('timeout')
    s=post('new',dict(checkpoint='optimized/initial.pt',human=0,seed='42',manual_deal=True,config={'device':'cpu','simulations':8,'candidates':4}))
    assert s['game']['waiting_deal']
    s=post('deal',{'colors':[i//4 for i in range(20)]})
    assert s['game']['round']==1 and not s['game']['waiting_deal']
    print('HTTP new/manual deal passed')
