import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from azul_ai.web_game import Match,Application,validate_config

root=Path(__file__).resolve().parents[1]
checkpoint=root/"runs/optimized/initial.pt"
config=validate_config({"device":"cpu","simulations":8,"candidates":4,"chance_cap":8})

match=Match(checkpoint,"optimized/initial.pt",0,42,config,False,"human")
assert match.state()["mode"]=="human" and match.state()["current"]==0
match.human_step(match.state()["legal"][0])
assert match.state()["current"]==1
match.human_step(match.state()["legal"][0])
assert match.moves==2

app=Application(root/"runs")
app.match=match
app.view=match.state()
app.advice_command("start",{"budget":"quick"})
for _ in range(400):
    result=app.state()["advice"]
    if result["status"] in ("ready","error"):break
    time.sleep(.025)
assert result["status"]=="ready",result
assert result["simulations"]==64 and len(result["suggestions"])==3
assert all(item["action"] in match.state()["legal"] for item in result["suggestions"])
app.advice_command("start",{"budget":"infinite"})
for _ in range(800):
    result=app.state()["advice"]
    if result["simulations"]>=1024:break
    time.sleep(.025)
assert result["status"]=="running" and result["simulations"]>=1024,result
app.advice_command("stop",{})
assert app.state()["advice"]["status"]=="stopped"
app.close()

human_ai=Match(checkpoint,"optimized/initial.pt",0,42,config,False,"ai")
human_ai.human_step(human_ai.state()["legal"][0])
app=Application(root/"runs");app.match=human_ai;app.view=human_ai.state()
try:
    app.advice_command("start",{"budget":"quick"})
    raise AssertionError("AI turn advice must be rejected")
except ValueError as exc:
    assert "玩家方" in str(exc)
app.close()
print("Local modes/advice passed: both human turns, finite, infinite, stop, AI-turn restriction")
