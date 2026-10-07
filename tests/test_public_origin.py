import http.client
import threading
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"python"))
from azul_ai.web_game import Application, make_server, normalize_public_origin

root=Path(__file__).resolve().parents[1]
assert normalize_public_origin('https://abc.ngrok-free.app')[0]=='https://abc.ngrok-free.app'
for invalid in ('abc.ngrok-free.app','https://abc.ngrok-free.app/path','ftp://abc.ngrok-free.app'):
    try: normalize_public_origin(invalid); raise AssertionError(invalid)
    except ValueError: pass

class App:
    token='test-token'
    def state(self): return {'ok':True}
    def advice_command(self,*args): pass
    def submit(self,*args): pass

server=make_server(App(),0,'https://abc.ngrok-free.app')
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
port=server.server_port
def get(host):
    conn=http.client.HTTPConnection('127.0.0.1',port);conn.request('GET','/api/state',headers={'Host':host});return conn.getresponse()
assert get(f'127.0.0.1:{port}').status==200
assert get('abc.ngrok-free.app').status==200
assert get('attacker.example').status==403
conn=http.client.HTTPConnection('127.0.0.1',port)
conn.request('POST','/api/options','{}',{'Host':'abc.ngrok-free.app','Origin':'https://attacker.example','X-Azul-Token':'test-token','Content-Length':'2'})
assert conn.getresponse().status==403
conn=http.client.HTTPConnection('127.0.0.1',port)
conn.request('POST','/api/options','{}',{'Host':'abc.ngrok-free.app','Origin':'https://abc.ngrok-free.app','X-Azul-Token':'test-token','Content-Length':'2'})
assert conn.getresponse().status==404  # origin passed; route is simply not an allowed mutating route
server.shutdown();server.server_close()
print('Public origin tests passed: local, configured ngrok, hostile host/origin')
