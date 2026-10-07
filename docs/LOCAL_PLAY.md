# 本地／ngrok 人机／人人对战

启动：

```powershell
./.venv/Scripts/python.exe scripts/play.py --port 8766
```

通过 ngrok 公开时，必须把公网源地址配置给服务端。ngrok 的地址不能依赖浏览器自动推断，否则服务端无法区分合法隧道和任意伪造 Host：

```powershell
# 例如 ngrok 显示的 Forwarding 地址为 https://abc.ngrok-free.app
./.venv/Scripts/python.exe scripts/play.py --port 8765 --public-origin https://abc.ngrok-free.app
```

也可以使用环境变量：

```powershell
$env:AZUL_PUBLIC_ORIGIN = "https://abc.ngrok-free.app"
./.venv/Scripts/python.exe scripts/play.py --port 8765
```

首次通过 ngrok 免费域名打开时，ngrok 可能先显示 “You are about to visit...” 提示页。点击其中的 Visit Site 后即可进入 Azul；这是 ngrok 的入口提示，不是 Azul 服务错误。浏览器正常进入应用后，API 请求会使用同一个公网 origin。

服务仍只监听本机 `127.0.0.1`，由 ngrok 转发公网请求。配置后允许本机地址和这个精确的公网 origin；其他 Host/Origin 返回 403。POST 请求仍要求页面随机生成的 `X-Azul-Token`，状态修改还要求 revision 匹配。不要把 `--public-origin` 设置成带路径、查询参数或通配符的地址，也不要把服务直接绑定到 `0.0.0.0`。

选择 `人机对战` 或 `人人对战` 后开始新对局。人人模式由玩家 1、玩家 2 在同一页面轮流操作，状态栏和蓝色棋盘边框标识当前走棋方；不会触发 AI 自动落子。手动发牌可与两种模式组合使用。

AI 建议面板随时可以展开。人机模式只在玩家回合提供建议，AI 回合隐藏建议内容；人人模式为当前走棋方提供建议。固定预算为 64、256、1024、4096 次模拟；`无限` 以 1024 次为一批持续更新，直到点击停止、关闭面板或局面发生变化。每次局面变化都会使旧建议作废并在面板保持展开时重新分析当前方。

建议列出前三个合法动作和搜索策略权重；点击“在棋盘标出”会选中对应来源并高亮目标行/地板，不会自动落子。权重是搜索策略占比，不是该步的胜率。局势条继续显示 checkpoint 的 W/D/L 预测。

无限搜索不是数学意义上完成无限次计算，而是无预设预算上限的持续任务。它会占用所选 CPU/GPU，尤其在训练同时运行时；可切换对战 AI 配置的设备，或随时停止建议。建议分析基于局面快照，绝不修改真实对局。
