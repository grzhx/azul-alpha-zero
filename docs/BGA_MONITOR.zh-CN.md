# BGA 双人 / 三人监控

服务入口：`http://127.0.0.1:8770/?relay=1`。

## 启动

在项目根目录执行：

```powershell
./scripts/build-msvc.ps1 -Configuration Release -OutputDirectory build-bga-3p
./.venv/Scripts/python.exe scripts/monitor_bga.py --port 8770
```

监控使用独立 `build-bga-3p/azul.dll`，无需替换训练中的 `build-release/azul.dll`。
编译脚本执行双人引擎、双人搜索和三人引擎测试。

将监控页的“挂载 Azul 监控”拖到书签栏，在 BGA 页面点击该书签。
升级采集器或重启服务后，重新从监控页更新书签并挂载。
采集器只读取公开对局信息，不执行 BGA 走棋操作。

## 人数与模型

- 双人：5 个工厂和中央，沿用 `Batch / PipelineSearch`。
- 三人：7 个工厂和中央，使用 `Batch3 / Search3 / Evaluator3`。
- 工厂按 `factory1` 到 `factory7` 的 DOM ID 读取，排除玩家临时持砖区。
- DOM 暂不可用时，BGA `factories[0]` 为中央，`factories[1..N]` 为工厂。
- 模型兼容性通过 checkpoint 的 `engine_version` 判断，当前支持 `azul3-v1` 和 `azul3-stable-v2`，模型列表按当前人数筛选。
- 三人默认 `runs/three_player_maxn_longrun/latest.pt`；双人默认 `runs/optimized/latest.pt`。
- 两种人数分别记住用户选择的模型。新分析会检查文件版本，读取训练发布的新权重。
- 支持 `actor-3p-*.pt` 三人发布文件。

三人搜索复用现有 C++ Gumbel / Max-N 实现。CUDA 可用时使用 GPU、BF16 推理、固定批量
CUDA Graph 和 pinned-memory 传输；同一模型与批量尺寸的 Graph 会跨局面复用。
局面改变或点击停止会在搜索叶节点请求之间取消旧任务。

## 搜索预算

预算是所有并行搜索根的模拟数之和，不是单棵树的深度。

| 档位 | 并行根 | 每根模拟 | 合计 |
| --- | ---: | ---: | ---: |
| 快速 | 1 | 64 | 64 |
| 标准 | 4 | 64 | 256 |
| 深入 | 8 | 128 | 1024 |
| 极深（默认） | 16 | 256 | 4096 |
| 无限 | 16 | 256 | 持续叠加各轮结果 |

仅在 BGA `chooseTile` 阶段提供完整拿砖 / 放行建议；玩家持砖等待选行、轮末计分等中间
阶段暂停分析，避免将尚未放置的砖误当成袋内或弃置区砖。

## 局势与袋子

双人保持固定左侧绿、右侧红的胜 / 平 / 负局势条。
三人依个人版图固定顺序使用绿、红、金色的名次条，不随当前行动方轮换。
三人模型输出排名效用，显示 `预期名次 = 2 - utility`，范围为 1 到 3；这不是获胜概率。
网络的三个输出按行动方旋转，需要转回固定玩家顺序后显示。
终局显示确定名次，同分时使用完整横行数判定并列。

完整公开行动记录可以重建袋内颜色。重建结果需要与当前图案行、地板、袋内总数以及
可见砖颜色守恒校验一致才能标为精确；历史缺失或动画中间状态时使用约束估计。

## 回归验证

```powershell
node tests/test_bga_collector.js
./.venv/Scripts/python.exe tests/test_bga_three.py -v
```

采集器回归覆盖 2/3 人工厂编号、DOM / gamedatas 备用解析、真实三人桌的完整历史重建
和中间阶段识别。Python 回归覆盖三种座位的公开状态导入、非法导入保持原状态、三人
GPU 搜索、双人建议、模型人数校验、终局和搜索取消。
