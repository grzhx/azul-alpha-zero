# Azul AlphaZero

[English](README.md)

面向原版《花砖物语 Azul》的 C++20 内核和 GPU 纯自对弈训练框架，支持双人和三人固定彩墙。
实现 Full Gumbel AlphaZero、显式随机补砖节点、chance 样本复用、批量并行环境和
PyTorch 策略/价值网络；提供本地对战界面和 BGA 只读分析面板。
不包含四人环境、灰墙变体或 Azul 系列其他游戏。

## 棋力和训练信息

**双人和三人模型均远胜于所遇到的人类玩家。**

双人内部评测确认的最优 checkpoint 为 **第 1800 代**。“代”指模型发布迭代，
不是单局游戏或对固定数据集训练一遍。

| 项目 | 双人模型 | 三人实验 |
| --- | --- | --- |
| 训练进度 | 1800 次模型发布 | 已续训到 500 次迭代 |
| 初始化 | 随机权重，纯自对弈 | 从双人第 1800 代迁移 |
| 网络 | 工厂盘等变，宽度 256，4 个残差块 | 7 工厂等变，宽度 256，4 个残差块 |
| 参数量 | 623,618 | 639,490 |
| 搜索 | Full Gumbel + 随机节点 | Full Gumbel + 三维 Max-N |
| 末期训练搜索 | 每步 256 次模拟 | 每步 64 次模拟 |
| learner 更新 | 累计 447,533 次 | 前期每代 4 次；401-500 代每代 16 次 |
| 完成对局 | checkpoint 计数为 4,207,953 局 | 每代 128 个并行环境，旧续训段曾重置累计计数 |
| 实测硬件 | Ryzen 7 9800X3D + RTX 5070 Ti | 相同 CPU/GPU |

双人第 1401 代起加入自身历史对手池。1701-1800 代采用 1400/1500/1600/1699 代，
权重 1:2:3:4，约 40% learner 策略位置来自历史对局。
训练不使用人类棋谱、外部引擎或人工策略标签。

1800 代以双方 256 搜索对战 1699 代，2000 局共 1068 胜、42 和、890 负，得分率
**54.45%**，95% 配对区间 **52.375%-56.60%**。双方 1024 搜索下 1000 局得分率
52.15%，区间 49.20%-55.10%，尚不能确认该预算下的优势。

三人训练仍属实验性。500 代是最新本机模型。
当前三人评测晋级门槛为 0.5，而同模型基准约为 1/3；未晋级不能直接解释为退化。
详细限制见[训练指南](docs/TRAINING.md)。

仓库只上传源代码，模型、replay、完整训练日志和下载的规则书不随 Git 分发。
小规模示例只验证安装，达到上述棋力需要长程训练。

## 构建和安装

要求 C++20 编译器、CMake >= 3.20。内核和原生测试不依赖 Python。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Windows 安装 Visual Studio C++ 工具后可使用：

```powershell
./scripts/build-msvc.ps1 -Configuration Release
```

脚本自动运行双人规则、搜索和三人规则测试，输出在 `build-release/`。
Python 自动发现 `build-release/`、`build/Release/` 或 `build/` 的动态库。

训练使用 Python >= 3.10、NumPy、PyTorch。CUDA 版本依据
[官方安装选择器](https://pytorch.org/get-started/locally/)安装。

```sh
python -m venv .venv
# 激活 .venv 后执行以下命令。
python -m pip install -r requirements.txt
python examples/selfplay.py
```

CPU 使用 `--device cpu`。GPU 支持 pinned memory、BF16 推理、CUDA Graph；
双人优化路径另有流水线、GPU replay 和图捕获 learner。

## 训练和评测

小规模双人安装检查：

```sh
python scripts/train_optimized.py --output runs/demo --iterations 3 --envs 16 --threads 4 --queues 1 --positions-per-update 1000 --simulations 16 --candidates 8 --width 64 --blocks 2 --batch-size 64 --warmup 64 --replay-capacity 20000 --updates-per-iteration 4
```

续训到总计 100 次发布：

```sh
python scripts/train_optimized.py --resume runs/demo/latest.pt --iterations 100
```

在两个 actor 尚未清理时评测，或先 pin：

```sh
python scripts/evaluate.py --candidate runs/demo/actor-000003.pt --opponent runs/demo/actor-000002.pt --simulations 128 --pairs 500 --output runs/evaluation.json
```

完整配置、三人迁移、续训和模型存储见[训练指南](docs/TRAINING.md)。
旧版 MLP 训练入口 `scripts/train.py` 保留。

## 本地对战和 BGA

```sh
python scripts/play.py --port 8765
python scripts/monitor_bga.py --port 8770
```

本地双人对战：`http://127.0.0.1:8765/`；BGA 双人/三人面板：`http://127.0.0.1:8770/`。
模型从 `runs/` 读取，AI 分析前需训练或提供兼容 checkpoint。
BGA 采集器只读公开状态，不提交走棋操作。

- [本地对战](docs/LOCAL_PLAY.md)
- [手动发牌](docs/MANUAL_DEAL.md)
- [BGA 面板](docs/BGA_MONITOR.zh-CN.md)
- [接口约定](docs/ARCHITECTURE.md)
- [双人规则](docs/RULES.zh-CN.md)
- [三人规则](docs/RULES_3P.zh-CN.md)

## 源码和测试

`include/azul/`、`src/` 为原生内核；`python/` 为封装、网络、训练和服务端；
`scripts/` 为命令入口；`tests/` 保留差分规则、搜索、续训、历史池和 UI 回归；
`web/`、`bga_web/` 为网页源码。测试命令见[英文 README](README.md#tests)。

观测包含三位或两位玩家的全部版面，以及分数、图案行、地板、公共砖源和颜色计数。
不包含 RNG 或未来抽砖顺序。袋中和弃砖计数基于完整公开历史；历史缺失时，BGA 分析估计。
