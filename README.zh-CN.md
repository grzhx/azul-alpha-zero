# Azul AlphaZero

[English](README.md)

C++20 原版 Azul 游戏内核与 GPU 纯自对弈框架，支持双人和三人固定彩墙。
本次为**仅算法代码版本**，整理至双人 **2300 代**：Full Gumbel AlphaZero、显式随机补砖、
工厂规范化与 chance 样本合并、批量搜索、策略/价值网络、replay、checkpoint 和离线评测。

最新版源码树不包含网页前端、BGA 采集器、HTTP 服务、本地对战或浏览器自动化。
不支持四人局、灰墙及 Azul 系列其他游戏。

## 棋力和发布基线

**双人和三人模型均远胜于所遇到的人类玩家。** BGA 实测所使用的浏览器工具不包含在本次算法发布中。

此次选择双人 **2300 代**为发布基线。“代”是模型发布迭代，不是单局或数据集 epoch。
模型为工厂等变网络、宽度 256、4 个共享残差块、45 维规则结束特征和独立的 2 块价值塔，
共 **1,031,427 参数**。三人实验仍为从双人 1800 迁移的 500 代模型，639,490 参数，采用三维 Max-N。

2201–2300 阶段完成 **332,484 局、13,118,656 个决策样本、25,623 次更新**，已完成 cohort 零截断。

| 项目 | 2300 阶段配置 |
| --- | --- |
| 搜索 | 每步 2048 模拟、16 候选、深度 96 |
| 并发环境 / 每代决策目标 | 2048 / 131072 |
| chance 上限 | 256 次原始抽样，按出现次数合并概率质量 |
| replay / batch / sample reuse | 1000000 / 1024 / 2 |
| 学习率 | 5e-5 |
| 历史池 | 1800 / 1900 / 2002 / 2100，权重 1:2:3:4 |
| 历史策略样本比例 | 约 40% |
| 策略 / 价值 / utility 辅助权重 | 1 / 1 / 0.1 |
| 标签平滑 / afterstate loss | 0.01 / 0.25 |
| 额外续局和动作排序监督 | 本阶段关闭 |
| 记录硬件 | Ryzen 7 9800X3D + RTX 5070 Ti |

纯自对弈不使用人类棋谱、外部引擎或人工选招标签，历史模型也来自自身训练。
2300 对 2200 的同预算正式评测如下，平局计半分，区间按交换座位的 seed 对计算：

| 双方每步预算 | 局数 | 胜 / 平 / 负 | 得分率 | 配对 95% 区间 |
| --- | ---: | --- | ---: | --- |
| 1024 | 1000 | 541 / 13 / 446 | 54.75% | 51.65%–57.85% |
| 2048 | 1000 | 519 / 17 / 464 | 52.75% | 49.65%–55.85% |
| 4096 | 1000 | 533 / 11 / 456 | 53.85% | 50.75%–56.95% |

后续 2400 对 2300 的三档评测均未显示可靠提升，也不能据此证明退化。
本次排除后续规则结构/CUDA 分支、自适应 chance、均衡复用及 2400 自动续训控制器。
完整改动范围与证据见[2300 发布说明](docs/RELEASE_2300.md)。

Git 不上传模型、replay、完整日志和二进制。可从源码训练；小规模示例只用于安装检查。

## 构建和安装

要求 C++20、CMake >= 3.20，原生测试无需 Python。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Windows 安装 Visual Studio C++ 工具后也可运行：

```powershell
./scripts/build-msvc.ps1 -Configuration Release
```

Python 自动查找 `build-release/`、`build/Release/`、`build/`。
可用 `--native-library` 指定独立训练库；工厂规范化要求 revision-2 动态库。
CMake 提供 `AZUL_NATIVE`、`AZUL_SANITIZE`、`AZUL_SEARCH_MAX_SIMULATIONS`、`AZUL_ANYTIME` 可选配置。

Python 要求 **>= 3.11**（checkpoint 哈希使用 `hashlib.file_digest`）、NumPy 和 PyTorch。
CUDA 版本依据[官方安装选择器](https://pytorch.org/get-started/locally/)安装。

```sh
python -m venv .venv
# 激活 .venv 后执行。
python -m pip install -r requirements.txt
python examples/selfplay.py
```

支持 CPU `--device cpu`；CUDA 路径使用 BF16、pinned memory、推理图、GPU replay 和 learner 图捕获。

## 训练和离线评测

小规模双人检查：

```sh
python scripts/train_optimized.py --output runs/demo --iterations 3 --envs 16 --threads 4 --queues 1 --positions-per-update 1000 --simulations 16 --candidates 8 --width 64 --blocks 2 --end-features --batch-size 64 --warmup 64 --replay-capacity 20000 --updates-per-iteration 4
```

续训到总计 100 次发布：

```sh
python scripts/train_optimized.py --resume runs/demo/latest.pt --iterations 100
```

两个 actor 尚未清理时评测，或先 pin：

```sh
python scripts/evaluate.py --candidate runs/demo/actor-000003.pt --opponent runs/demo/actor-000002.pt --simulations 128 --pairs 500 --output runs/evaluation.json
```

通用评测使用基础搜索接口；`scripts/audit_ai.py` 使用完整优化 tuning 和 CUDA；
`scripts/evaluate_tree_counts.py` 比较多棵独立树的最终动作投票。不同协议的结果不能混同。

- [训练指南](docs/TRAINING.md)
- [结束特征](docs/ENDING_FEATURES.md)
- [历史池](docs/HISTORICAL_POOL_TRAINING.md)
- [工厂对称搜索](docs/CHANCE_FACTORY_SYMMETRY.md)
- [接口](docs/ARCHITECTURE.md)
- [双人规则](docs/RULES.zh-CN.md)
- [三人规则](docs/RULES_3P.zh-CN.md)

## 源码和测试

`include/azul/`、`src/` 为规则、搜索、C ABI 和基准；`python/` 为封装、模型、训练和存储；
`scripts/` 为构建、离线评测与诊断入口；`tests/` 为规则差分、搜索、学习和续训测试。
测试命令见[英文 README](README.md#tests)。

观测包含所有玩家版面、分数、图案行、地板、砖源与公开历史颜色计数，不暴露 RNG 或未来抽砖。
原生局部结果、搜索进度和连续搜索接口作为算法能力保留，不依赖网页或 HTTP。
