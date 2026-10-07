# 接口、并行与训练约定

本文描述规则内核和基础批量环境。新增 Full Gumbel 搜索、chance 样本复用、GPU 网络与训练/续训接口见 [TRAINING.md](TRAINING.md)。搜索协议已经追加到同一 C 头文件，基础环境 ABI 保持 1，搜索 ABI 独立为 1。

## 文件入口

| 文件 | 职责 |
|---|---|
| `include/azul/engine.hpp` | 头文件形式的完整 C++20 规则内核 |
| `include/azul/c_api.h` | 固定 C ABI、批量环境、快照接口 |
| `src/c_api.cpp` | 独立环境存储和持久线程池 |
| `python/azul.py` | 纯标准库 ctypes 接口，可创建 NumPy 零拷贝视图 |
| `examples/selfplay.py` | 完整 Python 调用和快照重放示例 |
| `tests/reference.hpp` | 独立数组/逐砖式参考规则实现 |
| `tests/test_engine.cpp` | 专项规则、随机差分和并行一致性测试 |

## 状态与生命周期

`State` 本次 MSVC x64 构建为 **112 字节**；编译期要求不超过 128 字节且可平凡复制。两人墙面分别用 25 位位图表达，图案行用颜色+数量，工厂/中央/袋/弃砖/地板用每色计数表示。地板保留彩色数量以严格验证每色 20 块守恒；没有把尚未结算的地板砖提前混入弃砖。

随机袋用每色数量和无偏有界随机抽样实现，概率等价于洗匀后逐块抽取，不存储对策略有潜在泄漏风险的整袋顺序。RNG 使用 SplitMix64；`bounded(n)` 使用乘法映射和拒绝采样，不直接 `% n` 引入模偏差。seed=0 是合法 seed。这里不提供密码学随机性。

每局有 `draft / chance / terminal` 三种阶段：

```text
initial_state(seed, starting_player) -> draft
draft --apply(最后一组砖)--> 自动铺墙/处罚 -> chance 或 terminal
chance --deal()--> draft
draft --step()--> apply + 若需要则 deal
```

- `legal_actions(s)` 返回栈上固定容量 `ActionList`，使用 `size` 个有效元素；其余数组元素不初始化以减少写入。不得读取尾部。
- `legal(s,a)` 判断动作，`apply/step` 校验失败返回 false 且不修改状态。
- `apply_unchecked/step_unchecked` 要求动作已经合法；Debug 会断言，Release 不再重复检查，错误输入不受支持。
- `apply` 自动执行确定性的本轮结算，停在 `chance`，不会消耗未来补砖 RNG。
- `deal` 仅能在 `chance` 调用，正常 `step` 会自动调用它。
- `winner(s)`：未结束 `-2`；平局 `-1`；玩家 0/1 获胜返回 `0/1`。
- `validate(s)` 检查每色数量守恒、墙面/图案行兼容、工厂容量、阶段等，供调试和测试，不放入训练热路径。

`State` 是高级 C++ 调用者可访问的结构体，常规使用应通过 API 转移，不随意修改字段。完整赋值 `auto child = parent;` 就是搜索分支快照。低级帮助函数如 `settle_round` 仅应在无供应砖的合适阶段调用。

## 动作编号

固定 180 维，编号 0–179：

```text
action = (source * 5 + color) * 6 + destination
source      = 0..4 工厂盘，5 中央
color       = 0 蓝，1 黄，2 红，3 黑，4 白/浅青花纹
destination = 0..4 图案行（容量 1..5），5 地板
```

解码：`source = action / 30`，`color = (action / 6) % 5`，`destination = action % 6`。工厂索引始终保持，不按砖排序，以保证动作 ID 稳定。相同内容的不同工厂仍保留各自的动作编号；未做可能改变策略先验权重的对称裁剪。

`action_mask` 输出 180 个 uint8 的 0/1。终局和待抽砖节点全部为 0。策略必须在 mask 上屏蔽非法 logits；不能对所有 180 项直接采样。小规模随机 rollout 可直接从 `ActionList` 有效区采样。

## 172 维观测

输出 float32，视角由 `observe(s,perspective,out)` 指定。C ABI 默认每局的当前行动玩家。墙面本身不旋转，颜色/工厂编号保持全局固定；仅交换双方板的顺序。

以下用 Python 半开切片表示：

| 切片/索引 | 内容 | 数值约定 |
|---|---|---|
| `[0:30]` | 6 个来源 × 5 色，来源优先 | 数量 / 20 |
| `[30:55]` | 本方墙面 25 格，行优先 | 0/1 |
| `[55:80]` | 本方 5 图案行 × 5 色 one-hot | 空行全 0 |
| `[80:85]` | 本方图案行数量 | 各自除以行容量 |
| `[85:90]` | 本方地板各色彩砖数 | 数量 / 7 |
| `90` | 本方地板占格数，包含放得下的标记 | / 7 |
| `91` | 本方分数 | / 200 |
| `[92:154]` | 对方同样的 62 维字段 | 同上 |
| `[154:159]` | 袋中每色数量 | / 20 |
| `[159:164]` | 弃砖区每色数量 | / 20 |
| `164` | 起始标记仍在中央 | 0/1 |
| `165` | `next_start == perspective` | 0/1 |
| `166` | `current == perspective` | 0/1 |
| `[167:170]` | draft/chance/terminal one-hot | 0/1 |
| `170` | 当前轮数 | / 10 |
| `171` | 袋中总数 | / 100 |

`next_start` 在标记未被拿走前记的是本轮原先手，若整个拿砖阶段无人拿标记则继续先手。分数与轮数只是缩放，**不裁剪**到 [0,1]；轮数可以超过 10，分数也可能超过 200。公开观测不含 RNG state、seed 或未来抽砖。

## MCTS 与随机节点

搜索在 `draft` 选择动作后调用 `apply`；若得到 `chance`，用搜索专用 RNG 样本对状态副本执行 `deal`。例如：

```cpp
auto child = root;
azul::apply_unchecked(child, action);
if (child.phase == azul::Phase::chance) {
    child.rng.state = search_rng.next64();
    azul::deal(child);
}
```

模拟器内部 `rng` 用于重现，不是玩家信息。若直接复制真实状态并始终沿真实 RNG 推演未来，就会把未来抽砖当作确定信息。应独立采样并聚合随机节点的价值；不能把策略 action seed 与环境抽砖 RNG 混用。

## C ABI 批量环境

`azul_batch_create(n,threads,base_seed)` 创建 n 局和一个持久线程池。初始化时先手按 `i & 1` 交替，种子用 `episode_seed(base_seed,i)` 派生。`threads=0` 自动取硬件线程数，线程数不会超过 n。`threads=1` 直接串行执行，无条件变量等待。

- `azul_batch_observe`：可选输出 `float[n][172]`、`uint8[n][180]` 和 `uint8[n]` 当前玩家，任意输出可以传 nullptr。
- `azul_batch_step`：输入 `uint16[n]` 动作；可选输出 `AzulStepResult[n]`，并自动处理下一轮补砖。
- `azul_batch_reset`：重置全部环境，可更换 base seed。
- `azul_batch_reset_at`：按指定 seed 与先手重置某一局，用于异步终止后重开。
- C 函数成功返回 0，参数错误/捕获异常返回 -1；创建失败返回 nullptr。
- 某局动作不合法时不修改那局，结果 `invalid_action=1`；不会中止同批其他局。C 函数总返回值仍是 0，调用者需要检查 per-env 结果。

每个 `AzulStepResult` 恰好 12 字节，包含终局奖励、双方绝对序号的分数、实际行动玩家 actor、terminated、round_finished、invalid_action。

**没有自动重置终局。** 终止状态及终局观测会保留，待训练端先保存 transition 后显式 reset。对已经终止的状态再次 step，会保持终止并标记非法动作，奖励为 0，防止重复发终局奖励。

**同一个 batch handle 不可重入。** 不要同时从两个调用线程 observe/step/reset 同一 handle；内部工作线程已经处理并行。不同 handle 可以并发，独立 `State` 也可以在任意线程运行。不存在全局可变状态或全局 RNG。

批量线程池按连续索引分配工作范围，每次调用同步一次；不在每步创建线程，不在每局内部加锁。对很小的 batch，线程唤醒成本可能高于规则计算，优先设 threads=1。GPU 批推理时建议先测 1024–8192 环境，再调线程数。Python `ctypes.CDLL` 在 C 调用期间释放 GIL；但你自己在 Python 写逐局选动作仍可能成为瓶颈。

## 奖励、终止、样本归属

`terminal_reward` 和 C ABI `reward` 采用稀疏胜负奖励：未结束 0；终局胜 +1、负 -1、共享胜利 0。终局比较包含同分时的横行数量，不是简单的分差符号。

C ABI 的 reward 对应 **results.actor，即这一步动作前的行动者**。返回的下一状态当前玩家可能不同。跨轮时同一个玩家可以连续行动，训练代码不能每一步无条件 `value = -value`；应比较前后玩家编号，再决定是否变号。若训练保存整局所有玩家样本，终局应按各样本的玩家身份分别回填最终结果。

需要分数差、密集奖励或自定义 shaping 时，使用 `scores[2]` 或 C++ 原始状态自行计算；这些不是额外的桌游规则。

规则本身允许一直主动弃砖的无限对局。框架不偷偷设置终局上限；训练器需跟踪 cutoff，将超时区分为 `truncated`。本仓库基准和示例设置 4096 步防护，且不会把截断冒充平局。

## 快照、持久化与复现

C++ 值复制可精确保留状态及 RNG。C ABI `azul_snapshot_size`、`azul_batch_snapshot`、`azul_batch_restore` 提供本进程/同构建快照；restore 先验证状态，错误时不修改环境。批量 restore 不接受 chance 状态，因为其 API 总是自动 deal。

原始快照含结构体布局、字节序和 padding，**不是跨版本磁盘格式，也不是不可信外部文件解析器**。长期保存建议记录：引擎版本、规则模式、明确的初始 seed、先手、完整 action ID 序列。固定版本会逐步确定重放。若使用批量 base_seed，单局实际初始 seed 是 `episode_seed(base_seed,index)`。

`episode_seed` 使用 64 位无符号算术，跨编译器一致。基准的对局 ID 不依赖线程调度；1/2/4/8/16 线程结果应有相同 moves、胜负计数和 checksum。跨版本调整规则、动作空间或 RNG 算法需要版本化，不承诺新旧模型可直接混用。

## 验证范围

- 原版规则书：逐条核对准备、拿砖、图案行、地板、从上到下结算、邻接分数、补袋、终局和平局。
- 穷举 25 个落点 × 32 个横向模式 × 32 个纵向模式验证连续计分。
- 3000 局随机对战逐动作对比独立参考合法性和状态转移；参考模型逐块搬砖、二维格逐格计分，不调用生产版规则函数。随机补砖单独检查确定性、数量守恒、用完旧袋后补弃砖、恰好空袋、不足补砖等。
- 129 个环境，1 与 4 工作线程，300 个批次：比对状态、观测、mask、终局结果及重置。
- Python DLL 示例检查完整对局和快照确定重放。

参考模型共享 `State` 存储格式和由生产版生成的每轮发牌输入，因此差分测试不是第二个独立抽袋算法的证明；概率和补袋验证由专项测试覆盖。Windows Release 和 Debug 已通过；ASan 链接因本机未装运行库受阻，未宣称已经通过 ASan/UBSan 或 ThreadSanitizer。
