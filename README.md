# MLIR 动态形状内存规划

2026-11-20 选题进度。当前候选关注 MLIR/IREE 在动态形状下如何确定临时缓冲区的布局。一份编译产物固定冲突缓冲区的相对地址顺序，运行时用实际大小计算偏移。首轮结果支持继续比较布局表示，论文贡献仍待验证。

## BERT-tiny 先导结果

公开 ONNX 模型经 IREE `O2` 编译至 LLVM CPU。在同一 Stream 阶段恢复编译，原版与改写版使用相同的后续 pass。原始发布配置、恢复编译对照与改写版在 11 个输入形状上输出逐字节相同。

| 批大小 × 序长 | IREE 队列峰值（字节） | 固定地址顺序（字节） | 降幅 |
| --- | ---: | ---: | ---: |
| 1 × 16 | 119,936 | 58,432 | 51.3% |
| 1 × 64 | 540,800 | 245,824 | 54.5% |
| 1 × 128 | 1,245,312 | 524,352 | 57.9% |
| 3 × 97 | 2,650,688 | 1,155,968 | 56.4% |
| 1 × 256 | 3,145,856 | 1,704,000 | 45.8% |

峰值来自 IREE `local-task` 的队列分配事件，包含输出缓冲区。每个形状运行三次，记录一致。HAL 内存池的底层提交量在两版均为 64 MiB。另按编译器给出的生命周期计算各时刻同时存活的已对齐临时缓冲区大小，固定地址顺序在 11 个探索形状上均达到这一布局下界。改写版推理延迟待测。[原始内存事件汇总](results/bert-o2-memory.csv) · [输出比较及布局下界](results/bert-o2-correctness.csv)

用同一份 IREE 生命周期和大小表达式回放原生 TorchInductor 2.5.1 的内存池规划。固定 `1×128` 作为提示形状，两种表示使用相同的插入顺序。对于 `1×16` 和 `1×64`，地址顺序方案比树形布局所需容量少 12.5% 和 7.1%。`1×128` 两者均达到下界。采用 `1×64` 作为提示时，原生树在两个较短序列同样达到下界。下一步用另一种具有独立动态维度的模型检验单个固定布局的适用范围，并与运行时择优的两棵树比较。最长路径计算本身已有经典先例。[Inductor 回放表](results/bert-o2-inductor.csv) · [Inductor 源码](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_inductor/codegen/memory_planning.py) · [经典约束图布局](https://janders.eecg.utoronto.ca/1387/readings/seq_pair.pdf)

在保留断言的配置下，普通断言去重将 BERT 的执行区从 9 个降到 2 个，但序长 64 和 128 的四轮配对计时均变慢。随机权重 T5 decoder 夹具的同类改写将执行区从 43 个降到 3 个，证据来自编译 IR。断言去重作为工程对照归档。稀疏交集分派在 SNAP ca-GrQc 上即使逐查询免费选择 merge 或 gallop，比较次数也仅比固定长度比规则少 2.63%，暂不优先立题。[IREE Stream 分区](https://iree.dev/reference/mlir-passes/Stream/) · [SNAP 数据集](https://snap.stanford.edu/data/ca-GrQc.html)

## 复现

使用同一套 IREE 工具及其 Python 环境，版本 `3.13.0rc20261016`，提交 `89ece87b89852100c0a4ac3d515e14842d090967`。该 Python 环境需要 NumPy 和 ONNX，本轮使用 ONNX 1.23.2。硬件为 Intel Core i5-13400。脚本从公开模型的固定版本下载并校验 ONNX，将生成文件放入指定目录。

```bash
IREE_BIN=/path/to/iree-venv/bin bash mlir/run.sh "$(mktemp -d)"
```

在含有 PyTorch 2.5.1 CPU 与 SymPy 的环境中回放树形规划。将 `OUT` 指向上一条命令的输出目录。

```bash
python mlir/inductor_replay.py --dump "$OUT/release/passes.log" \
  --hint-batch 1 --hint-sequence 128 --priority area-bytes
```
