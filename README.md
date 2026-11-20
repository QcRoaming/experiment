# MLIR 编译实验记录

在本机 Intel Core i5-13400 上，BERT 的动态缓冲区布局曾显著降低 IREE 队列峰值。迁移到 T5 并加入双树基线后，固定地址顺序没有达到预设的继续门槛。另在单步解码中定位了一个可用现有 IREE pass 修复的 T5 编译失败。两项工作均为公开模型的探索实验。

## T5 动态内存布局

目标序长和源序长独立变化，批大小固定为 1。一个方案用提示形状 (32,32) 冻结冲突缓冲区的相对地址顺序，执行时根据实际大小计算偏移。对照取自原生 TorchInductor 2.5.1 的分配树。用三个提示形状和两种插入顺序生成六棵树，仅按提示集上最差的容量与下界之比选出两棵，平局先比较三项比值之和，再按计划键排序。两个入选提示为 (8,128) 和 (32,32)，均按面积稳定排序。将两棵树的偏移表达式编入一份 IREE 产物，执行时选容量较小的一棵。[Inductor 实现](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_inductor/codegen/memory_planning.py)

| 目标长 / 源长 | 固定顺序临时区（字节） | 双树临时区（字节） | 固定顺序节省 | 双树后整队列还能节省的上限 |
| --- | ---: | ---: | ---: | ---: |
| 17 / 95 | 120,576 | 102,464 | −17.68% | 0.24% |
| 95 / 17 | 734,592 | 734,656 | +0.01% | 0.55% |
| 63 / 65 | 326,016 | 332,160 | +1.85% | 0.44% |
| 65 / 63 | 346,752 | 346,816 | +0.02% | 0.38% |

若只允许一棵树，按同样规则选中 (32,32) 的默认树。离线回放四个额外形状时，固定顺序平均少用 19.60% 的临时空间。加入两棵树在运行时择优的对照后，同四个预选形状上的平均节省降为 **−3.95%**，继续门槛为 +5%。原版、固定顺序与双树均执行完整 T5 解码器，每个形状的 21 个输出逐字节一致。队列峰值来自 IREE `local-task` 内存事件。每次有两项共存的分配，其中输出占双树队列峰值的 94.4%–96.1%。上表最后一列用 146 个临时切片的同时存活字节数计算，前提是输出接口及调度固定。各版底层内存池的提交峰值相同，均为 64 MiB。[运行时记录](results/t5-memory.csv) · [临时区与下界](results/t5-scope.csv)

## 带 KV 缓存的单步解码

同一版 IREE 的 LLVM CPU 后端能以默认 O2 编译 BART、Marian、Pegasus。T5 在 Flow 到 Stream 转换时失败。某个宿主 `tensor.cast` 将静态维度 1 放宽成动态维度。直接从同一份 Flow 快照恢复仍失败。在宿主 `util.func` 内重跑一次已有的 Tensor-to-Flow 转换后，默认 O2 后段编译成功。IREE 已实现将这种 cast 降为保留维度的 reshape。[IREE 转换实现](https://github.com/iree-org/iree/blob/89ece87b89852100c0a4ac3d515e14842d090967/compiler/src/iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Patterns.cpp#L128)

在批大小 / 历史长 / 源长为 1/32/64 和 2/17/95 的两个输入上，修复产物的 11 个输出逐项通过 ONNX Runtime CPU 对照，最大绝对误差为 4.55×10⁻¹²。普通 O0、关闭断言的 O0，以及在 O2 下将 global optimization 降为 O0 的配置也通过。和关闭断言的 O0 配对比较，宿主修复在两个形状的六轮中分别快 4 轮、3 轮，尚无稳定延迟优势。将同一模型的批大小、历史长、源长分别或组合固定，得到七种原版 O2 均失败的签名。原样应用同一修复，只恢复了固定源长的一种。Marian 与 Pegasus 仅进行了编译检查。这个例子保留作版本相关的工程案例。[编译与输出](results/decode-step.csv) · [配对计时](results/decode-timing.csv) · [固定轴结果](results/decode-specialized.csv)

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

用同一份 IREE 生命周期和大小表达式回放原生 TorchInductor 2.5.1 的内存池规划。固定 `1×128` 作为提示形状，两种表示使用相同的插入顺序。对于 `1×16` 和 `1×64`，地址顺序方案比树形布局所需容量少 12.5% 和 7.1%。`1×128` 两者均达到下界。采用 `1×64` 作为提示时，原生树在两个较短序列同样达到下界。据此选择了上文的 T5 迁移实验与双树对照。最长路径计算本身已有经典先例。[Inductor 回放表](results/bert-o2-inductor.csv) · [Inductor 源码](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_inductor/codegen/memory_planning.py) · [经典约束图布局](https://janders.eecg.utoronto.ca/1387/readings/seq_pair.pdf)

在保留断言的配置下，普通断言去重将 BERT 的执行区从 9 个降到 2 个，但序长 64 和 128 的四轮配对计时均变慢。随机权重 T5 decoder 夹具的同类改写将执行区从 43 个降到 3 个，证据来自编译 IR。断言去重作为工程对照归档。稀疏交集分派在 SNAP ca-GrQc 上即使逐查询免费选择 merge 或 gallop，比较次数也仅比固定长度比规则少 2.63%，暂不优先立题。[IREE Stream 分区](https://iree.dev/reference/mlir-passes/Stream/) · [SNAP 数据集](https://snap.stanford.edu/data/ca-GrQc.html)

## BERT 复现

使用同一套 IREE 工具及其 Python 环境，版本 `3.13.0rc20261016`，提交 `89ece87b89852100c0a4ac3d515e14842d090967`。该 Python 环境需要 NumPy 和 ONNX，本轮使用 ONNX 1.23.2。硬件为 Intel Core i5-13400。脚本从公开模型的固定版本下载并校验 ONNX，将生成文件放入指定目录。

```bash
IREE_BIN=/path/to/iree-venv/bin bash mlir/run.sh "$(mktemp -d)"
```

在含有 PyTorch 2.5.1 CPU 与 SymPy 的环境中回放树形规划。将 `OUT` 指向上一条命令的输出目录。

```bash
python mlir/inductor_replay.py --dump "$OUT/release/passes.log" \
  --hint-batch 1 --hint-sequence 128 --priority area-bytes
```

## T5 布局与单步解码复现

使用上述固定提交的 IREE 编译器、运行时及其 Python 环境，安装 ONNX 1.23.2 和 NumPy。布局实验另需 PyTorch 2.5.1 CPU 与 SymPy。单步解码参考环境用 ONNX Runtime 1.31.1。脚本会下载固定版本的公开随机权重模型并核对 SHA256。每条命令用独立空目录。

```bash
export IREE_BIN=/path/to/iree-venv/bin
TORCH_PYTHON=/path/to/torch-venv/bin/python bash mlir/t5/run.sh "$(mktemp -d)"
DECODE=$(mktemp -d)
PYTHON=/path/to/ort-venv/bin/python bash mlir/decode/run.sh "$DECODE"
```

布局脚本输出 `three/results.csv`、`scope.csv`、六棵树在 13 组形状的静态容量，以及各阶段耗时。解码脚本保存编译诊断和每个输出的误差。断言关闭配置的数值与计时均限于有效输入。用下列命令复现宿主修复与两种现成配置的新一组计时，脚本对三者的六种顺序各运行一轮。

```bash
python3 mlir/decode/latency.py "$IREE_BIN/iree-benchmark-module" \
  "$DECODE/t5/model_params.irpa" "$DECODE/t5/check" \
  --prefix b1p32s64-input --prefix b2p17s95-input \
  --module "O0-strip=$DECODE/t5/O0-strip.vmfb" \
  --module "host-reflow=$DECODE/t5/reflow/host-reflow.vmfb" \
  --module "O2-global-O0=$DECODE/t5/O2-global-O0.vmfb"
```

要复查固定轴，运行 `python mlir/decode/specialize.py "$DECODE/t5/model.onnx" S bound.onnx`，其中第二项可替换为 B、P、S 的任一非空组合。在空工作目录用 `iree-import-onnx bound.onnx --externalize-params -o bound.mlir` 导入，再运行 `bash /path/to/mlir/decode/reflow.sh "$IREE_BIN" bound.mlir out`。另外两款公开模型在结果表中给出了版本及 SHA256，使用相同的导入命令与 `iree-compile --iree-opt-level=O2 --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host` 编译。
