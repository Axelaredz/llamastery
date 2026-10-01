# 自动调优：方法、陷阱与已验证的规律

调优引擎是 `tools/tune_models.py`。它分两个阶段，这正是它值得存在的
主要原因：廉价的筛选阶段淘汰垃圾，昂贵的验证阶段淘汰看似合理实则不可用的配置。

## 两个阶段

1. **筛选** —— 在短上下文上跑 `llama-bench` / `llama-sweep-bench`。
   成本低，用来剔除明显很弱的 `n-cpu-moe` × `ubatch` × `t` 组合。
2. **验证** —— 真实的 `llama-server`：完整上下文、真实 KV、
   读取空闲显存、在短与深上下文上各做一次检索测试（大海捞针），
   并检查 draft 质量。

每次运行的产物：`~/.config/llama/tune-results/<阶段>/<时间-pid>/` ——
`results.json`（机器可读）、`screening.json`、`run-NNN.log`。

`results.json` 正是 `llamastery ingest` 汇入 `measurements.json` 的东西。
值得看的字段：

| 字段 | 含义 |
|---|---|
| `short.gen_tps` / `deep.gen_tps` | 4k 与完整上下文下的生成速度 |
| `short.prefill_tps` | 提示处理速度 |
| `min_observed_free_mib` | 运行期间空闲显存的最小值 —— 关键数字 |
| `needle_ok` | 检索测试是否通过（模型未在长上下文下失效） |
| `ok` / `error` | 运行是否撑住了，附原因 |
| `config`、`config_extra` | 候选配置的完整参数集合 |

从实测数据中取**最差值**，而不是平均值：规划要按最坏情况来。

## 调优器不做什么

* 不搜索 `c` —— 那是用户的决定，不是网格搜索。
* 不在模型之间做选择。
* 不检查多个 preset 能否同时装下 —— 那是 `llamastery budget` 的事。

## 实践中遇到的陷阱

### `spec-type` 多值 + 外部 MTP draft

在 Faks 构建中，`--spec-type ngram-mod` 与 `--model-draft`（MTP 头）
同时使用会在加载时 segfault。外部 MTP 头本身就要慢一倍
（24 t/s 对 48），因此在这类模型上，不带 draft 的 `ngram-mod`
是唯一可用的加速器。

### `ubatch-size`

`ubatch-size = 2048` 在 12 GiB 显存上是已知的 OOM 成因，1024 可用。
但这是启发式而非定律：如果某个具体配置已有成功的实测数据，
`llamastery validate` 就会保持沉默。

实测补充的细节：单用 `ubatch-size = 2048` 没问题 ——
实测为 26.4 t/s、剩余 1616 MiB。会崩的是 2048 **搭配**推测解码：
验证图所需的 compute buffer 大于已分配的部分。

### `parallel`

`parallel > 1` 会成倍放大 KV 池。在 12 GiB 上请保持 `parallel = 1`。

### `mmproj` 放在 CPU

`mmproj-offload = 0`（投影器放内存）对生成速度影响很小 ——
实测 38.9 对 38.8 t/s，而且不占显存。在显存紧张的卡上，
视觉模型几乎总是这样选。

### `cache-reuse` 与 mmproj 不兼容

在源码中核实过，`tools/server/server-context.cpp:1220`：

```cpp
if (params_base.n_cache_reuse) {
    params_base.n_cache_reuse = 0;
    SRV_WRN("cache_reuse is not supported by multimodal, it will be disabled");
}
```

加载多模态模型时，`cache-reuse` 会被强制关闭，并在日志中给出警告。
第二道屏障（同文件，约 3164 行）：`can_cache_reuse` 要求
`!slot.prompt.tokens.has_mtmd`，也就是说任何带图片的请求都会
为该槽位关闭 reuse。

**对措辞的重要修正：** 该限制针对 mmproj，而不是 `ngram-mod`，
也不是推测解码。models.ini 中较早的一条记录错误地把它们关联在了一起。

### `spec-draft-*` 与 `ngram-mod` 同时设置无效

已核实：`common/speculative.cpp` 中的 `common_speculative_n_max()`
只遍历 `spec->types` 里**被选中**的类型：

```cpp
case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:      // 以及 EAGLE3、DFLASH
    n_max = max(n_max, spec->draft.n_max);   // draft 模型的旋钮
    break;
case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
    n_max = max(n_max, spec->ngram_mod.n_max);  // 它自己的旋钮
    break;
```

因此当 `spec-type = ngram-mod` 时，`spec-draft-n-max` 与
`spec-draft-p-min` 是失效的 —— 它们控制的是另一个 draft 模型。
ngram-mod 自己的旋钮是 `--spec-ngram-mod-n-max`（64）、
`--spec-ngram-mod-n-min`（48）、`--spec-ngram-mod-n-match`（24）。
一个同时写了 `spec-draft-n-max = 1` 和 `spec-type = ngram-mod` 的
preset，会在无人察觉的情况下使用默认值运行。

### ngram-mod 实际带来什么

在 llama-upstream 0.5.0-dev、Qwen3.8 MiniPlus、114688 tokens 上做的 A/B 实测，
提示相同，唯一差别是 `spec-type` 那一行：

| 探针文本 | 启用 ngram-mod | 不启用 ngram |
|---|---|---|
| lorem（有重复） | **84–92 t/s** | 26.3–26.5 t/s |
| 无重复 | 30.0 t/s | 28.5–30.5 t/s |
| 4k，有重复 | 98.6 t/s | — |
| 4k，无重复 | 49.5 t/s | — |

结论：在重复文本上（代码、模板、对话）加速器带来 ×3.2；
在唯一文本上收益恰好为零，且不产生损害。
因此 `spec-type = ngram-mod` 值得针对有重复的任务按需启用，
而不是「以防万一一直开着」。

方法学上的提醒：调优器的探针（大海捞针）与 `probe` 命令
对同一个 preset 会给出不同数字 —— 48 对 26.5 t/s。
这是两种不同任务，而不是测量不一致：只能在同一方法内部比较。
上面的 A/B 之所以用 `probe` 对 `probe`，正是出于这个原因。

在两个模型上的扩展实测（Qwen3.8 与 Tiel-Coder NanoPlus，faks）：

| 提示 | 启用 ngram-mod | 不启用 ngram |
|---|---|---|
| 重复文本 | 96–107 t/s | 26–28 t/s |
| 110k 上的真实代码 | 25.6 t/s | 28.8 t/s |

在真实代码上加速器**反而有害**：它要为每次落空付出代价，
而且让测量变得更不稳定（两次试次之间相差 5.3–6.6 t/s，
而不启用时只有 0.1–0.5）。

### `--n-cpu-moe` 是如何生效的

它不是 params 中的数值字段，而是逐层的缓冲区覆盖
（`common/arg.cpp:2607`）：对每一层 `i < N`，都会把
`llm_ffn_exps_block_regex(i)` 以强制 CPU 缓冲区的形式加入
`tensor_buft_overrides`。这就是为什么在代码里搜 `n_cpu_moe`
什么都找不到 —— 应当去找 `tensor_buft_overrides` / `ffn_exps`。

对空闲显存的实测影响（Tiel，c=114688，b=2048，`ok=True` 的运行）：
moe24 → 2963 MiB，moe32 → 5028，moe40 → 7129。
斜率是每层 258–263 MiB。解析式算出的专家权重占比（~0.978）
把效果高估了约 13%，这就是 `llamastery calibrate --from-tune`
会自动拟合出的 `offload_realization = 0.89` 系数的由来。

### 防复读

`repeat-last-n = 256` 搭配 `repeat-penalty = 1.02` 是对抗复读的有效组合。
遇到 JSON 损坏或工具调用问题时可以降到 1.05，再低就没有意义了。

## 速度衰减规律

早先 `models.ini` 表头写的是 `tg(D) = 1000 / (16.1 + 0.0025·D)`。
公式是错的：在 D = 114688 时它给出 3.3 t/s，
而同一处表头记录的却是 32.5 t/s —— 系数低了大约 20 倍。

用 Tiel-Coder 的四次实测重新拟合（128k，RTX 3060，未启用 ngram）：

| D | 实测 |
|---|---|
| 9 864 | 48.6 t/s |
| 32 327 | 42.1 t/s |
| 63 932 | 36.1 t/s |
| 114 012 | 32.5 t/s |

得到 `tg(D) = 1 / (0.01961 + 9.786e-8 · D)`：
8k → 49.0，32k → 43.8，65k → 38.4，114688 → 32.4 t/s。

**但应当依赖你自己模型的实测值，而不是公式。** Qwen3.8 系列在 114688 下
实测为 39–41.5 t/s，高于对 Tiel 的拟合值。该规律适用于判断数量级，
以及在同一模型上比较两种配置，但不适用于在模型之间搬运数字。

实际推论：**在较大深度上，翻倍上下文大约只损失 1 t/s。**
因此用于长会话的 128k preset 在速度上几乎不吃亏。

## 测量中的假象

* 在已预热的缓存上测 prefill 会得到虚高的数字。65k 下真实的冷启动
  prefill 约需 50 分钟（约 20 t/s）。
* draft 接受率 1.00 是重复提示带来的假象，不可相信。
  在有变化的文本上接受率是 0.93。
* 会话的第一个请求不会因 draft 而加速（索引还是空的）。

## 值得了解的调优器局限

* **事实抽取检查会误伤推理模型。** 调优器要求模型把一个密钥插入长提示，
  并期望它在 64 个 token（`--n-predict`）之内复述出来。
  带 `<think>` 的模型会把整个预算用在推理上，答案为空，
  `finish_reason: length` —— 于是这次运行被判定为失败，
  尽管它的速度相当不错。用 `--n-predict 256` 可以缓解。
  区分「模型很差」与「模型在思考」要看 `gen_tps`，而不是 `ok`。
* **深度阶段很贵。** 在 114688 上冷启动 prefill 需要几十分钟，
  所以 128k 下的 `--deep --apply` 是以小时计的。实际替代做法是：
  用调优器在短上下文上选定参数，再单独测量真实深度下的速度 ——
  `llamastery probe --tokens 110000` 作用在运行中的服务器上。
  这样更便宜，而且测的正是需要测的东西。
* **可用键的集合随构建的参数表扩展。** 调优器从目标二进制的 `--help`
  取参数，因此带分支参数的 preset（`load-mode`、`ctx-checkpoints`、
  `no-mmproj-offload`）能够通过，这些知识也没有在调优器代码里重复。
* **`--with-spec` 会把加速器保留在测量中。** 默认情况下 `spec-type`
  会从调优运行中剔除：探针由重复文本构成，在其上 ngram 赢得并不真实，
  而内存里的 draft 模型会让显存读数失真。
  传 `--with-spec` 则是明确地要带着它测。

## 挑选 preset 的工作顺序

1. `llamastery schema --grep <子串>` —— 确认这个参数在该构建中确实存在
   （各分支会添加自己的参数，而 `--draft` / `--draft-min`
   在新版本中已被声明为移除）。
2. 确定 `c`（任务的容量）和模型。
3. `llamastery budget <section>` —— 估算显存；不够就降低
   `n-cpu-moe`（MoE 上最强的杠杆）或 `c`。
4. `llamastery tune ... --extra deep` —— 调优器会遍历
   `n-cpu-moe`/`ubatch`/`t`。
5. `llamastery ingest --apply` —— 固定实测数据。
6. 对所有 section 执行 `llamastery budget` —— 检查在 `--models-max`
   限制下多个 preset 能否同时装下。