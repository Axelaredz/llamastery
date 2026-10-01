# 显存预算、实测与槽位规划

## `budget` 会显示什么

```bash
llamastery budget                              # 每个 preset，以及 --models-max 的槽位预测
llamastery budget <preset> --models-max 2 --reserve 1024 --explain
```

它把显存拆分为若干项：权重（已计入 `n-cpu-moe`）、KV 缓存、mmproj、
compute buffer。它还能回答另一个问题 —— N 个 preset 能否同时装下？
路由会让每个已加载的模型各自成为一个独立进程，因此显存是所有模型累加的。

```bash
llamastery calibrate <preset> --free-mib 1147
llamastery calibrate a=1147,b=992 --free-mib     # 逐个 section 精确指定
llamastery calibrate --from-tune                # 依据调优器的运行结果
llamastery calibrate --from-log                 # 依据服务器自身的报告
```

在 `calibrate` 运行之前，估值偏低 —— 工具会打印警告，这不是 bug。
残差离散度很大意味着结构化模型不可信，应当依赖实测数据。

## 实测胜过推算

如果某个 preset 已有真实的 `used_mib`，「装不装得下」的判断就以它为准，
估值并列显示以便核对。否则，一个能正常工作但模型不够精确的 preset
会显得装不下：`tiel-coder-nanoplus-128ctx-mmproj-moe16` 就是这样 ——
估值给出 11.54 GiB，而实际只用了 10.42 GiB。

```bash
llamastery measure                    # 从显卡读取并记录
llamastery measure --recalibrate      # 同时重算 compute buffer
```

只有在 preset 已加载时，测量才有意义。

`calibrate --from-log` 比 `--from-tune` 更准确：它直接从服务器日志中读取
紧邻 `n_ubatch` 的 `CUDA0 compute buffer size` 行，也就是缓冲区由服务器自己
报出。缺点是服务器并非每次加载都会打印这个分解（在路由模式下，
实例的日志有时不会进入共享日志），所以数据点很少。
较旧的做法（用总量减去其余部分求余量）会把固定开销算进去，
因而把缓冲区高估约 0.3 GiB。

## 真实深度下的速度

调优器用浅上下文省钱，而真正的工作发生在长上下文上。
因此深度要在运行中的服务器上单独测量。

```bash
llamastery probe --tokens 110000 --max-tokens 96 --repeats 2
llamastery probe --tokens 4096 --repeats 1          # 短深度，用于画曲线
llamastery probe --from-file server.cpp --tokens 110000   # 用真实文件作为提示
llamastery probe --image ~/photo.jpg                # 验证视觉（mmproj）
llamastery probe ... --record --preset <section>   # 写入 measurements.json
```

测量通过一次普通请求完成：服务器自己统计 timings，
因此得到 `prompt_n / predicted_n / *_ms`。

**提示缓存被显式关闭：** 请求中带有 `cache_prompt: false`。
若不这样做，重复同一段文本会命中已预热的 KV，`prompt_n` 变成 4 而不是 110000，
该次测量的 prefill 会低上数倍 —— 在图表上看起来就像「突然变快了」。
这样的试次会被标记为 🔥（预热）并不计入 prefill 统计。
如果一整轮试次都是预热的，工具会如实说明其中不存在冷启动 prefill。

**测什么比重复几次更重要。** 用编造的词表拼出的「无重复」提示会让模型
部分陷入循环，于是结果取决于该构建如何应对循环，而不是真实速度。
要得到诚实的数字，请用 `--from-file` 加上真实的代码或文本；
文件被截断处会补一个续写尾巴，否则模型会认为文件已结束，
在第一个 token 就输出 EOS。尾巴会逐个尝试直到模型开口，
因为同样的提示语并非在所有构建上都有效。

在 114688 上的实测差异（Qwen3.8 与 Tiel-Coder NanoPlus，faks）：

| 提示 | ngram-mod | 不启用 ngram |
|---|---|---|
| 重复文本 | 96–107 t/s | 26–28 t/s |
| 110k 上的真实代码 | 25.6 t/s | 28.8 t/s |

加速器在重复文本上有 ×3–4 的收益，而在唯一文本上**反而有害**：
它要为每一次落空付出代价。应按有重复的任务来启用，而不是一直开着。

## 崩溃日志

```bash
llamastery crashes                       # 哪些 preset 会让服务崩溃，原因是什么
llamastery crashes --forget <preset>     # 清除记录：该 preset 已验证可用
```

某个 preset 可能加载成功、通过参数校验，却只在深上下文下崩溃 ——
`ubatch 2048 + ngram-mod` 在 114688 上就是这样让 CUDA 挂掉的。
这类崩溃会被自动记录（加载失败时，以及 `probe` 期间实例挂掉时），之后：

* `validate` 会警告那些曾让服务崩溃的 preset；
* `presets annotate` 会在 preset 区块里写入 💥（该 preset 曾让服务崩溃）一行；
* 一次成功的测量会自动清除记录 —— 已验证的 preset 不该继续被列为会崩溃。

存储位置：`~/.local/state/llamastery/crashes.json`。

## 汇总实测数据

```bash
llamastery ingest             # 在 tune-results 和 models.ini 注释中找到了什么
llamastery ingest --apply     # 保存到 ~/.local/state/llamastery/measurements.json
```

来自 `tune-results/*/results.json`（由调优器生成）和 `models.ini` 各 section
上方注释的实测数据，会被合并进同一个存储，并代入 `budget` 和 `validate`。
键是配置签名，因此只差一个参数的 preset 会落到自己的记录里，
不会继承别人的数字。

## 速度衰减规律

早先 `models.ini` 的表头写着 `tg(D) = 1000 / (16.1 + 0.0025·D)`。
这个公式是错的：在 D = 114688 时它给出 3.3 t/s，
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

## 计算方式

```
每 token 字节数 = 注意力层数 × n_kv_heads × (bytes(K) × head_dim + bytes(V) × v_head_dim)
缓存大小        = 每 token 字节数 × ctx × 槽位数
```

其中 `bytes(q8_0) = 1.0625`，`bytes(f16) = 2`，`bytes(q4_0) = 0.5625`。

两个陷阱：

* **GQA。** `n_kv_heads` 通常远小于 `n_head`（在 Qwen3-35B-A3B 上是 16 对 2），
  所以 KV 比「朴素」算法算出来的小好几倍。
* **混合架构。** 在 qwen35moe / Nemotron-H / Jamba 中，部分层是 SSM（Mamba 类），
  它们的状态不随上下文增长，也不保存 KV。
  只有每隔 `full_attention_interval` 层才有完整注意力。
  `budget` 会考虑这一点；「把所有层都算上」会把显存低估好几倍。

`kv-unified`（多数现代 preset 都已开启）表示整个上下文共用一个 KV 池：
`parallel` 不再成倍放大缓存。关掉它，成倍关系就会回来。