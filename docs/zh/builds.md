# 构建、分支与参数表

## 注册表

记录源码目录、remote、是否支持路由（`--models-preset`）、环境变量以及端口。
并非所有构建都有路由：`ik_llama` 是单进程模式，它的 preset 必须手工翻译成
argv。

```bash
llamastery builds list                        # 已注册哪些、是否构建完成、版本
llamastery builds detect                      # 在常见目录中查找分支（只读）
llamastery builds detect --apply              # 写入注册表
llamastery builds show faks
llamastery builds add mine --path ~/git/mine-fork \
              --remote https://github.com/u/mine --no-router --port 8098
```

当 8099 已被占用时，无路由的构建需要 `--port`：否则第二个构建会以
`couldn't bind to server socket` 失败。构建的环境变量在同一处设置，
启动时自动生效：

```bash
llamastery builds add faks --path ~/git/llama-faks \
              --env GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1
```

## 版本新鲜度

```bash
llamastery builds stale
llamastery builds stale --no-fetch            # 不访问网络
```

它区分三种容易混淆的状态：

| 状态 | 含义 | 处理方式 |
|---|---|---|
| 落后于上游 | 存在上游没有的提交 | `git pull` |
| ahead | 存在上游没有的本地提交 | 无需处理：重新构建会保留它们 |
| 二进制比源码旧 | HEAD 动过，但构建没有重新编译 | 重新构建 |

它还会指出是否改动了携带参数的文件（`arg.cpp`、`server-context.cpp`）：
在分支上，新提交改变的不只是速度，还有参数集合 —— 最好在测量之前就知道。
`doctor` 里也会打印同一行，让这项检查成为日常流程的一部分。
它不会自动重建：决定权仍在人手里。

更新之后：

```bash
cd ~/git/llama-upstream && git pull --ff-only
cmake --build build -j$(nproc)
llamastery schema --build upstream --refresh   # 刷新参数缓存
```

## 参数表

参数表从具体二进制的 `--help` 解析而来，因此分支专属参数
（`load-mode`、`image-min-tokens`、`ctx-checkpoints`、`spec-type`、
`kv-unified`）会自动出现。缓存以二进制文件的 mtime 为键，
所以重新构建会自动使旧缓存失效。

```bash
llamastery schema --build faks                # 这个构建有多少参数
llamastery schema --build faks --grep moe     # 按关键词搜索
llamastery schema --build faks --json | jq '."--spec-type"'
llamastery schema --refresh                   # 不使用缓存
```

如果需要弄清楚「这个分支到底能做什么」，就从这里开始。

参数表在三处被使用：把 preset 的键翻译成 argv、`validate`，
以及自动调优器。正因如此，关于分支参数的知识没有在代码中重复。

## 无路由的构建

`ik_llama` 不支持 `--models-preset`。`llamastery load --build ik`
会把 preset 的 section 翻译成 argv，并重启一个单进程服务器：

```bash
llamastery load <preset> --build ik --dry-run   # 只看 argv，不启动
llamastery load <preset> --build ik
```

翻译遵循该构建自己的参数表：它没有的参数会被丢弃并给出警告，
`gpu-layers` 会被识别为与 `n-gpu-layers` 相同的键。

这类构建的寻址方式不同：服务器地址来自注册表和 `LLAMA_SERVER`，
而不是默认的 8099。`probe` 和 `measure` 通过 pid 文件恢复构建信息 ——
每次 CLI 调用都是独立进程。

## 进程识别

构建的识别方式是把 `/proc/<pid>/exe` 与注册表中的 `server_bin` 路径做比较。
因此，如果你把某个构建重建到了新路径，却没有同步更新注册表条目，
`status` 就无法再认出占用端口的进程。

## xing4_0 移植版

原生 llama.cpp 中没有 `xing4_0` 架构 —— 本地构建和上游都没有。唯一可用的
是 `jmarceno/llama.cpp-xing4` 仓库的 `xing4_0-port` 分支（此前由
`shuxiaoqiong` 维护同一分支）。在 `detect` 中它叫 `xing4`
（`~/git/llama-xing4`）。

需要记住三点：

1. **没有 fork 专属参数。** 这几乎就是纯上游：没有 `n-cpu-moe`，也没有
   `load-mode`/`ctx-checkpoints`。专家混合 offload 用原生
   `-ot`/`override-tensor`（`blk.(…).ffn_*_exps.weight=CPU`）完成 ——
   `budget` 按 GGUF 张量表精确计算。
2. **KV 按 MLA 计算。** `xing4_0` 使用压缩 KV（`kv_lora_rank` + rope 部分，
   每 token 每层 576 个元素），普通 GQA 公式会高估近一倍 —— `budget`
   会自动采用 MLA 公式。
3. **提交影响速度。** 已验证：`b2056929`（“量化 MLA KV 解码加速”）在
   RTX 3060 上处处慢于父提交 `63c16fb`（gen 30.7 对 36.7 t/s @10k，
   prefill 178 对 639 t/s，深上下文崩溃）。锁定提交是有意为之，
   `builds stale` 会提示新提交。