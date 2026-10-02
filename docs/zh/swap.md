# llama-swap：导出与后端推荐

`llamastery` 是唯一事实来源（validate + budget + 实测直连 `llama-server`），
`llama-swap` 只是运行时代理：统一端口、热切换、`ttl`。本页说明如何导出、
后端应该用哪个构建。

## 命令

```bash
llamastery swap export --build faks -o ~/.config/llama-swap/config.yaml --dry-run  # 先检查
llamastery swap export --build faks -o ~/.config/llama-swap/config.yaml             # 写入（自动 .bak-<时间> 备份）
llamastery swap export --build faks --only qwen3.8-35B-A3B-miniplus-v2.1-128ctx     # 子集输出到 stdout
llamastery swap status                                # 代理 :8080 + 直连服务器状态
llamastery swap install                               # 下载 v261 二进制到 ~/.local/bin（仅 Linux x64）
```

## 规则

1. 只改 `models.ini`，swap YAML 只能由 `export` 生成，禁止手改。
2. `validate / budget / measure / probe / tune` 只走直连（`:8099`/`:8098`），
   不经过 `:8080`。经过代理测速会失真。
3. 一个 swap 配置只用一个构建。禁止在同一文件混用 `faks` 和 `ik`：
   参数表不同，部分参数会被静默丢弃。
4. 路由构建（`faks`、`upstream`）经 `preset_to_argv` 以 single 模式导出。
   禁止「swap → router → instance」套娃。

## llama-swap 后端推荐哪个构建（推荐）

参考硬件：RTX 3060 12GB + Ryzen 5700X + 32GB RAM。

| 构建 | 什么时候用 | Swap 状态 |
|---|---|---|
| `faks` | **默认。** `models.ini` 预设就是按它写的：`n-cpu-moe`、`fa`、`ctx-checkpoints`、`kv-unified`，env `GGML_CUDA_REGISTER_HOST=1` + `GGML_SCHED_PREFETCH_EXPERTS=1`。导出干净、无警告。 | ✅ 主用 |
| `upstream` | 备用，`faks` 坏了或需要 `ggml-org` 最新特性时。 | ✅ 备用 |
| `ik` | MoE 裸速度最强（IQK/Trellis、FlashMLA），但参数表旧（2024-08 同步）：导出会丢 `n-gpu-layers`、`load-mode`、`kv-unified`、`n-predict`。冲纪录用 `llamastery load --build ik` 直连；放进 swap 只能单独建配置并检查警告。 | ⚠️ 单独 |
| `xing4` | 仅 `xing4` 架构模型。 | ❌ 不混用 |

结论：**swap 默认用 `faks`**（`swap export --build faks`），
`ik` 在 swap 之外手冲纪录。

## 运行

```bash
~/.local/bin/llama-swap --config ~/.config/llama-swap/config.yaml --listen 0.0.0.0:8080
```

安装方式选择二进制：宿主机 CUDA 已可用，docker 没有 nvidia runtime，
unified 镜像还会丢掉 Faks 补丁并要求重映射 HF 缓存路径。
