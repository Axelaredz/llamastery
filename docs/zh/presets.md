# llama.cpp 的 preset 格式

三个层级的配置。它们之间的混淆是几乎所有误解的根源，所以先看这张表。

| 层级 | 文件 | 键 | 是什么 |
|---|---|---|---|
| 每模型 preset | `models.ini` / `presets.ini` | `--models-preset PATH` | section = 模型，`[*]` = 共享默认值 |
| 具名、可共享 | 空 HF 仓库中的 `preset.ini` | `-hf user/repo` | 把 preset 当作带标签的「模型」 |
| 所有二进制通用 | `/etc/llama.cpp/config.ini`、`~/.config/llama.cpp/config.ini` | 无需参数 | 对 `llama-cli` 同样生效；只读取 `[*]` 和第一个标题之前的 section |

生效顺序（由弱到强）：
`config.ini` → 环境变量 → 模型 preset → 路由自身的命令行参数。
也就是说，命令行传入的参数**优先于** preset 中的值。

关于「谁控制什么」的权威来源是 `tools/server/server-models.cpp`
中的 `unset_reserved_args()`：

* 会从任何 preset 中剔除：`ssl-key-file`、`ssl-cert-file`、`api-key`、
  `models-dir`、`models-max`、`models-preset`、`models-autoload`；
* 路由在派生子进程时覆盖：`port`、`host`、`alias`；
* 在每模型 preset 中，`model` / `mmproj` / `hf-repo` **就是**模型本身，
  因此是合法的（路由只从基础 preset 中剔除它们）。

## 同一个键的三种写法

等价，可以在同一个文件里混用：

```ini
ctx-size = 32768     # 长写法
c = 32768            # 短写法
LLAMA_ARG_CTX_SIZE = 32768   # 环境变量名
```

逻辑：先把各种写法去重（短写法与长写法算作同一个参数），
再与路由的基础参数合并。

布尔值：`on` / `off`、`true` / `false`、`1` / `0`、`enabled` / `disabled`。
无值参数（`kv-unified`、`jinja`）—— 值留空即可。

## 例子

```ini
version = 1

; 所有模型共享的默认值
[*]
n-gpu-layers = 99
fa = true
cache-type-k = q8_0
cache-type-v = q8_0

[my-model-65k]
m = /models/Qwen3-30B-A3B-Q4_K_M.gguf
c = 65536
n-cpu-moe = 18
temp = 0.6
top-p = 0.95
top-k = 20

[my-model-65k-vision]
m = /models/Qwen3-30B-A3B-Q4_K_M.gguf
mmproj = /models/mmproj-Q8_0.gguf
mmproj-offload = 0
image-min-tokens = 1024
c = 65536
n-cpu-moe = 18
```

## 路由从哪里找到模型

1. `~/.cache/llama.cpp`（或 `LLAMA_CACHE`）—— 已缓存的 HF 模型；
2. `--models-dir PATH` —— 只看直接子目录，**不递归**；
3. `--models-preset` —— 带显式路径（`m = ...`）的 section。

名称冲突时优先级为：preset > models-dir > cache。
对于多片段模型和 `mmproj`，文件放在子目录中，
投影器文件名应以 `mmproj` 开头。

路由的相关参数：`--models-max N`（默认 4，`0` 表示不限）、
`--models-autoload` / `--no-models-autoload`。

## 关于显存需要理解什么

每个已加载的模型都是一个**独立的 `llama-server` 进程**，
跑在 `127.0.0.1` 上某个空闲端口，路由只做请求转发。因此：

* 显存是所有同时加载的模型累加的，而不是按「preset 大小」计算；
* `LLAMA_SERVER_ROUTER_PORT` 会传给子进程；
* 单个 preset 内的 `parallel` 是另一条轴：它在单个进程内部成倍放大 KV 池。

`llamastery budget` 正是用来回答「N 个 preset 能否同时装下」这个问题，
现有的各类 GUI 启动器都没有回答它。

## KV 缓存结构

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
  `llamastery budget` 会考虑这一点；「把所有层都算上」会把显存低估好几倍。

`kv-unified`（多数现代 preset 都已开启）表示整个上下文共用一个 KV 池：
`parallel` 不再成倍放大缓存。关掉它，成倍关系就会回来。

## 对 preset 的操作

```bash
llamastery presets list                      # 所有 section
llamastery presets show <section>             # 该 section 的键
llamastery presets globals                   # [*] section
llamastery presets export -o - <section>...   # 导出子集
llamastery presets annotate --dry-run        # 格式化工具会改什么
llamastery presets annotate --apply          # 应用
```

### 导入他人的 preset

来源：本地文件、URL，或 `git-仓库#分支:内部路径`。

```bash
llamastery presets import --source ./foreign-presets.ini --dry-run
llamastery presets import --source 'https://github.com/u/repo#main:presets.ini' --dry-run
llamastery presets import --source git@github.com:u/repo.git --only my-model-128k
llamastery presets import --source ./p.ini --on-conflict new      # 不覆盖
llamastery presets import --source ./p.ini --on-conflict overwrite --conflicts
```

行为：默认 `--dry-run`；冲突时 `skip`；写入前会生成 `.bak-<时间戳>`。
`--rename 旧名=新名` 用来重命名 section。目标文件不存在时会创建。

### 校验

```bash
llamastery validate                          # 当前 models.ini 的所有 section
llamastery validate <section> --json
llamastery validate --build ik               # 针对指定构建校验
llamastery validate --no-paths                # 不访问磁盘（更快）
```

它能发现：未知的键、路由的控制参数（`api-key`、`models-max` 会被剔除；
`port`/`host`/`alias` 会被覆盖）、不存在的 GGUF 文件、
超过训练上下文的 `c`、超过层数的 `n-cpu-moe`、已知的分支陷阱，
以及出现在崩溃日志中的 preset。

每个构建的参数表都不同，因此为某个分支写的 preset 在 ik_llama 上会产生警告：
有一部分参数在那里根本不存在。提示信息会指明具体的参数名。

## 加载与卸载

```bash
llamastery runtime start --build faks      # 启动路由
llamastery runtime status                  # 端口、pid、构建、显存中有什么
llamastery load <preset>                   # 加载进显存
llamastery runtime unload [model]          # 卸载，服务器继续运行
llamastery runtime restart --build faks
llamastery runtime stop
llamastery runtime logs -n 100
```

`llamastery` 自己管理服务器 —— 不需要、也不使用任何独立管理器
（`llama`、`llama-faks`、`llama-ik`）。这是有意为之：
为每个分支单独写脚本的人并不多，而工具应当在没装这些脚本的人手里也能工作。

加载之前会针对**该**构建的参数表校验，并预测显存：

```
preset: qwen3.8-35B-A3B-miniplus-21-128ctx-ngram-mmproj
build: faks   binary: /home/axel/git/llama-faks/build/bin/llama-server
  VRAM: 8.67 GiB / 12.00 GiB，余量 +2.96 GiB
```

有错误时不会执行加载（`--force` 可强制）。

其他模型会先被卸载：显存只有一份。能同时加载几个模型由路由的
`--models-max` 决定。

section 上方注释的书写规范见 [comments.md](comments.md)。