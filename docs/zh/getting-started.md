# 安装与上手

## 需要什么

* Python 3.11 或更高版本。仅此而已：没有任何第三方依赖，也不会全局安装任何东西。
* 至少一个已构建好的 `llama-server`（某个构建或分支）。本工具不负责编译，
  只负责注册已经构建好的东西。

## 获取

```bash
git clone <仓库> ~/git/llamastery
```

仓库是自包含的：`bin/llamastery` 在任何位置都能运行。下文用 `$LM` 指代它。

```bash
LM=~/git/llamastery
$LM/bin/llamastery --help
```

## 作为 Agent 技能安装

```bash
ln -s ~/git/llamastery ~/.config/opencode/skills/llamastery
ln -s ~/git/llamastery ~/.claude/skills/llamastery
```

请使用符号链接而不是复制，这样任何修改所有 Agent 立刻可见。
仓库根目录的 `SKILL.md` 是速查表，其余内容在 `docs/` 中。

## 首次运行

```bash
$LM/bin/llamastery doctor
```

`doctor` 会显示它能看到的一切：已注册的构建、当前 preset、实测数据条数、
compute buffer 校准值、显存总量，以及各构建是否处于最新状态。

如果注册表是空的：

```bash
$LM/bin/llamastery builds detect --apply
```

`detect` 会在常见目录（`~/git/*`、`~/llama*`）中查找分支。
不加 `--apply` 时只显示找到的结果。

## 最小可用流程

```bash
$LM/bin/llamastery validate                    # 所有 section 通过参数校验
$LM/bin/llamastery budget                     # 每个 preset 需要多少显存
$LM/bin/llamastery runtime start --build faks  # 启动路由
$LM/bin/llamastery load <preset>               # 把模型加载进显存
$LM/bin/llamastery measure                     # 实测显存占用
$LM/bin/llamastery probe --tokens 110000        # 真实深度下的速度
$LM/bin/llamastery runtime stop                # 停止
```

顺序很重要：`measure` 和 `probe` 只在已加载 preset 的情况下才有意义。

## 文件位置

| 内容 | 路径 |
|---|---|
| 构建注册表 | `~/.config/llamastery/builds.json` |
| 实测数据与校准 | `~/.local/state/llamastery/` |
| 崩溃日志 | `~/.local/state/llamastery/crashes.json` |
| 服务器日志 | `~/.local/state/llamastery/router.log` |
| 参数表缓存 | `~/.cache/llamastery/` |
| 路由 preset | `~/.config/llama/models.ini` 及其同级文件 |

可以通过 `LLAMASTERY_CONFIG_DIR`、`LLAMASTERY_STATE_DIR`、
`LLAMASTERY_CACHE_DIR` 覆盖 —— 见 README。
工具旧名称对应的前缀有意不再支持。

## 先验证你自己的数据

在信任任何数字之前，先确认确实存在实测数据：

```bash
$LM/bin/llamastery ingest            # 在 tune-results 和注释中找到了什么
$LM/bin/llamastery budget --explain  # 显存的逐项拆分
```

在 `calibrate` 运行之前，显存估值偏低 —— 工具会为此打印警告。
这不是 bug，而是说明校准尚未进行。