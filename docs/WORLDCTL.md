# worldctl：LingBot 命令行入口

## 当前交接

2026-10-22：从已合并的 main `335b91b91c0d815d97638e67ee9307ed138fff8b`
建立独立分支 `codex/worldctl-foundation-20261022`，工作树
`/home/zhang/ai/lingbot-worldctl-foundation-20261022`。原 community 工作树的未跟踪文件不变，
没有复制实验源码、模型或私有 examples。此文档维护 CLI 使用与交接，产品权威仍见下方冻结文档。

第一片已实现 status、doctor、help。自身 12 项 stdlib unittest 通过，
现役 11 个离线脚本的 129 项检查也通过，共 141 项；日志位于
`D:/study-agent-validation/worldctl-foundation-offline-20261022/summary.json`。
Ruff/格式与新模块 strict mypy 通过。受测代码提交446c346，之后只改本文档，
未重跑相同代码树；最终SHA由本文档提交恢复。
不重新验收模型、GPU、画面质量或真实呈现延迟。

## 怎么用

```bash
python3 worldctl.py status --json
python3 worldctl.py doctor --json
python3 worldctl.py doctor --dry-run --output /tmp/new-report.json --json
python3 worldctl.py help --json
```

- status：读取分支、精确提交和已跟踪/未跟踪改动数量，不显示未跟踪文件正文。
- doctor：检查 Python/Git/必要入口文档，读取依赖包的安装信息。
  不导入 torch/wan，不启动 CUDA、nvidia-smi 或网络，不读取模型权重或凭据。
  缺少重型依赖会明确显示 missing，但不影响离线 CLI 自身可用；runtime_ready 始终未知，
  release_qualified 始终 false，不能把环境检查 PASS 当成可以跑模型。
- 默认只读；显式 output 才写新报告，拒绝覆盖已有文件、符号链接及符号链接父目录。
  同目录临时文件与 hard-link 原子发布，不覆盖并发写入。
- dry-run 可以读取本地元数据，但不创建报告。stdout 始终是一份 agent-cli-result-v1 JSON。
  退出 0 表示本命令成功、2 参数错、3 前提不足、4 执行失败预留、5 超时/取消。
- head_sha 标记实际 CLI 代码；requested_repo_head_sha 单列被检查仓库，二者不混用。

## 命令与原入口映射

| 内容 | 原入口 | 当前处理 |
| --- | --- | --- |
| 当前源码状态 | Git metadata | 只读 status |
| 离线可用性 | pyproject.toml、run.sh/run.ps1、gpu_cleanliness.py | 静态 doctor，不运行原入口 |
| GPU 清洁门 | gpu_cleanliness.sample/verdict | 本片不执行；原 700/6000 MiB 阈值不变 |
| 原模型入口 | run.sh play/check/smoke | 本片不启动；check 也会导入模型，不能作默认 doctor |
| 实验目录与计划 | 已跟踪脚本、原参数与冻结证据 | 下一独立片只做白名单 list/plan，不提供 run |

## 产品边界

继续遵守 `docs/RC_FROZEN.md`、`docs/DEPLOYMENT_PRESETS_FROZEN.md`、
`docs/deployment/RTX5060_8GB_CURRENT_AUTHORITY.md`：当前部署几何 304x528，
预览、权威帧、renderer submit 和 physical present 不能混用；历史实验不是新发布资格。
第一版不添加全局安装包、数据库、服务或新的推理调度器。

## 下一片

静态登记已有实验及原始入口/副作用/参考文档，生成可审阅计划。
运行前仍需单独预算、输入与 GPU 清洁门；未跟踪实验不会自动纳入目录。
