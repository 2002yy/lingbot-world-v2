# worldctl：LingBot 命令行入口

## 当前交接

2026-10-22后续第一片：codex/worldctl-execution-20261022，固定main
7eff1b3bb5f4c8b3bd498f51e15043e921761d26，独立worktree
/home/zhang/ai/lingbot-worldctl-execution-20261022。新增内部worldctl_execution.execute：
Linux/WSL拥有的session、共享时限、原始退出码/日志、超时/取消/日志预算和清理状态。
不向CLI开放任意命令。自己的会话以外（例如主动setsid脱离的进程）不在清理保证内，
不是沙箱；其他系统返回BLOCKED。日志检测间隔可短暂超过阈值，清理最多另用约2.5秒。
子环境只继承OS变量，不继承提供商凭据/测试选项；设置离线与CUDA禁用，隔离HOME。
正常父进程退出也清理剩余会话成员，确认非zombie进程停止后才cleanup_complete。

9项真实轻量进程测试通过，包含子失败禁止下一步、后代超时停止、父退后清理、
取消/KeyboardInterrupt、dry-run零目录/零子进程、环境隔离、日志超限。
Ruff/格式与新模块Linux平台strict mypy通过；不把Windows分析时缺POSIX符号当运行错误。
本片尚不接GPU/模型实验，不修改模型、阈值或历史脚本。下一片只接既有离线检查白名单。

## 上一轮已合并记录

2026-10-22：两个功能片已合入main：PR #3（status/doctor）head
fac4d86d177ccaadca83e5f32749ea46b2a669c3，merge7249f4b31070126f681c410d73fc5899fabc3219；
PR #2（experiment list/plan）head3cff7afcfcc0a6df52416c913727c555ba89e7c8，
merge2f378cf76cc408129e5349ac5e63dfd6297352f1。均expected-head合并，完整review分页无未解决意见。
服务器main无保护/生效rules、无Actions工作流或check contexts，不冒充CI通过；以本地证据交付。

最终13个离线脚本150项通过，新两个模块strict mypy/Ruff/格式/diff-check/secret0通过；
合并main的完整树与3cff7af相同，未重跑相同源码全套。6条实际CLI回执含既有输入计划、
缺输入和坏预算；原community状态、输入元数据不变，未执行模型/GPU或创建实验输出目录。
受测源码de0a587后只有docs变更。最终文档关闭片是
codex/worldctl-handoff-20261022，实际提交/PR/主线SHA从Git恢复。
交接/验收清单同时保存在D:/study-agent-validation/CLI_IMPLEMENTATION_PLAN_2026-10-20.md。

下一唯一功能片：有预算的原入口执行器与超时/取消停止；先离线验收执行控制，
真实GPU实验仍受独立目的/预算、现有清洁门及输入信任边界约束。之后才做同配置比较。
完整GPU/模型质量/真实呈现资格仍未新增，冻结RC和阈值不变。

## 合并前开发记录

第二片（2026-10-22）：branch `codex/worldctl-catalog-20261022`，worktree
`/home/zhang/ai/lingbot-worldctl-catalog-20261022`，固定父片
`fac4d86d177ccaadca83e5f32749ea46b2a669c3`。实现 experiment list/plan，
只登记下方两个已跟踪历史脚本，不搜索/运行未跟踪实验。基础与新目录自测共21项通过；
现役13个离线脚本共150项检查通过，记录
`D:/study-agent-validation/worldctl-catalog-offline-20261022/summary.json`。
新两个模块 strict mypy/Ruff/格式通过，secret scan 0。受测代码de0a587，之后仅docs变更；
最终SHA由本段提交恢复。当前为本地通过的独立候选，不代表已合main或新GPU资格。

实际CLI业务回执：`D:/study-agent-validation/worldctl-business-e2e-20261022/manifest.json`。
status、doctor、catalog与使用既有模型/资产/缓存路径的memory计划PASS；
缺preview输入BLOCKED/3、预算0参数错FAIL/2。全部单JSON/stdout、stderr空，
没有启动GPU、没有创建实验输出目录；原community状态及TAE/cache文件大小/mtime未变。
没有读取或反序列化张量正文，不能把计划通过当成实验运行通过。

## 实验清单与计划

```bash
python3 worldctl.py experiment list --json
python3 worldctl.py experiment plan --experiment preview-handoff --timeout 120 \
  --tae /path/to/tae.pth --holdout scene=/path/to/trusted-holdout.pt \
  --out-dir /path/to/new-output --json
```

| ID | 原入口与参考 | 计划含义 |
| --- | --- | --- |
| preview-handoff | preview_2a_handoff.py / docs/PREVIEW_2A_HANDOFF.md | 原三帧切换审计；未来执行会加载可信holdout/TAE、使用CUDA并写preview_2a.json |
| memory-gate | m0_memgate.py / docs/M0_MEMORY_GATE.md | 原BF16显存实验，原参数512x320、65 chunks、seed42、temporal_stability_8_2；不是当前304x528部署推荐 |

memory-gate计划需显式 `--weights`、`--assets`、`--base` 目录、`--y-cache` 文件和 `--tae` 文件，
以及 `--timeout`、`--out-dir`。preview计划只接受1～8条唯一tag=path的holdout和TAE。
原脚本SHA256固定；被改写/缺失、符号链接或文档缺失时不可生成计划，不导入脚本或读取张量内容。
只检查输入路径存在和输出目录未存在；不证明模型兼容、数据可信或GPU清洁。

计划返回原argv、cwd、原始参数、后续副作用和显式预算。
`plan_only=true`、`run_available=false`、`execution_authorized=false`，
`budget_enforced=false`：预算只是未来受控执行器的输入，本命令没有启动任何工作。
不创建实验输出目录、不反序列化 .pt、不导入torch/wan、不运行nvidia-smi。
计划PASS仅表示静态计划可生成；原始GPU清洁门、依赖/模型兼容、可信输入和运行授权仍未验收。
run与compare尚未实现，不能把两份配置不明的历史结果说成公平对照。

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
