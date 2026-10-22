# worldctl：LingBot 命令行入口

## 当前交接

2026-10-22本轮已交付：PR #8 preflight head723a4b4eb683578516cbe2ec0a8292e1c98dd4ae，
merge a2b337e2c3176478df223bd183ac50d865d485a7；PR #9 GPU只读快照
head72bfeca0dc3b5ba2dfcbf3bd23ac5bc09c77c0b0，merge main
10fb4d2a9d6469d1c6973964cd633ca8f2bac045。两片均expected-head合并，完整review
分页无线程/无外部审阅结论；repo没有Actions工作流、main保护或有效规则，不冒充CI通过。

第一片16离线脚本178项PASS，第二片17脚本193项PASS；Ruff/格式/四模块Linux
strict mypy/diff/secret0通过。证据 D:/study-agent-validation 下
worldctl-preflight-offline-20261022/summary.json 和
worldctl-gpu-check-offline-20261022/summary.json。
真实六例 worldctl-preflight-gpu-e2e-20261022/manifest.json：既有memory输入静态
PASS；未确认信任BLOCKED/3；缺目的FAIL/2；GPU dry-run无设备调用；缺预算FAIL/2；
一次真实GPU快照PASS（72MiB不明占用、7828MiB free），只代表当时采样。
原输入stat和community状态不变，没有模型/GPU计算，没有创建实验输出目录。
main整树与受测72bfeca相同，status/list/GPU dry-run三条通过，原1259个未跟踪
文件保留，证据worldctl-preflight-gpu-main-20261022.json；不重复相同源码全套。
当前只读文档关闭片codex/worldctl-preflight-handoff-20261022，最终SHA/PR从Git恢复。

使用 `experiment preflight` 时沿用下文plan参数，另加
`--purpose 'check preview handoff' --trusted-inputs`；信任确认并非来源证明。
GPU快照单独用 `python worldctl.py gpu check --timeout 5 --json`；
它没有未来运行许可，也不能代替运行前即时重验。静态检查和快照均未验证模型兼容性。
下一唯一功能片：登记实验配置与输入来源的不可变指纹，之后才同条件比较；
实际GPU执行仍受明确目的/预算、输入来源、原清洁门和执行器接线约束。

## 本轮第二片开发记录

2026-10-22第二片：codex/worldctl-gpu-check-20261022，以已提交723a4b4为父，
不读取其他工作树WIP。新增 `python worldctl.py gpu check --timeout 5 --json`。
只读nvidia-smi两次查询共用1～10秒预算，不导入torch/wan、不启动模型或改GPU状态。
先核对gpu_cleanliness.py冻结SHA256，再调用其原verdict：不明占用<=700MiB且
free>=6000MiB。未知进程显存/查询失败/多GPU/归属错配/采样矛盾一律BLOCKED，
不把未知归属当0；超时退出5，其余无法确认退出3，参数错退出2。
快照通过仅说明当时干净，不代表未来执行许可或模型兼容/质量。
dry-run不查GPU；其他--repo不冒充实际CLI的门。查询子环境只传OS路径变量。
原冻结gate、模型、推理、阈值、部署预设不修改。新增15项模拟遥测检查，
含固定阈值边界、第二次查询失败、单预算、错误脱敏及CLI单JSON。
最终完整门/真实业务回执/PR状态见本轮外部清单与Git。
下一唯一片：登记实验运行配置与输入来源的不可变指纹，为之后同条件比较做准备；
仍不隐含授权GPU实验。

## 上一片实验静态准备检查

2026-10-22下一轮第一片：codex/worldctl-preflight-20261022，从干净main
151f00cb9053e4946dac98f0b45198b55b75d2d8独立开发。
`experiment preflight`沿用plan的输入选项，另要求 `--purpose` 和
`--trusted-inputs`。只允许实际干净CLI checkout，不接受祖先.env；检查输入
非空、无symlink路径、依赖包元数据及原脚本固定的 ~/ai/taehv/taehv.py。
读取输入stat，不读取/反序列化张量，不导入torch/wan/taehv，不查GPU、不建输出目录。
PASS只证明静态前置检查通过；信任确认不是来源证明，兼容性尚未验证，
runtime_started/execution_authorized/release_qualified仍false，不能据此启动实验。
10项新检查+直接CLI影响集通过；完整门及最终提交从Git和本轮外部清单恢复。
测试曾误用unittest discover递归导入wan，系统Python缺easydict；改为仓库现役
逐脚本入口后通过，没有修改模型依赖或生产代码来迎合错误测试命令。
下一片：只读GPU检查，复用冻结700/6000 MiB阈值，不启动模型。

## 前一轮交付

2026-10-22：本轮两个独立功能片，执行器 codex/worldctl-execution-20261022
从 main 7eff1b3bb5f4c8b3bd498f51e15043e921761d26 开始；CLI 接线
codex/worldctl-test-run-20261022 固定已提交父片 feeae71caba05c0cdbd4203a78c1e7369281a6f2。
不是依赖其他工作树的未提交内容。两片已合并：PR #5 head feeae71caba05c0cdbd4203a78c1e7369281a6f2，
merge c5b21fa7c79360d62a83d0f8a9fecae49de535d4；PR #6 head
b827afa12a3dc2d8adbf01e0cb0e6e665fd3b575，merge main
6e0dbdd282298e5337b8684bb16d4706a2d96ed3。均expected-head合并，完整审阅线程为空。
仓库没有Actions工作流、必需check或生效main规则；无CI可等待，不宣称CI成功。

新增 `test run --suite offline --timeout 120`，只运行下面白名单中11个既有CPU检查。
必须在实际CLI干净checkout运行，祖先 .env 也会阻止执行；每个脚本与固定
7eff1b3 基线blob核对，每一步重查HEAD/干净状态/来源。不会执行未跟踪实验。
使用含numpy的现有Python环境；缺依赖会按真实子进程失败返回，不伪装成通过。

```bash
/home/zhang/ai/lingbot-env/bin/python worldctl.py test run --suite offline --timeout 120 --json
/home/zhang/ai/lingbot-env/bin/python worldctl.py test run --suite offline --timeout 120 --dry-run --json
```

stdout是单个JSON，子进程输出转到stderr并保存在隔离的 `/tmp/worldctl-run-*`。
`--output` 报告应写在checkout外，禁止覆盖，保留各子日志链接。
失败立即停止后续检查；子失败退出4，超时/取消/日志超限退出5，前置条件阻塞退出3。
整组共用一个预算，不是每个脚本重置预算。dry-run不创建日志或子进程。

Linux/WSL执行器会在正常父进程退出、超时、取消时清理其拥有的session，
回传原始退出码与cleanup_complete；主动setsid脱离的进程不在清理保证内，
这不是沙箱。其他平台阻塞；日志检测会短暂超过阈值，清理另需最多约2.5秒。
子环境只继承OS变量，隔离HOME，禁用CUDA/在线模型下载，不继承提供商凭据。

执行器最终14脚本160项通过；接线最终15脚本168项通过（其中CLI共39项）。
Ruff/格式、Linux strict mypy、diff、secret scan 0通过，两个功能候选干净。
证据 worldctl-execution-final-20261022/summary.json、
worldctl-test-run-offline-20261022/summary.json 位于 D:/study-agent-validation。
六条真实CLI回执 worldctl-test-run-e2e-20261022/manifest.json：11脚本成功、
dry-run零子进程、1秒预算超时且清理完成、缺numpy真实失败、错误suite/零预算拒绝。
合并main与最终受测候选整树相同；status/doctor/离线dry-run三条主线检查通过，
未重复相同源码全套。worldctl-execution-main-20261022.json记录原community的
1259个未跟踪文件全部保留、无tracked脏改动。当前文档片只补交接，不改变受测行为。
此轮没有GPU/模型运行，不修改冻结RC、模型、阈值、预设或历史实验脚本。
下一唯一片：给已登记实验补执行前的预算/输入/环境核验，真实GPU执行仍需明确目的与预算。

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
