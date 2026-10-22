# worldctl：LingBot 命令行入口

## 当前交接：2026-10-23 审阅修复

本轮只收口四条 execution-record 审阅意见，不增加模型/GPU执行能力。
分支 `codex/worldctl-record-handoff-20261022`，基线/执行前 HEAD
`1a9f95928763cc3487644991c50f6f851d94290b`，PR #22 尚未合并。
阶段 A 已准备五个 Python 文件；本条记录补入第六个文件 `docs/WORLDCTL.md`。
阶段 B/C 已在 WSL/Linux 真实执行通过；代码与本文档已应用到目标工作树。
本次直接从 WSL 读取目标：HEAD/13205 字节原版吻合，但工作树当前干净；
交接提到的四项无关修改已不在当前 diff。仍保存并比较
eval_two_layer.py、lingbot_fp8.py、run.sh、setup.sh 的字节摘要，不修改或暂存它们。

冻结裁定：Popen → 同步 child_started 写入/flush/fsync → 立即检查截止时间/取消 →
监督循环。启动确认时间计入同一预算，暴露 `start_confirm_seconds`。
除 Linux 实测证明真实缺陷，不改成异步模型；四条意见在验收后用一个提交收口，
不拆散生产端/消费端共享的 `TERMINAL_STOP_CODES` 与 `step_outcome`。
记录目录/文件创建失败应结构化 BLOCKED3 且不启动子进程；新目录项同步；
inspect-run 必须严格配对停止原因和退出码，不能用独立允许集合冒充一致性。

两条接受的限制：
- fsync 没有硬超时。确认窗口计入总预算，只能在调用返回后立即检查和清理；
  fsync 卡住期间不保证到点终止。同步一次调用不等于有界时间。
- SIGKILL 不执行 finally；start_new_session 不会在父进程死亡时自动清理孤儿。
  崩溃探针自己的清理标注 `cleanup_by: probe`，不能算 worldctl 清理保证。

两窗口验收：确认 fsync 后杀父进程，必须保留 child_started、没有终态、inspect 拒绝3；
写确认前杀父进程，必须标为 uncertain-record-missing，绝不推断未启动。
两窗口均需证明真实 SIGKILL/PID，探针清理完自身子进程，不能把未触发窗口当 PASS。
该探针验证进程崩溃可观察性，不模拟断电，不证明磁盘/文件系统跨掉电持久性。

产物集中在 `D:/study-agent-validation/`：源码 worldctl-src/；补丁
worldctl-review-fix-20261023.patch；原五文件备份 worldctl-review-fix-backup-20261023/；
崩溃探针 worldctl-crash-probe-20261023.py；执行脚本 worldctl-review-fix-20261023.sh。
Linux 入口：`bash /mnt/d/study-agent-validation/worldctl-review-fix-20261023.sh`。
门顺序：精确分支/HEAD/范围断言 → 备份六文件 → 应用六文件 → 比较无关文件摘要 →
三组 unittest → 两窗口崩溃探针 → 全量 CPU/offline 脚本门。
输出 worldctl-review-fix-verify-20261023/，已存在时停止而不覆盖。
脚本不提交；任一失败即停，未通过 Linux 门前不提交或合并。
之后检查 Ruff/格式/Linux strict mypy、diff 范围，再单提交；PR 当前 head 审阅/必要门
确认后才能合并，不借旧 head 的检查，不声称 CI 或 GPU/发布验收通过。
本轮实测：三组 unittest 50 项 PASS；全量 25 脚本 308 项 PASS；
两窗口均 CLI -9、inspect3、无终态、探针清理成功，分别为 durable 与
uncertain-record-missing。四个无关文件 SHA256 前后相同。
Ruff/格式与三生产模块 strict mypy（Linux target）通过；两文件仅格式调整经 AST
一致性检查，不重复全量门；最终再跑三组 focused 门并保存精确文件摘要。
全量 summary 的 head 是测试时基线 HEAD，必须结合 validated-source.sha256/applied.patch
识别当时未提交的受测代码；不能把该 head 当作修复已提交的证据。
补丁、备份和探针为离线验收附件，Linux 日志在上述 verify 目录；崩溃样本另归档到其
crash-records/，不依赖 /tmp 留存。单提交/推送后的精确 SHA 从本节所属提交及 PR #22
恢复；本地验收通过不等于远端审阅通过。未自动合并 main。
启动确认窗口保留原 P1 讨论指出的同步 fsync 限制，作为接受的边界而非硬超时修复。
下一唯一执行片：审阅 PR #22 当前修复提交，确认这项接受的边界及其余三条修复后收口。

## 上次交接（2026-10-22）

2026-10-22本轮已交付：PR #20 execution record
head32c02f0a94687e328677edc46e962c976f1af21f，merge53f00d20da0e3d9c68e5eb529ed83abf90e87869；
PR #21 offline接线/inspect-run head7597ed920621953da811762298d4948f76ef3048，merge main
4b5270bb9837628bbefb6e2e8e054d38a542ce1b。均expected-head合并，完整review分页
无线程/无外部审阅结论；repo无Actions工作流、main保护或有效规则，不冒充CI通过。

第一片24脚本281项PASS，第二片25脚本295项PASS；Ruff/格式/五模块Linux
strict mypy/diff/secret0通过。证据 D:/study-agent-validation 下
worldctl-execution-record-offline-20261022/summary.json 与
worldctl-run-inspect-offline-20261022/summary.json。
11条实际CLI worldctl-record-e2e-20261022/manifest.json：真实11脚本offline成功并记录；
成功/1秒超时/缺numpy失败的记录重验保留0/5/4；改日志FAIL4；截断/旧hashBLOCKED3；
dry-run无日志或记录、缺参数FAIL2。记录及日志只在隔离临时目录，原community状态不变。
缺依赖失败是特意用系统Python的环境负例，不当作实现回归；实际正确环境的offline门通过。

记录覆盖实际PID/session启动、UTC时间、单调耗时、退出码、清理状态和日志大小/限额内的SHA（超限仅记大小）；
逐事件flush/fsync。记录失败不继续启动；终态写失败保留观察到的退出码并阻塞。
Popen失败不再误称已开始。inspect-run只读链/事件/旁边固定名称日志，不执行所存命令；
缺结束或成功标题覆盖失败、合并stdout/stderr超预算等都会拒绝。
记录只证明保存内容的内部一致性，非签名/来源认证；不授予模型/手工关联/发布资格。
清理仍限拥有的Linux session，不是沙箱，主动脱离session不在保证内。

第三片留存修复/交接：codex/worldctl-record-handoff-20261022。主线4b5270b源码与
受测7597ed9相同，但复验发现原/tmp两份记录已不存在；旧主线复验未通过，不能借旧文件
作留存证据，具体临时文件清除机制未确认。改offline CLI默认保存到
~/.local/state/worldctl/runs（不改配置），独立0700运行目录/日志/记录均保留在checkout外。
内部record_root可选；拒绝symlink路径，dry-run不建目录，测试使用自己的临时fixture根。
新增3项留存/路径检查，最终候选代码d42ea61b8eccff287f969ca6ccb513a0a765fb0f通过25脚本298项；
Ruff/格式/五模块Linux strict mypy/diff/secret0通过，随后仅文档变更。
worldctl-record-retention-offline-20261022/summary.json与
worldctl-record-persistent-e2e-20261022/manifest.json记录11条新实际CLI、
跨进程留存复验及3份记录/相邻日志的D盘归档副本，源与副本保留0/5/4结果。
第三片PR #22已从文档改为留存修复+交接；旧head的README P2意见要求限定哈希声明，
现已明确仅限预算内日志，超限无hash且不称已验证。最终HEAD/合并状态从Git/PR恢复。
主线最终复验见worldctl-record-main-20261022.json（修复前曾失败，不借旧结果）。
旧临时目录E2E保留为历史样本，不覆盖；新证据另存worldctl-record-persistent-e2e-20261022。
实际合并/最终主线复验状态由本片最终提交和外部清单恢复；修复是行为变更，重新跑最终门。
下一唯一功能片：将受控执行记录与产物/配置身份作可追溯关联，先离线fixture，不追认旧运行。

## 本轮第二片开发记录

2026-10-22第二片：codex/worldctl-run-inspect-20261022，固定已提交32c02f0，
不依赖其他工作树WIP。offline test run默认启用执行记录并写入suite/执行HEAD/基线
上下文，artifacts保留日志+execution.jsonl；dry-run仍无目录/子进程/完整记录。
新增 `test inspect-run --record /path/to/execution.jsonl --json`，
只读4MiB/512事件以内常规非symlink记录，不执行其中命令。校验序号/hash链/时间格式/
事件顺序/实际启动与退出/停止原因/终态，核对旁边固定命名stdout/stderr的大小和SHA。
缺结束记录/非法类型/越界路径/伪造成功终态拒绝；日志变化FAIL4、缺失BLOCKED3；
完整记录保留原失败4/超时5，日志超限保留5且不声称全部hash验证。
检查的是保存记录与文件一致性，不认证执行来源，也不升级任何手工关联/模型/发布资格。
原cleanup仅限拥有的Linux session，不是沙箱；写失败/原始观察码分别报告。
14项新消费端检查+13项记录端检查通过，第一片完整24脚本281项通过；最终门见外部清单。
下一唯一片：将受控执行记录与产物/配置身份作可追溯关联，先离线fixture，不追认旧运行。

## 本轮第一片开发记录

2026-10-22本轮第一片：codex/worldctl-execution-record-20261022，从main
aa47ead58469f1d62dd6fe7b87c08d84ed6eed5e独立开发。内部execute增加可选record，
默认不改变现有调用。开启时在隔离日志目录逐步flush/fsync execution.jsonl：
run_started/step_started/child_started/step_finished/run_finished，包含UTC时间、
单调时钟耗时、实际PID/session/原始退出码/清理状态和stdout/stderr大小/SHA。
链式hash只检查内部一致性，不是签名或可信来源；recording_origin_authenticated=false。
记录失败阻止下一次启动/停止当前进程；缺结束记录不能视为完整；最终写失败保留观察到
的原退出码并BLOCKED3。log超限只记大小不hash超额内容，仍保留原退出5。
record模式限制128命令/参数长度/上下文allowlist/1MiB日志预算，避免记录无限增长。
同时修正启动失败时execution_started误为true：必须实际Popen成功才为true。
13项新轻量进程测试与原10项通过，含记录故障/超时/取消/零启动/dry-run。
未接模型/GPU，不改既有会话清理边界（不是沙箱，主动脱离session无保证）。
下一片：接到offline test run并增加只读inspect-run，核对记录/日志，不授予执行来源权威。

## 上一轮交付

2026-10-22本轮已交付：PR #17 associate
head571bcaeb44d63ee44940548a8f49f96b83da087f，merge1ef5b983215373af89b4527033443cda92ad909f；
PR #18 verify-association head85a8f377510cace7536b860a4137c806039654de，merge main
54b238f61c513f00f5219b1e305757318b8c64c3。均expected-head合并，完整review分页
无线程/无外部审阅结论；repo无Actions工作流、main保护或有效规则，不冒充CI通过。

第一片22脚本255项PASS，第二片23脚本268项PASS；Ruff/格式/三模块Linux
strict mypy/diff/secret0通过。证据 D:/study-agent-validation 下
worldctl-association-offline-20261022/summary.json 与
worldctl-association-verify-offline-20261022/summary.json。
13条真实CLI worldctl-association-e2e-20261022/manifest.json：成功使用明确合成输入，
创建与重验手工关联；改raw字节/换合法fingerprint均FAIL4；篡改hash/权限声明BLOCKED3；
dry-run不生成完整记录且拒绝重验；真实历史preview与合成scene标签不符FAIL4；
合成memory原本失败时，即使关联文件重验匹配仍FAIL4。历史raw hash/mtime及原community
状态不变，无新模型/GPU实验，无真实模型输入hash或来源验收。

关联核对的是文件和已记录字段：preview场景、memory的weight/preset/chunks/chunk_size；
旧结果缺seed/geometry/source-tree/真实消耗输入hash/process-exit/预算/GPU快照时仍未绑定。
手工关联可存失败/未知结果，必须保留原FAIL/BLOCKED；不能把文件一致变成实验通过。
supplied_files_rechecked是当前提供文件校验，不是模型输入/运行来源证明；
run_binding_verified、provenance_verified、results_comparable和执行/发布许可始终false。

主线54b238f整树与受测85a8f37相同，status/合成associate/verify三例通过，原1259
未跟踪文件保留；worldctl-association-main-20261022.json。当前仅文档关闭片
codex/worldctl-association-handoff-20261022，最终SHA/PR由Git恢复；文档不重复相同源码全套。
下一唯一功能片：为受控执行器记录开始/结束、退出码和产物信息，先离线验收，不追认旧运行。

## 本轮第二片开发记录

2026-10-22第二片：codex/worldctl-association-verify-20261022，固定已提交571bcae，
不依赖其他工作树WIP。新增
`experiment verify-association --association saved.json --result raw.json
--fingerprint fingerprint.json --json`（命令一行）。
校验完整手工关联回执的schema/原状态/权限标记/内部hash，再对当前提供的raw和fingerprint
重新检视并构建同一记录。比较canonical hash，不把3.0和3这种不同类型隐式等同。
替换/篡改/未知schema/声称有运行权威的回执拒绝；匹配时保留原FAIL4/BLOCKED3，
不把“文件没变”升级为实验通过。只重验提供的文件，不重新hash模型输入或认证执行来源。
supplied_files_rechecked=true不等于current_inputs_rechecked/run_binding_verified，
后两者及results_comparable/发布/执行权威始终false；dry-run不落报告，报告不覆盖。
新增13项自测和相邻影响集通过；第一片22脚本255项通过，最终门见外部清单。
下一唯一片：为受控执行器记录开始/结束、退出码和产物信息，先离线验收，不追认旧运行。

## 本轮第一片开发记录

2026-10-22本轮第一片：codex/worldctl-association-20261022，从干净main
361c8964c4d5202f050c4d965d7ef0b476613727独立开发。新增
`experiment associate --experiment NAME --result raw.json --fingerprint fingerprint.json
--purpose 'manual historical association' --output new-report.json --json`（命令一行）。
读取合法完整fingerprint并重新inspect原始结果，核对实验类型、内存已记录配置或preview
scene标签；把结果字节SHA、配置/输入指纹SHA、登记schema脚本SHA和缺失证据记入手工关联。
两次读取结果不一致则阻塞。检查失败/指标未知保留原FAIL4/BLOCKED3，不升级为PASS。
结果缺的seed/geometry/source tree/过程输入hash/process-exit/预算/GPU快照明确未验证。
association_kind=manual-association、run_binding_verified/provenance_verified/
results_comparable/execution_authorized/release_qualified始终false；不能追认旧运行。
dry-run可校验但不生成完整关联记录、不落盘；--output沿用非覆盖原子写入。
12项新增检查通过；最终完整门/样本/提交见外部清单和Git。
下一片：重验关联回执与当前提供的两个文件，识别替换/篡改/过期，但不认证执行来源。

## 上一轮交付

2026-10-22本轮已交付：PR #14 memory inspect
headfb46d4659e9182aa12319d1506e58748752b9c30，merge6e859470f78e97461291280781656aae186ffdbb；
PR #15 preview inspect head2cb7211ee407925beef519b1417a959c8d758477，merge main
598c2ac05b87e38ad0019651c34bd075b846961f。均expected-head合并，完整review分页
无线程/无外部审阅结论；repo无Actions工作流、main保护或有效规则，不冒充CI通过。

第一片20脚本229项PASS，第二片21脚本243项PASS；Ruff/格式/三模块Linux
strict mypy/diff/secret0通过。证据 D:/study-agent-validation 下
worldctl-memory-result-offline-20261022/summary.json 与
worldctl-preview-result-offline-20261022/summary.json。
10条真实CLI worldctl-result-e2e-20261022/manifest.json：现存m0/preview原始文件检视
PASS；故意修改的副本能检测汇总矛盾、截断、伪造计数/比值、方向缺失、重复键和参数错；
dry-run不写报告。两个原始文件SHA/大小/mtime与原community状态不变，无新GPU实验。

找到的真实历史m0有65chunk，余量2724.5625MiB、reserved增长0.3236%；原p50标签
实际均值1383.066ms，独立中位数1378.649ms。真实preview有3scene/21交接，错误方向0、
overshoot2、gross0，correction中位0.6131；这些是原记录内一致性，不是新测量。
M0通过仅历史数值门；preview PASS仅逐项公式/汇总一致，可能仍含负面发现。
缺失方向不按0错误放行，旧else文案也不代替真实计数；group回退显式披露。
两个历史文件都缺进程退出、来源/输入/运行身份绑定，不能证明当前部署或发布资格。

主线598c2ac整树与受测2cb7211相同，status/真实M0/真实preview三例通过，原1259
未跟踪文件保留；worldctl-result-main-20261022.json。当前仅文档关闭片
codex/worldctl-result-handoff-20261022，最终SHA/PR由Git恢复；不为文档重复相同源码全套。
下一唯一功能片：提供结果与配置/输入指纹的显式关联；缺执行证据仍标未绑定，不补造来源。

## 本轮第二片开发记录

2026-10-22第二片：codex/worldctl-preview-result-20261022，以已提交fb46d46为父，
不依赖其他工作树WIP。新增同一inspect入口的 `--experiment preview-handoff`。
从逐scene/chunk重算距离比值、错误方向/overshoot/gross计数、整体/start/steady中位数。
未知方向或零baseline导致的NaN保留为缺失，输出合法JSON null并BLOCKED/3，
不能因为wrong_direction=0就判方向正确。缺字段/坏链/类型错BLOCKED3，汇总或公式矛盾FAIL4。
完整一致返回PASS0仅代表检视通过；即使记录含错误方向仍保留负面发现，不是质量验收。
原脚本空group会回退到全部rows，显式报告真实group样本数和fallback；原else文案可能
在wrong=0时也声称有错误方向，显式标记该误报，不改原脚本/结果/判定阈值。
检视不重算图像/latent指标，不证明来源/进程完成/当前部署或人工呈现通过。
新增14项测试与内存14项通过；第一片完整20脚本229项通过，第二片最终门见外部清单。
下一唯一片：给结果与配置/输入指纹提供显式关联；缺少执行证据时保留未绑定状态。

## 本轮第一片开发记录

2026-10-22本轮第一片：codex/worldctl-memory-result-20261022，base main
c8b380af9f66ad9483202e065a53da5ad36bb120。新增
`experiment inspect --experiment memory-gate --result /path/m0.json --json`。
只读历史原始JSON（4MiB、常规非symlink文件、重复键/变化/非法类型拒绝），不运行脚本。
固定原65 chunks / chunk_size 3 / bf16 / temporal_stability_8_2，核对5个内存点和
65个连续chunk，逐项重算增长<2%、margin>=500MiB以及原时延统计。
数值门失败或汇总与rows不符返回FAIL/4，缺失/未知格式BLOCKED/3；
头部overall=PASS不覆盖原始数据失败。没有process-exit记录，不能独立证明no-OOM。
保留原脚本与门；明确原ttfnf_p50_ms实际是排除warmup的均值，另算真实中位数，
不把旧标签误报为p50。历史JSON未绑定输入/代码/seed/geometry/GPU清洁快照，
检视PASS不授予当前部署、执行或发布资格。14项新增自测通过；最终门/回执见清单。
下一片：预览交接原始结果，重算计数与汇总，并把NaN缺失与0错误方向分开。

## 上一轮交付

2026-10-22本轮已交付：PR #11 fingerprint head40d3eeb3fbba5bb1b704fa78c6c849724b4926d5，
merge ac248b23c2ee268e2b8bd02043c3c14690dd8814；PR #12 compare
head1cc2fae55e29dad815da977e6eff519c9c09a272，merge main
8ed7dbb0be76a56dec97a0df20700ec63611d869。两片均expected-head合并，完整review
分页无线程/无外部审阅结论；repo没有Actions工作流、main保护或有效规则，不冒充CI通过。
第一片18脚本203项PASS，最终第二片19脚本215项PASS；Ruff/格式/三模块Linux
strict mypy/diff/secret0通过。证据 D:/study-agent-validation 下
worldctl-fingerprint-offline-20261022/summary.json 与
worldctl-compare-final-20261022/summary.json。比较代码复查修复Python把42.0与42
视为相等的类型漏洞，回执要求精确整数类型；新候选重跑一次完整门。

真实10条CLI回执 worldctl-fingerprint-e2e-20261022/manifest.json：用明确标注的合成
非模型/非张量文件生成两份相同指纹并比较PASS；改字节后比较FAIL/4且仅inputs不同；
篡改回执旧hash被BLOCKED/3；dry-run不hash；不足/缺少预算拒绝；既有真实模型路径
用1字节预算在读取前拒绝，不声称真实模型/张量资格。原真实输入stat及community状态
保持，没有GPU/网络/模型执行。main整树与受测1cc2fae相同，status/list/历史指纹比较
三例通过，原1259个未跟踪文件保留；worldctl-fingerprint-main-20261022.json。
当前文档片codex/worldctl-fingerprint-handoff-20261022只补说明和README入口；
最终SHA/PR由Git恢复，不因文档重复相同源码全套。

用experiment fingerprint时沿用preflight参数并明确设置--max-bytes/--max-files/
--hash-timeout，再用--output保存checkout外新回执；比较见下一节命令。
哈希只绑定快照内容，不认证作者或来源，不保证之后输入未变。相同仅指回执记录的
静态条件相同；没有观察或比较真实实验结果，不提供执行/发布许可。
下一唯一片：检查已登记历史实验的原始结果文件并明确其证据范围，不执行新GPU实验。

## 本轮第二片开发记录

2026-10-22第二片：codex/worldctl-compare-20261022，以已提交40d3eeb为父，
不依赖其他工作树WIP。新增
`experiment compare --left /path/left.json --right /path/right.json --json`。
只读两个experiment fingerprint的完整成功CLI回执，限常规非symlink文件，
每个最多64MiB；拒绝重复JSON键、未知schema、缺字段/类型错/重复角色、路径穿越、
非完整或声称执行/来源权威的回执，重算payload SHA256后才比较。
相同返回PASS/0，不同返回FAIL/4并列出字段；无效输入BLOCKED/3，缺参数FAIL/2。
校验的是回执内部一致性，不认证回执作者，不重查当前模型输入，也不比较实际实验结果。
source_tree绑定整Git树（包括文档），差异会保守标为不同，不默默忽略。
--output沿用非覆盖原子报告，dry-run不落盘；没有模型/GPU/网络操作。
第一片18脚本203项通过，第二片11项新检查及直接影响集通过；最终完整门与
真实CLI样本/合并状态记录在本轮外部清单和Git。
下一唯一片：检查已登记历史实验的原始结果文件并明确其证据范围，不执行新GPU实验。

## 本轮第一片开发记录

2026-10-22本轮第一片：codex/worldctl-fingerprint-20261022，从main
5a9dd8c8cfa743d67b957d21374a3ae990405f73独立开发。新增
`experiment fingerprint`：沿用preflight全部参数，再显式提供
`--max-bytes 67108864 --max-files 1024 --hash-timeout 10`。
只有静态准备检查通过才读完整输入内容并SHA256；不反序列化、不导入模型、不查GPU。
max-files计所有入口及目录，最多4096；max-bytes累计最多64GiB；
hash-timeout最多60秒，在目录遍历和每次分块读取之间检查，不保证打断内核中的慢I/O。
递归记录目录（含空目录）和文件，拒绝symlink/特殊文件/深度>32，检查读取期间及
最终stat/checkout稳定；超预算或变化不生成部分指纹。dry-run不读文件内容。
用CLI --output在checkout外原子保存且不覆盖；不同路径相同内容保持输入身份。
指纹绑定Git树、登记脚本、配置、依赖版本、计划时限和输入字节，不证明来源真实性，
不授予未来运行或发布许可。10项新增检查通过；完整门与最终SHA从本轮清单/Git恢复。
下一片：离线比较两个完整指纹回执，只判断静态条件一致，不冒充真实结果比较。

## 上一轮交付

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
