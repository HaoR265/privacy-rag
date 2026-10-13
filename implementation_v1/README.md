# 第一版隐私 RAG 实现

依据本项目 `项目文档/第一版机读规范` 的独立规范开始实施。实现目录与原有资料隔离；不使用历史验收草案。

**完整 GPU R1 整数参考已实际跑通；完整隐私系统 R2 尚未运行成功。** 管理、授权交付、实际密码原语、模型工件、独立 R0/R1 参考、原生模型图及会话调度已形成。当前数值配置为 `v1-f12-coordinated-20261013`，机读规范原文已同步。RTX2060 实际完成两段资料、问题、一次1024×28 Prefill与全部255次Decode，生成256个输出槽、30个有效Token，用时950.012秒，`numeric_failure=false`。运行前后工件与计算源码身份一致，未回退CPU。生产入口因完整固定预算R2运行依据尚不具备继续关闭。见[完整GPU R1记录](records/models_coordinated_full_r1_cuda_rag_run.json)、[当前数值配置](artifacts/constants/numeric_profile_v1_qk.json)与[条件范围依据](artifacts/constants/range_evidence_v1_coordinated.json)。

原运行环境已下载两个模型，并按锁定修订形成文件 SHA-256 清单及各 310 个规范张量的 floor f12 量化。本仓库分发清单；新克隆须另行下载和量化：

| 模型 | 固定修订 | 原 checkpoint / 量化目录 |
| --- | --- | --- |
| Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | `artifacts/models/embedding/checkpoint` / `quantized_f12` |
| Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | `artifacts/models/generation/checkpoint` / `quantized_f12` |

## 模块与当前边界

| 目录 | 已有代码 | 已实际核对 / 未接通部分 |
| --- | --- | --- |
| `private_rag/model_input.py` | D/U 本地规范化、双 tokenizer 切块、固定输入槽 | 原模型 tokenizer 与模板实际核对；服务端不收原文或问题 |
| `private_rag/delivery.py`、`vault.py` | 逐用户重新分享、HPKE、OS keystore、D 状态 AES-GCM | 篡改、绑定、份额等式、真实系统密钥库往返；准备输入为合成调试数据 |
| `private_rag/database.py`、`management.py`、`client.py` | PostgreSQL ACL/版本/发布/快照、真实 mTLS API 和 D/U 客户端 | 临时 PostgreSQL 与 TLS 1.3 实际调用；完整同步与撤权已核对 |
| `native/` | SCI OT/Beaver、布尔转换、精确 floor、SEAL BFV 及 c1-only/GC 解码 | 直接链接生产库的两独立进程、TLS、非零/零权重均实际运行；具条件解析噪声界 |
| `compute/` | 秘密数值算子、Top2/I7/提取、两个 28 层模型、模板、固定生成、客户端最终释放 | 数值/会话/工件探针已跑；完整真实模型 R2 未跑 |
| `compute/worker_main.cpp`、`private_rag/compute_process.py` | 独立 worker、仅本方输入的密封 memfd ABI、三连接心跳、启动包装 | 三端口 listener、事件ACK、准备持久化与终态已接代码；独立PG核对通过，完整worker因完整R2证据门关闭未运行 |
| `private_rag/application.py` | U 本地完整查询调用和 D 本地块准备调用 | 实际tokenizer、N=0提示、N>0门禁已核对；完整worker未启用 |
| `references/` | 独立 R0 浮点、R1 整数完整参考 | 完整 R0 的 CPU 运行仅为历史证据；CUDA R1已完成19项小算子及整条固定预算链路；从不被生产入口导入，不能据此声称R2成功 |

完整状态、运行证据和下一步见 [实施进度](records/implementation_progress.md)。文档一致、某个组件运行、完整系统运行是三种不同结论。

当前开发构建为 `prag-native-v1.0.12`，已加入 RTX2060 本方份额后端、共享 RoPE 系数批量查表、完整层连接和 GPU KV 存储。生产模型执行门仍关闭。以下批量内核数据属于已归档的 v1.0.11 构建，保持其原始源码及库身份。

- 批量 GC 实现范围守卫、精确 floor、比较/选择、A2B/B2A、秘密指数与移位、平方和预算树及公开 LUT，输出新鲜份额，失败位保密；固定最多32768个逻辑坐标。
- 完整 RMS 保留原数值域、floor 顺序、六轮 NR 和私有 gamma。主干及逐头归一化按公开最多16行分块。
- 共享 QK/PV 使用用户授权的 BFV 矩阵 Beaver 预处理。两次受保护 BFV 计算的系数是本方新鲜随机矩阵，C 始终持钥；真实 Q/K/V 仍是两方秘密份额。向量及逐元素乘法继续 OT。
- 普通私有权重矩阵、联合 QKV/gate-up、私有 gamma 和固定155648项 Token 查表保持既有语义及参数。它们此前的 v1.0.9 运行有独立源码归档，不冒充本次构建运行。

v1.0.11 当次生产构建、worker构建和诊断构建的密码库哈希一致，worker另存自身二进制哈希；该结论绑定原归档，不能套用后续修改。公开输入的独立整数参考全部在 RTX2060；真实两方密码内核使用 SEAL/SCI CPU 后端。输出重构仅在独立诊断程序内。

| v1.0.11归档运行对象 | 耗时 | 应用层字节 | 核对结果 |
| --- | ---: | ---: | --- |
| RMS：1×128 | 0.401秒 | 45,445,604 | 匹配 CUDA R1 |
| RMS：16×1024 | 14.139秒 | 3,378,020,915 | 匹配 CUDA R1 |
| RMS：16×2048 | 29.402秒 | 6,729,356,883 | 匹配 CUDA R1 |
| Attention：1个查询×1280个键×128维 | 7.216秒 | 651,757,962 | 完整 QK→Softmax→PV 匹配 |
| Attention：16个查询×1280个键×128维 | 56.950秒 | 6,222,295,830 | 完整正常输入匹配 |
| 同形状含无效行、缺失键及Q/K边界 | 56.708秒 | 6,222,295,830 | 输出及秘密失败位匹配 |

共9组完整RMS、4组完整注意力头、29组基础/矩阵核对及13类双方公开拒绝路径。这里的“核对”是实现调试，未设正式测试/验收标准。边界样例预期产生秘密失败位，不作为正常模型成功。RMS使用S端真实gamma映射与哈希/L1检查；Attention使用公开合成Q/K/V与真实数值常量，不声称加载或完成模型层。

16×1280正常头的阶段计量为：共享QK约669.54MB，QK缩放507.77MB，行最大值/掩码492.42MB，exp/LUT及相邻乘法3.5105GB，倒数六NR20.72MB，概率乘法/floor358.05MB，概率检查373.22MB，共享PV181.45MB，PV输出处理108.26MB。原始记录含双端峰值RSS、帧数、方向切换和OT/Beaver/GC材料数量；父阶段包含子阶段，不能重复相加，方向切换也不是网络RTT。时间是本机诊断观察，不外推模型性能。

既有v1.0.9约10.69TB仅是矩阵/Token/gamma的静态子集。当前完整RMS已含gamma，不能再把旧gamma成本加一次；剩余SwiGLU/RoPE/L2、检索、argmax和会话等未形成当前完整成本，完整查询通信与耗时保持未知。见 `records/batch111_cost_scope.json`。

混合后端已实际完成两种模型的完整第0层（各2行，真实权重、全部16Q/8KV头）：Embedding 的2048个最终整数与Generation的4096个整数及失败位均匹配独立CUDA R1，均无数值失败；C端分别约103.759秒/10.736GB、213.51秒/18.378GB。独立L2与两记录全扫描/Top2/I7/证据装配的4103个整数也已匹配，约32.0751秒/1.250GB。这些是有限公开开发样例，不代表完整Embedding或系统成功。

完整单Token Decode（真实28层、词表头、秘密argmax和KV更新）已完成，57345个整数及失败位匹配CUDA R1，numeric_failure=false，C端6249.32秒、应用层500658003539字节；两文档固定257/257/160槽私有Embedding和交付/检索仍在运行。Decode使用明确登记的公开零KV前缀，经新鲜分享后只读写本方GPU份额；没有声称已完成私有Prefill。本阶段不运行完整1024 Prefill及255次Decode。原生与Python生产入口仍在处理输入前返回 `BACKEND_NOT_READY`。六项收尾依据见 [M2 完成记录](records/hybrid112_M2_closure.md)。当前记录见 [混合后端实施记录](records/hybrid112_implementation.md)，原v1.0.11结果见 `records/crypto_batch111_stage_completion.json`。

## 构建

在本目录执行。宿主需要 x86-64 / AES-NI / AVX2、C++17 编译器、CMake、GMP 开发库、OpenSSL 开发库，以及 PostgreSQL。当前实际版本记录在 `records/crypto-dependencies.json`。不要更改系统服务或现有数据库来运行调试入口。

```bash
uv venv --python 3.11.14 .venv
uv pip sync --python .venv/bin/python requirements.lock \
  --extra-index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python --no-deps -e .
bash tools/build_crypto.sh --dependencies-only
PYTHONPATH=. .venv/bin/python -m tools.build_hybrid --name hybrid-release
.venv/bin/python -m private_rag.cli status
```

新克隆先完成本节构建，再按下文“首次重建模型工件”准备权重、tokenizer和配置；在所需文件齐备前，不调用配置物化或模型入口。公开记录的分发边界见[源码发布说明](../PUBLICATION.md)。

首次准备依赖使用`build_crypto.sh --dependencies-only`，不会重建旧原生目录或改写已登记Context。随后`tools.build_hybrid`必须使用新目录，已有名称会拒绝；不指定名称时自动生成新目录。它记录构建前后源码/依赖哈希，检查实际Context与锁定值一致，以及C/S两方prepare/query四个正式入口仍在处理输入前拒绝，完整记录写入新目录的`build_record.json`。本机使用CUDA 12.4及其GCC 13宿主工具，CUDA目标为sm75，C++密码库保持锁定SEAL/SCI。

旧`build_crypto.sh`默认路径仍会构建`native/build`并导出Context；`--materialize-evidence`只供对应历史构建与真实运行记录核对，不用于当前构建登记。源码改变后不能借用旧运行冒充当前结果。源码编译和门禁检查成功不等于模型已运行。

`requirements.lock` 记录本次实际 Python 环境；其中 PyTorch/CUDA、Transformers 供离线工件与独立参考使用。锁定 SEAL/SCI 的实际密码后端仍为 CPU，没有据此声称其天然支持 GPU；原有密码方案、安全假设、模型修订和固定预算不变。

**当前运行政策：R0/R1明文模型运算必须使用RTX2060，禁止CPU模型回退；R2按已授权混合分工运行。** CPU承担SEAL/SCI BFV、OT、GC与Beaver协议，RTX2060承担本方份额整数运算、公开布局和KV份额存储；只传输本方份额、S自有权重和公开常量，不在GPU重构秘密。管理、编译和工件处理可用CPU。实际后端与完整运行证据分别登记，局部核对不打开生产门禁。

RTX2060/CUDA 已恢复。此前 `cuInit=999`、`/dev/nvidia-uvm` 返回 `EIO` 的故障经普通卸载/重载 `nvidia_uvm` 处理，恢复记录中的 `cuInit=0`、`gpu_ready=true`，实际 CUDA FP64 矩阵乘法与 int64 重组小核对通过。外屏连接保持不变，原 KWin 与 OBS 进程保留；没有 GPU reset、PCI 解绑或桌面重启。见 [CUDA 恢复记录](records/cuda_uvm_recovery_run.json)。

任何后续模型命令都必须先在同一项目环境通过 GPU 预检；没有可用 GPU 时立即停止，不改用 CPU：

```bash
.venv/bin/python -m tools.check_cuda_environment
```

该入口检查真实设备打开、CUDA 初始化、RTX2060 选择和小规模 FP64/int64 运算，不加载模型。结果写入 `records/cuda_environment.json`；只有 `gpu_ready=true` 才继续。当前已通过，但不能以旧记录代替下一次运行前的实际预检。

R1 CUDA 后端已实际完成19项公开算子与独立整数参考逐值核对，以及完整固定预算RAG。两段资料各257×28、问题160×28，随后1024×28 Prefill与255次Decode全部执行；即使已产生EOS，也完成全部256输出槽。该公开样例的30个有效Token回答为“方案采用 X25519、HKDF-SHA256 和 AES-128-GCM 作为加密交付方式。”峰值张量分配5,102,529,536字节、CUDA保留5,337,251,840字节；运行后模型进程退出、显存释放。详见[公开回答及来源](records/models_cuda_full_public_answer.json)与[完成/清理核对](records/models_cuda_full_completion_review.json)。

首次GPU完整尝试曾因log2_floor构造57倍临时张量而OOM，后改为等价固定六步整数位移并重新核对19项算子，第二次完整执行成功。没有改模型、精度、数值配置或Token预算。旧CPU记录未保存完整向量，不声称完整CPU/CUDA逐值对照；这个公开输入成功也不保证所有输入均不触发数值失败。

复现此公开小样例（会执行完整固定预算模型，必须先通过上述实际GPU预检）：

```bash
.venv/bin/python -m model_tools.run_full_r1_rag
```

它是独立明文整数参考入口，不进入管理服务或私有worker。混合后端的完整R2仍需端到端协议执行证据。

首次重建模型工件（以下是下载、量化和公开工件导出命令，不是模型推理命令）：

```bash
.venv/bin/python model_tools/fetch_models.py
.venv/bin/python model_tools/quantize_weights.py --model embedding
.venv/bin/python model_tools/quantize_weights.py --model generation
.venv/bin/python model_tools/materialize_template.py
.venv/bin/python model_tools/materialize_constants.py
.venv/bin/python model_tools/export_native_constants.py
.venv/bin/python model_tools/materialize_numeric_profile.py
.venv/bin/python model_tools/range_evidence.py
.venv/bin/python -m model_tools.build_model_manifest
.venv/bin/python tools/materialize_numeric_header.py
.venv/bin/python tools/materialize_configuration.py
```

以上工具的命令参数见各工具源文件（下载/量化工具提供 `--help`）；已有下载在锁定修订目录保留。重新生成数值头后须重新构建 `build/compute`，使 worker 中的编译绑定与实际配置相同。`artifacts/configuration.json` 绑定真实文件哈希，不表示通过数值启用条件。当前配置登记653个文件，其中客户端核对33个公开文件/清单，服务端另核对620个量化权重文件；两个实际 tokenizer 也在绑定之内。实际复核结果见 `records/configuration_bindings_debug.json`。

## 管理服务运行

先创建独立的 PostgreSQL 数据库与账号，再初始化 schema。此示例只初始化项目自己的新 PKI 目录；已有目录会拒绝覆盖。`DB_DSN` 使用用户自己的数据库连接，密码不写入项目记录。

```bash
.venv/bin/python -m private_rag.cli init-pki --directory runtime/pki
.venv/bin/python -m private_rag.cli init-db --dsn "$DB_DSN" --objects runtime/objects
.venv/bin/python -m private_rag.cli serve --dsn "$DB_DSN" --objects runtime/objects \
  --cert runtime/pki/S-management.pem --key runtime/pki/S-management.key \
  --ca runtime/pki/ca.pem
```

使用 `enroll --account-id UUID --role D或U --certificate-sha256 SHA256 --dsn ... --objects ...` 将真实客户端证书关联到账户。公钥注册与资料管理必须在 mTLS 上进行。客户端 API 见 [客户端说明](records/management_client.md)。PKI 生成的是本地部署起点，不自动形成用户账户和授权。

`N=0` 可返回无授权材料；`N>0` 的正式模型调用目前返回固定 `BACKEND_NOT_READY`。没有 R0/R1 明文降级，也没有可通过环境变量开启的模拟路径。查询/释放不自动重试；新的查询需新的受理快照和新材料。

公开取消使用本人证书调用 `POST /v1/sessions/{session_id}/cancel`。已经 claim 的会话会先保持全局占用，协调器回收本任务 worker 和三条连接后才进入 aborted。`compute-once` 的正常 SIGTERM/Ctrl-C 走该清理路径。SIGKILL、断电或数据库不可达后，不自动认定旧计算已经停止；人工确认旧 worker 与连接均已停止后，才可使用内部的指定会话恢复接口。细节与真实独立核对见 [应用接线记录](records/application_review.md)。

## 实现调试入口

这些是实现期间的编译与运行核对，不是正式测试内容、规模或验收规范。只使用公开/合成材料；独立开发可执行文件不链接到生产 worker。

```bash
bash tools/build_crypto.sh --development
cmake -S compute -B build/compute -DCMAKE_BUILD_TYPE=Release \
  -DPRAG_BUILD_ISOLATED_DEVELOPMENT=ON -DPRAG_BUILD_MODEL_DEV=ON
cmake --build build/compute -j1
.venv/bin/python tools/check_management.py
.venv/bin/python tools/check_mtls_management.py
```

原生探针每次需要新的 ledger 路径；已存在路径会拒绝，不能删除旧账本再复用材料。运行方式与实际结果分别在 `records/crypto-implementation.md`、`records/models_progress.md` 和 `records/compute_numeric_debug.json`。临时数据库、证书和监听器由调试入口正常清理，长期模型权重与公开工件保留。

复核当前批量原语与完整内核（公开合成输入，独立两进程mTLS；每次自动生成新的记录UUID，不覆盖旧证据）：

```bash
cmake -S experiments/batch111 -B experiments/batch111/build -DCMAKE_BUILD_TYPE=Release \
  -DPRAG_BUILD_ISOLATED_DEVELOPMENT=OFF
cmake --build experiments/batch111/build -j1
PYTHONPATH=. .venv/bin/python experiments/batch111/run_primitives.py
PYTHONPATH=. .venv/bin/python experiments/batch111/run_failures.py
PYTHONPATH=. .venv/bin/python experiments/batch111/run_attention.py --rows 16 --keys 1280 --mode normal
PYTHONPATH=. .venv/bin/python -c 'from experiments.batch111.run_rms import run; run(2048,16,"normal")'
.venv/bin/python -m tools.check_private_model_policy
```

这些命令会执行隔离SEAL/SCI CPU密码内核，参考计算须通过RTX2060预检，不启用完整模型。`run_attention.py --mode mask_boundaries`用于明确预期秘密失败位的边界调试；不是正常问答成功样例。诊断最多300秒，与正式会话无总时限的规定分开。

`tools.build_batch111_evidence`及其后续整套登记命令属于v1.0.11归档构建；当前源码已经变化，不直接重跑它来给新构建登记旧成绩。源码或编译结果变化时先产生新运行记录，并按运行时的源码快照验证；旧记录不修改哈希来迁就新源码。规范副本通过 `tools.sync_spec_artifacts` 和 `tools.record_spec_revision` 同步，文档校验不运行模型。

v1.0.11归档运行清单位于 `records/batch111_run_registration.json`：`rms`登记9条记录，`attention`登记4条，`primitives`和`failures`各登记1条，`native_library_sha256`绑定当时诊断构建的密码库。每个独立运行命令输出新的UUID记录路径；更换构建后须将相应新记录显式登记，不能仅修改旧记录中的哈希。证据生成器逐项检查记录内全部来源/二进制哈希、账本终态与要求的形状集合；失败则不形成新工件。单独重跑某个核对不会自动覆盖现有清单。

既有 `experiments/pack109_production`、`experiments/batch110` 和 `experiments/batch111` 保留各自库与归档身份。`build_pack109_cost.py`是旧固定子集公式，各版本证据检查器只用于对应源码身份。无需删除旧账本或重建旧目录来获得新的运行记录。

当前混合后端开发构建及有限运行入口：

```bash
PYTHONPATH=. .venv/bin/python -m tools.build_hybrid \
  --name hybrid-next-development --development
PYTHONPATH=. .venv/bin/python experiments/hybrid112/run_layer.py \
  --model embedding --rows 2 --layer 0 --build-directory build/hybrid-next-development
PYTHONPATH=. .venv/bin/python experiments/hybrid112/run_decode.py \
  --build-directory build/hybrid-next-development
PYTHONPATH=. .venv/bin/python experiments/hybrid112/m3_reference.py \
  embeddings records/hybrid112_m3_CUDA_reference.json
PYTHONPATH=. .venv/bin/python experiments/hybrid112/run_m3.py \
  --build-directory build/hybrid-next-development
```

以上入口按需分别运行，已有同阶段活动任务时不要重复启动。完整层、Decode和M3入口均先核对新构建的源码、依赖和输出登记，拒绝没有构建登记或源码已变化的目录；运行前归档当次来源，复制并复核可执行文件，避免后续重编译替换活动程序。构建不执行模型，只有相应运行命令才执行CUDA参考或真实R2。运行记录与实际构建版本逐次绑定。

需要本机CUDA工具链、RTX2060、现有真实量化权重及可用OS密钥库。M3使用临时PostgreSQL（只监听独立Unix socket）、临时PKI和实际管理mTLS，依次执行两次完整私有准备、加密留存、授权HPKE交付、完整同步和一次私有查询Embedding/检索。没有计算总时限，也没有密码材料续算功能；正常完成或失败均关闭临时服务并删除仅本次生成的OS密钥。**已有同一运行未结束时不要重复启动或覆盖其构建目录。** `records/hybrid112_M3_live.log`和对应公开阶段trace反映当前执行进度，活动账本不是完成证据。

一次性读取当前已登记运行的公开进度：

```bash
PYTHONPATH=. .venv/bin/python -m tools.hybrid112_status
```

该命令只统计已写完的公开阶段，不读取秘密份额、不启动轮询服务、不恢复会话，也不把日志近期更新视为进程存活或最终成功的证明。当前1.0.12工件绑定顺序为`tools.build_hybrid112_manifest`、`tools.materialize_configuration`、`tools.materialize_spec_parameters`、`tools.sync_spec_artifacts`、`tools.record_spec_revision`；实际文件核对另用`tools.check_configuration_bindings`。历史运行只对照自身源码归档，尚未结束的运行单列pending。

## 尚需完成

1. 完整GPU R1此前已完成，本轮没有重跑完整模型。后续如有新数值/实现问题，仅按需要运行公开核对；每次先做实际GPU预检，没有GPU就不运行模型。继续保持完整模型禁止CPU回退，不能把安全替代后的零向量当成功，也不能把已运行的单个公开样例变成正式验收标准。
2. 验证已连接的受理快照、计算 socket、原生事件与最终释放在真实完整模型运行中的行为；当前只有独立组件和公开状态核对。
3. M2单Token Decode已完成并核对原始证据；继续完成M3固定预算两文档私有检索，按实际结果登记源码、工件和资源消耗。M1完整层与L2/检索独立入口已有实际依据；完整系统及问答质量尚未验证。不能缩减预算或用明文参考代替R2。
4. 将已形成的独立接口边界继续落实到所有内部张量表示、尺度和运算描述；当前图内部分量仍以固定约定的 C++ 标量/向量表示。

此前完整 R0 在 CPU 上跑完 1024 Prefill 和 255 Decode，属于历史浮点参考证据，不是当前 CPU 运行许可。两次 CPU 完整 R1 尝试均中断：第二次已完成两段资料和问题 Embedding，记录到 Prefill 18/28 层、0 次 Decode，随后按用户要求停止；这两次均没有完整 R1 结果。实际记录见 `records/models_r0_full_rag_run.json` 及两份 `records/models_coordinated_full_r1_rag_attempt*_interrupted.json`。

文档分别确认所记录公开样例的完整GPU R1和局部R2密码内核运行，不宣称形式安全认证、完整R2成功或正式验收完成。机读规范原文直接维护当前定义；工程决定与实际执行记录另存，旧调试失败不改写成成功。
