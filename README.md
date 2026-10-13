# Privacy RAG

基于 Qwen3、BFV 同态加密和两方秘密分享的隐私 RAG 第一版实现。资料提供方准备并授权资料，用户客户端与计算服务方共同执行私有 Embedding、检索、证据装配和生成。

**资料管理、授权交付、密码后端、模型计算图和会话管理的主要代码已实现；完整 GPU R1 与 28 层单 Token 私有 Decode 已取得实际运行结果。** 仓库包含 Python/C++/CUDA 实现、锁定依赖、独立机读规范、数值工件和运行记录，当前工作集中在固定预算链路的集成核对。

两文档私有 Embedding/交付/检索仍在运行，完整 R2 隐私问答系统尚未运行成功，生产入口保持关闭。各阶段结果按实际验证范围分别登记。

## 阅读入口

- [第一版机读规范](项目文档/第一版机读规范/README.md)：角色、安全边界、62 条技术规则、模型修订、数值规则、协议和固定预算。
- [构建与运行说明](implementation_v1/README.md)：依赖准备、模型工件生成、独立参考及开发入口。
- [模块接口](implementation_v1/records/module_contracts.md)：输入输出、形状、数值尺度、保密状态和参与方。
- [工程决定](implementation_v1/records/engineering_decisions.md)：具体实施选择及依据。
- [发布范围与证据边界](PUBLICATION.md)：本次源码快照包含什么，以及如何理解历史运行记录。

## 实现内容

| 模块 | 主要目录 | 当前内容 |
| --- | --- | --- |
| 资料与授权 | `implementation_v1/private_rag/` | 本地切块、授权、HPKE 交付、AES-GCM 存储、PostgreSQL、mTLS 管理 API、快照及会话 |
| 密码原语 | `implementation_v1/native/` | 锁定 SEAL/SCI 的 BFV、OT、Beaver、GC、精确截断及批量矩阵处理 |
| 私有计算 | `implementation_v1/compute/` | 两个 28 层模型图、秘密归一化、注意力、检索、证据装配、生成与最终释放 |
| GPU 与参考 | `implementation_v1/references/`、`model_tools/` | RTX2060 R0/R1 参考、数值工件、量化、实际权重身份与范围依据 |
| 开发运行 | `implementation_v1/experiments/` | 独立两方协议入口及公开输入的调试核对 |

模型固定为 Qwen3-Embedding-0.6B 和 Qwen3-1.7B，具体修订见机读规范。完整模型 R0/R1 使用 RTX2060，禁止 CPU 回退。R2 的 BFV/OT/GC/Beaver 使用 CPU 密码后端，RTX2060 处理本方份额运算、公开布局和 KV 份额存储；不在 GPU 上重构秘密。

## 当前运行依据

- 完整 GPU R1 公开样例已执行：两段资料、问题、1024 槽 Prefill 和 255 次 Decode，生成 256 个输出槽，`numeric_failure=false`。
- 两个模型各 2 行的完整第 0 层，以及独立 L2、两记录全扫描、Top-2、I7 和证据装配，已在真实两方协议中与 CUDA R1 匹配。
- 单 Token Decode 已完成全部 28 层、LM Head、秘密 argmax 和 KV 更新，57,345 个整数及失败位匹配 CUDA R1，`numeric_failure=false`；C 端约 6,249.32 秒，应用层双向累计 500,658,003,539 字节。公开零 KV 前缀不构成私有 Prefill 的运行证据。
- 两文档固定预算私有检索仍在运行，尚无完整结果。
- 正式测试内容、规模和验收标准尚未制定。编译通过、文档检查通过、局部运行成功和完整系统成功分别记录。

## 获取与构建

```bash
git clone https://github.com/HaoR265/privacy-rag.git
cd privacy-rag/implementation_v1
uv venv --python 3.11.14 .venv
uv pip sync --python .venv/bin/python requirements.lock \
  --extra-index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python --no-deps -e .
bash tools/build_crypto.sh --dependencies-only
PYTHONPATH=. .venv/bin/python -m tools.build_hybrid --name hybrid-release
.venv/bin/python -m private_rag.cli status
```

构建需要 Linux x86-64、AES-NI/AVX2、C++17、CMake、GMP/OpenSSL 开发库和 CUDA 工具链。当前 CUDA 目标为 RTX2060 的 `sm75`；完整依赖及工件准备顺序见[实施说明](implementation_v1/README.md)。上述构建和状态读取不执行模型。

模型权重、tokenizer 原文件、第三方源码、构建输出、密钥、数据库、会话账本与历史源码压缩包不随 Git 分发。模型下载和量化工具按锁定修订重新生成运行所需文件；公开清单中的哈希表示原始工件身份，不表示新克隆中已经具备这些文件。
