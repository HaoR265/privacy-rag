# 构建与运行边界

在 `implementation_v1/` 执行。需要 Python 3.11、C++17、CMake、GMP/OpenSSL 开发库及 CUDA 工具链。当前 GPU 目标为 sm75（RTX2060）；已有核对环境使用 CUDA 12.4 与 GCC 13 CUDA 宿主编译器。管理服务使用 PostgreSQL。

```bash
uv venv --python 3.11.14 .venv
uv pip sync --python .venv/bin/python requirements.lock \
  --extra-index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python --no-deps -e .
bash tools/build_crypto.sh --dependencies-only
.venv/bin/python tools/build_hybrid.py --name local-gpu \
  --token-gpu --matrix-gpu --silu-gpu --enc0-gpu --client-phase-gpu
.venv/bin/python -m private_rag.cli status
```

依赖脚本锁定 Microsoft SEAL 4.1.2（`119dc32e135cb89c1062076a69310d4413ebc824`）和 EzPC/SCI（`f24bf3e022dde3493cf0ffe3ebd92dd73936b734`），拒绝不一致或有跟踪修改的依赖。脚本会访问官方依赖仓库。构建工具要求新目录，生成源码、依赖与产物的哈希登记，不启动模型。

可添加 `--development` 编译隔离诊断入口；这不启用生产服务。目录 `experiments/` 中同时有被主库引用的 GPU 后端源码，不能整体删除。原始数据、临时输出和编译产物不属于源码依赖。

模型锁定为 Qwen3-Embedding-0.6B（`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`）及 Qwen3-1.7B（`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`）。下载与量化工具保留；载荷须另行准备。数值配置为 `v1-f12-coordinated-20261013`，不得将旧运行的配置与新构建随意混用。

本源码包仅提供构建及开发代码，不包含已绑定的新环境运行配置、身份凭据、模型权重、完整运行材料或生产启用依据。正式 worker 对 prepare/query 保持拒绝；不要将关闭门禁改为默认放行。完整模型 R0/R1 使用 RTX2060，不回退到 CPU；R2 的 CPU 密码工作与明文模型 CPU 推理是不同的计算。

发布副本的具体编译结果见验证说明；来源构建的局部协议结果不自动等于发布副本重新运行结果。完整模型未在发布准备过程中启动。

`build_crypto.sh --materialize-evidence` 是原工作区的历史证据整理选项，所需历史脚本及记录未包含在本包中，不能在本包使用。公开数值范围工件保留原字节与哈希；其中指向历史运行记录的字段只作来源标识，相关原始证据未全部随包提供。

开发控制器源码位于 `experiments/full_private_query/`，以构建来源登记和 GPU UUID/PCI 绑定约束子进程。它还需要单独准备的模型工件、输入及服务环境；不能把控制器存在解释为克隆后已具备完整运行条件。构建不启动控制器，也不自动创建定时任务。
