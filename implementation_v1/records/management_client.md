# D/U 管理客户端实现记录

`private_rag/client.py` 提供实际同步 `httpx` 客户端，使用现有 TLS1.3-only、X25519、内部 CA 和客户端证书；客户端核对服务端 Ed25519 叶证书，可额外 pin SHA256。关闭环境代理和 HTTP redirect，不自动重放写操作。所有传输/远端错误变为固定 `PublicError`，不回显响应内容；没有问题原文上传入口。客户端连接关闭由上下文管理器保障。

`UserClient` 在 OS `RecipientKeyStore` 中保留长期 HPKE 私钥，只注册公钥。服务增加仅本人可读的公开密钥状态和注册时的 expected_previous_version 事务比较，拒绝复用既往公钥，避免并发代次错配。新私钥先保存再注册，响应丢失时只读公开状态核对；不会删除可能已成功注册的私钥。

`synchronize()` 每次取得完整 manifest，创建新 `UserShareCache`，下载并认证全部 HPKE 包，验证全部绑定、尺度所属的配置标识和排序后才原子替换本地集合。`SynchronizedCollection` 包含规范 JSON manifest 和不可变 U 份额 tuple，原始秘密不会通过 repr 输出。失败使本实例不可受理新会话，之前已经取得的不可变集合仍可供已受理会话持有。`accept_query()` 只发送 manifest SHA256/config 和新随机 session UUID，核对服务返回的完整快照、相同 UUID 和固定形状；每次受理调用后，再开始新查询必须重新完整同步。

`ProviderClient` 接通文档版本、ACL、交付计划、重新分享、HPKE、stage 与原子 publish。`stage_prepared_block` 的输入是本地 Token/mask 和已完成准备的 D 单方 `PreparationShare`；发给服务的是独立 S 份额/差值和 U 密文，不上传 Token 明文。服务必须已有相同 preparation_id 的保留 S 份额。客户端没有创建完成准备、直接导入 S 准备或绕过发布完整性的 HTTP 入口。

使用方式（身份、证书和配置哈希由部署提供；首次注册与每次主动轮换才调用 register_new_key）：

```python
from pathlib import Path
from private_rag.client import TLSFiles, UserClient

tls = TLSFiles(Path("U.pem"), Path("U.key"), Path("ca.pem"), server_certificate_sha256)
with UserClient("https://localhost:8443", tls, user_uuid, model_numeric_sha256) as user:
    user.register_new_key()  # 首次配置或显式轮换；不要每次启动都轮换
    collection = user.synchronize()
    accepted = user.accept_query()
    # 若 accepted，则把 collection 的 U 单方份额交给独立 compute 接口。
    # 本管理模块不会发送问题原文，也不会重构 z/命中/证据。
```

`tools/check_mtls_management.py` 已实际运行该客户端，经真实 localhost TLS1.3 连接独立 PostgreSQL：OS keystore 注册、D 创建两个块及授权、逐用户重新分享/HPKE/交付/发布、U 连续两次全量下载四个包（N=2）、撤权后完整空集合、无后端受理固定错误；并保留原先无证书/错误 CA/身份头伪造拒绝核对。准备份额由脚本的公开合成入口驱动内部生命周期，`private_computation_executed=false`，没有执行真实 Embedding。运行记录在 `records/management_mtls_debug.json`。新建随机账号对应的 keystore 条目已经删除并确认不存在；临时监听与 PostgreSQL 已正常停止。

后续接线：application.py 已把 collection/本地问题输入连接 compute_process.run_worker，并仅在 U 最终释放后本地解码；ProviderApplication 连接选定本地块、准备受理、C worker 和 D 成功份额 AES256GCM 留存，失败只留在 D。ProviderClient.start_preparation 只发 document/version/block 公共元数据，不接收 Token。scheduler.py 与 CLI compute-once 接起 S 的三连接/快照/事件与完成事务，详细调用及实际边界见 compute_process.md。

新增应用核对实际执行绑定 tokenizer、本地 N0 问题和真实 mTLS 全同步/固定提示。N>0 应用再完整下载 N=2 的两个包后关闭门拒绝，因此本轮脚本总计下载 6 包；D 准备应用在门禁前未创建加密文件、受理会话或 worker。仍没有持久化 U 已解密缓存，服务真实后端激活保持关闭。管理、局部应用和传输成功均不等于 R2 模型推理或系统验收成功。

取消采用 `cancel_session(session_uuid)`，真实端点为本人 mTLS 认证的 `POST /v1/sessions/{session_uuid}/cancel`，无理由或秘密失败字段。D 准备和 U 查询均在提交前生成 UUID；响应丢失或公开异常时，只尝试一次与该 UUID 绑定的取消，不重放受理。取消先于受理到达时，数据库保存仅该账户有效的取消意图，后到受理不得再创建该会话；其他账户的意图不能阻止本人的新 UUID。应用已受理后的建连、进程、公开完成等待读失败或 KeyboardInterrupt 同样触发一次取消。HTTP 取消的 5 秒网络超时只限制该次管理请求，不构成计算总时限。取消请求也失联时保留原错误，S 依靠已有公开连接故障/心跳或操作员恢复，不虚构取消已成功。

`tools/check_cancellation_mtls.py` 已通过实际 TLS1.3/独立 PG 核对 D/U 本人取消、跨账户拒绝、关闭门后只有一次受理 POST 和一次取消，以及 claimed 状态不提前释放全局会话。该脚本只插入明确公开的内部生命周期样本，不启动或替换模型后端；结果为 `records/cancellation_mtls_debug.json`，临时监听和 PG 均正常关闭。
