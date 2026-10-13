# 资料交付与管理实现记录

本记录为 DEC-010 范围内实施者决定；不将实现细节写成用户逐项确认。涉及 CRY-013/014、DAT-001—006、RUN-001/004/007/008 及 CODE-P01—08；未改变模型、数值尺度、信任假设或预算。以下核对是实现调试，不是新测试或验收规范。

实现前仅有机读逻辑契约。现有 `delivery.py` 实现 D 的逐坐标独立重新分享、S 的 delta 适配、固定 1537 字交付、cryptography 49.0.0 的真实 HPKE Base 加密和 U 本地认证打开。没有重构接口。独立合成调试脚本的重构只针对脚本本身生成的公开数据。

选择官方 `Suite.encrypt/decrypt`，X25519/HKDF-SHA256/AES-128-GCM。该 API 的辅助认证信息由 `info` 传递，没有独立 AAD 参数；info 是固定域分隔加所有绑定字段的规范 JSON，AEAD AAD 为空。依据官方 https://cryptography.io/en/49.0.0/hazmat/primitives/hpke/ 。实际包固定 12344 字节，载荷是 12296 字节完整宽度小端余数。具体字节布局登记在 `artifacts/interface_delivery.json`。

PostgreSQL 负责账户、接收钥匙代次、ACL、资料版本、逐用户槽位、发布指针、不可变快照、全局单会话及一次性材料账本。大份额/密包写不可变 SHA256 对象，先 fsync 再在数据库单事务切换发布引用。新版本缺块、旧接收钥匙或失配授权代次均拒绝；保留上个完整版本。用户全量清单只含自身不透明槽位、允许版本和包摘要；U 的缓存必须所有包完成认证后才切换为新集合，生产 UserClient 每次重新取得完整集合，不走增量 endpoint。

管理 FastAPI 不读取问题/答案明文，不提供原文、向量、命中或 S 份额下载。认证只接受真实 TLS ingress 放入 scope 的证书 DER 和一致 SHA256，并查已登记账户；HTTP 用户名或证书头没有作用。生产创建 app 未接有效计算 activation_check 时，N>0 查询受理固定拒绝；N=0 返回规定提示。正式受理再次核对 ACL、清单、钥匙代次、所有对象及模型数值配置。准备产物只能通过内部计算协调器已完成会话留存，没有 public HTTP 准备伪造入口。

`vault.py` 使用 AES-256-GCM 和严格认可的系统密钥存储后端，不回退到文件密码。D 的保留对象通过 provider/key/object/version AAD 绑定；U 的 X25519 长期接收密钥使用独立服务名。没有把 Python 释放引用称为物理安全擦除。身份私钥和 BFV 会话密钥不在这里混用。

`Lifecycle` 只有进入最终释放并有完整确认才能标记 completed，释放中断一律 aborted；未确定消费的材料 burn。恢复函数必须由持有计算进程独占所有权的协调器在确认旧进程已死后调用，不从普通重连请求调用。活跃状态在 PostgreSQL 用唯一索引约束为全局一个；计算不持长数据库事务，不设总超时。

实际运行：`PYTHONPATH=. .venv/bin/python tools/check_management.py`。该入口自行新建独立临时 PostgreSQL 18.3 集群，仅 Unix socket，不接触用户现有数据库。公开合成数据驱动交付、发布、同步、快照及生命周期；没有执行真实 Embedding 或 R2。已核对 HPKE 篡改和版本绑定拒绝、份额配对、原子发布失败保旧、撤权后旧快照保持/新会话排除、重新授权新代次、换钥等待重新分享、并发 busy、材料消费/作废、HTTP header 不能伪造身份及错误响应不回显输入。SecretService 中本任务临时新条目的 AES-256-GCM 读写和错误 AAD 拒绝也实际通过；该条目已删除，临时数据库正常关闭。机器结果见 `management_debug.json`。

尚未证明：上述管理调试不是完整真实私有计算，不证明来源的恶意安全或模型质量。新增原生计算协调器按最终 ACK 后的 S 本方输出保存准备份额，并把登记和会话完成放在同一事务；只有 D 得知数值失败，D 本地失败时不创建可交付份额，不再另向 S 发送成功/失败消息。FastAPI 真实 TLS 监听由 tls_server.py 集成；本记录早期 TestClient scope 注入仅核对接口，不冒充真实 mTLS，独立真实 mTLS 结果另列。不可变旧对象暂不自动删除，快照结束后只是释放数据库引用。

接线现状：application.py、compute_process.py、scheduler.py 已连接本方应用、原生进程 ABI、三 lane 与 Repository/Lifecycle。生产门禁仍关闭，没有完整模型运行依据。准备可独立重做同一块，新的 session UUID 同时是双方 preparation_id，不复用原 session 材料；publish 用 DISTINCT block_index 检查完整覆盖。未知 total_bytes 记 NULL。不得通过 HTTP 或通用上传绕过准备状态，不得用公开合成调试数据作为生产准备结果。续做入口和 CLI 示例见 compute_process.md。

后续增量：另以 `tools/check_mtls_management.py` 实际启动本机独立 HTTPS listener、内部 Ed25519 CA/D/U 身份与临时 PostgreSQL，协商 TLSv1.3。U 公钥登记、N0 全同步/提示、D 元数据注册和角色隔离均通过；缺客户端证书及不同 CA 证书在 TLS 层被拒绝，HTTP 用户头不能改变证书身份。listener、临时密钥目录与数据库均正常收尾，结果见 `management_mtls_debug.json`。这仍仅为管理与传输实运行，未启动模型。并修复受理的两事务 N 检查窗口：最新版 ACL/完整清单/activation_check 在同一事务锁内核对，之后才插入活跃快照。
