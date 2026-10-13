# 本方应用泄漏与错误分支审查

本轮为代码审查和必要调试，不是完整信息流证明或新验收标准。范围 application.py、client.py、compute_process.py、scheduler.py 与数据库完成状态。

1. 发现并修复：D 准备初稿在秘密 client_failed=true 时直接本地返回，只有成功时查询 S 会话完成状态。S 可从 GET 请求有无推断失败位，违反“仅 D 获失败位”。现所有有效固定输出均先执行同一公开会话完成等待，直到 S 的 completed+release_acknowledged，再仅在 D 分支保留成功份额或返回失败。秘密位不进入 HTTP body/path/header、S 状态、S 份额登记或日志。成功后的显式 stage/publish 仍由已授权调用者决定，应用不自动发布。

2. U 的 N=0 分支依赖公开全授权块数，直接规定提示；N>0 的证据不足/数值失败/长度上限只在固定原生最终释放之后向 U 解释，后续没有管理 HTTP。问题正文只在 U tokenizer 和本方 memfd；D 原文/正文/Embedding Token 只在 D tokenizer、AES256GCM vault 和本方 memfd。管理只获得约定元数据、独立 S 份额/差值及 U 的 HPKE 包。

3. S 原生生命周期仅在公开 AUTHENTICATED/RELEASING 事件及其成功数据库提交后 ACK；不基于 D 的秘密失败改变状态。S 准备统一存自己的固定 1025 个环元素，并原子登记 completed；其结果结构没有 client_failed。D 和 S 的 preparation_id 都是会话 UUID。D 等待公开完成，避免 stage 与 S 原子登记竞态；若这一步网络失败，源块仍加密留存，但不能声称本次准备份额已保留或交付。

4. completed 状态的 release_acknowledged 必须 true、public_fault 必须 null、ended_at 必须存在；aborted 与 active 也核对对应状态字段。仅账户本人能读取公开会话状态。未知通信字节为 null；没有计算总时限、失败重跑或写请求自动重放。public TCP admission 时限和原生固定心跳独立。

5. Worker stdout/stderr 不进入日志，秘密结果 dataclass 禁用 repr。管理响应固定错误码，不回显验证输入；本地 InputRejected 使用固定原因，没有原文。依赖秘密的本地失败/答案分支位于已允许最终释放之后。原生收到的份额界检查只检查环余数编码，不把本方随机份额当成完整模型数值。

6. 公开取消上下文只覆盖受理后的绑定、建连、固定输出读取和公开完成等待；U 最终答案/数值失败解释以及 D 的 client_failed 判断与成功份额加密留存位于该上下文外。因此秘密失败本身不触发 HTTP cancel，也不改变成功/失败均执行的 D 公开完成 GET 路径。受理响应不确定时，客户端用预生成随机 UUID 发一次本人取消；服务原子串行化 claim/cancel/completed。claimed 后取消仅记 cancel_requested，待本方子进程及三 lane 确认回收后才终止会话并释放全局占用。

7. 原生事件 pipe 提前 EOF、子进程非零退出、异常输出长度均立即固定错误，不等候永远不会到达的 ACK；进程 pipe 仍并行排空。协调器异常先关闭 admission 连接/监听，再终止状态；worker 异常先终止并 wait 本方子进程，再关闭 owned fd，之后协调器才能释放状态。CLI 正常 SIGTERM 转换成公开取消；SIGKILL、断电、数据库不可达与无法确认旧子进程终止仍需操作员恢复，不能因 PostgreSQL owner lock 消失就断言旧计算已停止。旧 schema 升级遇到活跃会话保守标记 claimed，避免默认 false 允许过早释放。

实际边界：真实管理 mTLS/全同步、N0 应用、本地 tokenizer、关闭门禁、PostgreSQL 公开合成生命周期、HPKE 和 OS vault 已有独立运行记录。数值/BFV 工件更新期间，完整实际文件哈希核对曾正确触发 VERSION_MISMATCH；现统一配置 d023e3914e9a1bd6c6ce3576449fab9d00761ec6497c44ed3c423b7136596f21 形成后，已重跑完整管理 mTLS 与进程边界核对并通过，两份 JSON 记录均保存同一配置摘要。没有绕过 gate 或 mock 原生后端来执行秘密失败分支；上述分支等形依据是代码审查，尚不等于全模型 R2 实运行或协议安全证明。

取消专项实际核对为 `cancellation_debug.json` 与 `cancellation_mtls_debug.json`：真实独立 PG 中含旧库升级、先取消/先 claim/完成取消竞争；真实 pipe 子进程只使用公开事件和固定公开载荷，覆盖 8201 字节输出、提前 EOF、非零退出和运行中公开取消，无伪模型或生产门绕过。所有样本最终无活跃会话，子进程、临时监听和 PG 已回收。
