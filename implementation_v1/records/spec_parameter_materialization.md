# 已形成非数值参数导出

这是实施工件与源码核对记录，不是正式测试或验收规范。`tools/materialize_spec_parameters.py` 从本地公开工件、真实 tokenizer 与已审读源码导出《参数与接口.yaml》中的模板、模型清单和接口定义。重复运行得到相同字节内容，不执行模型、密码会话、数据库操作或生产开门；没有修改 configuration、activation、数值块或根端工件同步工具。

已补齐两模型各自的 checkpoint 清单、权重哈希、tokenizer 哈希与量化清单，量化文件各 310 个。导出时重新计算 checkpoint 中所有非 safetensors 公开文件的哈希，检查大权重文件长度及量化文件存在性；大权重与量化 payload 哈希引用已形成的清单，本轮没有重新计算所有大文件哈希。不能把这个清单核对说成模型运行成功。

模板使用真实生成 tokenizer 重新编码 4 段公开文本，核对 26 个保留控制字符串及对应 ID、官方角色模板摘要和原生模板二进制摘要。公开 Token 共 96 个，三个私有槽共 640 个，使用容量 736，尾部无效填充 288，总长度 1024。正文和公开段独立分词，固定偏移、换行与空白、dummy ID 0/validity 0、无效 query 的 dummy key 物理索引 0 均已登记。normalize_text 的 UTF-8 strict、单 BOM、换行归一化、NFC、控制串/ID 拒绝顺序按实际实现登记。

已登记原生 PRGCF001 帧的 96 字节头与公开操作决定的 payload，原生心跳为 96+8=104 个应用字节，不含 TLS 记录开销。Python PRG1 是独立的 24+4096+65536=69656 字节辅助格式；Python PHB1 是 32 字节辅助心跳，均未冒充原生 worker 实际格式。原生 3 条 TLS 心跳探针记录已有初始心跳/断连/burn/线程回收结果，但没有计时运行 180 秒 inactivity 分支，完整 R2 仍未执行。

接口登记包含本方密封 memfd、22 个 argv（含程序名）、事件 1/2 及同值 ACK、四种固定输出长度、交付绑定/info/固定 HPKE 载荷、库内错误枚举、公开生命周期及材料账本。Python Descriptor 尚未支持 gc_bitvector/bfv_ciphertext，部分原生算子仍依赖 Word/Bit 的静态调用约定，所以统一所有密码类型的 ABI 保持 partial。恢复条件按实际 lifecycle.py 明确要求旧 worker 和三 lane 均已停止以及独占 owner 锁，锁消失本身不证明可恢复。

运行命令：

```sh
.venv/bin/python tools/materialize_spec_parameters.py
.venv/bin/python ../项目文档/第一版机读规范/校验规范.py
```

再次导出字节不变。包内校验 5 个 YAML、62 条规则、8 条决策、5 个工件，errors 为空。数值流水线、非线性算法和密码后端三个原文块逐字节保持不变。检查结果分别保存在 `spec_parameter_materialization.json` 与 `spec_parameter_materialization_check.json`，后者的 SHA256 仅标识当次导出参数状态；后续根端同步数值工件可以合法改变整个参数文件摘要。

仍为 null 的三处中，rope_scaling 与 active_compute_total_timeout_seconds 是明确关闭选项，只有 fixed_execution_message_and_access_plan.actual 是尚未形成的工件。全局未验证、BFV 控制解码的既有验证状态和完整消息/访问计划未验证状态未伪装成通过；本次不扩大到数值或密码正确性验证。
