# RTX 2060 UVM 最小修复脚本

脚本：`tools/repair_cuda_uvm.sh`。本次只准备与静态审查，未执行 sudo、未停止服务、未卸载模块，也未运行模型。已核对 `bash -n` 和三段内嵌 Python 的 AST 语法。

当前依据：RTX 2060 的 `nvidia-smi` 正常，内核驱动与用户态库版本匹配；`cuInit(0)` 返回 999，strace 显示打开 `/dev/nvidia-uvm` 返回 EIO。只读观察中 UVM 持有者是桌面用户 asus 的 KDE 硬件监视服务所启动的 `nvidia-smi dmon -d 2 -s pucm`。脚本执行时会重新检查全部持有者，不使用这些 PID 作为固定操作目标。

在原来的 asus 桌面终端执行这一条命令，并在 sudo 提示时输入用户密码：

```bash
sudo /bin/bash '/srv/shared/Documents/ChatGPT/密码比赛/implementation_v1/tools/repair_cuda_uvm.sh'
```

需要 sudo 的原因是完整读取其他用户的 UVM 设备持有情况，以及卸载、装载内核模块；当前非交互 sudo 没有此权限。不要通过 root 登录终端或 `sudo -i` 执行，脚本需要核对发起 sudo 的桌面用户。

脚本仅执行以下有界操作：

1. 从 root 的 `/proc` 读取所有进程与线程的 fd，以及进程内存映射，并与 fuser 结果交叉核对。身份不明、扫描不完整、不是当前用户 KDE 监视器的持有者，都会使脚本退出。
2. 只有 UID、服务 cgroup、精确可执行文件和命令行、主进程祖先链全部成立，才临时停止该用户的 `plasma-ksystemstats.service`。检查服务没有覆盖配置或额外停止传播关系；不停止其他应用。停止后再次检查 UVM 没有任何持有者。
3. 普通执行 `rmmod nvidia_uvm`，随后 `modprobe nvidia_uvm`。不加 force。选用 rmmod 是因为本机 modprobe 手册明确说明 `modprobe -r` 还可能删除空闲依赖模块；rmmod 只卸指定的 UVM。内核发现竞态新增引用时会拒绝普通卸载，脚本不会强制绕过。
4. 以原桌面用户调用一次 `cuInit(0)`。不加载模型。退出 trap 恢复原本运行且由脚本停止的 KDE 监视服务，并核对 active/running；若卸载后尚未成功装载，退出路径也会再尝试装载同一个 UVM 模块。

脚本不重置 GPU，不卸载 nvidia/nvidia_drm/nvidia_modeset，不解绑 PCI，不重启系统、桌面或 OBS，不改变 AMD 显示设置。正常执行只会让 KDE 硬件监视数据短暂停顿。UVM 模块重载及 cuInit 仍可能失败；失败后不会扩大操作范围。所有可捕获退出路径均执行恢复；若操作系统拒绝恢复，会明确报告并返回失败。SIGKILL、断电或系统崩溃无法由 shell trap 恢复，执行期间不要强杀脚本。

看到 `cuInit(0) = 0` 且监视服务恢复确认，才说明这次最小 CUDA 初始化检查成功；不代表 PyTorch 模型、R2 密码计算或完整 RAG 已经运行成功。若仍返回 999 或其他错误，保留终端输出，继续诊断，不手动追加其他卸载或重置命令。

当前冻结脚本 SHA-256：`cc1dc10d8686632c617819667ccee63c2f343e2f057243bf6f1d4fe038978d33`。
