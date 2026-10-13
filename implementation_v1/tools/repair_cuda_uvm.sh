#!/bin/bash
# Only nvidia_uvm is reloaded. No GPU reset, core-driver removal, or PCI change.
# Run once with sudo from the logged-in desktop account. Never run via sudo -i.
set -Eeuo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin LC_ALL=C
unset LD_PRELOAD LD_LIBRARY_PATH PYTHONPATH PYTHONHOME MODPROBE_OPTIONS
umask 077

die() { printf 'UVM 修复停止：%s\n' "$*" >&2; exit 1; }
log() { printf '%s\n' "$*"; }
[[ $# == 0 ]] || die '此脚本不接受参数。'
[[ $EUID == 0 ]] || die '请由当前桌面用户用 sudo 执行；需要完整检查进程并重新装载 UVM 内核模块。'
[[ ${SUDO_UID:-} =~ ^[0-9]+$ && ${SUDO_UID} != 0 && -n ${SUDO_USER:-} ]] || die '无法确认发起 sudo 的非 root 桌面用户。'
for cmd in /usr/bin/python3 /usr/bin/fuser /usr/sbin/runuser /usr/bin/systemctl /usr/bin/timeout /usr/sbin/rmmod /usr/sbin/modprobe; do
    [[ -x $cmd ]] || die "缺少系统工具：$cmd"
done

owner_uid=$SUDO_UID
owner_name=$SUDO_USER
owner_home=$(/usr/bin/python3 -I - "$owner_uid" "$owner_name" <<'PY'
import pwd, sys
p = pwd.getpwuid(int(sys.argv[1]))
if p.pw_name != sys.argv[2] or not p.pw_dir.startswith('/'):
    raise SystemExit('sudo 用户与系统账户记录不一致')
print(p.pw_dir)
PY
) || die '无法核对桌面用户。'
[[ -S /run/user/$owner_uid/bus ]] || die '当前用户的 systemd/DBus 会话不可用。'
unit=plasma-ksystemstats.service
userctl() { /usr/bin/timeout 20s /usr/sbin/runuser -u "$owner_name" -- /usr/bin/env -i \
    HOME="$owner_home" USER="$owner_name" LOGNAME="$owner_name" \
    PATH="$PATH" LC_ALL=C XDG_RUNTIME_DIR="/run/user/$owner_uid" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$owner_uid/bus" \
    /usr/bin/systemctl --user "$@"; }
property() { userctl show "$unit" -p "$1" --value; }

restore_monitor=0
restore_module=0
cleanup() {
    local rc=$?
    trap - EXIT
    trap '' HUP INT TERM
    set +e
    if (( restore_module )); then
        log '退出前尝试恢复 nvidia_uvm 的装载状态。'
        if ! /usr/sbin/modprobe nvidia_uvm; then
            printf '恢复失败：nvidia_uvm 尚未成功装载。未尝试其他驱动操作。\n' >&2
            rc=1
        fi
    fi
    if (( restore_monitor )); then
        log "恢复此前运行的 $unit。"
        if ! userctl start "$unit"; then
            printf '恢复失败：请在原桌面用户终端执行 systemctl --user start plasma-ksystemstats.service。\n' >&2
            rc=1
        elif [[ $(property ActiveState) != active || $(property SubState) != running ]]; then
            printf '恢复未确认：KDE 监视服务尚未回到 active/running；请保留此输出供诊断。\n' >&2
            rc=1
        else
            log 'KDE 监视服务已恢复为 active/running。'
        fi
    fi
    (( rc == 0 )) && log 'UVM 最小修复完成；cuInit(0) 成功。这不代表模型或完整系统已运行。'
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -d /sys/module/nvidia_uvm ]] || die 'nvidia_uvm 当前未装载；与本脚本针对的故障不符。'
[[ -d /sys/module/nvidia && -d /sys/module/nvidia_modeset && -d /sys/module/nvidia_drm ]] || die '当前图形驱动状态与预期不符；拒绝操作。'
initial_active=$(property ActiveState) || die '无法读取用户服务状态。'
initial_sub=$(property SubState) || die '无法读取用户服务子状态。'
main_pid=$(property MainPID) || die '无法读取 KDE 监视服务主进程。'
service_cgroup=$(property ControlGroup) || die '无法读取 KDE 监视服务 cgroup。'
fragment=$(property FragmentPath) || die '无法读取 KDE 监视服务来源。'
dropins=$(property DropInPaths) || die '无法读取用户服务覆盖配置。'
[[ $fragment == /usr/lib/systemd/user/plasma-ksystemstats.service && -z $dropins ]] || die '监视服务不是预期的无覆盖 KDE 系统单元；拒绝停止。'
for dep in PropagatesStopTo ConsistsOf BoundBy; do
    value=$(property "$dep") || die '无法检查服务停止传播关系。'
    [[ -z $value ]] || die "服务存在额外停止传播关系 $dep；拒绝停止其他应用。"
done

# stdout contains only approved monitor PIDs. Unknown or unreadable holders fail.
# Inspect all process/thread fd tables plus memory mappings; cross-check fuser.
scan_uvm_holders() {
    /usr/bin/python3 -I - "$1" "$owner_uid" "$main_pid" "$service_cgroup" <<'PY'
import os, pathlib, stat, subprocess, sys

mode, owner, main, cg = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
proc = pathlib.Path('/proc')
nodes = [pathlib.Path('/dev/nvidia-uvm')]
if pathlib.Path('/dev/nvidia-uvm-tools').exists():
    nodes.append(pathlib.Path('/dev/nvidia-uvm-tools'))
rdevs, mapped_files = set(), set()
try:
    majors = [int(line.split()[0]) for line in (proc / 'devices').read_text().splitlines()
              if len(line.split()) == 2 and line.split()[1] == 'nvidia-uvm']
    if len(majors) != 1:
        raise RuntimeError('无法唯一确认 UVM 设备主编号')
    for index, node in enumerate(nodes):
        s = node.stat()
        if not stat.S_ISCHR(s.st_mode) or os.major(s.st_rdev) != majors[0] or os.minor(s.st_rdev) != index:
            raise RuntimeError(f'UVM 设备节点与内核注册不符：{node}')
        rdevs.add(s.st_rdev)
        mapped_files.add((os.major(s.st_dev), os.minor(s.st_dev), s.st_ino))
    holders = set()
    for entry in proc.iterdir():
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            tasks = list((entry / 'task').iterdir())
            for task in tasks:
                try:
                    fds = list((task / 'fd').iterdir())
                except FileNotFoundError:
                    continue  # This thread has exited.
                for fd in fds:
                    try:
                        s = fd.stat()
                    except FileNotFoundError:
                        continue  # The descriptor was closed during enumeration.
                    if stat.S_ISCHR(s.st_mode) and s.st_rdev in rdevs:
                        holders.add(pid)
            # A mapped device can remain held after its original fd is closed.
            for line in (entry / 'maps').read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) >= 5:
                    a, b = fields[3].split(':')
                    if (int(a, 16), int(b, 16), int(fields[4])) in mapped_files:
                        holders.add(pid)
        except (FileNotFoundError, ProcessLookupError):
            if entry.exists():
                raise RuntimeError(f'PID {pid} 仍存在但无法完成检查')
        except PermissionError:
            if entry.exists():
                raise RuntimeError(f'无法完整读取 PID {pid}，拒绝继续')
    check = subprocess.run(['/usr/bin/fuser', *map(str, nodes)], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    if check.returncode not in (0, 1):
        raise RuntimeError('fuser 交叉检查失败')
    # With LC_ALL=C, fuser puts only decimal PIDs on stdout, and path/access
    # annotations on stderr. Treat any unexpected stdout as a failed scan.
    for item in check.stdout.split():
        if not item.isdecimal():
            raise RuntimeError('fuser 返回了无法识别的持有者')
        holders.add(int(item))
    if 'denied' in check.stderr.lower() or 'cannot' in check.stderr.lower():
        raise RuntimeError('fuser 未能完成检查')

    def identity(pid):
        base = proc / str(pid)
        raw = (base / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        # After comm: state, ppid, ... starttime (field 22).
        result = {'ppid': int(fields[1]), 'start': fields[19]}
        uid_line = next(s for s in (base / 'status').read_text().splitlines() if s.startswith('Uid:'))
        result['uids'] = list(map(int, uid_line.split()[1:]))
        result['exe'] = os.readlink(base / 'exe')
        result['cmd'] = (base / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        result['cg'] = (base / 'cgroup').read_text().strip()
        return result

    def approved(pid):
        if mode != 'monitor' or main <= 1:
            return False
        prefix = f'/user.slice/user-{owner}.slice/user@{owner}.service/'
        if not cg.startswith(prefix) or not cg.endswith('/plasma-ksystemstats.service'):
            return False
        leader, candidate = identity(main), identity(pid)
        if leader['uids'] != [owner] * 4 or leader['exe'] != '/usr/bin/ksystemstats':
            return False
        if leader['cmd'] != [b'/usr/bin/ksystemstats'] or leader['cg'] != '0::' + cg:
            return False
        if candidate['uids'] != [owner] * 4 or candidate['cg'] != '0::' + cg:
            return False
        if pid != main:
            if candidate['exe'] != os.path.realpath('/usr/bin/nvidia-smi'):
                return False
            if candidate['cmd'] != [b'/usr/bin/nvidia-smi', b'dmon', b'-d', b'2', b'-s', b'pucm']:
                return False
            cursor, seen = pid, set()
            while cursor != main:
                if cursor <= 1 or cursor in seen:
                    return False
                seen.add(cursor)
                ancestor = identity(cursor)
                if ancestor['uids'] != [owner] * 4 or ancestor['cg'] != '0::' + cg:
                    return False
                cursor = ancestor['ppid']
        # Refuse PID reuse during validation.
        return identity(pid)['start'] == candidate['start'] and identity(main)['start'] == leader['start']

    remaining = []
    for pid in sorted(holders):
        if not (proc / str(pid)).exists():
            continue
        try:
            if not approved(pid):
                raise RuntimeError(f'存在未获准停止的 UVM 持有者 PID {pid}；未停止任何其他应用')
        except (FileNotFoundError, ProcessLookupError):
            if (proc / str(pid)).exists():
                raise RuntimeError(f'PID {pid} 的身份在检查期间发生变化')
            continue
        remaining.append(pid)
    print(' '.join(map(str, remaining)))
except Exception as exc:
    print(f'UVM 持有者检查失败：{exc}', file=sys.stderr)
    raise SystemExit(1)
PY
}

log '完整检查 UVM 持有者（包括其他用户进程）；不会结束未知应用。'
holders=$(scan_uvm_holders monitor) || die '持有者不满足最小修复条件。'
if [[ -n $holders ]]; then
    [[ $initial_active == active && $initial_sub == running ]] || die 'KDE 监视服务不是稳定运行状态。'
    [[ $(property MainPID) == "$main_pid" && $(property ControlGroup) == "$service_cgroup" ]] || die '服务身份已变化。'
    # Set before issuing stop so even a partially completed stop is restored.
    restore_monitor=1
    log "仅临时停止 $owner_name 的 $unit（已核对持有者 PID：$holders）。"
    userctl stop "$unit" || die '停止 KDE 监视服务未完成；退出并恢复。'
    [[ $(property ActiveState) == inactive ]] || die '监视服务没有停止或已被重新激活。'
fi
scan_uvm_holders empty >/dev/null || die '再次检查仍存在 UVM 持有者；不卸载模块。'

log '仅普通卸载 nvidia_uvm，再装载同一模块；不使用 force。'
# Unlike modprobe -r, rmmod cannot cascade to now-unused dependency modules.
# The kernel refuses removal if a racing/new holder still references UVM.
restore_module=1
/usr/sbin/rmmod nvidia_uvm || die 'UVM 普通卸载失败；不强制，不更改其他驱动。'
/usr/sbin/modprobe nvidia_uvm || die 'UVM 重新装载失败；退出时再尝试恢复装载状态。'
restore_module=0
[[ -d /sys/module/nvidia_uvm ]] || die '未能确认 UVM 已重新装载。'

log '以原桌面用户执行一次 cuInit(0)，不加载模型、不进行 GPU 重置。'
if /usr/bin/timeout 20s /usr/sbin/runuser -u "$owner_name" -- /usr/bin/env -i \
    HOME="$owner_home" USER="$owner_name" LOGNAME="$owner_name" PATH="$PATH" LC_ALL=C \
    /usr/bin/python3 -I - <<'PY'
import ctypes
cuda = ctypes.CDLL('libcuda.so.1')
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int
code = cuda.cuInit(0)
print(f'cuInit(0) = {code}', flush=True)
raise SystemExit(0 if code == 0 else 1)
PY
then
    :
else
    die 'cuInit 仍未成功；保留结果供诊断，不扩大修复范围。'
fi
