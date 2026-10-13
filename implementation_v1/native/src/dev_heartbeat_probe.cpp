// Isolated public-data lifecycle diagnostic; no inference or model inputs.
#include "prag/heartbeat.hpp"
#include "prag/secure_ops.hpp"
#include <csignal>
#include <iostream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
static int run(Role role, int fds[3], const std::string &prefix,
               const std::string &pki) {
  std::array<unsigned char, 16> session{};
  std::array<unsigned char, 32> configuration{};
  session[0] = 13;
  configuration[0] = 3;
  Channel compute(fds[0], session, configuration),
      first(fds[1], session, configuration),
      second(fds[2], session, configuration);
  const std::string own = role == Role::Client ? "U" : "S-compute";
  for (auto *c : {&compute, &first, &second})
    c->enable_tls(role, pki + "/ca.pem", pki + "/" + own + ".pem",
                  pki + "/" + own + ".key", "localhost");
  SecureOps op(role, compute,
               prefix + (role == Role::Client ? ".C.ledger" : ".S.ledger"));
  Channel &out = role == Role::Client ? first : second;
  Channel &in = role == Role::Client ? second : first;
  Heartbeat hb(out, in, compute, [&op] { op.abort_session(); });
  // This diagnostic timeout bounds the executable only. The Heartbeat class
  // itself never has a total-session timeout and retains fixed 30/180 seconds.
  auto diagnostic_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!hb.received() &&
         std::chrono::steady_clock::now() < diagnostic_deadline) {
    hb.check();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!hb.received())
    throw std::runtime_error("E_DIAGNOSTIC_HEARTBEAT_START");
  unsigned char ready = 1;
  if (role == Role::Server) {
    compute.recv_data(&ready, 1);
    compute.send_data(&ready, 1);
    out.abort(); // Deliberate public disconnection, then burn locally as well.
    op.abort_session();
    hb.close();
    return 0;
  }
  compute.send_data(&ready, 1);
  compute.recv_data(&ready, 1);
  while (!hb.failed() && std::chrono::steady_clock::now() < diagnostic_deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  hb.close(); // Join callback and all I/O threads before accessing lifecycle.
  if (!hb.failed() || op.session_active())
    throw std::runtime_error("E_DIAGNOSTIC_BURN");
  bool rejected = false;
  try {
    op.reserve_material("must_not_run", 1);
  } catch (const std::exception &) {
    rejected = true;
  }
  if (!rejected)
    throw std::runtime_error("E_DIAGNOSTIC_REUSE");
  std::cout << "{\"entry\":\"isolated_native_heartbeat_probe\",\"TLS13_mTLS_"
               "connections\":3,"
               "\"initial_heartbeats\":\"received_both_directions\","
               "\"disconnect_burn\":\"matched\","
               "\"reuse_after_abort\":\"rejected\",\"threads_joined\":true,"
               "\"interval_seconds\":30,"
               "\"inactivity_seconds\":180,\"elapsed_inactivity_branch_run\":"
               "false,\"full_system\":false}\n";
  return 0;
}
int main(int argc, char **argv) {
  std::signal(SIGPIPE, SIG_IGN);
  if (argc != 3)
    return 2;
  int pairs[3][2];
  for (auto &p : pairs)
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, p))
      return 3;
  pid_t child = fork();
  if (child < 0)
    return 3;
  int status = 1;
  try {
    int fds[3];
    for (int j = 0; j < 3; ++j) {
      close(pairs[j][child == 0 ? 0 : 1]);
      fds[j] = pairs[j][child == 0 ? 1 : 0];
    }
    status =
        run(child == 0 ? Role::Server : Role::Client, fds, argv[1], argv[2]);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
  if (child == 0)
    _exit(status);
  int state = 0;
  waitpid(child, &state, 0);
  return status || !WIFEXITED(state) || WEXITSTATUS(state);
}
