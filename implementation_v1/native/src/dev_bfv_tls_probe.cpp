// Public synthetic diagnostic: two BFV outputs with the production library.
#include "prag/secure_ops.hpp"
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;

static int run(Role role, int fd, const std::string &prefix,
               const std::string &certs) {
  std::array<unsigned char, 16> session{};
  std::array<unsigned char, 32> configuration{};
  session[0] = 104;
  configuration[0] = 61;
  Channel io(fd, session, configuration);
  const auto own = role == Role::Client ? "U" : "S-compute";
  io.enable_tls(role, certs + "/ca.pem", certs + "/" + own + ".pem",
                certs + "/" + own + ".key", "localhost");
  SecureOps op(role, io,
               prefix + (role == Role::Client ? ".C.ledger" : ".S.ledger"),
               false);
  if (!io.authenticated() || op.isolated_development())
    throw std::runtime_error("E_BFV_PRODUCTION_PROBE_PROFILE");
  const Word x = Word(-12345) & MASK, xc = 125934, y = 79, yc = 98867;
  const std::vector<Word> shares{
      role == Role::Client ? xc : (x - xc) & MASK,
      role == Role::Client ? yc : (y - yc) & MASK, op.public_word(12)};
  const std::vector<std::vector<Word>> weights =
      role == Role::Server
          ? std::vector<std::vector<Word>>{{2, Word(-3) & MASK, 5}, {0, 0, 0}}
          : std::vector<std::vector<Word>>{};
  io.set_stage(104);
  io.set_operation(1);
  auto output = op.private_linear(shares, weights, 2, 12, 12);
  io.set_operation(2);
  if (role == Role::Server) {
    io.send_data(output.data(), int(output.size() * sizeof(Word)));
    op.close_session();
    return 0;
  }
  // Reconstruct only the public synthetic values in this diagnostic executable.
  std::vector<Word> other(output.size());
  io.recv_data(other.data(), int(other.size() * sizeof(Word)));
  for (size_t j = 0; j < output.size(); ++j)
    output[j] = op.add(output[j], other[j]);
  if (output != std::vector<Word>{(2 * x - 3 * y + 60) & MASK, 0})
    throw std::runtime_error("E_BFV_PRODUCTION_PROBE_MISMATCH");
  op.close_session();
  std::cout << "{\"entry\":\"production_BFV_decode60_synthetic_probe\","
               "\"library\":\"prag_crypto\",\"native_version\":\"" PRAG_NATIVE_VERSION "\","
               "\"two_processes\":true,\"native_TLS13_mTLS\":true,"
               "\"isolated_development\":false,\"real_BFV_outputs\":2,"
               "\"public_columns\":3,\"nonzero_and_zero_weights\":\"matched\","
               "\"transport\":\"TLS13_over_local_AF_UNIX\",\"client_application_bytes\":"
            << io.bytes()
            << ",\"full_R2\":false,\"timing_comparison\":false}\n";
  return 0;
}

int main(int argc, char **argv) {
  std::signal(SIGPIPE, SIG_IGN);
  if (argc != 3)
    return 2;
  int sockets[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets))
    return 3;
  const pid_t child = fork();
  if (child < 0)
    return 3;
  int result = 1;
  if (child == 0) {
    close(sockets[0]);
    try {
      result = run(Role::Server, sockets[1], argv[1], argv[2]);
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
    }
    _exit(result);
  }
  close(sockets[1]);
  try {
    result = run(Role::Client, sockets[0], argv[1], argv[2]);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
  int status = 0;
  waitpid(child, &status, 0);
  return result || !WIFEXITED(status) || WEXITSTATUS(status);
}
