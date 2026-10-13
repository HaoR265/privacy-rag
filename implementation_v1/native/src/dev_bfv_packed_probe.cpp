// Bounded public synthetic multi-target BFV diagnostics, never a model run.
#include "prag/bfv.hpp"
#include "prag/secure_ops.hpp"
#include <csignal>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;

static int run(Role role, int fd, const std::string &prefix,
               const std::string &certs) {
  std::array<unsigned char, 16> session{};
  std::array<unsigned char, 32> configuration{};
  session[0] = 105;
  configuration[0] = 5;
  Channel io(fd, session, configuration);
  if (!certs.empty()) {
    const auto own = role == Role::Client ? "U" : "S-compute";
    io.enable_tls(role, certs + "/ca.pem", certs + "/" + own + ".pem",
                  certs + "/" + own + ".key", "localhost");
  }
  SecureOps op(role, io,
               prefix + (role == Role::Client ? ".C.ledger" : ".S.ledger"),
#ifdef PRAG_PRODUCTION_BFV_PACKED_PROBE
               false);
  if (!io.authenticated() || op.isolated_development())
    throw std::runtime_error("E_BFV_PRODUCTION_PROBE_PROFILE");
#else
               true);
#endif
  std::ostringstream cases;
  const size_t dimensions[] = {4096, 2730}, counts[] = {3, 4};
  for (size_t case_id = 0; case_id < 2; ++case_id) {
    const size_t d = dimensions[case_id], rows = counts[case_id];
    std::vector<Word> shares(d);
    std::vector<std::vector<Word>> weights(rows, std::vector<Word>(d));
    std::vector<Word> expected(rows, 0);
    for (size_t j = 0; j < d; ++j) {
      const Word whole = Word(int((j * 13 + 7) % 101) - 50) & MASK;
      const Word client = (UINT64_C(0x1928abcdef) + j * UINT64_C(125939)) & MASK;
      shares[j] = role == Role::Client ? client : (whole - client) & MASK;
      for (size_t row = 0; row < rows; ++row) {
        const Word weight = row == 1 || row == rows - 1
                                ? 0
                                : Word(int((j * (row + 3) + 5) % 17) - 8) & MASK;
        expected[row] = (expected[row] + whole * weight) & MASK;
        if (role == Role::Server)
          weights[row][j] = weight;
      }
    }
    io.set_stage(105 + case_id);
    const auto before = io.bytes();
    auto out = op.private_linear(shares,
        role == Role::Server ? weights : std::vector<std::vector<Word>>{},
        rows, 12, 12);
    const auto bytes = io.bytes() - before;
    // Only synthetic diagnostic output is reconstructed, in this executable.
    io.set_operation(100000 + case_id);
    if (role == Role::Server) {
      io.send_data(out.data(), int(out.size() * sizeof(Word)));
    } else {
      std::vector<Word> other(rows);
      io.recv_data(other.data(), int(other.size() * sizeof(Word)));
      for (size_t row = 0; row < rows; ++row)
        out[row] = op.add(out[row], other[row]);
      if (out != expected)
        throw std::runtime_error("E_BFV_PACKED_SYNTHETIC_MISMATCH");
      if (case_id)
        cases << ',';
      const size_t capacity = packed_bfv_rows_per_ciphertext(d);
      cases << "{\"columns\":" << d << ",\"rows\":" << rows
            << ",\"capacity\":" << capacity
            << ",\"returned_ciphertext_groups\":" << (rows + capacity - 1) / capacity
            << ",\"singleton_last_group\":true,\"zero_and_nonzero_rows\":\"matched\","
               "\"all_zero_last_group\":\"matched\",\"application_bytes_before_reconstruction\":"
            << bytes << '}';
    }
  }
  op.close_session();
  if (role == Role::Client)
    std::cout << "{\"entry\":\"bounded_multi_target_BFV_probe\","
                 "\"native_version\":\"" PRAG_NATIVE_VERSION "\","
                 "\"two_processes\":true,\"raw_fractional_bits\":24,"
                 "\"cases\":[" << cases.str()
              << "],\"native_TLS13_mTLS\":" << (io.authenticated() ? "true" : "false")
              << ",\"isolated_development\":" << (op.isolated_development() ? "true" : "false")
              << ",\"client_application_bytes\":" << io.bytes()
              << ",\"full_R2\":false,\"GPU_used\":false,\"timing_comparison\":false}\n";
  return 0;
}
int main(int argc, char **argv) {
  std::signal(SIGPIPE, SIG_IGN);
  if (argc < 2 || argc > 3)
    return 2;
#ifdef PRAG_PRODUCTION_BFV_PACKED_PROBE
  if (argc != 3)
    return 2;
#endif
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
      result = run(Role::Server, sockets[1], argv[1], argc == 3 ? argv[2] : "");
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
    }
    _exit(result);
  }
  close(sockets[1]);
  try {
    result = run(Role::Client, sockets[0], argv[1], argc == 3 ? argv[2] : "");
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
  int status = 0;
  waitpid(child, &status, 0);
  return result || !WIFEXITED(status) || WEXITSTATUS(status);
}
