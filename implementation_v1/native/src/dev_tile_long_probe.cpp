// Public synthetic diagnostics only: no model and no production reconstruction.
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
               const std::string &certs, bool failure_probe) {
  std::array<unsigned char, 16> session{};
  std::array<unsigned char, 32> configuration{};
  session[0] = 108;
  configuration[0] = 6;
  Channel io(fd, session, configuration);
  if (!certs.empty()) {
    const auto own = role == Role::Client ? "U" : "S-compute";
    io.enable_tls(role, certs + "/ca.pem", certs + "/" + own + ".pem",
                  certs + "/" + own + ".key", "localhost");
  }
  SecureOps op(role, io,
               prefix + (role == Role::Client ? ".C.ledger" : ".S.ledger"),
#ifdef PRAG_PRODUCTION_BFV_MERGED_PROBE
               false);
  if (!io.authenticated() || op.isolated_development() || failure_probe)
    throw std::runtime_error("E_BFV_PRODUCTION_PROBE_PROFILE");
#else
               true);
#endif
  if (failure_probe) {
    bool caught = false;
    try {
      op.private_linear_streamed({123}, 1, 12, 12,
          [](size_t, size_t, size_t, size_t) {
            return std::vector<std::vector<Word>>{}; // deliberate wrong shape
          });
    } catch (const std::runtime_error &e) {
      caught = role == Role::Client ||
          std::string(e.what()) == "E_BFV_WEIGHT_BLOCK_SHAPE";
    }
    if (!caught || op.session_active())
      throw std::runtime_error("E_CALLBACK_FAILURE_DID_NOT_ABORT");
    if (role == Role::Client)
      std::cout << "{\"entry\":\"merged_callback_failure\",\"native_version\":\""
                PRAG_NATIVE_VERSION "\",\"both_sessions_aborted\":true,"
                "\"peer_unblocked\":true,\"full_R2\":false,\"GPU_used\":false}\n";
    return 0;
  }
  const auto before_reject = io.bytes();
  bool rejected = false;
  try {
    op.private_linear_streamed(std::vector<Word>(19 * 8192 + 1), 1, 12, 12,
        [](size_t, size_t actual, size_t, size_t count) {
          return std::vector<std::vector<Word>>(count, std::vector<Word>(actual));
        });
  } catch (const std::runtime_error &e) {
    rejected = std::string(e.what()) == "E_SHAPE";
  }
  if (!rejected || io.bytes() != before_reject)
    throw std::runtime_error("E_BFV_K20_REJECTION");
  std::ostringstream cases;
  const size_t dimensions[] = {3, 8193, 19 * 8192}, counts[] = {3, 513, 1};
  size_t all_callback_calls = 0;
  for (size_t case_id = 0; case_id < 3; ++case_id) {
    const size_t columns = dimensions[case_id], rows = counts[case_id];
    const size_t blocks = 1 + (columns - 1) / 8192;
    const size_t d = std::min(size_t(8192), columns);
    const size_t capacity = (columns <= 8192 ? size_t(512) : size_t(1));
    const size_t groups = (rows + capacity - 1) / capacity;
    auto whole_at = [case_id](size_t j) -> Word {
      if (case_id == 1)
        return j == 0 || j == 8192 ? 7 : 0;
      return Word(int((j * 13 + 7) % 101) - 50) & MASK;
    };
    auto weight_at = [case_id](size_t row, size_t j) -> Word {
      if (case_id < 2 && row == 1)
        return 0;
      if (case_id == 1) {
        // Row0 cancels across ciphertexts, row1 is zero, row2 is negative.
        if (row == 0)
          return j == 0 ? 11 : (j == 8192 ? Word(-11) & MASK : 0);
        return j == 0 ? 3 : (j == 8192 ? Word(-5) & MASK : 0);
      }
      return Word(int((j * (row + 3) + 5) % 17) - 8) & MASK;
    };
    std::vector<Word> shares(columns), expected(rows, 0);
    for (size_t j = 0; j < columns; ++j) {
      const Word whole = whole_at(j);
      const Word client = (UINT64_C(0x1928abcdef) + j * UINT64_C(125939)) & MASK;
      shares[j] = role == Role::Client ? client : (whole - client) & MASK;
      for (size_t row = 0; row < rows; ++row)
        expected[row] = (expected[row] + whole * weight_at(row, j)) & MASK;
    }
    size_t callback_calls = 0;
    PrivateWeightBlock callback = [&](size_t begin, size_t actual,
                                      size_t first, size_t count) {
      if (role != Role::Server)
        throw std::runtime_error("E_CLIENT_WEIGHT_CALLBACK");
      if (begin % 8192 || actual != std::min(d, columns - begin) ||
          first >= rows || count != std::min(capacity, rows - first))
        throw std::runtime_error("E_BFV_CALLBACK_PUBLIC_SCHEDULE");
      ++callback_calls;
      std::vector<std::vector<Word>> weights(count, std::vector<Word>(actual));
      for (size_t row = 0; row < count; ++row)
        for (size_t j = 0; j < actual; ++j)
          weights[row][j] = weight_at(first + row, begin + j);
      return weights;
    };
    io.set_stage(106 + case_id);
    const auto before = io.bytes();
    std::vector<Word> out;
    if (case_id == 0) {
      std::vector<std::vector<Word>> weights;
      if (role == Role::Server)
        weights = callback(0, columns, 0, rows);
      out = op.private_linear(shares, weights, rows, 12, 12);
    } else {
      out = op.private_linear_streamed(shares, rows, 12, 12, callback);
    }
    const auto bytes = io.bytes() - before;
    if (callback_calls != (role == Role::Server ? blocks * groups : 0))
      throw std::runtime_error("E_BFV_CALLBACK_COUNT");
    all_callback_calls += callback_calls;
    io.set_operation(100000 + case_id);
    if (role == Role::Server) {
      io.send_data(out.data(), int(out.size() * sizeof(Word)));
    } else {
      std::vector<Word> other(rows);
      io.recv_data(other.data(), int(other.size() * sizeof(Word)));
      for (size_t row = 0; row < rows; ++row)
        out[row] = op.add(out[row], other[row]);
      if (out != expected)
        throw std::runtime_error("E_BFV_MERGED_SYNTHETIC_MISMATCH");
      if (case_id)
        cases << ',';
      cases << "{\"columns\":" << columns << ",\"rows\":" << rows
            << ",\"input_blocks\":" << blocks << ",\"capacity\":" << capacity
            << ",\"returned_ciphertext_groups\":" << groups
            << ",\"last_actual_columns\":" << (columns - (blocks - 1) * 8192)
            << ",\"raw_fractional_bits\":24,\"ring_outputs\":[";
      for (size_t row = 0; row < rows; ++row) {
        if (row) cases << ',';
        cases << out[row];
      }
      cases << "],\"result\":\"matched\",\"application_bytes_before_reconstruction\":"
            << bytes << '}';
    }
  }
  if (all_callback_calls != (role == Role::Server ? 1046 : 0))
    throw std::runtime_error("E_BFV_CALLBACK_TOTAL");
  op.close_session();
  if (role == Role::Client)
    std::cout << "{\"entry\":\"tile108_long_input_BFV_probe\","
                 "\"native_version\":\"" PRAG_NATIVE_VERSION "\","
                 "\"two_processes\":true,\"raw_fractional_bits\":24,"
                 "\"K20_public_rejected_before_network\":true,"
                 "\"client_callback_never_called\":true,\"server_callback_calls\":1046,"
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
  const bool failure = argc == 3 && std::string(argv[2]) == "--callback-failure";
#ifdef PRAG_PRODUCTION_BFV_MERGED_PROBE
  if (argc != 3 || failure)
    return 2;
#endif
  const std::string certs = argc == 3 && !failure ? argv[2] : "";
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
      result = run(Role::Server, sockets[1], argv[1], certs, failure);
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
    }
    _exit(result);
  }
  close(sockets[1]);
  try {
    result = run(Role::Client, sockets[0], argv[1], certs, failure);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
  int status = 0;
  waitpid(child, &status, 0);
  return result || !WIFEXITED(status) || WEXITSTATUS(status);
}
