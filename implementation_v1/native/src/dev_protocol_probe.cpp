// Isolated synthetic-data developer executable. Never linked into production.
#include "prag/secure_ops.hpp"
#include <csignal>
#include <fstream>
#include <iostream>
#include <openssl/rand.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
static int run(Role role, int fd, const std::string &prefix, int bfv,
               const std::string &certs) {
  std::array<unsigned char, 16> session{};
  std::array<unsigned char, 32> configuration{};
  session[0] = 12;
  configuration[0] = 2;
  Channel io(fd, session, configuration);
  if (!certs.empty()) {
    auto own = role == Role::Client ? "U" : "S-compute";
    auto peer = "localhost";
    io.enable_tls(role, certs + "/ca.pem", certs + "/" + own + ".pem",
                  certs + "/" + own + ".key", peer);
  }
  SecureOps op(role, io,
               prefix + (role == Role::Client ? ".C.ledger" : ".S.ledger"),
#ifdef PRAG_PRODUCTION_PRIMITIVE_PROBE
               false);
#else
               true);
#endif
  // Deterministic public synthetic inputs are separated into fixed
  // random-looking residues; protocol randomness itself always uses fresh OS
  // entropy.
  Word xc = 125934, x = Word(-12345) & MASK, y = 79, yc = 98867;
  Word xs = (x - xc) & MASK, ys = (y - yc) & MASK;
  Word a = role == Role::Client ? xc : xs, b = role == Role::Client ? yc : ys;
  std::vector<Word> out;
  out.push_back(op.mul(a, b));
  out.push_back(op.b2a(op.a2b(a)));
  out.push_back(op.floor(a, 4));
  out.push_back(op.b2a_bit(op.gt(a, b)));
  out.push_back(op.b2a_bit(op.equal(a, a)));
  out.push_back(op.select(op.gt(b, a), b, a));
  if (bfv) {
    const size_t columns = bfv == 2 ? 8193 : 3;
    std::vector<Word> vx(columns, op.public_word(0)), weights(columns, 0);
    vx[0] = a;
    vx[1] = b;
    vx.back() = op.public_word(12);
    weights[0] = 2;
    weights[1] = Word(-3) & MASK;
    weights.back() = 5;
    auto r = op.private_linear(
        vx,
        role == Role::Server
            ? std::vector<std::vector<Word>>{weights, std::vector<Word>(columns, 0)}
            : std::vector<std::vector<Word>>{},
        2, 12, 12);
    out.insert(out.end(), r.begin(), r.end());
  }
  // Reconstruction exists only here for public synthetic input diagnostics.
  if (role == Role::Server) {
    io.send_data(out.data(), int(out.size() * 8));
    return 0;
  }
  std::vector<Word> other(out.size());
  io.recv_data(other.data(), int(other.size() * 8));
  for (size_t j = 0; j < out.size(); ++j)
    out[j] = (out[j] + other[j]) & MASK;
  std::vector<Word> expected{(x * y) & MASK, x, Word(-772) & MASK, 0, 1, y};
  if (bfv) {
    expected.push_back((2 * x - 3 * y + 60) & MASK);
    expected.push_back(0);
  }
  if (out != expected)
    throw std::runtime_error("E_SYNTHETIC_MISMATCH");
  std::cout << "{\"entry\":\"isolated_synthetic_protocol_probe\",\"library\":\""
#ifdef PRAG_PRODUCTION_PRIMITIVE_PROBE
            << "prag_crypto"
#else
            << "prag_crypto_development"
#endif
            << "\",\"two_"
               "processes\":true,\"OT_Beaver_A2B_B2A_floor_compare_select\":"
               "\"matched\",\"BFV_controlled_GC_decode\":\""
            << (bfv ? "matched_nonzero_and_zero_weights" : "not_run")
            << "\",\"BFV_public_columns\":" << (bfv == 2 ? 8193 : (bfv ? 3 : 0))
            << ",\"transport\":\"local_AF_UNIX_development_transport\","
               "\"client_bytes\":"
            << io.bytes() << ",\"native_TLS13_mTLS\":"
            << (io.authenticated() ? "true" : "false")
            << ",\"full_system\":false}\n";
  return 0;
}
int main(int argc, char **argv) {
  std::signal(SIGPIPE, SIG_IGN);
  if (argc < 2 || argc > 4)
    return 2;
  const int bfv = argc < 3 ? 0 : (std::string(argv[2]) == "bfv-long" ? 2 : 1);
  int sv[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv))
    return 3;
  pid_t child = fork();
  if (child < 0)
    return 3;
  int result = 1;
  try {
    if (child == 0) {
      close(sv[0]);
      result = run(Role::Server, sv[1], argv[1], bfv,
                   argc == 4 ? argv[3] : "");
      _exit(result);
    }
    close(sv[1]);
    result =
        run(Role::Client, sv[0], argv[1], bfv, argc == 4 ? argv[3] : "");
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    if (child == 0)
      _exit(1);
  }
  int state = 0;
  waitpid(child, &state, 0);
  return result || !WIFEXITED(state) || WEXITSTATUS(state);
}
