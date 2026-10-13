// Bounded public-synthetic diagnostics; absent from production builds.
#include "prag/bfv.hpp"
#include "prag/development_bfv_decode.hpp"
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;

static int run(Role role, int fd, const std::string &prefix) {
  std::array<unsigned char, 16> session{};
  std::array<unsigned char, 32> configuration{};
  session[0] = 104;
  configuration[0] = 60;
  Channel io(fd, session, configuration);
  SecureOps op(role, io,
               prefix + (role == Role::Client ? ".C.ledger" : ".S.ledger"),
               true);
  auto primes = locked_bfv_parameters().coeff_modulus();
  primes.pop_back();
  const mpz_class q = modulus_product(primes), t = mpz_class(1) << 59,
                  h = q / 2;
  const mpz_class first = (q - h + t - 1) / t,
                  last = (t * q - h + t - 1) / t;
  // Both input owners, the first and final rounding transitions, the 60th
  // quotient bit, phase=q, and the largest possible phase sum 2q-2.
  const std::vector<std::pair<mpz_class, mpz_class>> cases{
      {0, 0}, {first - 1, 0}, {0, first}, {last - 1, 0},
      {0, last}, {q - 1, 0}, {q - 1, 1}, {q - 1, q - 1}};
  std::vector<mpz_class> local;
  std::vector<Word> expected;
  bool full_quotient_bit59_covered = false;
  for (const auto &item : cases) {
    local.push_back(role == Role::Client ? item.first : item.second);
    const mpz_class phase = (item.first + item.second) % q;
    const mpz_class whole = (phase * t + h) / q;
    const mpz_class result = whole % t;
    expected.push_back(result.get_ui());
    full_quotient_bit59_covered |= bool(mpz_tstbit(whole.get_mpz_t(), 59));
  }
  if (!full_quotient_bit59_covered)
    throw std::runtime_error("E_DECODE60_BOUNDARY_COVERAGE");
  io.set_stage(104);
  io.set_operation(1);
  const auto before_gc = io.bytes();
  const auto decoded = development_bfv59_decode(op, local);
  const auto phase_bytes = io.bytes() - before_gc;
  if (role == Role::Client && decoded != expected)
    throw std::runtime_error("E_DECODE60_PHASE_MISMATCH");
  if (role == Role::Server)
    for (auto value : decoded)
      if (value != 0)
        throw std::runtime_error("E_DECODE60_OUTPUT_ROLE");

  // Two complete real BFV controlled returns, including the zero-weight path.
  // The whole inputs below are public synthetic values; protocol entropy is
  // fresh. Reconstruction is confined to this development executable.
  const Word full_x = Word(-12345) & MASK, xc = 125934, full_y = 79,
             yc = 98867;
  const std::vector<Word> x{
      role == Role::Client ? xc : (full_x - xc) & MASK,
      role == Role::Client ? yc : (full_y - yc) & MASK, op.public_word(12)};
  const std::vector<std::vector<Word>> weights =
      role == Role::Server
          ? std::vector<std::vector<Word>>{{2, Word(-3) & MASK, 5}, {0, 0, 0}}
          : std::vector<std::vector<Word>>{};
  io.set_operation(2);
  const auto before_bfv = io.bytes();
  auto output = op.private_linear(x, weights, 2, 12, 12);
  const auto bfv_bytes = io.bytes() - before_bfv;
  io.set_operation(3);
  if (role == Role::Server) {
    io.send_data(output.data(), int(output.size() * sizeof(Word)));
    op.close_session();
    return 0;
  }
  std::vector<Word> other(output.size());
  io.recv_data(other.data(), int(other.size() * sizeof(Word)));
  for (size_t i = 0; i < output.size(); ++i)
    output[i] = op.add(output[i], other[i]);
  if (output != std::vector<Word>{(2 * full_x - 3 * full_y + 60) & MASK, 0})
    throw std::runtime_error("E_DECODE60_BFV_MISMATCH");
  op.close_session();
  std::cout << "{\"entry\":\"isolated_bfv_decode60_probe\","
               "\"native_version\":\"" PRAG_NATIVE_VERSION "\",\"two_processes\":true,"
               "\"phase_boundary_cases\":8,\"phase_results\":\"matched\","
               "\"full_quotient_bit59_covered\":true,"
               "\"real_BFV_outputs\":2,\"BFV_nonzero_and_zero_weights\":\"matched\","
               "\"phase_batch_application_bytes\":"
            << phase_bytes << ",\"BFV_batch_application_bytes\":" << bfv_bytes
            << ",\"client_application_bytes\":" << io.bytes()
            << ",\"transport\":\"local_AF_UNIX_development_transport\","
               "\"mTLS\":false,\"full_R2\":false,\"timing_comparison\":false}\n";
  return 0;
}

int main(int argc, char **argv) {
  std::signal(SIGPIPE, SIG_IGN);
  if (argc != 2)
    return 2;
  int sv[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv))
    return 3;
  const pid_t child = fork();
  if (child < 0)
    return 3;
  int result = 1;
  if (child == 0) {
    close(sv[0]);
    try {
      result = run(Role::Server, sv[1], argv[1]);
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
    }
    _exit(result);
  }
  close(sv[1]);
  try {
    result = run(Role::Client, sv[0], argv[1]);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
  int state = 0;
  waitpid(child, &state, 0);
  return result || !WIFEXITED(state) || WEXITSTATUS(state);
}
