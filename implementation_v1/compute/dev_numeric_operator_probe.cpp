// Isolated developer probe: every input and opened result is public synthetic.
// Uses real SecureOps and materialized constants; never linked into prag_worker.
#include "artifacts.hpp"
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
using namespace prag;
using namespace prag::model;

static std::map<std::string, Row> read_expected(const std::string& path) {
  std::ifstream f(path); std::string magic, label;
  if (!(f >> magic) || magic != "PRDEVNUM1") throw std::runtime_error("E_PROBE_FIXTURE");
  std::map<std::string, Row> cases;
  std::size_t count;
  while (f >> label >> count) {
    if (count < 2 || count > 3 || cases.count(label)) throw std::runtime_error("E_PROBE_FIXTURE");
    Row values(count);
    for (auto& v : values) { int64_t signed_value; if (!(f >> signed_value)) throw std::runtime_error("E_PROBE_FIXTURE"); v = Word(signed_value) & MASK; }
    cases.emplace(label, std::move(values));
  }
  if (!f.eof() || cases.size() != 7) throw std::runtime_error("E_PROBE_FIXTURE");
  return cases;
}

static int run(Role role, int fd, const std::string& prefix, const std::string& constants_path,
               const std::string& fixture, const std::string& selected) {
  const auto expected = read_expected(fixture);
  if (selected != "all" && !expected.count(selected)) throw std::runtime_error("E_PROBE_CASE");
  Constants constants(constants_path);
  std::array<unsigned char,16> session{}; session[0] = 67;
  std::array<unsigned char,32> configuration{}; configuration[0] = 9;
  Channel io(fd, session, configuration);
  SecureOps op(role, io, prefix + (role == Role::Client ? ".C.ledger" : ".S.ledger"), true);
  // Fixed nontrivial additive/XOR shares of known public synthetic values.
  auto share = [&](int64_t v) { return role == Role::Client ? Word{912837} : (Word(v) - 912837) & MASK; };
  auto bit = [&](bool v) { return role == Role::Client ? true : !v; };
  for (const auto& label : {"exp_zero", "exp_32", "exp_tail", "silu_positive", "silu_negative", "softmax_valid", "softmax_invalid"}) {
    if (selected != "all" && selected != label) continue; // Public developer-case selection only.
    Numerics n(op, constants.exp); Row output;
    const auto before = io.bytes(); const auto started = std::chrono::steady_clock::now();
    const std::string name(label);
    if (name == "exp_zero") output.push_back(n.exp_negative(share(0)));
    else if (name == "exp_32") output.push_back(n.exp_negative(share(32 << 12)));
    else if (name == "exp_tail") output.push_back(n.exp_negative(share((32 << 12) + 1)));
    else if (name == "silu_positive") output.push_back(n.silu(share(4096)));
    else if (name == "silu_negative") output.push_back(n.silu(share(-4096)));
    else {
      const bool valid = name == "softmax_valid";
      output = n.softmax({share(4096), share(-4096)}, {bit(valid), bit(true)}, bit(valid));
    }
    output.push_back(op.b2a_bit(n.failure()));
    // Opening is confined to this development executable and known public fixture.
    if (role == Role::Server) io.send_data(output.data(), int(output.size() * sizeof(Word)));
    else {
      Row other(output.size()); io.recv_data(other.data(), int(other.size() * sizeof(Word)));
      for (std::size_t i = 0; i < output.size(); ++i) output[i] = op.add(output[i], other[i]);
      if (output != expected.at(name)) throw std::runtime_error("E_NUMERIC_OPERATOR_MISMATCH:" + name);
      const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      std::cout << "{\"entry\":\"isolated_public_numeric_operator_probe\",\"case\":\"" << name
                << "\",\"matched_R1\":true,\"two_processes\":true,\"real_SecureOps\":true,\"actual_constants\":true,\"client_channel_bytes\":"
                << io.bytes()-before << ",\"elapsed_seconds\":" << elapsed
                << ",\"full_model_not_run\":true,\"production_TLS_not_run\":true}" << std::endl;
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 5) { std::cerr << "usage: prag_dev_numeric_operator_probe FRESH_LEDGER_PREFIX CONSTANTS_BIN PUBLIC_R1_FIXTURE CASE_OR_all\n"; return 2; }
  int sockets[2]; if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets)) return 3;
  pid_t child = fork(); if (child < 0) { close(sockets[0]); close(sockets[1]); return 3; }
  int rc = 1;
  try {
    if (child == 0) { close(sockets[0]); rc = run(Role::Server, sockets[1], argv[1], argv[2], argv[3], argv[4]); _exit(rc); }
    close(sockets[1]); rc = run(Role::Client, sockets[0], argv[1], argv[2], argv[3], argv[4]);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; shutdown(sockets[child == 0 ? 1 : 0], SHUT_RDWR);
    if (child == 0) _exit(1);
  }
  int state = 0; waitpid(child, &state, 0);
  return rc || !WIFEXITED(state) || WEXITSTATUS(state);
}
