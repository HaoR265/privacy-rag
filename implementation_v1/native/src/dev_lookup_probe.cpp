// Public synthetic diagnostics only; no reconstruction code enters the library.
#include "prag/secure_ops.hpp"
#include <csignal>
#include <iostream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
static int run(Role role, int fd, const std::string &prefix,
               const std::string &certs) {
  std::array<unsigned char, 16> session{};
  std::array<unsigned char, 32> configuration{};
  session[0] = 13;
  configuration[0] = 3;
  Channel io(fd, session, configuration);
  const std::string own = role == Role::Client ? "U" : "S-compute";
  io.enable_tls(role, certs + "/ca.pem", certs + "/" + own + ".pem",
                certs + "/" + own + ".key", "localhost");
  SecureOps op(role, io, prefix + (role == Role::Client ? ".C.ledger" : ".S.ledger"));
  bool empty_rejected = false;
  try {
    op.lookup(op.public_word(0), {});
  } catch (const std::runtime_error &e) {
    empty_rejected = std::string(e.what()) == "E_SHAPE";
  }
  if (!empty_rejected)
    throw std::runtime_error("E_EMPTY_LOOKUP_BEHAVIOR");
  const std::vector<size_t> lengths{1, 3, 256, 1280};
  const std::vector<std::vector<Word>> indices{
      {0, 1, MASK, Word(1) << 58},
      {0, 1, 2, 3, 4, Word(1) << 40, MASK},
      {0, 255, 256, MASK, (Word(1) << 40) + 17},
      {0, 1279, 1280, 2047, (Word(1) << 40) + 1279}};
  std::vector<Word> outputs, expected;
  std::vector<uint64_t> bytes;
  for (size_t group = 0; group < lengths.size(); ++group) {
    std::vector<Word> table(lengths[group]);
    for (size_t j = 0; j < table.size(); ++j) {
      switch (j % 4) {
      case 0: table[j] = (Word(j) + 29) & MASK; break;
      case 1: table[j] = (Word(0) - 7 - Word(j)) & MASK; break;
      case 2: table[j] = (Word(1) << 58) - 1; break;
      default: table[j] = Word(1) << 58; break;
      }
    }
    uint64_t before = io.bytes();
    for (Word index : indices[group]) {
      Word client_share = (Word(outputs.size() + 1) * 7919 + 456789) & MASK;
      Word local = role == Role::Client ? client_share : (index - client_share) & MASK;
      outputs.push_back(op.lookup(local, table));
      expected.push_back(index < table.size() ? table[index] : 0);
    }
    bytes.push_back(io.bytes() - before);
  }
  const size_t lookup_cases = outputs.size();
  std::vector<Word> ax(18), by(18);
  for (size_t j = 0; j < ax.size(); ++j) {
    Word x = (Word(j) - 9) & MASK, y = (Word(7 * j) - 11) & MASK;
    Word xc = 7919 * (j + 1), yc = 4567 * (j + 3);
    ax[j] = role == Role::Client ? xc : (x - xc) & MASK;
    by[j] = role == Role::Client ? yc : (y - yc) & MASK;
    expected.push_back((x * y) & MASK);
  }
  auto products = op.mul_many(ax, by);
  outputs.insert(outputs.end(), products.begin(), products.end());
  if (role == Role::Server) {
    io.send_data(outputs.data(), int(outputs.size() * sizeof(Word)));
    return 0;
  }
  std::vector<Word> other(outputs.size());
  io.recv_data(other.data(), int(other.size() * sizeof(Word)));
  for (size_t j = 0; j < outputs.size(); ++j)
    if (op.add(outputs[j], other[j]) != expected[j])
      throw std::runtime_error("E_SYNTHETIC_LOOKUP_MISMATCH");
  std::cout << "{\"entry\":\"isolated_public_lookup_probe\",\"library\":\"prag_crypto\","
               "\"two_processes\":true,\"native_TLS13_mTLS\":true,\"empty_E_SHAPE\":true,"
               "\"public_lengths\":[1,3,256,1280],\"public_cases_per_length\":[4,7,5,5],"
               "\"lookup_cases_matched\":" << lookup_cases
            << ",\"cross_1024_OT_mul_many_cases_matched\":18"
            << ",\"public_length_group_bytes\":[";
  for (size_t j = 0; j < bytes.size(); ++j)
    std::cout << (j ? "," : "") << bytes[j];
  std::cout << "],\"client_bytes\":" << io.bytes()
            << ",\"full_model\":false}\n";
  return 0;
}
int main(int argc, char **argv) {
  std::signal(SIGPIPE, SIG_IGN);
  if (argc != 3)
    return 2;
  int sockets[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets))
    return 3;
  pid_t child = fork();
  if (child < 0)
    return 3;
  int result = 1;
  try {
    if (child == 0) {
      close(sockets[0]);
      result = run(Role::Server, sockets[1], argv[1], argv[2]);
      _exit(result);
    }
    close(sockets[1]);
    result = run(Role::Client, sockets[0], argv[1], argv[2]);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    if (child == 0)
      _exit(1);
  }
  int state = 0;
  waitpid(child, &state, 0);
  return result || !WIFEXITED(state) || WEXITSTATUS(state);
}
