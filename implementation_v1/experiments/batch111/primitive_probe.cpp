// Isolated reconstruction harness, never linked into the worker/library.
#include "prag/batch.hpp"
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
static Word read(std::ifstream &f) {
  Word x;
  if (!f.read(reinterpret_cast<char *>(&x), 8))
    throw std::runtime_error("FIXTURE");
  return x;
}
static int run(Role role, int fd, const std::string &prefix,
               const std::string &cert, const std::string &fixture) {
  std::array<unsigned char, 16> sid{};
  sid[0] = 111;
  sid[1] = 25;
  std::array<unsigned char, 32> cfg{};
  cfg[0] = 111;
  Channel io(fd, sid, cfg);
  auto own = role == Role::Client ? "U" : "S-compute";
  io.enable_tls(role, cert + "/ca.pem", cert + "/" + own + ".pem",
                cert + "/" + own + ".key", "localhost");
  SecureOps s(role, io,
              prefix + (role == Role::Client ? ".C.ledger" : ".S.ledger"),
              false);
  BatchOps batch(s);
  std::ifstream f(fixture, std::ios::binary);
  auto cases = read(f);
  if (cases > 1000)
    throw std::runtime_error("FIXTURE");
  if (role == Role::Client)
    std::cout << "{\"cases\":[";
  for (size_t c = 0; c < cases; ++c) {
    auto code = read(f);
    BatchSpec d{BatchCode(code), read(f), read(f), {read(f), read(f), read(f)}};
    auto nw = read(f), nb = read(f), ow = read(f), ob = read(f);
    if (nw > 65536 || nb > 59 * 32768 || ow > 65536 || ob > 59 * 32768)
      throw std::runtime_error("FIXTURE");
    std::vector<Word> w(nw), wanted(ow);
    std::vector<uint8_t> b(nb), wanted_bits(ob);
    std::vector<Word> table;
    if (code == 11) {
      table.resize(256);
      for (auto &v : table)
        v = read(f);
    }
    for (size_t j = 0; j < nw; ++j) {
      Word whole = read(f),
           share = (UINT64_C(0x43765213abcdef) + 125939 * j) & MASK;
      w[j] = role == Role::Client ? share : (whole - share) & MASK;
    }
    for (size_t j = 0; j < nb; ++j) {
      char value;
      if (!f.get(value))
        throw std::runtime_error("FIXTURE");
      uint8_t share = (j * 7 + 3) % 2;
      b[j] = role == Role::Client ? share : share ^ uint8_t(value);
    }
    for (auto &v : wanted)
      v = read(f);
    if (ob && !f.read(reinterpret_cast<char *>(wanted_bits.data()), ob))
      throw std::runtime_error("FIXTURE");
    auto start = std::chrono::steady_clock::now();
    auto before = io.bytes(), frames = io.frames_received() + io.frames_sent(),
         runs = io.direction_runs();
    auto stats = batch.stats();
    BatchResult result;
    if (code == 101) {
      const size_t split = d.rows * d.parameters[0];
      result.words = batch.matrix(std::vector<Word>(w.begin(), w.begin()+split),
          std::vector<Word>(w.begin()+split, w.end()), d.rows, d.parameters[0], d.cols,
          0, 0, (Word{1} << 56)-1);
    } else if (code == 100) {
      result.words =
          batch.multiply(std::vector<Word>(w.begin(), w.begin() + nw / 2),
                         std::vector<Word>(w.begin() + nw / 2, w.end()));
    } else if (code == 11)
      result.words = batch.lookup(w, table);
    else
      result = batch.run(d, w, b);
    auto bytes = io.bytes() - before;
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    auto after = batch.stats();
    auto frame_delta = io.frames_received() + io.frames_sent() - frames,
         run_delta = io.direction_runs() - runs;
    if (result.words.size() != ow || result.bits.size() != ob)
      throw std::runtime_error("OUTPUT_SHAPE");
    io.set_operation(1000000000 + c);
    if (role == Role::Server) {
      if (ow)
        io.send_data(result.words.data(), int(ow * 8));
      if (ob)
        io.send_data(result.bits.data(), int(ob));
    } else {
      std::vector<Word> other(ow);
      std::vector<uint8_t> other_bits(ob);
      if (ow)
        io.recv_data(other.data(), int(ow * 8));
      if (ob)
        io.recv_data(other_bits.data(), int(ob));
      for (size_t j = 0; j < ow; ++j)
        if (((other[j] + result.words[j]) & MASK) != wanted[j])
          throw std::runtime_error("WORD_MISMATCH_case_" + std::to_string(c) +
                                   "_index_" + std::to_string(j));
      for (size_t j = 0; j < ob; ++j)
        if ((other_bits[j] ^ result.bits[j]) != wanted_bits[j])
          throw std::runtime_error("BIT_MISMATCH_case_" + std::to_string(c) +
                                   "_index_" + std::to_string(j));
      if (c)
        std::cout << ',';
      std::cout << "{\"case\":" << c << ",\"opcode\":" << code
                << ",\"rows\":" << d.rows << ",\"cols\":" << d.cols
                << ",\"seconds\":" << sec << ",\"application_bytes\":" << bytes
                << ",\"frames\":" << frame_delta
                << ",\"direction_runs\":" << run_delta
                << ",\"GC_AND\":" << after.and_gates - stats.and_gates
                << ",\"GC_input_OT\":"
                << after.input_ot_choices - stats.input_ot_choices
                << ",\"matched\":true}" << std::flush;
    }
  }
  if (f.peek() != EOF)
    throw std::runtime_error("FIXTURE_TAIL");
  s.close_session();
  if (role == Role::Client)
    std::cout << "],\"result\":\"matched_CUDA_integer_oracle\",\"native_mTLS\":"
                 "true,\"model_executed\":false}\n";
  return 0;
}
int main(int argc, char **argv) {
  if (argc != 4)
    return 2;
  std::signal(SIGPIPE, SIG_IGN);
  int fds[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds))
    return 3;
  auto child = fork();
  if (child < 0)
    return 3;
  int result = 1;
  auto call = [&](Role role, int fd) {
    return run(role, fd, argv[1], argv[2], argv[3]);
  };
  if (!child) {
    close(fds[0]);
    try {
      result = call(Role::Server, fds[1]);
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
    }
    _exit(result);
  }
  close(fds[1]);
  try {
    result = call(Role::Client, fds[0]);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
  int status = 0;
  waitpid(child, &status, 0);
  return result || !WIFEXITED(status) || WEXITSTATUS(status);
}
