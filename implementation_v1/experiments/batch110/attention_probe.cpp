// Real production shared QK -> Softmax -> shared PV; diagnostic reconstruction
// is outside timed kernels and this executable never links into the worker.
#include "artifacts.hpp"
#include "meter.hpp"
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <iostream>
#include <mutex>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
using namespace prag;
using namespace prag::model;
static Word read(std::ifstream &f) {
  Word v;
  if (!f.read(reinterpret_cast<char *>(&v), 8))
    throw std::runtime_error("FIXTURE");
  return v;
}
static int run(Role role, int fd, const std::string &prefix,
               const std::string &cert, const std::string &fixture,
               const std::string &constants) {
  std::ifstream file(fixture, std::ios::binary);
  size_t rows = read(file), keys = read(file);
  if (!rows || rows > 16 || !keys || keys > 1280)
    throw std::runtime_error("FIXTURE_SHAPE");
  auto words = [&](size_t n) {
    Row out(n);
    for (size_t j = 0; j < n; ++j) {
      auto whole = read(file),
           share = (UINT64_C(0x24bacdef013579) + j * 731123) & MASK;
      out[j] = role == Role::Client ? share : (whole - share) & MASK;
    }
    return out;
  };
  auto bits = [&](size_t n) {
    std::vector<uint8_t> out(n);
    for (size_t j = 0; j < n; ++j) {
      char b;
      if (!file.get(b))
        throw std::runtime_error("FIXTURE");
      uint8_t share = (j + 3) % 2;
      out[j] = role == Role::Client ? share : share ^ uint8_t(b);
    }
    return out;
  };
  auto q = words(rows * 128), k = words(keys * 128), v = words(keys * 128);
  auto allowed = bits(rows * keys), valid = bits(rows);
  Row expected(rows * 128);
  for (auto &x : expected)
    x = read(file);
  char expected_failed;
  if (!file.get(expected_failed) || file.peek() != EOF)
    throw std::runtime_error("FIXTURE");
  Constants c(constants);
  std::array<unsigned char, 16> sid{};
  sid[0] = 110;
  sid[1] = 28;
  std::array<unsigned char, 32> config{};
  config[0] = 110;
  Channel io(fd, sid, config);
  auto own = role == Role::Client ? "U" : "S-compute";
  io.enable_tls(role, cert + "/ca.pem", cert + "/" + own + ".pem",
                cert + "/" + own + ".key", "localhost");
  SecureOps s(role, io,
              prefix + (role == Role::Client ? ".C.ledger" : ".S.ledger"),
              false);
  PhaseMeter meter(io, prefix + (role == Role::Client ? ".C" : ".S"));
  std::vector<std::unique_ptr<PhaseMeter::Scope>> phases;
  Numerics n(s, c.exp, [&](const char *name, bool begin) {
    if (begin)
      phases.push_back(std::make_unique<PhaseMeter::Scope>(meter, name));
    else
      phases.pop_back();
  });
  std::mutex mutex;
  std::condition_variable cv;
  bool done = false;
  std::thread deadline([&] {
    std::unique_lock<std::mutex> lock(mutex);
    if (!cv.wait_for(lock, std::chrono::seconds(300), [&] { return done; }))
      io.abort();
  });
  auto start = std::chrono::steady_clock::now();
  auto before = io.bytes();
  Row result;
  std::string error;
  try {
    PhaseMeter::Scope all(meter, "complete_attention_head");
    result = n.attention(q, k, v, rows, keys, allowed, valid,
                         numeric_profile::generation, c.attention_scale);
  } catch (const std::exception &e) {
    error = e.what();
  }
  {
    std::lock_guard<std::mutex> lock(mutex);
    done = true;
  }
  cv.notify_all();
  deadline.join();
  auto bytes = io.bytes() - before;
  double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  if (!error.empty()) {
    s.abort_session();
    if (role == Role::Client)
      std::cout << "{\"result\":\"incomplete\",\"error\":\"" << error
                << "\",\"application_bytes\":" << bytes
                << ",\"seconds\":" << seconds << "}\n";
    return 1;
  }
  io.set_operation(1000000000);
  if (role == Role::Server) {
    io.send_data(result.data(), int(result.size() * 8));
    auto flag = n.failure();
    io.send_data(&flag, 1);
  } else {
    Row other(result.size());
    io.recv_data(other.data(), int(other.size() * 8));
    Bit flag;
    io.recv_data(&flag, 1);
    for (size_t j = 0; j < result.size(); ++j)
      if (((other[j] + result[j]) & MASK) != expected[j])
        throw std::runtime_error("ATTENTION_MISMATCH_" + std::to_string(j));
    if ((flag != n.failure()) != bool(expected_failed))
      throw std::runtime_error("FAILURE_FLAG_MISMATCH");
  }
  s.close_session();
  if (role == Role::Client)
    std::cout << "{\"result\":\"matched_CUDA_R1\",\"rows\":" << rows
              << ",\"keys\":" << keys
              << ",\"head_dimension\":128,\"application_bytes\":" << bytes
              << ",\"seconds\":" << seconds
              << ",\"private_weights_as_shared_inputs\":false,\"native_mTLS\":"
                 "true,\"model_executed\":false}\n";
  return 0;
}
int main(int argc, char **argv) {
  if (argc != 5)
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
    return run(role, fd, argv[1], argv[2], argv[3], argv[4]);
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
