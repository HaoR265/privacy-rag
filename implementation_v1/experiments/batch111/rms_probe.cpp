// One complete 128-wide RMS kernel using real Q gamma; no Transformer is run.
// Reconstruction is confined to this diagnostic and is never linked to worker.
#include "artifacts.hpp"
#include "batch_numerics.hpp"
#include "meter.hpp"
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <iostream>
#include <mutex>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
using namespace prag;
using namespace prag::model;
static int run(Role role, int fd, const std::string &prefix,
               const std::string &cert, const std::string &weights,
               const std::string &reference, bool diagonal,
               const std::string &digest, Word row_l1, size_t dimension,
               size_t batch) {
  if (!diagonal)
    throw std::runtime_error("RMS_GAMMA_ONLY");

  std::unique_ptr<MappedWeights> matrix;
  if (role == Role::Server) {
    matrix = std::make_unique<MappedWeights>(
        weights, dimension, diagonal ? 1 : dimension, diagonal);
    // Same real mapped-byte and per-row bound checker used by worker.
    matrix->require_row_bound(digest.c_str(), row_l1,
                              diagonal ? Word(192 << 12)
                                       : numeric_profile::main_abs_f12);
  }
  std::array<unsigned char, 16> session{};
  session[0] = 111;
  session[1] = 11 + diagonal;
  std::array<unsigned char, 32> config{};
  config[0] = 111;
  Channel io(fd, session, config);
  const std::string own = role == Role::Client ? "U" : "S-compute";
  io.enable_tls(role, cert + "/ca.pem", cert + "/" + own + ".pem",
                cert + "/" + own + ".key", "localhost");
  SecureOps op(role, io,
               prefix + (role == Role::Client ? ".C.ledger" : ".S.ledger"),
               false);
  Row x(dimension * batch), synthetic(dimension * batch);
  std::ifstream input_file(reference + ".input", std::ios::binary);
  if (!input_file.read(reinterpret_cast<char *>(synthetic.data()),
                       synthetic.size() * 8) ||
      input_file.peek() != EOF)
    throw std::runtime_error("INPUT_SHAPE");
  for (size_t j = 0; j < dimension * batch; ++j) {
    Word whole = synthetic[j];
    Word c = (UINT64_C(0x21fedcba987654) + j * 125939) & MASK;
    x[j] = role == Role::Client ? c : (whole - c) & MASK;
  }
  PrivateMatrix gamma{dimension, 1, 12, {}, {}};
  if (role == Role::Server)
    gamma.server_reader = [&](size_t r, size_t c) { return matrix->at(r, c); };
  std::array<Row, 3> unused_exp;
  for (auto &row : unused_exp)
    row.resize(256);
  PhaseMeter meter(io, prefix + (role == Role::Client ? ".C" : ".S"));
  current_meter = &meter;
  std::vector<std::unique_ptr<PhaseMeter::Scope>> phases;
  Numerics numerics(op, unused_exp, [&](const char *name, bool start) {
    if (start)
      phases.push_back(std::make_unique<PhaseMeter::Scope>(meter, name));
    else
      phases.pop_back();
  }); // RMS never accesses exponential tables.
  auto start = std::chrono::steady_clock::now();
  auto before = io.bytes();
  std::mutex mutex;
  std::condition_variable cv;
  bool done = false;
  std::thread deadline([&] {
    std::unique_lock<std::mutex> lock(mutex);
    if (!cv.wait_for(lock, std::chrono::seconds(300), [&] { return done; }))
      io.abort();
  });
  Row y;
  Bit numeric_failure = false;
  std::string failure;
  try {
    PhaseMeter::Scope all(meter, "full_RMS");
    Rows rows;
    for (size_t i = 0; i < batch; ++i)
      rows.emplace_back(x.begin() + i * dimension,
                        x.begin() + (i + 1) * dimension);
    auto out =
        numerics.rms_many(rows, gamma,
                          dimension == 128 ? numeric_profile::generation.q_norm
                                           : numeric_profile::main_abs_f12);
    for (auto &row : out)
      y.insert(y.end(), row.begin(), row.end());
    numeric_failure = numerics.failure();
  } catch (const std::exception &e) {
    failure = e.what();
  }
  {
    std::lock_guard<std::mutex> lock(mutex);
    done = true;
  }
  cv.notify_all();
  deadline.join();
  if (!failure.empty()) {
    const auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    op.abort_session();
    if (role == Role::Client)
      std::cout << "{\"result\":\"incomplete_deadline_or_protocol_error\","
                   "\"completed\":false,\"elapsed_seconds\":"
                << elapsed << ",\"application_bytes\":" << io.bytes() - before
                << ",\"dimension\":" << dimension << ",\"batch\":" << batch
                << ",\"error\":\"" << failure
                << "\",\"deadline_seconds\":300}\n";
    return 0;
  }
  auto bytes = io.bytes() - before;
  double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  io.set_operation(1000000000);
  if (role == Role::Server) {
    io.send_data(y.data(), int(y.size() * 8));
    auto failed = numeric_failure;
    io.send_data(&failed, sizeof(failed));
  } else {
    Row other(dimension * batch), wanted(dimension * batch);
    io.recv_data(other.data(), int(other.size() * 8));
    std::ifstream f(reference, std::ios::binary);
    if (!f.read(reinterpret_cast<char *>(wanted.data()), wanted.size() * 8) ||
        f.peek() != EOF)
      throw std::runtime_error("REFERENCE_SHAPE");
    for (size_t j = 0; j < dimension * batch; ++j)
      if (((other[j] + y[j]) & MASK) != wanted[j])
        throw std::runtime_error("RMS_MISMATCH");
    Bit server_failed;
    io.recv_data(&server_failed, sizeof(server_failed));
    uint8_t expected_failure = 0;
    std::ifstream fflag(reference + ".flag", std::ios::binary);
    if (!fflag.read(reinterpret_cast<char *>(&expected_failure), 1))
      throw std::runtime_error("REFERENCE_FLAG");
    if ((server_failed != numeric_failure) != bool(expected_failure))
      throw std::runtime_error("RMS_NUMERIC_FAILURE_MISMATCH");
  }
  op.close_session();
  if (role == Role::Client)
    std::cout << "{\"result\":\"matched_CUDA_R1\",\"application_bytes\":"
              << bytes << ",\"elapsed_seconds\":" << seconds
              << ",\"dimension\":" << dimension << ",\"batch\":" << batch
              << ",\"diagonal_gamma\":" << (diagonal ? "true" : "false")
              << ",\"raw_fractional_bits\":12,\"native_TLS13_mTLS\":true,"
                 "\"isolated_development\":false,\"model_executed\":false}\n";
  return 0;
}
int main(int argc, char **argv) {
  if (argc != 10)
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
    return run(role, fd, argv[1], argv[2], argv[3], argv[4], std::stoi(argv[5]),
               argv[6], std::stoull(argv[7]), std::stoull(argv[8]),
               std::stoull(argv[9]));
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
