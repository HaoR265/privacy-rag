// Public failure-path probe only; no output or input reconstruction.
#include "prag/batch.hpp"
#include <csignal>
#include <iostream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace prag;
static int run(Role role, int fd, const std::string &prefix,
               const std::string &cert, int mode) {
  alarm(20);
  std::array<unsigned char, 16> sid{};
  sid[0] = 111;
  sid[1] = 31;
  std::array<unsigned char, 32> cfg{};
  cfg[0] = 111;
  Channel io(fd, sid, cfg);
  auto own = role == Role::Client ? "U" : "S-compute";
  io.enable_tls(role, cert + "/ca.pem", cert + "/" + own + ".pem",
                cert + "/" + own + ".key", "localhost");
  auto ledger = prefix + (role == Role::Client ? ".C.ledger" : ".S.ledger");
  SecureOps s(role, io, ledger, false);
  BatchOps b(s);
  bool caught = false;
  std::string error;
  try {
    switch (mode) {
    case 0:
      b.run({BatchCode::Floor, 1, 1, {role == Role::Client ? 12u : 24u, 0, 0}},
            {s.public_word(3)});
      break;
    case 1:
      b.run({BatchCode::Floor, 1, 32769, {12, 0, 0}}, {});
      break;
    case 2:
      b.run({BatchCode::Floor, 1, 1, {12, 0, 0}},
            {role == Role::Client ? MASK + 1 : Word(0)});
      break;
    case 3: {
      std::vector<Word> table(256);
      if (role == Role::Server)
        table[255] = 1;
      b.lookup({0}, table);
      break;
    }
    case 4:
      s.close_session();
      b.run({BatchCode::Floor, 1, 1, {12, 0, 0}}, {0});
      break;
    case 5: {
      SecureOps reused(role, io, ledger, false);
      break;
    }
    case 6:
      b.run({BatchCode::GuardedSum, 1, 1, {Word{1} << 56, 0, 0}}, {0}, {0});
      break;
    case 7:
      b.matrix({0}, {0}, 1, 1281, 1, 12, 12, 1);
      break;
    case 8:
      b.matrix({0,0}, {0,0}, 1, 2, 1, 0, 0, 1,
               [](const char *phase, bool start) {
                 if(start && std::string(phase)=="matrix_local_preprocessing")
                   throw std::runtime_error("PROBE_ABORT_AFTER_BFV_CROSSES");
               });
      break;
    case 9:
      b.matrix({MASK+1}, {0}, 1, 1, 1, 0, 0, 1);
      break;
    case 10: {
      size_t columns=role==Role::Client?3:4;
      b.matrix({0,0}, std::vector<Word>(2*columns), 1, 2, columns, 0, 0, 1);
      break;
    }
    case 11:
      b.matrix({0,0}, {0,0}, 1, 2, 1, 0, 0, 1,
               [](const char *phase, bool start) {
                 if(!start && std::string(phase)=="matrix_local_preprocessing")
                   throw std::runtime_error("PROBE_ABORT_AT_PHASE_END");
               });
      break;
    case 12:
      b.matrix({0,0}, {0,0}, 1, 2, 1, 0, 0, 1,
               [&](const char *phase, bool start) {
                 if(std::string(phase)!="matrix_cross_A_C_B_S") return;
                 if(start) s.abort_session();
                 else throw std::runtime_error("PROBE_SECONDARY_CALLBACK_FAILURE");
               });
      break;
    default:
      throw std::runtime_error("PROBE_MODE");
    }
  } catch (const std::exception &e) {
    caught = true;
    error = e.what();
  }
  if (mode == 5)
    s.abort_session();
  bool good = caught && !s.session_active();
  if (mode == 11) good = good && error == "PROBE_ABORT_AT_PHASE_END";
  if (mode == 12) good = good && error != "PROBE_SECONDARY_CALLBACK_FAILURE";
  std::string line =
      "{\"party\":\"" + std::string(role == Role::Client ? "C" : "S") +
      "\",\"mode\":" + std::to_string(mode) +
      ",\"expected_rejection\":" + (good ? "true" : "false") + ",\"error\":\"" +
      error + "\",\"ledger\":\"" + ledger + "\"}\n";
  write(STDOUT_FILENO, line.data(), line.size());
  alarm(0);
  return good ? 0 : 1;
}
int main(int argc, char **argv) {
  if (argc != 4)
    return 2;
  std::signal(SIGPIPE, SIG_IGN);
  for (int mode = 0; mode < 13; ++mode) {
    int fds[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds))
      return 3;
    auto child = fork();
    if (child < 0)
      return 3;
    int result = 1;
    auto call = [&](Role role, int fd) {
      return run(role, fd, std::string(argv[1]) + "_" + std::to_string(mode),
                 argv[2], mode);
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
    if (result || !WIFEXITED(status) || WEXITSTATUS(status))
      return 1;
  }
  return 0;
}
