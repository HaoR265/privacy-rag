#pragma once
#if !defined(__x86_64__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "Native protocol version 1 requires the locked x86_64 little-endian ABI"
#endif
#include "utils/io_channel.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <openssl/ssl.h>
#include <string>
#include <vector>
namespace prag {
enum class Role { Client, Server };
// Fixed-width public frame header. Low-level SCI points override its native
// size_t encoding, using one canonical 65-byte uncompressed P-256 point.
class Channel : public sci::IOChannel<Channel> {
  int fd_;
  SSL_CTX *context_ = nullptr;
  SSL *tls_ = nullptr;
  std::array<unsigned char, 16> session_;
  std::array<unsigned char, 32> config_;
  bool authenticated_ = false;
  uint64_t stage_ = 1, operation_ = 0;
  uint64_t sent_ = 0, received_ = 0;
  char diagnostic_last_direction_ = 0;
  uint64_t diagnostic_direction_runs_ = 0;
  std::atomic<uint64_t> bytes_{0};
  void write_exact(const void *, size_t);
  void read_exact(void *, size_t);

public:
  Channel(int owned_fd, std::array<unsigned char, 16> session,
          std::array<unsigned char, 32> config);
  ~Channel();
  Channel(const Channel &) = delete;
  Channel &operator=(const Channel &) = delete;
  void enable_tls(Role, const std::string &ca, const std::string &certificate,
                  const std::string &key, const std::string &peer_dns);
  void set_stage(uint64_t stage) { stage_ = stage; }
  void set_operation(uint64_t operation) { operation_ = operation; }
  bool authenticated() const { return authenticated_; }
  // Interrupt blocking I/O; safe from the independent heartbeat watchdog.
  // The owner must join all I/O threads before destroying this object.
  void abort() noexcept;
  std::string session_hex() const;
  std::string configuration_hex() const;
  std::string peer_fingerprint_sha256() const;
  void send_data(const void *, int);
  void recv_data(void *, int);
  void flush() {}
  void send_pt(emp::Point *, int num_pts = 1);
  void recv_pt(emp::Group *, emp::Point *, int num_pts = 1);
  uint64_t bytes() const { return bytes_.load(); }
  uint64_t frames_sent() const { return sent_; }
  uint64_t frames_received() const { return received_; }
  // Diagnostic transcript alternations, not transport RTT/latency estimates.
  uint64_t direction_runs() const { return diagnostic_direction_runs_; }
};
} // namespace prag
