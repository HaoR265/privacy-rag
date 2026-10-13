#pragma once
#include "prag/channel.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
namespace prag {
// Two independently authenticated TLS connections, one per direction. Each SSL
// object has exactly one I/O thread. All three channels must outlive this
// guard. The public session close transaction precedes close(); no
// total-session timer.
class Heartbeat {
  Channel &out_, &in_, &compute_;
  std::function<void()> on_abort_;
  std::atomic<bool> stopped_{false}, failed_{false};
  std::atomic<uint64_t> received_{0};
  std::mutex mutex_;
  std::condition_variable changed_;
  std::chrono::steady_clock::time_point last_;
  std::thread sender_, receiver_, watchdog_;
  void fail() noexcept;
  void send_loop() noexcept;
  void receive_loop() noexcept;
  void watch_loop() noexcept;

public:
  static constexpr unsigned interval_seconds = 30;
  static constexpr unsigned inactivity_seconds = 180;
  Heartbeat(Channel &outbound, Channel &inbound, Channel &compute,
            std::function<void()> on_abort);
  ~Heartbeat();
  Heartbeat(const Heartbeat &) = delete;
  void close() noexcept;
  void check() const;
  bool failed() const noexcept { return failed_.load(); }
  uint64_t received() const noexcept { return received_.load(); }
};
} // namespace prag
