#include "prag/heartbeat.hpp"
#include <array>
#include <stdexcept>
namespace prag {
Heartbeat::Heartbeat(Channel &out, Channel &in, Channel &compute,
                     std::function<void()> on_abort)
    : out_(out), in_(in), compute_(compute), on_abort_(std::move(on_abort)),
      last_(std::chrono::steady_clock::now()) {
  if (&out == &in || &out == &compute || &in == &compute || !on_abort_)
    throw std::runtime_error("E_HEARTBEAT_CHANNELS");
  for (auto *c : {&out, &in}) {
    if (!c->authenticated() || !compute.authenticated() ||
        c->session_hex() != compute.session_hex() ||
        c->configuration_hex() != compute.configuration_hex() ||
        c->peer_fingerprint_sha256() != compute.peer_fingerprint_sha256())
      throw std::runtime_error("E_HEARTBEAT_BINDING");
    c->set_stage(
        0x4842); // Public reserved heartbeat stage, distinct connection.
    c->set_operation(0);
  }
  try {
    sender_ = std::thread(&Heartbeat::send_loop, this);
    receiver_ = std::thread(&Heartbeat::receive_loop, this);
    watchdog_ = std::thread(&Heartbeat::watch_loop, this);
  } catch (...) {
    close();
    throw;
  }
}
Heartbeat::~Heartbeat() { close(); }
void Heartbeat::fail() noexcept {
  if (stopped_.exchange(true))
    return;
  failed_.store(true);
  compute_.abort();
  out_.abort();
  in_.abort();
  changed_.notify_all();
  try {
    on_abort_();
  } catch (...) {
    // Transport is already closed. Do not expose callback internals or secrets.
  }
}
void Heartbeat::close() noexcept {
  stopped_.store(true);
  changed_.notify_all();
  out_.abort();
  in_.abort();
  for (auto *t : {&sender_, &receiver_, &watchdog_})
    if (t->joinable())
      t->join();
}
void Heartbeat::check() const {
  if (failed())
    throw std::runtime_error("E_HEARTBEAT_ABORT");
}
void Heartbeat::send_loop() noexcept {
  try {
    uint64_t counter = 0;
    auto next = std::chrono::steady_clock::now();
    while (!stopped_.load()) {
      std::array<unsigned char, 8> payload{};
      uint64_t v = counter++;
      for (int j = 7; j >= 0; --j) {
        payload[j] = v & 255;
        v >>= 8;
      }
      out_.send_data(payload.data(), payload.size());
      next += std::chrono::seconds(interval_seconds);
      std::unique_lock<std::mutex> lock(mutex_);
      changed_.wait_until(lock, next, [this] { return stopped_.load(); });
    }
  } catch (...) {
    fail();
  }
}
void Heartbeat::receive_loop() noexcept {
  try {
    uint64_t expected = 0;
    while (!stopped_.load()) {
      std::array<unsigned char, 8> payload{};
      in_.recv_data(payload.data(), payload.size());
      uint64_t v = 0;
      for (auto b : payload)
        v = (v << 8) | b;
      if (v != expected++)
        throw std::runtime_error("E_HEARTBEAT_SEQUENCE");
      {
        std::lock_guard<std::mutex> lock(mutex_);
        last_ = std::chrono::steady_clock::now();
      }
      ++received_;
      changed_.notify_all();
    }
  } catch (...) {
    fail();
  }
}
void Heartbeat::watch_loop() noexcept {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!stopped_.load()) {
    auto deadline = last_ + std::chrono::seconds(inactivity_seconds);
    if (std::chrono::steady_clock::now() >= deadline) {
      lock.unlock();
      fail();
      return;
    }
    changed_.wait_until(lock, deadline);
  }
}
} // namespace prag
