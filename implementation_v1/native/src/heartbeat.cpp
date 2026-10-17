#include "prag/heartbeat.hpp"
#include <array>
#include <limits>
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
void Heartbeat::finish() {
  try {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if(finish_requested_||stopped_.load())
        throw std::runtime_error("E_HEARTBEAT_FINISH_STATE");
      finish_requested_=true;
    }
    changed_.notify_all();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      changed_.wait(lock,[this]{return stopped_.load()||(terminal_sent_&&terminal_received_);});
      if(stopped_.load())throw std::runtime_error("E_HEARTBEAT_ABORT");
    }
    // The receiver has consumed an authenticated terminal frame; it has not
    // merely observed EOF. The watchdog remains active through this barrier.
    compute_.set_stage(0x4843);compute_.set_operation(0);
    const std::array<unsigned char,8> terminal={'P','R','A','G','E','N','D','1'};
    std::array<unsigned char,8> peer{};
    compute_.send_data(terminal.data(),terminal.size());
    compute_.recv_data(peer.data(),peer.size());
    if(peer!=terminal)throw std::runtime_error("E_HEARTBEAT_FINISH_FRAME");
    check();
    close(); // All peers have consumed their terminal frame before any close.
    check(); // Also catch a watchdog failure racing the preceding check.
  } catch(...) {fail();close();throw;}
}
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
      bool terminal=false;
      {std::lock_guard<std::mutex> lock(mutex_);terminal=finish_requested_;}
      std::array<unsigned char, 8> payload{};
      if(counter==std::numeric_limits<uint64_t>::max())
        throw std::runtime_error("E_HEARTBEAT_SEQUENCE");
      uint64_t v = terminal?std::numeric_limits<uint64_t>::max():counter++;
      for (int j = 7; j >= 0; --j) {
        payload[j] = v & 255;
        v >>= 8;
      }
      out_.send_data(payload.data(), payload.size());
      if(terminal){
        {std::lock_guard<std::mutex> lock(mutex_);terminal_sent_=true;}
        changed_.notify_all();return;
      }
      next += std::chrono::seconds(interval_seconds);
      std::unique_lock<std::mutex> lock(mutex_);
      changed_.wait_until(lock, next, [this] { return stopped_.load()||finish_requested_; });
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
      if(v==std::numeric_limits<uint64_t>::max()){
        {std::lock_guard<std::mutex> lock(mutex_);
          terminal_received_=true;last_=std::chrono::steady_clock::now();}
        changed_.notify_all();return;
      }
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
