#pragma once
#include "prag/channel.hpp"
#include <array>
namespace prag {
// Public fixed-shape buffering for half-gate tables only. Base OT, private
// input labels, and output labels remain on their existing bound Channel.
class GcBatchChannel : public sci::IOChannel<GcBatchChannel> {
  Channel &wire_;
  bool sender_, active_ = false;
  std::size_t expected_ = 0, consumed_ = 0, cursor_ = 0, available_ = 0;
  std::array<unsigned char, 65536> buffer_{};

public:
  static constexpr std::size_t chunk_bytes = 65536;
  GcBatchChannel(Channel &wire, bool sender) : wire_(wire), sender_(sender) {}
  ~GcBatchChannel();
  GcBatchChannel(const GcBatchChannel &) = delete;
  void begin(std::size_t public_and_gates);
  void end();
  void send_data(const void *, int);
  void recv_data(void *, int);
};
} // namespace prag
