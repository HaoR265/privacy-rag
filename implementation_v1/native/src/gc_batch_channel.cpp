#include "prag/gc_batch_channel.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <openssl/crypto.h>
#include <stdexcept>
namespace prag {
GcBatchChannel::~GcBatchChannel() {
  OPENSSL_cleanse(buffer_.data(), buffer_.size());
}
void GcBatchChannel::begin(std::size_t gates) {
  if (active_ || gates == 0 || gates > std::numeric_limits<std::size_t>::max() / 32)
    throw std::runtime_error("E_GC_BATCH_STATE");
  active_ = true;
  expected_ = gates * 32;
  consumed_ = cursor_ = available_ = 0;
}
void GcBatchChannel::send_data(const void *p, int n) {
  if (!active_) {
    wire_.send_data(p, n);
    return;
  }
  if (!sender_ || n != 32 || expected_ - consumed_ < 32)
    throw std::runtime_error("E_GC_BATCH_SHAPE");
  std::memcpy(buffer_.data() + cursor_, p, 32);
  cursor_ += 32;
  consumed_ += 32;
  if (cursor_ == chunk_bytes) {
    wire_.send_data(buffer_.data(), chunk_bytes);
    cursor_ = 0;
  }
}
void GcBatchChannel::recv_data(void *p, int n) {
  if (!active_) {
    wire_.recv_data(p, n);
    return;
  }
  if (sender_ || n != 32 || expected_ - consumed_ < 32)
    throw std::runtime_error("E_GC_BATCH_SHAPE");
  if (cursor_ == available_) {
    wire_.recv_data(buffer_.data(), chunk_bytes);
    available_ = chunk_bytes;
    cursor_ = 0;
  }
  std::memcpy(p, buffer_.data() + cursor_, 32);
  cursor_ += 32;
  consumed_ += 32;
}
void GcBatchChannel::send_tables(const void *p, std::size_t n) {
  if (!active_ || !sender_ || n % 32 || n > expected_ - consumed_ || (n && !p))
    throw std::runtime_error("E_GC_BATCH_SHAPE");
  auto source = static_cast<const unsigned char *>(p);
  while (n) {
    if (cursor_ == 0 && n >= chunk_bytes) {
      consumed_ += chunk_bytes;
      wire_.send_data(source, chunk_bytes);
      source += chunk_bytes; n -= chunk_bytes;
    } else {
      const auto take = std::min(n, chunk_bytes - cursor_);
      std::memcpy(buffer_.data() + cursor_, source, take);
      source += take; n -= take; cursor_ += take; consumed_ += take;
      if (cursor_ == chunk_bytes) {
        wire_.send_data(buffer_.data(), chunk_bytes);
        cursor_ = 0;
      }
    }
  }
}
void GcBatchChannel::recv_tables(void *p, std::size_t n) {
  if (!active_ || sender_ || n % 32 || n > expected_ - consumed_ || (n && !p))
    throw std::runtime_error("E_GC_BATCH_SHAPE");
  auto destination = static_cast<unsigned char *>(p);
  while (n) {
    if (cursor_ == available_ && n >= chunk_bytes) {
      wire_.recv_data(destination, chunk_bytes);
      destination += chunk_bytes; n -= chunk_bytes; consumed_ += chunk_bytes;
      cursor_ = available_ = 0;
    } else {
      if (cursor_ == available_) {
        wire_.recv_data(buffer_.data(), chunk_bytes);
        cursor_ = 0; available_ = chunk_bytes;
      }
      const auto take = std::min(n, available_ - cursor_);
      std::memcpy(destination, buffer_.data() + cursor_, take);
      destination += take; n -= take; cursor_ += take; consumed_ += take;
    }
  }
}
void GcBatchChannel::end() {
  if (!active_ || consumed_ != expected_)
    throw std::runtime_error("E_GC_BATCH_SHAPE");
  if (sender_ && cursor_) {
    std::fill(buffer_.begin() + cursor_, buffer_.end(), 0);
    wire_.send_data(buffer_.data(), chunk_bytes);
  } else if (!sender_) {
    for (std::size_t j = cursor_; j < available_; ++j)
      if (buffer_[j] != 0)
        throw std::runtime_error("E_GC_BATCH_PADDING");
  }
  OPENSSL_cleanse(buffer_.data(), buffer_.size());
  active_ = false;
}
} // namespace prag
