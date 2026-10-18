#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <string>
#include <memory>
namespace prag::model {
using Word=std::uint64_t;
// Receives one participant's shares only. No network, decryption, selection by
// a secret index, or conversion to plaintext belongs in this interface.
struct ShareStorage {
  virtual ~ShareStorage() = default;
  virtual void write(size_t offset, const std::vector<Word>&) = 0;
  virtual std::vector<Word> read(size_t offset, size_t count) = 0;
  virtual void clear() noexcept = 0;
};
struct LocalShareStats {
  uint64_t kernels=0, uploaded_bytes=0, downloaded_bytes=0;
  size_t live_bytes=0, peak_bytes=0;
};
struct LocalShares {
  virtual ~LocalShares() = default;
  virtual std::vector<Word> add(const std::vector<Word>&, const std::vector<Word>&) = 0;
  virtual std::vector<Word> scale(const std::vector<Word>&, Word public_factor) = 0;
  // Public mapping: output[i] = input[indices[i]].
  virtual std::vector<Word> layout(const std::vector<Word>&, const std::vector<size_t>& public_indices) = 0;
  virtual std::vector<Word> sum_rows(const std::vector<Word>&, size_t columns) = 0;
  virtual std::unique_ptr<ShareStorage> allocate(size_t words) = 0;
  virtual LocalShareStats stats() const = 0;
};
std::unique_ptr<LocalShares> make_rtx2060_shares();
}
