#pragma once
#include "prag/secure_ops.hpp"
#include <algorithm>
#include <openssl/crypto.h>
#include <openssl/rand.h>
#include <stdexcept>
// Not part of prag_crypto or the worker. This investigates a different matrix
// triple preparation schedule; production BatchOps::matrix remains OT based.
namespace prag::candidate {
inline std::vector<Word> matrix(SecureOps &s, const std::vector<Word> &x,
                                const std::vector<Word> &y, size_t rows,
                                size_t inner, size_t columns) {
  try {
    if (!rows || rows > 16 || !inner || inner > 1280 || !columns ||
        columns > 1280 || x.size() != rows * inner ||
        y.size() != inner * columns)
      throw std::runtime_error("CANDIDATE_SHAPE");
    std::array<Word, 4> descriptor{11001, rows, inner, columns}, peer{};
    s.reserve_material("candidate_BFV_matrix_triple", 1);
    auto &io = s.channel();
    if (s.role() == Role::Client) {
      io.send_data(descriptor.data(), 32);
      io.recv_data(peer.data(), 32);
    } else {
      io.recv_data(peer.data(), 32);
      io.send_data(descriptor.data(), 32);
    }
    if (peer != descriptor)
      throw std::runtime_error("CANDIDATE_DESCRIPTOR");
    std::vector<Word> a(x.size()), b(y.size()), t(rows * columns),
        wire(a.size() + b.size()), other(wire.size()), dx(a.size()),
        dy(b.size());
    s.reserve_material("candidate_matrix_A_random_words", a.size());
    s.reserve_material("candidate_matrix_B_random_words", b.size());
    if (RAND_priv_bytes(reinterpret_cast<unsigned char *>(a.data()),
                        int(a.size() * 8)) != 1 ||
        RAND_priv_bytes(reinterpret_cast<unsigned char *>(b.data()),
                        int(b.size() * 8)) != 1)
      throw std::runtime_error("E_RNG");
    for (auto &v : a)
      v &= MASK;
    for (auto &v : b)
      v &= MASK;
    std::vector<std::vector<Word>> ac(rows, std::vector<Word>(inner));
    if (s.role() == Role::Client)
      for (size_t r = 0; r < rows; ++r)
        std::copy(a.begin() + r * inner, a.begin() + (r + 1) * inner,
                  ac[r].begin());
    auto cross1 = s.private_matrix(
        ac, columns, 0, 0,
        [&](size_t first, size_t width, size_t row, size_t count) {
          std::vector<std::vector<Word>> w(count, std::vector<Word>(width));
          for (size_t j = 0; j < count; ++j)
            for (size_t k = 0; k < width; ++k)
              w[j][k] = b[(first + k) * columns + row + j];
          return w;
        });
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < columns; ++c)
        t[r * columns + c] = cross1[r][c];
    // C encrypts its B share transposed; S privately applies its A share.
    // The key still belongs to C. S's input share is zero in both calls.
    for (size_t first = 0; first < columns; first += 16) {
      size_t count = std::min(size_t(16), columns - first);
      std::vector<std::vector<Word>> bc(count, std::vector<Word>(inner));
      if (s.role() == Role::Client)
        for (size_t c = 0; c < count; ++c)
          for (size_t k = 0; k < inner; ++k)
            bc[c][k] = b[k * columns + first + c];
      auto cross2 = s.private_matrix(
          bc, rows, 0, 0, [&](size_t kb, size_t width, size_t row, size_t nr) {
            std::vector<std::vector<Word>> w(nr, std::vector<Word>(width));
            for (size_t r = 0; r < nr; ++r)
              for (size_t k = 0; k < width; ++k)
                w[r][k] = a[(row + r) * inner + kb + k];
            return w;
          });
      for (size_t c = 0; c < count; ++c)
        for (size_t r = 0; r < rows; ++r)
          t[r * columns + first + c] =
              (t[r * columns + first + c] + cross2[c][r]) & MASK;
    }
    for (size_t r = 0; r < rows; ++r)
      for (size_t k = 0; k < inner; ++k)
        for (size_t c = 0; c < columns; ++c)
          t[r * columns + c] =
              (t[r * columns + c] + a[r * inner + k] * b[k * columns + c]) &
              MASK;
    for (size_t j = 0; j < a.size(); ++j)
      wire[j] = (x[j] - a[j]) & MASK;
    for (size_t j = 0; j < b.size(); ++j)
      wire[a.size() + j] = (y[j] - b[j]) & MASK;
    s.reserve_material("candidate_matrix_masked_opening", wire.size());
    if (s.role() == Role::Client) {
      io.send_data(wire.data(), int(wire.size() * 8));
      io.recv_data(other.data(), int(other.size() * 8));
    } else {
      io.recv_data(other.data(), int(other.size() * 8));
      io.send_data(wire.data(), int(wire.size() * 8));
    }
    for (size_t j = 0; j < a.size(); ++j)
      dx[j] = (wire[j] + other[j]) & MASK;
    for (size_t j = 0; j < b.size(); ++j)
      dy[j] = (wire[a.size() + j] + other[a.size() + j]) & MASK;
    for (size_t r = 0; r < rows; ++r)
      for (size_t k = 0; k < inner; ++k)
        for (size_t c = 0; c < columns; ++c) {
          size_t i = r * columns + c;
          Word term = dx[r * inner + k] * b[k * columns + c] +
                      a[r * inner + k] * dy[k * columns + c];
          if (s.role() == Role::Server)
            term += dx[r * inner + k] * dy[k * columns + c];
          t[i] = (t[i] + term) & MASK;
        }
    for (auto *values : {&a, &b, &dx, &dy, &wire, &other})
      OPENSSL_cleanse(values->data(), values->size() * 8);
    return t;
  } catch (...) {
    s.abort_session();
    throw;
  }
}
} // namespace prag::candidate
