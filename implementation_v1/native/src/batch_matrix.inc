// Included in namespace prag after BatchOps::Impl. Both operands stay shared.
namespace {
struct MatrixWords {
  std::vector<Word> values;
  explicit MatrixWords(size_t n = 0) : values(n) {}
  ~MatrixWords() { OPENSSL_cleanse(values.data(), values.size() * sizeof(Word)); }
  MatrixWords(const MatrixWords &) = delete;
};
struct MatrixRows {
  std::vector<std::vector<Word>> values;
  MatrixRows(size_t rows = 0, size_t cols = 0)
      : values(rows, std::vector<Word>(cols)) {}
  ~MatrixRows() {
    for (auto &row : values)
      OPENSSL_cleanse(row.data(), row.size() * sizeof(Word));
  }
  MatrixRows(const MatrixRows &) = delete;
};
struct MatrixPhase {
  const std::function<void(const char *, bool)> &callback;
  const char *name;
  const int exceptions_on_entry = std::uncaught_exceptions();
  MatrixPhase(const std::function<void(const char *, bool)> &f, const char *n)
      : callback(f), name(n) { if (callback) callback(name, true); }
  ~MatrixPhase() noexcept(false) {
    if (!callback) return;
    if (std::uncaught_exceptions() > exceptions_on_entry) {
      // Preserve the protocol exception so the outer catch aborts and burns.
      // A diagnostic callback must not terminate an already-unwinding process.
      try { callback(name, false); } catch (...) {}
    } else {
      // Normal-exit callback failures also reach matrix()'s abort-and-burn path.
      callback(name, false);
    }
  }
};
}
std::vector<Word> BatchOps::matrix(
    const std::vector<Word> &x, const std::vector<Word> &y, size_t rows,
    size_t inner, size_t columns, unsigned fx, unsigned fy, Word bound,
    const std::function<void(const char *, bool)> &phase) {
  try {
    if (!rows || rows > 16 || !inner || inner > 1280 || !columns ||
        columns > 1280 || x.size() != rows * inner ||
        y.size() != inner * columns || fx > 58 || fy > 58 || fx + fy > 58 ||
        bound >= (Word{1} << 56))
      throw std::runtime_error("E_SHARED_MATRIX_SHAPE");
    for (auto v : x) if (v > MASK) throw std::runtime_error("E_BATCH_WORD");
    for (auto v : y) if (v > MASK) throw std::runtime_error("E_BATCH_WORD");
    // 102 identifies matrix-triple preparation, distinct from v110 scalar OT.
    const bool qk_reverse64=(rows==1 && inner==128);
    p_->descriptor(qk_reverse64?103:102, rows, columns, {inner, Word(fx * 64 + fy), bound});
    auto &s = p_->op;
    auto &io = s.channel();
    MatrixWords a(x.size()), b(y.size()), t(rows * columns),
        wire(x.size() + y.size()), peer(wire.values.size()),
        dx(x.size()), dy(y.size());
    {
      MatrixPhase meter(phase, "matrix_triple_randomness");
      s.reserve_material("BFV_matrix_beaver", 1);
      s.reserve_material("matrix_A_random_words", a.values.size());
      s.reserve_material("matrix_B_random_words", b.values.size());
      for (auto *v : {&a.values, &b.values}) {
        if (RAND_priv_bytes(reinterpret_cast<unsigned char *>(v->data()), int(v->size() * 8)) != 1)
          throw std::runtime_error("E_RNG");
        for (auto &w : *v) w &= MASK;
      }
    }
    {
      MatrixPhase meter(phase, "matrix_cross_A_C_B_S");
      MatrixRows ac(rows, inner), cross;
      if (s.role() == Role::Client)
        for (size_t r = 0; r < rows; ++r)
          std::copy(a.values.begin() + r * inner, a.values.begin() + (r + 1) * inner, ac.values[r].begin());
      cross.values = s.private_matrix(ac.values, columns, 0, 0,
          [&](size_t kb, size_t width, size_t first, size_t count) {
            std::vector<std::vector<Word>> weights(count, std::vector<Word>(width));
            for (size_t c = 0; c < count; ++c)
              for (size_t k = 0; k < width; ++k)
                weights[c][k] = b.values[(kb + k) * columns + first + c];
            return weights;
          });
      for (size_t r = 0; r < rows; ++r)
        std::copy(cross.values[r].begin(), cross.values[r].end(), t.values.begin() + r * columns);
    }
    {
      MatrixPhase meter(phase, "matrix_cross_A_S_B_C");
      // C encrypts B_C^T; S supplies A_S. C remains the BFV key owner.
      const size_t reverse_rows=qk_reverse64?64:16;
      for (size_t first = 0; first < columns; first += reverse_rows) {
        size_t count = std::min(reverse_rows, columns - first);
        MatrixRows bc(count, inner), cross;
        if (s.role() == Role::Client)
          for (size_t c = 0; c < count; ++c)
            for (size_t k = 0; k < inner; ++k)
              bc.values[c][k] = b.values[k * columns + first + c];
        PrivateWeightBlock reverse_reader=
            [&](size_t kb, size_t width, size_t r0, size_t nr) {
              std::vector<std::vector<Word>> weights(nr, std::vector<Word>(width));
              for (size_t r = 0; r < nr; ++r)
                for (size_t k = 0; k < width; ++k)
                  weights[r][k] = a.values[(r0 + r) * inner + kb + k];
              return weights;
            };
      cross.values=qk_reverse64?s.private_qk_reverse64(bc.values,reverse_reader):s.private_matrix(bc.values,rows,0,0,reverse_reader);
      for (size_t c = 0; c < count; ++c)
          for (size_t r = 0; r < rows; ++r)
            t.values[r * columns + first + c] =
                (t.values[r * columns + first + c] + cross.values[c][r]) & MASK;
      }
    }
    {
      MatrixPhase meter(phase, "matrix_local_preprocessing");
      for (size_t r = 0; r < rows; ++r)
        for (size_t k = 0; k < inner; ++k)
          for (size_t c = 0; c < columns; ++c)
            t.values[r * columns + c] = (t.values[r * columns + c] +
                a.values[r * inner + k] * b.values[k * columns + c]) & MASK;
    }
    {
      MatrixPhase meter(phase, "matrix_masked_opening");
      s.reserve_material("matrix_masked_opening_words", wire.values.size());
      for (size_t j = 0; j < x.size(); ++j) wire.values[j] = (x[j] - a.values[j]) & MASK;
      for (size_t j = 0; j < y.size(); ++j) wire.values[x.size() + j] = (y[j] - b.values[j]) & MASK;
      // Locked x86-64 ABI: fixed-width LE ring words; public size is descriptor-bound.
      if (s.role() == Role::Client) {
        io.send_data(wire.values.data(), int(wire.values.size() * 8));
        io.recv_data(peer.values.data(), int(peer.values.size() * 8));
      } else {
        io.recv_data(peer.values.data(), int(peer.values.size() * 8));
        io.send_data(wire.values.data(), int(wire.values.size() * 8));
      }
      for (auto v : peer.values) if (v > MASK) throw std::runtime_error("E_BATCH_WORD");
      for (size_t j = 0; j < x.size(); ++j) dx.values[j] = (wire.values[j] + peer.values[j]) & MASK;
      for (size_t j = 0; j < y.size(); ++j) dy.values[j] = (wire.values[x.size()+j] + peer.values[x.size()+j]) & MASK;
    }
    {
      MatrixPhase meter(phase, "matrix_online_local");
      for (size_t r = 0; r < rows; ++r)
        for (size_t k = 0; k < inner; ++k)
          for (size_t c = 0; c < columns; ++c) {
            Word term = dx.values[r * inner + k] * b.values[k * columns + c] +
                a.values[r * inner + k] * dy.values[k * columns + c];
            if (s.role() == Role::Server)
              term += dx.values[r * inner + k] * dy.values[k * columns + c];
            t.values[r * columns + c] = (t.values[r * columns + c] + term) & MASK;
          }
    }
    // Unsigned mod 2^64 intermediates reduce exactly into the divisor ring 2^59.
    return std::move(t.values);
  } catch (...) {
    p_->op.abort_session();
    throw;
  }
}
