// Isolated candidate: existing MatrixWords/Rows/Phase helpers precede this.
std::vector<Word> BatchOps::logical_matrix(
    const std::vector<Word> &x, const std::vector<Word> &y, size_t rows,
    size_t inner, size_t columns, unsigned fx, unsigned fy, Word bound,
    const std::array<Word,5>& identity,
    const std::function<void(const char *, bool)> &phase) {
  try {
    if (!rows || rows > 1024 || !inner || inner > 1280 || !columns ||
        columns > 1280 || x.size() != rows * inner ||
        y.size() != inner * columns || fx > 58 || fy > 58 || fx + fy > 58 ||
        bound >= (Word{1} << 56))
      throw std::runtime_error("E_SHARED_MATRIX_SHAPE");
    for (auto v : x) if (v > MASK) throw std::runtime_error("E_BATCH_WORD");
    for (auto v : y) if (v > MASK) throw std::runtime_error("E_BATCH_WORD");
    if(!identity[0]||identity[0]<=p_->last_logical_matrix_sequence||
       identity[1]>=28||identity[2]>=16||identity[3]>256||
       (identity[4]!=1&&identity[4]!=2)||
       (identity[4]==1&&(inner!=128||fx!=12||fy!=12))||
       (identity[4]==2&&(columns!=128||fx!=24||fy!=12)))
      throw std::runtime_error("E_LOGICAL_MATRIX_IDENTITY");
    // Consume the public identity before any material or diagnostic callback.
    // Failure aborts the whole session; no retry can recover this context.
    p_->last_logical_matrix_sequence=identity[0];
    p_->descriptor(1100, rows, columns, {inner, Word(fx * 64 + fy), bound});
    p_->descriptor(1101,identity[0],identity[1],{identity[2],identity[3],identity[4]});
    p_->descriptor(1102,16,16,{(rows+15)/16,(columns+15)/16,1});
    auto &s = p_->op;
    auto &io = s.channel();
    MatrixWords a(x.size()), b(y.size()), t(rows * columns),
        wire(x.size() + y.size()), peer(wire.values.size()),
        dx(x.size()), dy(y.size());
    {
      MatrixPhase meter(phase, "matrix_triple_randomness");
      std::string material="BFV_logical_matrix_beaver";
      for(auto id:identity)material+="_"+std::to_string(id);
      material+="_"+std::to_string(rows)+"x"+std::to_string(inner)+"x"+std::to_string(columns);
      s.reserve_material(material.c_str(),1);
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
      for(size_t row0=0;row0<rows;row0+=16) {
        const size_t nr=std::min(size_t(16),rows-row0);
        MatrixRows ac(nr, inner), cross;
        if(s.role()==Role::Client)
          for(size_t r=0;r<nr;++r)
            std::copy_n(a.values.begin()+(row0+r)*inner,inner,ac.values[r].begin());
        cross.values=s.private_matrix(ac.values,columns,0,0,
          [&](size_t kb,size_t width,size_t first,size_t count){
            std::vector<std::vector<Word>> weights(count,std::vector<Word>(width));
            for(size_t c=0;c<count;++c)for(size_t k=0;k<width;++k)
              weights[c][k]=b.values[(kb+k)*columns+first+c];
            return weights;
          });
        for(size_t r=0;r<nr;++r)
          std::copy(cross.values[r].begin(),cross.values[r].end(),t.values.begin()+(row0+r)*columns);
      }
    }
    {
      MatrixPhase meter(phase, "matrix_cross_A_S_B_C");
      // C encrypts B_C^T; S supplies A_S. C remains the BFV key owner.
      for (size_t first = 0; first < columns; first += 16) {
        size_t count = std::min(size_t(16), columns - first);
        MatrixRows bc(count, inner), cross;
        if (s.role() == Role::Client)
          for (size_t c = 0; c < count; ++c)
            for (size_t k = 0; k < inner; ++k)
              bc.values[c][k] = b.values[k * columns + first + c];
        cross.values = s.private_matrix(bc.values, rows, 0, 0,
            [&](size_t kb, size_t width, size_t r0, size_t nr) {
              std::vector<std::vector<Word>> weights(nr, std::vector<Word>(width));
              for (size_t r = 0; r < nr; ++r)
                for (size_t k = 0; k < width; ++k)
                  weights[r][k] = a.values[(r0 + r) * inner + kb + k];
              return weights;
            });
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
