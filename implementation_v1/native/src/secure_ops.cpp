#include "prag/secure_ops.hpp"
#include "OT/split-iknp.h"
#include <algorithm>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <openssl/crypto.h>
#include <openssl/rand.h>
#include <stdexcept>
#include <unistd.h>
namespace prag {
static Word random_word() {
  Word v;
  if (RAND_priv_bytes((unsigned char *)&v, sizeof(v)) != 1)
    throw std::runtime_error("E_RNG");
  return v & MASK;
}
struct SecureOps::Impl {
  Role role;
  Channel &io;
  bool development;
  int ledger = -1;
  uint64_t sequence = 0;
  std::atomic<bool> active{true};
  std::mutex ledger_mutex;
  sci::SplitIKNP<Channel> c_to_s, s_to_c;
  Impl(Role r, Channel &i, const std::string &p, bool d)
      : role(r), io(i), development(d),
        c_to_s(r == Role::Client ? sci::ALICE : sci::BOB, &i),
        s_to_c(r == Role::Server ? sci::ALICE : sci::BOB, &i) {
#ifndef PRAG_ISOLATED_DEVELOPMENT
    if (d)
      throw std::runtime_error("E_DEVELOPMENT_NOT_LINKED");
#endif
    if (!d && !i.authenticated())
      throw std::runtime_error("E_MTLS_REQUIRED");
    ledger = open(p.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
    if (ledger < 0)
      throw std::runtime_error("E_FRESH_LEDGER_REQUIRED");
    try {
      std::string binding = "format=1 session=" + io.session_hex() +
                            " configuration=" + io.configuration_hex() + "\n";
      if (write(ledger, binding.data(), binding.size()) !=
              (ssize_t)binding.size() ||
          fsync(ledger) != 0)
        throw std::runtime_error("E_LEDGER");
      reserve("session", 0);
    } catch (...) {
      // A constructor failure does not run Impl's destructor. Preserve the
      // burned pathname but release the fd; no materials were dispatched yet.
      close(ledger);
      ledger = -1;
      throw;
    }
  }
  ~Impl() {
    terminate(false);
    if (ledger >= 0) {
      fsync(ledger);
      close(ledger);
    }
  }
  void terminate(bool abort) noexcept {
    if (abort)
      io.abort();
    std::lock_guard<std::mutex> lock(ledger_mutex);
    if (!active.exchange(false))
      return;
    constexpr char aborted[] = "terminal=aborted all_reserved=burned\n";
    constexpr char closed[] = "terminal=closed all_reserved=burned\n";
    const char *row = abort ? aborted : closed;
    const size_t length = abort ? sizeof(aborted) - 1 : sizeof(closed) - 1;
    // Even if this final marker cannot be persisted, prior fsynced
    // reservations and O_EXCL admission permanently prevent reuse.
    if (ledger >= 0) {
      (void)write(ledger, row, length);
      (void)fsync(ledger);
    }
  }
  void reserve(const char *kind, size_t n) {
    std::lock_guard<std::mutex> lock(ledger_mutex);
    if (!active.load())
      throw std::runtime_error("E_SESSION_CLOSED");
    io.set_operation(sequence);
    std::string row = std::to_string(sequence++) + " reserved " + kind + " " +
                      std::to_string(n) + " ring59\n";
    if (write(ledger, row.data(), row.size()) != (ssize_t)row.size() ||
        fsync(ledger) != 0)
      throw std::runtime_error("E_LEDGER");
  }
  std::vector<Word> cross(const std::vector<Word> &a, Role sender,
                          unsigned bits = 59) {
    if (a.size() > size_t(std::numeric_limits<int>::max() / bits))
      throw std::runtime_error("E_SHAPE");
    reserve(sender == Role::Client ? "ot_C_to_S" : "ot_S_to_C",
            a.size() * bits);
    auto &ot = sender == Role::Client ? c_to_s : s_to_c;
    size_t n = a.size() * bits;
    std::vector<Word> out(a.size(), 0);
    if (role == sender) {
      std::vector<std::array<Word, 2>> m(n);
      std::vector<Word *> p(n);
      for (size_t i = 0; i < a.size(); ++i)
        for (unsigned j = 0; j < bits; ++j) {
          size_t k = i * bits + j;
          Word r = random_word();
          m[k] = {(-r) & MASK, ((a[i] << j) - r) & MASK};
          p[k] = m[k].data();
          out[i] = (out[i] + r) & MASK;
        }
      // The locked SCI sender emits at most AES_BATCH_SIZE/2 messages per
      // post frame, while its receiver otherwise requests AES_BATCH_SIZE.
      // Bound each invocation so the strict Channel frame shapes coincide.
      for (size_t first = 0; first < n; first += 1024)
        ot.send(p.data() + first, int(std::min(size_t(1024), n - first)), 59);
      OPENSSL_cleanse(m.data(), m.size() * sizeof(m[0]));
    } else {
      std::vector<uint8_t> choices(n);
      std::vector<Word> received(n);
      for (size_t i = 0; i < a.size(); ++i)
        for (unsigned j = 0; j < bits; ++j)
          choices[i * bits + j] = (a[i] >> j) & 1;
      for (size_t first = 0; first < n; first += 1024)
        ot.recv(received.data() + first, choices.data() + first,
                int(std::min(size_t(1024), n - first)), 59);
      for (size_t i = 0; i < a.size(); ++i)
        for (unsigned j = 0; j < bits; ++j)
          out[i] = (out[i] + received[i * bits + j]) & MASK;
      OPENSSL_cleanse(received.data(), received.size() * sizeof(Word));
    }
    return out;
  }
  bool cross_bit(bool a, Role sender) {
    reserve("boolean_cross_OT", 1);
    auto &ot = sender == Role::Client ? c_to_s : s_to_c;
    if (role == sender) {
      uint8_t r = uint8_t(random_word() & 1);
      uint8_t data[2] = {r, uint8_t(r ^ a)};
      uint8_t *p = data;
      ot.send(&p, 1, 1);
      return r;
    }
    uint8_t b = a, out;
    ot.recv(&out, &b, 1, 1);
    return out & 1;
  }
  // Fixed public batches used by the lookup tree. Existing scalar operators
  // keep their protocol schedule; this is the same one-bit cross-product OT.
  std::vector<uint8_t> cross_bits(const std::vector<uint8_t> &a, Role sender) {
    constexpr size_t block_size = 1024;
    std::vector<uint8_t> out(a.size());
    auto &ot = sender == Role::Client ? c_to_s : s_to_c;
    for (size_t first = 0; first < a.size(); first += block_size) {
      const size_t count = std::min(block_size, a.size() - first);
      reserve("boolean_cross_OT", count);
      if (role == sender) {
        std::vector<uint8_t> random(count);
        std::vector<std::array<uint8_t, 2>> messages(count);
        std::vector<uint8_t *> pointers(count);
        if (RAND_priv_bytes(random.data(), int(count)) != 1)
          throw std::runtime_error("E_RNG");
        for (size_t j = 0; j < count; ++j) {
          out[first + j] = random[j] & 1;
          messages[j] = {out[first + j], uint8_t(out[first + j] ^ a[first + j])};
          pointers[j] = messages[j].data();
        }
        ot.send(pointers.data(), int(count), 1);
        OPENSSL_cleanse(random.data(), random.size());
        OPENSSL_cleanse(messages.data(), messages.size() * sizeof(messages[0]));
      } else {
        std::vector<uint8_t> choices(a.begin() + first, a.begin() + first + count);
        ot.recv(out.data() + first, choices.data(), int(count), 1);
        for (size_t j = 0; j < count; ++j)
          out[first + j] &= 1;
        OPENSSL_cleanse(choices.data(), choices.size());
      }
    }
    return out;
  }
  std::vector<uint8_t> and_bits(const std::vector<uint8_t> &a,
                                const std::vector<uint8_t> &b) {
    if (a.empty() || a.size() != b.size())
      throw std::runtime_error("E_SHAPE");
    auto c = cross_bits(role == Role::Client ? a : b, Role::Client);
    auto s = cross_bits(role == Role::Server ? a : b, Role::Server);
    std::vector<uint8_t> out(a.size());
    for (size_t j = 0; j < a.size(); ++j)
      out[j] = (a[j] & b[j]) ^ c[j] ^ s[j];
    OPENSSL_cleanse(c.data(), c.size());
    OPENSSL_cleanse(s.data(), s.size());
    return out;
  }
};
SecureOps::SecureOps(Role r, Channel &i, const std::string &p, bool d)
    : p_(new Impl(r, i, p, d)) {}
SecureOps::~SecureOps() = default;
void SecureOps::reserve_material(const char *kind, size_t n) {
  p_->reserve(kind, n);
}
void SecureOps::abort_session() noexcept { p_->terminate(true); }
void SecureOps::close_session() noexcept { p_->terminate(false); }
bool SecureOps::session_active() const noexcept { return p_->active.load(); }
Role SecureOps::role() const { return p_->role; }
Channel &SecureOps::channel() { return p_->io; }
bool SecureOps::isolated_development() const { return p_->development; }
Word SecureOps::public_word(Word v) const {
  return role() == Role::Server ? v & MASK : 0;
}
Bit SecureOps::public_bit(bool v) const { return role() == Role::Server && v; }
std::vector<Word> SecureOps::mul_many(const std::vector<Word> &x,
                                      const std::vector<Word> &y) {
  if (x.size() != y.size() || x.empty())
    throw std::runtime_error("E_SHAPE");
  size_t n = x.size();
  p_->reserve("beaver", n);
  std::vector<Word> a(n), b(n), t(n), dx(n), dy(n), other(2 * n), wire(2 * n),
      out(n);
  for (size_t i = 0; i < n; ++i) {
    a[i] = random_word();
    b[i] = random_word();
  }
  auto pc = p_->cross(role() == Role::Client ? a : b, Role::Client);
  auto ps = p_->cross(role() == Role::Server ? a : b, Role::Server);
  for (size_t i = 0; i < n; ++i) {
    t[i] = (a[i] * b[i] + pc[i] + ps[i]) & MASK;
    wire[i] = (x[i] - a[i]) & MASK;
    wire[n + i] = (y[i] - b[i]) & MASK;
  }
  // Payload words are full 8-byte little endian on the locked x86_64 profile.
  if (role() == Role::Client) {
    p_->io.send_data(wire.data(), int(16 * n));
    p_->io.recv_data(other.data(), int(16 * n));
  } else {
    p_->io.recv_data(other.data(), int(16 * n));
    p_->io.send_data(wire.data(), int(16 * n));
  }
  for (size_t i = 0; i < n; ++i) {
    dx[i] = (wire[i] + other[i]) & MASK;
    dy[i] = (wire[n + i] + other[n + i]) & MASK;
    out[i] = (t[i] + dx[i] * b[i] + a[i] * dy[i] +
              (role() == Role::Server ? dx[i] * dy[i] : 0)) &
             MASK;
  }
  for (auto *v : {&a, &b, &t, &dx, &dy, &pc, &ps, &wire, &other})
    OPENSSL_cleanse(v->data(), v->size() * sizeof(Word));
  return out;
}
Word SecureOps::mul(Word a, Word b) { return mul_many({a}, {b})[0]; }
std::vector<Word> SecureOps::private_diagonal(const std::vector<Word>& x,
    const std::vector<Word>& weights,unsigned fx,unsigned fw) {
  if(!isolated_development()&&!channel().authenticated())throw std::runtime_error("E_MTLS_REQUIRED");
  if(!session_active())throw std::runtime_error("E_SESSION_CLOSED");
  if(x.empty()||x.size()>8192||fx>58||fw>58||fx+fw>58||
     (role()==Role::Server?weights.size()!=x.size():!weights.empty()))throw std::runtime_error("E_DIAGONAL_SHAPE");
  try {
    std::array<unsigned char,40> descriptor{},other{};
    const uint64_t v[]={0x5052414744494133ULL,3,x.size(),fx,fw};
    for(size_t j=0;j<5;++j)for(size_t b=0;b<8;++b)descriptor[8*j+b]=v[j]>>(56-8*b);
    reserve_material("private_diagonal_descriptor",1);
    if(role()==Role::Client){channel().send_data(descriptor.data(),40);channel().recv_data(other.data(),40);}
    else{channel().recv_data(other.data(),40);if(other!=descriptor)throw std::runtime_error("E_DIAGONAL_DESCRIPTOR");channel().send_data(descriptor.data(),40);}
    if(other!=descriptor)throw std::runtime_error("E_DIAGONAL_DESCRIPTOR");
    auto out=p_->cross(role()==Role::Server?weights:x,Role::Server);
    if(role()==Role::Server)for(size_t i=0;i<x.size();++i)out[i]=(out[i]+x[i]*weights[i])&MASK;
    return out; // Direct asymmetric OT cross term; no Beaver openings/floor.
  }catch(...){abort_session();throw;}
}
std::vector<Word> SecureOps::one_hot_shared(Word index,size_t n) {
  if(!isolated_development()&&!channel().authenticated())throw std::runtime_error("E_MTLS_REQUIRED");
  if(!session_active())throw std::runtime_error("E_SESSION_CLOSED");
  if(n!=155648)throw std::runtime_error("E_TOKEN_VOCABULARY");
  try {
    reserve_material("token_onehot_fixed_155648",1);
    const auto bits=a2b(index);
    Bit high=public_bit(false);for(size_t j=18;j<59;++j)high=bor(high,bits[j]);
    std::vector<uint8_t> nodes{uint8_t(bnot(high))};
    for(size_t level=0;level<18;++level) {
      std::vector<uint8_t> selector(nodes.size(),uint8_t(bits[17-level]));
      auto right=p_->and_bits(nodes,selector);std::vector<uint8_t> next(2*nodes.size());
      for(size_t j=0;j<nodes.size();++j){next[2*j]=nodes[j]^right[j];next[2*j+1]=right[j];}
      OPENSSL_cleanse(nodes.data(),nodes.size());nodes=std::move(next);
    }
    std::vector<Word> binary(n);for(size_t j=0;j<n;++j)binary[j]=nodes[j];
    auto cross=p_->cross(binary,Role::Server,1);
    for(size_t j=0;j<n;++j)binary[j]=(binary[j]-2*cross[j])&MASK;
    OPENSSL_cleanse(nodes.data(),nodes.size());OPENSSL_cleanse(cross.data(),cross.size()*8);
    return binary;
  }catch(...){abort_session();throw;}
}
Bit SecureOps::band(Bit a, Bit b) {
  bool c = p_->cross_bit(role() == Role::Client ? a : b, Role::Client),
       s = p_->cross_bit(role() == Role::Server ? a : b, Role::Server);
  return (a && b) ^ c ^ s;
}
Bits SecureOps::a2b(Word x) {
  Bits out{};
  Bit carry = public_bit(false);
  for (unsigned j = 0; j < 59; ++j) {
    Bit a = role() == Role::Client && ((x >> j) & 1),
        b = role() == Role::Server && ((x >> j) & 1), propagate = bxor(a, b);
    out[j] = bxor(propagate, carry);
    carry = bxor(band(a, b), band(propagate, carry));
  }
  return out;
}
Word SecureOps::b2a_bit(Bit x) {
  auto p = p_->cross({Word(x)}, Role::Client, 1);
  return (Word(x) - 2 * p[0]) & MASK;
}
Word SecureOps::b2a(const Bits &x) {
  std::vector<Word> v(59);
  for (unsigned j = 0; j < 59; ++j)
    v[j] = x[j];
  auto p = p_->cross(v, Role::Client, 1);
  Word out = 0;
  for (unsigned j = 0; j < 59; ++j)
    out = (out + ((v[j] - 2 * p[j]) << j)) & MASK;
  return out;
}
Bit SecureOps::gt(Word a, Word b, bool sign) {
  auto x = a2b(a), y = a2b(b);
  if (sign) {
    x[58] = bnot(x[58]);
    y[58] = bnot(y[58]);
  }
  Bit e = public_bit(true), g = public_bit(false);
  for (int j = 58; j >= 0; --j) {
    Bit d = band(band(e, x[j]), bnot(y[j]));
    g = bxor(g, d);
    e = band(e, bnot(bxor(x[j], y[j])));
  }
  return g;
}
Bit SecureOps::equal(Word a, Word b) {
  auto x = a2b(a), y = a2b(b);
  Bit e = public_bit(true);
  for (unsigned j = 0; j < 59; ++j)
    e = band(e, bnot(bxor(x[j], y[j])));
  return e;
}
Word SecureOps::select(Bit b, Word x, Word y) {
  return add(y, mul(b2a_bit(b), sub(x, y)));
}
Word SecureOps::floor(Word x, unsigned shift) {
  auto b = a2b(x);
  Bits y{};
  for (unsigned j = 0; j < 59; ++j)
    y[j] = shift >= 59 || j + shift >= 59 ? b[58] : b[j + shift];
  return b2a(y);
}
Word SecureOps::lookup(Word i, const std::vector<Word> &table) {
  if (table.empty())
    throw std::runtime_error("E_SHAPE");
  size_t leaves = 1;
  unsigned width = 0;
  while (leaves < table.size()) {
    if (width == 59)
      throw std::runtime_error("E_SHAPE");
    leaves <<= 1;
    ++width;
  }
  auto index = a2b(i);
  // All high bits must be zero. This includes the sign bit and preserves
  // zero for out-of-range ring indices instead of silently reducing modulo L.
  std::vector<uint8_t> valid;
  for (unsigned bit = width; bit < 59; ++bit)
    valid.push_back(bnot(index[bit]));
  while (valid.size() > 1) {
    std::vector<uint8_t> a(valid.size() / 2), b(a.size());
    for (size_t j = 0; j < a.size(); ++j) {
      a[j] = valid[2 * j];
      b[j] = valid[2 * j + 1];
    }
    auto next = p_->and_bits(a, b);
    if (valid.size() & 1)
      next.push_back(valid.back());
    OPENSSL_cleanse(valid.data(), valid.size());
    valid = std::move(next);
  }
  std::vector<uint8_t> selectors{
      valid.empty() ? uint8_t(public_bit(true)) : valid[0]};
  for (unsigned bit = width; bit-- > 0;) {
    std::vector<uint8_t> repeated(selectors.size(), index[bit]);
    auto right = p_->and_bits(selectors, repeated);
    std::vector<uint8_t> next(2 * selectors.size());
    for (size_t j = 0; j < selectors.size(); ++j) {
      next[2 * j] = selectors[j] ^ right[j];
      next[2 * j + 1] = right[j];
    }
    OPENSSL_cleanse(selectors.data(), selectors.size());
    OPENSSL_cleanse(right.data(), right.size());
    selectors = std::move(next);
  }
  // Batched Boolean-to-arithmetic conversion of the public table's leaves.
  // Padding leaves have no contribution, so non-power-of-two lengths also
  // return zero for every index >= table.size().
  Word out = 0;
  constexpr size_t block_size = 1024;
  for (size_t first = 0; first < table.size(); first += block_size) {
    const size_t count = std::min(block_size, table.size() - first);
    std::vector<Word> local(count);
    for (size_t j = 0; j < count; ++j)
      local[j] = selectors[first + j];
    auto cross = p_->cross(local, Role::Client, 1);
    for (size_t j = 0; j < count; ++j)
      out = add(out, mul_public((local[j] - 2 * cross[j]) & MASK, table[first + j]));
    OPENSSL_cleanse(local.data(), local.size() * sizeof(Word));
    OPENSSL_cleanse(cross.data(), cross.size() * sizeof(Word));
  }
  OPENSSL_cleanse(index.data(), sizeof(index));
  OPENSSL_cleanse(selectors.data(), selectors.size());
  OPENSSL_cleanse(valid.data(), valid.size());
  return out;
}
} // namespace prag
