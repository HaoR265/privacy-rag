#include "prag/bfv.hpp"
#include "layout.hpp"
#include "GC/halfgate_eva.h"
#include "GC/halfgate_gen.h"
#include "OT/iknp.h"
#include "prag/gc_batch_channel.hpp"
#include "prag/secure_ops.hpp"
#include <fstream>
#include <openssl/crypto.h>
#include <openssl/rand.h>
#include <seal/util/ntt.h>
#include <seal/util/polyarithsmallmod.h>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#if SEAL_VERSION_MAJOR != 4 || SEAL_VERSION_MINOR != 1 ||                      \
    SEAL_VERSION_PATCH != 2
#error "ART-002 requires the locked SEAL 4.1.2 implementation"
#endif
#ifdef SEAL_USE_GAUSSIAN_NOISE
#error "ART-002 requires the bounded CBD sampler"
#endif
#ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
#error "The fixed zero-weight schedule requires transparent intermediates"
#endif
#ifdef SEAL_USE_INTEL_HEXL
#error "The locked native profile was materialized without Intel HEXL"
#endif
namespace prag {
static_assert(std::is_same<seal::SEAL_DEFAULT_PRNG_FACTORY,
                           seal::Blake2xbPRNGFactory>::value,
              "The locked native profile uses Blake2xb");
using Label = sci::block128;
using Wires = std::vector<Label>;
seal::EncryptionParameters locked_bfv_parameters() {
  seal::EncryptionParameters p(seal::scheme_type::bfv);
  p.set_poly_modulus_degree(16384);
  p.set_coeff_modulus(
      seal::CoeffModulus::Create(16384, {60, 60, 60, 60, 60, 60, 60}));
  p.set_plain_modulus(uint64_t(1) << 59);
  constexpr uint64_t expected[] = {
      1152921504599080961ULL, 1152921504600260609ULL, 1152921504604979201ULL,
      1152921504605962241ULL, 1152921504606584833ULL, 1152921504606683137ULL,
      1152921504606748673ULL};
  for (size_t j = 0; j < 7; ++j)
    if (p.coeff_modulus()[j].value() != expected[j])
      throw std::runtime_error("E_ART002_PARAMETER_PROFILE");
  return p;
}
mpz_class modulus_product(const std::vector<seal::Modulus> &p) {
  mpz_class q = 1;
  for (auto &m : p)
    q *= m.value();
  return q;
}
std::size_t packed_bfv_rows_per_ciphertext(std::size_t d) {
  if (d == 0 || d > 8192)
    throw std::runtime_error("E_BFV_PACKED_SHAPE");
  return std::min(std::size_t(16384) / (2 * d - 1), std::size_t(8192) / d);
}
static void verify_art002_context(const seal::SEALContext &ctx) {
  auto key = ctx.key_context_data(), data = ctx.first_context_data();
  if (!ctx.parameters_set() || !key || !data ||
      key->parms().coeff_modulus().size() != 7 ||
      data->parms().coeff_modulus().size() != 6 ||
      data->parms().poly_modulus_degree() != 16384 ||
      data->parms().plain_modulus().value() != (uint64_t(1) << 59) ||
      data->qualifiers().sec_level != seal::sec_level_type::tc128)
    throw std::runtime_error("E_ART002_CONTEXT_PROFILE");
  auto locked = locked_bfv_parameters();
  for (size_t j = 0; j < 6; ++j)
    if (data->parms().coeff_modulus()[j] != locked.coeff_modulus()[j])
      throw std::runtime_error("E_ART002_CONTEXT_PROFILE");
  const mpz_class q = modulus_product(data->parms().coeff_modulus());
  const mpz_class t = mpz_class(1) << 59;
  // Legacy vector calls have <=19 products; matrix tiles have <=512.
  // Each weight polynomial still has <=8192 centered nonzero coefficients.
  const mpz_class bound = 8194 * (512 * (mpz_class(1) << 71) + 1);
  if (4 * t * bound >= q)
    throw std::runtime_error("E_ART002_DECODE_BOUND");
}
mpz_class rns_coefficient(const uint64_t *a, size_t n, size_t index,
                          const std::vector<seal::Modulus> &primes) {
  mpz_class q = modulus_product(primes), out = 0;
  for (size_t j = 0; j < primes.size(); ++j) {
    mpz_class p = primes[j].value(), part = q / p, inverse;
    mpz_invert(inverse.get_mpz_t(), part.get_mpz_t(), p.get_mpz_t());
    out += mpz_class(a[j * n + index]) * part * inverse;
  }
  return out % q;
}
static std::string id_string(const seal::parms_id_type &id) {
  std::ostringstream s;
  for (size_t i = 0; i < 4; ++i) {
    if (i)
      s << ",";
    s << '"' << id[i] << '"';
  }
  return s.str();
}
void write_bfv_context_artifact(const std::string &path) {
  seal::SEALContext ctx(locked_bfv_parameters(), true,
                        seal::sec_level_type::tc128);
  verify_art002_context(ctx);
  std::ofstream f(path);
  f << "{\n\"artifact\":\"ART-002\",\"seal_version\":\"4.1.2\",\"security\":"
       "\"tc128\",\"polynomial_degree\":16384,\"plaintext_modulus\":"
       "\"576460752303423488\",\"context_parameters_set\":true,\"production_"
       "enabled\":false,\"primitive_profile_verified\":true,\"session_noise_"
       "union_bound\":0,\"levels\":[\n";
  bool first = true;
  for (auto c = ctx.key_context_data(); c; c = c->next_context_data()) {
    if (!first)
      f << ",\n";
    first = false;
    auto &qpr = c->parms().coeff_modulus();
    auto q = modulus_product(qpr);
    size_t b = mpz_sizeinbase(q.get_mpz_t(), 2);
    f << "{\"chain_index\":" << c->chain_index()
      << ",\"key_level\":" << (c == ctx.key_context_data() ? "true" : "false")
      << ",\"parms_id_u64\":[" << id_string(c->parms_id()) << "],\"primes\":[";
    for (size_t i = 0; i < qpr.size(); ++i) {
      if (i)
        f << ",";
      f << '"' << qpr[i].value() << '"';
    }
    f << "],\"Q\":\"" << q.get_str() << "\",\"Q_bits\":" << b
      << ",\"phase_add_bits\":" << b + 1 << ",\"numerator_bits\":" << b + 60
      << ",\"division_remainder_bits\":" << b + 1 << "}";
  }
  f << "\n]}\n";
  if (!f)
    throw std::runtime_error("E_ARTIFACT_WRITE");
}
class Decoder {
  Role role;
  Channel &io;
  sci::IKNP<Channel> ot;
  GcBatchChannel gate_io;
  std::unique_ptr<sci::CircuitExecution> c;
  sci::HalfGateGen<GcBatchChannel> *generator = nullptr;
  Label zero() { return c->public_label(false); }
  Label one() { return c->public_label(true); }
  Label X(Label a, Label b) { return c->xor_gate(a, b); }
  Label A(Label a, Label b) { return c->and_gate(a, b); }
  Label N(Label a) { return c->not_gate(a); }
  Label M(Label b, Label x, Label y) { return X(y, A(b, X(x, y))); }
  Wires constant(const mpz_class &v, size_t width) {
    Wires w(width);
    for (size_t i = 0; i < width; ++i)
      w[i] = mpz_tstbit(v.get_mpz_t(), i) ? one() : zero();
    return w;
  }
  Wires input(const mpz_class &v, size_t width, Role owner) {
    Wires w(width);
    if (role == Role::Server) {
      Wires alternate(width), selected(width);
      for (size_t j = 0; j < width; ++j) {
        sci::PRG128().random_block(&w[j], 1);
        alternate[j] = w[j] ^ generator->delta;
        selected[j] = mpz_tstbit(v.get_mpz_t(), j) ? alternate[j] : w[j];
      }
      if (owner == Role::Server)
        io.send_block(selected.data(), int(width));
      else
        ot.send(w.data(), alternate.data(), int(width));
    } else if (owner == Role::Server)
      io.recv_block(w.data(), int(width));
    else {
      std::unique_ptr<bool[]> b(new bool[width]);
      for (size_t j = 0; j < width; ++j)
        b[j] = mpz_tstbit(v.get_mpz_t(), j);
      ot.recv(w.data(), b.get(), int(width));
    }
    return w;
  }
  Wires add(const Wires &a, const Wires &b) {
    Wires o(a.size());
    Label carry = zero();
    for (size_t j = 0; j < a.size(); ++j) {
      auto p = X(a[j], b[j]);
      o[j] = X(p, carry);
      carry = X(A(a[j], b[j]), A(p, carry));
    }
    return o;
  }
  // Unsigned fixed-width subtraction; output no_borrow secretly selects result.
  Wires sub(const Wires &a, const Wires &b, Label &no_borrow) {
    Wires o(a.size());
    Label carry = one();
    for (size_t j = 0; j < a.size(); ++j) {
      auto nb = N(b[j]), p = X(a[j], nb);
      o[j] = X(p, carry);
      carry = X(A(a[j], nb), A(p, carry));
    }
    no_borrow = carry;
    return o;
  }

  // Private to the locked BFV decode construction below, not a general
  // numerator/divisor API. Its caller establishes N=v*2^59+floor(q/2),
  // 0<=v<q and the locked 360-bit q. Thus N<q*2^60, so the high 360
  // quotient bits are public zero and N>>60 is already a remainder <q.
  Wires divide_locked_bfv59_numerator(const Wires &numerator,
                                     const Wires &qc) {
    constexpr size_t width = 361, nw = 420, rounds = 60;
    if (numerator.size() != nw || qc.size() != width)
      throw std::runtime_error("E_BFV_DECODE_WIRE_LAYOUT");
    Wires remainder(width, zero()), quotient(nw, zero());
    for (size_t j = 0; j < nw - rounds; ++j)
      remainder[j] = numerator[j + rounds];
    // Keep bit 59: at v=q-1 the full quotient can equal t=2^59, whose
    // low 59 bits must be zero. Every shifted remainder is <2q and fits
    // the unchanged 361-bit representation before one subtraction.
    for (size_t step = rounds; step-- > 0;) {
      for (size_t j = width - 1; j > 0; --j)
        remainder[j] = remainder[j - 1];
      remainder[0] = numerator[step];
      Label greater;
      auto diff = sub(remainder, qc, greater);
      for (size_t j = 0; j < width; ++j)
        remainder[j] = M(greater, diff[j], remainder[j]);
      quotient[step] = greater;
    }
    return quotient;
  }

public:
  Decoder(Role r, Channel &i)
      : role(r), io(i), ot(&i), gate_io(i, r == Role::Server) {}
  #include "bfv_tile_decoder.inc"
#ifdef PRAG_ISOLATED_DEVELOPMENT
  Word decode_locked_bfv59(const mpz_class &local, const mpz_class &q) {
    static const mpz_class locked_q = [] {
      auto primes = locked_bfv_parameters().coeff_modulus();
      primes.pop_back(); // first data level, never the seven-prime key level
      return modulus_product(primes);
    }();
    if (q != locked_q || mpz_sizeinbase(q.get_mpz_t(), 2) != 360 ||
        local < 0 || local >= q)
      throw std::runtime_error("E_BFV_DECODE_CANONICAL_PROFILE");
    // Every target has fresh delta, public/input labels and hash state. The
    // independent IKNP stream advances across disjoint batches; never reset it.
    c.reset();
    generator = nullptr;
    if (role == Role::Server) {
      auto *g = new sci::HalfGateGen<GcBatchChannel>(&gate_io);
      generator = g;
      c.reset(g);
    } else {
      c.reset(new sci::HalfGateEva<GcBatchChannel>(&gate_io));
    }
    constexpr size_t b = 360, width = 361, nw = 420, rounds = 60;
    auto u = input(local, width, Role::Client),
         c0 = input(local, width, Role::Server);
    // Only the public-zero quotient prefix is elided. Phase/numerator/
    // remainder widths, exact rounding and low-59 output wires are unchanged.
    gate_io.begin(5 * width + 2 * nw + 3 * width * rounds);
    auto phase = add(u, c0), qc = constant(q, width);
    Label greater;
    auto reduced = sub(phase, qc, greater);
    for (size_t j = 0; j < width; ++j)
      phase[j] = M(greater, reduced[j], phase[j]);
    Wires numerator(nw, zero());
    for (size_t j = 0; j < b; ++j)
      numerator[j + 59] = phase[j];
    numerator = add(numerator, constant(q / 2, nw));
    auto quotient = divide_locked_bfv59_numerator(numerator, qc);
    gate_io.end();
    uint8_t permutation[59];
    Word out = 0;
    if (role == Role::Server) {
      for (size_t j = 0; j < 59; ++j)
        permutation[j] = sci::getLSB(quotient[j]);
      io.send_data(permutation, 59);
    } else {
      io.recv_data(permutation, 59);
      for (size_t j = 0; j < 59; ++j)
        out |= Word(bool(sci::getLSB(quotient[j])) ^ bool(permutation[j])) << j;
    }
    return out;
  }
#endif
};
#ifdef PRAG_ISOLATED_DEVELOPMENT
// Synthetic phase diagnostics only; this symbol is absent from prag_crypto.
std::vector<Word>
development_bfv59_decode(SecureOps &op,
                         const std::vector<mpz_class> &local_components) {
  if (!op.isolated_development() || !op.session_active())
    throw std::runtime_error("E_DEVELOPMENT_ONLY");
  auto primes = locked_bfv_parameters().coeff_modulus();
  primes.pop_back();
  const mpz_class q = modulus_product(primes);
  op.reserve_material("development_BFV_decoder_independent_OT", 1);
  op.reserve_material("development_BFV_decoder_GC", local_components.size());
  Decoder decoder(op.role(), op.channel());
  std::vector<Word> out;
  for (const auto &local : local_components)
    out.push_back(decoder.decode_locked_bfv59(local, q));
  return out;
}
#endif
class BfvEngine {
  Role role;
  Channel &io;
  seal::SEALContext context;
  seal::PublicKey pk;
  seal::SecretKey sk;
  std::unique_ptr<seal::Encryptor> enc;
  Decoder decoder;

public:
  BfvEngine(Role r, Channel &i)
      : role(r), io(i),
        context(locked_bfv_parameters(), true, seal::sec_level_type::tc128),
        decoder(r, i) {
    verify_art002_context(context);
    size_t count = 16384 * 7 * 2;
    if (role == Role::Client) {
      seal::KeyGenerator kg(context);
      sk = kg.secret_key();
      kg.create_public_key(pk);
      io.send_data(pk.data().data(), int(count * 8));
    } else {
      pk.data().resize(context, context.key_parms_id(), 2);
      pk.data().is_ntt_form() = true;
      io.recv_data(pk.data().data(), int(count * 8));
      if (!seal::is_valid_for(pk, context))
        throw std::runtime_error("E_BFV_KEY");
    }
    enc.reset(new seal::Encryptor(context, pk));
  }
  ~BfvEngine() {
    if (role == Role::Client)
      OPENSSL_cleanse(sk.data().data(),
                      sk.data().coeff_count() * sizeof(uint64_t));
  }
  seal::Ciphertext input_block(const std::vector<Word> &x) {
    constexpr size_t n = 16384;
    if (x.empty() || x.size() > 16384)
      throw std::runtime_error("E_BFV_PACKED_SHAPE");
    const size_t limbs =
        n * context.first_context_data()->parms().coeff_modulus().size();
    seal::Ciphertext ct(context);
    if (role == Role::Client) {
      seal::Plaintext plain(n);
      for (size_t j = 0; j < x.size(); ++j)
        plain[j] = x[j] & MASK;
      enc->encrypt(plain, ct);
      io.send_data(ct.data(), int(2 * limbs * 8));
    } else {
      ct.resize(context, context.first_parms_id(), 2);
      io.recv_data(ct.data(), int(2 * limbs * 8));
      if (!seal::is_valid_for(ct, context))
        throw std::runtime_error("E_BFV_INPUT");
    }
    return ct;
  }
  #include "bfv_matrix.inc"
  std::vector<Word> decode_output_tile(SecureOps &op,
                                     const std::vector<mpz_class>& values) {
    op.reserve_material("bfv_linear_tile_GC", values.size());
    return decoder.decode_tile(values, modulus_product(context.first_context_data()->parms().coeff_modulus()));
  }
  std::vector<Word> merged_blocks(
      SecureOps &op, const std::vector<seal::Ciphertext> &encrypted_inputs,
      const std::vector<Word> &x, size_t first_row, size_t rows,
      const PrivateWeightBlock &server_weights,
      std::vector<mpz_class>& pending_phases) {
    const size_t n = 16384, d = std::min(size_t(8192), x.size());
    const size_t blocks = encrypted_inputs.size();
    if (x.empty() || x.size() > 19 * size_t(8192) || !blocks || blocks > 19 ||
        blocks != (x.size() + 8191) / 8192 ||
        !rows || rows > packed_bfv_rows_per_ciphertext(d))
      throw std::runtime_error("E_BFV_PACKED_SHAPE");
    const size_t stride = 2 * d - 1;
    auto cd = context.first_context_data();
    const size_t k = cd->parms().coeff_modulus().size(), limbs = n * k;
    std::vector<Word> local(rows, 0);
    std::vector<mpz_class> value(rows);
    seal::Ciphertext ct;
    seal::Evaluator eval(context);
    if (role == Role::Server) {
      for (size_t block = 0; block < blocks; ++block) {
        const size_t begin = block * 8192;
        const size_t actual = std::min(d, x.size() - begin);
        const auto weights = server_weights(begin, actual, first_row, rows);
        if (weights.size() != rows)
          throw std::runtime_error("E_BFV_WEIGHT_BLOCK_SHAPE");
        for (const auto &row : weights)
          if (row.size() != actual)
            throw std::runtime_error("E_BFV_WEIGHT_BLOCK_SHAPE");
        seal::Plaintext weight(n);
        for (size_t row = 0; row < rows; ++row)
          for (size_t j = 0; j < actual; ++j) {
            weight[row * stride + d - 1 - j] = weights[row][j] & MASK;
            local[row] = (local[row] + x[begin + j] * weights[row][j]) & MASK;
          }
        // The immutable inputs are never modified. Every public block, even
        // an all-zero weight block, undergoes the full fixed-shape multiply.
        seal::Ciphertext product(encrypted_inputs[block]);
        std::vector<uint64_t> lifted(n * k);
        for (size_t j = 0; j < k; ++j) {
          auto qi = cd->parms().coeff_modulus()[j].value();
          auto *p = lifted.data() + j * n;
          for (size_t z = 0; z < n; ++z) {
            Word v = weight[z];
            const Word negative = Word(0) - (v >> 58);
            p[z] = v + ((qi - (uint64_t(1) << 59)) & negative);
          }
          seal::util::ntt_negacyclic_harvey(p, cd->small_ntt_tables()[j]);
          for (size_t component_index = 0; component_index < 2;
               ++component_index) {
            auto *c = product.data(component_index) + j * n;
            seal::util::ntt_negacyclic_harvey(c, cd->small_ntt_tables()[j]);
            seal::util::dyadic_product_coeffmod(
                c, p, n, cd->parms().coeff_modulus()[j], c);
            seal::util::inverse_ntt_negacyclic_harvey(c,
                                                      cd->small_ntt_tables()[j]);
          }
        }
        if (block == 0)
          ct = std::move(product); // initialize from a valid same-level object
        else
          eval.add_inplace(ct, product); // exact mod-q add; no rescaling
      }
    }
    // On S this is after all public products. On C no plaintext multiplication
    // is performed; both parties reserve the same next operation before c1.
    op.reserve_material("bfv_packed_mask_Enc0", n);
    if (role == Role::Server) {
      seal::Plaintext mask(n);
      if (RAND_priv_bytes((unsigned char *)mask.data(), int(n * 8)) != 1)
        throw std::runtime_error("E_RNG");
      for (size_t j = 0; j < n; ++j)
        mask[j] &= MASK;
      eval.sub_plain_inplace(ct, mask);
      seal::Ciphertext zero_ct;
      enc->encrypt_zero(zero_ct);
      eval.add_inplace(ct, zero_ct);
      io.send_data(ct.data(1), int(limbs * 8));
      for (size_t row = 0; row < rows; ++row) {
        const size_t target = row * stride + d - 1;
        value[row] = rns_coefficient(ct.data(0), n, target,
                                     cd->parms().coeff_modulus());
        local[row] = (local[row] + mask[target]) & MASK;
      }
    } else {
      std::vector<uint64_t> component(limbs);
      io.recv_data(component.data(), int(limbs * 8));
      for (size_t j = 0; j < k; ++j)
        for (size_t z = 0; z < n; ++z)
          if (component[j * n + z] >= cd->parms().coeff_modulus()[j].value())
            throw std::runtime_error("E_BFV_COMPONENT");
      for (size_t j = 0; j < k; ++j) {
        auto *p = component.data() + j * n;
        seal::util::ntt_negacyclic_harvey(p, cd->small_ntt_tables()[j]);
        seal::util::dyadic_product_coeffmod(p, sk.data().data() + j * n, n,
                                            cd->parms().coeff_modulus()[j], p);
        seal::util::inverse_ntt_negacyclic_harvey(p, cd->small_ntt_tables()[j]);
      }
      for (size_t row = 0; row < rows; ++row)
        value[row] = rns_coefficient(component.data(), n, row * stride + d - 1,
                                     cd->parms().coeff_modulus());
    }
    // Retain only target phases locally. The caller batches at most 512
    // outputs across returned ciphertexts; no joint phase is opened.
    pending_phases.insert(pending_phases.end(),value.begin(),value.end());
    return local;
  }
};
std::vector<Word>
SecureOps::private_linear(const std::vector<Word> &x,
                          const std::vector<std::vector<Word>> &weights,
                          size_t rows, unsigned fx, unsigned fw) {
  if (x.empty() || x.size() > 19 * size_t(8192) || !rows)
    throw std::runtime_error("E_SHAPE");
  PrivateWeightBlock fetch;
  if (role() == Role::Server) {
    if (weights.size() != rows)
      throw std::runtime_error("E_SHAPE");
    for (const auto &w : weights)
      if (w.size() != x.size())
        throw std::runtime_error("E_SHAPE");
    fetch = [&weights](size_t begin, size_t actual, size_t first, size_t count) {
      std::vector<std::vector<Word>> block(count, std::vector<Word>(actual));
      for (size_t row = 0; row < count; ++row)
        std::copy_n(weights[first + row].begin() + begin, actual, block[row].begin());
      return block;
    };
  }
  return private_linear_streamed(x, rows, fx, fw, fetch);
}
std::vector<Word>
SecureOps::private_linear_streamed(const std::vector<Word> &x, size_t rows,
                                   unsigned fx, unsigned fw,
                                   const PrivateWeightBlock &server_weights) {
  if (!isolated_development() && !channel().authenticated())
    throw std::runtime_error("E_MTLS_REQUIRED");
  if (!session_active())
    throw std::runtime_error("E_SESSION_CLOSED");
  if (x.empty() || x.size() > 19 * size_t(8192) || !rows ||
      fx > 58 || fw > 58 || fx + fw > 58 ||
      (role() == Role::Server && !server_weights))
    throw std::runtime_error("E_SHAPE");
  if (rows > MATRIX_COLUMNS_MAX)throw std::runtime_error("E_SHAPE");
  // The ordinary vector API uses precisely the same batched matrix protocol.
  if (x.size() <= MATRIX_INNER_MAX)
    return private_matrix({x},rows,fx,fw,server_weights).front();
  // Validate the public K bound before any ciphertext allocation or network.
  const size_t blocks = 1 + (x.size() - 1) / 8192;
  if (blocks > 19)
    throw std::runtime_error("E_SHAPE");
  const size_t dimension = std::min(size_t(8192), x.size());
  const size_t capacity = packed_bfv_rows_per_ciphertext(dimension);
  try {
    // ABI v2 binds long-input shape, scales and tile size before any BFV input.
    std::array<unsigned char,80> descriptor{},received{};
    const uint64_t fields[]={0x505241474c494e32ULL,2,1,x.size(),rows,fx,fw,8192,512,19};
    for(size_t f=0;f<10;++f)for(size_t b=0;b<8;++b)
      descriptor[f*8+b]=static_cast<unsigned char>(fields[f]>>(56-8*b));
    reserve_material("bfv_linear_descriptor",1);
    if(role()==Role::Client) {
      channel().send_data(descriptor.data(),descriptor.size());
      channel().recv_data(received.data(),received.size());
    } else {
      channel().recv_data(received.data(),received.size());
      if(received!=descriptor)throw std::runtime_error("E_BFV_LINEAR_DESCRIPTOR");
      channel().send_data(descriptor.data(),descriptor.size());
    }
    if(received!=descriptor)throw std::runtime_error("E_BFV_LINEAR_DESCRIPTOR");
    if (!bfv_) {
      reserve_material("bfv_key_and_GC_independent_OT", 1);
      bfv_ = std::make_shared<BfvEngine>(role(), channel());
    }
    std::vector<seal::Ciphertext> encrypted_inputs;
    encrypted_inputs.reserve(blocks);
    for (size_t block = 0; block < blocks; ++block) {
      const size_t begin = block * 8192;
      const size_t actual = std::min(dimension, x.size() - begin);
      std::vector<Word> xb(dimension, 0);
      std::copy_n(x.begin() + begin, actual, xb.begin());
      reserve_material("bfv_input_encryption", 1);
      encrypted_inputs.push_back(bfv_->input_block(xb));
    }
    std::vector<Word> out(rows, 0);
    for (size_t tile = 0; tile < rows; tile += 512) {
      const size_t count=std::min(size_t(512),rows-tile);
      std::vector<mpz_class> phases;phases.reserve(count);
      for(size_t first=tile;first<tile+count;first+=capacity) {
        const size_t group_rows=std::min(capacity,tile+count-first);
        const auto group=bfv_->merged_blocks(*this,encrypted_inputs,x,first,
                                             group_rows,server_weights,phases);
        std::copy(group.begin(),group.end(),out.begin()+first);
      }
      const auto shares=bfv_->decode_output_tile(*this,phases);
      for(size_t i=0;i<count;++i)out[tile+i]=(out[tile+i]+shares[i])&MASK;
    }
    return out; // f_X+f_W; no per-block or final implicit truncation
  } catch (...) {
    // Once protocol material may exist, callback/transport/allocation failure
    // burns this session and disconnects the waiting peer. Never resume it.
    abort_session();
    throw;
  }
}

std::vector<std::vector<Word>> SecureOps::private_matrix(
    const std::vector<std::vector<Word>>& x, size_t outputs,
    unsigned fx, unsigned fw, const PrivateWeightBlock& reader) {
  if(!isolated_development() && !channel().authenticated())throw std::runtime_error("E_MTLS_REQUIRED");
  if(!session_active())throw std::runtime_error("E_SESSION_CLOSED");
  if(x.empty() || x.size()>MATRIX_BATCH_MAX || x[0].empty() ||
      x[0].size()>MATRIX_INNER_MAX || !outputs || outputs>MATRIX_COLUMNS_MAX ||
      fx>58 || fw>58 || fx+fw>58 || (role()==Role::Server && !reader))
    throw std::runtime_error("E_SHAPE");
  for(const auto& row:x)if(row.size()!=x[0].size())throw std::runtime_error("E_SHAPE");
  try {
    // Explicit public ABI v2, big endian u64 fields, protected by the current
    // PRGCF001 session/configuration/stage/operation/sequence and mTLS.
    // Matrix BFV uses no HPKE; there is no new unbound AAD/info string.
    std::array<unsigned char,80> descriptor{},received{};
    const auto packing=matrix_packing(x.size(),x[0].size(),outputs);
    const uint64_t fields[]={0x505241474d415431ULL,3,x.size(),x[0].size(),outputs,
                             fx,fw,packing.reduction,
                             packing.columns,512};
    for(size_t f=0;f<10;++f)for(size_t b=0;b<8;++b)
      descriptor[f*8+b]=static_cast<unsigned char>(fields[f]>>(56-8*b));
    reserve_material("bfv_matrix_descriptor",1);
    if(role()==Role::Client) {
      channel().send_data(descriptor.data(),descriptor.size());
      channel().recv_data(received.data(),received.size());
    } else {
      channel().recv_data(received.data(),received.size());
      if(received!=descriptor)throw std::runtime_error("E_MATRIX_DESCRIPTOR");
      channel().send_data(descriptor.data(),descriptor.size());
    }
    if(received!=descriptor)throw std::runtime_error("E_MATRIX_DESCRIPTOR");
    if(!bfv_) {
      reserve_material("bfv_key_and_GC_independent_OT",1);
      bfv_=std::make_shared<BfvEngine>(role(),channel());
    }
    return bfv_->matrix(*this,x,outputs,reader);
  } catch(...) {abort_session();throw;}
}
} // namespace prag
