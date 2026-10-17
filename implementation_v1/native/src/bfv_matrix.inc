// Included inside BfvEngine. Fixed coefficient matrix multiplication, not a
// network batch of scalar linear calls. x[B,K] * W[O,K]^T -> y[B,O].
std::vector<std::vector<Word>> matrix(
    SecureOps &op, const std::vector<std::vector<Word>> &x, size_t outputs,
    const PrivateWeightBlock &reader,bool qk_reverse64=false) {
  constexpr size_t n=16384;
  const size_t batch=x.size(), inner=x[0].size();
  if(qk_reverse64 && (inner!=128||outputs!=1||!batch||batch>64))throw std::runtime_error("E_QK_REVERSE64_SHAPE");
  const auto packing=qk_reverse64?matrix_packing_qk_reverse64(batch):matrix_packing(batch,inner,outputs);
  const size_t d=packing.reduction, stride=2*d-1, blocks=packing.inputs;
  const size_t columns_per_tile=512/batch;
  auto cd=context.first_context_data();
  const auto &primes=cd->parms().coeff_modulus();
  const size_t k=primes.size(), limbs=n*k;
  seal::Evaluator eval(context);
  std::vector<seal::Ciphertext> encrypted;
  encrypted.reserve(blocks);
  // C packs all B row segments in one input polynomial for each reduction
  // block. Ciphertexts are immutable and reused across public output tiles.
  for(size_t block=0;block<blocks;++block) {
    std::vector<Word> packed((batch-1)*stride+d,0);
    for(size_t b=0;b<batch;++b)
      for(size_t j=0;j<d && block*d+j<inner;++j)
        packed[b*stride+j]=x[b][block*d+j];
    op.reserve_material("bfv_matrix_input",1);
    encrypted.push_back(input_block(packed));
  }
#ifdef PRAG_EXPERIMENTAL_MATRIX_GPU
  // Coefficient-form input enters the GPU exactly once. This compile-time
  // backend never retries a failed CUDA operation through the CPU backend.
  std::unique_ptr<matrix_gpu::Matrix> gpu;
  auto close_gpu = [&]() noexcept {
    return gpu ? gpu->close() : matrix_gpu::Cleanup{};
  };
  try {
    if(role==Role::Server)
      gpu=std::make_unique<matrix_gpu::Matrix>(context,encrypted);
#else
  // Ciphertexts are immutable and local to this invocation. Reuse their exact
  // forward transform across every output group; no cross-session material.
  if(role==Role::Server) {
    for(auto& ct:encrypted)for(size_t j=0;j<k;++j)for(size_t c=0;c<2;++c)
      seal::util::ntt_negacyclic_harvey(ct.data(c)+j*n,cd->small_ntt_tables()[j]);
  }
#endif
  std::vector<std::vector<Word>> result(batch,std::vector<Word>(outputs));
  for(size_t tile=0;tile<outputs;tile+=columns_per_tile) {
    const size_t tile_count=std::min(columns_per_tile,outputs-tile);
    std::vector<mpz_class> phases;phases.reserve(batch*tile_count);
    for(size_t first=tile;first<tile+tile_count;first+=packing.columns) {
    const size_t count=std::min(packing.columns,tile+tile_count-first);
    seal::Ciphertext sum;
    if(role==Role::Server) {
#ifdef PRAG_EXPERIMENTAL_MATRIX_GPU
      std::vector<uint64_t> ring(blocks*n);
      struct RingWiper {
        std::vector<uint64_t>& values;
        ~RingWiper() { OPENSSL_cleanse(values.data(),values.size()*sizeof(uint64_t)); }
      } ring_wiper{ring};
      for(size_t block=0;block<blocks;++block) {
        const size_t begin=block*d, actual=std::min(d,inner-begin);
        const auto weights=reader(begin,actual,first,count);
        if(weights.size()!=count)throw std::runtime_error("E_BFV_WEIGHT_BLOCK_SHAPE");
        for(const auto& w:weights)
          if(w.size()!=actual)throw std::runtime_error("E_BFV_WEIGHT_BLOCK_SHAPE");
        for(size_t o=0;o<count;++o)for(size_t j=0;j<actual;++j) {
          ring[block*n+o*batch*stride+d-1-j]=weights[o][j]&MASK;
          for(size_t b=0;b<batch;++b)
            result[b][first+o]=(result[b][first+o]+x[b][begin+j]*weights[o][j])&MASK;
        }
      }
      std::vector<uint64_t> public_targets;public_targets.reserve(batch*count);
      for(size_t o=0;o<count;++o)for(size_t b=0;b<batch;++b)public_targets.push_back((o*batch+b)*stride+d-1);
      auto coeffs=gpu->evaluate_selective(ring,public_targets);
      RingWiper coefficients_wiper{coeffs};
      sum.resize(context,context.first_parms_id(),2);
      std::copy(coeffs.begin(),coeffs.end(),sum.data());
#else
      for(size_t block=0;block<blocks;++block) {
        const size_t begin=block*d, actual=std::min(d,inner-begin);
        const auto weights=reader(begin,actual,first,count);
        if(weights.size()!=count)throw std::runtime_error("E_BFV_WEIGHT_BLOCK_SHAPE");
        for(const auto &w:weights)
          if(w.size()!=actual)throw std::runtime_error("E_BFV_WEIGHT_BLOCK_SHAPE");
        seal::Plaintext weight(n);
        for(size_t o=0;o<count;++o)
          for(size_t j=0;j<actual;++j) {
            weight[o*batch*stride+d-1-j]=weights[o][j]&MASK;
            for(size_t b=0;b<batch;++b)
              result[b][first+o]=(result[b][first+o]+x[b][begin+j]*weights[o][j])&MASK;
          }
        // Every block, including zero weights and public tails, follows the
        // same full NTT path. No rotations, relinearization or modulus switch.
        seal::Ciphertext product(encrypted[block]);
        std::vector<uint64_t> lifted(limbs);
        for(size_t j=0;j<k;++j) {
          const auto qi=primes[j].value();auto *w=lifted.data()+j*n;
          for(size_t z=0;z<n;++z) {
            const Word v=weight[z], negative=Word(0)-(v>>58);
            w[z]=v+((qi-(Word(1)<<59))&negative);
          }
          seal::util::ntt_negacyclic_harvey(w,cd->small_ntt_tables()[j]);
          for(size_t c=0;c<2;++c) {
            auto *p=product.data(c)+j*n;
            seal::util::dyadic_product_coeffmod(p,w,n,primes[j],p);
          }
        }
        if(block==0)sum=std::move(product);
        else for(size_t c=0;c<2;++c)for(size_t j=0;j<k;++j)
          seal::util::add_poly_coeffmod(sum.data(c)+j*n,product.data(c)+j*n,n,primes[j],sum.data(c)+j*n);
      }
      // Inverse NTT is linear over each q_j: sum before inverse is identical
      // coefficient-by-coefficient. Return to canonical coefficient domain
      // before any SEAL plaintext/mask operation. Ciphertext metadata stays
      // coefficient-form because the temporary NTT arrays are never exported.
      for(size_t c=0;c<2;++c)for(size_t j=0;j<k;++j)
        seal::util::inverse_ntt_negacyclic_harvey(sum.data(c)+j*n,cd->small_ntt_tables()[j]);
#endif
    }
    op.reserve_material("bfv_matrix_tile_mask_Enc0",n);
    // Values below are local c0 or local c1*sk coefficients, never the phase.
    std::vector<mpz_class> local(batch*count);
    if(role==Role::Server) {
      seal::Plaintext mask(n);
      if(RAND_priv_bytes(reinterpret_cast<unsigned char*>(mask.data()),int(n*8))!=1)
        throw std::runtime_error("E_RNG");
      for(size_t j=0;j<n;++j)mask[j]&=MASK;
      eval.sub_plain_inplace(sum,mask);
      seal::Ciphertext zero;enc->encrypt_zero(zero);eval.add_inplace(sum,zero);
      io.send_data(sum.data(1),int(limbs*8)); // ONE c1 for the entire B x count tile
      for(size_t o=0;o<count;++o)for(size_t b=0;b<batch;++b) {
        const size_t target=(o*batch+b)*stride+d-1;
        result[b][first+o]=(result[b][first+o]+mask[target])&MASK;
        local[o*batch+b]=rns_coefficient(sum.data(0),n,target,primes);
      }
    } else {
      std::vector<uint64_t> component(limbs);
      io.recv_data(component.data(),int(limbs*8));
      for(size_t j=0;j<k;++j) {
        auto *p=component.data()+j*n;
        for(size_t z=0;z<n;++z)if(p[z]>=primes[j].value())throw std::runtime_error("E_BFV_COMPONENT");
        seal::util::ntt_negacyclic_harvey(p,cd->small_ntt_tables()[j]);
        seal::util::dyadic_product_coeffmod(p,sk.data().data()+j*n,n,primes[j],p);
        seal::util::inverse_ntt_negacyclic_harvey(p,cd->small_ntt_tables()[j]);
      }
      for(size_t o=0;o<count;++o)for(size_t b=0;b<batch;++b)
        local[o*batch+b]=rns_coefficient(component.data(),n,(o*batch+b)*stride+d-1,primes);
    }
    phases.insert(phases.end(),local.begin(),local.end());
    }
    const auto q=modulus_product(primes);
    op.reserve_material("bfv_matrix_tile_GC",batch*tile_count);
    const auto shares=decoder.decode_tile(phases,q);
    for(size_t o=0;o<tile_count;++o)for(size_t b=0;b<batch;++b)
      result[b][tile+o]=(result[b][tile+o]+shares[o*batch+b])&MASK;
  }
#ifdef PRAG_EXPERIMENTAL_MATRIX_GPU
  const auto cleanup=close_gpu();
  if(gpu && (cleanup.first_cuda_error || !cleanup.closed))
    throw std::runtime_error("E_MATRIX_GPU_CLEANUP");
#endif
  return result; // No floor. Output physical order is row-major B x O.
#ifdef PRAG_EXPERIMENTAL_MATRIX_GPU
  } catch(...) {
    // Preserve the first failure and still attempt zero/free on every buffer.
    // Normal-path cleanup failures above propagate to the session abort path.
    close_gpu();
    throw;
  }
#endif
}
