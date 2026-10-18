// Included inside BfvEngine. Fixed coefficient matrix multiplication, not a
// network batch of scalar linear calls. x[B,K] * W[O,K]^T -> y[B,O].
std::vector<std::vector<Word>> matrix(
    SecureOps &op, const std::vector<std::vector<Word>> &x, size_t outputs,
    const PrivateWeightBlock &reader,bool qk_reverse64=false,bool input_scope=false,bool operation_scope=false) {
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
  ClientPhaseScope local_client_phase(context,"matrix_standard_key");
  auto& client_phase=operation_scope?matrix_client_phase():local_client_phase;
#ifdef PRAG_ENC0_GPU
  std::unique_ptr<enc0_gpu::PublicZero> enc0_backend;
#endif
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
#ifdef PRAG_MATRIX_INPUT_ENC_GPU
    if(role==Role::Client && input_scope) {
      if(packed.empty()||packed.size()>n)throw std::runtime_error("E_BFV_PACKED_SHAPE");
      if(!matrix_input_backend_){
        const char* uuid=std::getenv("PRAG_ENC0_GPU_UUID");
        const char* pci=std::getenv("PRAG_ENC0_GPU_PCI");
        if(!uuid||!pci)throw std::runtime_error("E_ENC0_DEVICE_BINDING");
        matrix_input_backend_=std::make_unique<enc0_gpu::PublicZero>(context,pk,uuid,pci);
      }
      seal::Plaintext plain(n);
      struct WipePlain { seal::Plaintext& p; ~WipePlain(){OPENSSL_cleanse(p.data(),p.coeff_count()*sizeof(uint64_t));} } wipe_plain{plain};
      for(size_t j=0;j<packed.size();++j)plain[j]=packed[j]&MASK;
      if(!seal::is_valid_for(plain,context)||plain.is_ntt_form())throw std::runtime_error("E_BFV_INPUT_PLAIN");
      auto ct=matrix_input_backend_->encrypt_zero();
      ++matrix_input_calls_;
      eval.add_plain_inplace(ct,plain);
      io.send_data(ct.data(),int(2*limbs*8));
      encrypted.push_back(std::move(ct));
    } else encrypted.push_back(input_block(packed));
#else
    encrypted.push_back(input_block(packed));
#endif
  }
#ifdef PRAG_EXPERIMENTAL_MATRIX_GPU
  // Coefficient-form input enters the GPU exactly once. This compile-time
  // backend never retries a failed CUDA operation through the CPU backend.
  std::unique_ptr<matrix_gpu::Matrix> gpu;
  auto close_gpu = [&]() noexcept {
    return gpu ? gpu->close() : matrix_gpu::Cleanup{};
  };
#ifdef PRAG_MATRIX_PENDING_EVALUATE
  // At most one async evaluate, joined before another launch or GPU cleanup.
  std::future<void> pending;
#endif
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
#ifdef PRAG_MATRIX_PENDING_EVALUATE
#ifndef PRAG_EXPERIMENTAL_MATRIX_GPU
#error "Pending evaluate requires the GPU matrix backend"
#endif
  const bool overlap = outputs > std::min(packing.columns,columns_per_tile);
  struct PipelineWork {
    std::vector<uint64_t> ring, targets, coefficients;
    size_t first, count;
    PipelineWork(size_t words,size_t f,size_t c):ring(words),first(f),count(c){}
    ~PipelineWork(){
      OPENSSL_cleanse(ring.data(),ring.size()*sizeof(uint64_t));
      OPENSSL_cleanse(coefficients.data(),coefficients.size()*sizeof(uint64_t));
    }
  };
  std::shared_ptr<PipelineWork> next_work;
  auto launch_next=[&](size_t first,size_t count){
    if(role!=Role::Server||pending.valid())throw std::runtime_error("E_MATRIX_PENDING_STATE");
    auto work=std::make_shared<PipelineWork>(blocks*n,first,count);
    // Reader and local share accumulation always remain on the protocol thread.
    for(size_t block=0;block<blocks;++block){
      const size_t begin=block*d,actual=std::min(d,inner-begin);
      const auto weights=reader(begin,actual,first,count);
      if(weights.size()!=count)throw std::runtime_error("E_BFV_WEIGHT_BLOCK_SHAPE");
      for(const auto&w:weights)if(w.size()!=actual)throw std::runtime_error("E_BFV_WEIGHT_BLOCK_SHAPE");
      for(size_t o=0;o<count;++o)for(size_t j=0;j<actual;++j){
        work->ring[block*n+o*batch*stride+d-1-j]=weights[o][j]&MASK;
        for(size_t b=0;b<batch;++b)
          result[b][first+o]=(result[b][first+o]+x[b][begin+j]*weights[o][j])&MASK;
      }
    }
    work->targets.reserve(batch*count);
    for(size_t o=0;o<count;++o)for(size_t b=0;b<batch;++b)
      work->targets.push_back((o*batch+b)*stride+d-1);
    next_work=work;
    auto evaluate=[engine=gpu.get(),work]{
      // evaluate_selective binds its CUDA device on this worker itself.
      // No reader, RNG, material allocation, TLS or decoder access here.
      work->coefficients=engine->evaluate_selective(work->ring,work->targets);
    };
    if(overlap) pending=std::async(std::launch::async,evaluate);
    else evaluate(); // Single public group has no possible overlap.
  };
  if(role==Role::Server)launch_next(0,std::min(packing.columns,std::min(columns_per_tile,outputs)));
#endif
  for(size_t tile=0;tile<outputs;tile+=columns_per_tile) {
    const size_t tile_count=std::min(columns_per_tile,outputs-tile);
    std::vector<mpz_class> phases;phases.reserve(batch*tile_count);
    for(size_t first=tile;first<tile+tile_count;first+=packing.columns) {
    const size_t count=std::min(packing.columns,tile+tile_count-first);
    seal::Ciphertext sum;
    if(role==Role::Server) {
#ifdef PRAG_EXPERIMENTAL_MATRIX_GPU
#ifdef PRAG_MATRIX_PENDING_EVALUATE
      if(pending.valid()!=overlap||!next_work||next_work->first!=first||next_work->count!=count)
        throw std::runtime_error("E_MATRIX_PENDING_ORDER");
      if(overlap) pending.get(); // propagate worker failure before return material use
      auto current_work=std::move(next_work);
      sum.resize(context,context.first_parms_id(),2);
      if(current_work->coefficients.size()!=2*limbs)throw std::runtime_error("E_MATRIX_PENDING_COEFFICIENTS");
      std::copy(current_work->coefficients.begin(),current_work->coefficients.end(),sum.data());
      // Prepare and launch only the immediately following public output group.
      const size_t following=first+count;
      if(following<outputs){
        const size_t following_tile_end=following<tile+tile_count
          ?tile+tile_count:following+std::min(columns_per_tile,outputs-following);
        launch_next(following,std::min(packing.columns,following_tile_end-following));
      }
#else
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
#endif
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
      seal::Ciphertext zero;
#ifdef PRAG_ENC0_GPU
      if(!enc0_backend){const char* uuid=std::getenv("PRAG_ENC0_GPU_UUID");const char* pci=std::getenv("PRAG_ENC0_GPU_PCI");if(!uuid||!pci)throw std::runtime_error("E_ENC0_DEVICE_BINDING");enc0_backend=std::make_unique<enc0_gpu::PublicZero>(context,pk,uuid,pci);}
      zero=enc0_backend->encrypt_zero();
#else
      enc->encrypt_zero(zero);
#endif
      eval.add_inplace(sum,zero);
      io.send_data(sum.data(1),int(limbs*8)); // ONE c1 for the entire B x count tile
      for(size_t o=0;o<count;++o)for(size_t b=0;b<batch;++b) {
        const size_t target=(o*batch+b)*stride+d-1;
        result[b][first+o]=(result[b][first+o]+mask[target])&MASK;
        local[o*batch+b]=rns_coefficient(sum.data(0),n,target,primes);
      }
    } else {
      std::vector<uint64_t> component(limbs);
      struct WipePhaseComponent {
        std::vector<uint64_t>& value;
        ~WipePhaseComponent()noexcept { OPENSSL_cleanse(value.data(),value.size()*sizeof(uint64_t)); }
      } wipe_component{component};
      io.recv_data(component.data(),int(limbs*8));
      client_phase.apply(component,sk);
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
#ifdef PRAG_MATRIX_PENDING_EVALUATE
  if(pending.valid())throw std::runtime_error("E_MATRIX_PENDING_UNCONSUMED");
#endif
  const auto cleanup=close_gpu();
  if(gpu && (cleanup.first_cuda_error || !cleanup.closed))
    throw std::runtime_error("E_MATRIX_GPU_CLEANUP");
#endif
#ifdef PRAG_ENC0_GPU
  if(enc0_backend){auto c=enc0_backend->close();if(!c.closed||c.first_cuda_error)throw std::runtime_error("E_ENC0_CLEANUP");}
#endif
  if(!operation_scope)client_phase.finish();
  return result; // No floor. Output physical order is row-major B x O.
#ifdef PRAG_EXPERIMENTAL_MATRIX_GPU
  } catch(...) {
    // Preserve the first failure and still attempt zero/free on every buffer.
    // Normal-path cleanup failures above propagate to the session abort path.
#ifdef PRAG_MATRIX_PENDING_EVALUATE
    // Join before Matrix::close: neither buffers nor state may race evaluation.
    // Preserve the first exception; a secondary worker failure never hides it.
    if(pending.valid())try{pending.get();}catch(...){}
#endif
    close_gpu();
#ifdef PRAG_ENC0_GPU
    if(enc0_backend){auto c=enc0_backend->close();if(!c.closed||c.first_cuda_error)std::cerr<<"E_ENC0_CLEANUP_DURING_FAILURE\n";}
#endif
    throw;
  }
#endif
}
