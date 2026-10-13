// Included inside BfvEngine. Fixed coefficient matrix multiplication, not a
// network batch of scalar linear calls. x[B,K] * W[O,K]^T -> y[B,O].
std::vector<std::vector<Word>> matrix(
    SecureOps &op, const std::vector<std::vector<Word>> &x, size_t outputs,
    const PrivateWeightBlock &reader) {
  constexpr size_t n=16384, d=MATRIX_REDUCTION_TILE, stride=2*d-1;
  const size_t batch=x.size(), inner=x[0].size(), blocks=(inner+d-1)/d;
  const size_t columns_per_tile=MATRIX_OUTPUT_COORDINATES/batch;
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
  std::vector<std::vector<Word>> result(batch,std::vector<Word>(outputs));
  for(size_t first=0;first<outputs;first+=columns_per_tile) {
    const size_t count=std::min(columns_per_tile,outputs-first);
    seal::Ciphertext sum;
    if(role==Role::Server) {
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
            seal::util::ntt_negacyclic_harvey(p,cd->small_ntt_tables()[j]);
            seal::util::dyadic_product_coeffmod(p,w,n,primes[j],p);
            seal::util::inverse_ntt_negacyclic_harvey(p,cd->small_ntt_tables()[j]);
          }
        }
        if(block==0)sum=std::move(product);
        else eval.add_inplace(sum,product);
      }
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
    const auto q=modulus_product(primes);
    op.reserve_material("bfv_matrix_tile_GC",batch*count);
    const auto shares=decoder.decode_tile(local,q);
    for(size_t o=0;o<count;++o)for(size_t b=0;b<batch;++b)
      result[b][first+o]=(result[b][first+o]+shares[o*batch+b])&MASK;
  }
  return result; // No floor. Output physical order is row-major B x O.
}
