// Inside BfvEngine. The complete 155648-entry input is arithmetic-shared.
// This candidate is not used by production until the separate mTLS run passes.
std::vector<seal::GaloisKeys> token_keys;
seal::SecretKey token_final_sk;
std::unique_ptr<seal::Encryptor> token_enc;
void setup_token_keys(SecureOps& op) {
  if(token_enc)return;
  constexpr size_t n=16384,limbs=n*7;
  op.reserve_material("token_acyclic_keys_9_levels",108);
  const auto before=io.bytes();
  token_keys.resize(9);seal::SecretKey old;
  if(role==Role::Client)old=sk;
  seal::PublicKey final_pk;
  for(unsigned level=0;level<9;++level) {
    seal::GaloisKeys keys;
    if(role==Role::Client){seal::KeyGenerator next(context);keys=acyclic_keys(old,next.secret_key(),context,level);
      OPENSSL_cleanse(old.data().data(),old.data().coeff_count()*8);old=next.secret_key();
      if(level==8)next.create_public_key(final_pk);
    }else{keys.parms_id()=context.key_parms_id();keys.data().resize(n);}
    for(uint32_t elt:{uint32_t(1),uint32_t((1u<<(level+1))+1)}) {
      auto& group=keys.data()[seal::GaloisKeys::get_index(elt)];
      if(role==Role::Server)group.resize(6);
      for(auto& key:group){
        if(role==Role::Client)io.send_data(key.data().data(),2*limbs*8);
        else{key.data().resize(context,context.key_parms_id(),2);key.data().is_ntt_form()=true;io.recv_data(key.data().data(),2*limbs*8);if(!seal::is_valid_for(key,context))throw std::runtime_error("E_TOKEN_SWITCH_KEY");}
      }
    }
    if(role==Role::Server)token_keys[level]=std::move(keys);
  }
  if(role==Role::Client){token_final_sk=std::move(old);io.send_data(final_pk.data().data(),2*limbs*8);}
  else{final_pk.data().resize(context,context.key_parms_id(),2);final_pk.data().is_ntt_form()=true;io.recv_data(final_pk.data().data(),2*limbs*8);if(!seal::is_valid_for(final_pk,context))throw std::runtime_error("E_TOKEN_FINAL_KEY");}
  token_enc.reset(new seal::Encryptor(context,final_pk));
  if(role==Role::Client)std::cerr<<"token109_keys_application_bytes="<<(io.bytes()-before)<<'\n';
}
seal::Ciphertext token_raw(const std::vector<seal::Ciphertext>& inputs,
    const std::vector<Word>& x,size_t output,const PrivateWeightBlock& reader) {
  constexpr size_t n=16384,d=8192,k=6;
  const auto cd=context.first_context_data();const auto& primes=cd->parms().coeff_modulus();
  seal::Ciphertext sum(context);sum.resize(context,context.first_parms_id(),2);
  std::fill_n(sum.data(),2*n*k,0);Word local=0;
  std::vector<uint64_t> lifted(n),product(n);
  for(size_t block=0;block<19;++block) {
    auto w=reader(block*d,d,output,1);
    if(w.size()!=1||w[0].size()!=d)throw std::runtime_error("E_BFV_WEIGHT_BLOCK_SHAPE");
    for(size_t j=0;j<d;++j){Word v=w[0][j]&MASK;if(v>(Word(1)<<20)&&v<((Word(1)<<59)-(Word(1)<<20)))throw std::runtime_error("E_TOKEN_WEIGHT_BOUND");local=(local+x[block*d+j]*v)&MASK;}
    for(size_t p=0;p<k;++p) {
      std::fill(lifted.begin(),lifted.end(),0);const Word q=primes[p].value();
      for(size_t j=0;j<d;++j){const Word v=w[0][j]&MASK;lifted[d-1-j]=v+((q-(Word(1)<<59))&(Word(0)-(v>>58)));}
      seal::util::ntt_negacyclic_harvey(lifted.data(),cd->small_ntt_tables()[p]);
      for(size_t c=0;c<2;++c){auto*dst=sum.data(c)+p*n;seal::util::dyadic_product_coeffmod(inputs[block].data(c)+p*n,lifted.data(),n,primes[p],product.data());seal::util::add_poly_coeffmod(dst,product.data(),n,primes[p],dst);}
    }
  }
  for(size_t c=0;c<2;++c)for(size_t p=0;p<k;++p)seal::util::inverse_ntt_negacyclic_harvey(sum.data(c)+p*n,cd->small_ntt_tables()[p]);
  seal::Plaintext own(n);own[d-1]=local;seal::Evaluator(context).add_plain_inplace(sum,own);
  monomial_shift(sum,2*n-(d-1),context);return sum;
}
std::vector<Word> token_packed(SecureOps& op,const std::vector<Word>& onehot,
    size_t hidden,const PrivateWeightBlock& reader) {
  constexpr size_t n=16384,k=6,limbs=n*k;
  if(onehot.size()!=155648||(hidden!=1024&&hidden!=2048))throw std::runtime_error("E_TOKEN_SHAPE");
  setup_token_keys(op);
  const auto cd=context.first_context_data();const auto& primes=cd->parms().coeff_modulus();
  std::vector<seal::Ciphertext> inputs;inputs.reserve(19);
  for(size_t block=0;block<19;++block){op.reserve_material("token_fixed_input",1);
    auto ct=input_block(std::vector<Word>(onehot.begin()+block*8192,onehot.begin()+(block+1)*8192));
    if(role==Role::Server)for(size_t c=0;c<2;++c)for(size_t p=0;p<k;++p)seal::util::ntt_negacyclic_harvey(ct.data(c)+p*n,cd->small_ntt_tables()[p]);
    inputs.push_back(std::move(ct));
  }
  std::vector<Word> result(hidden);seal::Evaluator eval(context);
  for(size_t tile=0;tile<hidden;tile+=512) {
    op.reserve_material("token_return_tile_mask_Enc0",n);
    std::vector<mpz_class> phases(512);std::vector<Word> masks(512,0);
    if(role==Role::Server) {
      std::vector<seal::Ciphertext> stack(10);std::array<bool,10> used{};
      for(size_t coordinate=0;coordinate<512;++coordinate) {
        auto current=token_raw(inputs,onehot,tile+coordinate,reader);unsigned level=0;
        while(used[level]){current=pack_pair(std::move(stack[level]),std::move(current),level,token_keys[level],context);used[level]=false;++level;}
        stack[level]=std::move(current);used[level]=true;
      }
      auto& ct=stack[9];seal::Plaintext mask(n);
      if(RAND_priv_bytes(reinterpret_cast<unsigned char*>(mask.data()),n*8)!=1)throw std::runtime_error("E_RNG");for(size_t j=0;j<n;++j)mask[j]&=MASK;
      eval.sub_plain_inplace(ct,mask);seal::Ciphertext zero;token_enc->encrypt_zero(zero);eval.add_inplace(ct,zero);
      io.send_data(ct.data(1),limbs*8);
      for(size_t i=0;i<512;++i){const auto target=packed_target(i);phases[i]=rns_coefficient(ct.data(0),n,target,primes);masks[i]=mask[target];}
    }else{
      std::vector<uint64_t> component(limbs);io.recv_data(component.data(),limbs*8);
      for(size_t p=0;p<k;++p){auto*a=component.data()+p*n;for(size_t j=0;j<n;++j)if(a[j]>=primes[p].value())throw std::runtime_error("E_BFV_COMPONENT");seal::util::ntt_negacyclic_harvey(a,cd->small_ntt_tables()[p]);seal::util::dyadic_product_coeffmod(a,token_final_sk.data().data()+p*n,n,primes[p],a);seal::util::inverse_ntt_negacyclic_harvey(a,cd->small_ntt_tables()[p]);}
      for(size_t i=0;i<512;++i)phases[i]=rns_coefficient(component.data(),n,packed_target(i),primes);
    }
    op.reserve_material("token_exact_tile_GC",512);
    const auto shares=decoder.decode_token_tile(phases,masks,modulus_product(primes));
    std::copy(shares.begin(),shares.end(),result.begin()+tile);
    OPENSSL_cleanse(masks.data(),masks.size()*8);
  }
  return result;
}
