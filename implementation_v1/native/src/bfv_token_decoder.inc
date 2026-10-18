// Inside Decoder. Only Token's proved |weight|<=2^20 and exact factor512
// permit this controlled unscale. Not a generic division or truncation API.
std::vector<Word> decode_token_tile(const std::vector<mpz_class>& local,
    const std::vector<Word>& server_mask,const mpz_class& q) {
  constexpr size_t count=512;
  auto primes=locked_bfv_parameters().coeff_modulus();primes.pop_back();
  const mpz_class t=mpz_class(1)<<59;
  const mpz_class e0=8194*(19*(mpz_class(1)<<71)+1)+1;
  const mpz_class bound=(mpz_class(1)<<18)*e0+174762*(mpz_class(1)<<24)+8194;
  if(local.size()!=count||server_mask.size()!=count||q!=modulus_product(primes)||4*t*bound>=q)
    throw std::runtime_error("E_TOKEN_DECODE_PROFILE");
  std::vector<Word> rho(count),answer(count);mpz_class cbits=0,sbits=0;
  if(role==Role::Server){if(RAND_priv_bytes(reinterpret_cast<unsigned char*>(rho.data()),count*8)!=1)throw std::runtime_error("E_RNG");for(auto&v:rho)v&=MASK;}
  for(size_t i=0;i<count;++i) {
    if(local[i]<0||local[i]>=q)throw std::runtime_error("E_BFV_TILE_CANONICAL");
    const mpz_class scaled=local[i]*t,high=scaled/q,rem=scaled%q,quarter=(4*rem)/q;
    if(role==Role::Client)cbits|=(quarter+(high<<2))<<(61*i);
    else sbits|=(quarter+(mpz_class((high.get_ui()+server_mask[i])&MASK)<<2)+(mpz_class(rho[i])<<61))<<(120*i);
  }
  c.reset();generator=nullptr;
  if(role==Role::Server){auto*g=new sci::HalfGateGen<GcBatchChannel>(&gate_io);generator=g;c.reset(g);}
  else c.reset(new sci::HalfGateEva<GcBatchChannel>(&gate_io));
  const auto u=input(cbits,61*count,Role::Client),v=input(sbits,120*count,Role::Server);
  std::vector<Wires> output;output.reserve(count);gate_io.begin(358*count);
  for(size_t i=0;i<count;++i) {
    const auto carry=A(u[61*i],v[120*i]),p=X(u[61*i+1],v[120*i+1]);
    const auto s1=X(p,carry),s2=X(A(u[61*i+1],v[120*i+1]),A(p,carry));
    Wires a(59),b(59),mask(59),correction(59,zero());
    correction[0]=X(s1,s2);correction[1]=A(s1,s2);
    for(size_t j=0;j<59;++j){a[j]=u[61*i+2+j];b[j]=v[120*i+2+j];mask[j]=v[120*i+61+j];}
    const auto value=add(add(a,b),correction);Wires unscaled(59);
    for(size_t j=0;j<59;++j)unscaled[j]=value[std::min(j+9,size_t(58))];
    Label unused;output.push_back(sub(unscaled,mask,unused));
  }
  gate_io.end();std::vector<unsigned char> perm(59*count);
  if(role==Role::Server){for(size_t i=0;i<count;++i)for(size_t j=0;j<59;++j)perm[59*i+j]=sci::getLSB(output[i][j]);io.send_data(perm.data(),perm.size());answer=rho;}
  else{io.recv_data(perm.data(),perm.size());for(size_t i=0;i<count;++i)for(size_t j=0;j<59;++j)answer[i]|=Word(bool(sci::getLSB(output[i][j]))^bool(perm[59*i+j]))<<j;}
  OPENSSL_cleanse(rho.data(),rho.size()*8);return answer;
}
