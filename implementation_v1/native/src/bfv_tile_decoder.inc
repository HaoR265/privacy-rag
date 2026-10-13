// Inside Decoder. A single GC produces a whole tile of arithmetic shares.
// Valid ONLY for the locked BFV profile and the proved phase error bound.
std::vector<Word> decode_tile(const std::vector<mpz_class>& local,
                              const mpz_class& q) {
  const size_t count=local.size();
  auto primes=locked_bfv_parameters().coeff_modulus();primes.pop_back();
  const mpz_class t=mpz_class(1)<<59;
  const mpz_class bound=8194*(512*(mpz_class(1)<<71)+1);
  if(count==0 || count>512 || q!=modulus_product(primes) || 4*t*bound>=q)
    throw std::runtime_error("E_BFV_TILE_PROFILE");
  std::vector<Word> high(count),rho(count,0),answer(count);
  mpz_class client_bits=0,server_bits=0;
  if(role==Role::Server) {
    if(RAND_priv_bytes(reinterpret_cast<unsigned char*>(rho.data()),int(count*8))!=1)
      throw std::runtime_error("E_RNG");
    for(auto& v:rho)v&=MASK;
  }
  for(size_t i=0;i<count;++i) {
    if(local[i]<0 || local[i]>=q)throw std::runtime_error("E_BFV_TILE_CANONICAL");
    mpz_class scaled=local[i]*t, quotient=scaled/q, remainder=scaled%q;
    mpz_class quarter=(4*remainder)/q;
    high[i]=quotient.get_ui();
    if(role==Role::Client)client_bits|=quarter<<(2*i);
    else server_bits|=(quarter+(mpz_class(rho[i])<<2))<<(61*i);
  }
  // Fresh circuit/labels/hash state once PER TILE. The independent IKNP
  // stream advances once over the complete 2*count evaluator-input vector.
  c.reset();generator=nullptr;
  if(role==Role::Server) {
    auto *g=new sci::HalfGateGen<GcBatchChannel>(&gate_io);generator=g;c.reset(g);
  } else c.reset(new sci::HalfGateEva<GcBatchChannel>(&gate_io));
  const auto u=input(client_bits,2*count,Role::Client);
  const auto v=input(server_bits,61*count,Role::Server);
  std::vector<Wires> output;output.reserve(count);
  gate_io.begin(122*count); // 4 AND carry kernel + 59-bit subtraction (118).
  for(size_t i=0;i<count;++i) {
    const Label a0=u[2*i],a1=u[2*i+1],b0=v[61*i],b1=v[61*i+1];
    const auto carry1=A(a0,b0),p1=X(a1,b1);
    const auto sum1=X(p1,carry1),sum2=X(A(a1,b1),A(p1,carry1));
    Wires correction(59,zero()),mask(59);
    // floor((quarter_C+quarter_S+2)/4) in {0,1,2}.
    correction[0]=X(sum1,sum2);correction[1]=A(sum1,sum2);
    for(size_t j=0;j<59;++j)mask[j]=v[61*i+2+j];
    Label unused;
    output.push_back(sub(correction,mask,unused)); // no correction bit opened
  }
  gate_io.end();
  std::vector<unsigned char> permutation(59*count);
  if(role==Role::Server) {
    for(size_t i=0;i<count;++i)for(size_t j=0;j<59;++j)
      permutation[59*i+j]=sci::getLSB(output[i][j]);
    io.send_data(permutation.data(),int(permutation.size()));
    for(size_t i=0;i<count;++i)answer[i]=(high[i]+rho[i])&MASK;
  } else {
    io.recv_data(permutation.data(),int(permutation.size()));
    for(size_t i=0;i<count;++i) {
      Word masked=0;
      for(size_t j=0;j<59;++j)
        masked|=Word(bool(sci::getLSB(output[i][j]))^bool(permutation[59*i+j]))<<j;
      answer[i]=(high[i]+masked)&MASK;
    }
  }
  OPENSSL_cleanse(rho.data(),rho.size()*sizeof(Word));
  return answer;
}
