// Outside BfvEngine. Session-bound production protocol, no plaintext path.
std::vector<Word> SecureOps::private_token_lookup(Word index,size_t vocabulary,
    size_t hidden,const PrivateWeightBlock& reader,TokenLookupStats* telemetry) {
  auto& op=*this;TokenLookupStats local;auto& stats=telemetry?*telemetry:local;stats={};
  if(!op.session_active())throw std::runtime_error("E_SESSION_CLOSED");
  if(!op.channel().authenticated())throw std::runtime_error("E_MTLS_REQUIRED");
  if((hidden!=1024&&hidden!=2048)||vocabulary>155648||!vocabulary||
     (op.role()==Role::Server&&!reader))throw std::runtime_error("E_TOKEN_SHAPE");
  try {
    std::array<unsigned char,80> descriptor{},other{};
    const uint64_t values[]={0x50524147544f4b33ULL,3,vocabulary,155648,hidden,512,8192,19,9,12};
    for(size_t j=0;j<10;++j)for(size_t b=0;b<8;++b)descriptor[8*j+b]=values[j]>>(56-8*b);
    op.reserve_material("token_descriptor",1);auto& io=op.channel();
    if(op.role()==Role::Client){io.send_data(descriptor.data(),80);io.recv_data(other.data(),80);}
    else{io.recv_data(other.data(),80);if(other!=descriptor)throw std::runtime_error("E_TOKEN_DESCRIPTOR");io.send_data(descriptor.data(),80);}
    if(other!=descriptor)throw std::runtime_error("E_TOKEN_DESCRIPTOR");
    const auto start=std::chrono::steady_clock::now();const auto before=io.bytes();
    auto onehot=op.one_hot_shared(index,155648);
    stats.onehot_bytes=io.bytes()-before;stats.onehot_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const auto bkey=io.bytes();
    if(!bfv_){op.reserve_material("bfv_key_and_GC_independent_OT",1);bfv_=std::make_shared<BfvEngine>(op.role(),io);}
    auto& engine=*bfv_;stats.initial_key_bytes=io.bytes()-bkey;
    engine.setup_token_keys(op);stats.switch_key_bytes=io.bytes()-bkey-stats.initial_key_bytes;
    const auto bonline=io.bytes();auto result=engine.token_packed(op,onehot,hidden,reader);stats.online_bfv_gc_bytes=io.bytes()-bonline;
    OPENSSL_cleanse(onehot.data(),onehot.size()*8);return result;
  }catch(...){op.abort_session();throw;}
}
