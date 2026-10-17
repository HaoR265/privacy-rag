// Outside BfvEngine. Session-bound production protocol, no plaintext path.
std::vector<Word> SecureOps::private_token_lookup(Word index,size_t vocabulary,
    size_t hidden,const PrivateWeightBlock& reader,TokenLookupStats* telemetry) {
  auto& op=*this;TokenLookupStats local;auto& stats=telemetry?*telemetry:local;stats={};
  if(!op.session_active())throw std::runtime_error("E_SESSION_CLOSED");
  if(!op.channel().authenticated())throw std::runtime_error("E_MTLS_REQUIRED");
  if((hidden!=1024&&hidden!=2048)||vocabulary>155648||!vocabulary||
     (op.role()==Role::Server&&!reader))throw std::runtime_error("E_TOKEN_SHAPE");
  const auto total_start=std::chrono::steady_clock::now();
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
    struct WipeOnehot {std::vector<Word>& row;~WipeOnehot(){OPENSSL_cleanse(row.data(),row.size()*sizeof(Word));}} wipe_onehot{onehot};
    stats.onehot_bytes=io.bytes()-before;stats.onehot_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const auto bkey=io.bytes();const auto key_start=std::chrono::steady_clock::now();
    if(!bfv_){op.reserve_material("bfv_key_and_GC_independent_OT",1);bfv_=std::make_shared<BfvEngine>(op.role(),io);}
    auto& engine=*bfv_;stats.initial_key_bytes=io.bytes()-bkey;
    const auto switch_start=std::chrono::steady_clock::now();
    stats.initial_key_seconds=std::chrono::duration<double>(switch_start-key_start).count();
    engine.setup_token_keys(op);stats.switch_key_bytes=io.bytes()-bkey-stats.initial_key_bytes;
    stats.switch_key_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-switch_start).count();
    const auto bonline=io.bytes();auto result=engine.token_packed(op,onehot,hidden,reader,telemetry);stats.online_bfv_gc_bytes=io.bytes()-bonline;
    stats.total_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-total_start).count();
    return result;
  }catch(...){op.abort_session();throw;}
}

std::vector<std::vector<Word>> SecureOps::private_token_lookup_many(
    const std::vector<Word>& indices,size_t vocabulary,size_t hidden,
    const PrivateWeightBlock& reader,TokenLookupStats* telemetry) {
  TokenLookupStats local;auto& stats=telemetry?*telemetry:local;stats={};
  if(!session_active())throw std::runtime_error("E_SESSION_CLOSED");
  if(!channel().authenticated())throw std::runtime_error("E_MTLS_REQUIRED");
  if(indices.empty()||indices.size()>4||(hidden!=1024&&hidden!=2048)||
      !vocabulary||vocabulary>155648||(role()==Role::Server&&!reader)){
    abort_session();throw std::runtime_error("E_TOKEN_BATCH_SHAPE");
  }
  std::vector<std::vector<Word>> onehots;
  struct WipeRows {std::vector<std::vector<Word>>& rows;
    ~WipeRows(){for(auto& row:rows)OPENSSL_cleanse(row.data(),row.size()*sizeof(Word));}} wipe{onehots};
  const auto total_start=std::chrono::steady_clock::now();
  try{
    // PRAGTKB4: 12 big-endian u64. Schedule 1 means slot-major input,
    // coordinate/block/prime-major shared weights, tile/slot-major returns.
    std::array<unsigned char,96> descriptor{},other{};
    const uint64_t values[]={0x50524147544b4234ULL,4,vocabulary,155648,hidden,512,8192,19,9,12,indices.size(),1};
    for(size_t j=0;j<12;++j)for(size_t byte=0;byte<8;++byte)descriptor[8*j+byte]=values[j]>>(56-8*byte);
    reserve_material("token_batch_descriptor",1);auto& io=channel();
    if(role()==Role::Client){io.send_data(descriptor.data(),descriptor.size());io.recv_data(other.data(),other.size());}
    else{io.recv_data(other.data(),other.size());if(other!=descriptor)throw std::runtime_error("E_TOKEN_BATCH_DESCRIPTOR");io.send_data(descriptor.data(),descriptor.size());}
    if(other!=descriptor)throw std::runtime_error("E_TOKEN_BATCH_DESCRIPTOR");
    const auto onehot_start=std::chrono::steady_clock::now();const auto onehot_bytes=io.bytes();
    onehots.reserve(indices.size());for(auto index:indices)onehots.push_back(one_hot_shared(index,155648));
    stats.onehot_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-onehot_start).count();
    stats.onehot_bytes=io.bytes()-onehot_bytes;
    const auto key_start=std::chrono::steady_clock::now();const auto key_bytes=io.bytes();
    if(!bfv_){reserve_material("bfv_key_and_GC_independent_OT",1);bfv_=std::make_shared<BfvEngine>(role(),io);}
    stats.initial_key_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-key_start).count();
    stats.initial_key_bytes=io.bytes()-key_bytes;
    const auto switch_start=std::chrono::steady_clock::now();const auto switch_bytes=io.bytes();
    bfv_->setup_token_keys(*this);
    stats.switch_key_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-switch_start).count();
    stats.switch_key_bytes=io.bytes()-switch_bytes;
    const auto online_bytes=io.bytes();
    auto result=bfv_->token_packed_many(*this,onehots,hidden,reader,telemetry);
    stats.online_bfv_gc_bytes=io.bytes()-online_bytes;
    stats.total_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-total_start).count();
    return result;
  }catch(...){abort_session();throw;}
}
