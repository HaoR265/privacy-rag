#if defined(PRAG_TOKEN_LOCAL_RING_GPU) && !defined(PRAG_TOKEN_FULL_DEVICE_RESIDENCY)
#error "Local ring GPU requires full device residency"
#endif
// Inside BfvEngine. Fixed public B<=4; only S's deterministic weight
// transforms are shared. No input ciphertext or material survives this call.
struct TokenWordsWiper {
  std::vector<Word>& words;
  ~TokenWordsWiper(){OPENSSL_cleanse(words.data(),words.size()*sizeof(Word));}
};
struct TokenPlaintextWiper {
  seal::Plaintext& value;
  ~TokenPlaintextWiper(){OPENSSL_cleanse(value.data(),value.coeff_count()*sizeof(Word));}
};
std::vector<seal::Ciphertext> token_raw_many(
    const std::vector<std::vector<seal::Ciphertext>>& inputs,
    const std::vector<std::vector<Word>>& onehots,size_t output,
    const PrivateWeightBlock& reader,TokenLookupStats* stats
#ifdef PRAG_EXPERIMENTAL_TOKEN_GPU
    ,experimental::resident::TokenWeights* gpu
#endif
    ) {
  constexpr size_t n=16384,d=8192,k=6;
  const size_t slots=onehots.size();
  const auto cd=context.first_context_data();const auto& primes=cd->parms().coeff_modulus();
  std::vector<seal::Ciphertext> sums(slots);
  for(auto& sum:sums){sum.resize(context,context.first_parms_id(),2);std::fill_n(sum.data(),2*n*k,0);}
  std::vector<Word> local(slots,0);TokenWordsWiper wipe_local{local};
  std::vector<uint64_t> lifted(n),product(n);
#ifdef PRAG_EXPERIMENTAL_TOKEN_GPU
  std::vector<Word> gpu_weights(gpu?19*d:0);TokenWordsWiper wipe_gpu_weights{gpu_weights};
#endif
  for(size_t block=0;block<19;++block){
    const auto read_start=stats?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
    auto weights=reader(block*d,d,output,1);
    if(stats){stats->server_weight_read_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-read_start).count();
      ++stats->server_weight_blocks;stats->server_weight_values+=d;}
    if(weights.size()!=1||weights[0].size()!=d)throw std::runtime_error("E_BFV_WEIGHT_BLOCK_SHAPE");
    for(size_t j=0;j<d;++j){
      const Word value=weights[0][j]&MASK;
      if(value>(Word(1)<<20)&&value<((Word(1)<<59)-(Word(1)<<20)))throw std::runtime_error("E_TOKEN_WEIGHT_BOUND");
      for(size_t slot=0;slot<slots;++slot)local[slot]=(local[slot]+onehots[slot][block*d+j]*value)&MASK;
    }
#ifdef PRAG_EXPERIMENTAL_TOKEN_GPU
    if(gpu){
      for(size_t j=0;j<d;++j)gpu_weights[block*d+j]=weights[0][j]&MASK;
      // Logical transforms move to the device; CPU transform time stays zero.
      if(stats)stats->server_weight_ntts+=k;
      continue;
    }
#endif
    for(size_t prime=0;prime<k;++prime){
      const auto weight_start=stats?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
      std::fill(lifted.begin(),lifted.end(),0);const Word modulus=primes[prime].value();
      for(size_t j=0;j<d;++j){const Word value=weights[0][j]&MASK;
        lifted[d-1-j]=value+((modulus-(Word(1)<<59))&(Word(0)-(value>>58)));}
      seal::util::ntt_negacyclic_harvey(lifted.data(),cd->small_ntt_tables()[prime]);
      if(stats){++stats->server_weight_ntts;stats->server_weight_lift_ntt_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-weight_start).count();}
      {
      const auto dyadic_start=stats?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
      for(size_t slot=0;slot<slots;++slot)for(size_t component=0;component<2;++component){
        auto* dst=sums[slot].data(component)+prime*n;
        seal::util::dyadic_product_coeffmod(inputs[slot][block].data(component)+prime*n,lifted.data(),n,primes[prime],product.data());
        seal::util::add_poly_coeffmod(dst,product.data(),n,primes[prime],dst);
      }
      if(stats)stats->server_dyadic_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-dyadic_start).count();
      }
    }
  }
#ifdef PRAG_EXPERIMENTAL_TOKEN_GPU
  if(gpu){
    const auto begin=std::chrono::steady_clock::now();auto flat=gpu->evaluate(gpu_weights);TokenWordsWiper wipe_flat{flat};
    if(flat.size()!=slots*2*k*n)throw std::runtime_error("E_GPU_TOKEN_OUTPUT_SHAPE");
    for(size_t slot=0;slot<slots;++slot)std::copy_n(flat.data()+slot*2*k*n,2*k*n,sums[slot].data());
    if(stats){const auto t=gpu->timing();stats->server_dyadic_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();stats->gpu_weight_upload_seconds+=t.weight_upload_ms/1000;stats->gpu_weight_lift_seconds+=t.lift_ms/1000;stats->gpu_weight_ntt_seconds+=t.ntt_ms/1000;stats->gpu_dyadic_seconds+=t.dyadic_ms/1000;stats->gpu_kernel_seconds+=(t.lift_ms+t.ntt_ms+t.dyadic_ms)/1000;stats->gpu_output_download_seconds+=t.download_ms/1000;}
  }
#endif
  for(size_t slot=0;slot<slots;++slot){
    auto& sum=sums[slot];
    for(size_t component=0;component<2;++component)for(size_t prime=0;prime<k;++prime)
      seal::util::inverse_ntt_negacyclic_harvey(sum.data(component)+prime*n,cd->small_ntt_tables()[prime]);
    seal::Plaintext own(n);TokenPlaintextWiper wipe_own{own};own[d-1]=local[slot];seal::Evaluator(context).add_plain_inplace(sum,own);
    monomial_shift(sum,2*n-(d-1),context);
  }
  return sums;
}

#if defined(PRAG_EXPERIMENTAL_TOKEN_GPU) && defined(PRAG_TOKEN_COORDINATE_PIPELINE)
void token_prepare_coordinate(candidate_token_pipeline::Coordinate& prepared,
 const std::vector<std::vector<Word>>& onehots,const PrivateWeightBlock& reader,bool local_gpu=false){
 const auto prepare_begin=std::chrono::steady_clock::now();
 constexpr size_t d=8192;prepared.local.fill(0);prepared.read_seconds=0;prepared.blocks=prepared.values=0;
 for(size_t block=0;block<19;++block){
  auto begin=std::chrono::steady_clock::now();auto weights=reader(block*d,d,prepared.index,1);
  struct WipeRows{std::vector<std::vector<Word>>& rows;~WipeRows(){for(auto& row:rows)OPENSSL_cleanse(row.data(),row.size()*sizeof(Word));}} wipe{weights};
  prepared.read_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();++prepared.blocks;prepared.values+=d;
  if(weights.size()!=1||weights[0].size()!=d)throw std::runtime_error("E_BFV_WEIGHT_BLOCK_SHAPE");
  for(size_t j=0;j<d;++j){const Word value=weights[0][j]&MASK;
   if(value>(Word(1)<<20)&&value<((Word(1)<<59)-(Word(1)<<20)))throw std::runtime_error("E_TOKEN_WEIGHT_BOUND");
   prepared.weights[block*d+j]=value;
   if(!local_gpu)for(size_t slot=0;slot<prepared.slots;++slot)prepared.local[slot]=(prepared.local[slot]+onehots[slot][block*d+j]*value)&MASK;
  }
 }
 prepared.prepare_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-prepare_begin).count();
}
std::vector<seal::Ciphertext> token_consume_coordinate(
 const candidate_token_pipeline::Coordinate& prepared,experimental::resident::TokenWeights* gpu,TokenLookupStats* stats){
 constexpr size_t n=16384,d=8192,k=6;const size_t slots=prepared.slots;
 const auto cd=context.first_context_data();const auto& primes=cd->parms().coeff_modulus();
 std::vector<seal::Ciphertext> sums(slots);for(auto& sum:sums)sum.resize(context,context.first_parms_id(),2);
 if(stats){stats->pipeline_prepare_seconds+=prepared.prepare_seconds;stats->server_weight_read_seconds+=prepared.read_seconds;stats->server_weight_blocks+=prepared.blocks;stats->server_weight_values+=prepared.values;stats->server_weight_ntts+=19*k;}
    const auto begin=std::chrono::steady_clock::now();auto flat=gpu->evaluate(prepared.weights);TokenWordsWiper wipe_flat{flat};
    if(flat.size()!=slots*2*k*n)throw std::runtime_error("E_GPU_TOKEN_OUTPUT_SHAPE");
    for(size_t slot=0;slot<slots;++slot)std::copy_n(flat.data()+slot*2*k*n,2*k*n,sums[slot].data());
    if(stats){const auto t=gpu->timing();stats->server_dyadic_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();stats->gpu_weight_upload_seconds+=t.weight_upload_ms/1000;stats->gpu_weight_lift_seconds+=t.lift_ms/1000;stats->gpu_weight_ntt_seconds+=t.ntt_ms/1000;stats->gpu_dyadic_seconds+=t.dyadic_ms/1000;stats->gpu_kernel_seconds+=(t.lift_ms+t.ntt_ms+t.dyadic_ms)/1000;stats->gpu_output_download_seconds+=t.download_ms/1000;}
  for(size_t slot=0;slot<slots;++slot){
    auto& sum=sums[slot];
    for(size_t component=0;component<2;++component)for(size_t prime=0;prime<k;++prime)
      seal::util::inverse_ntt_negacyclic_harvey(sum.data(component)+prime*n,cd->small_ntt_tables()[prime]);
    seal::Plaintext own(n);TokenPlaintextWiper wipe_own{own};own[d-1]=prepared.local[slot];seal::Evaluator(context).add_plain_inplace(sum,own);
    monomial_shift(sum,2*n-(d-1),context);
  }
  return sums;
}
#endif

std::vector<std::vector<Word>> token_packed_many(SecureOps& op,
    const std::vector<std::vector<Word>>& onehots,size_t hidden,
    const PrivateWeightBlock& reader,TokenLookupStats* stats) {
  constexpr size_t n=16384,k=6,limbs=n*k;
  const size_t slots=onehots.size();
  if(!slots||slots>4||(hidden!=1024&&hidden!=2048))throw std::runtime_error("E_TOKEN_BATCH_SHAPE");
  for(const auto& row:onehots)if(row.size()!=155648)throw std::runtime_error("E_TOKEN_BATCH_SHAPE");
  setup_token_keys(op);
  const auto cd=context.first_context_data();const auto& primes=cd->parms().coeff_modulus();
  std::vector<std::vector<seal::Ciphertext>> inputs(slots);
  const auto input_start=stats?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
  for(size_t slot=0;slot<slots;++slot){
    inputs[slot].reserve(19);
    for(size_t block=0;block<19;++block){
      op.reserve_material("token_fixed_input",1);
      std::vector<Word> part(onehots[slot].begin()+block*8192,onehots[slot].begin()+(block+1)*8192);
      TokenWordsWiper wipe_part{part};auto ct=input_block(part);
#ifndef PRAG_EXPERIMENTAL_TOKEN_GPU
      if(role==Role::Server)for(size_t component=0;component<2;++component)for(size_t prime=0;prime<k;++prime)
        seal::util::ntt_negacyclic_harvey(ct.data(component)+prime*n,cd->small_ntt_tables()[prime]);
#endif
      inputs[slot].push_back(std::move(ct));if(stats)++stats->input_ciphertexts;
    }
  }
#ifdef PRAG_EXPERIMENTAL_TOKEN_GPU
  std::unique_ptr<experimental::resident::TokenWeights> gpu;
  std::unique_ptr<experimental_full_switch::Packing> gpu_packing;
  if(role==Role::Server){const auto begin=std::chrono::steady_clock::now();gpu=std::make_unique<experimental::resident::TokenWeights>(context,inputs);if(stats){const auto t=gpu->timing();stats->gpu_dyadic_used=true;stats->gpu_constructor_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();stats->gpu_input_upload_seconds=t.initial_input_upload_ms/1000;stats->gpu_device_bytes=t.device_bytes;}}
  if(role==Role::Server){
    const auto begin=std::chrono::steady_clock::now();
    gpu_packing=std::make_unique<experimental_full_switch::Packing>(context,token_keys);
    if(stats){stats->gpu_packing_constructor_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();stats->gpu_packing_device_bytes=gpu_packing->stats().device_bytes;}
  }
  auto close_gpu=[&]{
    const auto begin=std::chrono::steady_clock::now();bool closed=true;int error=0;size_t wiped=0,freed=0;
    if(gpu){const auto r=gpu->close();closed&=r.closed;error=r.first_cuda_error;wiped+=r.wipe_synchronized_bytes;freed+=r.freed_bytes;}
    if(gpu_packing){const auto r=gpu_packing->close();closed&=r.closed;if(!error)error=r.first_cuda_error;wiped+=r.wipe_synchronized_bytes;freed+=r.freed_bytes;}
    if(stats){stats->gpu_cleanup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();stats->gpu_cleanup_closed=closed;stats->gpu_cleanup_error=error;stats->gpu_wipe_synchronized_bytes=wiped;stats->gpu_freed_bytes=freed;}
    if(!closed||error)throw std::runtime_error("E_GPU_TOKEN_CLEANUP");
  };
#endif
  if(stats)stats->input_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-input_start).count();
  std::vector<std::vector<Word>> result(slots,std::vector<Word>(hidden));
  bool local_ring_selected=false;
#ifdef PRAG_TOKEN_LOCAL_RING_GPU
  local_ring_selected=(slots==1&&hidden==1024)||(slots==4&&hidden==2048);
#endif
  try {
#ifdef PRAG_TOKEN_LOCAL_RING_GPU
    if(role==Role::Server&&local_ring_selected){gpu->bind_local_shares(onehots);
      if(stats)stats->gpu_device_bytes=gpu->timing().device_bytes;
    }
#endif
    seal::Evaluator eval(context);
    for(size_t tile=0;tile<hidden;tile+=512){
      std::vector<std::vector<seal::Ciphertext>> stacks(slots,std::vector<seal::Ciphertext>(10));
      std::vector<std::array<bool,10>> used(slots,std::array<bool,10>{});
#ifdef PRAG_TOKEN_FULL_DEVICE_RESIDENCY
      if(role==Role::Server){
        auto consume_device=[&](const candidate_token_pipeline::Coordinate& prepared){
          auto begin=std::chrono::steady_clock::now();
          std::vector<Word> local(prepared.local.begin(),prepared.local.begin()+slots);TokenWordsWiper wipe_local{local};
          #ifdef PRAG_TOKEN_LOCAL_RING_GPU
          auto batch=local_ring_selected?gpu->evaluate_local_finished(prepared.weights):gpu->evaluate_finished(prepared.weights,local);
#else
          auto batch=gpu->evaluate_finished(prepared.weights,local);
#endif
          if(stats){stats->server_raw_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();stats->pipeline_prepare_seconds+=prepared.prepare_seconds;stats->server_weight_read_seconds+=prepared.read_seconds;stats->server_weight_blocks+=prepared.blocks;stats->server_weight_values+=prepared.values;stats->server_weight_ntts+=19*k;
            const auto t=gpu->timing();stats->gpu_weight_upload_seconds+=t.weight_upload_ms/1000;stats->gpu_weight_lift_seconds+=t.lift_ms/1000;stats->gpu_weight_ntt_seconds+=t.ntt_ms/1000;stats->gpu_dyadic_seconds+=t.dyadic_ms/1000;stats->gpu_kernel_seconds+=(t.lift_ms+t.ntt_ms+t.dyadic_ms+t.finish_ms)/1000;}
          constexpr size_t ct_words=2*k*n;
          for(size_t slot=0;slot<slots;++slot){
            experimental::resident::DeviceCipherBatch current{batch.data+slot*ct_words,ct_words,batch.device,batch.ready,batch.owner};
            unsigned level=0;
            while(used[slot][level]){
              const auto t=std::chrono::steady_clock::now();
              current=gpu_packing->pair_device(gpu->stack_view(slot,level),current,level);
              used[slot][level]=false;++level;
              if(stats){++stats->server_pack_merges;stats->server_packing_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();}
            }
            gpu->stack_store(slot,level,current);used[slot][level]=true;
          }
        };
        const auto pipeline_begin=std::chrono::steady_clock::now();
        candidate_token_pipeline::TwoCoordinates pipeline;
        bool parallel=private_reader_worker_safe(reader);if(stats)stats->coordinate_pipeline_used=parallel;
        if(parallel)pipeline.run(tile,512,slots,[&](candidate_token_pipeline::Coordinate& p){token_prepare_coordinate(p,onehots,reader,local_ring_selected);},consume_device,true);
        else{candidate_token_pipeline::Coordinate prepared;prepared.slots=slots;
          for(size_t coordinate=0;coordinate<512;++coordinate){prepared.index=tile+coordinate;token_prepare_coordinate(prepared,onehots,reader,local_ring_selected);consume_device(prepared);prepared.erase();}}
        if(stats)stats->pipeline_tiles_wall_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-pipeline_begin).count();
        for(size_t slot=0;slot<slots;++slot){const auto download_begin=std::chrono::steady_clock::now();auto flat=gpu->download(gpu->stack_view(slot,9));TokenWordsWiper wipe{flat};
          if(stats)stats->gpu_output_download_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-download_begin).count();
          stacks[slot][9].resize(context,context.first_parms_id(),2);std::copy(flat.begin(),flat.end(),stacks[slot][9].data());}
      }
#else
      auto pack_current=[&](std::vector<seal::Ciphertext> current){
        for(size_t slot=0;slot<slots;++slot){
          unsigned level=0;
          while(used[slot][level]){
            const auto pack_start=stats?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
#ifdef PRAG_EXPERIMENTAL_TOKEN_GPU
            current[slot]=gpu_packing->pair(std::move(stacks[slot][level]),std::move(current[slot]),level);
#else
            current[slot]=pack_pair(std::move(stacks[slot][level]),std::move(current[slot]),level,token_keys[level],context);
#endif
            used[slot][level]=false;++level;
            if(stats){++stats->server_pack_merges;stats->server_packing_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-pack_start).count();}
          }
          stacks[slot][level]=std::move(current[slot]);used[slot][level]=true;
        }
      };
#if defined(PRAG_EXPERIMENTAL_TOKEN_GPU) && defined(PRAG_TOKEN_COORDINATE_PIPELINE)
      if(role==Role::Server&&private_reader_worker_safe(reader)){
        const auto pipeline_begin=std::chrono::steady_clock::now();
        if(stats)stats->coordinate_pipeline_used=true;
        candidate_token_pipeline::TwoCoordinates pipeline;
        pipeline.run(tile,512,slots,
          [&](candidate_token_pipeline::Coordinate& prepared){token_prepare_coordinate(prepared,onehots,reader);},
          [&](const candidate_token_pipeline::Coordinate& prepared){
            const auto begin=std::chrono::steady_clock::now();auto current=token_consume_coordinate(prepared,gpu.get(),stats);
            if(stats)stats->server_raw_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            pack_current(std::move(current));
          },true);
        if(stats)stats->pipeline_tiles_wall_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-pipeline_begin).count();
      }else
#endif
      if(role==Role::Server)for(size_t coordinate=0;coordinate<512;++coordinate){
        const auto raw_start=stats?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        auto current=token_raw_many(inputs,onehots,tile+coordinate,reader,stats
#ifdef PRAG_EXPERIMENTAL_TOKEN_GPU
          ,gpu.get()
#endif
          );
        if(stats)stats->server_raw_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-raw_start).count();
        pack_current(std::move(current));
      }
#endif
      // The descriptor binds tile-major, then slot-major protected returns.
      for(size_t slot=0;slot<slots;++slot){
        op.reserve_material("token_return_tile_mask_Enc0",n);
        const auto return_start=stats?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        std::vector<mpz_class> phases(512);std::vector<Word> masks(512,0);TokenWordsWiper wipe_masks{masks};
        if(role==Role::Server){
          auto& ct=stacks[slot][9];seal::Plaintext mask(n);TokenPlaintextWiper wipe_mask{mask};
          if(RAND_priv_bytes(reinterpret_cast<unsigned char*>(mask.data()),n*8)!=1)throw std::runtime_error("E_RNG");
          for(size_t j=0;j<n;++j)mask[j]&=MASK;
          eval.sub_plain_inplace(ct,mask);seal::Ciphertext zero;token_enc->encrypt_zero(zero);eval.add_inplace(ct,zero);
          io.send_data(ct.data(1),limbs*8);
          for(size_t coordinate=0;coordinate<512;++coordinate){const auto target=packed_target(coordinate);
            phases[coordinate]=rns_coefficient(ct.data(0),n,target,primes);masks[coordinate]=mask[target];}
        }else{
          std::vector<uint64_t> component(limbs);TokenWordsWiper wipe_component{component};io.recv_data(component.data(),limbs*8);
          for(size_t prime=0;prime<k;++prime){auto* values=component.data()+prime*n;
            for(size_t j=0;j<n;++j)if(values[j]>=primes[prime].value())throw std::runtime_error("E_BFV_COMPONENT");
            seal::util::ntt_negacyclic_harvey(values,cd->small_ntt_tables()[prime]);
            seal::util::dyadic_product_coeffmod(values,token_final_sk.data().data()+prime*n,n,primes[prime],values);
            seal::util::inverse_ntt_negacyclic_harvey(values,cd->small_ntt_tables()[prime]);}
          for(size_t coordinate=0;coordinate<512;++coordinate)
            phases[coordinate]=rns_coefficient(component.data(),n,packed_target(coordinate),primes);
        }
        if(stats)stats->return_phase_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-return_start).count();
        const auto gc_start=stats?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        op.reserve_material("token_exact_tile_GC",512);
        auto shares=decoder.decode_token_tile(phases,masks,modulus_product(primes));TokenWordsWiper wipe_shares{shares};
        std::copy(shares.begin(),shares.end(),result[slot].begin()+tile);
        if(stats){++stats->returned_tiles;stats->gc_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-gc_start).count();}
      }
    }
#ifdef PRAG_EXPERIMENTAL_TOKEN_GPU
    close_gpu();
#endif
    return result;
  }catch(...){for(auto& row:result)OPENSSL_cleanse(row.data(),row.size()*sizeof(Word));
#ifdef PRAG_EXPERIMENTAL_TOKEN_GPU
    // Preserve the original protocol/operator error. Cleanup failure is
    // separately observable in telemetry and as a fixed nonsecret error code.
    try{close_gpu();}catch(...){std::cerr<<"E_GPU_TOKEN_CLEANUP_DURING_FAILURE\n";}
#endif
    throw;}
}
