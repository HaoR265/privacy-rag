// Source-only candidate. No global plan or secret workspace cache.
static void validate_stream(const StreamPlan&p){
 if(p.wires<180||p.wires>40000||p.roots.size()!=60)throw std::runtime_error("STREAM_SHAPE");
 const uint32_t starts[]={0,58,3532},counts[]={58,3474,58};
 std::vector<uint8_t> produced(p.wires);std::fill(produced.begin(),produced.begin()+180,1);
 for(unsigned z=0;z<3;++z){auto&q=p.phases[z];if(q.gate_begin!=starts[z]||q.gates!=counts[z]||q.nodes.empty()||q.levels.size()<2||q.levels.front()!=0||q.levels.back()!=q.nodes.size()||q.live_in!=(z?p.phases[z-1].live_out:std::vector<uint32_t>{}))throw std::runtime_error("STREAM_PHASE");
  std::vector<int> lv(p.wires,-1);std::fill(lv.begin(),lv.begin()+180,0);for(auto w:q.live_in){if(w<180||w>=p.wires||!produced[w]||lv[w]>=0)throw std::runtime_error("STREAM_LIVE_IN");lv[w]=0;}
  std::vector<uint8_t> gids(q.gates);for(size_t l=1;l<q.levels.size();++l){if(q.levels[l]<q.levels[l-1])throw std::runtime_error("STREAM_LEVEL");for(size_t i=q.levels[l-1];i<q.levels[l];++i){auto a=q.nodes[i];if(a.op>2||a.a>=p.wires||a.b>=p.wires||a.out>=p.wires||produced[a.out]||lv[a.a]<0||lv[a.b]<0||lv[a.a]>=int(l)||lv[a.b]>=int(l))throw std::runtime_error("STREAM_DEPENDENCY");produced[a.out]=1;lv[a.out]=l;if(a.op==1){if(a.gate<q.gate_begin||a.gate>=q.gate_begin+q.gates||gids[a.gate-q.gate_begin]++)throw std::runtime_error("STREAM_GATE");}}}
  for(auto g:gids)if(g!=1)throw std::runtime_error("STREAM_GATE_COUNT");std::vector<uint8_t> seen(p.wires);for(auto w:q.live_out)if(w<180||w>=p.wires||lv[w]<0||seen[w]++)throw std::runtime_error("STREAM_LIVE_OUT");if(z==2){if(!q.live_out.empty())throw std::runtime_error("STREAM_LAST_LIVE");for(auto w:p.roots)if(w>=p.wires||lv[w]<0)throw std::runtime_error("STREAM_ROOT");}
 }
 for(auto used:produced)if(!used)throw std::runtime_error("STREAM_UNUSED_WIRE");
}
template<bool G>__global__ void stream_kernel(Label*work,uint32_t wires,const Instruction*nodes,const uint32_t*levels,uint32_t nl,const uint32_t*livein,uint32_t ni,const uint32_t*liveout,uint32_t no,const uint32_t*roots,const Label*ci,const Label*si,const Label*boundary_in,Label*boundary_out,Label*outputs,Label*tables,const Config*cfg,uint32_t phase,uint32_t gb,uint32_t gates,size_t n,size_t off,size_t k){
 size_t lane=blockIdx.x,global=off+lane;Label*all=work+lane*wires;
 if(threadIdx.x==0){all[0]=cfg->zero;all[1]=cfg->one;}
 for(uint32_t j=threadIdx.x;j<178;j+=blockDim.x)all[2+j]=j<59?ci[59*global+j]:j<118?si[59*global+j-59]:j<177?si[59*n+59*global+j-118]:si[118*n+global];
 for(uint32_t j=threadIdx.x;j<ni;j+=blockDim.x)all[livein[j]]=boundary_in[global*ni+j];__syncthreads();
 for(uint32_t l=1;l<nl;++l){for(uint32_t i=levels[l-1]+threadIdx.x;i<levels[l];i+=blockDim.x){auto op=nodes[i];Label a=all[op.a],b=all[op.b];if(op.op==0){all[op.out]=X(a,b);continue;}if(op.op==2){all[op.out]=X(a,cfg->one);continue;}size_t local=lane*gates+op.gate-gb;uint64_t globalgate=uint64_t(gb)*n+global*gates+op.gate-gb,id=2*(cfg->first+globalgate);Label w;
 if(G){Label d=cfg->delta,ha[2]={a,X(a,d)},hb[2]={b,X(b,d)};hash(ha,cfg->seed,id);hash(hb,cfg->seed,id+1);Label t0=X(X(ha[0],ha[1]),mask(d,b.w[0]&1)),t1=X(hb[0],hb[1]);tables[2*local]=t0;tables[2*local+1]=X(t1,a);w=X(X(X(ha[0],mask(t0,a.w[0]&1)),hb[0]),mask(t1,b.w[0]&1));}
 else {Label ha[1]={a},hb[1]={b};hash(ha,cfg->seed,id);hash(hb,cfg->seed,id+1);w=X(X(X(X(ha[0],hb[0]),mask(tables[2*local],a.w[0]&1)),mask(tables[2*local+1],b.w[0]&1)),mask(a,b.w[0]&1));}all[op.out]=w;
 }__syncthreads();}
 for(uint32_t j=threadIdx.x;j<no;j+=blockDim.x)boundary_out[global*no+j]=all[liveout[j]];
 if(phase==2)for(uint32_t j=threadIdx.x;j<60;j+=blockDim.x)outputs[j<59?59*global+j:59*n+global]=all[roots[j]];
}
struct StreamResources {
 struct Allocation{void*p;size_t bytes;};std::vector<Allocation>dev,host;cudaStream_t streams[2]{};cudaEvent_t events[2]{};size_t bytes=0,hostbytes=0;bool closed=false;
 void*alloc(size_t n){dev.push_back({nullptr,n});ck(cudaMalloc(&dev.back().p,n));bytes+=n;return dev.back().p;}
 void*pinned(size_t n){host.push_back({nullptr,n});ck(cudaHostAlloc(&host.back().p,n,cudaHostAllocDefault));hostbytes+=n;return host.back().p;}
 void close(){cudaError_t error=cudaSuccess;auto check=[&](cudaError_t e){if(error==cudaSuccess&&e!=cudaSuccess)error=e;};for(auto s:streams)if(s)check(cudaStreamSynchronize(s));ck(error);for(auto&a:dev)if(a.p)check(cudaMemset(a.p,0,a.bytes));check(cudaDeviceSynchronize());ck(error);for(auto&a:dev)if(a.p){auto e=cudaFree(a.p);if(e==cudaSuccess)a.p=nullptr;check(e);}for(auto&a:host)if(a.p){OPENSSL_cleanse(a.p,a.bytes);auto e=cudaFreeHost(a.p);if(e==cudaSuccess)a.p=nullptr;check(e);}for(auto&e:events)if(e){auto r=cudaEventDestroy(e);if(r==cudaSuccess)e=nullptr;check(r);}for(auto&s:streams)if(s){auto r=cudaStreamDestroy(s);if(r==cudaSuccess)s=nullptr;check(r);}ck(error);closed=true;}
 ~StreamResources(){if(!closed)try{close();}catch(...){cleanup_fail_stop("GPU_STREAM_LAST_RESORT_CLEANUP_FAILED");}}
};
template<bool G>static Times stream_api(const DeviceBinding&target,const StreamPlan&p,size_t n,const void*ci,const void*si,void*out,Label seed,Label delta,Label zero,Label one,uint64_t first,const Transfer&transfer){
 auto began=Clock::now();WipeGuard dw(&delta,sizeof(delta));validate_stream(p);if(!n||n>6144||!ci||!si||!out||!transfer||first>UINT64_MAX/2-3590*n||(G&&!(delta.w[0]&1)))throw std::runtime_error("STREAM_ARGUMENT");
 constexpr size_t tile=256;size_t nk=std::min(n,tile),maxnodes=0,maxlevels=0,maxlive=0;for(auto&q:p.phases){maxnodes=std::max(maxnodes,q.nodes.size());maxlevels=std::max(maxlevels,q.levels.size());maxlive=std::max(maxlive,std::max(q.live_in.size(),q.live_out.size()));}
 size_t metabytes=maxnodes*sizeof(Instruction)+(maxlevels+2*maxlive+60)*4,workbytes=p.wires*nk*16,tablebytes=3474*nk*32,boundbytes=(p.phases[0].live_out.size()+p.phases[1].live_out.size())*n*16;
 size_t device_total=178*n*16+60*n*16+boundbytes+metabytes+sizeof(Config)+2*(workbytes+tablebytes);
 if(device_total>256*1024*1024||2*tablebytes>128*1024*1024)throw std::runtime_error("STREAM_RESOURCE_LIMIT");
 Times t;DeviceScope selected(target);StreamResources r;Config cfg{seed,delta,first,zero,one};WipeGuard cw(&cfg,sizeof(cfg));
 try{
 auto*c=static_cast<Label*>(r.alloc(59*n*16));auto*s=static_cast<Label*>(r.alloc(119*n*16));auto*result=static_cast<Label*>(r.alloc(60*n*16));auto*b0=static_cast<Label*>(r.alloc(p.phases[0].live_out.size()*n*16));auto*b1=static_cast<Label*>(r.alloc(p.phases[1].live_out.size()*n*16));auto*dc=static_cast<Config*>(r.alloc(sizeof(Config)));auto*nodes=static_cast<Instruction*>(r.alloc(maxnodes*sizeof(Instruction)));auto*levels=static_cast<uint32_t*>(r.alloc(maxlevels*4));auto*lin=static_cast<uint32_t*>(r.alloc(maxlive*4));auto*lout=static_cast<uint32_t*>(r.alloc(maxlive*4));auto*roots=static_cast<uint32_t*>(r.alloc(60*4));Label*work[2],*dt[2];void*ht[2];
 for(unsigned i=0;i<2;++i){work[i]=static_cast<Label*>(r.alloc(workbytes));dt[i]=static_cast<Label*>(r.alloc(tablebytes));ht[i]=r.pinned(tablebytes);ck(cudaStreamCreateWithFlags(&r.streams[i],cudaStreamNonBlocking));ck(cudaEventCreateWithFlags(&r.events[i],cudaEventDisableTiming));}t.device_bytes=r.bytes;t.initialize=elapsed(began);
 auto at=Clock::now();ck(cudaMemcpy(c,ci,59*n*16,cudaMemcpyHostToDevice));ck(cudaMemcpy(s,si,119*n*16,cudaMemcpyHostToDevice));ck(cudaMemcpy(dc,&cfg,sizeof(cfg),cudaMemcpyHostToDevice));ck(cudaMemcpy(roots,p.roots.data(),60*4,cudaMemcpyHostToDevice));t.upload_bytes=178*n*16+sizeof(cfg)+240;t.upload=elapsed(at);
 for(unsigned phase=0;phase<3;++phase){auto&q=p.phases[phase];at=Clock::now();ck(cudaMemcpy(nodes,q.nodes.data(),q.nodes.size()*sizeof(Instruction),cudaMemcpyHostToDevice));ck(cudaMemcpy(levels,q.levels.data(),q.levels.size()*4,cudaMemcpyHostToDevice));if(!q.live_in.empty())ck(cudaMemcpy(lin,q.live_in.data(),q.live_in.size()*4,cudaMemcpyHostToDevice));if(!q.live_out.empty())ck(cudaMemcpy(lout,q.live_out.data(),q.live_out.size()*4,cudaMemcpyHostToDevice));t.upload+=elapsed(at);t.upload_bytes+=q.nodes.size()*sizeof(Instruction)+(q.levels.size()+q.live_in.size()+q.live_out.size())*4;
  size_t tiles=(n+tile-1)/tile;bool pending[2]{};auto finish=[&](unsigned i){if(pending[i]){ck(cudaEventSynchronize(r.events[i]));pending[i]=false;}};
  auto launch=[&](size_t j){unsigned i=j%2;size_t off=j*tile,k=std::min(tile,n-off),bytes=q.gates*k*32;if(!G){ck(cudaMemcpyAsync(dt[i],ht[i],bytes,cudaMemcpyHostToDevice,r.streams[i]));t.upload_bytes+=bytes;}stream_kernel<G><<<k,128,0,r.streams[i]>>>(work[i],p.wires,nodes,levels,q.levels.size(),lin,q.live_in.size(),lout,q.live_out.size(),roots,c,s,phase==1?b0:phase==2?b1:nullptr,phase==0?b0:phase==1?b1:nullptr,result,dt[i],dc,phase,q.gate_begin,q.gates,n,off,k);ck(cudaGetLastError());if(G){ck(cudaMemcpyAsync(ht[i],dt[i],bytes,cudaMemcpyDeviceToHost,r.streams[i]));t.download_bytes+=bytes;}ck(cudaEventRecord(r.events[i],r.streams[i]));pending[i]=true;};
  at=Clock::now();if(G){for(size_t j=0;j<std::min(size_t(2),tiles);++j)launch(j);for(size_t j=0;j<tiles;++j){unsigned i=j%2;finish(i);transfer(ht[i],q.gates*std::min(tile,n-j*tile)*32);OPENSSL_cleanse(ht[i],q.gates*std::min(tile,n-j*tile)*32);if(j+2<tiles)launch(j+2);}}
  else {for(size_t j=0;j<tiles;++j){unsigned i=j%2;finish(i);transfer(ht[i],q.gates*std::min(tile,n-j*tile)*32);launch(j);}}
  for(unsigned i=0;i<2;++i)finish(i);t.kernel+=elapsed(at); // includes transfer/waits: deliberately not a kernel-only measure
 }
 at=Clock::now();ck(cudaMemcpy(out,result,60*n*16,cudaMemcpyDeviceToHost));t.download=elapsed(at);t.download_bytes+=60*n*16;at=Clock::now();r.close();cw.clear();dw.clear();selected.restore();t.cleanup=elapsed(at);t.cleaned=true;t.host_config_wiped=cw.wiped;t.whole_api=elapsed(began);return t;
 }catch(...){auto original=std::current_exception();r.close();cw.clear();dw.clear();selected.restore();std::rethrow_exception(original);}
}
Times stream_complete(bool g,const DeviceBinding&d,const StreamPlan&p,size_t n,const void*c,const void*s,void*out,Label seed,Label delta,Label zero,Label one,uint64_t first,const Transfer&io){WipeGuard w(&delta,sizeof(delta));return g?stream_api<true>(d,p,n,c,s,out,seed,delta,zero,one,first,io):stream_api<false>(d,p,n,c,s,out,seed,{},zero,one,first,io);}
