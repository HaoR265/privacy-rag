#pragma once
#include "prag/channel.hpp"
#include <chrono>
#include <fstream>
#include <map>
#include <sstream>
#include <sys/resource.h>
// Public diagnostics only: no operand, share, failure flag or output is logged.
struct PhaseMeter {
  prag::Channel& io; std::ofstream trace; std::ifstream ledger;
  std::map<std::string,uint64_t> reservations;
  PhaseMeter(prag::Channel& c,const std::string& prefix):io(c),trace(prefix+".trace.jsonl"),ledger(prefix+".ledger"){}
  void poll(){ledger.clear();std::string line;while(std::getline(ledger,line)){std::istringstream s(line);std::string seq,action,kind;uint64_t n;if(s>>seq>>action>>kind>>n && action=="reserved")reservations[kind]+=n;}}
  struct Scope {
    PhaseMeter& m;std::string name;uint64_t bytes,tx,rx,runs;int exceptions;
    std::map<std::string,uint64_t> before;
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
    Scope(PhaseMeter& m,std::string n):m(m),name(std::move(n)),bytes(m.io.bytes()),tx(m.io.frames_sent()),rx(m.io.frames_received()),runs(m.io.direction_runs()),exceptions(std::uncaught_exceptions()){
      m.poll();before=m.reservations;m.trace<<"{\"event\":\"begin\",\"phase\":\""<<name<<"\"}\n";m.trace.flush();
    }
    ~Scope(){m.poll();rusage usage{};getrusage(RUSAGE_SELF,&usage);
      m.trace<<"{\"event\":\""<<(std::uncaught_exceptions()>exceptions?"aborted":"end")<<"\",\"phase\":\""<<name<<"\",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"application_bytes\":"<<m.io.bytes()-bytes<<",\"frames_sent\":"<<m.io.frames_sent()-tx<<",\"frames_received\":"<<m.io.frames_received()-rx<<",\"direction_runs\":"<<m.io.direction_runs()-runs<<",\"peak_rss_kib\":"<<usage.ru_maxrss<<",\"reservations\":{";
      bool first=true;for(auto& [k,v]:m.reservations){auto delta=v-before[k];if(delta){if(!first)m.trace<<',';first=false;m.trace<<'"'<<k<<"\":"<<delta;}}
      m.trace<<"}}\n";m.trace.flush();
    }
  };
};
inline thread_local PhaseMeter* current_meter=nullptr;
