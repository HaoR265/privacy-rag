#pragma once
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace prag {
struct MatrixPacking {std::size_t reduction, columns, returns, inputs;};
inline MatrixPacking matrix_packing_qk_reverse64(std::size_t b) {
  if(!b||b>64)throw std::invalid_argument("E_QK_REVERSE64_SHAPE");
  // Dedicated K=128,O=1: one full reduction, one return, no wrap.
  return {128,1,1,1};
}
inline MatrixPacking matrix_packing(std::size_t b,std::size_t k,std::size_t o) {
  if(!b||b>16||!k||k>8192||!o||o>155648)throw std::invalid_argument("E_MATRIX_SHAPE");
  MatrixPacking best{};uint64_t cost=std::numeric_limits<uint64_t>::max();
  const size_t gc=512/b;
  for(size_t d=16;d<=8192/b;d*=2) {
    const size_t cap=std::min(gc,8192/(b*d));
    const size_t full=o/gc,tail=o%gc;
    const size_t returns=full*((gc+cap-1)/cap)+(tail?(tail+cap-1)/cap:0);
    const size_t inputs=(k+d-1)/d;
    const uint64_t bytes=inputs*UINT64_C(1572960)+returns*UINT64_C(786528);
    if(bytes<cost){cost=bytes;best={d,cap,returns,inputs};}
  }
  return best; // Public dimensions only; ties use smallest reduction.
}
}
