#pragma once
#include <cstdint>
#include <vector>
#include <memory>
namespace seal {class SEALContext;class Ciphertext;}
namespace matrix_gpu {
constexpr size_t degree=16384,primes=6;
struct Cleanup{int first_cuda_error=0;size_t wiped=0,freed=0;bool closed=false;};
class Matrix {
 struct State;std::shared_ptr<State> s;
 void initialize(const std::vector<uint64_t>& input,const std::vector<uint64_t>& q,const std::vector<uint64_t>& roots,const std::vector<uint64_t>& inv,const std::vector<uint64_t>& invn);
 std::vector<uint64_t> evaluate_impl(const std::vector<uint64_t>&,const std::vector<uint64_t>*);
public:
 Matrix(const seal::SEALContext&,const std::vector<seal::Ciphertext>&);
 ~Matrix();Matrix(const Matrix&)=delete;Matrix& operator=(const Matrix&)=delete;
 // Exactly [input_blocks][16384] packed canonical ring59 coefficients.
 // Returns coefficient form [2][6][16384]. No secret keys or reconstruction.
 std::vector<uint64_t> evaluate(const std::vector<uint64_t>&);
 // Output still [2][6][16384], but non-target c0 is ZERO and is NOT
 // a full equivalent ciphertext. Only coefficientwise postprocessing allowed.
 // Public strictly increasing coefficient indices, 1..512 entries.
 std::vector<uint64_t> evaluate_selective(const std::vector<uint64_t>&,const std::vector<uint64_t>& targets);
 size_t device_bytes()const;Cleanup close()noexcept;
};
}
