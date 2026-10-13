cmake_minimum_required(VERSION 3.16)
project(prag_tile108_candidate LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
find_package(OpenSSL 3 REQUIRED)
find_package(Threads REQUIRED)
find_package(SEAL 4.1.2 EXACT REQUIRED PATHS "${ROOT}/third_party/SEAL/build/cmake" NO_DEFAULT_PATH)
add_executable(prag_production_matrix_probe matrix_probe.cpp bfv_candidate.cpp)
target_include_directories(prag_production_matrix_probe PRIVATE "${ROOT}/native/include" "${ROOT}/third_party/EzPC/SCI/src")
target_compile_options(prag_production_matrix_probe PRIVATE -maes -msse4.1 -mavx -mavx2 -Wno-deprecated-declarations -Wno-ignored-attributes)
target_compile_definitions(prag_production_matrix_probe PRIVATE EMP_USE_RANDOM_DEVICE=1 PRAG_NATIVE_VERSION="1.0.8-candidate")
# The candidate supplies the complete BFV object. The static linker selects
# only unchanged Channel/SecureOps/GC objects from the formed production lib.
target_link_libraries(prag_production_matrix_probe PRIVATE "${ROOT}/native/build-matrix107/libprag_crypto.a" SEAL::seal OpenSSL::SSL OpenSSL::Crypto Threads::Threads gmpxx gmp)
