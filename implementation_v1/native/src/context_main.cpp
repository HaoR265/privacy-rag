#include "prag/bfv.hpp"
#include <iostream>
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      return 2;
    prag::write_bfv_context_artifact(argv[1]);
    std::cout << "BFV_CONTEXT_MATERIALIZED_PRIMITIVE_PROFILE_VERIFIED_FULL_SYSTEM_GATED\n";
    return 0;
  } catch (const std::exception &) {
    std::cerr << "E_CONTEXT\n";
    return 1;
  }
}
