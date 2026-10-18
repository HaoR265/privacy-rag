#pragma once
#include "GC/circuit_execution.h"
#include "prag/gc_batch_channel.hpp"
#include <array>
#include <vector>
namespace prag::silu_gpu_bridge {
// Independent diagnostic candidate. Configuration is explicit, validated and
// source-bound by the controller; an unavailable backend never falls back.
struct Binding { std::array<unsigned char,32> digest; };
Binding prepare(); // public template/device preparation; included in call time
struct Configuration {
 sci::block128 seed{}, delta{}, zero{}, one{};
 ~Configuration();
 Configuration() = default;
 Configuration(const Configuration&)=delete;
 Configuration& operator=(const Configuration&)=delete;
};
std::vector<sci::block128> execute(bool sender, size_t full_lanes,
 const std::vector<sci::block128>& ci,
 const std::vector<sci::block128>& si,
 const Configuration&, GcBatchChannel&);
}
