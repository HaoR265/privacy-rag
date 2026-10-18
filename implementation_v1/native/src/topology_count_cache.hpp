#pragma once
#include "prag/batch.hpp"
#include <deque>
namespace prag {
// Public topology only; owned by one BatchOps instance and never shared.
// Full table comparison avoids digest collisions and includes every public bit.
class TopologyCountCache {
  struct Entry { BatchSpec spec; std::vector<Word> table; uint64_t gates; };
  std::deque<Entry> entries_;
  size_t table_words_ = 0;
public:
  static constexpr size_t max_entries = 32, max_table_words = 327680;
  bool find(const BatchSpec& spec, const std::vector<Word>& table, uint64_t& gates) const {
    for (const auto& e : entries_)
      if(e.spec.code == spec.code && e.spec.rows == spec.rows &&
         e.spec.cols == spec.cols && e.spec.parameters == spec.parameters && e.table == table) {
        gates=e.gates; return true;
      }
    return false;
  }
  void insert(const BatchSpec& spec, const std::vector<Word>& table, uint64_t gates) {
    if(table.size() > max_table_words) return;
    while(entries_.size() >= max_entries || table_words_ + table.size() > max_table_words) {
      table_words_ -= entries_.front().table.size(); entries_.pop_front();
    }
    entries_.push_back({spec,table,gates}); table_words_ += table.size();
  }
  size_t size() const { return entries_.size(); }
  size_t table_words() const { return table_words_; }
};
}
