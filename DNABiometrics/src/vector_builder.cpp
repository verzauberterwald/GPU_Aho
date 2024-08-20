#include "../include/vector_builder.hpp"

std::map<automaton_node *, int> mapped_to;
std::map<int, automaton_node *> mapped_back;

void build_vec(node *cur, Node vec[], size_t &size,
               int parent_idx /*index of cur in vector*/,
               size_t &idx /*next available space in vector*/) {

  if (!cur->child.size())
    return;

  vec[parent_idx].offset = idx;
  size_t child_idx = idx;
  for (auto c : cur->child) {

    mapped_to[c.second] = idx;
    mapped_back[idx] = c.second;

    vec[parent_idx].bitmap |= (1 << (c.first - 'a'));

    idx++;
  }

  for (auto c : cur->child) {
    build_vec(c.second, vec, size, child_idx, idx);
    child_idx++;
  }
}

void fill_data(Node vec[], size_t &size) {

  for (int i = 0; i < size; ++i) {
    vec[i].fail_link = mapped_to[mapped_back[i]->suffix_link];
    vec[i].output_link = mapped_to[mapped_back[i]->output_link];
    vec[i].pattern_idx = mapped_back[i]->pattern_ind;
  }
}

void flatten_tree(node *cur, Node vec[], size_t &size,
                  int parent_idx /*index of cur in vector*/,
                  size_t &idx /*next available space in vector*/) {

  mapped_to[cur] = 0;
  mapped_back[0] = cur;

  build_vec(cur, vec, size, 0, idx);
  fill_data(vec, size);
}