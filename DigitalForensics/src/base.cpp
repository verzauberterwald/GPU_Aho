#include "../include/base.hpp"

node *add_node() {
  node *temp = new node;
  temp->suffix_link = nullptr;
  temp->output_link = nullptr;
  temp->pattern_ind = -1;
  return temp;
}

void build_automata(node *root, std::vector<std::string> &patterns,
                    size_t &size) {
  for (int i = 0; i < patterns.size(); i++) {
    node *cur = root;
    for (auto c : patterns[i]) {
      if (cur->child.count(c))
        cur = cur->child[c];
      else {
        node *new_child = add_node();
        size++;
        cur->child.insert({c, new_child});
        cur = new_child;
      }
    }
    cur->pattern_ind = i;
  }
}

void build_failure_output_links(node *root) {
  root->suffix_link = root;
  std::queue<node *> qu;
  for (auto &it : root->child) {
    qu.push(it.second);
    it.second->suffix_link = root;
  }

  while (!qu.empty()) {
    node *cur_state = qu.front();
    qu.pop();

    for (auto &it : cur_state->child) {
      char c = it.first;
      node *temp = cur_state->suffix_link;
      while (temp != root && temp->child.count(c) == 0)
        temp = temp->suffix_link;

      if (temp->child.count(c))
        it.second->suffix_link = temp->child[c];
      else
        it.second->suffix_link = root;

      qu.push(it.second);
    }

    if (cur_state->suffix_link->pattern_ind >= 0)
      cur_state->output_link = cur_state->suffix_link;
    else
      cur_state->output_link = cur_state->suffix_link->output_link;
  }
}
