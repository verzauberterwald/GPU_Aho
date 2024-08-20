#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <queue>
#include <vector>

// Definition of the automaton node structure
struct automaton_node {
  std::map<unsigned char, automaton_node *> child;
  automaton_node *suffix_link;
  automaton_node *output_link;
  int pattern_ind;
};

typedef struct automaton_node node;

// Function to add a new node
node *add_node();
// Function to build the automata using the input patterns
void build_automata(node *root, std::vector<std::string> &patterns,
                    size_t &size);
// Function to build failure and output links for the automaton nodes
void build_failure_output_links(node *root);
