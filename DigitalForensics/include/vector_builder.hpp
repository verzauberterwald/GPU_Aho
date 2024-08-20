#pragma once
#include "../include/base.hpp"
#include <iostream>
#include <map>
#include <queue>
#include <vector>

// Structure of a node in the trie.
struct Node {
  unsigned int bitmap[8]; // Bitmap for fast lookup
  unsigned int offset;    // Offset to child nodes in array
  int fail_link;          // Link to node on failure
  int output_link;        // Link to node on output match
  int pattern_idx;        // Index of the pattern in the trie
};

// Function to populate the flat vector of Nodes
void build_vec(node *cur, Node vec[], size_t &size,
               int parent_idx /*index of cur in vector*/,
               size_t &idx /*next available space in vector*/);
// Function to populate fail and output links after building the trie vector
void fill_data(Node vec[], size_t &size);
// Function flatten the finite automaton into a vector
void flatten_tree(node *cur, Node vec[], size_t &size,
                  int parent_idx /*index of cur in vector*/,
                  size_t &idx /*next available space in vector*/);
