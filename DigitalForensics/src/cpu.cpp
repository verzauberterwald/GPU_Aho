#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <vector>

#include "../include/base.hpp"
#include "../include/read_helper.hpp"
#include "../include/vector_builder.hpp"

void search_pattern(Node vec[], std::string &text, auto &indices) {

  int cur = 0;

  for (int i = 0; i < text.length(); i++) {
    int c = static_cast<unsigned char>(text[i]);
    unsigned int pos = c - (c / 32) * 32;

    if (vec[cur].bitmap[c / 32] &
        (1 << pos)) { // if link to character exists follow it

      // Apply the mask to the number to zero out bits beyond the i-th bit
      unsigned int mask = ((1LL << (pos + 1LL)) - 1LL);
      unsigned int maskedNumber = vec[cur].bitmap[c / 32] & mask;

      int bits_before = 0;
      for (int j = 0; j < (c / 32); ++j) {
        bits_before += __builtin_popcount(vec[cur].bitmap[j]);
      }
      cur =
          vec[cur].offset + bits_before + __builtin_popcount(maskedNumber) - 1;

      if (vec[cur].pattern_idx >= 0) {
        indices[vec[cur].pattern_idx].push_back(i);
      }
      int temp = vec[cur].output_link;
      while (temp != 0) {
        indices[vec[temp].pattern_idx].push_back(i);
        temp = vec[temp].output_link;
      }
    } else {
      while (cur != 0 && ((vec[cur].bitmap[c / 32] & (1 << pos)) == 0)) {
        cur = vec[cur].fail_link;
      }

      if ((vec[cur].bitmap[c / 32] & (1 << pos)))
        i--;
    }
  }
}

int main(int argc, char *argv[]) {

  if (argc != 3 || (argc > 1 && std::string(argv[1]) == "--help")) {
    std::cout << "Usage: " << argv[0] << " <input_file> <patterns_file>\n";
    std::cout << "  <input_file> : The name of the text file to process.\n";
    std::cout << "  <patterns_file> : The file containing patterns to "
                 "search within the input file.\n";
    return 1;
  }

  std::string inputFileName = argv[1];
  std::string patternsFileName = argv[2];

  const std::string isoFilePath = inputFileName;
  std::string text = readISOFileAsBytes(isoFilePath);

  std::cout << "File size in bytes: " << text.size() << std::endl;

  int k = 4;
  std::string filename = patternsFileName;
  std::vector<std::string> patterns = readHexFile(filename, k);

  size_t size = 1;
  node *root = add_node();

  build_automata(root, patterns, size);

  build_failure_output_links(root);

  Node *vec = new Node[size];
  memset(vec, 0, sizeof(Node) * size);

  size_t next_idx = 1;
  flatten_tree(root, vec, size, 0, next_idx);

  std::vector<std::vector<int>> indices(k, std::vector<int>());

  auto start = std::chrono::high_resolution_clock::now();
  search_pattern(vec, text, indices);
  auto end = std::chrono::high_resolution_clock::now();

  // Duration in milliseconds
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "Time taken: " << duration.count() << " milliseconds"
            << std::endl;

  for (int i = 0; i < patterns.size(); i++) {
    std::cout << "\nTotal occurrences of \"";
    std::cout << patterns[i] << "\": " << indices[i].size();
    std::cout << " Positions\n";
    for (int j = 0; j < indices[i].size(); ++j) {
      std::cout << indices[i][j] << ' ';
    }
    std::cout << '\n';
  }

  return 0;
}