#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <vector>

#include "../include/base.hpp"
#include "../include/vector_builder.hpp"

void search_pattern(Node vec[], std::string &text, auto &indices) {

  int cur = 0;

  for (int i = 0; i < text.length(); i++) {
    int c = text[i] - 'a';
    if (vec[cur].bitmap & (1 << c)) { // if link to character exists follow it

      // Apply the mask to the number to zero out bits beyond the i-th bit
      unsigned char maskedNumber = vec[cur].bitmap & ((1 << (c + 1)) - 1);
      // Move to child node
      cur = vec[cur].offset + __builtin_popcount(maskedNumber) - 1;

      if (vec[cur].pattern_idx >= 0) {
        indices[vec[cur].pattern_idx].push_back(i);
      }
      int temp = vec[cur].output_link;
      while (temp != 0) {
        indices[vec[temp].pattern_idx].push_back(i);
        temp = vec[temp].output_link;
      }
    } else {
      while (cur != 0 && ((vec[cur].bitmap & (1 << c)) == 0)) {
        cur = vec[cur].fail_link;
      }

      if ((vec[cur].bitmap & (1 << c)))
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

  std::ifstream in_file(inputFileName);
  std::stringstream buffer;
  std::string text;

  if (in_file.is_open()) {
    buffer << in_file.rdbuf();
    text = buffer.str();
    in_file.close();
  } else {
    std::cerr << "Unable to open file" << std::endl;
  }

  for (int j = 0; j < text.size(); ++j) {
    if (text[j] == 'g')
      text[j] = 'b';
    if (text[j] == 't')
      text[j] = 'd';
  }

  int k = 6;
  std::vector<std::string> patterns;
  std::ifstream file(patternsFileName);
  std::string line;
  int count = 0;

  if (file.is_open()) {
    while (std::getline(file, line) && count < k) {
      patterns.push_back(line);
      // patterns[patterns.size() - 1].pop_back();
      count++;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file" << std::endl;
  }

  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < patterns[i].size(); ++j) {
      if (patterns[i][j] == 'g')
        patterns[i][j] = 'b';
      if (patterns[i][j] == 't')
        patterns[i][j] = 'd';
    }
  }

  size_t size = 1;
  node *root = add_node();

  build_automata(root, patterns, size);

  build_failure_output_links(root);

  Node *vec = new Node[size];
  memset(vec, 0, size * sizeof(Node));

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

  // Print results
  for (int i = 0; i < patterns.size(); i++) {
    std::cout << "Total occurrences of \"";
    for (int j = 0; j < patterns[i].size(); ++j) {
      if (patterns[i][j] == 'b')
        std::cout << 'g';
      else if (patterns[i][j] == 'd')
        std::cout << 't';
      else
        std::cout << patterns[i][j];
    }

    std::cout << "\": " << indices[i].size() << std::endl;
    std::cout << "Positions: ";
    for (int j = 0; j < indices[i].size(); j++) {
      std::cout << indices[i][j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}