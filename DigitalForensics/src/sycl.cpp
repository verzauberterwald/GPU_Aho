#include <CL/sycl.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <vector>

#include "../include/base.hpp"
#include "../include/read_helper.hpp"
#include "../include/vector_builder.hpp"

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

  int text_length = text.length();
  std::cout << "File size in bytes: " << text.size() << std::endl;

  int k = 4; // Number of patterns
  std::string filename = patternsFileName;
  std::vector<std::string> patterns = readHexFile(filename, k);

  int max_pattern_length = 0;
  for (int i = 0; i < k; ++i) {
    max_pattern_length =
        std::max(max_pattern_length, (int)(patterns[i].size()));
  }

  size_t size = 1;
  node *root = add_node();

  build_automata(root, patterns, size);

  build_failure_output_links(root);

  Node *vec = new Node[size];
  memset(vec, 0, sizeof(Node) * size);

  size_t next_idx = 1;
  flatten_tree(root, vec, size, 0, next_idx);

  // Host vectors to store sizes and indices
  int max_size = 100;
  std::vector<unsigned int> h_sizes(patterns.size(), 0);
  std::vector<int> h_indices(patterns.size() * max_size, 0);
  try {
    // Select a GPU device
    sycl::gpu_selector gpuSel;
    sycl::queue q(gpuSel);

    // Print out the device information
    std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    {
      sycl::buffer<Node, 1> vec_buffer(vec, sycl::range<1>(size));
      sycl::buffer<char, 1> text_buffer(text.c_str(),
                                        sycl::range<1>(text_length));
      sycl::buffer<unsigned int, 1> sizes_buffer(
          h_sizes.data(), sycl::range<1>(patterns.size()));
      sycl::buffer<int, 1> indices_buffer(
          h_indices.data(), sycl::range<1>(patterns.size() * max_size));

      int thread_pool = 4;

      q.submit([&](sycl::handler &h) {
        auto vec_acc = vec_buffer.get_access<sycl::access::mode::read>(h);
        auto text_acc = text_buffer.get_access<sycl::access::mode::read>(h);
        auto sizes_acc =
            sizes_buffer.get_access<sycl::access::mode::read_write>(h);
        auto indices_acc =
            indices_buffer.get_access<sycl::access::mode::read_write>(h);

        h.parallel_for(sycl::range<1>(thread_pool), [=](sycl::id<1> tid) {
          int start_index = (text_length / thread_pool) * tid.get(0);
          int end_index = start_index + (text_length / thread_pool) +
                          max_pattern_length - 2;
          if (end_index >= text_length)
            end_index = text_length - 1;

          int cur = 0;
          for (int i = start_index; i <= end_index; i++) {
            int c = static_cast<unsigned char>(text_acc[i]);
            unsigned int pos = c - (c / 32) * 32;

            if (vec_acc[cur].bitmap[c / 32] & (1 << pos)) {
              unsigned int mask = ((1LL << (pos + 1LL)) - 1LL);

              unsigned int maskedNumber = vec_acc[cur].bitmap[c / 32] & mask;

              int bits_before = 0;
              for (int j = 0; j < (c / 32); ++j) {
                bits_before += sycl::popcount(vec_acc[cur].bitmap[j]);
              }

              cur = vec_acc[cur].offset + bits_before +
                    sycl::popcount(maskedNumber) - 1;

              int overlap = ((start_index != 0) &&
                             (i - start_index + 1 < max_pattern_length));

              if (overlap) {
                continue;
              }

              if (vec_acc[cur].pattern_idx >= 0) {

                auto idx =
                    sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(
                        sizes_acc[vec_acc[cur].pattern_idx])
                        .fetch_add(1);

                indices_acc[vec_acc[cur].pattern_idx * max_size + idx] = i;
              }
              int temp = vec_acc[cur].output_link;

              while (temp != 0) {
                int output_pattern_index = vec_acc[temp].pattern_idx;
                auto idx =
                    sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(
                        sizes_acc[output_pattern_index])
                        .fetch_add(1);

                indices_acc[output_pattern_index * max_size + idx] = i;

                temp = vec_acc[temp].output_link;
              }
            } else {
              while (cur != 0 && ((vec_acc[cur].bitmap[c / 32] & (1 << pos)) ==
                                  0)) { // follow suffix links till matching
                                        // suffix or root is found
                cur = vec_acc[cur].fail_link;
              }

              if ((vec_acc[cur].bitmap[c / 32] & (1 << pos)))
                i--;
            }
          }
        });
      });
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Duration in milliseconds
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Time taken: " << duration.count() << " milliseconds"
              << std::endl;

    // Print results
    for (int i = 0; i < k; i++) {
      std::cout << "Total occurrences of \"" << patterns[i];
      std::cout << "\": " << h_sizes[i] << std::endl;
      std::cout << "Positions: ";
      for (int j = 0; j < h_sizes[i]; j++) {
        std::cout << h_indices[i * max_size + j] << " ";
      }
      std::cout << std::endl;
    }

  } catch (const std::exception &e) {
    std::cerr << "An exception has been caught: " << e.what() << std::endl;
    std::cerr << "Could not find a GPU device." << std::endl;
    return -1;
  }
  return 0;
}

// clang-format off
// clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda src/sycl_ext.cpp -o sycl-app

//. /opt/intel/oneapi/setvars.sh --include-intel-llvm

// clang-format on