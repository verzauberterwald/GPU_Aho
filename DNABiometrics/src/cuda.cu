#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <vector>

#include "../include/base.hpp"
#include "../include/vector_builder.hpp"

__global__ void search_pattern_cuda(const Node *const vec, const char *text,
                                    int text_length, int *indices,
                                    unsigned int *sizes, int max_size,
                                    unsigned int *locks, int thread_count,
                                    int max_pattern_length)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int start_index = (text_length / thread_count) * tid;
    int end_index =
        start_index + (text_length / thread_count) + max_pattern_length - 2;
    if (end_index >= text_length)
    {
        end_index = text_length - 1;
    }

    if (tid == 0)
    {
        start_index = 0;
    }

    int cur = 0;
    for (int i = start_index; i <= end_index; i++)
    {
        int c = text[i] - 'a';
        if (vec[cur].bitmap & (1 << c))
        {

            unsigned char maskedNumber = vec[cur].bitmap & ((1 << (c + 1)) - 1);

            cur = vec[cur].offset + __popc(maskedNumber) - 1;

            int overlap = ((start_index != 0) &&
                           (i - start_index + 1 < max_pattern_length));

            if (overlap)
            {
                continue;
            }

            if ((vec[cur].pattern_idx >= 0))
            {

                int pattern_index = vec[cur].pattern_idx;
                // Locking
                bool leaveLoop = false;
                while (!leaveLoop)
                {
                    if (atomicExch(&locks[pattern_index], 1u) != 0u)
                    {
                        // critical section
                        unsigned int idx =
                            atomicInc(&sizes[pattern_index], max_size);
                        atomicExch(&indices[pattern_index * max_size + idx], i);

                        leaveLoop = true;
                        // unlocking
                        atomicExch(&locks[pattern_index], 0u);
                    }
                }
            }

            int temp = vec[cur].output_link;
            while (temp != 0)
            {
                int output_pattern_index = vec[temp].pattern_idx;
                // Locking
                bool leaveLoop = false;
                while (!leaveLoop)
                {
                    if (atomicExch(&locks[output_pattern_index], 1u) != 0u)
                    {
                        //  Critical section
                        unsigned int idx =
                            atomicInc(&sizes[output_pattern_index], max_size);
                        atomicExch(
                            &indices[output_pattern_index * max_size + idx], i);

                        leaveLoop = true;
                        // Unlocking
                        atomicExch(&locks[output_pattern_index], 0u);
                    }
                }
                temp = vec[temp].output_link;
            }
        }
        else
        {
            while (cur != 0 && ((vec[cur].bitmap & (1 << c)) == 0))
            {
                cur = vec[cur].fail_link;
            }

            if ((vec[cur].bitmap & (1 << c)))
                i--;
        }
    }
}

int main(int argc, char *argv[])
{

    if (argc != 3 || (argc > 1 && std::string(argv[1]) == "--help"))
    {
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

    if (in_file.is_open())
    {
        buffer << in_file.rdbuf();
        text = buffer.str();
        in_file.close();
    }
    else
    {
        std::cerr << "Unable to open file" << std::endl;
    }

    for (int j = 0; j < text.size(); ++j)
    {
        if (text[j] == 'g')
            text[j] = 'b';
        if (text[j] == 't')
            text[j] = 'd';
    }

    int text_length = text.size();

    int k = 6;
    std::vector<std::string> patterns;
    std::ifstream file(patternsFileName);
    std::string line;
    int count = 0;

    if (file.is_open())
    {
        while (std::getline(file, line) && count < k)
        {
            patterns.push_back(line);
            // patterns[patterns.size() - 1].pop_back();
            count++;
        }
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file" << std::endl;
    }

    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < patterns[i].size(); ++j)
        {
            if (patterns[i][j] == 'g')
                patterns[i][j] = 'b';
            if (patterns[i][j] == 't')
                patterns[i][j] = 'd';
        }
    }

    int max_pattern_length = 0;
    for (int i = 0; i < k; ++i)
    {
        max_pattern_length = max(max_pattern_length, (int)(patterns[i].size()));
    }

    size_t size = 1;
    node *root = add_node();
    build_automata(root, patterns, size);

    build_failure_output_links(root);

    Node *vec = new Node[size];
    memset(vec, 0, sizeof(vec));

    std::cout << "Nr. patterns = " << k << " Nr. of nodes : " << size
              << " Size : " << size * sizeof(Node) << '\n';

    size_t next_idx = 1;
  flatten_tree(root, vec, size, 0, next_idx);

    // Host pointers
    unsigned int *h_sizes = new unsigned int[k]();
    int max_size = 200; // Maximum number of occurrences per pattern
    int *h_indices = new int[k * max_size];

    // Device pointers
    Node *d_vec;
    char *d_text;
    int *d_indices;
    unsigned int *d_sizes;

    // Allocate memory on the device
    cudaMalloc(&d_vec, sizeof(Node) * size);
    cudaMalloc(&d_text, sizeof(char) * text_length);
    cudaMalloc(&d_sizes, sizeof(unsigned int) * k);
    cudaMalloc(&d_indices, sizeof(int) * k * max_size);

    // Copy data from host to device
    cudaMemcpy(d_text, text.c_str(), sizeof(char) * text_length,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, h_sizes, sizeof(unsigned int) * k,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vec, sizeof(Node) * size, cudaMemcpyHostToDevice);

    unsigned int *d_locks;
    cudaMalloc(&d_locks, sizeof(unsigned int) * k);
    cudaMemset(d_locks, 0, sizeof(unsigned int) * k);

    int thread_count = 4;
    int block_count = 1;
    auto start = std::chrono::high_resolution_clock::now();

    search_pattern_cuda<<<block_count, thread_count>>>(
        d_vec, d_text, text_length, d_indices, d_sizes, max_size, d_locks,
        thread_count * block_count, max_pattern_length);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_indices, d_indices, sizeof(int) * k * max_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sizes, d_sizes, sizeof(unsigned int) * k,
               cudaMemcpyDeviceToHost);

    // Duration in milliseconds
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Time taken: " << duration.count() << " milliseconds"
              << std::endl;

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error after synchronize: "
                  << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    // Print results
    for (int i = 0; i < k; i++)
    {
        std::cout << "Total occurrences of \"";
        for (int j = 0; j < patterns[i].size(); ++j)
        {
            if (patterns[i][j] == 'b')
                std::cout << 'g';
            else if (patterns[i][j] == 'd')
                std::cout << 't';
            else
                std::cout << patterns[i][j];
        }

        std::cout << "\": " << h_sizes[i] << std::endl;
        std::cout << "Positions: ";
        // sort(h_indices[i], h_indices[i] + h_sizes[i]);
        for (int j = 0; j < h_sizes[i]; j++)
        {
            std::cout << h_indices[i * max_size + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_indices);
    cudaFree(d_sizes);
    cudaFree(d_text);
    cudaFree(d_vec);
    cudaFree(d_locks);
    delete[] h_indices;
    delete[] h_sizes;
    return 0;
}