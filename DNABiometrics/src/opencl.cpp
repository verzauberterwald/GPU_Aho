#include <CL/cl2.hpp>
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

#define checkError(err, msg)                                                   \
  if (err != CL_SUCCESS) {                                                     \
    fprintf(stderr, "%s failed: %d\n", msg, err);                              \
    exit(EXIT_FAILURE);                                                        \
  }

// Utility function to load OpenCL kernel source code
const char *loadKernelSource(const char *filename) {
  std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for reading: " << filename << std::endl;
    exit(1);
  }

  size_t size = file.tellg();
  char *buffer = new char[size + 1];
  file.seekg(0, std::ios::beg);
  file.read(buffer, size);
  buffer[size] = '\0';
  file.close();

  return buffer;
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

  size_t text_length = text.size();

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
  memset(vec, 0, sizeof(vec));

  size_t next_idx = 1;
  flatten_tree(root, vec, size, 0, next_idx);

  cl_int err;
  cl_uint numPlatforms;
  clGetPlatformIDs(0, nullptr, &numPlatforms);
  std::vector<cl_platform_id> platforms(numPlatforms);
  clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

  cl_device_id device_id = nullptr;
  for (auto platform : platforms) {
    cl_uint numDevices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (numDevices > 0) {
      std::vector<cl_device_id> devices(numDevices);
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(),
                     nullptr);
      device_id = devices[0];
      break;
    }
  }

  if (device_id == nullptr) {
    std::cerr << "Failed to find a GPU device." << std::endl;
    return -1;
  }

  cl_context context =
      clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
  checkError(err, "clCreateContext");

  cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);
  checkError(err, "clCreateCommandQueue");

  const char *source_str = loadKernelSource("src/opencl_kernel.cl");

  cl_program program =
      clCreateProgramWithSource(context, 1, &source_str, nullptr, &err);
  checkError(err, "clCreateProgramWithSource");
  delete[] source_str;

  err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t logSize;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &logSize);

    char *log = (char *)malloc(logSize);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logSize,
                          log, NULL);

    fprintf(stderr, "Build log:\n%s\n", log);
    free(log);
  }
  checkError(err, "clBuildProgram");

  cl_kernel kernel = clCreateKernel(program, "search_pattern_opencl", &err);

  auto start = std::chrono::high_resolution_clock::now();
  // Host pointers
  unsigned int *h_sizes = new unsigned int[k]();
  unsigned int *h_locks = new unsigned int[k]();
  int **h_indices = new int *[k];
  int max_size = 200; // Maximum number of occurrences per pattern

  for (int i = 0; i < k; ++i) {
    h_indices[i] = new int[max_size]();
  }

  // Create memory buffers on the device
  cl_mem vec_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     size * sizeof(Node), nullptr, &err);
  cl_mem text_buffer = clCreateBuffer(
      context, CL_MEM_READ_ONLY, text_length * sizeof(char), nullptr, &err);
  cl_mem sizes_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       k * sizeof(unsigned int), nullptr, &err);
  cl_mem indices_buffer = clCreateBuffer(
      context, CL_MEM_READ_WRITE, k * max_size * sizeof(int), nullptr, &err);
  cl_mem locks_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       k * sizeof(unsigned int), nullptr, &err);

  err = clEnqueueWriteBuffer(queue, vec_buffer, CL_TRUE, 0, size * sizeof(Node),
                             vec, 0, nullptr, nullptr);
  err = clEnqueueWriteBuffer(queue, text_buffer, CL_TRUE, 0,
                             text_length * sizeof(char), text.c_str(), 0,
                             nullptr, nullptr);
  err = clEnqueueWriteBuffer(queue, sizes_buffer, CL_TRUE, 0,
                             k * sizeof(unsigned int), h_sizes, 0, nullptr,
                             nullptr);
  err = clEnqueueWriteBuffer(queue, locks_buffer, CL_TRUE, 0,
                             k * sizeof(unsigned int), h_locks, 0, nullptr,
                             nullptr);
  for (int i = 0; i < k; ++i) {
    size_t offset = i * max_size * sizeof(int);
    err = clEnqueueWriteBuffer(queue, indices_buffer, CL_TRUE, offset,
                               max_size * sizeof(int), h_indices[i], 0, NULL,
                               NULL);
    if (err != CL_SUCCESS) {
      std::cerr << "Error writing indices data to buffer: " << err << std::endl;
    }
  }

  int thread_count = 4;

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&vec_buffer);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&text_buffer);
  err = clSetKernelArg(kernel, 2, sizeof(int), &text_length);
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &indices_buffer);
  err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &sizes_buffer);
  err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &locks_buffer);
  err = clSetKernelArg(kernel, 6, sizeof(int), &thread_count);
  err = clSetKernelArg(kernel, 7, sizeof(int), &max_pattern_length);
  err = clSetKernelArg(kernel, 8, sizeof(int), &k);
  err = clSetKernelArg(kernel, 9, sizeof(int), &max_size);

  size_t global_item_size = thread_count;
  size_t local_item_size = 1;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_item_size,
                               &local_item_size, 0, nullptr, nullptr);

  clFinish(queue);

  // Read the updated data back to the host for the sizes array
  err = clEnqueueReadBuffer(queue, sizes_buffer, CL_TRUE, 0,
                            k * sizeof(unsigned int), h_sizes, 0, nullptr,
                            nullptr);

  auto end = std::chrono::high_resolution_clock::now();

  // Duration in milliseconds
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "Time taken: " << duration.count() << " milliseconds"
            << std::endl;

  if (err != CL_SUCCESS) {
    std::cerr << "Error reading sizes data from buffer: " << err << std::endl;
  }

  for (int i = 0; i < k; ++i) {
    size_t offset = i * max_size * sizeof(int);
    err = clEnqueueReadBuffer(queue, indices_buffer, CL_TRUE, offset,
                              h_sizes[i] * sizeof(int), h_indices[i], 0, NULL,
                              NULL);
    if (err != CL_SUCCESS) {
      std::cerr << "Error reading indices data to buffer for pattern " << i
                << ": " << err << std::endl;
    }
  }

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
    std::cout << "\": " << h_sizes[i] << std::endl;
    std::cout << "Positions: ";
    for (int j = 0; j < h_sizes[i]; j++) {
      std::cout << h_indices[i][j] << " ";
    }
    std::cout << std::endl;
  }

  delete[] vec;
  clReleaseMemObject(vec_buffer);
  clReleaseMemObject(text_buffer);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}