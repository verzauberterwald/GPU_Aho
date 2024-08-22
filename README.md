This project provides the necessary source code for a comparison between CUDA, OpenCL and SYCL in an accelerated impleentation of Aho-Corasick algorithm for parallel computing on CPUs and GPGPUs in the context of DNA Biometrics and Digital Forensics applications.

The project is divided into two separate experiments:
  - DNA Biometrics
    This experiment uses as input data DNA sequences which are composed on only 4 nucleotides (therefore 4 distinct characters). 
  - Digital Forensics
    This experiment uses as input sequences of text whih contain up to distinct 256 characters.

Prerequisites:
- NVIDIA GPU (for CUDA targets)
- CMake
- GCC
- CUDA
- [DPC++](https://developer.codeplay.com/products/oneapi/nvidia/2024.1.0/guides/get-started-guide-nvidia#supported-platforms "using DPC++ to run SYCL™ applications on NVIDIA")

For compilation use './build.sh' in each of the 2 folders separately.

```
Usage: <program_binary> <input_file> <patterns_file>
    <program_binary> : The binary of the app to be executed.
    <input_file> : The name of the text file to process.
    <patterns_file> : The file containing patterns to search within the input file.
```
