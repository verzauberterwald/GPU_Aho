#include "../include/read_helper.hpp"

char hexToByte(const std::string &hex) {
  char byte = static_cast<char>(std::stoi(hex, nullptr, 16));
  return byte;
}

std::vector<std::string> readHexFile(const std::string &filename, int k) {
  std::vector<std::string> byteStrings;
  std::ifstream file(filename);
  std::string line;

  int i = 0;
  while (std::getline(file, line)) {
    std::string byteString;
    for (size_t j = 0; j < line.length(); j += 2) {
      if (j + 1 < line.length()) {
        byteString.push_back(hexToByte(line.substr(j, 2)));
      }
    }
    byteStrings.push_back(byteString);
    i++;
    if (i == k)
      break;
  }

  return byteStrings;
}

std::string readISOFileAsBytes(const std::string &filepath) {
  std::ifstream file(filepath, std::ios::binary);

  if (!file) {
    std::cerr << "Error: Could not open file." << std::endl;
    return "";
  }

  std::string fileBytes((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

  file.close();

  return fileBytes;
}
