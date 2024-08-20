#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Helper function to convert a hex string to a byte (char)
char hexToByte(const std::string &hex);

// Function to read hex data from a file and convert it to a vector of byte
// strings
std::vector<std::string> readHexFile(const std::string &filename, int k);

// Function to read a binary file and return its contents as a string of bytes
std::string readISOFileAsBytes(const std::string &filepath);
