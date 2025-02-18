#include "common/io/results.h"

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <format>
#include <cstdlib>

std::vector<std::string> split (const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss (s);
    std::string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}

__uint128_t strtouint128 (const std::string &s) {
    __uint128_t result = 0;

    for (char c : s) {
        if (isdigit(c)) {
            result = result * 10 + (c - '0');
        } else {
            throw std::domain_error("Invalid character when parsing int (" + std::to_string(c) + ").");
        }
    }

    return result;
}

std::string uint128tostr(__uint128_t num) {
    std::string str;
    do {
        int digit = num % 10;
        str = std::to_string(digit) + str;
        num = (num - digit) / 10;
    } while (num != 0);
    return str;
}

Results Results::read_from_file(const std::string &file_name) {
    std::ifstream in(file_name);
    if (!in.good()) {
        throw std::invalid_argument("CSV file does not exist: " + file_name);
    }

    std::string line;

    // Skip header line
    std::getline(in, line);

    Results results(file_name);
    while (std::getline(in, line)) {
        std::vector<std::string> row = split(line, ',');
        results.add_result(
            row[0],
            std::stoll(row[1]),
            std::stoll(row[2]),
            // Append 0x to force strtod to parse as hex
            std::strtod(("0x" + row[3]).c_str(), nullptr),
            strtouint128(row[4])
        );
    }

    return results;
}

void Results::flush_to_file() {
    std::ofstream out(output_path);
    out << "sketch_type,k,n_trial,estimate,exact" << std::endl;

    // Write all data currently in results
    for (const auto& [sketch_type, ks] : data) {
        for (const auto& [k, trials] : ks) {
            for (auto [trial, estimate] : trials) {
                out << sketch_type << "," <<
                    k << "," <<
                    trial << "," <<
                    std::format("{:a}", estimate.first) << "," <<
                    uint128tostr(estimate.second) << std::endl;
            }
        }
    }

    out.close();
}
