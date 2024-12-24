#include "results.h"

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

Results Results::read_from_file(const std::string &file_name) {
    std::ifstream in(file_name);
    if (!in.good()) {
        throw std::invalid_argument("CSV file does not exist");
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
            std::strtod(("0x" + row[3]).c_str(), nullptr)
        );
    }

    return results;
}

void Results::flush_to_file() {
    std::ofstream out(output_path);
    out << "sketch_type,k,n_trial,estimate" << std::endl;

    // Write all data currently in results
    for (const auto& [sketch_type, ks] : data) {
        for (const auto& [k, trials] : ks) {
            for (auto [trial, estimate] : trials) {
                out << sketch_type << "," << k << "," << trial << "," << std::format("{:a}", estimate) << std::endl;
            }
        }
    }

    out.close();
}
