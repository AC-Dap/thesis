#ifndef RESULTS_H
#define RESULTS_H

#include <map>
#include <string>

/**
* The results of our simulations are stored in a CSV file with 5 columns:
* - sketch_type: a string describing the kind of sketch used, including all parameters
* - k: the space used
* - n_trial: what number trial this row is for
* - estimate: the estimate of the moment
* - exact: the exact value of the moment
*/
struct Results {
    std::map<std::string,
      std::map<size_t,
          std::map<size_t,
            std::pair<double, __uint128_t>>>> data;
    std::string output_path;

    explicit Results(const std::string& output_path): output_path(output_path) {}

    bool has(const std::string& sketch_type, const size_t k, const size_t n_trial) {
        return data.contains(sketch_type) &&
               data[sketch_type].contains(k) &&
               data[sketch_type][k].contains(n_trial);
    }

    void add_result(const std::string& sketch_type, size_t k, size_t n_trial, double estimate, __uint128_t exact) {
        data[sketch_type][k][n_trial] = {estimate, exact};
    }

    static Results read_from_file(const std::string& file_name);

    /**
     * Write the current contents of the results to the file.
     * This overwrites the file if it already exists.
     */
    void flush_to_file();
};

#endif //RESULTS_H
