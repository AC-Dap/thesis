#include <fstream>
#include <iostream>
#include <format>
#include <string>
#include <functional>
#include <filesystem>
#include <results.h>

#include "dataset.h"
#include "simulations.h"

using namespace std;

constexpr char const *TRAIN_PATH = "../data/AOL-user-ct-collection/user-ct-test-collection-01.txt";
constexpr char const *TEST_PATH = "../data/AOL-user-ct-collection/user-ct-test-collection-02.txt";

// Write mode to use for data file, if the file already exists.
enum FileWriteMode {
    SKIP,
    OVERWRITE,
};

void run_sims(Results &results, vector<size_t> &ks, size_t n_sims,
              string sketch_type, function<vector<double>(size_t, size_t)> run_n_sims, FileWriteMode mode) {
    for (size_t k: ks) {
        // Get which trials we need to run
        vector<size_t> trials;
        for (int i = 1; i <= n_sims; i++) {
            if (mode == OVERWRITE || !results.has(sketch_type, k, i)) {
                trials.push_back(i);
            }
        }

        if (trials.empty()) {
            cout << "Skipping " << sketch_type << " k=" << k << endl;
            continue;
        }

        // Run the trials
        cout << sketch_type << " k=" << k << ": ";
        vector<double> estimates = run_n_sims(k, trials.size());

        // Write trials to results
        for (size_t i = 0; i < trials.size(); i++) {
            results.add_result(sketch_type, k, trials[i], estimates[i]);
        }

        // Make sure our results file is updated with the newest sims
        results.flush_to_file();
    }
}

int main() {
    BackingItems backing_items = get_backing_items({TRAIN_PATH, TEST_PATH});

    Dataset ds_train(backing_items), ds_test(backing_items);
    ds_train.read_from_file(TRAIN_PATH);
    ds_test.read_from_file(TEST_PATH);

    constexpr size_t total_trials = 30;
    vector<size_t> degs = {3, 4};
    vector<size_t> ks = {1 << 6, 1 << 7, 1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16};

    constexpr size_t min_freq = 7;
    constexpr double ep = 0.05;

    MockOracleAbsoluteError o_abs(ep, ds_train);
    MockOracleRelativeError o_rel(ep, ds_train);
    MockOracleBinomialError o_bin(ep, ds_train);
    ExactOracle o_exact(ds_train);
    vector<MockOracle*> os = {&o_abs, &o_rel, &o_bin, &o_exact};

    FileWriteMode mode = SKIP;
    for (auto deg: degs) {
        // Load results
        string file_path = format("results/deg={}/deg={}_results.csv", deg, deg);
        Results results(file_path);
        try {
            results = Results::read_from_file(file_path);
        } catch (std::invalid_argument const &ex) {
            cout << ex.what() << ", creating new csv at " << file_path << endl;
        }

        // PPSWOR
        run_sims(
            results, ks, total_trials,
            "ppswor",
            [&](size_t k, size_t n_trials) {
                return run_n_ppswor_sims(k, deg, n_trials, ds_test);
            },
            mode
        );

        // Exact PPSWOR
        run_sims(
            results, ks, total_trials,
            "exact",
            [&](size_t k, size_t n_trials) {
                return run_n_swa_sims(0, k, 0, o_exact, deg, n_trials, ds_test);
            },
            mode
        );

        // SWA
        for (auto* o: os) {
            if (o->name() == "exact") continue;

            // 0 k 0
            run_sims(
                results, ks, total_trials,
                format("swa_{}_kh=0_kp=k_ku=0", o->name()),
                [&](size_t k, size_t n_trials) {
                    return run_n_swa_sims(0, k, 0, *o, deg, n_trials, ds_test);
                },
                mode
            );

            // k/2 k/2 0
            run_sims(
                results, ks, total_trials,
                format("swa_{}_kh=k/2_kp=k/2_ku=0", o->name()),
                [&](size_t k, size_t n_trials) {
                    return run_n_swa_sims(k / 2, k / 2, 0, *o, deg, n_trials, ds_test);
                },
                mode
            );

            // 0 k/2 k/2
            run_sims(
                results, ks, total_trials,
                format("swa_{}_kh=0_kp=k/2_ku=k/2", o->name()),
                [&](size_t k, size_t n_trials) {
                    return run_n_swa_sims(0, k / 2, k / 2, *o, deg, n_trials, ds_test);
                },
                mode
            );
        }

        // Avg bucket estimators
        vector<pair<function<Buckets(size_t)>, string>> bucket_types = {
            {[&](size_t k) { return generate_exponential_buckets(min_freq, k); }, "expo"},
            {generate_linear_buckets, "linear"}
        };
        vector<pair<function<double(double, double, double)>, string>> n_estimates = {
            {n_estimate_left, "lower"},
            {n_estimate_right, "upper"},
            {n_estimate_arith_avg, "arith"},
            {n_estimate_geo_avg, "geo"},
            {n_estimate_harm_avg, "harm"}
        };
        for (auto &[bucket_gen, bucket_name]: bucket_types) {
            for (auto &[n_estimate, estimate_name]: n_estimates) {
                for (auto* o: os) {
                    // k 0
                    run_sims(
                        results, ks, total_trials,
                        format("bucket_{}_{}_{}_k=k_kh=0", bucket_name, estimate_name, o->name()),
                        [&](size_t k, size_t n_trials) {
                            auto buckets = [&] {
                                return bucket_gen(k);
                            };

                            auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                                return bucket_sketch(0, deg, n_estimate, b, o, ds);
                            };

                            return run_n_bucket_sims(
                                buckets,
                                sketch,
                                n_trials,
                                *o,
                                ds_test
                            );
                        },
                        mode
                    );

                    // k/2 k/2
                    run_sims(
                        results, ks, total_trials,
                        format("bucket_{}_{}_{}_k=k/2_kh=k/2", bucket_name, estimate_name, o->name()),
                        [&](size_t k, size_t n_trials) {
                            auto buckets = [&] {
                                return bucket_gen(k / 2);
                            };

                            auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                                return bucket_sketch(k / 2, deg, n_estimate, b, o, ds);
                            };

                            return run_n_bucket_sims(
                                buckets,
                                sketch,
                                n_trials,
                                *o,
                                ds_test
                            );
                        },
                        mode
                    );
                }
            }
        }


        // Alt bucket estimator
        for (auto &[bucket_gen, bucket_name]: bucket_types) {
            for (auto* o: os) {
                // k 0
                run_sims(
                    results, ks, total_trials,
                    format("bucket_{}_alt_{}_k=k_kh=0", bucket_name, o->name()),
                    [&](size_t k, size_t n_trials) {
                        auto buckets = [&] {
                            return bucket_gen(k);
                        };

                        auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                            return alt_bucket_sketch(0, deg, b, o, ds);
                        };

                        return run_n_bucket_sims(
                            buckets,
                            sketch,
                            n_trials,
                            *o,
                            ds_test
                        );
                    },
                    mode
                );

                // k/2 k/2
                run_sims(
                    results, ks, total_trials,
                    format("bucket_{}_alt_{}_k=k/2_kh=k/2", bucket_name, o->name()),
                    [&](size_t k, size_t n_trials) {
                        auto buckets = [&] {
                            return bucket_gen(k / 2);
                        };

                        auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                            return alt_bucket_sketch(k / 2, deg, b, o, ds);
                        };

                        return run_n_bucket_sims(
                            buckets,
                            sketch,
                            n_trials,
                            *o,
                            ds_test
                        );
                    },
                    mode
                );
            }
        }

        // Cond bucket estimator
        for (auto &[bucket_gen, bucket_name]: bucket_types) {
            for (auto* o: os) {
                // 1 k
                run_sims(
                    results, ks, total_trials,
                    format("cond_bucket_{}_{}_k=1_kh=k", bucket_name, o->name()),
                    [&](size_t k, size_t n_trials) {
                        auto buckets = [&] {
                            return bucket_gen(1);
                        };

                        auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                            return cond_bucket_sketch(k, deg, b, o, ds);
                        };

                        return run_n_bucket_sims(
                            buckets,
                            sketch,
                            n_trials,
                            *o,
                            ds_test
                        );
                    },
                    mode
                );
            }
        }
    }
}
