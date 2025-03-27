#include "moments/simulations.h"

#include <vector>
#include <string>
#include <chrono>
#include <iostream>

#include "common/ppswor.h"
#include "common/mock_oracle.h"
#include "common/simulations.h"
#include "common/io/dataset.h"
#include "moments/estimator.h"
#include "moments/sketches/fake_swa.h"
#include "moments/sketches/bucket_sketch.h"
#include "moments/sketches/parsimonious.h"

using namespace std;

namespace moments {
    void run_all_sims(Dataset &ds, vector<MockOracle *> &os,
                      size_t total_trials, string &output_name, FileWriteMode mode) {
        vector<size_t> degs = {3, 4};
        vector<size_t> ks = {
            1 << 6, 1 << 7, 1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16
        };

        for (auto deg: degs) {
            // Load results
            string file_path = format("results/{}_moments_deg={}.csv", output_name, deg);
            Results results(file_path);
            try {
                results = Results::read_from_file(file_path);
            } catch (std::invalid_argument const &ex) {
                cout << ex.what() << ", creating new csv at " << file_path << endl;
            }

            __uint128_t exact_moment = calculate_exact_moment(ds, deg);

            // PPSWOR
            run_sims(
                results, ks, total_trials,
                "ppswor",
                [&](size_t k, size_t n_trials) {
                    return run_n_ppswor_sims(k, deg, n_trials, ds);
                },
                exact_moment,
                mode
            );

            // SWA
            for (auto *o: os) {
                if (o->name == "exact") continue;

                // 0 k 0
                run_sims(
                    results, ks, total_trials,
                    format("swa_{}_kh=0_kp=k_ku=0", o->name),
                    [&](size_t k, size_t n_trials) {
                        return run_n_swa_sims(0, k, 0, *o, deg, n_trials, ds);
                    },
                    exact_moment,
                    mode
                );

                // k/2 k/2 0
                run_sims(
                    results, ks, total_trials,
                    format("swa_{}_kh=k/2_kp=k/2_ku=0", o->name),
                    [&](size_t k, size_t n_trials) {
                        return run_n_swa_sims(k / 2, k / 2, 0, *o, deg, n_trials, ds);
                    },
                    exact_moment,
                    mode
                );

                // k/2 k/4 k/4
                run_sims(
                    results, ks, total_trials,
                    format("swa_{}_kh=k/2_kp=k/4_ku=k/4", o->name),
                    [&](size_t k, size_t n_trials) {
                        return run_n_swa_sims(k/2, k / 4, k / 4, *o, deg, n_trials, ds);
                    },
                    exact_moment,
                    mode
                );
            }

            for (auto *o : os) {
                vector<pair<function<Buckets(double, size_t)>, string> > bucket_types = {
                    {generate_exponential_buckets, "expo"},
                    {[&](double min_freq, size_t k) { return generate_linear_buckets(k);}, "linear"}
                };

                // Avg bucket estimators
                for (auto &[bucket_gen, bucket_name]: bucket_types) {
                    // k 0
                    run_sims(
                        results, ks, total_trials,
                        format("central_bucket_{}_{}_k=k_kh=0", bucket_name, o->name),
                        [&](size_t k, size_t n_trials) {
                            auto buckets = [&](double min_freq) {
                                return bucket_gen(min_freq, k);
                            };

                            auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                                return central_bucket_sketch(0, deg, b, o, ds);
                            };

                            return run_n_bucket_sims(
                                buckets,
                                sketch,
                                n_trials,
                                *o,
                                ds
                            );
                        },
                        exact_moment,
                        mode
                    );

                    // k/2 k/2
                    run_sims(
                        results, ks, total_trials,
                        format("central_bucket_{}_{}_k=k/2_kh=k/2", bucket_name, o->name),
                        [&](size_t k, size_t n_trials) {
                            auto buckets = [&](double min_freq) {
                                return bucket_gen(min_freq, k / 2);
                            };

                            auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                                return central_bucket_sketch(k / 2, deg, b, o, ds);
                            };

                            return run_n_bucket_sims(
                                buckets,
                                sketch,
                                n_trials,
                                *o,
                                ds
                            );
                        },
                        exact_moment,
                        mode
                    );
                }

                // k 0
                run_sims(
                    results, ks, total_trials,
                    format("unbiased_bucket_{}_kh=0", o->name),
                    [&](size_t k, size_t n_trials) {
                        auto buckets = [&](double min_freq) {
                            return generate_linear_buckets(1);
                        };

                        auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                            return unbiased_bucket_sketch(0, deg, b, o, ds);
                        };

                        return run_n_bucket_sims(
                            buckets,
                            sketch,
                            n_trials,
                            *o,
                            ds
                        );
                    },
                    exact_moment,
                    mode
                );

                // k/2 k/2
                run_sims(
                    results, ks, total_trials,
                    format("unbiased_bucket_{}_kh=k", o->name),
                    [&](size_t k, size_t n_trials) {
                        auto buckets = [&](double min_freq) {
                            return generate_linear_buckets(1);
                        };

                        auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                            return unbiased_bucket_sketch(k, deg, b, o, ds);
                        };

                        return run_n_bucket_sims(
                            buckets,
                            sketch,
                            n_trials,
                            *o,
                            ds
                        );
                    },
                    exact_moment,
                    mode
                );

                for (auto &[bucket_gen, bucket_name]: bucket_types) {
                    // k 0
                    run_sims(
                        results, ks, total_trials,
                        format("counting_bucket_{}_{}_k=k_kh=0", bucket_name, o->name),
                        [&](size_t k, size_t n_trials) {
                            auto buckets = [&](double min_freq) {
                                return bucket_gen(min_freq, k);
                            };

                            auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                                return counting_bucket_sketch(0, deg, b, o, ds);
                            };

                            return run_n_bucket_sims(
                                buckets,
                                sketch,
                                n_trials,
                                *o,
                                ds
                            );
                        },
                        exact_moment,
                        mode
                    );

                    // k/2 k/2
                    run_sims(
                        results, ks, total_trials,
                        format("counting_bucket_{}_{}_k=k/2_kh=k/2", bucket_name, o->name),
                        [&](size_t k, size_t n_trials) {
                            auto buckets = [&](double min_freq) {
                                return bucket_gen(min_freq, k / 2);
                            };

                            auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                                return counting_bucket_sketch(k / 2, deg, b, o, ds);
                            };

                            return run_n_bucket_sims(
                                buckets,
                                sketch,
                                n_trials,
                                *o,
                                ds
                            );
                        },
                        exact_moment,
                        mode
                    );
                }

                for (auto &[bucket_gen, bucket_name]: bucket_types) {
                    // k 0
                    run_sims(
                        results, ks, total_trials,
                        format("sampling_bucket_{}_{}_k=k/16_ku=16_kh=0", bucket_name, o->name),
                        [&](size_t k, size_t n_trials) {
                            auto buckets = [&](double min_freq) {
                                return bucket_gen(min_freq, k / 16);
                            };

                            auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                                return sampling_bucket_sketch(0, 16, deg, b, o, ds);
                            };

                            return run_n_bucket_sims(
                                buckets,
                                sketch,
                                n_trials,
                                *o,
                                ds
                            );
                        },
                        exact_moment,
                        mode
                    );

                    // k/2 k/2
                    run_sims(
                        results, ks, total_trials,
                        format("sampling_bucket_{}_{}_k=k/32_ku=16_kh=k/2", bucket_name, o->name),
                        [&](size_t k, size_t n_trials) {
                            auto buckets = [&](double min_freq) {
                                return bucket_gen(min_freq, k / 32);
                            };

                            auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                                return sampling_bucket_sketch(k / 2, 16, deg, b, o, ds);
                            };

                            return run_n_bucket_sims(
                                buckets,
                                sketch,
                                n_trials,
                                *o,
                                ds
                            );
                        },
                        exact_moment,
                        mode
                    );
                }
            }

            // Make sure all results are flushed in the end
            results.flush_to_file();
        }
    }

    vector<double> run_n_ppswor_sims(size_t k, size_t deg, size_t nsims, Dataset &ds) {
        auto t1 = chrono::high_resolution_clock::now();

        PPSWOR pp(k, 2, ds.item_counts.size());

        vector<double> estimates(nsims);
        for (int i = 0; i < nsims; i++) {
            if (i != 0) pp.reset();

            for (auto item: ds.lines) {
                pp.update(item, 1);
            }

            auto sample = pp.sample();
            auto weights = get<1>(sample), probs = get<2>(sample);

            estimates[i] = estimate_moment(weights, probs, deg);

            cout << "." << flush;
        }
        cout << endl;

        auto t2 = chrono::high_resolution_clock::now();
        cout << " | " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() / 1000.0 / nsims
                << " seconds on average" << endl;

        return estimates;
    }

    vector<double> run_n_swa_sims(size_t kh, size_t kp, size_t ku, MockOracle &o, size_t deg, size_t nsims,
                                  Dataset &ds) {
        auto t1 = chrono::high_resolution_clock::now();

        vector<double> estimates(nsims);
        for (int i = 0; i < nsims; i++) {
            o.reset_estimates();

            auto sample = fake_swa_sample(kh, kp, ku, deg, o, ds);
            auto weights = get<1>(sample), probs = get<2>(sample);

            estimates[i] = estimate_moment(weights, probs, deg);

            cout << "." << flush;
        }
        cout << endl;

        auto t2 = chrono::high_resolution_clock::now();
        cout << " | " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() / 1000.0 / nsims
                << " seconds on average" << endl;

        return estimates;
    }

    vector<double> run_n_bucket_sims(
        const function<Buckets(double)> bucket_gen,
        const function<double(Buckets &, MockOracle &, Dataset &)> sketch,
        size_t nsims, MockOracle &o, Dataset &ds) {
        auto t1 = chrono::high_resolution_clock::now();

        vector<double> estimates(nsims);
        for (int i = 0; i < nsims; i++) {
            o.reset_estimates();

            double min_freq = 1;
            for (ItemId item = 0; item < ds.item_counts.size(); item++) {
                if (ds.item_counts[item] > 0) {
                    if (o.estimates[item] == 0) o.estimates[item] = 1e-9;
                    min_freq = min(min_freq, o.estimates[item]);
                }
            }
            Buckets b = bucket_gen(min_freq);

            estimates[i] = sketch(b, o, ds);

            cout << "." << flush;
        }
        cout << endl;

        auto t2 = chrono::high_resolution_clock::now();
        cout << " | " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() / 1000.0 / nsims
                << " seconds on average" << endl;

        return estimates;
    }
}
