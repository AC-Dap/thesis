#include "threshold/simulations.h"

#include <vector>
#include <string>
#include <chrono>
#include <iostream>

#include "common/ppswor.h"
#include "common/mock_oracle.h"
#include "common/simulations.h"
#include "common/io/dataset.h"
#include "threshold/estimator.h"
#include "threshold/sketches/fake_swa.h"
#include "threshold/sketches/bucket_sketch.h"

using namespace std;

namespace threshold {
    void run_all_sims(Dataset &ds, vector<MockOracle *> &os,
                      size_t total_trials, string &output_name, FileWriteMode mode) {
        double threshold = 0.00001;
        constexpr size_t min_freq = 7;
        vector<size_t> ks = {1 << 6, 1 << 7, 1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12};

        // Load results
        string file_path = format("results/{}_threshold.csv", output_name);
        Results results(file_path);
        try {
            results = Results::read_from_file(file_path);
        } catch (std::invalid_argument const &ex) {
            cout << ex.what() << ", creating new csv at " << file_path << endl;
        }

        __uint128_t exact_threshold = calculate_exact_threshold(ds, threshold);

        // PPSWOR
        run_sims(
            results, ks, total_trials,
            "ppswor",
            [&](size_t k, size_t n_trials) {
                return run_n_ppswor_sims(k, threshold, n_trials, ds);
            },
            exact_threshold,
            mode
        );

        // SWA
        for (auto *o: os) {
            // 0 k 0
            run_sims(
                results, ks, total_trials,
                format("swa_{}_kh=0_kp=k_ku=0", o->name),
                [&](size_t k, size_t n_trials) {
                    return run_n_swa_sims(0, k, 0, *o, threshold, n_trials, ds);
                },
                exact_threshold,
                mode
            );

            // k/2 k/2 0
            run_sims(
                results, ks, total_trials,
                format("swa_{}_kh=k/2_kp=k/2_ku=0", o->name),
                [&](size_t k, size_t n_trials) {
                    return run_n_swa_sims(k / 2, k / 2, 0, *o, threshold, n_trials, ds);
                },
                exact_threshold,
                mode
            );

            // k/2 k/4 k/4
            run_sims(
                results, ks, total_trials,
                format("swa_{}_kh=k/2_kp=k/4_ku=k/4", o->name),
                [&](size_t k, size_t n_trials) {
                    return run_n_swa_sims(k/2, k/4, k/4, *o, threshold, n_trials, ds);
                },
                exact_threshold,
                mode
            );
        }

        // Avg bucket estimators
        vector<pair<function<Buckets(size_t)>, string> > bucket_types = {
            {[&](size_t k) { return generate_exponential_buckets(min_freq, k); }, "expo"},
            {generate_linear_buckets, "linear"}
        };
        for (auto &[bucket_gen, bucket_name]: bucket_types) {
            for (auto *o: os) {
                // k 0
                run_sims(
                    results, ks, total_trials,
                    format("central_bucket_{}_{}_k=k_kh=0", bucket_name, o->name),
                        [&](size_t k, size_t n_trials) {
                            auto buckets = [&] {
                                return bucket_gen(k);
                            };

                            auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                                return central_bucket_sketch(0, threshold, b, o, ds);
                            };

                            return run_n_bucket_sims(
                                buckets,
                                sketch,
                                n_trials,
                                *o,
                                ds
                            );
                        },
                    exact_threshold,
                    mode
                );

                // k/2 k/2
                run_sims(
                    results, ks, total_trials,
                    format("central_bucket_{}_{}_k=k/2_kh=k/2", bucket_name, o->name),
                        [&](size_t k, size_t n_trials) {
                            auto buckets = [&] {
                                return bucket_gen(k / 2);
                            };

                            auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                                return central_bucket_sketch(k / 2, threshold, b, o, ds);
                            };

                            return run_n_bucket_sims(
                                buckets,
                                sketch,
                                n_trials,
                                *o,
                                ds
                            );
                        },
                    exact_threshold,
                    mode
                );
            }
        }

        for (auto &[bucket_gen, bucket_name]: bucket_types) {
            for (auto *o: os) {
                // k 0
                run_sims(
                    results, ks, total_trials,
                    format("counting_bucket_{}_{}_k=k_kh=0", bucket_name, o->name),
                    [&](size_t k, size_t n_trials) {
                        auto buckets = [&] {
                            return bucket_gen(k);
                        };

                        auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                            return counting_bucket_sketch(0, threshold, b, o, ds);
                        };

                        return run_n_bucket_sims(
                            buckets,
                            sketch,
                            n_trials,
                            *o,
                            ds
                        );
                    },
                    exact_threshold,
                    mode
                );

                // k/2 k/2
                run_sims(
                    results, ks, total_trials,
                    format("counting_bucket_{}_{}_k=k/2_kh=k/2", bucket_name, o->name),
                    [&](size_t k, size_t n_trials) {
                        auto buckets = [&] {
                            return bucket_gen(k / 2);
                        };

                        auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                            return counting_bucket_sketch(k / 2, threshold, b, o, ds);
                        };

                        return run_n_bucket_sims(
                            buckets,
                            sketch,
                            n_trials,
                            *o,
                            ds
                        );
                    },
                    exact_threshold,
                    mode
                );
            }
        }

        for (auto &[bucket_gen, bucket_name]: bucket_types) {
            for (auto *o: os) {
                // k 0
                run_sims(
                    results, ks, total_trials,
                    format("sampling_bucket_{}_{}_k=k/16_ku=16_kh=0", bucket_name, o->name),
                    [&](size_t k, size_t n_trials) {
                        auto buckets = [&] {
                            return bucket_gen(k / 16);
                        };

                        auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                            return sampling_bucket_sketch(0, 16, threshold, b, o, ds);
                        };

                        return run_n_bucket_sims(
                            buckets,
                            sketch,
                            n_trials,
                            *o,
                            ds
                        );
                    },
                    exact_threshold,
                    mode
                );

                // k/2 k/2
                run_sims(
                    results, ks, total_trials,
                    format("sampling_bucket_{}_{}_k=k/32_ku=16_kh=k/2", bucket_name, o->name),
                    [&](size_t k, size_t n_trials) {
                        auto buckets = [&] {
                            return bucket_gen(k / 32);
                        };

                        auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
                            return sampling_bucket_sketch(k / 2, 16, threshold, b, o, ds);
                        };

                        return run_n_bucket_sims(
                            buckets,
                            sketch,
                            n_trials,
                            *o,
                            ds
                        );
                    },
                    exact_threshold,
                    mode
                );
            }
        }

        // // Smart bucket estimators
        // for (auto *o: os) {
        //     auto &n_estimate = n_estimate_arith_avg;
        //
        //     // k/2 k/2, a
        //     run_sims(
        //         results, ks, total_trials,
        //         format("smart_a_expo_arith_{}_k=k/2_kh=k/2", o->name),
        //         [&](size_t k, size_t n_trials) {
        //             auto buckets = [&] {
        //                 return generate_exponential_buckets(min_freq, k / 2);
        //             };
        //
        //             auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
        //                 return smart_a_bucket_sketch(k / 2, threshold, n_estimate, b, o, ds);
        //             };
        //
        //             return run_n_bucket_sims(
        //                 buckets,
        //                 sketch,
        //                 n_trials,
        //                 *o,
        //                 ds
        //             );
        //         },
        //         exact_threshold,
        //         mode
        //     );
        //
        //     // k/2 k/2, b
        //     run_sims(
        //         results, ks, total_trials,
        //         format("smart_b_expo_arith_{}_k=k/2_kh=k/2", o->name),
        //         [&](size_t k, size_t n_trials) {
        //             auto buckets = [&] {
        //                 return generate_exponential_buckets(min_freq, k / 2);
        //             };
        //
        //             auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
        //                 return smart_b_bucket_sketch(k / 2, threshold, n_estimate, b, o, ds);
        //             };
        //
        //             return run_n_bucket_sims(
        //                 buckets,
        //                 sketch,
        //                 n_trials,
        //                 *o,
        //                 ds
        //             );
        //         },
        //         exact_threshold,
        //         mode
        //     );
        // }
        //
        // // Alt bucket estimator
        // for (auto &[bucket_gen, bucket_name]: bucket_types) {
        //     for (auto *o: os) {
        //         // k 0
        //         run_sims(
        //             results, ks, total_trials,
        //             format("bucket_{}_alt_{}_k=k_kh=0", bucket_name, o->name),
        //             [&](size_t k, size_t n_trials) {
        //                 auto buckets = [&] {
        //                     return bucket_gen(k);
        //                 };
        //
        //                 auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
        //                     return alt_bucket_sketch(0, threshold, b, o, ds);
        //                 };
        //
        //                 return run_n_bucket_sims(
        //                     buckets,
        //                     sketch,
        //                     n_trials,
        //                     *o,
        //                     ds
        //                 );
        //             },
        //             exact_threshold,
        //             mode
        //         );
        //
        //         // k/2 k/2
        //         run_sims(
        //             results, ks, total_trials,
        //             format("bucket_{}_alt_{}_k=k/2_kh=k/2", bucket_name, o->name),
        //             [&](size_t k, size_t n_trials) {
        //                 auto buckets = [&] {
        //                     return bucket_gen(k / 2);
        //                 };
        //
        //                 auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
        //                     return alt_bucket_sketch(k / 2, threshold, b, o, ds);
        //                 };
        //
        //                 return run_n_bucket_sims(
        //                     buckets,
        //                     sketch,
        //                     n_trials,
        //                     *o,
        //                     ds
        //                 );
        //             },
        //             exact_threshold,
        //             mode
        //         );
        //     }
        // }
        //
        // // SWA bucket estimator
        // for (auto &[bucket_gen, bucket_name]: bucket_types) {
        //     for (auto *o: os) {
        //         // sqrt(k) sqrt(k) 0
        //         run_sims(
        //             results, ks, total_trials,
        //             format("swa_bucket_{}_{}_k=sqrt(k)_kp=sqrt(k)_kh=0", bucket_name, o->name),
        //             [&](size_t k, size_t n_trials) {
        //                 auto buckets = [&] {
        //                     return bucket_gen(size_t(sqrt(k)));
        //                 };
        //
        //                 auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
        //                     return swa_bucket_sketch(0, size_t(sqrt(k)), threshold, b, o, ds);
        //                 };
        //
        //                 return run_n_bucket_sims(
        //                     buckets,
        //                     sketch,
        //                     n_trials,
        //                     *o,
        //                     ds
        //                 );
        //             },
        //             exact_threshold,
        //             mode
        //         );
        //
        //         // 64 k 0
        //         run_sims(
        //             results, ks, total_trials,
        //             format("swa_bucket_{}_{}_k=64_kp=k_kh=0", bucket_name, o->name),
        //             [&](size_t k, size_t n_trials) {
        //                 auto buckets = [&] {
        //                     return bucket_gen(64);
        //                 };
        //
        //                 auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
        //                     return swa_bucket_sketch(0, k / 64, threshold, b, o, ds);
        //                 };
        //
        //                 return run_n_bucket_sims(
        //                     buckets,
        //                     sketch,
        //                     n_trials,
        //                     *o,
        //                     ds
        //                 );
        //             },
        //             exact_threshold,
        //             mode
        //         );
        //
        //         // 64 k/2 k/2
        //         run_sims(
        //             results, ks, total_trials,
        //             format("swa_bucket_{}_{}_k=64_kp=k/2_kh=k/2", bucket_name, o->name),
        //             [&](size_t k, size_t n_trials) {
        //                 auto buckets = [&] {
        //                     return bucket_gen(64);
        //                 };
        //
        //                 auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
        //                     return swa_bucket_sketch(k / 2, k / 128, threshold, b, o, ds);
        //                 };
        //
        //                 return run_n_bucket_sims(
        //                     buckets,
        //                     sketch,
        //                     n_trials,
        //                     *o,
        //                     ds
        //                 );
        //             },
        //             exact_threshold,
        //             mode
        //         );
        //     }
        // }
        //
        // // Unif bucket estimator
        // for (auto &[bucket_gen, bucket_name]: bucket_types) {
        //     for (auto *o: os) {
        //         // sqrt(k) sqrt(k) 0
        //         run_sims(
        //             results, ks, total_trials,
        //             format("unif_bucket_{}_{}_k=sqrt(k)_ku=sqrt(k)_kh=0", bucket_name, o->name),
        //             [&](size_t k, size_t n_trials) {
        //                 auto buckets = [&] {
        //                     return bucket_gen(size_t(sqrt(k / 2.)));
        //                 };
        //
        //                 auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
        //                     return unif_bucket_sketch(k / 2., size_t(sqrt(k / 2.)), threshold, b, o, ds);
        //                 };
        //
        //                 return run_n_bucket_sims(
        //                     buckets,
        //                     sketch,
        //                     n_trials,
        //                     *o,
        //                     ds
        //                 );
        //             },
        //             exact_threshold,
        //             mode
        //         );
        //
        //         // 64 k 0
        //         run_sims(
        //             results, ks, total_trials,
        //             format("unif_bucket_{}_{}_k=64_ku=k_kh=0", bucket_name, o->name),
        //             [&](size_t k, size_t n_trials) {
        //                 auto buckets = [&] {
        //                     return bucket_gen(64);
        //                 };
        //
        //                 auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
        //                     return unif_bucket_sketch(0, k / 64, threshold, b, o, ds);
        //                 };
        //
        //                 return run_n_bucket_sims(
        //                     buckets,
        //                     sketch,
        //                     n_trials,
        //                     *o,
        //                     ds
        //                 );
        //             },
        //             exact_threshold,
        //             mode
        //         );
        //
        //         // 64 k/2 k/2
        //         run_sims(
        //             results, ks, total_trials,
        //             format("unif_bucket_{}_{}_k=64_ku=k/2_kh=k/2", bucket_name, o->name),
        //             [&](size_t k, size_t n_trials) {
        //                 auto buckets = [&] {
        //                     return bucket_gen(64);
        //                 };
        //
        //                 auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
        //                     return unif_bucket_sketch(k / 2, k / 128, threshold, b, o, ds);
        //                 };
        //
        //                 return run_n_bucket_sims(
        //                     buckets,
        //                     sketch,
        //                     n_trials,
        //                     *o,
        //                     ds
        //                 );
        //             },
        //             exact_threshold,
        //             mode
        //         );
        //     }
        // }
        //
        // // Unif2 bucket estimator
        // for (auto &[bucket_gen, bucket_name]: bucket_types) {
        //     for (auto *o: os) {
        //         // sqrt(k) sqrt(k) 0
        //         run_sims(
        //             results, ks, total_trials,
        //             format("unif2_bucket_{}_{}_k=sqrt(k)_ku=sqrt(k)_kh=0", bucket_name, o->name),
        //             [&](size_t k, size_t n_trials) {
        //                 auto buckets = [&] {
        //                     return bucket_gen(size_t(sqrt(k / 2.)));
        //                 };
        //
        //                 auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
        //                     return unif2_bucket_sketch(k / 2., size_t(sqrt(k / 2.)), threshold, b, o, ds);
        //                 };
        //
        //                 return run_n_bucket_sims(
        //                     buckets,
        //                     sketch,
        //                     n_trials,
        //                     *o,
        //                     ds
        //                 );
        //             },
        //             exact_threshold,
        //             mode
        //         );
        //
        //         // 64 k 0
        //         run_sims(
        //             results, ks, total_trials,
        //             format("unif2_bucket_{}_{}_k=64_ku=k_kh=0", bucket_name, o->name),
        //             [&](size_t k, size_t n_trials) {
        //                 auto buckets = [&] {
        //                     return bucket_gen(64);
        //                 };
        //
        //                 auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
        //                     return unif2_bucket_sketch(0, k / 64, threshold, b, o, ds);
        //                 };
        //
        //                 return run_n_bucket_sims(
        //                     buckets,
        //                     sketch,
        //                     n_trials,
        //                     *o,
        //                     ds
        //                 );
        //             },
        //             exact_threshold,
        //             mode
        //         );
        //
        //         // 64 k/2 k/2
        //         run_sims(
        //             results, ks, total_trials,
        //             format("unif2_bucket_{}_{}_k=64_ku=k/2_kh=k/2", bucket_name, o->name),
        //             [&](size_t k, size_t n_trials) {
        //                 auto buckets = [&] {
        //                     return bucket_gen(64);
        //                 };
        //
        //                 auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
        //                     return unif2_bucket_sketch(k / 2, k / 128, threshold, b, o, ds);
        //                 };
        //
        //                 return run_n_bucket_sims(
        //                     buckets,
        //                     sketch,
        //                     n_trials,
        //                     *o,
        //                     ds
        //                 );
        //             },
        //             exact_threshold,
        //             mode
        //         );
        //     }
        // }

        // Make sure all results are flushed in the end
        results.flush_to_file();
    }

    vector<double> run_n_ppswor_sims(size_t k, double threshold, size_t nsims, Dataset &ds) {
        auto t1 = chrono::high_resolution_clock::now();

        PPSWOR pp(k, 2, ds.item_counts.size());

        vector<double> estimates(nsims);
        size_t threshold_count = threshold * ds.lines.size();
        for (int i = 0; i < nsims; i++) {
            if (i != 0) pp.reset();

            for (auto item: ds.lines) {
                pp.update(item, 1);
            }

            auto sample = pp.sample();
            auto weights = get<1>(sample), probs = get<2>(sample);

            estimates[i] = estimate_threshold(weights, probs, threshold_count);

            cout << "." << flush;
        }
        cout << endl;

        auto t2 = chrono::high_resolution_clock::now();
        cout << " | " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() / 1000.0 / nsims
                << " seconds on average" << endl;

        return estimates;
    }

    vector<double> run_n_swa_sims(size_t kh, size_t kp, size_t ku, MockOracle &o, double threshold, size_t nsims,
                                  Dataset &ds) {
        auto t1 = chrono::high_resolution_clock::now();

        vector<double> estimates(nsims);
        size_t threshold_count = threshold * ds.lines.size();
        for (int i = 0; i < nsims; i++) {
            o.reset_estimates();

            auto sample = fake_swa_sample(kh, kp, ku, threshold, o, ds);
            auto weights = get<1>(sample), probs = get<2>(sample);

            estimates[i] = estimate_threshold(weights, probs, threshold_count);

            cout << "." << flush;
        }
        cout << endl;

        auto t2 = chrono::high_resolution_clock::now();
        cout << " | " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() / 1000.0 / nsims
                << " seconds on average" << endl;

        return estimates;
    }

    vector<double> run_n_bucket_sims(
        const function<Buckets()> bucket_gen,
        const function<double(Buckets &, MockOracle &, Dataset &)> sketch,
        size_t nsims, MockOracle &o, Dataset &ds) {
        auto t1 = chrono::high_resolution_clock::now();

        Buckets b = bucket_gen();

        vector<double> estimates(nsims);
        for (int i = 0; i < nsims; i++) {
            o.reset_estimates();

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
