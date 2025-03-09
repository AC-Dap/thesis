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

        constexpr size_t min_freq = 7;

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
                            auto buckets = [&] {
                                return bucket_gen(k / 2);
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
            }

            for (auto *o: os) {
                // k 0
                run_sims(
                    results, ks, total_trials,
                    format("unbiased_bucket_{}_kh=0", o->name),
                    [&](size_t k, size_t n_trials) {
                        auto buckets = [&] {
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
                        auto buckets = [&] {
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
                            auto buckets = [&] {
                                return bucket_gen(k / 2);
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
                            auto buckets = [&] {
                                return bucket_gen(k / 32);
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
            //
            // // Smart bucket estimators
            // for (auto &[n_estimate, estimate_name]: n_estimates) {
            //     for (auto *o: os) {
            //         // k/2 k/2 A
            //         run_sims(
            //             results, ks, total_trials,
            //             format("smart_a_expo_{}_{}_k=k/2_kh=k/2", estimate_name, o->name),
            //             [&](size_t k, size_t n_trials) {
            //                 auto buckets = [&] {
            //                     return generate_exponential_buckets(min_freq, k);
            //                 };
            //
            //                 auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
            //                     return smart_a_bucket_sketch(k / 2, deg, n_estimate, b, o, ds);
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
            //             exact_moment,
            //             mode
            //         );
            //
            //         // k/2 k/2 B
            //         run_sims(
            //             results, ks, total_trials,
            //             format("smart_b_expo_{}_{}_k=k/2_kh=k/2", estimate_name, o->name),
            //             [&](size_t k, size_t n_trials) {
            //                 auto buckets = [&] {
            //                     return generate_exponential_buckets(min_freq, k);
            //                 };
            //
            //                 auto sketch = [&](Buckets &b, MockOracle &o, Dataset &ds) {
            //                     return smart_b_bucket_sketch(k / 2, deg, n_estimate, b, o, ds);
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
            //             exact_moment,
            //             mode
            //         );
            //     }
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
            //                     return alt_bucket_sketch(0, deg, b, o, ds);
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
            //             exact_moment,
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
            //                     return alt_bucket_sketch(k / 2, deg, b, o, ds);
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
            //             exact_moment,
            //             mode
            //         );
            //     }
            // }
            //
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
            //                     return swa_bucket_sketch(0, size_t(sqrt(k)), deg, b, o, ds);
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
            //             exact_moment,
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
            //                     return swa_bucket_sketch(0, k / 64, deg, b, o, ds);
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
            //             exact_moment,
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
            //                     return swa_bucket_sketch(k / 2, k / 128, deg, b, o, ds);
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
            //             exact_moment,
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
            //                     return unif_bucket_sketch(k / 2, size_t(sqrt(k / 2.)), deg, b, o, ds);
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
            //             exact_moment,
            //             mode
            //         );
            //         results.flush_to_file();
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
            //                     return unif_bucket_sketch(0, k / 64, deg, b, o, ds);
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
            //             exact_moment,
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
            //                     return unif_bucket_sketch(k / 2, k / 128, deg, b, o, ds);
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
            //             exact_moment,
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
            //                     return unif2_bucket_sketch(k / 2, size_t(sqrt(k / 2.)), deg, b, o, ds);
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
            //             exact_moment,
            //             mode
            //         );
            //         results.flush_to_file();
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
            //                     return unif2_bucket_sketch(0, k / 64, deg, b, o, ds);
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
            //             exact_moment,
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
            //                     return unif2_bucket_sketch(k / 2, k / 128, deg, b, o, ds);
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
            //             exact_moment,
            //             mode
            //         );
            //     }
            // }

            // for (auto *o: os) {
            //     run_sims(
            //         results, ks, 2,
            //         format("3parsimonious_bucket_{}", o->name),
            //         [&](size_t k, size_t n_trials) {
            //             auto buckets = [&] {
            //                 return generate_exponential_buckets(min_freq, k);
            //             };
            //
            //             return run_n_parsimonious_bucket_sims(k, buckets, *o, deg, n_trials, ds);
            //         },
            //         exact_moment,
            //         mode
            //     );
            //
            //     run_sims(
            //         results, ks, 1,
            //         format("parsimonious_swa_{}", o->name),
            //         [&](size_t k, size_t n_trials) {
            //             return run_n_parsimonious_swa_sims(k/4, k/4, k/2, *o, deg, n_trials, ds);
            //         },
            //         exact_moment,
            //         mode
            //     );
            // }

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

    vector<double> run_n_parsimonious_bucket_sims(size_t k, function<Buckets()> bucket_gen,
                                           MockOracle &o, size_t deg, size_t nsims, Dataset &ds) {
        auto t1 = chrono::high_resolution_clock::now();

        Buckets b = bucket_gen();

        vector<double> estimates(nsims);
        for (int i = 0; i < nsims; i++) {
            o.reset_estimates();

            estimates[i] = parsimonious_bucket_sketch(k, deg, b, o, ds);

            cout << "." << flush;
        }
        cout << endl;

        auto t2 = chrono::high_resolution_clock::now();
        cout << " | " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() / 1000.0 / nsims
                << " seconds on average" << endl;

        return estimates;
    }

    vector<double> run_n_parsimonious_swa_sims(size_t kh, size_t kp, size_t k,
                                           MockOracle &o, size_t deg, size_t nsims, Dataset &ds) {
        auto t1 = chrono::high_resolution_clock::now();

        vector<double> estimates(nsims);
        for (int i = 0; i < nsims; i++) {
            o.reset_estimates();

            estimates[i] = parsimonious_swa_sketch(kh, kp, k, deg, o, ds);

            cout << "." << flush;
        }
        cout << endl;

        auto t2 = chrono::high_resolution_clock::now();
        cout << " | " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() / 1000.0 / nsims
                << " seconds on average" << endl;

        return estimates;
    }
}
