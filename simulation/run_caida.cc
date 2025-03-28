#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include <format>

#include "common/simulations.h"
#include "common/mock_oracle.h"
#include "common/ppswor.h"
#include "common/io/dataset.h"
#include "moments/estimator.h"
#include "moments/sketches/fake_swa.h"
#include "moments/sketches/bucket_sketch.h"

using namespace std;

size_t num_unique_items = 11211749;

vector<double> run_caida_buckets(Buckets& b, const function<double(Buckets &, MockOracle &, Dataset &)> sketch, MockOracle &o) {
    auto t1 = chrono::high_resolution_clock::now();

    Dataset ds(num_unique_items);

    vector<double> estimates(10);
    for(int i = 0; i < 10; i++){
        // Read in a and b
        ds.clear();
        ds.add_from_file("../data/processed/CAIDA/" + to_string(1 + i) + ".txt");
        ds.add_from_file("../data/processed/CAIDA/" + to_string(2 + i) + ".txt");

        estimates[i] = sketch(b, o, ds);

        cout << "." << flush;
    }

    auto t2 = chrono::high_resolution_clock::now();
    cout << " | " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() / 1000.0 / 10
            << " seconds on average" << endl;
    return estimates;
}

void run_caida_moment_sims(vector<MockOracle*> &os, string output_name);

int main(int argc, const char** argv) {
    // Read in datasets
    Dataset ds_train(num_unique_items), ds_test_all(num_unique_items);
    ds_train.add_from_file("../data/processed/CAIDA/train.txt");
    for(int i = 1; i <= 11; i++) {
        ds_test_all.add_from_file("../data/processed/CAIDA/" + to_string(i) + ".txt");
    }

    // Create oracles
    MockOracleAbsoluteError o_abs(0.001, "abs_0.001", ds_test_all);
    MockOracleRelativeError o_rel(0.05, "rel_0.05", ds_test_all);
    ExactOracle o_train("train", ds_train);
    vector<MockOracle*> os = {&o_abs, &o_rel, &o_train};

    // Run all sims
    run_caida_moment_sims(os, "caida");
}

void run_caida_moment_sims(vector<MockOracle*> &os, string output_name) {
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

        // Calculate exact moments
        vector<__uint128_t> exact_moments(10);
        Dataset ds(num_unique_items);

        for(int i = 0; i < 10; i++){
            // Read in a and b
            ds.clear();
            ds.add_from_file("../data/processed/CAIDA/" + to_string(1 + i) + ".txt");
            ds.add_from_file("../data/processed/CAIDA/" + to_string(2 + i) + ".txt");

            exact_moments[i] = moments::calculate_exact_moment(ds, deg);
        }

        cout << "Bucketing sketches:" << endl;
        for (auto k : ks) {
            for (auto *o : os) {
                o->reset_estimates();

                double min_freq = 1;
                for (ItemId item = 0; item < ds.item_counts.size(); item++) {
                    if (o->estimates[item] == 0) o->estimates[item] = 1e-9;
                    min_freq = min(min_freq, o->estimates[item]);
                }
                if (min_freq == 0) {
                    cerr << "Warning: min_freq is 0" << endl;
                    return;
                }

                vector<pair<function<Buckets(size_t)>, string> > bucket_types = {
                    {[&](size_t k) { return generate_exponential_buckets(min_freq, k); }, "expo"},
                    {generate_linear_buckets, "linear"}
                };

                cout << "Running for k = " << k << ", oracle = " << o->name << endl;
                // Central estimator
                for (auto &[bucket_gen, bucket_name]: bucket_types) {
                    Buckets b = bucket_gen(k);
                    auto estimates = run_caida_buckets(b, [&](Buckets &b, MockOracle &o, Dataset &ds) {
                        return moments::central_bucket_sketch(0, deg, b, o, ds);
                    }, *o);

                    b = bucket_gen(k / 2);
                    auto estimates2 = run_caida_buckets(b, [&](Buckets &b, MockOracle &o, Dataset &ds) {
                        return moments::central_bucket_sketch(k / 2, deg, b, o, ds);
                    }, *o);

                    for(int i = 0; i < 10; i++){
                        results.add_result("central_bucket_" + bucket_name + "_" + o->name + "_k=k_kh=0", k, i + 1, estimates[i], exact_moments[i]);
                        results.add_result("central_bucket_" + bucket_name + "_" + o->name + "_k=k/2_kh=k/2", k, i + 1, estimates2[i], exact_moments[i]);
                    }
                }

                // Unbiased estimator
                Buckets b = generate_linear_buckets(1);
                auto estimates1 = run_caida_buckets(b, [&](Buckets &b, MockOracle &o, Dataset &ds) {
                    return moments::unbiased_bucket_sketch(0, deg, b, o, ds);
                }, *o);
                auto estimates2 = run_caida_buckets(b, [&](Buckets &b, MockOracle &o, Dataset &ds) {
                    return moments::unbiased_bucket_sketch(k, deg, b, o, ds);
                }, *o);

                for(int i = 0; i < 10; i++){
                    results.add_result("unbiased_bucket_" + o->name + "_kh=0", k, i + 1, estimates1[i], exact_moments[i]);
                    results.add_result("unbiased_bucket_" + o->name + "_kh=k", k, i + 1, estimates2[i], exact_moments[i]);
                }

                // Counting estimator
                for (auto &[bucket_gen, bucket_name]: bucket_types) {
                    Buckets b = bucket_gen(k);
                    auto estimates = run_caida_buckets(b, [&](Buckets &b, MockOracle &o, Dataset &ds) {
                        return moments::counting_bucket_sketch(0, deg, b, o, ds);
                    }, *o);

                    b = bucket_gen(k / 2);
                    auto estimates2 = run_caida_buckets(b, [&](Buckets &b, MockOracle &o, Dataset &ds) {
                        return moments::counting_bucket_sketch(k / 2, deg, b, o, ds);
                    }, *o);

                    for(int i = 0; i < 10; i++){
                        results.add_result("counting_bucket_" + bucket_name + "_" + o->name + "_k=k_kh=0", k, i + 1, estimates[i], exact_moments[i]);
                        results.add_result("counting_bucket_" + bucket_name + "_" + o->name + "_k=k/2_kh=k/2", k, i + 1, estimates2[i], exact_moments[i]);
                    }
                }
            }
        }

        // Make sure all results are flushed in the end
        results.flush_to_file();
    }
}
