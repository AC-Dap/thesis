#ifndef BUCKET_SKETCH_H
#define BUCKET_SKETCH_H
#include <functional>
#include <vector>
#include <cmath>

#include "dataset.h"
#include "mock_oracle.h"

using namespace std;

typedef vector<double> Buckets;

Buckets generate_exponential_buckets(double min_freq, size_t k);
Buckets generate_linear_buckets(size_t k);

double bucket_sketch(size_t k_hh, size_t deg, const function<double(double, double, double)>& n_estimate, const Buckets& buckets, MockOracle& o, const Dataset& ds);

double cond_bucket_sketch(size_t k_hh, size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds);

double alt_bucket_sketch(size_t k_hh, size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds);

double n_estimate_left(double S, double left, double right);

double n_estimate_right(double S, double left, double right);

double n_estimate_left_round(double S, double left, double right);

double n_estimate_right_round(double S, double left, double right);

double n_estimate_arith_avg(double S, double left, double right);

double n_estimate_geo_avg(double S, double left, double right);

double n_estimate_harm_avg(double S, double left, double right);


#endif //BUCKET_SKETCH_H
