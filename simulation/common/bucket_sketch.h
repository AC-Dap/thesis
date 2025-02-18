#ifndef BUCKET_SKETCH_H
#define BUCKET_SKETCH_H
#include <functional>
#include <vector>
#include "common/io/dataset.h"
#include "common/mock_oracle.h"
#include "common/heap.h"

using namespace std;

typedef vector<double> Buckets;

Buckets generate_exponential_buckets(double min_freq, size_t k);
Buckets generate_linear_buckets(size_t k);

void process_ds_to_buckets(function<void(ItemId)> process_item, Heap<tuple<double, ItemId>>& top_h, MockOracle& o, const Dataset& ds);

double n_estimate_left(double S, double left, double right);

double n_estimate_right(double S, double left, double right);

double n_estimate_left_round(double S, double left, double right);

double n_estimate_right_round(double S, double left, double right);

double n_estimate_arith_avg(double S, double left, double right);

double n_estimate_geo_avg(double S, double left, double right);

double n_estimate_harm_avg(double S, double left, double right);


#endif //BUCKET_SKETCH_H
