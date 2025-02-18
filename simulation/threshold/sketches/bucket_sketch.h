#ifndef THRESHOLD_BUCKET_SKETCH_H
#define THRESHOLD_BUCKET_SKETCH_H
#include <functional>
#include <cmath>
#include "common/io/dataset.h"
#include "common/mock_oracle.h"
#include "common/bucket_sketch.h"

namespace threshold {

double bucket_sketch(size_t k_hh, double threshold,
                     const function<double(double, double, double)>& n_estimate, const Buckets& buckets, MockOracle& o, const Dataset& ds);

double smart_a_bucket_sketch(size_t k_hh, double threshold,
                     const function<double(double, double, double)>& n_estimate, const Buckets& buckets, MockOracle& o, const Dataset& ds);

double smart_b_bucket_sketch(size_t k_hh, double threshold,
                     const function<double(double, double, double)>& n_estimate, const Buckets& buckets, MockOracle& o, const Dataset& ds);

double alt_bucket_sketch(size_t k_hh, double threshold, const Buckets& buckets, MockOracle& o, const Dataset& ds);

double swa_bucket_sketch(size_t k_hh, size_t k_p, double threshold, const Buckets& buckets, MockOracle& o, const Dataset& ds);

double unif_bucket_sketch(size_t k_hh, size_t k_p, double threshold, const Buckets& buckets, MockOracle& o, const Dataset& ds);

double unif2_bucket_sketch(size_t k_hh, size_t k_p, double threshold, const Buckets& buckets, MockOracle& o, const Dataset& ds);

}

#endif //THRESHOLD_BUCKET_SKETCH_H
