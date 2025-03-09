#ifndef THRESHOLD_BUCKET_SKETCH_H
#define THRESHOLD_BUCKET_SKETCH_H
#include <functional>
#include <cmath>
#include "common/io/dataset.h"
#include "common/mock_oracle.h"
#include "common/bucket_sketch.h"

namespace threshold {

    double central_bucket_sketch(size_t k_hh, double threshold, const Buckets& buckets, MockOracle& o, const Dataset& ds);

    double counting_bucket_sketch(size_t k_hh, double threshold, const Buckets& buckets, MockOracle& o, const Dataset& ds);

    double sampling_bucket_sketch(size_t k_hh, size_t k_u, double threshold, const Buckets& buckets, MockOracle& o, const Dataset& ds);

}

#endif //THRESHOLD_BUCKET_SKETCH_H
