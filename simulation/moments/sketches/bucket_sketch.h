#ifndef MOMENTS_BUCKET_SKETCH_H
#define MOMENTS_BUCKET_SKETCH_H
#include <functional>
#include <cmath>
#include "common/io/dataset.h"
#include "common/mock_oracle.h"
#include "common/bucket_sketch.h"

namespace moments {
    double central_bucket_sketch(size_t k_hh, size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds);

    double unbiased_bucket_sketch(size_t k_hh, size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds);

    double counting_bucket_sketch(size_t k_hh, size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds);

    double sampling_bucket_sketch(size_t k_hh, size_t k_u, size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds);
}

#endif //MOMENTS_BUCKET_SKETCH_H
