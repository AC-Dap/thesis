#ifndef MOMENTS_PARSIMONIOUS_H
#define MOMENTS_PARSIMONIOUS_H

#include <cmath>

#include "common/io/dataset.h"
#include "common/mock_oracle.h"
#include "common/bucket_sketch.h"

using namespace std;

namespace moments {

double parsimonious_bucket_sketch(size_t k, size_t deg, Buckets& b, MockOracle& o, Dataset& ds);
double parsimonious_swa_sketch(size_t kh, size_t kp, size_t k, size_t deg, MockOracle& o, Dataset& ds);

}

#endif //MOMENTS_PARSIMONIOUS_H
