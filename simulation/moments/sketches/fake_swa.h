#ifndef MOMENTS_FAKE_SWA_H
#define MOMENTS_FAKE_SWA_H

#include <vector>
#include <tuple>

#include "common/mock_oracle.h"
#include "common/io/dataset.h"

using namespace std;

namespace moments {

/**
 * We note that a SWA sample does not depend on the order of items received.
 * This means we can directly generate a sample without needing to simulate the data stream.
 *
 * We pass `items` by value, since we will modify it.
 */
tuple<vector<ItemId>, vector<double>, vector<double>> fake_swa_sample(
    size_t kh, size_t kp, size_t ku, size_t deg, MockOracle& oracle, Dataset& ds);

}

#endif