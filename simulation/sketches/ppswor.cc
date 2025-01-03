#include "ppswor.h"

#include "count_sketch.h"
#include "hashing.h"

using namespace std;

void PPSWOR::update(ItemId item, double count) {
    double w = count / seed[item];
    cs.update(item, w);
}

tuple<vector<ItemId>, vector<double>, vector<double>> PPSWOR::sample() {
    auto hh = cs.heavy_hitters();
    vector<ItemId> items(k);
    vector<double> weights(k), probs(k);

    auto tau = cs.estimate(hh[0]);
    for(int i = 0; i < k; i++){
        auto s = hh[1 + i];
        auto est = cs.estimate(s);
        auto sample_count = est * seed[s];
        auto sample_prob = 1 - exp(-pow(sample_count/tau, deg));

        items[i] = s;
        weights[i] = sample_count;
        probs[i] = sample_prob;
    }

    return {items, weights, probs};
}