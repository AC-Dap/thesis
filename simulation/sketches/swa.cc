#include "swa.h"

/**
 * Tries inserting `item` into the heap `h`, return true if an overflow occurs.
 * If the heap is full, we update `o_item` and `o_count` with the overflowed item.
 */
bool try_insert(Heap<tuple<double, const string*, size_t>>& h, tuple<double, const string*, size_t> item, const string*& o_item, size_t& o_count) {
    if(h.len < h.cap) {
        h.push(item);
        return false;
    }

    auto overflow = h.pushpop(item);
    o_item = get<1>(overflow);
    o_count = get<2>(overflow);
    return true;
}

void SWA::update(const string* item, size_t count) {
    // Check if item is already in heap; if so, just update count
    // Shouldn't need to reheapify, since heap is not ordered by count
    for(int i = 0; i < h_heap.len; i++){
        if(get<1>(h_heap.heap[i]) == item) {
            get<2>(h_heap.heap[i]) += count;
            return;
        }
    }
    for(int i = 0; i < p_heap.len; i++){
        if(get<1>(p_heap.heap[i]) == item) {
            get<2>(p_heap.heap[i]) += count;
            return;
        }
    }
    for(int i = 0; i < u_heap.len; i++){
        if(get<1>(u_heap.heap[i]) == item) {
            get<2>(u_heap.heap[i]) += count;
            return;
        }
    }

    // Insertion order: h_heap -> p_heap -> u_heap
    auto weight = oracle.estimate(item);
    const string* o_item = item;
    size_t o_count = count;
    if(!try_insert(h_heap, {weight, o_item, o_count}, o_item, o_count)) return;

    weight = pow(oracle.estimate(o_item), deg) / seed[o_item];
    if(!try_insert(p_heap, {weight, o_item, o_count}, o_item, o_count)) return;

    weight = -seed[o_item];
    try_insert(u_heap, {weight, o_item, o_count}, o_item, o_count);
}

tuple<vector<const string*>, vector<double>, vector<double>> SWA::sample() {
    vector<const string*> s(kh + kp + ku);
    vector<double> weights(kh + kp + ku), probs(kh + kp + ku);

    for(int i = 0; i < kh; i++){
        s[i] = get<1>(h_heap.heap[i]);
        weights[i] = get<2>(h_heap.heap[i]);
        probs[i] = 1;
    }

    auto tau = get<0>(p_heap.heap[0]);
    for(int i = 0; i < kp; i++){
        s[kh + i] = get<1>(p_heap.heap[i+1]);
        weights[kh + i] = get<2>(p_heap.heap[i+1]);
        probs[kh + i] = 1 - exp(-pow(oracle.estimate(s[kh + i]), deg) / tau);
    }

    tau = get<0>(u_heap.heap[0]);
    for(int i = 0; i < ku; i++) {
        s[kh + ku + i] = get<1>(u_heap.heap[i+1]);
        weights[kh + ku + i] = get<2>(u_heap.heap[i+1]);
        probs[kh + ku + i] = 1 - exp(-tau);
    }

    return {s, weights, probs};
}