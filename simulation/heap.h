#ifndef HEAP_H
#define HEAP_H

#include <vector>
#include <utility>

using namespace std;

template<typename T>
struct Heap {
    Heap(size_t cap): len(0), cap(cap), heap(cap) {}

    void push(T el);
    T pop();
    T pushpop(T el);
    void heapify(size_t i);

    size_t len, cap;
    vector<T> heap;
};

template<typename T>
void Heap<T>::push(T el) {
    if(len == cap) throw "Pushing to filled heap";

    heap[len] = el;

    int curr_i = len;
    while(curr_i != 0) {
        int parent_i = (curr_i - 1) / 2;
        if(heap[parent_i] > heap[curr_i]) {
            std::swap(heap[parent_i], heap[curr_i]);
            curr_i = parent_i;
        } else {
            break;
        }
    }
    len++;
}

template<typename T>
T Heap<T>::pop() {
    if(len == 0) throw "Popping on empty heap";

    T ret = heap[0];

    if(len == 1) {
        len = 0;
    } else {
        std::swap(heap[0], heap[len - 1]);
        len--;

        heapify(0);
    }

    return ret;
}

template<typename T>
T Heap<T>::pushpop(T el) {
    if(len == 0 || el < heap[0]) return el;

    T ret = heap[0];
    heap[0] = el;
    heapify(0);
    return ret;
}

template<typename T>
void Heap<T>::heapify(size_t i) {
    int curr_i = i;
    while(true) {
        int left_i = 2*curr_i + 1;
        int right_i = 2*curr_i + 2;

        int smallest_i = curr_i;
        if (left_i < len && heap[left_i] < heap[smallest_i]) {
            smallest_i = left_i;
        }
        if (right_i < len && heap[right_i] < heap[smallest_i]) {
            smallest_i = right_i;
        }

        if (smallest_i != curr_i) {
            std::swap(heap[curr_i], heap[smallest_i]);
            curr_i = smallest_i;
        } else {
            break;
        }
    }
}

#endif