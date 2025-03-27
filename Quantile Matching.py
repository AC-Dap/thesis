# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
def expo_buckets(min_freq, k):
    step_size = np.exp(-np.log(min_freq) / (k-2))
    buckets = 1 / step_size ** np.arange(k)
    buckets = np.append(buckets, 0)
    return buckets[::-1]

def linear_buckets(k):
    return np.arange(k+1) / k

# Place each item into a bucket
def find_bucket_index(buckets, value):
    if value == 0:
        return 0
    return np.searchsorted(buckets, value, side='left') - 1

def place_in_buckets(buckets, est_freqs, true_counts):
    k = len(buckets) - 1
    filled_buckets = [[] for _ in range(k)]

    sorted_freqs = est_freqs.sort_values(ascending=True)
    curr_bucket = 0
    for item, freq in sorted_freqs.items():
        while buckets[curr_bucket + 1] < freq:
            curr_bucket += 1
        filled_buckets[curr_bucket].append(true_counts[item])

    for i in range(k):
        filled_buckets[i] = np.array(filled_buckets[i])
    return filled_buckets


# %%
# Helper Functions
rng = np.random.default_rng()

def rel_oracle_est(freqs, ep, N):
    oracle_freqs = freqs.copy()
    err = rng.uniform(1-ep, 1+ep, len(freqs))
    i = 0
    for item, freq in oracle_freqs.items():
        oracle_freqs[item] = min(1, max(0, err[i] * freq))
        if oracle_freqs[item] == 0:
            oracle_freqs[item] = 1e-9
        i += 1
    return oracle_freqs

def abs_oracle_est(freqs, ep, N):
    oracle_freqs = freqs.copy()
    err = rng.uniform(-ep, ep, len(freqs))
    i = 0
    for item, freq in oracle_freqs.items():
        oracle_freqs[item] = min(1, max(0, err[i] + freq))
        if oracle_freqs[item] == 0:
            oracle_freqs[item] = 1e-9
        i += 1
    return oracle_freqs

def train_oracle_est(freqs, train_freqs, N):
    oracle_freqs = freqs.copy()
    for item, freq in oracle_freqs.items():
        if item in train_freqs:
            oracle_freqs[item] = train_freqs[item]
        else:
            oracle_freqs[item] = 1/N
    return oracle_freqs / np.sum(oracle_freqs)


# %%
def bucket_quantile_est(buckets, est_freqs, true_counts):
    k = len(buckets) - 1
    N = np.sum(true_counts)

    bucket_count_est = np.zeros(k)
    bucket_count_act = np.zeros(k)
    bucket_sum = np.zeros(k)
    bucket_min_f = np.full(k, np.inf, dtype=float)
    bucket_min_count = np.full(k, np.inf, dtype=float)
    bucket_max_f = np.full(k, 0, dtype=float)
    bucket_max_count = np.full(k, 0, dtype=float)

    curr_bucket = 0
    curr_sample = []
    sorted_freqs = est_freqs.sort_values(ascending=True)
    for item, f in sorted_freqs.items():
        while buckets[curr_bucket + 1] < f:
            if len(curr_sample) > 0:
                s = np.random.choice(curr_sample, min(len(curr_sample), 16))
                bucket_min_count[curr_bucket] = np.min(s)
                bucket_max_count[curr_bucket] = np.max(s)
            curr_sample = []
            curr_bucket += 1
        curr_sample.append(true_counts[item])
        bucket_count_act[curr_bucket] += 1
        bucket_sum[curr_bucket] += true_counts[item]
        bucket_count_est[curr_bucket] += true_counts[item] / (N * f)
    s = np.random.choice(curr_sample, min(len(curr_sample), 16))
    bucket_min_count[curr_bucket] = np.min(s)
    bucket_max_count[curr_bucket] = np.max(s)
        # if f < bucket_min_f[curr_bucket]:
        #     bucket_min_f[curr_bucket] = f
        #     bucket_min_count[curr_bucket] = true_counts[item]
        # if f > bucket_max_f[curr_bucket]:
        #     bucket_max_f[curr_bucket] = f
        #     bucket_max_count[curr_bucket] = true_counts[item]

    # for i in range(k):
    #     if bucket_sum[i] > 0:
    #         bucket_count_est[i] = int(np.ceil(bucket_sum[i] / (N * (buckets[i] + buckets[i+1])/2)))
    #     bucket_min_count[i] = N * buckets[i]
    #     bucket_max_count[i] = N * buckets[i+1]

    print("Est count:", np.sum(bucket_count_est))
    print("Act count:", np.sum(bucket_count_act))

    return bucket_count_est, bucket_sum, bucket_min_count, bucket_max_count, bucket_count_act

    # Generate fake data
    # els = []
    # for i in range(k):
    #     if bucket_sum[i] > 0:
    #         els.append(np.linspace(N * buckets[i], N * buckets[i+1], int(np.ceil(bucket_count_est[i])), endpoint=True))
    # els = np.concatenate(els)
    # els.sort()

    # return els[::-1], np.array(true_counts)

def estimate_rank(x, bucket_count_est, bucket_min_count, bucket_max_count):
    k = len(bucket_count_est)
    num_greater = 0
    for i in range(k):
        if bucket_count_est[i] == 0:
            continue
        
        m, M, n = bucket_min_count[i], bucket_max_count[i], bucket_count_est[i]
        m, M = min(m, M), max(m, M)
        if x < m: 
            num_greater += 0
        elif x >= M:
            num_greater += n
        else:
            d = (M - m) / (n - 1)
            k = int(np.floor((x - m) / d))
            num_greater += k + 1
    return num_greater / np.sum(bucket_count_est)


# %%
### Load data
def read_processed_data(file):
    df = pd.read_csv(f'data/processed/{file}', names=['id'])
    unique_counts = df['id'].value_counts()
    unique_freqs = unique_counts / len(df)
    return unique_counts, unique_freqs


aol_train_counts, aol_train_freqs = read_processed_data('AOL/train.txt')
aol_test_counts, aol_test_freqs = read_processed_data('AOL/test.txt')

# %%
k=4096
rel_est_freqs = rel_oracle_est(aol_test_freqs, 0.05, np.sum(aol_test_counts))
abs_est_freqs = abs_oracle_est(aol_test_freqs, 0.001, np.sum(aol_test_counts))
train_est_freqs = train_oracle_est(aol_test_freqs, aol_train_freqs, np.sum(aol_test_counts))


# %%
def plot_quantile_errors(k, est_freqs, counts, label):
    buckets = expo_buckets(np.min(est_freqs), k)
    a, b, c = bucket_quantile_est(buckets, est_freqs, counts)

    for i in range(k):
        if a[i] > 0:
            print(b[i], c[i])

    xs = np.logspace(np.log10(np.min(counts)), np.log10(np.max(counts)), 1000)
    qs_act = np.zeros(len(xs))
    qs_est = np.zeros(len(xs))
    for i, x in enumerate(xs):
        qs_act[i], qs_est[i] = quantile(counts, x, a, b, c)
    
    err = np.abs(qs_act - qs_est) / qs_act
    # plt.plot(xs, qs_act, c='yellow')
    plt.plot(qs_act, err, label=label)
    plt.xscale('log')
    plt.yscale('log')

k = 16384
# plot_quantile_errors(k, rel_est_freqs, aol_test_counts, "rel")
# plot_quantile_errors(k, abs_est_freqs, aol_test_counts, "abs")
plot_quantile_errors(k, train_est_freqs, aol_test_counts, "train")

plt.legend()
plt.show()

# quantile(aol_test_counts, 1, a, b, c)
# quantile(aol_test_counts, 16, a, b, c)
# quantile(aol_test_counts, 64, a, b, c)
# quantile(aol_test_counts, 256, a, b, c)
# quantile(aol_test_counts, 2048, a, b, c)
# quantile(aol_test_counts, 65536, a, b, c)

# %%
def compute_f(s, N, M, m):
    k_M = int(np.floor((N / M)**(1/s)))
    k_m = int(np.ceil((N / m)**(1/s)))
    ks = np.arange(k_M, k_m + 1)
    total = np.sum(np.floor(N / (ks ** s)))
    return total

def find_s(target_f, N, M, m, tol=1e-5, s_min=1.0, s_max=5.0):
    while s_max - s_min > tol:
        s_mid = (s_min + s_max) / 2
        f_mid = compute_f(s_mid, N, M, m)
        
        if f_mid < target_f:
            s_max = s_mid
        else:
            s_min = s_mid
    
    return (s_min + s_max) / 2


# %%
def est_rank_2(x, N, b_sums, b_mins, b_maxes, fitted_s):
    k = len(b_mins)
    num_smaller = 0
    count = 0
    for i in range(k):
        if b_sums[i] == 0:
            continue
        
        m, M, s = b_mins[i], b_maxes[i], fitted_s[i]
        k_M = int(np.floor((N / M)**(1/s)))
        k_m = int(np.ceil((N / m)**(1/s)))
        
        # How many np.floor(N / (k**s)) <= x:
        threshold = (N / (x + 1)) ** (1 / s)
        l_min = np.floor(threshold) + 1
        l_min = max(l_min, k_M)
        
        if l_min <= k_m:
            print(m, M, k_m, k_M, l_min)
            num_smaller += k_m - l_min + 1

        count += k_m - k_M + 1
    return num_smaller / count


# %%
k = 4096
# buckets = np.concatenate((np.linspace(0, 0.5, k, endpoint=False), expo_buckets(0.5, k)[2:]))
buckets = expo_buckets(np.min(rel_est_freqs), k)
b_counts, b_sums, b_mins, b_maxs, b_count_real = bucket_quantile_est(buckets, rel_est_freqs, aol_test_counts)

fitted_s = np.zeros(k)
for i in range(k):
    if b_sums[i] > 0:
        fitted_s[i] = find_s(b_sums[i], N, b_maxs[i], b_mins[i])

# %%
# def quantile(data, x, a, b, c):
#     act = np.searchsorted(data[::-1], x, side='right') / len(data)
#     10, np.sum(aol_test_counts), b_sums, b_mins, b_maxs, fitted_s)
#     est = est_rank2(x, a, b, c)
#     return act, est
    # err = np.abs(act - est) / act
    # print(f"Rank of {x}: {act} (actual), {est} (est), {err}")

xs = np.logspace(np.log10(np.min(aol_test_counts)), np.log10(np.max(aol_test_counts)), 1000)
qs_act = np.zeros(len(xs))
qs_est = np.zeros(len(xs))
N = np.sum(aol_test_counts)
for i, x in enumerate(xs):
    qs_act[i] = np.searchsorted(aol_test_counts[::-1], x, side='right') / len(aol_test_counts)
    qs_est[i] = est_rank_2(x, N, b_sums, b_mins, b_maxs, fitted_s)
    # qs_act[i], qs_est[i] = quantile(aol_test_counts, x, b_counts, b_mins, b_maxs)

err = np.abs(qs_act - qs_est)
# plt.plot(xs, qs_act, c='yellow')
plt.plot(qs_act, err)
plt.axhline(0.1)
plt.xscale('log')
plt.yscale('log')

# %%
print(k, len(b_mins), len(b_maxs), len(fitted_s))
est_rank_2(10, np.sum(aol_test_counts), b_sums, b_mins, b_maxs, fitted_s)

# %%
print(estimate_rank(1, b_counts, b_mins, b_maxs))
# print(est_rank_2(1, N, b_sums, b_mins, b_maxs, fitted_s))
N = np.sum(aol_test_counts)
c_tot = 0
for i in range(k):
    if b_counts[i] > 0:
        m, M = b_mins[i], b_maxs[i]
        s = find_s(b_sums[i], N, M, m)
        c = int(np.ceil((N / m)**(1/s))) - int(np.floor((N / M)**(1/s))) + 1
        c_m = np.ceil(b_sums[i] / b_maxs[i])
        c_M = np.ceil(b_sums[i] / b_mins[i])
        print(b_mins[i], b_maxs[i], b_sums[i], b_counts[i], c, max(c_m, min(c, c_M)), b_count_real[i])
        c_tot += c
print(c_tot)

# %%
"""
1.0 5.0 128323.0 85711.80342410224 44835.0
1.0 79.0 77359.0 25835.50649953942 22492.0
1.0 22.0 55924.0 12451.254243033076 13864.0
1.0 7.0 35276.0 4712.435927443685 6837.0
"""
N = np.sum(aol_test_counts)
M = 22
m = 1
for s in np.linspace(1, 2, 20):
    # 1/k^s == M/N -> k=(N/M)**(1/s)
    k_M = (N / M)**(1/s)
    k_m = (N / m)**(1/s)
    # print(f"{s}: {k_M} - {k_m}")
    k_M = int(np.floor(k_M))
    k_m = int(np.ceil(k_m))
    tot = 0
    for k in range(k_M, k_m+1):
        tot += 1/(k**s)
    print(f"{s}: {tot * N} - {k_m - k_M + 1}")

# %%
k=4096
est_freqs = abs_oracle_est(aol_test_freqs, 0.001, np.sum(aol_test_counts))
buckets = expo_buckets(np.min(est_freqs), k)
est, act = bucket_quantile_est(buckets, est_freqs, aol_test_counts)
print("Predicted els", len(est))
print("Total els", len(act))
# print(est[-20:])
# print(act[-20:])

plt.plot(np.arange(len(act))[::-1], act, label="Real")
plt.plot(np.arange(len(est))[::-1], est, label="Predicted")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

# %%
# k=4096
est_freqs = train_oracle_est(aol_test_freqs, aol_train_freqs, np.sum(aol_test_counts))
buckets = expo_buckets(np.min(est_freqs), k)
est, act = bucket_quantile_est(buckets, est_freqs, aol_test_counts)
print("Predicted els", len(est))
print("Total els", len(act))
# print(est[-20:])
# print(act[-20:])

def quantile(q):
    plt.axvline(len(act) * q, ls='--')
    print(est[int(len(est) *  q)], act[int(len(act) *  q)])

plt.plot(np.arange(len(act)), act, label="Real")
plt.plot(np.arange(len(est)), est, label="Predicted")
quantile(0.01)
quantile(0.5)
quantile(0.99)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

# %%
import cvxpy as cp
import numpy as np
import scipy

# Problem data
N = 1000
M = 10
low, high = 0.09, 0.11
ps = np.random.uniform(low=low, high=high, size=M)
ps /= np.sum(ps)
data = np.random.multinomial(N, ps)

moments = [np.sum(data**i) for i in range(0, 6)]

# Moment matching
def falling_power(n, k):
    return scipy.special.poch(n - k + 1, k)

def power(p, k):
    return p**k

def generate_moment_constraint(m, p, m_val):
    moment_eq = 0
    for k in range(1, m+1):
        moment_eq += scipy.special.stirling2(m, k) * falling_power(N, k) * power(p, k)
    return np.sum(moment_eq) - m_val

def obj(p):
    center = (high + low)/2
    epsilon = 1e-8
    return np.sum((p - center)**2) + epsilon*np.sum(p**2)

constraints = [
    {'type': 'eq', 'fun': lambda p: np.sum(p) - 1},
    {'type': 'eq', 'fun': lambda p: generate_moment_constraint(2, p, moments[2])},
    # {'type': 'eq', 'fun': lambda p: generate_moment_constraint(3, p, moments[3])},
    # {'type': 'eq', 'fun': lambda p: generate_moment_constraint(4, p, moments[4])},
    # {'type': 'eq', 'fun': lambda p: generate_moment_constraint(5, p, moments[5])},
]
bounds = [(0, 1)] * M
p_init = np.ones(M) / M
res = scipy.optimize.minimize(obj, p_init, constraints=constraints, bounds=bounds, method='SLSQP', options={'ftol': 1e-9})

print(res)

# %%
import cvxpy as cp
import numpy as np
import scipy

# Problem data
N = 10000
M = 1000
low, high = 0.99*1/M, 1.01*1/M
ps = np.random.uniform(low=low, high=high, size=M)
ps /= np.sum(ps)
data = np.random.multinomial(N, ps)

moments = [np.sum(data**i) for i in range(0, 6)]
print(moments)

# Moment matching
def obj(x):
    center = (high + low)/2 * N
    return np.sum((x - center)**2)

constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - N},
    {'type': 'eq', 'fun': lambda x: np.sum(x**2) - moments[2], 'jac': lambda x: 2 * x},
    # {'type': 'eq', 'fun': lambda x: np.sum(x**3) - moments[3]},
    # {'type': 'eq', 'fun': lambda x: np.sum(x**4) - moments[4]},
    # {'type': 'eq', 'fun': lambda x: np.sum(x**5) - moments[5]},
]
bounds = [(0, N)] * M
x_init = np.ones(M) * N / M
res = scipy.optimize.minimize(obj, x_init, constraints=constraints, bounds=bounds, method='COBYLA', options={'ftol': 1e-3})

print(res)

# %%
import cvxpy as cp
import numpy as np
import scipy

# Problem data
N = 1000
M = 10
low, high = 0.099, 0.101
ps = np.random.uniform(low=low, high=high, size=M)
ps /= np.sum(ps)
print(ps)
data = np.random.multinomial(N, ps)

moments = [np.sum(data**i) for i in range(0, 6)]
print(moments)

# Moment matching
center = (high + low)/2

def falling_power(n, k):
    return scipy.special.poch(n - k + 1, k)

def power(dels, k):
    # Estimate (center + del)^k with linear expansion
    return center ** k + k * (center ** (k-1)) * dels + k * (k-1) / 2 * (center ** (k-2)) * (dels ** 2)

def generate_moment_constraint(m, dels, m_val):
    moment_eq = 0
    for k in range(1, m+1):
        moment_eq += scipy.special.stirling2(m, k) * falling_power(N, k) * power(dels, k)
    return cp.sum(moment_eq) == m_val

# Model each probability as center + delta
deltas = cp.Variable(M)
objective = cp.Minimize(cp.sum_squares(deltas))

constraints = [
    center * M + cp.sum(deltas) == 1,
    deltas >= -center
]
for m in range(2, 3):
    tolerance = 1e-2
    constraints.append(cp.abs(generate_moment_constraint(m, deltas, moments[m]).args[0]) <= tolerance)
    # constraints.append(generate_moment_constraint(m, deltas, moments[m]))

prob = cp.Problem(objective, constraints)

# Solve the problem (solver can be chosen explicitly if desired)
prob.solve(solver=cp.SCS, verbose=True)

optimized_probs = center + deltas.value
print("Optimized probabilities:", optimized_probs)
print("Sum of probabilities (should be 1):", np.sum(optimized_probs))

# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval, chebvander

def clenshaw_curtis(N):
    """
    Compute Clenshaw–Curtis quadrature nodes and weights on the interval [-1,1].
    This implementation is O(N^2) which is acceptable for moderate N (e.g., 50–100).
    """
    if N == 0:
        return np.array([0.0]), np.array([2.0])
    j = np.arange(0, N + 1)
    # Nodes: cosine-spaced between -1 and 1.
    x = np.cos(np.pi * j / N)
    w = np.zeros(N + 1)
    # Compute weights using the standard formula:
    for i in range(N + 1):
        s = 0.0
        # k runs from 0 to floor(N/2)
        for k in range(0, N // 2 + 1):
            if 2 * k == N:
                b = 1.0
            else:
                b = 2.0
            s += b * np.cos(2 * k * np.pi * i / N) / (1 - 4 * k * k)
        w[i] = 2.0 / N * s
    return x, w

class ProductionChebyshevMomentSolver:
    def __init__(self, target_cheb_moments, xmin, xmax, order, quad_points=100):
        """
        Parameters:
          target_cheb_moments: array-like of target Chebyshev moments for degrees 0..order-1.
                               For a properly normalized density, target_cheb_moments[0] should equal 1.
          xmin, xmax:        the original support of the data.
          order:             the number of moments (and hence number of parameters θ).
          quad_points:       the number of quadrature points used in Clenshaw–Curtis integration.
        """
        self.target_moments = np.array(target_cheb_moments)  # target moments in the Chebyshev basis
        self.xmin = xmin
        self.xmax = xmax
        self.order = order
        self.quad_points = quad_points
        # Precompute quadrature nodes and weights on [-1, 1]
        self.u_nodes, self.weights = clenshaw_curtis(quad_points)
        # Precompute the Chebyshev Vandermonde matrix evaluated at the quadrature nodes.
        # Each row i contains [T_0(u_i), T_1(u_i), ..., T_{order-1}(u_i)]
        self.T = chebvander(self.u_nodes, order - 1)
        # Initialize θ to zeros.
        self.theta = np.zeros(order)
        self.converged = False

        print("U",self.u_nodes)

    def potential_integrals(self, theta):
        """
        For a given parameter vector θ, compute the following approximations using quadrature:
          Z = ∫₋₁¹ exp(Σ_j θ_j T_j(u)) du
          m_i = (1/Z) * ∫₋₁¹ T_i(u) exp(Σ_j θ_j T_j(u)) du,  for i = 0, …, order-1
          Q_ij = (1/Z) * ∫₋₁¹ T_i(u) T_j(u) exp(Σ_j θ_j T_j(u)) du
        """
        # Compute the exponent at each quadrature node.
        print("T",theta)
        exponent = self.T.dot(theta)  # shape: (num_nodes,)
        print("E",exponent)
        f_vals = np.exp(exponent)       # unnormalized density at nodes
        # Approximate Z via quadrature.
        Z = np.sum(self.weights * f_vals)
        # Compute the first moments (m) in the Chebyshev basis.
        m = np.dot(self.weights * f_vals, self.T) / Z  # shape: (order,)
        # Compute the second moments matrix Q.
        Q = np.zeros((self.order, self.order))
        for i in range(self.order):
            for j in range(self.order):
                Q[i, j] = np.sum(self.weights * f_vals * self.T[:, i] * self.T[:, j]) / Z
        return Z, m, Q

    def compute_gradient_hessian(self, theta):
        """
        Define the potential function:
          L(θ) = log(Z) - Σ_i θ_i * (target moment)_i,
        whose gradient is
          ∇L = m - (target moments)
        and whose Hessian is
          H = Q - m mᵀ.
        """
        Z, m, Q = self.potential_integrals(theta)
        grad = m - self.target_moments
        H = Q - np.outer(m, m)
        return grad, H

    def solve_theta(self, tol=1e-8, max_iter=100, damping=1e-8):
        """
        Use Newton's method to solve for θ so that the estimated Chebyshev
        moments match the target moments. A simple backtracking line search
        (by reducing the step) is applied if the potential does not decrease.
        """
        theta = self.theta.copy()
        for it in range(max_iter):
            grad, H = self.compute_gradient_hessian(theta)
            norm_grad = np.linalg.norm(grad)
            print(norm_grad)
            if norm_grad < tol:
                self.converged = True
                break
            try:
                delta = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                # In case of singular Hessian, use least-squares.
                delta = np.linalg.lstsq(H, -grad, rcond=None)[0]
            # Apply damping/line search.
            print("D",delta)
            step = damping
            theta_new = theta + step * delta
            Z_new, _, _ = self.potential_integrals(theta_new)
            L_new = np.log(Z_new) - np.dot(theta_new, self.target_moments)
            Z_old, _, _ = self.potential_integrals(theta)
            L_old = np.log(Z_old) - np.dot(theta, self.target_moments)
            while L_new > L_old and step > 1e-6:
                step *= 0.5
                theta_new = theta + step * delta
                Z_new, _, _ = self.potential_integrals(theta_new)
                L_new = np.log(Z_new) - np.dot(theta_new, self.target_moments)
            theta = theta_new
            # (Optional: log iteration details, e.g., print(f"Iter {it}, ||grad||={norm_grad:.2e}, L={L_new:.6f}"))
        self.theta = theta
        return theta

    def pdf(self, u):
        """
        Evaluate the estimated density on the scaled domain u ∈ [–1, 1]:
          f(u) = exp(Σ_j θ_j T_j(u)) / Z.
        u can be a scalar or a NumPy array.
        """
        T_u = np.polynomial.chebyshev.chebvander(u, self.order - 1)
        exponent = T_u.dot(self.theta)
        f_unnormalized = np.exp(exponent)
        Z, _, _ = self.potential_integrals(self.theta)
        return f_unnormalized / Z

    def cdf(self, u_grid):
        """
        Compute the cumulative distribution function (CDF) over a sorted grid u_grid ⊆ [–1, 1]
        using cumulative trapezoidal integration.
        """
        pdf_vals = self.pdf(u_grid)
        # Approximate CDF using np.cumsum with gradient (trapz rule).
        cdf_vals = np.cumsum(pdf_vals * np.gradient(u_grid))
        cdf_vals /= cdf_vals[-1]
        return cdf_vals

    def estimate_quantile(self, phi, grid_points=1000):
        """
        Estimate the quantile corresponding to probability φ in the original domain.
        This is done by computing the CDF on a fine grid of u values, inverting it,
        and then mapping back to x via:
            x = ((u + 1) / 2) * (xmax - xmin) + xmin.
        """
        u_grid = np.linspace(-1, 1, grid_points)
        cdf_vals = self.cdf(u_grid)
        idx = np.searchsorted(cdf_vals, phi)
        u_quant = u_grid[idx] if idx < len(u_grid) else u_grid[-1]
        # Map from u ∈ [–1, 1] back to x ∈ [xmin, xmax]
        x_quant = ((u_quant + 1) / 2) * (self.xmax - self.xmin) + self.xmin
        return x_quant



# %%

# ─────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────

# In production, the target moments (in the Chebyshev basis) would be computed from your data.
# For a given dataset and using the scaling function s(x) = (2*(x - xmin)/(xmax - xmin)) - 1,
# one can compute the Chebyshev moments as: μ_i = (1/n) ∑_x T_i(s(x)).
# For demonstration purposes, we provide a hypothetical set of target moments.
order = 6
# For example, for a normalized density, we set:
# degree 0: 1 (normalization); other moments are placeholders.
target_cheb_moments = [np.int64(10), np.int64(1000), np.int64(100914), np.int64(10279618), np.int64(1057208262), np.int64(109785241810)]

xmin, xmax = 0, 10
solver = ProductionChebyshevMomentSolver(target_cheb_moments, xmin, xmax, order, quad_points=100)

# Solve for the optimal parameters θ.
theta_est = solver.solve_theta()
if solver.converged:
    print("Newton's method converged.")
else:
    print("Newton's method did not converge within the maximum iterations.")

# Estimate, for example, the 95th percentile in the original x-domain.
q_95 = solver.estimate_quantile(0.95)
print(f"Estimated 95th percentile: {q_95:.4f}")

# %%
np.random.seed(42)
N = 10000
# Sample from an exponential distribution with scale=1 (mean=1)
samples = np.random.normal(size=N)

# For a robust finite support, choose:
xmin = np.min(samples) - 1
# Use the 99.9th percentile as xmax to avoid extreme outliers
xmax = np.max(samples) + 1
print(f"Using support: [{xmin}, {xmax:.4f}]")

# Scale the samples to the Chebyshev domain: u = 2*(x - xmin)/(xmax - xmin) - 1
u_samples = 2 * (samples - xmin) / (xmax - xmin) - 1
# Choose the order (number of moments); for example, order=5
order = 5
# Compute target Chebyshev moments from the samples:
# For each degree i, target_moment[i] = average of T_i(u_samples)
T_matrix = chebvander(u_samples, order - 1)  # shape: (N, order)
target_cheb_moments = np.mean(T_matrix, axis=0)
print("Target Chebyshev moments:", target_cheb_moments)

# Instantiate the solver with the target moments and domain [xmin, xmax]
solver = ProductionChebyshevMomentSolver(target_cheb_moments, xmin, xmax, order, quad_points=100)

# Solve for the maximum entropy parameters
theta_est = solver.solve_theta()
if solver.converged:
    print("Newton's method converged.")
else:
    print("Newton's method did not converge within the maximum iterations.")

# Estimate the 95th percentile using our solver
estimated_q95 = solver.estimate_quantile(0.95)
print(f"Estimated 95th percentile (via max-ent): {estimated_q95:.4f}")

# Compare with the empirical 95th percentile from the samples
empirical_q95 = np.quantile(samples, 0.95)
print(f"Empirical 95th percentile: {empirical_q95:.4f}")

# And compare with the theoretical 95th percentile for an Exponential(1):
theoretical_q95 = -np.log(1 - 0.95)  # = -log(0.05)
print(f"Theoretical 95th percentile: {theoretical_q95:.4f}")

# Optionally, plot the estimated PDF and CDF for visualization.
x_grid = np.linspace(xmin, xmax, 1000)
# Map x_grid to u_grid:
u_grid = 2 * (x_grid - xmin) / (xmax - xmin) - 1
pdf_est = solver.pdf(u_grid)
cdf_est = solver.cdf(u_grid)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_grid, pdf_est, label="Estimated PDF")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Maximum Entropy PDF")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_grid, cdf_est, label="Estimated CDF")
plt.axhline(0.95, color="red", linestyle="--", label="phi = 0.95")
plt.xlabel("x")
plt.ylabel("CDF")
plt.title("Maximum Entropy CDF")
plt.legend()

plt.tight_layout()
plt.show()
