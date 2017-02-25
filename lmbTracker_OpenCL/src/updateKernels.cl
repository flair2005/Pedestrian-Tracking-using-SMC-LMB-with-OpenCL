#define CLRNG_SINGLE_PRECISION
#include "clRNG/mrg31k3p.clh"

// TODO: read these defines from global.h
// TODO: avoid MAX_TARGETS if you can!!
#define MAX_TARGETS 50
#define NUM_PARTICLES 512
#define WG_THREADS_PER_BLOCK 256
#define NUM_ELEMS_PER_BLOCK (WG_THREADS_PER_BLOCK * 2)
#define X_DIM 4
#define Z_DIM 2

// TODO: merge these defines with global defines later

typedef struct {
  uint l[2];
  float w[NUM_PARTICLES];
  float x[NUM_PARTICLES * X_DIM];
} lmb1; // P_lmb

typedef struct {
  uint l;
  float r;
  float x[NUM_PARTICLES * X_DIM];
} lmb2; // U_lmb

typedef struct { float w[NUM_PARTICLES]; } lmb3; // update_lmb_tmp

global clrngMrg31k3pStream dev_streams[MAX_TARGETS * NUM_PARTICLES];

kernel void initKernel(global clrngMrg31k3pHostStream *streams) {
  clrngMrg31k3pCopyOverStreamsFromGlobal(1, &dev_streams[get_global_id(0)],
                                         &streams[get_global_id(0)]);
}

inline void atomicFloatAdd(global float *addr, float val) {
  union {
    uint u32;
    float f32;
  } next, expected, current;
  current.f32 = *addr;
  do {
    expected.f32 = current.f32;
    next.f32 = expected.f32 + val;
    current.u32 = atomic_cmpxchg((global uint *)addr, expected.u32, next.u32);
  } while (current.u32 != expected.u32);
}

inline void atomicFloatAddLocal(local float *addr, float val) {
  union {
    uint u32;
    float f32;
  } next, expected, current;
  current.f32 = *addr;
  do {
    expected.f32 = current.f32;
    next.f32 = expected.f32 + val;
    current.u32 = atomic_cmpxchg((local uint *)addr, expected.u32, next.u32);
  } while (current.u32 != expected.u32);
}

inline float computePd(float *x) {
  // TODO: later return the parametrized pd from filter parameters!!
  //  return pd;
  return 0.95;
}

// TODO: right now most of the stuff here is hard-coded based on fixed model
// params...later make it flexible
inline float computeLikelihood(float *x, float *z) {
  //  arma::fmat D = arma::diagmat(arma::fvec({10,10}));
  //    arma::fmat R = D*D.t();
  //    arma::fmat H = {{1,0,0,0}, {0,0,1,0}};

  //    arma::fvec numer = -0.5 * (z-H*x).t() * R.i() * (z-H*x);
  //    return (exp(numer(0)) / sqrt(pow(2*M_PI, z.size()) * arma::det(R)));

  float a = z[0] - x[0];
  float b = z[1] - x[2];

  // return (exp(-0.005*(a*a + b*b)) / sqrt(pow(2*M_PI, Z_DIM) * 10000));
  return (exp(-0.005 * (a * a + b * b)) / 628.3185);
}

kernel void preUpdate(global float *Zk, global lmb1 *P_lmb, uint Zk_len,
                      uint P_lmb_len, global lmb3 *update_lmb_tmp,
                      global float *P_avpd, global float *allcostm) {
  uint this_z = get_global_id(0) / (NUM_PARTICLES * P_lmb_len);
  uint this_l = (get_global_id(0) / NUM_PARTICLES) % P_lmb_len;
  uint this_idx = get_global_id(0) % NUM_PARTICLES;

  float w = (float)P_lmb[this_l].w[this_idx];
  float x[X_DIM];
  for (size_t i = 0; i < X_DIM; ++i) {
    x[i] = (float)P_lmb[this_l].x[this_idx * X_DIM + i];
  }
  float z[Z_DIM];
  for (size_t i = 0; i < Z_DIM; ++i) {
    z[i] = (float)Zk[this_z * Z_DIM + i];
  }

  local float this_wg_allcostm;
  local float this_wg_P_avpd;
  if (get_local_id(0) == 0) {
    this_wg_allcostm = 0;
    if (this_z == 0) {
      this_wg_P_avpd = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float tmp, tmp2;
  float tmp3;
  tmp = computePd(x) * w;
  tmp2 = tmp * computeLikelihood(x, z);

  if (this_z == 0) {
    atomicFloatAddLocal(&this_wg_P_avpd, tmp);
  }
  atomicFloatAddLocal(&this_wg_allcostm, tmp2);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_local_id(0) == 0) {
    if (this_z == 0) {
      atomicFloatAdd(&P_avpd[this_l], this_wg_P_avpd);
    }
    atomicFloatAdd(&allcostm[this_z * P_lmb_len + this_l], this_wg_allcostm);
  }
  update_lmb_tmp[P_lmb_len * (1 + this_z) + this_l].w[this_idx] = tmp2;
}

kernel void preUpdate1(uint P_lmb_len, global float *allcostm,
                       global lmb3 *update_lmb_tmp) {
  uint this_z = get_global_id(0) / (NUM_PARTICLES * P_lmb_len);
  uint this_l = (get_global_id(0) / NUM_PARTICLES) % P_lmb_len;
  uint this_idx = get_global_id(0) % NUM_PARTICLES;

  float tmp = allcostm[this_z * P_lmb_len + this_l];
  if (tmp) {
    update_lmb_tmp[P_lmb_len * (1 + this_z) + this_l].w[this_idx] /= tmp;
  }
}

kernel void updateKernel(uint P_lmb_len, global lmb3 *update_lmb_tmp,
                         global float *update_dglmb_tmp_w,
                         global uint *update_dglmb_tmp_i, global uint *offsets,
                         global uint *offsets_map, global atomic_uint *counts,

                         global lmb2 *U_lmb, global float *cum_w) {
  uint this_lmb_id = update_dglmb_tmp_i[get_global_id(0)];
  float this_lmb_update_w = update_dglmb_tmp_w[get_global_id(0)];
  uint U_lmb_idx = (this_lmb_id % P_lmb_len);
  uint this_idx =
      offsets[offsets_map[U_lmb_idx]] +
      atomic_fetch_add_explicit(&counts[U_lmb_idx], 1, memory_order_relaxed,
                                memory_scope_device);
  if (this_idx == offsets[offsets_map[U_lmb_idx]]) {
    U_lmb[offsets_map[U_lmb_idx]].l = U_lmb_idx;
  }

  enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT,
                 ndrange_1D(NUM_PARTICLES), ^{
      cum_w[this_idx * NUM_PARTICLES + get_global_id(0)] =
          update_lmb_tmp[this_lmb_id].w[get_global_id(0)] * this_lmb_update_w;
  });
}

kernel void scan(global float *w, global float *sums, uint N) {
  local float this_w[NUM_ELEMS_PER_BLOCK];
  if (get_local_id(0) == 0) {
    for (uint i = 0; i < NUM_ELEMS_PER_BLOCK; ++i) {
      this_w[i] = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint dp = 1;

  this_w[2 * lid] = w[2 * gid];
  if (2 * gid + 1 < N) {
    this_w[2 * lid + 1] = w[2 * gid + 1];
  }

  for (uint s = NUM_ELEMS_PER_BLOCK >> 1; s > 0; s >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < s) {
      uint i = dp * (2 * lid + 1) - 1;
      uint j = dp * (2 * lid + 2) - 1;
      this_w[j] += this_w[i];
    }
    dp <<= 1;
  }

  if (lid == 0) {
    sums[get_group_id(0)] = this_w[NUM_ELEMS_PER_BLOCK - 1];
    this_w[NUM_ELEMS_PER_BLOCK - 1] = 0;
  }

  for (uint s = 1; s < NUM_ELEMS_PER_BLOCK; s <<= 1) {
    dp >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < s) {
      uint i = dp * (2 * lid + 1) - 1;
      uint j = dp * (2 * lid + 2) - 1;
      float t = this_w[j];
      this_w[j] += this_w[i];
      this_w[i] = t;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  w[2 * gid] = this_w[2 * lid];
  if (2 * gid + 1 < N) {
    w[2 * gid + 1] = this_w[2 * lid + 1];
  }
}

kernel void gatherTop(global float *w, global float *sums,
                      global clk_event_t *events, uint N, uint sums_N, uint K) {
  if (get_global_id(0) == 0 && sums_N > 1) {
    float B = (float)NUM_ELEMS_PER_BLOCK;

    uint tail = sums_N - 1;
    uint head = tail - ceil(N / pow(B, K));

    for (uint i = K; i > 1; --i) {
      tail = head;
      head = tail - ceil(N / pow(B, i - 1));

      enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT,
                     ndrange_1D(tail - head), 1, &events[i], &events[i - 1], ^{
          sums[head + get_global_id(0)] +=
              sums[tail + get_global_id(0) / (uint)B];
      });
      release_event(events[i]);
    }

    // this is the final kernel launch which puts increments from sums to w
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT,
                   ndrange_1D(N), 1, &events[1], &events[0], ^{
        w[get_global_id(0)] += sums[get_global_id(0) / (uint)B];
    });
    release_event(events[1]);
    release_event(events[0]);
  }
}

kernel void scanTop(global float *w, global float *sums,
                    global clk_event_t *events, uint N) {
  // scatter

  // i) first launching scan kernel on w
  enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT,
                 ndrange_1D((N / 2 + N % 2), WG_THREADS_PER_BLOCK), 0, NULL,
                 &events[0], ^{ scan(w, sums, N); });

  // ii) now doing scans on sums solely
  uint head = 0;
  uint tail = (uint)ceil(N / (float)(NUM_ELEMS_PER_BLOCK));
  uint iterations = 0;
  while (1) {
    uint this_N = (tail - head);
    if (this_N == 1) {
      break;
    }
    iterations++;

    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT,
                   ndrange_1D((this_N / 2 + this_N % 2), WG_THREADS_PER_BLOCK),
                   1, &events[iterations - 1], &events[iterations],
                   ^{ scan(&sums[head], &sums[tail], this_N); });
    release_event(events[iterations - 1]);
    head = tail;
    tail += (uint)ceil(this_N / (float)(NUM_ELEMS_PER_BLOCK));
  }

  // gather
  enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(1),
                 1, &events[iterations], NULL,
                 ^{ gatherTop(w, sums, events, N, tail, iterations); });
  release_event(events[iterations]);
}

kernel void multiScanTop(global float *w, global uint *offsets,
                         global float *sums, global uint *sums_offsets,
                         global clk_event_t *events,
                         global uint *events_offsets) {
  uint w_off = offsets[get_global_id(0)] * NUM_PARTICLES;
  uint sums_off = sums_offsets[get_global_id(0)];
  uint events_off = events_offsets[get_global_id(0)];
  uint N = (offsets[get_global_id(0) + 1] - offsets[get_global_id(0)]) *
           NUM_PARTICLES;

  enqueue_kernel(
      get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(1),
      ^{ scanTop(&w[w_off], &sums[sums_off], &events[events_off], N); });
}

kernel void updateKernel1(global lmb1 *P_lmb, global float *cum_w,
                          global uint *offsets, global float *sums,
                          global uint *sums_offsets,

                          global lmb2 *U_lmb, global float *debug) {
  uint this_U_lmb_id = get_global_id(0) / NUM_PARTICLES;
  uint this_idx = get_global_id(0) % NUM_PARTICLES;
  uint this_P_lmb = U_lmb[this_U_lmb_id].l;

  if (this_idx == 0) {
    U_lmb[this_U_lmb_id].r = sums[sums_offsets[this_U_lmb_id + 1] - 1];
  }

  uint start_tmp = offsets[this_U_lmb_id];
  uint end_tmp = offsets[this_U_lmb_id + 1];
  uint tmp;

  float rand_v = clrngMrg31k3pRandomU01(&dev_streams[get_global_id(0)]) *
                 cum_w[end_tmp * NUM_PARTICLES - 1];

  // corner-case
  if (rand_v > cum_w[end_tmp * NUM_PARTICLES - 1]) {
    for (uint j = 0; j < X_DIM; ++j) {
      U_lmb[this_U_lmb_id].x[this_idx * X_DIM + j] =
          P_lmb[this_P_lmb].x[(NUM_PARTICLES - 1) * X_DIM + j];
    }
    return;
  }

  uint debug_cnt = 0; // to avoid infinite while loops in GPU
  while (1) {
    debug_cnt++;
    tmp = (end_tmp - start_tmp) / 2 + start_tmp;

    if (cum_w[tmp * NUM_PARTICLES] <= rand_v &&
        rand_v <= cum_w[(tmp + 1) * NUM_PARTICLES - 1]) {
      // required particle is in here
      // TODO: maybe do binary search here as well if you find this too slow!!
      //       else its lot simpler to do brute force search
      for (uint i = 0; i < NUM_PARTICLES; ++i) {
        if (rand_v <= cum_w[tmp * NUM_PARTICLES + i]) {
          for (uint j = 0; j < X_DIM; ++j) {
            U_lmb[this_U_lmb_id].x[this_idx * X_DIM + j] =
                P_lmb[this_P_lmb].x[(i - 1) * X_DIM + j];
          }
          return;
        }
      }
    } else {
      if (rand_v < cum_w[tmp * NUM_PARTICLES]) {
        if (rand_v > cum_w[tmp * NUM_PARTICLES - 1]) {
          // corner case
          for (uint j = 0; j < X_DIM; ++j) {
            U_lmb[this_U_lmb_id].x[this_idx * X_DIM + j] =
                P_lmb[this_P_lmb].x[j];
          }
          return;
        } else {
          end_tmp = tmp;
        }
      } else {
        if (rand_v < cum_w[(tmp + 1) * NUM_PARTICLES]) {
          // corner case
          for (uint j = 0; j < X_DIM; ++j) {
            U_lmb[this_U_lmb_id].x[this_idx * X_DIM + j] =
                P_lmb[this_P_lmb].x[j];
          }
          return;
        } else {
          start_tmp = tmp;
        }
      }
    }

    if (debug_cnt >= 10000) {
      *debug = 1;
      return;
    }
  }
}
