#ifndef __GLOBAL_H__
#define __GLOBAL_H__

// C++ STL includes
#include <iostream>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <map>
#include <unordered_map>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <limits>

#define X_DIM 4
#define Z_DIM 2
#define INF std::numeric_limits<float>::infinity()
#define EPS std::numeric_limits<float>::epsilon()
// #define DEBUG
#define NUM_PARTICLES 512 // TODO: make this a multiple of WG size automatically
#define WG_THREADS_PER_BLOCK 256
#define NUM_ELEMS_PER_BLOCK (2 * WG_THREADS_PER_BLOCK)
#define MAX_TARGETS 50

#endif
