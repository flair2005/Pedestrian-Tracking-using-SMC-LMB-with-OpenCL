#include "lmbTracker.hpp"
#include "shortestPaths.hpp"
#include "bestAssignments.hpp"

std::string readFile(std::string src_file_loc) {
  std::ifstream t{src_file_loc.c_str()};
  if (!t.good()) {
    std::cout << "kernel source file not found\n";
    assert(0);
  }
  std::string src_str{std::istreambuf_iterator<char>(t),
                      std::istreambuf_iterator<char>()};
  return src_str;
}

inline float lmbTracker::computePs(const arma::fvec &x) { return ps; }

inline float lmbTracker::computePd(const arma::fvec &x) {
  assert(x.size());
  return pd;
}

inline float lmbTracker::clutterDensity() { return (lambda_c / scan_volume); }

arma::fvec lmbTracker::esfMahler(const arma::fvec &x) {
  if (!x.size()) {
    return arma::fvec({1});
  }

  arma::fmat F = arma::zeros<arma::fmat>(2, x.size());
  size_t idx = 0;
  for (int n = 1; n <= (int)x.size(); n++) {
    F(idx, 0) = F(!idx, 0) + x(n - 1);
    for (int k = 2; k <= n; k++) {
      if (k <= n) {
        if (k == n) {
          F(idx, k - 1) = x(n - 1) * F(!idx, k - 2);
        } else {
          F(idx, k - 1) = F(!idx, k - 1) + x(n - 1) * F(!idx, k - 2);
        }
      }
    }
    idx = !idx;
  }

  return (arma::join_vert(arma::fvec({1}), arma::trans(F.row(!idx))));
}

lmbTracker::lmbTracker()
    : program(readFile("../src/updateKernels.cl")), init_kern_ftor_ptr(NULL),
      pre_update_kern_ftor_ptr(NULL), pre_update1_kern_ftor_ptr(NULL),
      update_kern_ftor_ptr(NULL), scan_kern_ftor_ptr(NULL),
      update1_kern_ftor_ptr(NULL) {
  initFilter();
}

lmbTracker::~lmbTracker() {
  delete init_kern_ftor_ptr;
  delete pre_update_kern_ftor_ptr;
  delete pre_update1_kern_ftor_ptr;
  delete update_kern_ftor_ptr;
  delete scan_kern_ftor_ptr;
  delete update1_kern_ftor_ptr;
}

void lmbTracker::initFilter() {
  current_time_step = 0;
  delta = 1;
  birth_r_max = 0.25;
  birth_V_max = 60;
  track_r_max = 0.5;
  track_r_min = 0.1;
  track_merge_U = 30;
  num_particles = (unsigned)NUM_PARTICLES;
  birth_Hmax = 50;
  survive_Hmax = 200;
  update_Hmax = 200;
  pd = 0.95;
  ps = 0.98;
  lambda_c = 4;
  scan_volume = (2000 * 2000);

  arma::fmat A0 = {{1, delta}, {0, 1}};
  F = arma::join_vert(arma::join_horiz(A0, arma::zeros<arma::fmat>(2, 2)),
                      arma::join_horiz(arma::zeros<arma::fmat>(2, 2), A0));
  sigma_v = 10;
  H = {{1, 0, 0, 0}, {0, 0, 1, 0}};
  arma::fmat D = arma::diagmat(arma::fvec({10, 10}));
  R = D * D.t();

  P._lmb.clear();
  P._dglmb.clear();
  U._lmb.clear();
  X.clear();
  Zk.clear();
  zk_r.clear();

  // opencl stuff
  // setting up OpenCL-2.0 default platform
  cl::Platform::get(&platforms);
  cl::Platform plat;
  for (auto &p : platforms) {
    std::string plat_ver = p.getInfo<CL_PLATFORM_VERSION>();
    if (plat_ver.find("OpenCL 2.") != std::string::npos) {
      plat = p;
      if (cl::Platform::setDefault(plat) != plat) {
        std::cout << "Error: Couldnt setup the default OpenCL platform\n";
        assert(0);
      }
      break;
    }
  }
  if (!plat()) {
    std::cout << "Error: No OpenCL 2.0 platform found\n";
    assert(0);
  }

  // building the OpenCL source code
  std::string program_build_flags =
      "-cl-std=CL2.0 -I../../clRNG_package/include/";
  try {
    program.build(program_build_flags.c_str());
  } catch (...) {
    err_status = CL_SUCCESS;
    auto build_info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&err_status);
    for (auto &pair : build_info) {
      std::cout << pair.second << std::endl << std::endl;
    }
    exit(EXIT_FAILURE);
  }

  // Initializing OpenCL environment
  // i) setting up random streams on GPU aka device so as to be able to generate
  //    uniform random numbers directly there
  size_t stream_buf_size;
  unsigned max_targets_expected = (unsigned)MAX_TARGETS;
  clrngMrg31k3pStream *streams =
      clrngMrg31k3pCreateStreams(NULL, max_targets_expected * num_particles,
                                 &stream_buf_size, (clrngStatus *)&err_status);
  if (err_status != CL_SUCCESS) {
    std::cout << "Error: Couldnt set up random streams using clRNG library\n";
  }
  cl::Buffer stream_buf_dev(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            stream_buf_size, streams);

  cmd_q = cl::CommandQueue(cl::Context::getDefault(), cl::Device::getDefault(),
                           CL_QUEUE_PROFILING_ENABLE);
  dev_cmd_q = cl::DeviceCommandQueue::makeDefault(cl::Context::getDefault(),
                                                  cl::Device::getDefault());
  init_kern_ftor_ptr =
      new cl::KernelFunctor<cl::Buffer &>(program, "initKernel");
  pre_update_kern_ftor_ptr =
      new cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl_uint, cl_uint,
                            cl::Buffer &, cl::Buffer &, cl::Buffer &>(
          program, "preUpdate");
  pre_update1_kern_ftor_ptr =
      new cl::KernelFunctor<cl_uint, cl::Buffer &, cl::Buffer &>(program,
                                                                 "preUpdate1");
  update_kern_ftor_ptr = new cl::KernelFunctor<
      cl_uint, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &,
      cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &>(program,
                                                              "updateKernel");
  scan_kern_ftor_ptr =
      new cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &,
                            cl::Buffer &, cl::Buffer &, cl::Buffer &>(
          program, "multiScanTop");
  update1_kern_ftor_ptr = new cl::KernelFunctor<
      cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &,
      cl::Buffer &, cl::Buffer &>(program, "updateKernel1");

  kernel_perf_event = (*init_kern_ftor_ptr)(
      cl::EnqueueArgs(cmd_q, cl::NDRange(max_targets_expected * num_particles)),
      stream_buf_dev);
  kernel_perf_event.wait();
}

std::vector<std::pair<lmbTracker::label_t, arma::fvec>>
lmbTracker::runFilter(const std::vector<arma::fvec> &_Zk) {
  current_time_step++;

  Zk_minus_1 = Zk;
  Zk = _Zk;
  zk_r.clear();
  zk_r.resize(Zk.size());  

  Update();
  
  Predict();
  
  pruneUpdate();  
  estimateState();

  return X;
}

// UPDATE
void lmbTracker::dglmbUpdate() {
  if (P._lmb.size() == 0) {
    // for no P._lmb component: no update_lmb_tmp is created -> no U._lmb so returning from here 
    return;
  }
  
  std::vector<float> P_avpd(P._lmb.size());  
  std::vector<float> allcostm(P._lmb.size() * Zk.size());
  std::vector<lmb3_dev> update_lmb_tmp((Zk.size() + 1) * P._lmb.size());     
  cl::Buffer update_lmb_tmp_dev(
      CL_MEM_READ_WRITE, sizeof(lmb3_dev) * (P._lmb.size() * (1 + Zk.size())));      
  std::vector<dglmb> update_dglmb_tmp;  
  
  if (Zk.size()) {
    // for miss-detections, we form update components as is
    for (size_t l = 0; l < P._lmb.size(); ++l) {
      for (size_t n = 0; n < NUM_PARTICLES; ++n) {
        update_lmb_tmp[l].w[n] = P._lmb[l].w[n];
      }
    }
    
    // for detections
    std::vector<float> Zk2;
    for (auto z : Zk) {
      for (size_t i = 0; i < Z_DIM; ++i) {
        Zk2.emplace_back(z(i));
      }
    }
    
    // going to gpu for finding update_lmb_tmp, P_avpd, allcostm  
    cl::Buffer Zk_dev(Zk2.begin(), Zk2.end(), true);
    // passing 0s as P_avpd, allcostm initially to GPU
    cl::Buffer P_avpd_dev(P_avpd.begin(), P_avpd.end(), false);
    cl::Buffer allcostm_dev(allcostm.begin(), allcostm.end(), false);
    
    kernel_perf_event = (*pre_update_kern_ftor_ptr)(
        cl::EnqueueArgs(cmd_q,
                        cl::NDRange(P._lmb.size() * Zk.size() * NUM_PARTICLES),
                        cl::NDRange(256)),
        Zk_dev, P_lmb_dev, Zk.size(), P._lmb.size(), update_lmb_tmp_dev,
        P_avpd_dev, allcostm_dev);
    kernel_perf_event.wait();
  
    // normalization
    kernel_perf_event = (*pre_update1_kern_ftor_ptr)(
        cl::EnqueueArgs(cmd_q,
                        cl::NDRange(P._lmb.size() * Zk.size() * NUM_PARTICLES),
                        cl::NDRange(256)),
        P._lmb.size(), allcostm_dev, update_lmb_tmp_dev);
    kernel_perf_event.wait();  

    cl::copy(P_avpd_dev, P_avpd.begin(), P_avpd.end());
    cl::copy(allcostm_dev, allcostm.begin(), allcostm.end());
    

    // update_dglmb_tmp
    float P_dglmb_sqrt_w_sum = 0;
    for (auto &l : P._dglmb) {
      P_dglmb_sqrt_w_sum += sqrt(l.w);
    }

    for (auto &l : P._dglmb) {
      if (!l.I.size()) {
        dglmb tmp;
        tmp.w =
            -lambda_c + Zk.size() * log(lambda_c * clutterDensity()) + log(l.w);
        update_dglmb_tmp.emplace_back(tmp);
      } else {
        arma::fmat CM(l.I.size(), Zk.size());
        for (size_t i = 0; i < l.I.size(); i++) {
          for (size_t j = 0; j < Zk.size(); j++) {
            CM(i, j) =
                -log(allcostm[j * P._lmb.size() + l.I[i]] /
                     (lambda_c * clutterDensity() * (1 - P_avpd[l.I[i]])));
          }
        }

        std::vector<std::pair<std::vector<int>, float>> assigns;
        unsigned m = round(update_Hmax * sqrt(l.w) / P_dglmb_sqrt_w_sum);
        murtyKBestAssignmentWrapper(CM, m, assigns);

        for (size_t i = 0; i < assigns.size(); i++) {
          dglmb tmp;
          float log_sum = 0;
          for (size_t ii = 0; ii < l.I.size(); ii++) {
            tmp.I.emplace_back(l.I[ii] +
                               P._lmb.size() * (assigns[i].first[ii] + 1));
            log_sum += log(1 - P_avpd[l.I[ii]]);
          }
          tmp.w = -lambda_c + Zk.size() * log(lambda_c * clutterDensity()) +
                  log_sum + log(l.w) - assigns[i].second;
          update_dglmb_tmp.emplace_back(tmp);

          for (auto &ii : assigns[i].first) {
            if (ii >= 0) {
              zk_r[ii] += exp(tmp.w);
            }
          }
        }
      }
    } 
  } else {          // Zk.size() == 0
    // P_avpd
    for (size_t l = 0; l < P._lmb.size(); l++) {
      float avpd_sum = 0;
      for (size_t n = 0; n < num_particles; n++) {
        avpd_sum += computePd(P._lmb[l].x[n]) * P._lmb[l].w[n];
      }
      P_avpd[l] = avpd_sum;
    }
        
    // update_lmb_tmp    
    for (size_t l = 0; l < P._lmb.size(); ++l) {
      for (size_t n = 0; n < NUM_PARTICLES; ++n) {
        update_lmb_tmp[l].w[n] = P._lmb[l].w[n];
      }
    }    
    
    // update_dglmb_tmp
    for (auto &l : P._dglmb) {
      dglmb tmp;
      float log_sum = 0;
      for (auto &i : l.I) {
        log_sum += log(1 - P_avpd[i]);
      }
      tmp.w = -lambda_c * log_sum * log(l.w);
      tmp.I = l.I;
      update_dglmb_tmp.emplace_back(tmp);
    }    
  }
  // update_dglmb_tmp normalization of weights/probabilities
  // using high precision doubles here to avoid missing out on very small data
  double sumexp_w = 0;
  for (auto &d : update_dglmb_tmp) {
    sumexp_w += exp((double)d.w);
  }

  for (auto &d : update_dglmb_tmp) {
    d.w = exp(d.w - (float)log(sumexp_w));
  }
  for (auto &r : zk_r) {
    r = (float)((double)r / sumexp_w);
  }   





  // going to gpu to find U._lmb 
  cl::copy(update_lmb_tmp.begin(), update_lmb_tmp.begin() + P._lmb.size(),
           update_lmb_tmp_dev);  
  
  std::vector<cl_uint> update_dglmb_tmp_i;
  std::vector<cl_float> update_dglmb_tmp_w;
  std::vector<cl_uint> offsets(P._lmb.size() + 1);
  std::vector<cl_uint> offsets_map(P._lmb.size());
  for (auto &d : update_dglmb_tmp) {
    for (auto &i : d.I) {
      update_dglmb_tmp_i.emplace_back(i);
      update_dglmb_tmp_w.emplace_back(d.w);
      offsets[(i % P._lmb.size()) + 1]++;
    }
  }
  unsigned tmp = 0;
  for (size_t o = 1; o < offsets.size(); ++o) {
    if (offsets[o] != 0) {
      offsets_map[o - 1] = tmp++;
    }
  }

  offsets.erase(std::remove_if(offsets.begin() + 1, offsets.end(),
                               [](const uint &o) { return (o == 0); }),
                offsets.end());

  std::vector<cl_uint> sums_offsets(offsets.size());
  std::vector<cl_uint> events_offsets(offsets.size());
  for (size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] += offsets[i - 1];
    unsigned blocks = (offsets[i] - offsets[i - 1]) * num_particles;
    if (blocks == 0) {
      std::cout << "Error: offsets here not set properly\n";
      assert(0);
    }

    events_offsets[i] =
        events_offsets[i - 1] +
        (unsigned)ceil(log(blocks) / (double)log(NUM_ELEMS_PER_BLOCK));
    sums_offsets[i] = sums_offsets[i - 1];
    while (blocks != 1) {
      blocks = (unsigned)ceil(blocks / (double)NUM_ELEMS_PER_BLOCK);
      sums_offsets[i] += blocks;
    }
  }

  std::vector<lmb2_dev> U_lmb2(offsets.size() - 1);
  std::vector<cl_uint> thread_atomic_counts(P._lmb.size());
  std::vector<cl_float> debug(1);

  cl::Buffer update_dglmb_tmp_w_dev(update_dglmb_tmp_w.begin(),
                                    update_dglmb_tmp_w.end(), true);
  cl::Buffer update_dglmb_tmp_i_dev(update_dglmb_tmp_i.begin(),
                                    update_dglmb_tmp_i.end(), true);
  cl::Buffer offsets_map_dev(offsets_map.begin(), offsets_map.end(), true);
  cl::Buffer offsets_dev(offsets.begin(), offsets.end(), true);
  cl::Buffer sums_offsets_dev(sums_offsets.begin(), sums_offsets.end(), true);
  cl::Buffer events_offsets_dev(events_offsets.begin(), events_offsets.end(),
                                true);
  cl::Buffer thread_atomic_counts_dev(thread_atomic_counts.begin(),
                                      thread_atomic_counts.end(), false);
  cl::Buffer cum_w_dev(CL_MEM_READ_WRITE, sizeof(cl_float) *
                                              offsets[U_lmb2.size()] *
                                              num_particles);
  cl::Buffer sums_dev(CL_MEM_READ_WRITE,
                      sizeof(cl_float) * sums_offsets[U_lmb2.size()]);
  cl::Buffer events_dev(CL_MEM_READ_WRITE,
                        sizeof(cl_event) * events_offsets[U_lmb2.size()]);
  cl::Buffer U_lmb_dev(CL_MEM_READ_WRITE, sizeof(lmb2_dev) * U_lmb2.size());
  cl::Buffer debug_dev(debug.begin(), debug.end(), false);

  kernel_perf_event = (*update_kern_ftor_ptr)(
      cl::EnqueueArgs(cmd_q, cl::NDRange(offsets[U_lmb2.size()])),
      P._lmb.size(), update_lmb_tmp_dev, update_dglmb_tmp_w_dev,
      update_dglmb_tmp_i_dev, offsets_dev, offsets_map_dev,
      thread_atomic_counts_dev, U_lmb_dev, cum_w_dev);
  kernel_perf_event.wait();

  kernel_perf_event = (*scan_kern_ftor_ptr)(
      cl::EnqueueArgs(cmd_q, cl::NDRange(U_lmb2.size())), cum_w_dev,
      offsets_dev, sums_dev, sums_offsets_dev, events_dev, events_offsets_dev);
  kernel_perf_event.wait();

  kernel_perf_event = (*update1_kern_ftor_ptr)(
      cl::EnqueueArgs(cmd_q, cl::NDRange(num_particles * U_lmb2.size())),
      P_lmb_dev, cum_w_dev, offsets_dev, sums_dev, sums_offsets_dev, U_lmb_dev,
      debug_dev);
  kernel_perf_event.wait();

  cl::copy(debug_dev, debug.begin(), debug.end());
  if (debug[0] != 0) {
    std::cout << "Error: Infinite while loops in GPU\n";
    assert(0);
  }

  cl::copy(U_lmb_dev, U_lmb2.begin(), U_lmb2.end());
  for (auto &l : U_lmb2) {
    if (l.r >= 1) {
      l.r = 1 - 0.000001;
    }
    lmb tmp;
    tmp.r = l.r;
    tmp.l = P._lmb[l.l].l;
    tmp.w.resize(num_particles);
    tmp.x.resize(num_particles);
    for (size_t n = 0; n < num_particles; ++n) {
      tmp.w[n] = 1 / (float)num_particles;
      tmp.x[n] = arma::zeros<arma::fvec>(X_DIM);
      for (size_t j = 0; j < X_DIM; ++j) {
        tmp.x[n](j) = l.x[n * X_DIM + j];
      }
    }
    U._lmb.emplace_back(tmp);
  }
}
#if 0
void lmbTracker::dglmb2lmb(std::vector<lmb> *lmb_tmp,
                           std::vector<dglmb> *dglmb_tmp) {
  std::map<label_t, size_t> unique_labels_map;
  for (auto &l : *lmb_tmp) {
    if (unique_labels_map.find(l.l) == unique_labels_map.end()) {
      unique_labels_map[l.l] = U._lmb.size();
      lmb tmp;
      tmp.l = l.l;
      U._lmb.emplace_back(tmp);
    }
  }

  for (auto &d : *dglmb_tmp) {
    for (auto &i : d.I) {
      for (size_t n = 0; n < num_particles; n++) {
        U._lmb[unique_labels_map[(*lmb_tmp)[i].l]].x.emplace_back(
            (*lmb_tmp)[i].x[n]);
        U._lmb[unique_labels_map[(*lmb_tmp)[i].l]].w.emplace_back(
            d.w * (*lmb_tmp)[i].w[n]);
      }
    }
  }
  // weight normalization
  for (auto &l : U._lmb) {
    l.r = 0;
    for (auto &v : l.w) {
      l.r += v;
    }
    for (auto &v : l.w) {
      v /= l.r;
    }

    // TODO: figure out why l.r >= 1..shouldnt happen!!
    //       maybe b/c of float precision issues!
    if (l.r >= 1) {
      l.r = (1 - EPS);
    }
  }

  // removing components that have not been assigned by any hypothesis
  U._lmb.erase(std::remove_if(U._lmb.begin(), U._lmb.end(),
                              [](const lmb &c) { return (c.r == 0); }),
               U._lmb.end());
}
#endif

void lmbTracker::Update() {
  U._lmb.clear();
  dglmbUpdate();
}

// PREDICTION
void lmbTracker::Predict() {
  P._lmb.clear();
  P._dglmb.clear();

  std::vector<lmb> P_birth_lmb;
  unsigned target_cnt = 0;
  for (size_t r = 0; r < zk_r.size(); r++) {
    if ((1 - zk_r[r]) > track_r_min) {
      float min_speed = INF;
      size_t min_speed_idx = 0;
      for (size_t z = 0; z < Zk_minus_1.size(); z++) {
        float this_speed =
            arma::norm(arma::fvec({(Zk[r](0) - Zk_minus_1[z](0)) / delta,
                                   (Zk[r](1) - Zk_minus_1[z](1)) / delta}));
        if (this_speed < min_speed) {
          min_speed = this_speed;
          min_speed_idx = z;
        }
      }

      // TODO: could try to filter targets having ~0 min_speed as this usually
      //      mean clutter
      //          if (min_speed < 0.1) {
      //
      //          } else
      if (min_speed < birth_V_max) {
        lmb tmp;
        tmp.r = (1 - zk_r[r]);
        if (tmp.r > birth_r_max) {
          tmp.r = birth_r_max;
        }
        tmp.l.first = current_time_step;
        tmp.l.second = target_cnt++;
        float x_speed = (Zk[r](0) - Zk_minus_1[min_speed_idx](0)) / delta;
        float y_speed = (Zk[r](1) - Zk_minus_1[min_speed_idx](1)) / delta;
        arma::fvec m = arma::fvec({Zk[r](0) + x_speed * delta, x_speed,
                                   Zk[r](1) + y_speed * delta, y_speed});
        tmp.x.resize(num_particles);
        tmp.w.resize(num_particles);
        for (size_t n = 0; n < num_particles; n++) {
          tmp.x[n] = newStatePredict(m);
          tmp.w[n] = 1 / (float)(num_particles);
        }
        P_birth_lmb.emplace_back(tmp);
      }
    }
  }
  P._lmb = P_birth_lmb;

  // TODO: parallelize P_survive_lmb on GPU via OpenCL
  std::vector<lmb> P_survive_lmb;
  for (auto &l : U._lmb) {
    if (l.r >= track_r_min) {
      lmb tmp;
      tmp.l = l.l;
      tmp.x.resize(num_particles);
      tmp.w.resize(num_particles);
      float sum_w = 0;
      for (size_t n = 0; n < num_particles; n++) {
        tmp.x[n] = newStatePredict(l.x[n]);
        tmp.w[n] = computePs(l.x[n]) * l.w[n];
        sum_w += tmp.w[n];
      }
      tmp.r = sum_w * l.r;
      for (size_t n = 0; n < num_particles; n++) {
        tmp.w[n] /= sum_w;
      }
      P_survive_lmb.emplace_back(tmp);
    }
  }
  P._lmb.insert(P._lmb.end(), P_survive_lmb.begin(), P_survive_lmb.end());

  std::vector<dglmb> P_birth_dglmb;
  lmb2dglmb(&P_birth_dglmb, &P_birth_lmb, birth_Hmax);

  std::vector<dglmb> P_survive_dglmb;
  lmb2dglmb(&P_survive_dglmb, &P_survive_lmb, survive_Hmax);

  float sum_w = 0;
  for (auto &b : P_birth_dglmb) {
    for (auto &s : P_survive_dglmb) {
      dglmb tmp;
      tmp.w = b.w * s.w;
      sum_w += tmp.w;
      tmp.I = b.I;
      for (auto &i : s.I) {
        tmp.I.emplace_back(i + P_birth_lmb.size());
      }
      P._dglmb.emplace_back(tmp);
    }
  }
  // weight normalization
  for (auto &d : P._dglmb) {
    d.w /= sum_w;
  }

  if (P._lmb.size()) {
    std::vector<lmb1_dev> P_lmb2(P._lmb.size());
    for (size_t l = 0; l < P._lmb.size(); ++l) {
      P_lmb2[l].l[0] = P._lmb[l].l.first;
      P_lmb2[l].l[1] = P._lmb[l].l.second;
      for (size_t n = 0; n < NUM_PARTICLES; ++n) {
        P_lmb2[l].w[n] = P._lmb[l].w[n];
        for (size_t j = 0; j < X_DIM; ++j) {
          P_lmb2[l].x[n * X_DIM + j] = P._lmb[l].x[n](j);
        }
      }
    }
    P_lmb_dev = cl::Buffer(P_lmb2.begin(), P_lmb2.end(), true);
  }
}

arma::fvec lmbTracker::newStatePredict(arma::fvec old_x) {
  // via Linear Gaussian Model for now!
  // TODO: maybe use a non-linear model later for better tracking accuracy
  arma::fvec new_x = F * old_x + sigma_v * arma::randn<arma::fvec>(X_DIM);
  return new_x;
}

void lmbTracker::lmb2dglmb(std::vector<dglmb> *_dglmb, std::vector<lmb> *_lmb,
                           unsigned k) {
  assert(k > 0);
  assert(_dglmb->size() == 0);

  std::vector<float> costv;
  for (auto &l : *_lmb) {
    costv.emplace_back(-log(l.r / (1 - l.r)));
  }
  std::vector<size_t> costv_sorted_idx(costv.size());
  iota(costv_sorted_idx.begin(), costv_sorted_idx.end(), 0);
  std::sort(costv_sorted_idx.begin(), costv_sorted_idx.end(),
            [&costv](size_t i1, size_t i2) { return (costv[i1] > costv[i2]); });

  std::vector<Edge> E;
  for (unsigned i = 0; i < costv.size() + 1; i++) {
    for (unsigned j = i + 1; j < costv.size() + 1; j++) {
      E.emplace_back(i, j, costv[costv_sorted_idx[j - 1]]);
    }
    E.emplace_back(i, costv.size() + 1, EPS);
  }
  std::vector<std::pair<std::vector<uint>, float>> shortest_paths;
  // YenKShortestPaths(E, costv.size() + 2, 0, costv.size() + 1, k,
  //                   shortest_paths);
  EppsteinKShortestPaths(E, costv.size() + 2, 0, costv.size() + 1, k,
                         shortest_paths);
  for (auto &sp : shortest_paths) {
    dglmb tmp;
    tmp.w = sp.second;
    for (auto &i : sp.first) {
      if (i != 0 && i != (costv.size() + 1)) {
        tmp.I.emplace_back(costv_sorted_idx[i - 1]);
      }
    }
    _dglmb->emplace_back(tmp);
  }

  // weight normalization
  float sum_w = 0;
  for (auto &l : *_lmb) {
    sum_w += log(1 - l.r);
  }
  float sum_exp_w = 0;
  for (auto &d : *_dglmb) {
    d.w = sum_w - d.w;
    sum_exp_w += exp(d.w);
  }
  for (auto &d : *_dglmb) {
    d.w = exp(d.w - log(sum_exp_w));
  }
}

void lmbTracker::pruneUpdate() {
  // merging close-by targets
  std::vector<lmb> lmb_new;
  float track_r_max_local = track_r_max;
  std::sort(U._lmb.begin(), U._lmb.end(),
            [](const lmb &lhs, const lmb &rhs) { return (lhs.r > rhs.r); });
  for (int i = 0; i < (int)(U._lmb.size() - 1); i++) {
    if (U._lmb[i].r < 0) {
      continue;
    }
    arma::fvec this_x = arma::zeros<arma::fvec>(X_DIM);
    for (size_t n = 0; n < num_particles; n++) {
      this_x += U._lmb[i].w[n] * U._lmb[i].x[n];
    }
    for (size_t j = i + 1; j < U._lmb.size(); j++) {
      if (U._lmb[j].r >= 0) {
        arma::fvec that_x = arma::zeros<arma::fvec>(X_DIM);
        for (size_t n = 0; n < num_particles; n++) {
          that_x += U._lmb[j].w[n] * U._lmb[j].x[n];
        }
        float dist = arma::norm(arma::fvec({this_x(0), this_x(2)}) -
                                arma::fvec({that_x(0), that_x(2)}));
        if (dist < track_merge_U) {
          lmb tmp;
          tmp.r = U._lmb[i].r + U._lmb[j].r;
          if (U._lmb[i].l.first < U._lmb[j].l.first) {
            tmp.l = U._lmb[i].l;
          } else {
            tmp.l = U._lmb[j].l;
          }
          tmp.x.resize(num_particles);
          tmp.w.resize(num_particles);
          float sum_w = 0;
          for (size_t n = 0; n < num_particles; n++) {
            tmp.w[n] = (U._lmb[i].r / tmp.r) * U._lmb[i].w[n] +
                       (U._lmb[j].r / tmp.r) * U._lmb[j].w[n];
            sum_w += tmp.w[n];
            tmp.x[n] = (U._lmb[i].r / tmp.r) * U._lmb[i].x[n] +
                       (U._lmb[j].r / tmp.r) * U._lmb[j].x[n];
          }
          for (size_t n = 0; n < num_particles; n++) {
            tmp.w[n] /= sum_w;
          }
          if (tmp.r >= 1) {
            tmp.r = 1 - 0.000001;
          }
          lmb_new.emplace_back(tmp);

          U._lmb[i].r = -1;
          U._lmb[j].r = -1;
          // TODO: right now merging is done only upto 2 components so as to
          // save computational costs...see if this is enough or else
          // merging of all possible components in *_tentative is needed???
          break;
        }
      }
    }
  }
  U._lmb.erase(std::remove_if(U._lmb.begin(), U._lmb.end(),
                              [track_r_max_local](const lmb &l) {
                 return (l.r == -1 || l.r < track_r_max_local);
               }),
               U._lmb.end());

  // ii) TODO: cap on max number of tracks

  U._lmb.insert(U._lmb.end(), lmb_new.begin(), lmb_new.end());
}

void lmbTracker::estimateState() {
  X.clear();
  arma::fvec r_over_1minusr(U._lmb.size());
  float prod_r = 1;
  for (size_t i = 0; i < U._lmb.size(); i++) {
    r_over_1minusr(i) = U._lmb[i].r / (1 - U._lmb[i].r);
    prod_r *= (1 - U._lmb[i].r);
  }
  arma::fvec cdn = prod_r * esfMahler(r_over_1minusr);
  
  unsigned num_targets = 0;
  float max_val = 0;
  for (size_t i = 0; i < cdn.size(); i++) {
    if (max_val < cdn[i]) {
      max_val = cdn[i];
      num_targets = i;
    }
  }
  std::sort(U._lmb.begin(), U._lmb.end(),
            [](const lmb &lhs, const lmb &rhs) { return (lhs.r > rhs.r); });
  for (size_t i = 0; i < num_targets; i++) {
    std::pair<label_t, arma::fvec> x;
    x.first = U._lmb[i].l;
    x.second = arma::zeros<arma::fvec>(X_DIM);
    for (size_t n = 0; n < num_particles; n++) {
      x.second += U._lmb[i].x[n] * U._lmb[i].w[n];
    }
    X.emplace_back(x);
  }
}

// TODO: suitable assertions to check possible runtime error conditions!!

// TODO: add random seeds for filtering operation if not debugging!!

// TODO: documentation
