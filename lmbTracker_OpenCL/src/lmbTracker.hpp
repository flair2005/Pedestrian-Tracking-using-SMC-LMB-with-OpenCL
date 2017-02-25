#include <opencv2/opencv.hpp>
#include "global.hpp"
#include <armadillo>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include "CL/cl2.hpp"
#include "clRNG/clRNG.h"
#include "clRNG/mrg31k3p.h"

class lmbTracker {
  // sub-structures/classes
public:
  typedef std::pair<unsigned, unsigned> label_t;

private:
  struct lmb {
    label_t l;
    float r;
    std::vector<arma::fvec> x;
    std::vector<float> w;

    void print() {
      std::cout << "r: " << r << std::endl;
      std::cout << "l:<" << l.first << "," << l.second << ">" << std::endl;
      std::cout << "particles:" << std::endl << "------------" << std::endl;
      arma::fvec tmp = arma::zeros<arma::fvec>(X_DIM);
      for (size_t n = 0; n < x.size(); n++) {
        tmp += w[n] * x[n];
      }
      tmp.print("x");
    }
  };
  struct dglmb {
    float w;
    std::vector<unsigned> I;

    void print() {
      std::cout << "w: " << w << "; I:(";
      for (auto &i : I) {
        std::cout << i << ", ";
      }
      std::cout << ")" << std::endl;
    }
  };
  struct lmbDensity {
    std::vector<lmb> _lmb;
    std::vector<dglmb> _dglmb;
  };
  struct lmb1_dev { // P_lmb
    cl_uint l[2];
    cl_float w[NUM_PARTICLES];
    cl_float x[NUM_PARTICLES * X_DIM];
  };
  struct lmb2_dev { // U_lmb
    cl_uint l;
    cl_float r;
    cl_float x[NUM_PARTICLES * X_DIM];
  };
  struct lmb3_dev { // update_lmb_tmp
    cl_float w[NUM_PARTICLES];
  };

  // member variables
public:
private:
  std::vector<arma::fvec> Zk, Zk_minus_1;
  std::vector<std::pair<label_t, arma::fvec>> X;
  // TODO: can we avoid double zk_r??
  std::vector<double> zk_r;
  lmbDensity P, U;

  unsigned current_time_step;
  float delta;
  float birth_r_max;
  float birth_V_max;
  float track_r_max;
  float track_r_min;
  float track_merge_U;
  unsigned num_particles;
  unsigned birth_Hmax, survive_Hmax, update_Hmax;

  float pd, ps;
  float lambda_c;
  float scan_volume;

  arma::fmat F, H, R;
  float sigma_v;

  // opencl stuff
  cl_int err_status;
  std::vector<cl::Platform> platforms;
  cl::Program program;
  cl::CommandQueue cmd_q;
  cl::DeviceCommandQueue dev_cmd_q;
  cl::Event kernel_perf_event;
  cl::KernelFunctor<cl::Buffer &> *init_kern_ftor_ptr;
  cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl_uint, cl_uint, cl::Buffer &,
                    cl::Buffer &, cl::Buffer &> *pre_update_kern_ftor_ptr;
  cl::KernelFunctor<cl_uint, cl::Buffer &, cl::Buffer &> *
      pre_update1_kern_ftor_ptr;
  cl::KernelFunctor<cl_uint, cl::Buffer &, cl::Buffer &, cl::Buffer &,
                    cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &,
                    cl::Buffer &> *update_kern_ftor_ptr;
  cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &,
                    cl::Buffer &, cl::Buffer &> *scan_kern_ftor_ptr;
  cl::KernelFunctor<cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &,
                    cl::Buffer &, cl::Buffer &,
                    cl::Buffer &> *update1_kern_ftor_ptr;
  cl::Buffer P_lmb_dev;
  // member functions
public:
  lmbTracker();
  ~lmbTracker();
  std::vector<std::pair<lmbTracker::label_t, arma::fvec>>
  runFilter(const std::vector<arma::fvec> &_Zk);

private:
  void initFilter();

  void Predict();
  arma::fvec newStatePredict(arma::fvec old_x);
  void lmb2dglmb(std::vector<dglmb> *dglmb, std::vector<lmb> *lmb, unsigned k);

  void dglmbUpdate();
  //void dglmb2lmb(std::vector<lmb> *lmb_tmp, std::vector<dglmb> *dglmb_tmp);
  void pruneUpdate();
  void Update();

  void estimateState();

  float computePs(const arma::fvec &x);
  float computePd(const arma::fvec &x);
  float clutterDensity();
  arma::fvec esfMahler(const arma::fvec &z);
};

// TODO:
// documentation
