/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include "pwu_kernel_parameter.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_custom_device.h"

namespace RPU {

template <typename T> struct UpdateFunctorCustom {

  __device__ __forceinline__ void operator()(
      T &w,
      uint32_t n,
      uint32_t negative,
      const param4_t par_4,
      const param2_t par_2,
      T &par_1,
      const T *global_par,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    // note that only w and par_1 will be written back when used. Thus it can be a "hidden_weights"
    // type note that we here assume that stoch_value is < 1, or if larger, then it did not hit the
    // bound.
    UNUSED(global_params_count);
    UNUSED(global_par);
    UNUSED(par_1);
    UNUSED(par_2);

    T dw = (negative > 0) ? ((T)par_4.w) : (-(T)par_4.y);
    T wmax = (T)par_4.z;
    T wmin = (T)par_4.x;
    T sigma = noise_std_dw;
    // n is larger 0 in any case
    if (n == 1) {
      if (sigma > (T)0.0) {
        T stoch_value = (T)curand_normal(&local_state);
        stoch_value *= sigma;
        w += dw * ((T)1.0 + stoch_value);
      } else {
        w += dw;
      }
    } else {
      if (sigma > (T)0.0) {
        T stoch_value = (T)curand_normal(&local_state);
        stoch_value *= sigma;
        w += dw * (T)n * ((T)1.0 + rsqrt((T)n) * stoch_value); // rsqrt(x) = 1/sqrt(x) is faster
      } else {
        w += dw * (T)n;
      }
    }

    // better always check both bounds
    w = (w > wmax) ? wmax : w;
    w = (w < wmin) ? wmin : w;
  }
};


template <typename T> struct UpdateFunctorCustomLargeNoise {

  __device__ __forceinline__ void operator()(
      T &w,
      uint32_t n,
      uint32_t negative,
      const param4_t par_4,
      const param2_t par_2,
      T &par_1,
      const T *global_par,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(global_params_count);
    UNUSED(global_par);
    UNUSED(par_1);
    UNUSED(par_2);
    // negative > 0 means going up here ...
    // here we assume that noise_std_dw>0 at least
    T wmax = par_4.z;                                   // [2];
    T wmin = par_4.x;                                   //[0];
    float dw = (negative > 0) ? (par_4.w) : (-par_4.y); // [3], [1]
    float sigma = noise_std_dw;

    // n is larger 0 in any case
    if (n == 1) { // short-cut without loop
      float stoch_value = curand_normal(&local_state);
      stoch_value *= sigma;
      w += dw * ((float)1.0 + stoch_value);

      w = (w > wmax) ? wmax : w;
      w = (w < wmin) ? wmin : w;

    } else {
      for (int i = 0; i < n; i++) { // need to loop here because noise can be large and hit the
                                    // boundary and retract again because of sign reverse
        float stoch_value = curand_normal(&local_state);
        stoch_value *= sigma;
        w += dw * ((float)1.0 + stoch_value);

        w = (w > wmax) ? wmax : w;
        w = (w < wmin) ? wmin : w;
      }
    }
  }
};

#define ARGS(NAME)                                                                                 \
  (this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,           \
   getPar().getName() + #NAME)

template <typename T>
pwukpvec_t<T> CustomRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  pwukpvec_t<T> v;

  if (getPar().dw_min_std > (T)0.33) { // 3 sigma
    v.push_back(
        RPU::make_unique<
            PWUKernelParameterSingleFunctor<T, UpdateFunctorCustomLargeNoise<T>, 1>>
            ARGS(FunctorLargeNoise));
    v.push_back(
        RPU::make_unique<
            PWUKernelParameterBatchFunctor<T, UpdateFunctorCustomLargeNoise<T>, 1>>
            ARGS(FunctorLargeNoise));
    v.push_back(
        RPU::make_unique<
            PWUKernelParameterBatchSharedFunctor<T, UpdateFunctorCustomLargeNoise<T>, 1>>
            ARGS(FunctorLargeNoise));
    v.push_back(
        RPU::make_unique<PWUKernelParameterBatchSharedWeightOutputFunctor<
            T, UpdateFunctorCustomLargeNoise<T>, 1>> ARGS(FunctorLargeNoise));

  } else {
    // use summing approximation is save in this case
    // Update functor and kernels are in pwu_kernels.h
    v.push_back(
        RPU::make_unique<PWUKernelParameterBatchSharedFunctor<T, UpdateFunctorCustom<T>, 1>>
            ARGS(Functor));
    v.push_back(
        RPU::make_unique<
            PWUKernelParameterBatchSharedWeightOutputFunctor<T, UpdateFunctorCustom<T>, 1>>
            ARGS(Functor));
    v.push_back(
        RPU::make_unique<PWUKernelParameterBatchFunctor<T, UpdateFunctorCustom<T>, 1>> ARGS(
            Functor));

    v.push_back(
        RPU::make_unique<PWUKernelParameterSingleFunctor<T, UpdateFunctorCustom<T>, 1>> ARGS(
            Functor));
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedSum<T>> ARGS(Sum));
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSharedSumBoundCheck<T>> ARGS(SumBC));

    v.push_back(RPU::make_unique<PWUKernelParameterBatchSum<T>> ARGS(Sum));
    v.push_back(RPU::make_unique<PWUKernelParameterBatchSumBoundCheck<T>> ARGS(SumBC));
  }

  return v;
}

#undef ARGS

template class CustomRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class CustomRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class CustomRPUDeviceCuda<half_t>;
#endif

} // namespace RPU
