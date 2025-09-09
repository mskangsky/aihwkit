/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include "rpu_custom_device.h"
#include "utility_functions.h"
#include <iostream>
// #include <random>
#include <chrono>
#include <cmath>
#include <limits>

namespace RPU {

/********************************************************************************
 * Custom RPU Device
 *********************************************************************************/

/**
 * @brief Populates the device with the given parameters.
 *
 * @tparam T The data type of the weights.
 * @param p The meta-parameter for the device.
 * @param rng A pointer to the random number generator.
 */
template <typename T>
void CustomRPUDevice<T>::populate(
    const CustomRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {
  // Call the populate function of the base class.
  PulsedRPUDevice<T>::populate(p, rng);
}

/**
 * @brief Performs a sparse update of the weights.
 *
 * This method is called when the update is sparse, i.e., only a subset of the weights are updated.
 *
 * @tparam T The data type of the weights.
 * @param weights A pointer to the weights of the device.
 * @param i The index of the weight to update.
 * @param x_signed_indices A pointer to the signed indices of the input vector.
 * @param x_count The number of non-zero elements in the input vector.
 * @param d_sign The sign of the error.
 * @param rng A pointer to the random number generator.
 */
template <typename T>
void CustomRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  // Get the pointers to the device parameters.
  T *scale_down = this->w_scale_down_[i];
  T *scale_up = this->w_scale_up_[i];
  T *w = weights[i];
  T *min_bound = this->w_min_bound_[i];
  T *max_bound = this->w_max_bound_[i];
  // Get the standard deviation of the weight change.
  T dw_min_std = getPar().dw_min_std;

  // Check if there is any noise.
  if (dw_min_std > (T)0.0) {
    // If there is noise, add it to the weight change.
    PULSED_UPDATE_W_LOOP(
        T dw = 0; if (sign > 0) {
          // Calculate the weight change with noise.
          dw = ((T)1.0 + dw_min_std * rng->sampleGauss()) * scale_down[j];
          // Update the weight.
          w[j] -= dw;
        } else {
          // Calculate the weight change with noise.
          dw = ((T)1.0 + dw_min_std * rng->sampleGauss()) * scale_up[j];
          // Update the weight.
          w[j] += dw;
        }
        // Clip the weight to the bounds.
        w[j] = MIN(w[j], max_bound[j]);
        w[j] = MAX(w[j], min_bound[j]););
  } else {
    // If there is no noise, just update the weight.
    PULSED_UPDATE_W_LOOP(
        if (sign > 0) {
          // Update the weight.
          w[j] -= scale_down[j];
        } else {
          // Update the weight.
          w[j] += scale_up[j];
        }
        // Clip the weight to the bounds.
        w[j] = MIN(w[j], max_bound[j]);
        w[j] = MAX(w[j], min_bound[j]););
  }
}

/**
 * @brief Performs a dense update of the weights.
 *
 * This method is called when the update is dense, i.e., all the weights are updated.
 *
 * @tparam T The data type of the weights.
 * @param weights A pointer to the weights of the device.
 * @param coincidences A pointer to the coincidences vector.
 * @param rng A pointer to the random number generator.
 */
template <typename T>
void CustomRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  // Get the pointers to the device parameters.
  T *scale_down = this->w_scale_down_[0];
  T *scale_up = this->w_scale_up_[0];
  T *w = weights[0];
  T *min_bound = this->w_min_bound_[0];
  T *max_bound = this->w_max_bound_[0];
  // Get the standard deviation of the weight change.
  T dw_min_std = getPar().dw_min_std;

  // This macro loops over all the weights and updates them.
  PULSED_UPDATE_W_LOOP_DENSE(
      // Calculate the weight change with noise.
      T dw = dw_min_std > (T)0.0 ? dw_min_std * rng->sampleGauss() : (T)0.0;
      if (sign > 0) {
        // Calculate the weight change with noise.
        dw = ((T)1.0 + dw) * scale_down[j];
        // Update the weight.
        w[j] -= dw;
      } else {
        // Calculate the weight change with noise.
        dw = ((T)1.0 + dw) * scale_up[j];
        // Update the weight.
        w[j] += dw;
      }
      // Clip the weight to the bounds.
      w[j] = MIN(w[j], max_bound[j]);
      w[j] = MAX(w[j], min_bound[j]);

  );
}

// Instantiate the class for the different data types.
template class CustomRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class CustomRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class CustomRPUDevice<half_t>;
#endif

} // namespace RPU
