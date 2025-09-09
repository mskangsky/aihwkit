/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

// Includes for the random number generator, the base pulsed device, and utility functions.
#include "rng.h"
#include "rpu_pulsed_device.h"
#include "utility_functions.h"

// The RPU namespace encapsulates all the classes and functions for the RPU simulator.
namespace RPU {

// Forward declaration of the CustomRPUDevice class.
// This is necessary because the BUILD_PULSED_DEVICE_META_PARAMETER macro uses the class name.
template <typename T> class CustomRPUDevice;

// This macro generates the meta-parameter class for the CustomRPUDevice.
// It defines the CustomRPUDeviceMetaParameter class, which holds the parameters for the device.
BUILD_PULSED_DEVICE_META_PARAMETER(
    Custom, // The name of the device.
    /*implements*/
    DeviceUpdateType::ConstantStep, // The type of update this device implements.
    /*parameter def*/
    , // Additional parameter definitions can be added here.
    /*print body*/
    , // Additional print statements for the parameters can be added here.
    /* calc weight granularity body */
    return this->dw_min; // The body of the calcWeightGranularity function.
    ,
    /*add*/
);

/**
 * @brief Custom Resistive Processing Unit (RPU) device.
 *
 * This class implements a custom RPU device. It inherits from the PulsedRPUDevice class
 * and can be used as a template for creating new devices with custom behavior.
 *
 * @tparam T The data type of the weights (e.g., float, double).
 */
template <typename T> class CustomRPUDevice : public PulsedRPUDevice<T> {

  // This macro generates the constructors, destructor, and other boilerplate code for the device.
  BUILD_PULSED_DEVICE_CONSTRUCTORS(
      CustomRPUDevice, // The name of the device class.
      /* ctor*/
      , // Additional code for the constructor can be added here.
      /* dtor*/
      , // Additional code for the destructor can be added here.
      /* copy */
      , // Additional code for the copy constructor can be added here.
      /* move assignment */
      , // Additional code for the move assignment operator can be added here.
      /* swap*/
      , // Additional code for the swap function can be added here.
      /* dp names*/
      , // Additional device parameter names can be added here.
      /* dp2vec body*/
      , // The body of the device parameters to vector function can be added here.
      /* vec2dp body*/
      , // The body of the vector to device parameters function can be added here.
      /*invert copy DP */
  );

  /**
   * @brief Performs a sparse update of the weights.
   *
   * This method is called when the update is sparse, i.e., only a subset of the weights are updated.
   *
   * @param weights A pointer to the weights of the device.
   * @param i The index of the weight to update.
   * @param x_signed_indices A pointer to the signed indices of the input vector.
   * @param x_count The number of non-zero elements in the input vector.
   * @param d_sign The sign of the error.
   * @param rng A pointer to the random number generator.
   */
  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;

  /**
   * @brief Performs a dense update of the weights.
   *
   * This method is called when the update is dense, i.e., all the weights are updated.
   *
   * @param weights A pointer to the weights of the device.
   * @param coincidences A pointer to the coincidences vector.
   * @param rng A pointer to the random number generator.
   */
  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;
};
} // namespace RPU
