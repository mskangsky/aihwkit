/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

// Includes for the base pulsed weight updater kernel parameter, the custom RPU device, and the base CUDA pulsed device.
#include "pwu_kernel_parameter_base.h"
#include "rpu_custom_device.h"
#include "rpucuda_pulsed_device.h"
#include <memory>

// The RPU namespace encapsulates all the classes and functions for the RPU simulator.
namespace RPU {

/**
 * @brief Custom Resistive Processing Unit (RPU) device for CUDA.
 *
 * This class implements a custom RPU device for CUDA. It inherits from the PulsedRPUDeviceCuda class
 * and can be used as a template for creating new devices with custom behavior.
 *
 * @tparam T The data type of the weights (e.g., float, double).
 */
template <typename T> class CustomRPUDeviceCuda : public PulsedRPUDeviceCuda<T> {

public:
  // This macro generates the constructors, destructor, and other boilerplate code for the CUDA device.
  BUILD_PULSED_DEVICE_CONSTRUCTORS_CUDA(
      CustomRPUDeviceCuda, // The name of the CUDA device class.
      CustomRPUDevice,     // The name of the corresponding CPU device class.
      /*ctor body*/
      , // Additional code for the constructor can be added here.
      /*dtor body*/
      , // Additional code for the destructor can be added here.
      /*copy body*/
      , // Additional code for the copy constructor can be added here.
      /*move assigment body*/
      , // Additional code for the move assignment operator can be added here.
      /*swap body*/
      , // Additional code for the swap function can be added here.
      /*host copy from cpu (rpu_device). Parent device params are copyied automatically*/
  )

  /**
   * @brief Gets the update kernels for the device.
   *
   * This method returns a vector of update kernels that are used to update the weights of the device.
   *
   * @param m_batch The size of the mini-batch.
   * @param nK32 The number of K32 blocks.
   * @param use_bo64 Whether to use 64-bit operations.
   * @param out_trans Whether to transpose the output.
   * @param up The pulsed update meta-parameter.
   * @return A vector of update kernels.
   */
  pwukpvec_t<T> getUpdateKernels(
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans,
      const PulsedUpdateMetaParameter<T> &up) override;
};

} // namespace RPU
