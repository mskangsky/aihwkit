/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

#include "pwu_kernel_parameter_base.h"
#include "rpu_custom_device.h"
#include "rpucuda_pulsed_device.h"
#include <memory>

namespace RPU {

template <typename T> class CustomRPUDeviceCuda : public PulsedRPUDeviceCuda<T> {

public:
  BUILD_PULSED_DEVICE_CONSTRUCTORS_CUDA(
      CustomRPUDeviceCuda,
      CustomRPUDevice,
      /*ctor body*/
      ,
      /*dtor body*/
      ,
      /*copy body*/
      ,
      /*move assigment body*/
      ,
      /*swap body*/
      ,
      /*host copy from cpu (rpu_device). Parent device params are copyied automatically*/
  )

  pwukpvec_t<T> getUpdateKernels(
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans,
      const PulsedUpdateMetaParameter<T> &up) override;
};

} // namespace RPU
