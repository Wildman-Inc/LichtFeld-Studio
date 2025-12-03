/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file hip_rand_compat.h
 * @brief cuRAND/hipRAND compatibility layer.
 *
 * This header provides unified random number generation API for both
 * CUDA (cuRAND) and HIP (hipRAND/rocRAND) backends.
 */

#ifndef LFS_HIP_RAND_COMPAT_H
#define LFS_HIP_RAND_COMPAT_H

#include "kernels/hip_runtime_compat.h"

#if LFS_USE_HIP

#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

// =============================================================================
// Device-side random state and functions (for use in kernels)
// =============================================================================

// State types - hipRAND uses different state types
using curandState = hiprandState;
using curandState_t = hiprandState_t;
using curandStateXORWOW = hiprandStateXORWOW;
using curandStateXORWOW_t = hiprandStateXORWOW_t;
using curandStatePhilox4_32_10 = hiprandStatePhilox4_32_10;
using curandStatePhilox4_32_10_t = hiprandStatePhilox4_32_10_t;
using curandStateMRG32k3a = hiprandStateMRG32k3a;
using curandStateMRG32k3a_t = hiprandStateMRG32k3a_t;
using curandStateMtgp32 = hiprandStateMtgp32;
using curandStateMtgp32_t = hiprandStateMtgp32_t;
using curandStateSobol32 = hiprandStateSobol32;
using curandStateSobol32_t = hiprandStateSobol32_t;
using curandStateScrambledSobol32 = hiprandStateScrambledSobol32;
using curandStateScrambledSobol32_t = hiprandStateScrambledSobol32_t;
using curandStateSobol64 = hiprandStateSobol64;
using curandStateSobol64_t = hiprandStateSobol64_t;
using curandStateScrambledSobol64 = hiprandStateScrambledSobol64;
using curandStateScrambledSobol64_t = hiprandStateScrambledSobol64_t;

// Device-side init and generation functions
#define curand_init hiprand_init
#define curand hiprand
#define curand_uniform hiprand_uniform
#define curand_uniform_double hiprand_uniform_double
#define curand_uniform2_double hiprand_uniform2_double
#define curand_uniform4 hiprand_uniform4
#define curand_normal hiprand_normal
#define curand_normal_double hiprand_normal_double
#define curand_normal2 hiprand_normal2
#define curand_normal2_double hiprand_normal2_double
#define curand_normal4 hiprand_normal4
#define curand_log_normal hiprand_log_normal
#define curand_log_normal_double hiprand_log_normal_double
#define curand_log_normal2 hiprand_log_normal2
#define curand_log_normal2_double hiprand_log_normal2_double
#define curand_log_normal4 hiprand_log_normal4
#define curand_poisson hiprand_poisson
#define curand_poisson4 hiprand_poisson4
#define curand_discrete hiprand_discrete
#define curand_discrete4 hiprand_discrete4

// =============================================================================
// Host-side generator API
// =============================================================================

// Generator type
using curandGenerator_t = hiprandGenerator_t;
using curandStatus_t = hiprandStatus_t;
using curandRngType_t = hiprandRngType_t;

// Status codes
constexpr auto CURAND_STATUS_SUCCESS = HIPRAND_STATUS_SUCCESS;
constexpr auto CURAND_STATUS_VERSION_MISMATCH = HIPRAND_STATUS_VERSION_MISMATCH;
constexpr auto CURAND_STATUS_NOT_INITIALIZED = HIPRAND_STATUS_NOT_INITIALIZED;
constexpr auto CURAND_STATUS_ALLOCATION_FAILED = HIPRAND_STATUS_ALLOCATION_FAILED;
constexpr auto CURAND_STATUS_TYPE_ERROR = HIPRAND_STATUS_TYPE_ERROR;
constexpr auto CURAND_STATUS_OUT_OF_RANGE = HIPRAND_STATUS_OUT_OF_RANGE;
constexpr auto CURAND_STATUS_LENGTH_NOT_MULTIPLE = HIPRAND_STATUS_LENGTH_NOT_MULTIPLE;
constexpr auto CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED;
constexpr auto CURAND_STATUS_LAUNCH_FAILURE = HIPRAND_STATUS_LAUNCH_FAILURE;
constexpr auto CURAND_STATUS_INTERNAL_ERROR = HIPRAND_STATUS_INTERNAL_ERROR;

// RNG types
constexpr auto CURAND_RNG_PSEUDO_DEFAULT = HIPRAND_RNG_PSEUDO_DEFAULT;
constexpr auto CURAND_RNG_PSEUDO_XORWOW = HIPRAND_RNG_PSEUDO_XORWOW;
constexpr auto CURAND_RNG_PSEUDO_MRG32K3A = HIPRAND_RNG_PSEUDO_MRG32K3A;
constexpr auto CURAND_RNG_PSEUDO_MTGP32 = HIPRAND_RNG_PSEUDO_MTGP32;
constexpr auto CURAND_RNG_PSEUDO_MT19937 = HIPRAND_RNG_PSEUDO_MT19937;
constexpr auto CURAND_RNG_PSEUDO_PHILOX4_32_10 = HIPRAND_RNG_PSEUDO_PHILOX4_32_10;
constexpr auto CURAND_RNG_QUASI_DEFAULT = HIPRAND_RNG_QUASI_DEFAULT;
constexpr auto CURAND_RNG_QUASI_SOBOL32 = HIPRAND_RNG_QUASI_SOBOL32;
constexpr auto CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
constexpr auto CURAND_RNG_QUASI_SOBOL64 = HIPRAND_RNG_QUASI_SOBOL64;
constexpr auto CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64;

// Generator functions
#define curandCreateGenerator hiprandCreateGenerator
#define curandCreateGeneratorHost hiprandCreateGeneratorHost
#define curandDestroyGenerator hiprandDestroyGenerator
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandSetStream hiprandSetStream
#define curandGetVersion hiprandGetVersion

// Generation functions
#define curandGenerate hiprandGenerate
#define curandGenerateLongLong hiprandGenerateLongLong
#define curandGenerateUniform hiprandGenerateUniform
#define curandGenerateUniformDouble hiprandGenerateUniformDouble
#define curandGenerateNormal hiprandGenerateNormal
#define curandGenerateNormalDouble hiprandGenerateNormalDouble
#define curandGenerateLogNormal hiprandGenerateLogNormal
#define curandGenerateLogNormalDouble hiprandGenerateLogNormalDouble
#define curandGeneratePoisson hiprandGeneratePoisson

// Helper function to get error string
inline const char* curandGetStatusString(curandStatus_t status) {
    switch (status) {
        case HIPRAND_STATUS_SUCCESS: return "HIPRAND_STATUS_SUCCESS";
        case HIPRAND_STATUS_VERSION_MISMATCH: return "HIPRAND_STATUS_VERSION_MISMATCH";
        case HIPRAND_STATUS_NOT_INITIALIZED: return "HIPRAND_STATUS_NOT_INITIALIZED";
        case HIPRAND_STATUS_ALLOCATION_FAILED: return "HIPRAND_STATUS_ALLOCATION_FAILED";
        case HIPRAND_STATUS_TYPE_ERROR: return "HIPRAND_STATUS_TYPE_ERROR";
        case HIPRAND_STATUS_OUT_OF_RANGE: return "HIPRAND_STATUS_OUT_OF_RANGE";
        case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE: return "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE";
        case HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case HIPRAND_STATUS_LAUNCH_FAILURE: return "HIPRAND_STATUS_LAUNCH_FAILURE";
        case HIPRAND_STATUS_INTERNAL_ERROR: return "HIPRAND_STATUS_INTERNAL_ERROR";
        default: return "Unknown hipRAND error";
    }
}

#else // CUDA backend

#include <curand.h>
#include <curand_kernel.h>

#endif // LFS_USE_HIP

// Common error checking macro
#define RAND_CHECK(call)                                                       \
    do {                                                                       \
        curandStatus_t status = (call);                                        \
        if (status != CURAND_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "RAND error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__,                                        \
                    curandGetStatusString(status));                            \
        }                                                                      \
    } while (0)

#endif // LFS_HIP_RAND_COMPAT_H
