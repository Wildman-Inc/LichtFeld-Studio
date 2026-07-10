/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file hipsparse/hipsparse.h
 * @brief Shim for hipsparse.h that ensures types are properly defined.
 *
 * This shim handles the issue where:
 * 1. ROCm 7.10's hipsparse-types.h uses #if(!defined(CUDART_VERSION)) to define hipsparseStatus_t
 * 2. If CUDART_VERSION is defined (by PyTorch headers), hipsparseStatus_t won't be defined
 * 3. This causes compilation errors in hipsparse-auxiliary.h
 *
 * Solution: Temporarily undefine CUDART_VERSION when including hipsparse headers.
 */

#ifndef _LFS_HIPSPARSE_SHIM_H_
#define _LFS_HIPSPARSE_SHIM_H_

// Save and temporarily undefine CUDART_VERSION
#pragma push_macro("CUDART_VERSION")
#ifdef CUDART_VERSION
#undef CUDART_VERSION
#endif

// Include the actual hipsparse header
#include_next <hipsparse/hipsparse.h>

// Restore CUDART_VERSION
#pragma pop_macro("CUDART_VERSION")

#endif // _LFS_HIPSPARSE_SHIM_H_
