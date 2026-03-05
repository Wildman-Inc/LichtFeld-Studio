/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIPCachingAllocator.h shim for Windows compatibility
 * 
 * On Windows, rpcndr.h defines `small` as `char`, which conflicts with
 * PyTorch's HIPCachingAllocator.h parameter names. This shim undefines
 * the macro before including the real header.
 */
#pragma once

// Windows rpcndr.h defines: #define small char
// This conflicts with parameter names in PyTorch's HIPCachingAllocator.h
#ifdef _WIN32
#pragma push_macro("small")
#undef small
#endif

#include_next <c10/hip/HIPCachingAllocator.h>

#ifdef _WIN32
#pragma pop_macro("small")
#endif
