// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Give the pages of a copied-out range of a read-only file mapping back to the OS (strata-inkling-2-mini GGUF loader).
//
// The GGUF is mmapped read-only and every tensor is COPIED out of the mapping (trunk arena, VRAM upload buffer, host
// expert arena). The mapped pages stay in the process's resident set after the copy, so a 15.7 GB UD-IQ4_XS sat next to
// its copies: peak RSS 32-36 GiB with 11-13 GiB of experts in RAM. Dropping the pages of a range that is already copied
// removes them from the process (they stay in the OS page cache, reclaimable, and fault back in if the range is read
// again), so nothing is lost but the double count. STRATA_INKLING2MINI_KEEP_MAPPED_PAGES=1 turns it off.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::core {

inline bool keep_mapped_pages() {
    static const bool keep = [] {
        const char* e = std::getenv("STRATA_INKLING2MINI_KEEP_MAPPED_PAGES");
        return e && e[0] == '1';
    }();
    return keep;
}

// The size of a page as the OS reports it (4 KiB on x86-64; larger on some other CPUs), 4096 if it cannot say.
inline uintptr_t mapped_page_size() noexcept {
    static const uintptr_t page = [] {
#ifdef _WIN32
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        const uintptr_t v = static_cast<uintptr_t>(si.dwPageSize);
#else
        const long r = sysconf(_SC_PAGESIZE);
        const uintptr_t v = r > 0 ? static_cast<uintptr_t>(r) : 0;
#endif
        return (v != 0 && (v & (v - 1)) == 0) ? v : uintptr_t{4096};
    }();
    return page;
}

// Only whole pages inside [p, p + n) are released (rounded inward, so a neighbour's page that is still to be read is
// never dropped). Returns the number of bytes handed back (0 when off, too small, or the OS call failed).
inline size_t release_mapped_pages(const void* p, size_t n) noexcept {
    if (!p || n == 0 || keep_mapped_pages()) return 0;
    const uintptr_t kPage = mapped_page_size();
    const uintptr_t b = (reinterpret_cast<uintptr_t>(p) + kPage - 1) & ~(kPage - 1);
    const uintptr_t e = (reinterpret_cast<uintptr_t>(p) + n) & ~(kPage - 1);
    if (e <= b) return 0;
#ifdef _WIN32
    // VirtualUnlock on pages that are not locked trims them from the working set (and reports ERROR_NOT_LOCKED, which
    // is expected here).
    (void)VirtualUnlock(reinterpret_cast<void*>(b), static_cast<SIZE_T>(e - b));
    return static_cast<size_t>(e - b);
#else
    return madvise(reinterpret_cast<void*>(b), static_cast<size_t>(e - b), MADV_DONTNEED) == 0
        ? static_cast<size_t>(e - b) : 0;
#endif
}

}  // namespace strata::core
