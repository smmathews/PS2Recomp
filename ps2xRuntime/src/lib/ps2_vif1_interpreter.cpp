// Based on Blackline Interactive implementation
#include "runtime/ps2_memory.h"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <unordered_map>
#include "ps2_log.h"

// Defined in ps2_gs_gpu.cpp: one process-wide monotonic epoch shared by every
// DQ8 probe, so timestamps taken in different TUs are directly comparable.
uint64_t dq8ProbeNowMs();

enum VIFCmd : uint8_t
{
    VIF_NOP = 0x00,
    VIF_STCYCL = 0x01,
    VIF_OFFSET = 0x02,
    VIF_BASE = 0x03,
    VIF_ITOP = 0x04,
    VIF_STMOD = 0x05,
    VIF_MSKPATH3 = 0x06,
    VIF_MARK = 0x07,
    VIF_FLUSHE = 0x10,
    VIF_FLUSH = 0x11,
    VIF_FLUSHA = 0x13,
    VIF_MSCAL = 0x14,
    VIF_MSCALF = 0x15,
    VIF_MSCNT = 0x17,
    VIF_STMASK = 0x20,
    VIF_STROW = 0x30,
    VIF_STCOL = 0x31,
    VIF_MPG = 0x4A,
    VIF_DIRECT = 0x50,
    VIF_DIRECTHL = 0x51,
};

namespace
{
    // Local twin of ps2DiagEnvLimit/ps2DiagLogBudget (ps2_runtime.h) for TUs
    // that do not include that header. A capped probe that drops events
    // silently is worse than no probe: the truncated tail reads as "the event
    // never happened again". See ps2_runtime.h for the canonical version.
    uint32_t localDiagEnvLimit(const char *envName, uint32_t fallback)
    {
        const char *raw = std::getenv(envName);
        if (!raw || raw[0] == '\0')
        {
            return fallback;
        }
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(raw, &end, 0);
        if (end == raw)
        {
            return fallback;
        }
        return static_cast<uint32_t>(parsed);
    }

    bool localDiagLogBudget(std::ostream &os, const char *tag, const char *envName,
                            uint32_t limit, uint32_t logIndex,
                            std::atomic<bool> &truncationAnnounced)
    {
        if (limit == 0u || logIndex < limit)
        {
            return true;
        }
        bool expected = false;
        if (truncationAnnounced.compare_exchange_strong(expected, true, std::memory_order_relaxed))
        {
            os << tag << " TRUNCATED after " << limit
               << " events; further events are NOT logged (raise or disable with "
               << envName << "=<n>, 0 = unlimited)" << std::endl;
        }
        return false;
    }

    std::atomic<uint32_t> s_debugVu1KickCount{0};
    std::atomic<uint32_t> s_debugVif1OpcodeCount{0};
    constexpr uint8_t kGifFmtImage = 2u;

    // ---- VIF1 opcode trace (PS2X_VIF1_TRACE=1) --------------------------
    // The pre-existing [vif1:cmd]/[vif1:mscal]/[vif1:mscnt] probes below use
    // RUNTIME_LOG, which expands to a no-op unless the runtime is compiled
    // with _DEBUG. In a release build they can never print, so "zero MSCAL in
    // the boot log" was never evidence that no MSCAL ran -- it was evidence
    // that the probe was compiled out. These counters are always maintained
    // (cheap relaxed atomics) and are dumped only when the trace is enabled,
    // so the MSCAL/MPG question can be answered from a normal release build.
    //
    // VALUE-tested gate, never getenv()!=nullptr: this tree has a documented
    // history of presence-gate bugs.
    bool vif1TraceOn()
    {
        static const bool on = []()
        {
            const char *e = std::getenv("PS2X_VIF1_TRACE");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        return on;
    }

    std::atomic<uint64_t> s_vif1Mpg{0};
    std::atomic<uint64_t> s_vif1Mscal{0};
    std::atomic<uint64_t> s_vif1Mscnt{0};
    std::atomic<uint64_t> s_vif1Direct{0};
    std::atomic<uint64_t> s_vif1Unpack{0};
    std::atomic<uint64_t> s_vif1Other{0};
    std::atomic<uint64_t> s_vif1Bytes{0};
    std::atomic<uint64_t> s_vif1Calls{0};

    // ---- VU1 data-memory quadword watch (PS2X_VU1WATCH="<lo>:<hi>") -------
    //
    // "Who writes VU1 qwNN, and is what they write finite?" A transform block
    // that arrives corrupt can only have come from one of two places: the EE
    // built it wrong and VIF unpacked it verbatim, or the VU1 microprogram
    // computed it. This instruments the first case. Every UNPACK landing in
    // the watched quadword range prints the destination qw, the four lanes as
    // raw hex and as floats, and the guest physical address of the DMA packet
    // the data came from -- which is what turns "the matrix is wrong" into an
    // EE address a producer trace can be hung on.
    //
    // Rate-limited to one report per destination quadword per 2 s so a
    // per-frame upload leaves a readable trickle in a 40-minute boot log
    // rather than gigabytes.
    bool vu1WatchEnabled(uint32_t &lo, uint32_t &hi)
    {
        static uint32_t s_lo = 0u, s_hi = 0u;
        static const bool on = [&]()
        {
            const char *e = std::getenv("PS2X_VU1WATCH");
            if (!e || !e[0])
                return false;
            char *end = nullptr;
            unsigned long a = std::strtoul(e, &end, 0);
            unsigned long b = a;
            if (end && *end == ':')
                b = std::strtoul(end + 1, nullptr, 0);
            if (b < a)
                std::swap(a, b);
            s_lo = static_cast<uint32_t>(a);
            s_hi = static_cast<uint32_t>(b);
            return true;
        }();
        lo = s_lo;
        hi = s_hi;
        return on;
    }

    uint32_t s_vu1WatchSrcPhys = 0xFFFFFFFFu; // published by the srcPhys overload
    // Raw source bytes consumed by the write being reported (unpack path only).
    const uint8_t *s_vu1WatchSrcVec = nullptr;
    uint32_t s_vu1WatchSrcLen = 0u;

    void vu1WatchReport(uint32_t destVec, const uint8_t *qw, const char *how)
    {
        static uint64_t s_lastMs[1024] = {};
        static std::atomic<uint64_t> s_lines{0};
        if (destVec >= 1024u)
            return;
        if (s_lines.load(std::memory_order_relaxed) > 20000u)
            return;
        const uint64_t now = dq8ProbeNowMs();
        if (now - s_lastMs[destVec] < 2000u)
            return;
        s_lastMs[destVec] = now;
        s_lines.fetch_add(1, std::memory_order_relaxed);

        uint32_t u[4];
        float f[4];
        std::memcpy(u, qw, sizeof(u));
        std::memcpy(f, qw, sizeof(f));
        int bad = 0;
        for (int i = 0; i < 4; ++i)
            if (((u[i] >> 23) & 0xFFu) == 0xFFu)
                ++bad;
        std::cerr << "[vu1:watch] t=" << std::dec << now << "ms qw" << destVec
                  << " via=" << how
                  << " src=0x" << std::hex << s_vu1WatchSrcPhys << std::dec
                  << " raw=" << std::hex << u[0] << ',' << u[1] << ',' << u[2] << ',' << u[3]
                  << std::dec
                  << " f=" << f[0] << ',' << f[1] << ',' << f[2] << ',' << f[3]
                  << " exp255lanes=" << bad;
        if (s_vu1WatchSrcVec && s_vu1WatchSrcLen)
        {
            std::cerr << " srcbytes=" << std::hex;
            for (uint32_t k = 0; k < s_vu1WatchSrcLen && k < 16u; ++k)
                std::cerr << (k ? "," : "") << static_cast<unsigned>(s_vu1WatchSrcVec[k]);
            std::cerr << std::dec;
        }
        else
        {
            std::cerr << " srcbytes=fill"; // no source consumed: ROW-fill cycle
        }
        std::cerr << std::endl;
    }

    // ---- Unbiased qw60/qw61/qw62 census v2 (PS2X_MTXCENSUS=1) -------------
    //
    // v1 (below, UNCHANGED -- keep its exact output line so old greps still
    // work) answered "is qw60 == qw61 a real invariant, or just what the
    // rate-limited watch happens to sample?": yes, it is real, ~45% of the
    // time. That does not localise the producer. v2 adds:
    //
    //   1. Per-phase (here: wall-clock epoch -- this TU has no game-phase
    //      signal; a phase flag would have to live on PS2Memory or
    //      R5900Context, and adding a data member to either changes ABI
    //      layout and segfaults instantly against the already-compiled
    //      corpus, so it is out) breakdown of the identical-rows fraction.
    //   2. For DISTINCT (non-identical) qw60/qw61 pairs: is the surviving
    //      basis actually sane? Row norms (qw60/qw61/qw62 as float xyz
    //      triples) and row0-vs-row1 orthogonality, so "55% distinct" can be
    //      read as either "unit-norm orthogonal bases" or "also garbage".
    //   3. Top-8 most frequent raw qw60 quadwords, to tell "one repeated
    //      placeholder" (a single degenerate code path) from "diverse
    //      garbage" (systemic).
    //
    // qw62 is folded in opportunistically: it is read only when it arrives
    // immediately after a qw60/qw61 pair (adjacent quadwords of the same
    // UNPACK, same as the v1 pairing trick), which is the common case for a
    // 3-or-4-row matrix upload. All of this is behind the same env gate as
    // v1 and is a no-op (one branch) when unset.
    struct MtxEpochStats
    {
        uint64_t pairs = 0u, equal = 0u, nan60 = 0u, distinct = 0u;
        double normSum[3] = {0.0, 0.0, 0.0}; // qw60,61,62 as xyz triples
        uint64_t normUnit[3] = {0u, 0u, 0u}; // within 1% of unit length
        uint64_t normSamples[3] = {0u, 0u, 0u};
        uint64_t orthoPairs = 0u; // distinct pairs where qw62 also landed
        uint64_t orthoOk = 0u;    // |dot(row0,row1)_normalised| < 0.01
    };

    struct Qw4
    {
        uint32_t v[4];
        bool operator==(const Qw4 &o) const { return std::memcmp(v, o.v, sizeof(v)) == 0; }
    };
    struct Qw4Hash
    {
        size_t operator()(const Qw4 &q) const
        {
            uint64_t h = 1469598103934665603ull;
            for (uint32_t i = 0; i < 4u; ++i)
            {
                h ^= q.v[i];
                h *= 1099511628211ull;
            }
            return static_cast<size_t>(h);
        }
    };

    inline float qwLaneF(uint32_t bits)
    {
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    // L2 length of the first 3 lanes (the xyz part of a matrix row).
    inline double rowNormXYZ(const uint32_t *lanes)
    {
        const double x = qwLaneF(lanes[0]), y = qwLaneF(lanes[1]), z = qwLaneF(lanes[2]);
        return std::sqrt(x * x + y * y + z * z);
    }

    inline double rowDotXYZ(const uint32_t *a, const uint32_t *b)
    {
        return qwLaneF(a[0]) * qwLaneF(b[0]) +
               qwLaneF(a[1]) * qwLaneF(b[1]) +
               qwLaneF(a[2]) * qwLaneF(b[2]);
    }

    // srcBytes/srcLen are the RAW SOURCE BYTES this unpack consumed (null if
    // unavailable) and srcGuest is the EE address they were DMA'd from
    // (0xFFFFFFFF if unmappable). They exist to answer the one question the
    // aggregate health numbers cannot: when a matrix row arrives saturated,
    // did the EE hand us a saturated row, or did our unpack manufacture it?
    void vu1MtxCensus(uint32_t destVec, const uint32_t *lanes,
                      const uint8_t *srcBytes, uint32_t srcLen, uint32_t srcGuest)
    {
        static const bool on = []()
        {
            const char *e = std::getenv("PS2X_MTXCENSUS");
            return e && e[0] && e[0] != '0';
        }();
        if (!on || (destVec != 60u && destVec != 61u && destVec != 62u))
            return;

        // ---- v1 state (kept byte-for-byte) ---------------------------------
        static uint32_t s_last60[4] = {0u, 0u, 0u, 0u};
        static bool s_have60 = false;
        static uint64_t s_pairs = 0u, s_equal = 0u, s_nan = 0u, s_lastMs = 0u;

        // ---- v2 state -------------------------------------------------------
        static uint32_t s_last61[4] = {0u, 0u, 0u, 0u};
        static bool s_have61ForTriple = false;   // last60/last61 still fresh, no qw62 consumed yet
        static bool s_lastPairDistinct = false;
        constexpr uint64_t kEpochMs = 30000u; // 30s wall-clock buckets (no phase signal in this TU)
        static std::map<uint64_t, MtxEpochStats> s_epochs;
        static std::unordered_map<Qw4, uint64_t, Qw4Hash> s_qw60Freq;
        // Bound cost if the population turns out diverse. NOTE: this cap is
        // NOT neutral. Once it is hit, only values already in the table keep
        // accumulating counts, so the top-8 becomes a picture of whenever the
        // table filled (typically early boot) rather than of the whole run,
        // and distinctKeys= reports the cap instead of the true diversity.
        // It must therefore announce itself; a silently saturated frequency
        // table reads as "the population is small and stable" when the truth
        // may be the exact opposite. Tunable via PS2X_MTXCENSUS_MAX_KEYS
        // (0 = unlimited).
        static const size_t kMaxQw60FreqKeys = []() -> size_t
        {
            const char *e = std::getenv("PS2X_MTXCENSUS_MAX_KEYS");
            if (!e || !e[0])
                return 20000u;
            char *end = nullptr;
            const unsigned long v = std::strtoul(e, &end, 0);
            return (end == e) ? 20000u : static_cast<size_t>(v);
        }();
        static bool s_qw60FreqSaturated = false;

        auto printReport = [&](uint64_t now)
        {
            // v1 line, unchanged, so old greps for "qw60==qw61=" still work.
            std::cerr << "[vu1:mtxcensus] t=" << std::dec << now
                      << "ms pairs=" << s_pairs
                      << " qw60==qw61=" << s_equal
                      << " (" << (s_pairs ? (100.0 * double(s_equal) / double(s_pairs)) : 0.0) << "%)"
                      << " qw60allNaN=" << s_nan << std::endl;

            // Item 1: per-epoch (phase-bucketed by wall-clock, see block
            // comment above for why epoch and not a real phase id) breakdown.
            for (const auto &kv : s_epochs)
            {
                const uint64_t epoch = kv.first;
                const MtxEpochStats &st = kv.second;
                std::cerr << "[vu1:mtxcensus:phase] epoch=" << epoch
                          << " tRangeMs=[" << (epoch * kEpochMs) << "," << (epoch * kEpochMs + kEpochMs) << ")"
                          << " pairs=" << st.pairs
                          << " equal=" << st.equal
                          << " (" << (st.pairs ? (100.0 * double(st.equal) / double(st.pairs)) : 0.0) << "%)"
                          << " nan60=" << st.nan60
                          << " distinct=" << st.distinct
                          << std::endl;

                // Item 2: row-norm and orthogonality health for the DISTINCT
                // pairs in this epoch only (identical pairs say nothing about
                // basis health -- they are the collapse itself).
                if (st.distinct == 0u)
                    continue;
                std::cerr << "[vu1:mtxcensus:health] epoch=" << epoch
                          << " distinct=" << st.distinct;
                static const char *rowName[3] = {"qw60", "qw61", "qw62"};
                for (int r = 0; r < 3; ++r)
                {
                    const double mean = st.normSamples[r] ? (st.normSum[r] / double(st.normSamples[r])) : 0.0;
                    const double unitFrac = st.normSamples[r]
                                                 ? (100.0 * double(st.normUnit[r]) / double(st.normSamples[r]))
                                                 : 0.0;
                    std::cerr << " " << rowName[r] << "Norm(n=" << st.normSamples[r]
                              << ",mean=" << mean << ",unit1pct=" << unitFrac << "%)";
                }
                const double orthoFrac = st.orthoPairs
                                             ? (100.0 * double(st.orthoOk) / double(st.orthoPairs))
                                             : 0.0;
                std::cerr << " ortho01(n=" << st.orthoPairs << ",abs<0.01=" << orthoFrac << "%)"
                          << std::endl;
            }

            // Item 3: top-8 most frequent raw qw60 quadwords.
            std::vector<std::pair<Qw4, uint64_t>> freq(s_qw60Freq.begin(), s_qw60Freq.end());
            std::partial_sort(freq.begin(), freq.begin() + std::min<size_t>(8u, freq.size()), freq.end(),
                               [](const auto &a, const auto &b) { return a.second > b.second; });
            const size_t topN = std::min<size_t>(8u, freq.size());
            for (size_t i = 0; i < topN; ++i)
            {
                const Qw4 &q = freq[i].first;
                std::cerr << "[vu1:mtxcensus:qw60top] rank=" << (i + 1u)
                          << " count=" << freq[i].second
                          << " raw=0x" << std::hex << q.v[0] << ",0x" << q.v[1]
                          << ",0x" << q.v[2] << ",0x" << q.v[3] << std::dec
                          << " distinctKeys=" << s_qw60Freq.size()
                          << (s_qw60FreqSaturated ? " keysTRUNCATED=1" : "")
                          << std::endl;
            }
        };

        if (destVec == 60u)
        {
            std::memcpy(s_last60, lanes, sizeof(s_last60));
            s_have60 = true;
            s_have61ForTriple = false;

            // ---- saturated-row attribution ------------------------------
            // A row whose xyz norm exceeds 1e8 cannot be a rotation basis.
            // Hardware (40 oracle arena dumps, 20884 blocks) has ZERO of
            // these, so every one is a defect. Tally is uncapped; the raw
            // dump is budgeted and announces its own truncation.
            {
                const double n0 = rowNormXYZ(lanes);
                if (n0 > 1e8 || !(n0 == n0))
                {
                    static const uint32_t satDumpMax = []() -> uint32_t
                    {
                        const char *e = std::getenv("PS2X_MTXCENSUS_SAT_DUMP");
                        if (!e || !e[0])
                            return 64u;
                        char *end = nullptr;
                        const unsigned long v = std::strtoul(e, &end, 0);
                        return (end == e) ? 64u : static_cast<uint32_t>(v);
                    }();
                    static uint64_t satTotal = 0u;
                    static uint32_t satDumped = 0u;
                    static bool satTruncated = false;
                    ++satTotal;
                    if (satDumpMax == 0u || satDumped < satDumpMax)
                    {
                        ++satDumped;
                        std::cerr << "[vu1:mtxcensus:sat] #" << satTotal
                                  << " t=" << std::dec << dq8ProbeNowMs() << "ms"
                                  << " norm=" << n0
                                  << " srcGuest=0x" << std::hex << srcGuest
                                  << " dst=0x" << lanes[0] << ",0x" << lanes[1]
                                  << ",0x" << lanes[2] << ",0x" << lanes[3];
                        // The discriminator: if src == dst the EE produced the
                        // saturated row and the VIF path is innocent; if they
                        // differ, the unpack is manufacturing it.
                        if (srcBytes && srcLen >= 16u)
                        {
                            uint32_t sw[4];
                            std::memcpy(sw, srcBytes, sizeof(sw));
                            std::cerr << " src=0x" << sw[0] << ",0x" << sw[1]
                                      << ",0x" << sw[2] << ",0x" << sw[3]
                                      << " srcEqDst="
                                      << (std::memcmp(sw, lanes, sizeof(sw)) == 0 ? 1 : 0);
                        }
                        else
                        {
                            std::cerr << " src=unavailable srcLen=" << std::dec << srcLen << std::hex;
                        }
                        std::cerr << std::dec << std::endl;
                    }
                    else if (!satTruncated)
                    {
                        satTruncated = true;
                        std::cerr << "[vu1:mtxcensus:sat] TRUNCATED after " << satDumpMax
                                  << " events; further saturated rows are NOT dumped"
                                  << " (raise or disable with PS2X_MTXCENSUS_SAT_DUMP=<n>,"
                                  << " 0 = unlimited). The satTotal= counter below stays complete."
                                  << std::endl;
                    }
                    if ((satTotal % 5000u) == 0u)
                    {
                        std::cerr << "[vu1:mtxcensus:sat] satTotal=" << std::dec << satTotal
                                  << " t=" << dq8ProbeNowMs() << "ms" << std::endl;
                    }
                }
            }

            Qw4 key;
            std::memcpy(key.v, lanes, sizeof(key.v));
            auto it = s_qw60Freq.find(key);
            if (it != s_qw60Freq.end())
            {
                ++it->second;
            }
            else if (kMaxQw60FreqKeys == 0u || s_qw60Freq.size() < kMaxQw60FreqKeys)
            {
                s_qw60Freq.emplace(key, 1u);
            }
            else if (!s_qw60FreqSaturated)
            {
                s_qw60FreqSaturated = true;
                std::cerr << "[vu1:mtxcensus:qw60top] TRUNCATED after "
                          << kMaxQw60FreqKeys
                          << " distinct keys; new distinct qw60 values are NOT counted from here on,"
                          << " so the top-8 and distinctKeys= are biased toward values seen earlier"
                          << " (raise or disable with PS2X_MTXCENSUS_MAX_KEYS=<n>, 0 = unlimited)"
                          << std::endl;
            }
            return;
        }

        if (destVec == 61u)
        {
            if (!s_have60)
                return;
            std::memcpy(s_last61, lanes, sizeof(s_last61));
            s_have61ForTriple = true;

            ++s_pairs;
            const bool eq = std::memcmp(s_last60, s_last61, sizeof(s_last60)) == 0;
            if (eq)
                ++s_equal;
            int bad = 0;
            for (int i = 0; i < 4; ++i)
                if (((s_last60[i] >> 23) & 0xFFu) == 0xFFu)
                    ++bad;
            if (bad == 4)
                ++s_nan;
            s_lastPairDistinct = !eq;

            const uint64_t now = dq8ProbeNowMs();
            MtxEpochStats &st = s_epochs[now / kEpochMs];
            ++st.pairs;
            if (eq)
                ++st.equal;
            if (bad == 4)
                ++st.nan60;
            if (!eq)
            {
                ++st.distinct;
                const double n0 = rowNormXYZ(s_last60), n1 = rowNormXYZ(s_last61);
                st.normSum[0] += n0; ++st.normSamples[0];
                if (n0 > 0.99 && n0 < 1.01) ++st.normUnit[0];
                st.normSum[1] += n1; ++st.normSamples[1];
                if (n1 > 0.99 && n1 < 1.01) ++st.normUnit[1];
                const double mag = n0 * n1;
                const double normDot = (mag > 1e-9) ? (rowDotXYZ(s_last60, s_last61) / mag) : 1.0;
                ++st.orthoPairs;
                if (std::fabs(normDot) < 0.01) ++st.orthoOk;
            }

            if (now - s_lastMs >= 5000u)
            {
                s_lastMs = now;
                printReport(now);
            }
            return;
        }

        // destVec == 62u: only usable if it lands immediately after a fresh
        // qw60/qw61 pair (same UNPACK), and only informative for DISTINCT
        // pairs -- an identical pair is the collapse itself, not a basis to
        // grade.
        if (destVec == 62u)
        {
            if (!s_have60 || !s_have61ForTriple)
                return;
            s_have61ForTriple = false; // consume; don't reuse for a later stray qw62
            if (!s_lastPairDistinct)
                return;
            const uint64_t now = dq8ProbeNowMs();
            MtxEpochStats &st = s_epochs[now / kEpochMs];
            const double n2 = rowNormXYZ(lanes);
            st.normSum[2] += n2; ++st.normSamples[2];
            if (n2 > 0.99 && n2 < 1.01) ++st.normUnit[2];
        }
    }

    uint32_t gifImageQwcFromTag(const uint8_t *data, uint32_t sizeBytes)
    {
        if (!data || sizeBytes < 16u)
            return 0u;

        uint64_t tagLo = 0u;
        std::memcpy(&tagLo, data, sizeof(tagLo));
        const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
        if (flg != kGifFmtImage)
            return 0u;

        return static_cast<uint32_t>(tagLo & 0x7FFFu);
    }
}

// Segment map of the VIF1 chain currently being interpreted. A TU-global rather
// than a PS2Memory member on purpose: adding a data member to PS2Memory changes
// its layout and breaks ABI against an already-compiled recompiled corpus.
std::vector<std::pair<uint32_t, uint32_t>> g_vif1ChainSegMap;

// Map a parse offset within the coalesced VIF1 chain buffer back to the guest
// physical address those bytes were DMA'd from. Diagnostics only: this is what
// lets the VU1 data-memory watch name the EE producer of a bad transform block
// instead of reporting an unknown source. Returns 0xFFFFFFFF when there is no
// map (non-chain path), when the segment came from scratchpad, or when the
// offset predates the first segment.
static uint32_t chainOffsetToGuest(uint32_t chainOffset)
{
    const auto &m_vif1ChainSegMap = g_vif1ChainSegMap;
    if (m_vif1ChainSegMap.empty())
        return 0xFFFFFFFFu;
    // Segments are appended in increasing offset order, so the owning segment
    // is the last one whose offset is <= chainOffset.
    size_t lo = 0, hi = m_vif1ChainSegMap.size();
    while (lo < hi)
    {
        const size_t mid = lo + (hi - lo) / 2;
        if (m_vif1ChainSegMap[mid].first <= chainOffset)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo == 0)
        return 0xFFFFFFFFu;
    const auto &seg = m_vif1ChainSegMap[lo - 1];
    if (seg.second == 0xFFFFFFFFu)
        return 0xFFFFFFFFu; // scratchpad-sourced segment
    return seg.second + (chainOffset - seg.first);
}

void PS2Memory::processVIF1Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (!m_rdram || !m_gsVRAM || sizeBytes == 0u)
        return;
    if (srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    s_vu1WatchSrcPhys = srcPhys;
    processVIF1Data(m_rdram + srcPhys, sizeBytes);
    s_vu1WatchSrcPhys = 0xFFFFFFFFu;
}

void PS2Memory::processVIF1Data(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || !m_gsVRAM || sizeBytes == 0u)
        return;

    {
        const uint64_t call = s_vif1Calls.fetch_add(1, std::memory_order_relaxed);
        s_vif1Bytes.fetch_add(sizeBytes, std::memory_order_relaxed);
        if (vif1TraceOn() && (call % 512u) == 0u)
        {
            std::cerr << "[vif1:census] calls=" << std::dec << (call + 1u)
                      << " bytes=" << s_vif1Bytes.load(std::memory_order_relaxed)
                      << " mpg=" << s_vif1Mpg.load(std::memory_order_relaxed)
                      << " mscal=" << s_vif1Mscal.load(std::memory_order_relaxed)
                      << " mscnt=" << s_vif1Mscnt.load(std::memory_order_relaxed)
                      << " direct=" << s_vif1Direct.load(std::memory_order_relaxed)
                      << " unpack=" << s_vif1Unpack.load(std::memory_order_relaxed)
                      << " other=" << s_vif1Other.load(std::memory_order_relaxed)
                      << std::endl;
        }
    }

    // Microprogram activation latch, shared by MSCAL / MSCALF / MSCNT.
    // Mirrors PCSX2 Vif_Codes.cpp vuExecMicro() for VIF1 exactly.
    auto latchVu1Activation = [&]()
    {
        vif1_regs.itop = vif1_regs.itops & 0x3FFu;
        vif1_regs.top = vif1_regs.tops & 0x3FFu;
        vif1_regs.stat ^= (1u << 7); // DBF ^= 1
        const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
        const uint32_t base = vif1_regs.base & 0x3FFu;
        const uint32_t ofst = vif1_regs.ofst & 0x3FFu;
        vif1_regs.tops = dbf ? ((base + ofst) & 0x3FFu) : base;
    };

    uint32_t pos = 0;

    while (pos + 4 <= sizeBytes)
    {
        uint32_t cmd;
        memcpy(&cmd, data + pos, 4);
        pos += 4;

        uint8_t opcode = (cmd >> 24) & 0x7F;
        uint16_t imm = cmd & 0xFFFF;
        uint8_t num = (cmd >> 16) & 0xFF;
        const bool irq = (cmd & 0x80000000u) != 0u;

        // Always-maintained opcode census (see vif1TraceOn() above).
        if ((opcode & 0x60) == 0x60)
            s_vif1Unpack.fetch_add(1, std::memory_order_relaxed);
        else if (opcode == VIF_MPG)
            s_vif1Mpg.fetch_add(1, std::memory_order_relaxed);
        else if (opcode == VIF_MSCAL || opcode == VIF_MSCALF)
            s_vif1Mscal.fetch_add(1, std::memory_order_relaxed);
        else if (opcode == VIF_MSCNT)
            s_vif1Mscnt.fetch_add(1, std::memory_order_relaxed);
        else if (opcode == VIF_DIRECT || opcode == VIF_DIRECTHL)
            s_vif1Direct.fetch_add(1, std::memory_order_relaxed);
        else
            s_vif1Other.fetch_add(1, std::memory_order_relaxed);

        const uint32_t opcodeIndex = s_debugVif1OpcodeCount.fetch_add(1, std::memory_order_relaxed);
        static const uint32_t kMaxVif1CmdTraceLogs = localDiagEnvLimit("PS2X_VIF1_CMD_TRACE_MAX_LOGS", 400u);
        static std::atomic<bool> s_vif1CmdTraceTruncated{false};
        if (vif1TraceOn() &&
            localDiagLogBudget(std::cerr, "[vif1:cmd]", "PS2X_VIF1_CMD_TRACE_MAX_LOGS",
                               kMaxVif1CmdTraceLogs, opcodeIndex, s_vif1CmdTraceTruncated))
        {
            std::cerr << "[vif1:cmd] idx=" << std::dec << opcodeIndex
                      << " opcode=0x" << std::hex << static_cast<uint32_t>(opcode)
                      << " imm=0x" << imm
                      << std::dec
                      << " num=" << static_cast<uint32_t>(num)
                      << " irq=" << static_cast<uint32_t>(irq ? 1u : 0u)
                      << std::endl;
        }
        if (opcodeIndex < 160u)
        {
            RUNTIME_LOG("[vif1:cmd] idx=" << opcodeIndex
                                          << " opcode=0x" << std::hex << static_cast<uint32_t>(opcode)
                                          << " imm=0x" << imm
                                          << std::dec
                                          << " num=" << static_cast<uint32_t>(num)
                                          << " irq=" << static_cast<uint32_t>(irq ? 1u : 0u)
                                          << std::endl);
        }

        // Track most-recent command for VIFn_CODE emulation.
        vif1_regs.code = cmd;
        vif1_regs.num = num;
        if (irq)
            vif1_regs.stat |= (1u << 11); // INT

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif1_regs.cycle = imm;
            continue;
        }
        else if (opcode == VIF_OFFSET)
        {
            // DC2 spec 04 / G28: OFFSET is exactly `DBF=0; OFST=imm;
            // TOPS=BASE` and must NOT write BASE. The previous code did
            // `BASE = old TOPS`, so every OFFSET advanced the double buffer's
            // origin by one half instead of resetting it -- the buffer marched
            // through VU memory rather than ping-ponging, eventually
            // overlapping VU1's input window with its XGKICK output window.
            // Verified against PCSX2 Vif_Codes.cpp vifCode_Offset.
            vif1_regs.stat &= ~(1u << 7); // DBF = 0
            vif1_regs.ofst = imm & 0x3FFu;
            vif1_regs.tops = vif1_regs.base & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_BASE)
        {
            // PCSX2 vifCode_Base writes BASE only; TOPS is (re)published by
            // OFFSET and by each microprogram activation, never by BASE.
            vif1_regs.base = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            // PCSX2 vifCode_ITop writes ITOPS (the staging register); ITOP --
            // what XITOP reads -- is latched from ITOPS at activation. Writing
            // ITOP directly let a running program see the ITOP belonging to
            // the NEXT batch, which is exactly what a double-buffered VIF
            // stream queues while the current batch is still executing.
            vif1_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif1_regs.mode = imm & 3u;
            continue;
        }
        else if (opcode == VIF_MSKPATH3)
        {
            // VIF command docs: MSKPATH3 uses IMMEDIATE bit 15.
            const bool wasMasked = m_path3Masked;
            m_path3Masked = (imm & 0x8000u) != 0u;
            if (wasMasked && !m_path3Masked)
                flushMaskedPath3Packets();
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif1_regs.mark = imm;
            vif1_regs.stat |= (1u << 6); // MRK
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_MSCAL || opcode == VIF_MSCALF)
        {
            // Microprogram activation (PCSX2 Vif_Codes.cpp vuExecMicro):
            //     ITOP <- ITOPS
            //     TOP  <- TOPS          (the half VIF1 just finished writing)
            //     DBF  ^= 1 ; TOPS <- DBF ? BASE+OFST : BASE
            //
            // TOP is what XTOP returns, and it must name the buffer the
            // program is about to READ. Previously TOP was never maintained
            // at all and XTOP was fed TOPS *after* the DBF flip -- i.e. the
            // half VIF1 will write NEXT. Every double-buffered VU1 batch
            // therefore transformed the other buffer's stale/unwritten
            // contents, which is how a correct microprogram ends up emitting
            // GIF packets whose tags never reach EOP.
            latchVu1Activation();
            uint32_t startPC = (uint32_t)imm * 8u;
            if (startPC >= 16384u)
            {
                static std::atomic<uint64_t> s_badMscal{0};
                static const uint32_t kMaxBadMscalLogs = localDiagEnvLimit("PS2X_VIF1_BADMSCAL_MAX_LOGS", 6u);
                static std::atomic<bool> s_badMscalTruncated{false};
                const uint64_t bn = s_badMscal.fetch_add(1, std::memory_order_relaxed);
                if (localDiagLogBudget(std::cerr, "[vif1:badmscal]", "PS2X_VIF1_BADMSCAL_MAX_LOGS",
                                       kMaxBadMscalLogs, static_cast<uint32_t>(bn), s_badMscalTruncated))
                {
                    std::cerr << "[vif1:badmscal] n=" << std::dec << bn
                              << " imm=0x" << std::hex << imm
                              << " pos=0x" << (pos - 4u)
                              << " size=0x" << sizeBytes << std::dec << std::endl;
                    const uint32_t from = (pos > 68u) ? (pos - 68u) : 0u;
                    const uint32_t to = (pos + 64u < sizeBytes) ? (pos + 64u) : sizeBytes;
                    for (uint32_t q = from; q + 4u <= to; q += 4u)
                    {
                        uint32_t w;
                        std::memcpy(&w, data + q, 4);
                        std::cerr << "[vif1:badmscal]   +0x" << std::hex << q
                                  << " = 0x" << w << std::dec
                                  << ((q == pos - 4u) ? "  <== here" : "") << std::endl;
                    }
                }
            }
            const uint32_t kickIndex = s_debugVu1KickCount.fetch_add(1, std::memory_order_relaxed);
            static const uint32_t kMaxMscalTraceLogs = localDiagEnvLimit("PS2X_VIF1_MSCAL_TRACE_MAX_LOGS", 200u);
            static std::atomic<bool> s_mscalTraceTruncated{false};
            if (vif1TraceOn() &&
                localDiagLogBudget(std::cerr, "[vif1:mscal]", "PS2X_VIF1_MSCAL_TRACE_MAX_LOGS",
                                   kMaxMscalTraceLogs, kickIndex, s_mscalTraceTruncated))
            {
                std::cerr << "[vif1:mscal] idx=" << std::dec << kickIndex
                          << " opcode=0x" << std::hex << static_cast<uint32_t>(opcode)
                          << " imm=0x" << imm
                          << " startPc=0x" << startPC
                          << " itop=0x" << vif1_regs.itop
                          << " cb=" << (m_vu1MscalCallback ? 1 : 0)
                          << std::dec << std::endl;
            }
            if (kickIndex < 48u)
            {
                RUNTIME_LOG("[vif1:mscal] idx=" << kickIndex
                                                << " opcode=0x" << std::hex << static_cast<uint32_t>(opcode)
                                                << " imm=0x" << imm
                                                << " startPc=0x" << startPC
                                                << " itop=0x" << vif1_regs.itop
                                                << std::dec << std::endl);
            }
            if (m_vu1MscalCallback)
                m_vu1MscalCallback(startPC, vif1_regs.itop);
            continue;
        }
        else if (opcode == VIF_MSCNT)
        {
            latchVu1Activation();
            const uint32_t kickIndex = s_debugVu1KickCount.fetch_add(1, std::memory_order_relaxed);
            if (kickIndex < 48u)
            {
                RUNTIME_LOG("[vif1:mscnt] idx=" << kickIndex
                                                << " itop=0x" << std::hex << vif1_regs.itop
                                                << " pc=resume"
                                                << std::dec << std::endl);
            }
            if (m_vu1MscntCallback)
                m_vu1MscntCallback(vif1_regs.itop);
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4 > sizeBytes)
                break;
            uint32_t maskValue = 0;
            std::memcpy(&maskValue, data + pos, sizeof(maskValue));
            vif1_regs.mask = maskValue;
            pos += 4;
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16 > sizeBytes)
                break;
            std::memcpy(vif1_regs.row, data + pos, 16);
            pos += 16;
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16 > sizeBytes)
                break;
            std::memcpy(vif1_regs.col, data + pos, 16);
            pos += 16;
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            uint32_t destAddr = (uint32_t)imm * 8u;
            // VIF MPG semantics: NUM==0 means 256 instructions (2048 bytes).
            // MPG payload is instruction-packed and should not be QW-aligned.
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            if (vif1TraceOn())
            {
                static std::atomic<uint64_t> s_mpgLogged{0};
                static std::atomic<uint64_t> s_mpgLoggedNonZero{0};
                static const uint32_t kMaxMpgLogged = localDiagEnvLimit("PS2X_VIF1_MPG_MAX_LOGS", 40u);
                static const uint32_t kMaxMpgNonZeroLogged = localDiagEnvLimit("PS2X_VIF1_MPG_NONZERO_MAX_LOGS", 60u);
                static std::atomic<bool> s_mpgTruncated{false};
                const uint64_t n = s_mpgLogged.fetch_add(1, std::memory_order_relaxed);
                // dest==0 uploads dominate the title phase and would otherwise
                // eat the whole cap before any field microcode lands.
                const bool wantNonZero =
                    (destAddr != 0u) &&
                    (s_mpgLoggedNonZero.fetch_add(1, std::memory_order_relaxed) < kMaxMpgNonZeroLogged);
                const bool withinMpgBudget =
                    (kMaxMpgLogged == 0u) || (n < static_cast<uint64_t>(kMaxMpgLogged)) || wantNonZero;
                if (withinMpgBudget)
                {
                    std::cerr << "[vif1:mpg] idx=" << std::dec << n
                              << " dest=0x" << std::hex << destAddr
                              << " instrs=" << std::dec << instructionCount
                              << " bytes=" << mpgBytes
                              << " avail=" << (sizeBytes - pos)
                              << " code=" << (m_vu1Code ? 1 : 0)
                              << std::endl;
                }
                else
                {
                    bool expected = false;
                    if (s_mpgTruncated.compare_exchange_strong(expected, true, std::memory_order_relaxed))
                    {
                        std::cerr << "[vif1:mpg] TRUNCATED after " << kMaxMpgLogged
                                  << " (+" << kMaxMpgNonZeroLogged << " non-zero-dest) events;"
                                  << " further events are NOT logged (raise or disable with"
                                  << " PS2X_VIF1_MPG_MAX_LOGS=<n>/PS2X_VIF1_MPG_NONZERO_MAX_LOGS=<n>,"
                                  << " 0 = unlimited)" << std::endl;
                    }
                }
            }
            if (m_vu1Code && destAddr < PS2_VU1_CODE_SIZE && mpgBytes > 0)
            {
                uint32_t copyBytes = mpgBytes;
                if (destAddr + copyBytes > PS2_VU1_CODE_SIZE)
                    copyBytes = PS2_VU1_CODE_SIZE - destAddr;
                if (pos + copyBytes <= sizeBytes)
                    std::memcpy(m_vu1Code + destAddr, data + pos, copyBytes);
            }
            pos += mpgBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else if (opcode == VIF_DIRECT || opcode == VIF_DIRECTHL)
        {
            uint32_t qwCount = imm;
            if (qwCount == 0)
                qwCount = 65536;
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            const bool truncated = qwCount > availableQw;
            if (qwCount > availableQw)
                qwCount = availableQw;

            const bool directHl = (opcode == VIF_DIRECTHL);
            uint32_t consumedQw = 0u;

            // PATH2 is a byte stream into GIF, so an IMAGE-mode GIFtag's
            // payload routinely continues into the FOLLOWING DIRECT packet(s)
            // -- DQ8 uploads every texture as `DIRECT 1` carrying the bare
            // IMAGE tag, then `DIRECT n` carrying the n quadwords of pixels.
            // Our GS consumer takes one self-contained packet per submit, so
            // each continuation chunk is re-wrapped in a synthetic IMAGE tag.
            //
            // This used to be drained from the raw VIF stream at the top of
            // the parse loop, which swallowed the *vifcodes* sitting between
            // the tag and its data and desynchronised the rest of the chain.
            // That only ever worked because the DMAtag halves carrying those
            // vifcodes were themselves being dropped by the chain walker (see
            // the tag-transfer fix in ps2_memory.cpp); with the stream intact,
            // the continuation must be taken from the next DIRECT's payload.
            if (m_vif1PendingPath2ImageQwc != 0u && qwCount > 0u)
            {
                const uint32_t chunkQw = std::min<uint32_t>(m_vif1PendingPath2ImageQwc, qwCount);
                std::vector<uint8_t> imagePacket(16u + static_cast<size_t>(chunkQw) * 16u, 0u);
                const uint64_t imageTag =
                    static_cast<uint64_t>(chunkQw & 0x7FFFu) |
                    ((m_vif1PendingPath2ImageQwc == chunkQw) ? (1ull << 15) : 0ull) |
                    (static_cast<uint64_t>(kGifFmtImage) << 58);
                std::memcpy(imagePacket.data(), &imageTag, sizeof(imageTag));
                std::memcpy(imagePacket.data() + 16u, data + pos, static_cast<size_t>(chunkQw) * 16u);
                submitGifPacket(GifPathId::Path2,
                                imagePacket.data(),
                                static_cast<uint32_t>(imagePacket.size()),
                                true,
                                m_vif1PendingPath2DirectHl);

                m_vif1PendingPath2ImageQwc -= chunkQw;
                if (m_vif1PendingPath2ImageQwc == 0u)
                    m_vif1PendingPath2DirectHl = false;
                consumedQw = chunkQw;
            }

            if (qwCount > consumedQw)
            {
                const uint8_t *packet = data + pos + static_cast<size_t>(consumedQw) * 16u;
                const uint32_t packetQw = qwCount - consumedQw;
                submitGifPacket(GifPathId::Path2, packet, packetQw * 16, true, directHl);

                const uint32_t imageQw = gifImageQwcFromTag(packet, packetQw * 16u);
                if (imageQw != 0u)
                {
                    const uint32_t inlineImageQw = packetQw - 1u;
                    if (imageQw > inlineImageQw)
                    {
                        m_vif1PendingPath2ImageQwc = imageQw - inlineImageQw;
                        m_vif1PendingPath2DirectHl = directHl;
                    }
                }
            }

            pos += qwCount * 16;
            if (truncated)
            {
                pos = sizeBytes;
                break;
            }
            continue;
        }
        else if ((opcode & 0x60) == 0x60)
        {
            uint8_t vn = (opcode >> 2) & 0x3;
            uint8_t vl = opcode & 0x3;
            const bool maskEnable = (opcode & 0x10u) != 0u;
            int components = vn + 1;
            int bitsPerComponent = 32;
            switch (vl)
            {
            case 0:
                bitsPerComponent = 32;
                break;
            case 1:
                bitsPerComponent = 16;
                break;
            case 2:
                bitsPerComponent = 8;
                break;
            case 3:
                bitsPerComponent = (vn == 3) ? 4 : 16;
                break;
            default:
                break;
            }
            int bitsPerVector = (vl == 3 && vn == 3) ? 16 : (components * bitsPerComponent);
            uint32_t bytesPerVector = (bitsPerVector + 7) / 8;
            // UNPACK semantics: NUM is 8-bit and NUM==0 means 256 vectors (writes).
            const uint32_t writeVectorCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);

            // STCYCL controls write cycles for UNPACK.
            uint32_t cl = vif1_regs.cycle & 0xFFu;
            uint32_t wl = (vif1_regs.cycle >> 8) & 0xFFu;
            if (cl == 0u)
                cl = 1u;
            if (wl == 0u)
                wl = 1u;

            uint32_t sourceVectorCount = writeVectorCount;
            if (cl < wl)
            {
                const uint32_t fullBlocks = writeVectorCount / wl;
                uint32_t remainder = writeVectorCount % wl;
                if (remainder > cl)
                    remainder = cl;
                sourceVectorCount = fullBlocks * cl + remainder;
            }

            uint32_t totalBytes = sourceVectorCount * bytesPerVector;
            totalBytes = (totalBytes + 3) & ~3u;

            uint32_t vuAddr = (uint32_t)imm & 0x3FFu;
            if ((imm & 0x8000u) != 0u)
                vuAddr = (vuAddr + (vif1_regs.tops & 0x3FFu)) & 0x3FFu;

            const bool zeroExtend = (imm & 0x4000u) != 0u;
            if (m_vu1Data && totalBytes > 0 && pos + totalBytes <= sizeBytes)
            {
                const uint8_t *srcBase = data + pos;
                uint32_t srcIndex = 0u;
                for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
                {
                    const uint32_t cyclePos = writeIndex % wl;
                    const bool sourceAvailable = (cl >= wl) || (cyclePos < cl);

                    uint32_t destVec = 0;
                    if (cl >= wl)
                    {
                        destVec = (vuAddr + (writeIndex / wl) * cl + cyclePos) & 0x3FFu;
                    }
                    else
                    {
                        destVec = (vuAddr + writeIndex) & 0x3FFu;
                    }

                    uint32_t destOff = destVec * 16u;
                    if (destOff + 16u > PS2_VU1_DATA_SIZE)
                    {
                        if (sourceAvailable && srcIndex < sourceVectorCount)
                            ++srcIndex;
                        continue;
                    }

                    uint32_t lanes[4] = {0u, 0u, 0u, 0u};
                    std::memcpy(lanes, m_vu1Data + destOff, sizeof(lanes));
                    uint32_t decompressed[4] = {lanes[0], lanes[1], lanes[2], lanes[3]};
                    bool decoded = false;

                    const uint8_t *srcVec = nullptr;
                    // Chain-buffer offset of the source vector actually consumed
                    // for THIS write, so the watch can report the exact guest
                    // address and the raw source bytes (not the offset of the
                    // whole VIF command). 0xFFFFFFFF = no source consumed, i.e.
                    // a fill-write cycle.
                    uint32_t usedSrcOff = 0xFFFFFFFFu;
                    if (sourceAvailable && srcIndex < sourceVectorCount)
                    {
                        srcVec = srcBase + srcIndex * bytesPerVector;
                        usedSrcOff = static_cast<uint32_t>(pos + srcIndex * bytesPerVector);
                        ++srcIndex;
                        decoded = true;
                    }

                    auto extend16 = [&](uint16_t raw) -> uint32_t
                    {
                        if (zeroExtend)
                            return static_cast<uint32_t>(raw);
                        return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(raw)));
                    };

                    auto extend8 = [&](uint8_t raw) -> uint32_t
                    {
                        if (zeroExtend)
                            return static_cast<uint32_t>(raw);
                        return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(raw)));
                    };

                    bool handledFormat = true;
                    if (!decoded)
                    {
                        handledFormat = false;
                    }
                    else if (vl == 0u)
                    {
                        if (components == 1)
                        {
                            uint32_t scalar = 0;
                            std::memcpy(&scalar, srcVec, sizeof(scalar));
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                uint32_t scalar = 0;
                                std::memcpy(&scalar, srcVec + c * 4u, sizeof(scalar));
                                decompressed[c] = scalar;
                            }
                        }
                    }
                    else if (vl == 1u)
                    {
                        if (components == 1)
                        {
                            uint16_t raw = 0;
                            std::memcpy(&raw, srcVec, sizeof(raw));
                            const uint32_t scalar = extend16(raw);
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                uint16_t raw = 0;
                                std::memcpy(&raw, srcVec + c * 2u, sizeof(raw));
                                decompressed[c] = extend16(raw);
                            }
                        }
                    }
                    else if (vl == 2u)
                    {
                        if (components == 1)
                        {
                            const uint32_t scalar = extend8(srcVec[0]);
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                decompressed[c] = extend8(srcVec[c]);
                            }
                        }
                    }
                    else if (vl == 3u && vn == 3u)
                    {
                        // V4-5: packed color-like format in a single 16-bit value.
                        uint16_t packed = 0;
                        std::memcpy(&packed, srcVec, sizeof(packed));
                        decompressed[0] = packed & 0x1Fu;
                        decompressed[1] = (packed >> 5) & 0x1Fu;
                        decompressed[2] = (packed >> 10) & 0x1Fu;
                        decompressed[3] = (packed >> 15) & 0x01u;
                    }
                    else
                    {
                        handledFormat = false;
                    }

                    // Unknown compressed format fallback: preserve legacy raw-copy behavior.
                    if (!handledFormat && decoded && !maskEnable && (vif1_regs.mode == 0u || vif1_regs.mode == 3u))
                    {
                        uint32_t copyBytes = (bytesPerVector < 16u) ? bytesPerVector : 16u;
                        std::memcpy(m_vu1Data + destOff, srcVec, copyBytes);
                        {
                            uint32_t wlo, whi;
                            if (vu1WatchEnabled(wlo, whi) && destVec >= wlo && destVec <= whi)
                                vu1WatchReport(destVec, m_vu1Data + destOff, "unpack-raw");
                        }
                        continue;
                    }

                    const bool canAdd = (vl != 3u || vn != 3u);
                    const uint32_t mode = vif1_regs.mode & 3u;
                    const uint32_t colIdx = (cyclePos > 3u) ? 3u : cyclePos;
                    const uint32_t maskCycle = (cyclePos > 3u) ? 3u : cyclePos;

                    for (uint32_t field = 0u; field < 4u; ++field)
                    {
                        uint32_t maskSpec = 0u;
                        if (maskEnable)
                        {
                            const uint32_t shift = ((maskCycle * 4u) + field) * 2u;
                            maskSpec = (vif1_regs.mask >> shift) & 0x3u;
                        }

                        // In fill-write cycles with suspended source reads, treat raw-data selections as row-fill.
                        if (!decoded && maskSpec == 0u)
                            maskSpec = 1u;

                        uint32_t writeVal = lanes[field];
                        if (maskSpec == 0u)
                        {
                            if (handledFormat)
                            {
                                writeVal = decompressed[field];
                                if (canAdd && (mode == 1u || mode == 2u))
                                {
                                    writeVal = writeVal + vif1_regs.row[field];
                                    if (mode == 2u)
                                        vif1_regs.row[field] = writeVal;
                                }
                            }
                        }
                        else if (maskSpec == 1u)
                        {
                            writeVal = vif1_regs.row[field];
                        }
                        else if (maskSpec == 2u)
                        {
                            writeVal = vif1_regs.col[colIdx];
                        }
                        else
                        {
                            continue; // write-protect
                        }

                        lanes[field] = writeVal;
                    }

                    std::memcpy(m_vu1Data + destOff, lanes, sizeof(lanes));
                    {
                        uint32_t wlo, whi;
                        if (vu1WatchEnabled(wlo, whi) && destVec >= wlo && destVec <= whi)
                        {
                            // Name the EE address these bytes were DMA'd from
                            // when the packet came through the coalesced chain
                            // buffer (which is the case for the transform
                            // blocks); the srcPhys overload cannot publish it.
                            // Computed per report and restored afterwards, so
                            // one chain's first segment does not get attributed
                            // to every later unpack in the same chain.
                            const uint32_t savedSrc = s_vu1WatchSrcPhys;
                            const uint32_t mappedSrc =
                                chainOffsetToGuest(usedSrcOff != 0xFFFFFFFFu ? usedSrcOff : pos);
                            if (mappedSrc != 0xFFFFFFFFu)
                                s_vu1WatchSrcPhys = mappedSrc;
                            // Publish the RAW SOURCE BYTES this write consumed.
                            // This is the question that decides whether the EE
                            // handed us two identical quadwords or our unpack
                            // duplicated one: if the sources differ but the
                            // destinations match, the bug is here, not in the
                            // guest.
                            s_vu1WatchSrcVec = srcVec;
                            s_vu1WatchSrcLen = srcVec ? bytesPerVector : 0u;
                            vu1WatchReport(destVec, m_vu1Data + destOff, "unpack");
                            s_vu1WatchSrcVec = nullptr;
                            s_vu1WatchSrcLen = 0u;
                            s_vu1WatchSrcPhys = savedSrc;
                        }
                    }
                    // Unbiased qw60/qw61 census (PS2X_MTXCENSUS=1). The watch
                    // above is rate-limited to one report per quadword per 2 s,
                    // so it only ever shows the FIRST packet in each window --
                    // useless for deciding whether "qw60 == qw61 always" is a
                    // real invariant or merely an artifact of which object
                    // happened to be sampled. This counts EVERY write instead.
                    vu1MtxCensus(destVec, lanes, srcVec,
                                 srcVec ? bytesPerVector : 0u,
                                 chainOffsetToGuest(usedSrcOff != 0xFFFFFFFFu ? usedSrcOff : pos));
                }
            }
            pos += totalBytes;

            if (pos > sizeBytes)
                break;
            continue;
        }
        else
        {
            continue;
        }
    }
}
