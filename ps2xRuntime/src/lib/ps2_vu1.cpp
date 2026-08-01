#include "runtime/ps2_vu1.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_gif_arbiter.h"
#include "runtime/ps2_memory.h"
#include "ps2_log.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

// Defined in ps2_gs_gpu.cpp: one process-wide monotonic epoch shared by every
// DQ8 probe, so timestamps taken in different TUs are directly comparable.
uint64_t dq8ProbeNowMs();

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

    std::atomic<uint32_t> s_debugVu1XgkickCount{0};

    // ---- second-and-later GIFtag REGS sample (PS2X_GIFTAG_REGS=1) ---------
    //
    // The only GIFtag sample anywhere in the runtime ([vu1:kicksample] /
    // [vu1:kickwalk] above) is always tag #0 of the kick, and DQ8's kicks
    // always open with a short NLOOP=1 header tag -- so the tag that
    // actually carries the vertex run (REGS: is RGBAQ/ST present per vertex,
    // or is it absent so verts inherit stale color/texcoord state?) has
    // never been inspected. Bounded to the first 64 occurrences so it can't
    // flood the log; gated behind its own env var so it costs nothing when
    // off.
    bool giftagRegsSampleOn()
    {
        static const bool on = []()
        {
            const char *e = std::getenv("PS2X_GIFTAG_REGS");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        return on;
    }

    std::atomic<uint32_t> s_giftagRegsSampleCount{0};

    // ---- accumulated GIFtag REGS histogram (PS2X_GIFTAG_REGS=1) -----------
    //
    // The bounded sample above only ever caught tag #1 of the FIRST few
    // kicks in the run (all "idx<N" samples came back identical: nreg=3
    // regs=[0x2,0x1,0x4], i.e. ST+RGBAQ+XYZF2), while the whole-run PACKED
    // vertex population histogram ([gs:packed]) shows an ST:XYZF2 ratio of
    // only ~0.25 -- so a second REGS signature (almost certainly no ST, and
    // maybe no RGBAQ either) exists and is simply never reached by a
    // bounded early sample. This tracks every second-and-later vertex-run
    // tag by its full (nreg, REGS nibbles, FLG) signature so the count
    // naturally covers the entire run, FIELD included, instead of whatever
    // happened to be first. Capped at kGiftagHistCap distinct signatures so
    // a pathological stream can't grow this without bound; anything past
    // the cap is folded into a single overflow bucket.
    struct GiftagHistEntry
    {
        uint32_t nreg = 0;
        uint64_t regsKey = 0; // REGS nibbles, masked to nreg*4 bits (nreg==16 -> full 64 bits)
        uint8_t flg = 0xFFu;  // 0=PACKED 1=REGLIST 2=IMAGE
        uint64_t tags = 0;
        // Sum of NLOOP, which is the number of REGISTER LOOPS, not the number of
        // vertices. A PACKED tag emits NREG descriptors per loop, so a signature
        // carrying eight XYZF2 per loop contributes eight vertices per loop. This
        // field was previously printed as "verts=", which under-counted such
        // signatures by 8x and led a reader to a share of 3.4% where the true
        // vertex share was 22.4%. It is printed as nloopSum= now; multiply by the
        // count of XYZF2/XYZ2/XYZF3/XYZ3 descriptors in the signature to get
        // vertices.
        uint64_t nloopSum = 0;
    };

    constexpr size_t kGiftagHistCap = 32;
    std::vector<GiftagHistEntry> s_giftagHist;
    uint64_t s_giftagHistOverflowTags = 0;
    uint64_t s_giftagHistOverflowVerts = 0;
    uint64_t s_giftagHistLastPrintMs = 0;

    void giftagHistPrint()
    {
        std::cout << "[gs:giftag-hist] --- t=" << std::dec << dq8ProbeNowMs()
                   << "ms distinct=" << s_giftagHist.size() << " ---" << std::endl;
        for (const auto &e : s_giftagHist)
        {
            std::cout << "[gs:giftag-hist] sig=nreg" << e.nreg << ":" << std::hex;
            for (uint32_t r = 0; r < e.nreg && r < 16u; ++r)
            {
                if (r)
                    std::cout << ',';
                std::cout << "0x" << ((e.regsKey >> (r * 4)) & 0xFu);
            }
            std::cout << std::dec << " flg=" << static_cast<uint32_t>(e.flg)
                       << " tags=" << e.tags << " nloopSum=" << e.nloopSum << std::endl;
        }
        if (s_giftagHistOverflowTags != 0u)
        {
            std::cout << "[gs:giftag-hist] sig=OVERFLOW tags=" << s_giftagHistOverflowTags
                       << " nloopSum=" << s_giftagHistOverflowVerts << std::endl;
        }
    }

    // Single-threaded by construction (the XGKICK case runs on the same
    // interpreter thread that owns every other piece of file-scope state in
    // this TU, e.g. s_vuInUpperOp below), so no locking here matches the
    // rest of this file's debug-probe state.
    void giftagHistNote(uint32_t nreg, uint64_t regsHi, uint8_t flg, uint32_t nloop)
    {
        const uint64_t regsKey = (nreg >= 16u) ? regsHi : (regsHi & ((1ull << (nreg * 4)) - 1ull));

        GiftagHistEntry *match = nullptr;
        for (auto &e : s_giftagHist)
        {
            if (e.nreg == nreg && e.regsKey == regsKey && e.flg == flg)
            {
                match = &e;
                break;
            }
        }
        if (!match)
        {
            if (s_giftagHist.size() < kGiftagHistCap)
            {
                s_giftagHist.push_back(GiftagHistEntry{nreg, regsKey, flg, 0, 0});
                match = &s_giftagHist.back();
            }
            else
            {
                s_giftagHistOverflowTags += 1u;
                s_giftagHistOverflowVerts += nloop;
                return;
            }
        }
        match->tags += 1u;
        match->nloopSum += nloop;

        const uint64_t now = dq8ProbeNowMs();
        if (s_giftagHistLastPrintMs == 0u)
        {
            s_giftagHistLastPrintMs = now;
            return;
        }
        if (now - s_giftagHistLastPrintMs < 5000u)
            return;
        s_giftagHistLastPrintMs = now;
        giftagHistPrint();
    }

    // True only while an FMAC (upper) instruction is executing -- see the MAC
    // flag block in VU1Interpreter::applyDest. File-scope on purpose: adding a
    // member would touch ps2_vu1.h, which ps2_runtime.h includes, which every
    // dq8/output TU includes.
    bool s_vuInUpperOp = false;

    // ---------------------------------------------------------------------
    // VU pipeline model (DC2 spec 04 / G71, G87, G138, G200).
    //
    // Cross-checked against PCSX2 VUops.cpp rather than tuned to a symptom
    // (G197/G198). What PCSX2's interpreter actually delays is NOT the vector
    // register file -- it *stalls the reader* on an FMAC hazard, which yields
    // the same values an immediate write does -- but three things that really
    // are visible late to the program:
    //
    //   * MAC / STATUS / CLIP flags: 4 cycles (_vuFMACflush). Each has an
    //     immediately-updated shadow that successive FMACs and CLIPs chain
    //     off, and an architectural copy that FMAND/FMEQ/FMOR/FCAND/FCOR/
    //     FCEQ/FCGET/FSAND/... read. Committing flags immediately makes a
    //     CLIP;...;FCAND or FMAC;...;FMAND gate read a value one iteration
    //     too new.
    //   * Q: DIV 7, SQRT 7, RSQRT 13 (_vuFDIVAdd). Microcode deliberately
    //     reads the OLDER Q without WAITQ; an immediate write feeds the fresh
    //     quotient into a perspective divide or ST scale a full iteration
    //     early. A second DIV issued while one is pending must busy-stall and
    //     then latch (G200) -- never silently overwrite the pending value.
    //   * P: EFU latencies per op (_vuEFUAdd), WAITP forces the commit.
    //
    // File-scope rather than VU1State members on purpose: widening
    // ps2_vu1.h would rebuild every one of dq8/output's ~7.8k TUs. There is
    // exactly one VU1 and it is driven from the VIF1 interpreter on a single
    // thread, the same assumption s_vuInUpperOp already makes.
    uint64_t s_vuCycle = 0;

    uint32_t s_macShadow = 0u;
    uint32_t s_statusShadow = 0u;
    uint32_t s_clipShadow = 0u;

    struct FlagPipeEntry
    {
        uint64_t due;
        uint32_t mac;
        uint32_t status;
        uint32_t clip;
        bool hasMacStatus;
        bool hasClip;
    };
    constexpr uint32_t kFlagPipeSlots = 8u;
    FlagPipeEntry s_flagPipe[kFlagPipeSlots];
    uint32_t s_flagRd = 0u;
    uint32_t s_flagCount = 0u;

    bool s_qPending = false;
    uint64_t s_qDue = 0u;
    float s_qValue = 0.0f;

    bool s_pPending = false;
    uint64_t s_pDue = 0u;
    float s_pValue = 0.0f;

    // Mirrors every file-scope pipeline-timing global above, byte-for-byte,
    // so save/restoreSharedPipeline can fence one VU1Interpreter instance's
    // use of them from another's. See VU1SharedPipelineSnapshot in
    // ps2_vu1.h for why this exists (VU0 macro mode and the real VU1 share
    // these globals despite being logically independent processors).
    struct SharedPipelineMirror
    {
        uint64_t vuCycle;
        uint32_t macShadow;
        uint32_t statusShadow;
        uint32_t clipShadow;
        FlagPipeEntry flagPipe[kFlagPipeSlots];
        uint32_t flagRd;
        uint32_t flagCount;
        bool qPending;
        uint64_t qDue;
        float qValue;
        bool pPending;
        uint64_t pDue;
        float pValue;
    };
    static_assert(sizeof(SharedPipelineMirror) <= sizeof(VU1SharedPipelineSnapshot),
                  "VU1SharedPipelineSnapshot::opaque is too small for SharedPipelineMirror");

    void vuPipeReset(VU1State &st)
    {
        s_vuCycle = 0u;
        s_flagRd = 0u;
        s_flagCount = 0u;
        s_qPending = false;
        s_pPending = false;
        s_macShadow = st.mac;
        s_statusShadow = st.status;
        s_clipShadow = st.clip;
    }

    // Commit every pipeline entry whose result is architecturally visible at
    // or before `now`.
    void vuPipeAdvance(VU1State &st, uint64_t now)
    {
        while (s_flagCount != 0u)
        {
            const FlagPipeEntry &e = s_flagPipe[s_flagRd];
            if (e.due > now)
                break;
            if (e.hasMacStatus)
            {
                st.mac = e.mac;
                st.status = e.status;
            }
            if (e.hasClip)
                st.clip = e.clip;
            s_flagRd = (s_flagRd + 1u) % kFlagPipeSlots;
            --s_flagCount;
        }
        if (s_qPending && s_qDue <= now)
        {
            st.q = s_qValue;
            s_qPending = false;
        }
        if (s_pPending && s_pDue <= now)
        {
            st.p = s_pValue;
            s_pPending = false;
        }
    }

    void vuPipePushFlags(VU1State &st, uint32_t mac, uint32_t status)
    {
        if (s_flagCount == kFlagPipeSlots)
            vuPipeAdvance(st, s_flagPipe[s_flagRd].due); // ring full: retire the oldest
        const uint32_t wr = (s_flagRd + s_flagCount) % kFlagPipeSlots;
        s_flagPipe[wr] = FlagPipeEntry{s_vuCycle + 4u, mac, status, 0u, true, false};
        ++s_flagCount;
    }

    void vuPipePushClip(VU1State &st, uint32_t clip)
    {
        if (s_flagCount == kFlagPipeSlots)
            vuPipeAdvance(st, s_flagPipe[s_flagRd].due);
        const uint32_t wr = (s_flagRd + s_flagCount) % kFlagPipeSlots;
        s_flagPipe[wr] = FlagPipeEntry{s_vuCycle + 4u, 0u, 0u, clip, false, true};
        ++s_flagCount;
    }

    // G200: issuing a second FDIV while one is in flight busy-stalls until the
    // first latches, then starts the new one -- it must not drop the pending
    // result.
    void vuPipeStartQ(VU1State &st, float value, uint32_t latency)
    {
        if (s_qPending)
        {
            if (s_qDue > s_vuCycle)
                s_vuCycle = s_qDue;
            vuPipeAdvance(st, s_vuCycle);
        }
        s_qPending = true;
        s_qDue = s_vuCycle + latency;
        s_qValue = value;
    }

    void vuPipeStartP(VU1State &st, float value, uint32_t latency)
    {
        if (s_pPending)
        {
            if (s_pDue > s_vuCycle)
                s_vuCycle = s_pDue;
            vuPipeAdvance(st, s_vuCycle);
        }
        s_pPending = true;
        s_pDue = s_vuCycle + latency;
        s_pValue = value;
    }

    void vuPipeWaitQ(VU1State &st)
    {
        if (s_qPending && s_qDue > s_vuCycle)
            s_vuCycle = s_qDue;
        vuPipeAdvance(st, s_vuCycle);
    }

    void vuPipeWaitP(VU1State &st)
    {
        if (s_pPending && s_pDue > s_vuCycle)
            s_vuCycle = s_pDue;
        vuPipeAdvance(st, s_vuCycle);
    }

    // PCSX2 _vuFlushAll(): everything in flight becomes visible when the
    // microprogram ends.
    void vuPipeFlushAll(VU1State &st)
    {
        uint64_t latest = s_vuCycle;
        for (uint32_t k = 0; k < s_flagCount; ++k)
        {
            const uint64_t due = s_flagPipe[(s_flagRd + k) % kFlagPipeSlots].due;
            if (due > latest)
                latest = due;
        }
        if (s_qPending && s_qDue > latest)
            latest = s_qDue;
        if (s_pPending && s_pDue > latest)
            latest = s_pDue;
        s_vuCycle = latest;
        vuPipeAdvance(st, latest);
    }

    // Compact always-on VU1 census, emitted at most once every 5 s. The
    // per-kick / per-run traces are either capped at the first few hundred
    // events (all of which land during the 2-D title) or gated behind
    // PS2X_VIF1_TRACE, whose output is a multi-GB firehose. This answers the
    // one question that matters for "no 3-D primitives reach the GS": are the
    // geometry programs running, are they kicking, and is anything in the
    // packets they kick?
    std::atomic<uint64_t> s_censusRuns{0};
    std::atomic<uint64_t> s_censusPairs{0};
    std::atomic<uint64_t> s_censusEbit{0};
    std::atomic<uint64_t> s_censusMaxCycles{0};
    std::atomic<uint64_t> s_censusKicks{0};
    std::atomic<uint64_t> s_censusKicksEmpty{0};
    std::atomic<uint64_t> s_censusKickBytes{0};
    std::atomic<uint64_t> s_censusKickNloop{0};

    // ---- VU1 data-memory store watch (PS2X_VU1WATCH="<lo>:<hi>") ---------
    // Companion to the VIF-side watch of the same name. Between them they
    // settle the only question that matters about a corrupt transform block:
    // did the EE hand it to us already wrong (VIF unpack) or did the
    // microprogram compute it (SQ/SQI/SQD)? Same 2 s per-quadword rate limit.
    bool vu1StoreWatchEnabled(uint32_t &lo, uint32_t &hi)
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

    void vu1StoreWatch(uint32_t byteAddr, const uint8_t *vuData, uint32_t pc, const char *how)
    {
        uint32_t lo, hi;
        if (!vu1StoreWatchEnabled(lo, hi))
            return;
        const uint32_t qw = byteAddr / 16u;
        if (qw < lo || qw > hi || qw >= 1024u)
            return;
        static uint64_t s_lastMs[1024] = {};
        const uint64_t now = dq8ProbeNowMs();
        if (now - s_lastMs[qw] < 2000u)
            return;
        s_lastMs[qw] = now;
        uint32_t u[4];
        float f[4];
        std::memcpy(u, vuData + byteAddr, sizeof(u));
        std::memcpy(f, vuData + byteAddr, sizeof(f));
        int bad = 0;
        for (int i = 0; i < 4; ++i)
            if (((u[i] >> 23) & 0xFFu) == 0xFFu)
                ++bad;
        std::cerr << "[vu1:watch] t=" << std::dec << now << "ms qw" << qw
                  << " via=" << how << " pc=0x" << std::hex << pc << std::dec
                  << " raw=" << std::hex << u[0] << ',' << u[1] << ',' << u[2] << ',' << u[3]
                  << std::dec
                  << " f=" << f[0] << ',' << f[1] << ',' << f[2] << ',' << f[3]
                  << " exp255lanes=" << bad << std::endl;
    }

    // ---- VU1 state dump probe (PS2X_VU1DUMP=<path prefix>) ---------------
    // The decisive instrument for "VU1 receives a batch but emits nothing":
    // snapshot micro memory + data memory so they can be diffed byte-for-byte
    // against a PCSX2 savestate's vu1MicroMem.bin / vu1Memory.bin, plus a
    // per-entry-PC histogram (which microprogram actually ran, how far it got,
    // how it stopped, whether it kicked).
    const char *vu1DumpPrefix()
    {
        static const char *p = []() -> const char *
        {
            const char *e = std::getenv("PS2X_VU1DUMP");
            return (e && e[0]) ? e : nullptr;
        }();
        return p;
    }

    struct Vu1EntryStat
    {
        uint32_t pc;
        uint64_t runs;
        uint64_t pairs;
        uint64_t kicks;
        uint64_t ebit;
        uint64_t maxcyc;
        uint64_t oob;
    };
    constexpr uint32_t kVu1EntrySlots = 48u;
    Vu1EntryStat s_entryStats[kVu1EntrySlots] = {};
    uint32_t s_entryStatCount = 0u;

    void vu1NoteEntry(uint32_t pc, uint32_t pairs, uint32_t kicks, char stop)
    {
        uint32_t i = 0;
        for (; i < s_entryStatCount; ++i)
            if (s_entryStats[i].pc == pc)
                break;
        if (i == s_entryStatCount)
        {
            if (s_entryStatCount >= kVu1EntrySlots)
                return;
            s_entryStats[i].pc = pc;
            ++s_entryStatCount;
        }
        Vu1EntryStat &s = s_entryStats[i];
        ++s.runs;
        s.pairs += pairs;
        s.kicks += kicks;
        if (stop == 'e')
            ++s.ebit;
        else if (stop == 'm')
            ++s.maxcyc;
        else
            ++s.oob;
    }

    const uint8_t *s_lastCode = nullptr;
    uint32_t s_lastCodeSize = 0u;
    const uint8_t *s_lastData = nullptr;
    uint32_t s_lastDataSize = 0u;

    void vu1DumpSnapshot()
    {
        const char *prefix = vu1DumpPrefix();
        if (!prefix)
            return;
        char path[512];
        if (s_lastCode && s_lastCodeSize)
        {
            std::snprintf(path, sizeof(path), "%s.code.bin", prefix);
            if (FILE *f = std::fopen(path, "wb"))
            {
                std::fwrite(s_lastCode, 1, s_lastCodeSize, f);
                std::fclose(f);
            }
        }
        if (s_lastData && s_lastDataSize)
        {
            std::snprintf(path, sizeof(path), "%s.data.bin", prefix);
            if (FILE *f = std::fopen(path, "wb"))
            {
                std::fwrite(s_lastData, 1, s_lastDataSize, f);
                std::fclose(f);
            }
        }
        for (uint32_t i = 0; i < s_entryStatCount; ++i)
        {
            const Vu1EntryStat &s = s_entryStats[i];
            std::cerr << "[vu1:entry] pc=0x" << std::hex << s.pc << std::dec
                      << " slot=" << (s.pc / 8u)
                      << " runs=" << s.runs
                      << " pairs=" << s.pairs
                      << " avgPairs=" << (s.runs ? (s.pairs / s.runs) : 0ull)
                      << " kicks=" << s.kicks
                      << " ebit=" << s.ebit
                      << " maxcyc=" << s.maxcyc
                      << " oob=" << s.oob
                      << std::endl;
        }
    }

    void vuCensusMaybeReport()
    {
        static std::atomic<uint64_t> s_lastMs{0};
        const uint64_t now = dq8ProbeNowMs();
        uint64_t prev = s_lastMs.load(std::memory_order_relaxed);
        if (now < prev + 5000ull)
            return;
        if (!s_lastMs.compare_exchange_strong(prev, now, std::memory_order_relaxed))
            return;
        vu1DumpSnapshot();
        std::cerr << "[vu1:census] t=" << std::dec << now << "ms"
                  << " runs=" << s_censusRuns.load(std::memory_order_relaxed)
                  << " pairs=" << s_censusPairs.load(std::memory_order_relaxed)
                  << " ebit=" << s_censusEbit.load(std::memory_order_relaxed)
                  << " maxcyc=" << s_censusMaxCycles.load(std::memory_order_relaxed)
                  << " kicks=" << s_censusKicks.load(std::memory_order_relaxed)
                  << " emptyKicks=" << s_censusKicksEmpty.load(std::memory_order_relaxed)
                  << " kickBytes=" << s_censusKickBytes.load(std::memory_order_relaxed)
                  << " kickNloopSum=" << s_censusKickNloop.load(std::memory_order_relaxed)
                  << std::endl;
    }

    // VU1 micro-program run trace (PS2X_VIF1_TRACE=1 -- same gate as the VIF1
    // opcode census, since the two questions are always asked together:
    // "did an MSCAL arrive" and "did the program it started get anywhere").
    // VALUE-tested gate, never getenv()!=nullptr.
    bool vu1TraceOn()
    {
        static const bool on = []()
        {
            const char *e = std::getenv("PS2X_VIF1_TRACE");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        return on;
    }
}

// Instruction field extraction helpers
static inline uint8_t DEST(uint32_t i) { return (uint8_t)((i >> 21) & 0xF); }
static inline uint8_t FT(uint32_t i) { return (uint8_t)((i >> 16) & 0x1F); }
static inline uint8_t FS(uint32_t i) { return (uint8_t)((i >> 11) & 0x1F); }
static inline uint8_t FD(uint32_t i) { return (uint8_t)((i >> 6) & 0x1F); }
static inline uint8_t BC(uint32_t i) { return (uint8_t)(i & 0x3); }

// Lower instruction field helpers
static inline uint8_t LIT(uint32_t i) { return (uint8_t)((i >> 16) & 0x1F); }
static inline uint8_t LIS(uint32_t i) { return (uint8_t)((i >> 11) & 0x1F); }
static inline uint8_t LID(uint32_t i) { return (uint8_t)((i >> 6) & 0x1F); }
static inline int16_t IMM11(uint32_t i) { return (int16_t)(int32_t)((int32_t)(i << 21) >> 21); }
static inline int16_t IMM15(uint32_t i)
{
    uint32_t lo11 = i & 0x7FF;
    uint32_t hi4 = (i >> 21) & 0xF;
    uint32_t raw = (hi4 << 11) | lo11;
    return (int16_t)(int32_t)((int32_t)(raw << 17) >> 17);
}

// ---------------------------------------------------------------------
// VU upper/lower same-cycle hazard detection, ported from upstream
// ran-j/PS2Recomp (ps2xRuntime/src/lib/vu/ps2_vu1_detail.h, carried
// through their PR #120 / #158).
//
// Real VU hardware issues the upper and lower halves of an instruction
// pair in the SAME cycle: both halves read the register file as it stood
// BEFORE the pair started, and only the upper half's write wins if both
// halves target the same register. Our run() loop below instead executes
// execUpper() then execLower() sequentially, so whenever the upper op
// writes a VF register that the lower op also reads or writes, the lower
// observes the upper's brand-new result instead of the previous
// iteration's value. `MADD vfX,... ; SQ vfX,(vi)` -- the software-
// pipelined store idiom a T&L kernel is built from -- is exactly this
// pattern, and it fires in 170 of DQ8's 6453 VU instruction pairs.
//
// vuLowerShouldRunBeforeUpper() is upstream's predicate for "this pair
// needs its lower half run first, so it sees pre-pair state, with the
// upper's write still applied afterward so it keeps priority." These
// four helpers depend only on DEST/FT/FS/FD/LIT/LIS (defined just above)
// and the same `(instr & 3) | ((instr >> 4) & 0x7C)` special-op packing
// our execUpper/execLower already dispatch on.
static inline uint8_t vuUpperVfWriteReg(uint32_t upper)
{
    const uint8_t op = upper & 0x3Fu;
    const uint8_t dest = DEST(upper);
    const uint8_t ft = FT(upper);
    const uint8_t fd = FD(upper);

    if (dest == 0u)
        return 0u;

    if (op <= 0x2Fu)
        return fd;

    if (op >= 0x3Cu)
    {
        const uint8_t specialOp = static_cast<uint8_t>((upper & 0x3u) | ((upper >> 4) & 0x7Cu));
        switch (specialOp)
        {
        // Upper special ops that write a VF register use FT as destination.
        case 0x10: // ITOF0
        case 0x11: // ITOF4
        case 0x12: // ITOF12
        case 0x13: // ITOF15
        case 0x14: // FTOI0
        case 0x15: // FTOI4
        case 0x16: // FTOI12
        case 0x17: // FTOI15
        case 0x1D: // ABS
            return ft;
        default:
            return 0u; // ACC/NOP/CLIP/etc.
        }
    }

    return 0u;
}

static inline void vuSetRegBit(uint32_t &mask, uint8_t reg)
{
    if (reg != 0u && reg < 32u)
        mask |= (1u << reg);
}

static inline void vuLowerVfReadWriteMasks(uint32_t lower, uint32_t &readMask, uint32_t &writeMask)
{
    readMask = 0u;
    writeMask = 0u;

    if (lower == 0u || lower == 0x8000033Cu)
        return;

    const uint8_t it = LIT(lower);
    const uint8_t is = LIS(lower);

    if ((lower & 0x80000000u) != 0u)
    {
        const uint8_t funct = lower & 0x3Fu;
        if (funct >= 0x3Cu && funct <= 0x3Fu)
        {
            const uint8_t specialOp = static_cast<uint8_t>((lower & 0x3u) | ((lower >> 4) & 0x7Cu));
            switch (specialOp)
            {
            case 0x30: // MOVE
            case 0x31: // MR32
                vuSetRegBit(readMask, is);
                vuSetRegBit(writeMask, it);
                return;
            case 0x34: // LQI
            case 0x36: // LQD
                vuSetRegBit(writeMask, it);
                return;
            case 0x35: // SQI
            case 0x37: // SQD
                vuSetRegBit(readMask, is);
                return;
            case 0x38: // DIV
            case 0x3A: // RSQRT
                vuSetRegBit(readMask, is);
                vuSetRegBit(readMask, it);
                return;
            case 0x39: // SQRT
                vuSetRegBit(readMask, it);
                return;
            case 0x3C: // MTIR
            case 0x3E: // ILWR: source base is an integer register, so the
                       // VF read below applies to MTIR only.
                if (specialOp == 0x3C)
                    vuSetRegBit(readMask, is);
                return;
            case 0x3D: // MFIR
            case 0x64: // MFP
                vuSetRegBit(writeMask, it);
                return;
            // EFU block (0x70-0x7E). Upstream's table stops at MFP and
            // falls through to `default: return 0`, which is safe there
            // only because upstream's EFU bodies are stubs. Ours are
            // fully implemented (see the ESADD..EEXP cases in execLower's
            // lower-special dispatch, further below in this file) and
            // every one of them reads vf[is]; WAITP (0x7B) touches no VF
            // register at all and is intentionally left out of this list.
            case 0x70: // ESADD
            case 0x71: // ERSADD
            case 0x72: // ELENG
            case 0x73: // ERLENG
            case 0x74: // EATANxy
            case 0x75: // EATANxz
            case 0x76: // ESUM
            case 0x78: // ESQRT
            case 0x79: // ERSQRT
            case 0x7A: // ERCPR
            case 0x7C: // ESIN
            case 0x7D: // EATAN
            case 0x7E: // EEXP
                vuSetRegBit(readMask, is);
                return;
            default:
                return;
            }
        }
        return;
    }

    switch ((lower >> 25) & 0x7Fu)
    {
    case 0x00: // LQ
        vuSetRegBit(writeMask, it);
        return;
    case 0x01: // SQ
        vuSetRegBit(readMask, is);
        return;
    default:
        return;
    }
}

// WAITQ (lower special 0x3B) and WAITP (0x7B) touch no VF register at all,
// so vuLowerVfReadWriteMasks() correctly reports 0/0 for both and the
// write-overlap predicate below can never fire for them. But hardware
// stalls the WHOLE pair until Q/P settles -- a Q-consuming upper op (e.g.
// MULq in DQ8's VU1 clipper, paired with WAITQ against a 7-cycle DIV
// latency) must see the value WAITQ forces, not the stale one still
// sitting in st.q. Our sequential execUpper-then-execLower default runs
// the upper's read first and only lets WAITQ latch the fresh quotient
// immediately after, so every clipped vertex's position uses the previous
// divide's result while its ST/colour (computed a couple of slots later,
// after WAITQ has advanced the pipe) come out correct.
//
// Hoisting WAITQ/WAITP ahead of the upper is always safe regardless of
// what the upper reads or writes: they have no VF side effect for the
// snapshot/restore dance further down in run() to get wrong, so it
// degrades to "just run the lower first" -- which is exactly the ordering
// hardware's whole-pair stall requires.
static inline bool vuLowerIsWaitQOrP(uint32_t lower)
{
    if ((lower & 0x80000000u) == 0u)
        return false;
    const uint8_t funct = lower & 0x3Fu;
    if (funct < 0x3Cu || funct > 0x3Fu)
        return false;
    const uint8_t specialOp = static_cast<uint8_t>((lower & 0x3u) | ((lower >> 4) & 0x7Cu));
    return specialOp == 0x3Bu || specialOp == 0x7Bu; // WAITQ, WAITP
}

static inline bool vuLowerShouldRunBeforeUpper(uint32_t upper, uint32_t lower)
{
    if (vuLowerIsWaitQOrP(lower))
        return true;

    const uint8_t upperWrite = vuUpperVfWriteReg(upper);
    if (upperWrite == 0u)
        return false;

    uint32_t lowerReads = 0u;
    uint32_t lowerWrites = 0u;
    vuLowerVfReadWriteMasks(lower, lowerReads, lowerWrites);

    const uint32_t upperBit = (1u << upperWrite);
    return ((lowerReads | lowerWrites) & upperBit) != 0u;
}

// Dest-masked write that must NOT touch the MAC/STATUS flags. ITOF/FTOI/ABS
// are the odd ones out in the upper "special" group: PCSX2's applyUnaryFunction
// writes vf[ft] straight through with no VU_MACx_UPDATE / VU_STAT_UPDATE, so
// routing them through applyDest (which now maintains the flags) would corrupt
// every FMAND/FMEQ gate that follows a vertex pack.
static inline void writeDestNoFlags(float *dst, const float *result, uint8_t dest)
{
    if (dest & 0x8)
        dst[0] = result[0];
    if (dest & 0x4)
        dst[1] = result[1];
    if (dest & 0x2)
        dst[2] = result[2];
    if (dest & 0x1)
        dst[3] = result[3];
}

// FTOI{0,4,12,15} -- PCSX2 VUops.cpp floatToInt<Offset>. The saturation clamp
// matters: DQ8's packer feeds FTOI4 with post-perspective coordinates that can
// legitimately exceed 2^31 for behind-camera vertices, and a plain C cast is
// undefined there (x86 yields 0x80000000 for *both* signs, mirroring a vertex
// to the opposite screen edge).
static inline uint32_t vuFloatToInt(float v, uint32_t shift)
{
    if (shift)
    {
        const uint32_t mulBits = 0x3F800000u + (shift << 23);
        float mul;
        std::memcpy(&mul, &mulBits, 4);
        v *= mul;
    }
    uint32_t u;
    std::memcpy(&u, &v, 4);
    if ((u & 0x7F800000u) >= 0x4F000000u)
        return (u & 0x80000000u) ? 0x80000000u : 0x7FFFFFFFu;
    return (uint32_t)(int32_t)v;
}

// ITOF{0,4,12,15} -- PCSX2 VUops.cpp intToFloat<Offset>.
static inline float vuIntToFloat(uint32_t bits, uint32_t shift)
{
    float f = (float)(int32_t)bits;
    if (shift)
    {
        const uint32_t mulBits = 0x3F800000u - (shift << 23);
        float mul;
        std::memcpy(&mul, &mulBits, 4);
        f *= mul;
    }
    return f;
}

VU1Interpreter::VU1Interpreter()
{
    reset();
}

void VU1Interpreter::reset()
{
    std::memset(&m_state, 0, sizeof(m_state));
    m_state.vf[0][3] = 1.0f; // VF0.w = 1.0
    m_state.q = 1.0f;
    vuPipeReset(m_state);
}

VU1SharedPipelineSnapshot VU1Interpreter::saveSharedPipeline() const
{
    VU1SharedPipelineSnapshot snap;
    SharedPipelineMirror mirror;
    mirror.vuCycle = s_vuCycle;
    mirror.macShadow = s_macShadow;
    mirror.statusShadow = s_statusShadow;
    mirror.clipShadow = s_clipShadow;
    std::memcpy(mirror.flagPipe, s_flagPipe, sizeof(s_flagPipe));
    mirror.flagRd = s_flagRd;
    mirror.flagCount = s_flagCount;
    mirror.qPending = s_qPending;
    mirror.qDue = s_qDue;
    mirror.qValue = s_qValue;
    mirror.pPending = s_pPending;
    mirror.pDue = s_pDue;
    mirror.pValue = s_pValue;
    std::memcpy(snap.opaque, &mirror, sizeof(mirror));
    return snap;
}

void VU1Interpreter::restoreSharedPipeline(const VU1SharedPipelineSnapshot &snap)
{
    SharedPipelineMirror mirror;
    std::memcpy(&mirror, snap.opaque, sizeof(mirror));
    s_vuCycle = mirror.vuCycle;
    s_macShadow = mirror.macShadow;
    s_statusShadow = mirror.statusShadow;
    s_clipShadow = mirror.clipShadow;
    std::memcpy(s_flagPipe, mirror.flagPipe, sizeof(s_flagPipe));
    s_flagRd = mirror.flagRd;
    s_flagCount = mirror.flagCount;
    s_qPending = mirror.qPending;
    s_qDue = mirror.qDue;
    s_qValue = mirror.qValue;
    s_pPending = mirror.pPending;
    s_pDue = mirror.pDue;
    s_pValue = mirror.pValue;
}

void VU1Interpreter::resyncPipelineShadow()
{
    // Re-seed the shadow flags from THIS instance's current (now fully
    // populated) m_state.mac/status/clip. reset() already called
    // vuPipeReset once, but that was before the caller copied real entry
    // state (e.g. copyVu0ContextToState) into m_state, so the shadow was
    // seeded from a still-zeroed state. Calling vuPipeReset again here,
    // now that mac/status/clip hold the true entry values, makes the
    // shadow agree with the architectural flags a program actually reads
    // at PC 0.
    vuPipeReset(m_state);
}

float VU1Interpreter::broadcast(const float *vf, uint8_t bc)
{
    return vf[bc & 3];
}

void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
{
    // MAC / STATUS flag update (DC2 spec 04 / G71: "MAC/STATUS flags must
    // actually be computed"). m_state.mac was declared and read by
    // FMEQ/FMAND/FMOR but NEVER written, so it was permanently 0 and every
    // MAC-gated branch in Level-5's VU1 microcode evaluated against the wrong
    // constant. Observed effect in DQ8: the geometry program entered at
    // VU1 pc 0x960 loops `FMAND vi8,vi6` / `IBEQ vi8,vi0,-22` forever, so it
    // never reaches its XGKICK.
    //
    // Only FMAC (upper) instructions set the flags. applyDest is also used by
    // lower ops (LQ/MOVE/MFIR/...) which must not touch them, so the upper-op
    // window is marked by s_vuInUpperOp rather than by adding a member (that
    // would touch ps2_vu1.h -> ps2_runtime.h -> ~7.8k output TUs).
    //
    // MAC layout (PS2 VU, 16 bits): O=15..12, U=11..8, S=7..4, Z=3..0, each
    // group ordered x,y,z,w from the high bit down. Lanes outside DEST read 0.
    //
    // The flag update is also where the VU's non-IEEE result conditioning
    // happens, and the conditioned value is what gets WRITTEN -- not just
    // flagged. PCSX2 VUops.cpp VU_MACx_UPDATE returns the rewritten bits:
    //     if (exp == 0)   { macflag = (macflag & ~0x1100) | 0x0001; return s; }
    //     if (exp == 255) { macflag = (macflag & ~0x1001) | 0x0100;
    //                       return s | 0x7f7fffff; }
    // i.e. an overflowing result becomes +/-Fmax and a denormal becomes signed
    // zero. This VU has no infinity and no NaN encoding, so leaving a host
    // Inf/NaN in vf or ACC hands the microprogram a value its own arithmetic
    // could never have produced, and it then spreads through every subsequent
    // FMAC, FMAND gate and vertex pack.
    float conditioned[4];
    if (s_vuInUpperOp)
    {
        uint32_t mac = 0u;
        for (int c = 0; c < 4; ++c)
        {
            conditioned[c] = result[c];
            if (!(dest & (0x8u >> c)))
                continue;
            const uint32_t bit = static_cast<uint32_t>(3 - c);
            const float v = result[c];
            uint32_t bits = 0u;
            std::memcpy(&bits, &v, sizeof(bits));
            if ((bits & 0x7FFFFFFFu) == 0u)
                mac |= (1u << bit); // Z
            if (bits & 0x80000000u)
                mac |= (1u << (4u + bit)); // S
            const uint32_t exp = (bits >> 23) & 0xFFu;
            if (exp == 0xFFu)
            {
                mac |= (1u << (12u + bit)); // O (VU saturates; inf/nan == overflow)
                const uint32_t sat = (bits & 0x80000000u) | 0x7F7FFFFFu;
                std::memcpy(&conditioned[c], &sat, sizeof(sat));
            }
            else if (exp == 0u && (bits & 0x7FFFFFu) != 0u)
            {
                mac |= (1u << (8u + bit)); // U (denormal)
                const uint32_t flushed = bits & 0x80000000u;
                std::memcpy(&conditioned[c], &flushed, sizeof(flushed));
            }
        }
        result = conditioned;
        // STATUS: bits 3..0 = current O,U,S,Z; bits 9..6 = sticky OS,US,SS,ZS.
        uint32_t cur = 0u;
        if (mac & 0x000Fu)
            cur |= 0x1u; // Z
        if (mac & 0x00F0u)
            cur |= 0x2u; // S
        if (mac & 0x0F00u)
            cur |= 0x4u; // U
        if (mac & 0xF000u)
            cur |= 0x8u; // O
        const uint32_t sticky = (s_statusShadow | (cur << 6)) & 0x3C0u;

        // Shadow updates immediately (successive FMACs chain off it); the
        // architectural MAC/STATUS become visible 4 cycles later, which is
        // what FMAND/FMEQ/FMOR/FSAND read.
        s_macShadow = mac;
        s_statusShadow = (s_statusShadow & ~0x3CFu) | cur | sticky;
        vuPipePushFlags(m_state, s_macShadow, s_statusShadow);
    }

    if (dest & 0x8)
        dst[0] = result[0]; // x
    if (dest & 0x4)
        dst[1] = result[1]; // y
    if (dest & 0x2)
        dst[2] = result[2]; // z
    if (dest & 0x1)
        dst[3] = result[3]; // w
}

void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

void VU1Interpreter::execute(uint8_t *vuCode, uint32_t codeSize,
                             uint8_t *vuData, uint32_t dataSize,
                             GS &gs, PS2Memory *memory,
                             uint32_t startPC, uint32_t itop,
                             uint32_t maxCycles)
{
    m_state.pc = startPC;
    m_state.ebit = false;
    m_state.itop = itop;
    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::resume(uint8_t *vuCode, uint32_t codeSize,
                            uint8_t *vuData, uint32_t dataSize,
                            GS &gs, PS2Memory *memory,
                            uint32_t itop, uint32_t maxCycles)
{
    m_state.ebit = false;
    m_state.itop = itop;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::run(uint8_t *vuCode, uint32_t codeSize,
                         uint8_t *vuData, uint32_t dataSize,
                         GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    const bool trace = vu1TraceOn();
    const uint32_t entryPc = m_state.pc;
    const uint32_t xgkickAtEntry = s_debugVu1XgkickCount.load(std::memory_order_relaxed);
    uint32_t executed = 0u;
    const char *stopReason = "maxCycles";
    // Does the microprogram image even contain anything at the entry point?
    // A zeroed code page is the signature of "MPG never landed", which is a
    // different failure from "MSCAL never arrived".
    uint32_t entryWords[2] = {0u, 0u};
    if (vuCode && entryPc + 8u <= codeSize)
        std::memcpy(entryWords, vuCode + entryPc, 8);

    // One-slot branch delay (DC2 spec 04 / G22). VU branch and jump
    // instructions take effect only AFTER the following 64-bit instruction
    // pair has executed. The previous "simplified branch delay" wrote the
    // target straight into m_state.pc, which SKIPPED the delay slot entirely
    // -- and Level-5 microcode routinely puts the loop-counter update
    // (IADDIU/ISUBIU) in that slot, so every counted loop ran forever.
    //
    // The branch state is kept local to run() on purpose: adding fields to
    // VU1State would touch ps2_vu1.h, which ps2_runtime.h includes, which
    // every one of dq8/output's ~7.8k translation units includes. A branch is
    // detected by observing that the lower op moved m_state.pc off the value
    // it was fetched from (the branch cases write `target - 8`).
    bool branchPending = false;
    uint32_t branchTarget = 0u;

    // Runaway-loop autopsy: remember the last kRing pcs so a maxCycles stop
    // can print the actual loop body instead of just its length.
    static constexpr uint32_t kRing = 64u;
    uint32_t ringPc[kRing];
    uint32_t ringN = 0u;
    const bool wantRing = trace;

    for (uint32_t cycle = 0; cycle < maxCycles; ++cycle)
    {
        if (m_state.pc + 8 > codeSize)
        {
            stopReason = "pcOutOfRange";
            break;
        }
        ++executed;

        // One VU cycle per 64-bit instruction pair (VU1 is dual-issue), then
        // retire anything whose latency has expired -- PCSX2 calls
        // _vuTestPipes before each instruction for exactly this reason.
        ++s_vuCycle;
        vuPipeAdvance(m_state, s_vuCycle);

        const uint32_t pcBefore = m_state.pc;
        if (wantRing)
            ringPc[(ringN++) % kRing] = pcBefore;
        uint32_t lower, upper;
        std::memcpy(&lower, vuCode + pcBefore, 4);
        std::memcpy(&upper, vuCode + pcBefore + 4, 4);

        bool eBit = (upper >> 30) & 1;

        // LOI (DC2 spec 04 / G27): signalled by the UPPER word's I-bit
        // (bit 31). When set, the LOWER word is a 32-bit float immediate
        // loaded into the I register and is NOT decoded as a lower
        // instruction -- but the UPPER instruction still executes.
        //
        // The previous rule keyed on `lower == 0x8000033C`, which is not the
        // LOI encoding at all: it is the canonical VU *lower NOP*. So every
        // ordinary "<real upper op> ; NOP" pair -- the most common pair in
        // any hand-scheduled VU program -- had its upper FMAC silently
        // dropped and clobbered I with the upper word's bits.
        const bool loi = (upper & 0x80000000u) != 0u;
        if (loi)
        {
            // VU User's Manual 3.1.4: "the content of the Lower OP field is
            // written to the I register at the T stage of the instruction...
            // It is used by the NEXT instruction to be executed, such as
            // ADDi/MULi." So this pair's own upper op must still observe the
            // OLD I value (whatever a prior LOI staged); the new immediate
            // only becomes visible starting with the next executed pair --
            // which, thanks to the delay-slot handling above, is correct
            // whether that next pair is a fall-through, a branch delay slot,
            // or a branch target: the write below happens once per iteration,
            // unconditionally, before nextPC is computed, so it lands in
            // program order regardless of control flow.
            //
            // The previous code wrote m_state.i BEFORE calling execUpper(),
            // so the LOI pair's own upper op (e.g. MULAi in DQ8's Euler->
            // matrix microprogram) illegally saw the brand-new immediate
            // instead of the previous one -- feeding sin/cos range reduction
            // a value like 12582912.0 instead of +/-1/2pi.
            s_vuInUpperOp = true;
            execUpper(upper);
            s_vuInUpperOp = false;
            std::memcpy(&m_state.i, &lower, 4);
        }
        else if (vuLowerShouldRunBeforeUpper(upper, lower))
        {
            // Same-cycle hazard (see the comment on vuLowerShouldRunBeforeUpper
            // above): the upper op writes a VF register this pair's lower op
            // also reads or writes. Real hardware issues both halves off the
            // SAME pre-pair register file, so run the lower half first --
            // it then sees the correct pre-pair value -- and the upper half
            // second so its write still lands with priority, matching
            // upstream ran-j/PS2Recomp's ordering.
            //
            // That reordering alone mishandles one more case: if the upper
            // op ALSO reads (as fs/ft) a VF register that the lower op
            // writes, neither execution order is correct on its own -- the
            // upper needs the pre-pair value of that register, but running
            // lower first just clobbered it. Snapshot the upper op's own
            // source registers before the lower op runs, substitute the
            // snapshot back in immediately before the upper op runs, then
            // restore the lower op's real result afterward -- unless the
            // upper op's own destination is that same register, in which
            // case the upper's write correctly wins and nothing is restored.
            const uint8_t upperFs = FS(upper);
            const uint8_t upperFt = FT(upper);
            const uint8_t upperWriteReg = vuUpperVfWriteReg(upper);

            uint32_t lowerReads = 0u;
            uint32_t lowerWrites = 0u;
            vuLowerVfReadWriteMasks(lower, lowerReads, lowerWrites);
            (void)lowerReads; // only the write mask matters for this snapshot

            const bool snapFs = upperFs != 0u && ((lowerWrites >> upperFs) & 1u) != 0u;
            const bool snapFt = upperFt != 0u && ((lowerWrites >> upperFt) & 1u) != 0u;

            float preFs[4], preFt[4];
            if (snapFs)
                std::memcpy(preFs, m_state.vf[upperFs], 16);
            if (snapFt)
                std::memcpy(preFt, m_state.vf[upperFt], 16);

            execLower(lower, vuData, dataSize, gs, memory, upper);

            float postLowerFs[4], postLowerFt[4];
            if (snapFs)
                std::memcpy(postLowerFs, m_state.vf[upperFs], 16);
            if (snapFt)
                std::memcpy(postLowerFt, m_state.vf[upperFt], 16);

            if (snapFs)
                std::memcpy(m_state.vf[upperFs], preFs, 16);
            if (snapFt)
                std::memcpy(m_state.vf[upperFt], preFt, 16);

            s_vuInUpperOp = true;
            execUpper(upper);
            s_vuInUpperOp = false;

            if (snapFs && upperFs != upperWriteReg)
                std::memcpy(m_state.vf[upperFs], postLowerFs, 16);
            if (snapFt && upperFt != upperWriteReg)
                std::memcpy(m_state.vf[upperFt], postLowerFt, 16);
        }
        else
        {
            s_vuInUpperOp = true;
            execUpper(upper);
            s_vuInUpperOp = false;
            execLower(lower, vuData, dataSize, gs, memory, upper);
        }

        // Enforce VF0 invariant
        m_state.vf[0][0] = 0.0f;
        m_state.vf[0][1] = 0.0f;
        m_state.vf[0][2] = 0.0f;
        m_state.vf[0][3] = 1.0f;
        // Enforce VI0 invariant
        m_state.vi[0] = 0;

        uint32_t nextPC;
        if (m_state.pc != pcBefore)
        {
            // A branch/jump was taken by this pair's lower op. Arm it and run
            // the delay slot (the pair right after the branch) first.
            branchTarget = (m_state.pc + 8u) & 0x3FFFu;
            branchPending = true;
            nextPC = pcBefore + 8u;
        }
        else if (branchPending)
        {
            branchPending = false;
            nextPC = branchTarget;
        }
        else
        {
            nextPC = pcBefore + 8u;
        }
        if (nextPC >= codeSize)
            nextPC = 0;
        m_state.pc = nextPC;

        if (m_state.ebit)
        {
            stopReason = "ebit";
            break;
        }

        if (eBit)
            m_state.ebit = true;
    }

    // PCSX2 _vuFlushAll(): the FMAC-flag, FDIV(Q) and EFU(P) pipes all become
    // visible when the microprogram stops, so the next program (and any EE-side
    // reader of VU1 state) sees settled values.
    vuPipeFlushAll(m_state);

    // Autopsy for a microprogram that either ran away (maxCycles) or finished
    // without ever reaching an XGKICK -- both mean no geometry came out.
    s_censusRuns.fetch_add(1, std::memory_order_relaxed);
    s_censusPairs.fetch_add(executed, std::memory_order_relaxed);
    if (stopReason[0] == 'e')
        s_censusEbit.fetch_add(1, std::memory_order_relaxed);
    else if (stopReason[0] == 'm')
        s_censusMaxCycles.fetch_add(1, std::memory_order_relaxed);
    if (vu1DumpPrefix())
    {
        s_lastCode = vuCode;
        s_lastCodeSize = codeSize;
        s_lastData = vuData;
        s_lastDataSize = dataSize;
        vu1NoteEntry(entryPc, executed,
                     s_debugVu1XgkickCount.load(std::memory_order_relaxed) - xgkickAtEntry,
                     stopReason[0]);
    }
    vuCensusMaybeReport();

    const uint32_t kicksThisRun = s_debugVu1XgkickCount.load(std::memory_order_relaxed) - xgkickAtEntry;
    if (trace && (stopReason[0] == 'm' || (kicksThisRun == 0u && executed >= 8u)))
    {
        static std::atomic<uint64_t> s_runawayLogged{0};
        const uint64_t rn = s_runawayLogged.fetch_add(1, std::memory_order_relaxed);
        if (rn < 6u && vuCode)
        {
            const uint32_t have = (ringN < kRing) ? ringN : kRing;
            const uint32_t base = (ringN < kRing) ? 0u : (ringN % kRing);
            std::cerr << "[vu1:runaway] n=" << std::dec << rn
                      << " entryPc=0x" << std::hex << entryPc << std::dec
                      << " executed=" << executed
                      << " stop=" << stopReason
                      << " lastPcs:" << std::endl;
            for (uint32_t k = 0; k < have; ++k)
            {
                const uint32_t pc = ringPc[(base + k) % kRing];
                uint32_t lo = 0u, up = 0u;
                if (pc + 8u <= codeSize)
                {
                    std::memcpy(&lo, vuCode + pc, 4);
                    std::memcpy(&up, vuCode + pc + 4, 4);
                }
                std::cerr << "[vu1:runaway]   pc=0x" << std::hex << pc
                          << " lo=0x" << lo << " up=0x" << up << std::dec << std::endl;
            }
            std::cerr << "[vu1:runaway]   vi:";
            for (int r = 0; r < 16; ++r)
                std::cerr << " " << std::dec << (int16_t)m_state.vi[r];
            std::cerr << std::endl;
        }
    }

    if (trace)
    {
        static std::atomic<uint64_t> s_runLogged{0};
        const uint64_t n = s_runLogged.fetch_add(1, std::memory_order_relaxed);
        const uint32_t kicks = s_debugVu1XgkickCount.load(std::memory_order_relaxed) - xgkickAtEntry;
        if (n < 200u || (n % 4096u) == 0u)
        {
            std::cerr << "[vu1:run] n=" << std::dec << n
                      << " entryPc=0x" << std::hex << entryPc
                      << " word0=0x" << entryWords[0]
                      << " word1=0x" << entryWords[1]
                      << std::dec
                      << " executed=" << executed
                      << " stop=" << stopReason
                      << " kicks=" << kicks
                      << std::endl;
        }
    }
}

// ============================================================================
// Upper instructions (FMAC pipeline)
// ============================================================================
void VU1Interpreter::execUpper(uint32_t instr)
{
    uint8_t dest = DEST(instr);
    uint8_t ft = FT(instr);
    uint8_t fs = FS(instr);
    uint8_t fd = FD(instr);
    uint8_t op = instr & 0x3F;

    float *vd = m_state.vf[fd];
    const float *vs = m_state.vf[fs];
    const float *vt = m_state.vf[ft];
    float result[4];

    // Upper opcode decoding (bits 5:0 of upper word)
    switch (op)
    {
    case 0x00:
    case 0x01:
    case 0x02:
    case 0x03: // ADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x07: // SUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x08:
    case 0x09:
    case 0x0A:
    case 0x0B: // MADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = m_state.acc[c] + vs[c] * bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x0C:
    case 0x0D:
    case 0x0E:
    case 0x0F: // MSUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = m_state.acc[c] - vs[c] * bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13: // MAXbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > bc) ? vs[c] : bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x17: // MINIbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < bc) ? vs[c] : bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B: // MULbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x1C: // MULq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * m_state.q;
        applyDest(vd, result, dest);
        return;
    case 0x1D: // MAXi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > m_state.i) ? vs[c] : m_state.i;
        applyDest(vd, result, dest);
        return;
    case 0x1E: // MULi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * m_state.i;
        applyDest(vd, result, dest);
        return;
    case 0x1F: // MINIi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < m_state.i) ? vs[c] : m_state.i;
        applyDest(vd, result, dest);
        return;
    case 0x20: // ADDq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + m_state.q;
        applyDest(vd, result, dest);
        return;
    case 0x21: // MADDq
        for (int c = 0; c < 4; c++)
            result[c] = m_state.acc[c] + vs[c] * m_state.q;
        applyDest(vd, result, dest);
        return;
    case 0x22: // ADDi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + m_state.i;
        applyDest(vd, result, dest);
        return;
    case 0x23: // MADDi
        for (int c = 0; c < 4; c++)
            result[c] = m_state.acc[c] + vs[c] * m_state.i;
        applyDest(vd, result, dest);
        return;
    case 0x24: // SUBq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - m_state.q;
        applyDest(vd, result, dest);
        return;
    case 0x25: // MSUBq
        for (int c = 0; c < 4; c++)
            result[c] = m_state.acc[c] - vs[c] * m_state.q;
        applyDest(vd, result, dest);
        return;
    case 0x26: // SUBi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - m_state.i;
        applyDest(vd, result, dest);
        return;
    case 0x27: // MSUBi
        for (int c = 0; c < 4; c++)
            result[c] = m_state.acc[c] - vs[c] * m_state.i;
        applyDest(vd, result, dest);
        return;
    case 0x28: // ADD
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + vt[c];
        applyDest(vd, result, dest);
        return;
    case 0x29: // MADD
        for (int c = 0; c < 4; c++)
            result[c] = m_state.acc[c] + vs[c] * vt[c];
        applyDest(vd, result, dest);
        return;
    case 0x2A: // MUL
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * vt[c];
        applyDest(vd, result, dest);
        return;
    case 0x2B: // MAX
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;
    case 0x2C: // SUB
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - vt[c];
        applyDest(vd, result, dest);
        return;
    case 0x2D: // MSUB
        for (int c = 0; c < 4; c++)
            result[c] = m_state.acc[c] - vs[c] * vt[c];
        applyDest(vd, result, dest);
        return;
    case 0x2E: // OPMSUB
        result[0] = m_state.acc[0] - vs[1] * vt[2];
        result[1] = m_state.acc[1] - vs[2] * vt[0];
        result[2] = m_state.acc[2] - vs[0] * vt[1];
        result[3] = 0.0f;
        applyDest(vd, result, dest);
        return;
    case 0x2F: // MINI
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;

    // ------------------------------------------------------------------
    // Upper "special" group (bits 5:2 == 0b1111).
    //
    // DC2 spec 04 / G138 -- opcode table cross-checked against PCSX2 master
    // VUops.cpp (_UPPER_FD_00/01/10/11_TABLE, each indexed by (code>>6)&0x1f
    // and selected by code&3). Flattening those four 32-entry tables gives a
    // single index
    //        uop = ((instr >> 6) & 0x1F) * 4 + (instr & 3)
    //            = (instr & 3) | ((instr >> 4) & 0x7C)
    // which is exactly the packing PCSX2 already uses for the LOWER special
    // group. Two independent checks fall out of it: the canonical upper NOP
    // 0x2FF lands on uop 0x2F, and CLIP 0x1FF lands on uop 0x1F.
    //
    // The previous code dispatched on `instr & 0x3F` (0x3C..0x3F) and then
    // used bits 10:6 as a *flat* index -- i.e. it used the concatenated table
    // index as if it were the per-table index. Consequences, all silent:
    //   * only ADDAx decoded correctly; MULAx ran as SUBAz, MADDAx as ADDAz,
    //     MADDAy/MADDAz/MULAy/MULAz/MULAw were unreachable;
    //   * ITOF*/FTOI*/ABS/CLIP were unreachable (their true uop values 0x10
    //     ..0x1F need instr&0x3F in 0x3C..0x3F, but the old 0x3C arm demanded
    //     bits 10:6 == 0x10..0x1F, which is OPMULA/NOP territory);
    //   * the whole `case 0x3E` / `case 0x3F` quarter returned unconditionally,
    //     which is why CLIP never wrote m_state.clip and every FCAND/FCOR gate
    //     evaluated against 0.
    // Level-5's vertex transform is the canonical
    //     MULAx ACC,mtx0,v.x / MADDAy / MADDAz / MADDw vout
    // and its packer ends in FTOI4, so before this fix essentially every
    // transformed vertex was arithmetic garbage even though geometry flowed.
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    {
        const uint32_t uop = (uint32_t)(instr & 3u) | ((instr >> 4) & 0x7Cu);
        switch (uop)
        {
        case 0x00:
        case 0x01:
        case 0x02:
        case 0x03: // ADDAbc
        {
            float bc = broadcast(vt, uop & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + bc;
            applyDestAcc(result, dest);
            return;
        }
        case 0x04:
        case 0x05:
        case 0x06:
        case 0x07: // SUBAbc
        {
            float bc = broadcast(vt, uop & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - bc;
            applyDestAcc(result, dest);
            return;
        }
        case 0x08:
        case 0x09:
        case 0x0A:
        case 0x0B: // MADDAbc
        {
            float bc = broadcast(vt, uop & 3);
            for (int c = 0; c < 4; c++)
                result[c] = m_state.acc[c] + vs[c] * bc;
            applyDestAcc(result, dest);
            return;
        }
        case 0x0C:
        case 0x0D:
        case 0x0E:
        case 0x0F: // MSUBAbc
        {
            float bc = broadcast(vt, uop & 3);
            for (int c = 0; c < 4; c++)
                result[c] = m_state.acc[c] - vs[c] * bc;
            applyDestAcc(result, dest);
            return;
        }
        // ITOF / FTOI / ABS: destination is ft (DC2 spec 04 / G32), source fs,
        // and they do NOT update MAC/STATUS (PCSX2 applyUnaryFunction).
        case 0x10:
        case 0x11:
        case 0x12:
        case 0x13: // ITOF0 / ITOF4 / ITOF12 / ITOF15
        {
            static const uint32_t kShift[4] = {0u, 4u, 12u, 15u};
            if (ft == 0)
                return;
            for (int c = 0; c < 4; c++)
            {
                uint32_t iv;
                std::memcpy(&iv, &vs[c], 4);
                result[c] = vuIntToFloat(iv, kShift[uop & 3]);
            }
            writeDestNoFlags(m_state.vf[ft], result, dest);
            return;
        }
        case 0x14:
        case 0x15:
        case 0x16:
        case 0x17: // FTOI0 / FTOI4 / FTOI12 / FTOI15
        {
            static const uint32_t kShift[4] = {0u, 4u, 12u, 15u};
            if (ft == 0)
                return;
            for (int c = 0; c < 4; c++)
            {
                const uint32_t iv = vuFloatToInt(vs[c], kShift[uop & 3]);
                std::memcpy(&result[c], &iv, 4);
            }
            writeDestNoFlags(m_state.vf[ft], result, dest);
            return;
        }
        case 0x18:
        case 0x19:
        case 0x1A:
        case 0x1B: // MULAbc
        {
            float bc = broadcast(vt, uop & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * bc;
            applyDestAcc(result, dest);
            return;
        }
        case 0x1C: // MULAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * m_state.q;
            applyDestAcc(result, dest);
            return;
        case 0x1D: // ABS
        {
            if (ft == 0)
                return;
            for (int c = 0; c < 4; c++)
            {
                uint32_t bits;
                std::memcpy(&bits, &vs[c], 4);
                bits &= 0x7FFFFFFFu;
                std::memcpy(&result[c], &bits, 4);
            }
            writeDestNoFlags(m_state.vf[ft], result, dest);
            return;
        }
        case 0x1E: // MULAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * m_state.i;
            applyDestAcc(result, dest);
            return;
        case 0x1F: // CLIP
        {
            // PCSX2 _vuCLIP: compare against |ft.w|, 6 new bits shifted into a
            // 24-bit (3-deep) clip register. Nothing else writes m_state.clip,
            // so FCAND/FCOR/FCEQ were reading a permanently-zero register.
            const float w = std::fabs(vt[3]);
            uint32_t flags = 0u;
            if (vs[0] > +w)
                flags |= 0x01u;
            if (vs[0] < -w)
                flags |= 0x02u;
            if (vs[1] > +w)
                flags |= 0x04u;
            if (vs[1] < -w)
                flags |= 0x08u;
            if (vs[2] > +w)
                flags |= 0x10u;
            if (vs[2] < -w)
                flags |= 0x20u;
            s_clipShadow = ((s_clipShadow << 6) | flags) & 0xFFFFFFu;
            vuPipePushClip(m_state, s_clipShadow);
            return;
        }
        case 0x20: // ADDAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + m_state.q;
            applyDestAcc(result, dest);
            return;
        case 0x21: // MADDAq
            for (int c = 0; c < 4; c++)
                result[c] = m_state.acc[c] + vs[c] * m_state.q;
            applyDestAcc(result, dest);
            return;
        case 0x22: // ADDAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + m_state.i;
            applyDestAcc(result, dest);
            return;
        case 0x23: // MADDAi
            for (int c = 0; c < 4; c++)
                result[c] = m_state.acc[c] + vs[c] * m_state.i;
            applyDestAcc(result, dest);
            return;
        case 0x24: // SUBAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - m_state.q;
            applyDestAcc(result, dest);
            return;
        case 0x25: // MSUBAq
            for (int c = 0; c < 4; c++)
                result[c] = m_state.acc[c] - vs[c] * m_state.q;
            applyDestAcc(result, dest);
            return;
        case 0x26: // SUBAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - m_state.i;
            applyDestAcc(result, dest);
            return;
        case 0x27: // MSUBAi
            for (int c = 0; c < 4; c++)
                result[c] = m_state.acc[c] - vs[c] * m_state.i;
            applyDestAcc(result, dest);
            return;
        case 0x28: // ADDA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + vt[c];
            applyDestAcc(result, dest);
            return;
        case 0x29: // MADDA
            for (int c = 0; c < 4; c++)
                result[c] = m_state.acc[c] + vs[c] * vt[c];
            applyDestAcc(result, dest);
            return;
        case 0x2A: // MULA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * vt[c];
            applyDestAcc(result, dest);
            return;
        case 0x2C: // SUBA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - vt[c];
            applyDestAcc(result, dest);
            return;
        case 0x2D: // MSUBA
            for (int c = 0; c < 4; c++)
                result[c] = m_state.acc[c] - vs[c] * vt[c];
            applyDestAcc(result, dest);
            return;
        case 0x2E: // OPMULA
            result[0] = vs[1] * vt[2];
            result[1] = vs[2] * vt[0];
            result[2] = vs[0] * vt[1];
            result[3] = 0.0f;
            applyDestAcc(result, dest);
            return;
        case 0x2F: // NOP
            return;
        default:
            return;
        }
    }

    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33: // iadd-like upper? No, these are valid upper ops
    default:
        // NOP / unimplemented upper
        return;
    }
}

// ============================================================================
// Lower instructions
// ============================================================================
void VU1Interpreter::execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr)
{
    (void)upperInstr;
    if (instr == 0x00000000 || instr == 0x8000033C) // NOP
        return;

    uint8_t opHi = (instr >> 25) & 0x7F;

    // The lower instruction encoding uses bits 31:25 for the primary opcode
    switch (opHi)
    {
    case 0x00: // LQ (Load Quadword from VU data memory)
    {
        uint8_t it = LIT(instr);
        uint8_t is = LIS(instr);
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            float tmp[4];
            std::memcpy(tmp, vuData + addr, 16);
            applyDest(m_state.vf[it], tmp, dest);
        }
        return;
    }
    case 0x01: // SQ (Store Quadword to VU data memory)
    {
        uint8_t is = LIS(instr);
        uint8_t it = LIT(instr);
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[it] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            float tmp[4];
            std::memcpy(tmp, vuData + addr, 16);
            if (dest & 0x8)
                tmp[0] = m_state.vf[is][0];
            if (dest & 0x4)
                tmp[1] = m_state.vf[is][1];
            if (dest & 0x2)
                tmp[2] = m_state.vf[is][2];
            if (dest & 0x1)
                tmp[3] = m_state.vf[is][3];
            std::memcpy(vuData + addr, tmp, 16);
            vu1StoreWatch(addr, vuData, m_state.pc, "vu1-sq");
        }
        return;
    }
    case 0x04: // ILW (Integer Load Word from VU data memory)
    {
        uint8_t it = LIT(instr);
        uint8_t is = LIS(instr);
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            int comp = 0;
            if (dest & 0x8)
                comp = 0;
            else if (dest & 0x4)
                comp = 1;
            else if (dest & 0x2)
                comp = 2;
            else
                comp = 3;
            uint32_t v;
            std::memcpy(&v, vuData + addr + comp * 4, 4);
            if (it != 0)
                m_state.vi[it] = (int32_t)(int16_t)(v & 0xFFFF);
        }
        return;
    }
    case 0x05: // ISW (Integer Store Word to VU data memory)
    {
        uint8_t it = LIT(instr);
        uint8_t is = LIS(instr);
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            uint32_t val = (uint32_t)(uint16_t)(m_state.vi[it] & 0xFFFF);
            if (dest & 0x8)
                std::memcpy(vuData + addr + 0, &val, 4);
            if (dest & 0x4)
                std::memcpy(vuData + addr + 4, &val, 4);
            if (dest & 0x2)
                std::memcpy(vuData + addr + 8, &val, 4);
            if (dest & 0x1)
                std::memcpy(vuData + addr + 12, &val, 4);
        }
        return;
    }
    case 0x08: // IADDIU
    {
        uint8_t it = LIT(instr);
        uint8_t is = LIS(instr);
        int16_t imm = (int16_t)(instr & 0x7FF) | ((instr >> 10) & 0x7800);
        if (it != 0)
            m_state.vi[it] = (int16_t)(m_state.vi[is] + imm);
        return;
    }
    case 0x09: // ISUBIU
    {
        uint8_t it = LIT(instr);
        uint8_t is = LIS(instr);
        int16_t imm = (int16_t)(instr & 0x7FF) | ((instr >> 10) & 0x7800);
        if (it != 0)
            m_state.vi[it] = (int16_t)(m_state.vi[is] - imm);
        return;
    }
    case 0x10: // FCEQ
    {
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            m_state.vi[1] = ((m_state.clip & 0xFFFFFF) == imm24) ? 1 : 0;
        return;
    }
    case 0x11: // FCSET
    {
        // Explicit software write: shadow and architectural copy together, so
        // a following CLIP chains off the value the program just set.
        s_clipShadow = instr & 0xFFFFFFu;
        m_state.clip = s_clipShadow;
        return;
    }
    case 0x12: // FCAND
    {
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            m_state.vi[1] = ((m_state.clip & imm24) != 0) ? 1 : 0;
        return;
    }
    case 0x13: // FCOR
    {
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            m_state.vi[1] = ((m_state.clip | imm24) == 0xFFFFFF) ? 1 : 0;
        return;
    }
    case 0x14: // FSEQ
    {
        uint16_t imm12 = instr & 0xFFF;
        if (1 != 0)
            m_state.vi[1] = ((m_state.status & 0xFFF) == imm12) ? 1 : 0;
        return;
    }
    case 0x15: // FSSET
    {
        // PCSX2 _vuFSSET: only the sticky half is writable; the live O/U/S/Z
        // bits keep whatever the last FMAC produced.
        s_statusShadow = (s_statusShadow & 0x3Fu) | (((instr >> 11) & 0x7FFu) & 0xFC0u);
        m_state.status = s_statusShadow;
        return;
    }
    case 0x16: // FSAND
    {
        uint16_t imm12 = instr & 0xFFF;
        if (1 != 0)
            m_state.vi[1] = (int32_t)(m_state.status & imm12);
        return;
    }
    case 0x17: // FSOR
    {
        uint16_t imm12 = instr & 0xFFF;
        if (1 != 0)
            m_state.vi[1] = ((m_state.status | imm12) == 0xFFF) ? 1 : 0;
        return;
    }
    // DC2 spec 04 / G138: the lower-opcode table had FMEQ (0x18) and FMAND
    // (0x1A) swapped -- the same transposition DC2 hit, which made a whole
    // MAC-gated branch cascade unsatisfiable. Canonical table (EE VU manual /
    // PCSX2 VUops): 0x18 FMEQ, 0x1A FMAND, 0x1C FMOR.
    case 0x18: // FMEQ
    {
        uint8_t it = LIT(instr);
        uint8_t is = LIS(instr);
        if (it != 0)
            m_state.vi[it] = ((m_state.mac & 0xFFFF) == (uint32_t)(uint16_t)m_state.vi[is]) ? 1 : 0;
        return;
    }
    case 0x1A: // FMAND
    {
        uint8_t it = LIT(instr);
        uint8_t is = LIS(instr);
        if (it != 0)
            m_state.vi[it] = (int32_t)(m_state.mac & (uint32_t)(uint16_t)m_state.vi[is]);
        return;
    }
    // PCSX2 _LOWER_OPCODE[128]: 0x18 FMEQ, 0x19 unknown, 0x1A FMAND,
    // 0x1B FMOR, 0x1C FCGET. FMOR was at 0x1C here (colliding with FCGET's
    // slot), so a real FMOR fell through to `default` and left its target
    // integer register stale.
    case 0x1B: // FMOR
    {
        uint8_t it = LIT(instr);
        uint8_t is = LIS(instr);
        if (it != 0)
            m_state.vi[it] = (int32_t)(m_state.mac | (uint32_t)(uint16_t)m_state.vi[is]);
        return;
    }
    case 0x1C: // FCGET
    {
        uint8_t it = LIT(instr);
        if (it != 0)
            m_state.vi[it] = (int32_t)(m_state.clip & 0xFFFu);
        return;
    }
    case 0x20: // B (unconditional branch)
    {
        int16_t imm = IMM11(instr);
        uint32_t target = (m_state.pc + 8 + imm * 8) & 0x3FFF;
        // Simplified branch delay: set PC so next iteration lands on target
        m_state.pc = target - 8;
        return;
    }
    case 0x21: // BAL (Branch and link)
    {
        uint8_t it = LIT(instr);
        int16_t imm = IMM11(instr);
        uint32_t target = (m_state.pc + 8 + imm * 8) & 0x3FFF;
        if (it != 0)
            m_state.vi[it] = (int32_t)((m_state.pc + 16) / 8);
        m_state.pc = target - 8;
        return;
    }
    case 0x24: // JR
    {
        uint8_t is = LIS(instr);
        uint32_t target = ((uint32_t)(uint16_t)m_state.vi[is] * 8u) & 0x3FFF;
        m_state.pc = target - 8;
        return;
    }
    case 0x25: // JALR
    {
        uint8_t it = LIT(instr);
        uint8_t is = LIS(instr);
        uint32_t target = ((uint32_t)(uint16_t)m_state.vi[is] * 8u) & 0x3FFF;
        if (it != 0)
            m_state.vi[it] = (int32_t)((m_state.pc + 16) / 8);
        m_state.pc = target - 8;
        return;
    }
    case 0x28: // IBEQ
    {
        uint8_t it = LIT(instr);
        uint8_t is = LIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)m_state.vi[is] == (int16_t)m_state.vi[it])
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & 0x3FFF;
            m_state.pc = target - 8;
        }
        return;
    }
    case 0x29: // IBNE
    {
        uint8_t it = LIT(instr);
        uint8_t is = LIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)m_state.vi[is] != (int16_t)m_state.vi[it])
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & 0x3FFF;
            m_state.pc = target - 8;
        }
        return;
    }
    case 0x2C: // IBLTZ
    {
        uint8_t is = LIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)m_state.vi[is] < 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & 0x3FFF;
            m_state.pc = target - 8;
        }
        return;
    }
    case 0x2D: // IBGTZ
    {
        uint8_t is = LIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)m_state.vi[is] > 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & 0x3FFF;
            m_state.pc = target - 8;
        }
        return;
    }
    case 0x2E: // IBLEZ
    {
        uint8_t is = LIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)m_state.vi[is] <= 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & 0x3FFF;
            m_state.pc = target - 8;
        }
        return;
    }
    case 0x2F: // IBGEZ
    {
        uint8_t is = LIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)m_state.vi[is] >= 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & 0x3FFF;
            m_state.pc = target - 8;
        }
        return;
    }

    case 0x40: // Lower special (opcode in bits 5:0)
    {
        uint8_t funct = instr & 0x3F;
        uint8_t it = LIT(instr);
        uint8_t is = LIS(instr);
        uint8_t id = LID(instr);
        uint8_t dest = (instr >> 21) & 0xF;

        switch (funct)
        {
        case 0x30: // IADD
            if (id != 0)
                m_state.vi[id] = (int16_t)(m_state.vi[is] + m_state.vi[it]);
            return;
        case 0x31: // ISUB
            if (id != 0)
                m_state.vi[id] = (int16_t)(m_state.vi[is] - m_state.vi[it]);
            return;
        case 0x32: // IADDI
        {
            int16_t imm5 = (int16_t)((int32_t)((instr >> 6) & 0x1F) << 27 >> 27);
            if (it != 0)
                m_state.vi[it] = (int16_t)(m_state.vi[is] + imm5);
            return;
        }
        case 0x34: // IAND
            if (id != 0)
                m_state.vi[id] = m_state.vi[is] & m_state.vi[it];
            return;
        case 0x35: // IOR
            if (id != 0)
                m_state.vi[id] = m_state.vi[is] | m_state.vi[it];
            return;

        default:
            break;
        }

        // ---- LowerOp group (bits 5:0 in 0x3C..0x3F) -------------------
        // DC2 spec 04 / G22 and G138. The previous code dispatched this
        // group on `(instr >> 6) & 0x1F` and treated bits 5:0 == 0x3D/0x3E/
        // 0x3F as XGKICK/XTOP/XITOP. Both are wrong. The canonical index
        // (EE VU manual; PCSX2 VUops.cpp `(code & 3) | ((code >> 4) & 0x7c)`)
        // puts XGKICK at 0x6C -- which has bits 5:0 == 0x3C, so every real
        // XGKICK fell into the old special2 sub-switch, found no matching
        // label, and was silently dropped. That is why an entire DQ8 boot
        // recorded ZERO [vu1:xgkick] even while VIF1 issued ~10^5 MSCALs.
        // The same misindexing dropped MOVE/MR32/LQI/SQI/DIV/RSQRT/MFP/
        // XTOP/XITOP and the EFU ops.
        if (funct < 0x3Cu)
            return;
        {
            const uint32_t funct2 = (uint32_t)(instr & 3u) | ((instr >> 4) & 0x7Cu);
            switch (funct2)
            {
            case 0x30: // MOVE
            {
                float tmp[4];
                std::memcpy(tmp, m_state.vf[is], 16);
                applyDest(m_state.vf[it], tmp, dest);
                return;
            }
            case 0x31: // MR32 (rotate right by 32 bits = shift xyzw -> yzwx)
            {
                float tmp[4] = {m_state.vf[is][1], m_state.vf[is][2], m_state.vf[is][3], m_state.vf[is][0]};
                applyDest(m_state.vf[it], tmp, dest);
                return;
            }
            case 0x3D: // MFIR (Move From Integer Register)
            {
                float result[4];
                int32_t val = (int32_t)(int16_t)(m_state.vi[is] & 0xFFFF);
                std::memcpy(&result[0], &val, 4);
                result[1] = result[0];
                result[2] = result[0];
                result[3] = result[0];
                applyDest(m_state.vf[it], result, dest);
                return;
            }
            case 0x3C: // MTIR (Move To Integer Register)
            {
                int comp = 0;
                if (dest & 0x8)
                    comp = 0;
                else if (dest & 0x4)
                    comp = 1;
                else if (dest & 0x2)
                    comp = 2;
                else
                    comp = 3;
                uint32_t fval;
                std::memcpy(&fval, &m_state.vf[is][comp], 4);
                if (it != 0)
                    m_state.vi[it] = (int32_t)(int16_t)(fval & 0xFFFF);
                return;
            }
            // ILWR / ISWR (PCSX2 LowerOP_T3_10/_T3_11 index 0x0F -> funct2
            // 0x3E / 0x3F). Register-indirect integer load/store with no
            // immediate; previously unimplemented, so any microcode that
            // walked a table of VU-memory indices (Level-5's per-strip
            // material/state fetch does exactly this) read a stale vi.
            case 0x3E: // ILWR
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[is]) * 16u;
                addr &= (dataSize - 1);
                if (it != 0 && addr + 16 <= dataSize)
                {
                    int comp = 0;
                    if (dest & 0x8)
                        comp = 0;
                    else if (dest & 0x4)
                        comp = 1;
                    else if (dest & 0x2)
                        comp = 2;
                    else
                        comp = 3;
                    uint32_t v;
                    std::memcpy(&v, vuData + addr + comp * 4, 4);
                    m_state.vi[it] = (int32_t)(int16_t)(v & 0xFFFFu);
                }
                return;
            }
            case 0x3F: // ISWR
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[is]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    const uint32_t val = (uint32_t)(uint16_t)(m_state.vi[it] & 0xFFFF);
                    if (dest & 0x8)
                        std::memcpy(vuData + addr + 0, &val, 4);
                    if (dest & 0x4)
                        std::memcpy(vuData + addr + 4, &val, 4);
                    if (dest & 0x2)
                        std::memcpy(vuData + addr + 8, &val, 4);
                    if (dest & 0x1)
                        std::memcpy(vuData + addr + 12, &val, 4);
                }
                return;
            }
            case 0x40: // RNEXT
                return;
            case 0x41: // RGET
                return;
            case 0x42: // RINIT
                return;
            case 0x43: // RXOR
                return;
            case 0x34: // LQI (Load Quadword, post-increment)
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[is]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    applyDest(m_state.vf[it], tmp, dest);
                }
                if (is != 0)
                    m_state.vi[is] = (int16_t)(m_state.vi[is] + 1);
                return;
            }
            case 0x35: // SQI (Store Quadword, post-increment)
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[it]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    if (dest & 0x8)
                        tmp[0] = m_state.vf[is][0];
                    if (dest & 0x4)
                        tmp[1] = m_state.vf[is][1];
                    if (dest & 0x2)
                        tmp[2] = m_state.vf[is][2];
                    if (dest & 0x1)
                        tmp[3] = m_state.vf[is][3];
                    std::memcpy(vuData + addr, tmp, 16);
                    vu1StoreWatch(addr, vuData, m_state.pc, "vu1-sq");
                }
                if (it != 0)
                    m_state.vi[it] = (int16_t)(m_state.vi[it] + 1);
                return;
            }
            case 0x36: // LQD (Load Quadword, pre-decrement)
            {
                if (is != 0)
                    m_state.vi[is] = (int16_t)(m_state.vi[is] - 1);
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[is]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    applyDest(m_state.vf[it], tmp, dest);
                }
                return;
            }
            case 0x37: // SQD (Store Quadword, pre-decrement)
            {
                if (it != 0)
                    m_state.vi[it] = (int16_t)(m_state.vi[it] - 1);
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[it]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    if (dest & 0x8)
                        tmp[0] = m_state.vf[is][0];
                    if (dest & 0x4)
                        tmp[1] = m_state.vf[is][1];
                    if (dest & 0x2)
                        tmp[2] = m_state.vf[is][2];
                    if (dest & 0x1)
                        tmp[3] = m_state.vf[is][3];
                    std::memcpy(vuData + addr, tmp, 16);
                    vu1StoreWatch(addr, vuData, m_state.pc, "vu1-sqd");
                }
                return;
            }
            // FDIV pipe (PCSX2 _vuFDIVAdd): DIV 7, SQRT 7, RSQRT 13 cycles.
            case 0x38: // DIV
            {
                int fsf = (instr >> 21) & 0x3;
                int ftf = (instr >> 23) & 0x3;
                float num = m_state.vf[is][fsf];
                float den = m_state.vf[it][ftf];
                float qv;
                if (den != 0.0f)
                {
                    qv = num / den;
                    // Q saturates like any other VU result; a finite/tiny
                    // quotient must not become a host infinity.
                    uint32_t qb;
                    std::memcpy(&qb, &qv, 4);
                    if ((qb & 0x7F800000u) == 0x7F800000u)
                    {
                        qb = (qb & 0x80000000u) | 0x7F7FFFFFu;
                        std::memcpy(&qv, &qb, 4);
                    }
                }
                else
                {
                    // PCSX2 _vuDIV: the sign is fs XOR ft, not the sign of the
                    // numerator alone -- a -0.0 divisor flips it.
                    uint32_t nb, db;
                    std::memcpy(&nb, &num, 4);
                    std::memcpy(&db, &den, 4);
                    const uint32_t qb = ((nb ^ db) & 0x80000000u) | 0x7F7FFFFFu;
                    std::memcpy(&qv, &qb, 4);
                }
                vuPipeStartQ(m_state, qv, 7u);
                return;
            }
            case 0x39: // SQRT
            {
                int ftf = (instr >> 23) & 0x3;
                float val = m_state.vf[it][ftf];
                vuPipeStartQ(m_state, std::sqrt(std::fabs(val)), 7u);
                return;
            }
            case 0x3A: // RSQRT
            {
                int fsf = (instr >> 21) & 0x3;
                int ftf = (instr >> 23) & 0x3;
                float num = m_state.vf[is][fsf];
                const float rad = m_state.vf[it][ftf];
                float den = std::sqrt(std::fabs(rad));
                float qv;
                if (den != 0.0f)
                {
                    qv = num / den;
                    uint32_t qb;
                    std::memcpy(&qb, &qv, 4);
                    if ((qb & 0x7F800000u) == 0x7F800000u)
                    {
                        qb = (qb & 0x80000000u) | 0x7F7FFFFFu;
                        std::memcpy(&qv, &qb, 4);
                    }
                }
                else
                {
                    // PCSX2 _vuRSQRT: sign is fs XOR ft, magnitude Fmax.
                    uint32_t nb, db;
                    std::memcpy(&nb, &num, 4);
                    std::memcpy(&db, &rad, 4);
                    const uint32_t qb = ((nb ^ db) & 0x80000000u) | 0x7F7FFFFFu;
                    std::memcpy(&qv, &qb, 4);
                }
                vuPipeStartQ(m_state, qv, 13u);
                return;
            }
            case 0x3B: // WAITQ
                vuPipeWaitQ(m_state);
                return;
            // EFU block. Numbering re-derived from PCSX2's LowerOP_T3_xx
            // tables (funct2 = index*4 + bc): ESADD 0x70, ERSADD 0x71,
            // ELENG 0x72, ERLENG 0x73, EATANxy 0x74, EATANxz 0x75, ESUM 0x76,
            // ESQRT 0x78, ERSQRT 0x79, ERCPR 0x7A, WAITP 0x7B, ESIN 0x7C,
            // EATAN 0x7D, EEXP 0x7E. ERCPR was at 0x77 (an unassigned slot),
            // WAITP at 0x7F, EATAN at 0x7B -- so ERCPR never ran and WAITP
            // was decoded as EATAN's slot.
            // EFU pipe (PCSX2 _vuEFUAdd + the VUREGS_PFS_* latency table):
            // ESADD 11, ERSADD 18, ELENG 18, ERLENG 24, EATANxy/xz 54,
            // ESUM 12, ERCPR 12, ESQRT 12, ERSQRT 18, ESIN 29, EATAN 54,
            // EEXP 44. P becomes readable by MFP only after that many cycles.
            case 0x70: // ESADD
            {
                const float v = m_state.vf[is][0] * m_state.vf[is][0] + m_state.vf[is][1] * m_state.vf[is][1] + m_state.vf[is][2] * m_state.vf[is][2];
                vuPipeStartP(m_state, v, 11u);
                return;
            }
            case 0x71: // ERSADD
            {
                const float sq = m_state.vf[is][0] * m_state.vf[is][0] + m_state.vf[is][1] * m_state.vf[is][1] + m_state.vf[is][2] * m_state.vf[is][2];
                vuPipeStartP(m_state, (sq != 0.0f) ? (1.0f / sq) : std::numeric_limits<float>::max(), 18u);
                return;
            }
            case 0x72: // ELENG
            {
                const float sq = m_state.vf[is][0] * m_state.vf[is][0] + m_state.vf[is][1] * m_state.vf[is][1] + m_state.vf[is][2] * m_state.vf[is][2];
                vuPipeStartP(m_state, std::sqrt(sq), 18u);
                return;
            }
            case 0x73: // ERLENG
            {
                const float sq = m_state.vf[is][0] * m_state.vf[is][0] + m_state.vf[is][1] * m_state.vf[is][1] + m_state.vf[is][2] * m_state.vf[is][2];
                const float len = std::sqrt(sq);
                vuPipeStartP(m_state, (len != 0.0f) ? (1.0f / len) : std::numeric_limits<float>::max(), 24u);
                return;
            }
            case 0x74: // EATANxy
            {
                const float x = m_state.vf[is][0], y = m_state.vf[is][1];
                vuPipeStartP(m_state, std::atan2(y, x), 54u);
                return;
            }
            case 0x75: // EATANxz
            {
                const float x = m_state.vf[is][0], z = m_state.vf[is][2];
                vuPipeStartP(m_state, std::atan2(z, x), 54u);
                return;
            }
            case 0x76: // ESUM
            {
                const float v = m_state.vf[is][0] + m_state.vf[is][1] + m_state.vf[is][2] + m_state.vf[is][3];
                vuPipeStartP(m_state, v, 12u);
                return;
            }
            case 0x78: // ESQRT
            {
                const int fsf = (instr >> 21) & 0x3;
                vuPipeStartP(m_state, std::sqrt(std::fabs(m_state.vf[is][fsf])), 12u);
                return;
            }
            case 0x79: // ERSQRT
            {
                const int fsf = (instr >> 21) & 0x3;
                const float d = std::sqrt(std::fabs(m_state.vf[is][fsf]));
                vuPipeStartP(m_state, (d != 0.0f) ? (1.0f / d) : std::numeric_limits<float>::max(), 18u);
                return;
            }
            case 0x7A: // ERCPR
            {
                const int fsf = (instr >> 21) & 0x3;
                const float val = m_state.vf[is][fsf];
                vuPipeStartP(m_state, (val != 0.0f) ? (1.0f / val) : std::numeric_limits<float>::max(), 12u);
                return;
            }
            case 0x7B: // WAITP
                vuPipeWaitP(m_state);
                return;
            case 0x7C: // ESIN
            {
                const int fsf = (instr >> 21) & 0x3;
                vuPipeStartP(m_state, std::sin(m_state.vf[is][fsf]), 29u);
                return;
            }
            case 0x7D: // EATAN
            {
                const int fsf = (instr >> 21) & 0x3;
                vuPipeStartP(m_state, std::atan(m_state.vf[is][fsf]), 54u);
                return;
            }
            case 0x7E: // EEXP
            {
                const int fsf = (instr >> 21) & 0x3;
                vuPipeStartP(m_state, std::exp(-m_state.vf[is][fsf]), 44u);
                return;
            }
            case 0x64: // MFP (Move From P register)
            {
                float result[4] = {m_state.p, m_state.p, m_state.p, m_state.p};
                applyDest(m_state.vf[it], result, dest);
                return;
            }
            case 0x6C: // XGKICK - send GIF packet from VU1 data memory
            {
            if (!vuData || dataSize < 16u)
                return;

            auto wrapOffset = [&](uint32_t off) -> uint32_t
            {
                return off % dataSize;
            };

            auto read64Wrap = [&](uint32_t off) -> uint64_t
            {
                uint8_t bytes[8];
                for (uint32_t i = 0; i < 8u; ++i)
                {
                    bytes[i] = vuData[wrapOffset(off + i)];
                }
                uint64_t value = 0;
                std::memcpy(&value, bytes, sizeof(value));
                return value;
            };

            uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[is]) * 16u;
            addr = wrapOffset(addr);
            uint32_t pktOff = addr;
            uint32_t totalBytes = 0u;
            bool done = false;
            uint32_t tagsWalked = 0u;
            uint64_t firstTags[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            uint32_t firstTagOff[8] = {0, 0, 0, 0, 0, 0, 0, 0};

            for (int safety = 0; safety < 256 && !done; ++safety)
            {
                uint64_t tagLo = read64Wrap(pktOff);
                uint32_t nloop = (uint32_t)(tagLo & 0x7FFFu);
                uint8_t flg = (uint8_t)((tagLo >> 58) & 0x3u);
                uint32_t nreg = (uint32_t)((tagLo >> 60) & 0xFu);
                if (nreg == 0u)
                    nreg = 16u;
                bool eop = ((tagLo >> 15) & 0x1ull) != 0ull;

                // Second-and-later GIFtag REGS sample. `tagsWalked` here is
                // still the pre-increment count of tags already consumed by
                // this kick, i.e. exactly the 0-based index of the tag being
                // decoded right now -- so ">= 1" means "not the first tag".
                if (giftagRegsSampleOn() && tagsWalked >= 1u)
                {
                    const uint64_t tagHi = read64Wrap(pktOff + 8u);
                    giftagHistNote(nreg, tagHi, flg, nloop);

                    const uint32_t idx = s_giftagRegsSampleCount.fetch_add(1, std::memory_order_relaxed);
                    static const uint32_t kMaxGiftagRegsSampleLogs =
                        localDiagEnvLimit("PS2X_GIFTAG_REGS_MAX_LOGS", 256u);
                    static std::atomic<bool> s_giftagRegsSampleTruncated{false};
                    if (localDiagLogBudget(std::cout,
                                           "[gs:giftag-regs]",
                                           "PS2X_GIFTAG_REGS_MAX_LOGS",
                                           kMaxGiftagRegsSampleLogs,
                                           idx,
                                           s_giftagRegsSampleTruncated))
                    {
                        std::cout << "[gs:giftag-regs] idx=" << idx
                                  << " tagNum=" << tagsWalked
                                  << " nloop=" << nloop
                                  << " nreg=" << nreg
                                  << " flg=" << static_cast<uint32_t>(flg)
                                  << " eop=" << (eop ? 1 : 0)
                                  << " regs=[" << std::hex;
                        for (uint32_t r = 0; r < nreg && r < 16u; ++r)
                        {
                            if (r)
                                std::cout << ',';
                            std::cout << "0x" << ((tagHi >> (r * 4)) & 0xFu);
                        }
                        std::cout << std::dec << "]" << std::endl;
                    }
                }

                if (tagsWalked < 8u)
                {
                    firstTags[tagsWalked] = tagLo;
                    firstTagOff[tagsWalked] = pktOff;
                }
                ++tagsWalked;

                uint32_t pktSize = 16u;
                if (flg == 0u)
                {
                    pktSize += nloop * nreg * 16u;
                }
                else if (flg == 1u)
                {
                    uint32_t regs = nloop * nreg;
                    pktSize += regs * 8u;
                    if ((regs & 1u) != 0u)
                        pktSize += 8u;
                }
                else if (flg == 2u)
                {
                    pktSize += nloop * 16u;
                }

                if (pktSize == 0u)
                    break;

                totalBytes += pktSize;
                pktOff = wrapOffset(pktOff + pktSize);
                if (eop)
                    done = true;
            }

            // Autopsy for a kick whose tag chain never reaches EOP inside VU
            // memory. On hardware XGKICK streams until the EOP tag, and a
            // microprogram always terminates its own packet, so "the walk ran
            // off the end of a 16 KB data memory" means the bytes at vi[is]
            // are not the packet the program thought it built. Bounded and
            // always-on: this is the difference between "geometry flows" and
            // "we submitted 64 KB of wrapped garbage to the GS".
            s_censusKicks.fetch_add(1, std::memory_order_relaxed);
            s_censusKickBytes.fetch_add(totalBytes, std::memory_order_relaxed);
            s_censusKickNloop.fetch_add((uint32_t)(firstTags[0] & 0x7FFFu), std::memory_order_relaxed);
            if (totalBytes <= 16u)
                s_censusKicksEmpty.fetch_add(1, std::memory_order_relaxed);

            {
                // Periodic sample of ORDINARY kicks too, so "what does a good
                // packet look like" and "what does a bad one look like" can be
                // read off the same log.
                static std::atomic<uint64_t> s_allKicks{0};
                const uint64_t an = s_allKicks.fetch_add(1, std::memory_order_relaxed);
                if ((an % 200000u) == 0u)
                {
                    std::cerr << "[vu1:kicksample] n=" << std::dec << an
                              << " addr=0x" << std::hex << addr << std::dec
                              << " tags=" << tagsWalked
                              << " totalBytes=" << totalBytes
                              << " reachedEop=" << (done ? 1 : 0)
                              << " tag0=0x" << std::hex << firstTags[0] << std::dec
                              << " nloop=" << (uint32_t)(firstTags[0] & 0x7FFFu)
                              << " flg=" << (uint32_t)((firstTags[0] >> 58) & 3u)
                              << " nreg=" << (uint32_t)((firstTags[0] >> 60) & 0xFu)
                              << std::endl;
                }
            }

            if (!done || totalBytes > dataSize)
            {
                static std::atomic<uint64_t> s_badKicks{0};
                const uint64_t bn = s_badKicks.fetch_add(1, std::memory_order_relaxed);
                if (bn < 24u || (bn % 100000u) == 0u)
                {
                    std::cerr << "[vu1:kickwalk] n=" << std::dec << bn
                              << " addr=0x" << std::hex << addr << std::dec
                              << " vi" << (int)is << "=" << (int)(int16_t)m_state.vi[is]
                              << " tags=" << tagsWalked
                              << " totalBytes=" << totalBytes
                              << " dataSize=" << dataSize
                              << " reachedEop=" << (done ? 1 : 0)
                              << std::endl;
                    const uint32_t shown = (tagsWalked < 8u) ? tagsWalked : 8u;
                    for (uint32_t t = 0; t < shown; ++t)
                    {
                        const uint64_t tg = firstTags[t];
                        std::cerr << "[vu1:kickwalk]   +0x" << std::hex << firstTagOff[t]
                                  << " tagLo=0x" << tg << std::dec
                                  << " nloop=" << (uint32_t)(tg & 0x7FFFu)
                                  << " eop=" << (uint32_t)((tg >> 15) & 1u)
                                  << " flg=" << (uint32_t)((tg >> 58) & 3u)
                                  << " nreg=" << (uint32_t)((tg >> 60) & 0xFu)
                                  << std::endl;
                    }
                }
            }

            if (totalBytes == 0u)
                return;

            // M-T1 diag (2026-07-18, generic, bounded, temporary): was
            // RUNTIME_LOG (silent outside a _DEBUG build) -- promoted to
            // always-on to check whether VU1's PATH1 XGKICK route ever
            // fires at all during a boot where PATH3-direct texture
            // uploads (dbw=4, ~229KB) are observed to declare a real
            // nloop but never receive a matching payload burst.
            const uint32_t debugIndex = s_debugVu1XgkickCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 300u)
            {
                std::cout << "[vu1:xgkick] idx=" << debugIndex
                                                << " addr=0x" << std::hex << addr
                                                << " totalBytes=0x" << totalBytes
                                                << std::dec
                                                << " wrap=" << static_cast<uint32_t>((addr + totalBytes > dataSize) ? 1u : 0u)
                                                << std::endl;
            }

            if (addr + totalBytes <= dataSize)
            {
                if (memory)
                    memory->submitGifPacket(GifPathId::Path1, vuData + addr, totalBytes);
                else
                    gs.processGIFPacket(vuData + addr, totalBytes);
            }
            else
            {
                std::vector<uint8_t> wrappedPacket(totalBytes);
                for (uint32_t i = 0; i < totalBytes; ++i)
                {
                    wrappedPacket[i] = vuData[wrapOffset(addr + i)];
                }

                if (memory)
                    memory->submitGifPacket(GifPathId::Path1, wrappedPacket.data(), totalBytes);
                else
                    gs.processGIFPacket(wrappedPacket.data(), totalBytes);
            }
            return;
        }
            case 0x68: // XTOP
            {
                if (it != 0)
                    m_state.vi[it] = (int32_t)m_state.xitop;
                return;
            }
            case 0x69: // XITOP
            {
                if (it != 0)
                    m_state.vi[it] = (int32_t)m_state.itop;
                return;
            }
            default:
                return;
            }
        }
    }
    default:
        break;
    }
}
