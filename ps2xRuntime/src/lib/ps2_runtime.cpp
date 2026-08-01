#include "ps2_runtime.h"
#include "ps2_log.h"
#include "ps2_stubs.h"
#include "ps2_syscalls.h"
#include "game_overrides.h"
#include "ps2_runtime_macros.h"
#include "runtime/ps2_gs_gpu.h"
#include "ThreadNaming.h"
#include "Kernel/Stubs/Audio.h"
#include "Kernel/Stubs/GS.h"
#include "Kernel/Stubs/MPEG.h"
#include "Kernel/Syscalls/Sync.h"
#include "ps2_host_backend.h"
#include "ps2_scheduler.h"
#include "runtime/ps2_guestwatch.h"
#include "runtime/ps2_vif0.h"
#include "runtime/ps2_vu1.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <chrono>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <sstream>
#include <vector>

namespace ps2_stubs
{
    void resetSifState();
}

// ===========================================================================
// Forensic write-watch policy.
//
// Every definition below used to be an `inline` function in ps2_runtime.h and
// was therefore duplicated into all ~7800 generated corpus translation units,
// each frozen at whatever the header said on its compile date. See the long
// comment above the declarations in ps2_runtime.h for the failure that caused.
// These live here, in libps2_runtime, so there is exactly one copy per process
// and policy changes are a runtime-only rebuild.
//
// Do not mark any of these `inline`, and do not move them back into a header.
// ===========================================================================

uint32_t ps2DiagEnvLimit(const char *envName, uint32_t fallback)
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

// Returns true if this event is within budget and should be logged. Emits a
// single "TRUNCATED after N events" line the first time an event is dropped,
// so a capped probe can never be mistaken for a probe that saw nothing.
bool ps2DiagLogBudget(std::ostream &os,
                      const char *tag,
                      const char *envName,
                      uint32_t limit,
                      uint32_t logIndex,
                      std::atomic<bool> &truncationAnnounced)
{
    if (limit == 0u || logIndex < limit)
    {
        return true;
    }
    bool expected = false;
    if (truncationAnnounced.compare_exchange_strong(expected, true, std::memory_order_relaxed))
    {
        auto flags = os.flags();
        os.flags(std::ios::dec | (flags & ~std::ios::basefield));
        os << tag << " TRUNCATED after " << limit
           << " events; further events are NOT logged (raise or disable with "
           << envName << "=<n>, 0 = unlimited)" << std::endl;
        os.flags(flags);
    }
    return false;
}

uint32_t ps2PathWatchMaxLogs()
{
    static const uint32_t limit =
        ps2DiagEnvLimit("PS2X_PATH_WATCH_MAX_LOGS", PS2_PATH_WATCH_MAX_LOGS_DEFAULT);
    return limit;
}

uint32_t ps2SlotWatchMaxLogs()
{
    static const uint32_t limit =
        ps2DiagEnvLimit("PS2X_SLOT_WATCH_MAX_LOGS", PS2_SLOT_WATCH_MAX_LOGS_DEFAULT);
    return limit;
}

uint32_t ps2PathWatchPhysAddr()
{
    return g_ps2PathWatchBase.load(std::memory_order_relaxed) & PS2_RAM_MASK;
}

uint32_t ps2PathWatchWatchBytes()
{
    return g_ps2PathWatchBytes.load(std::memory_order_relaxed);
}

namespace
{
    void ps2PathWatchDumpPrefix(const uint8_t *rdram)
    {
        if (!rdram)
        {
            return;
        }

        const uint32_t base = ps2PathWatchPhysAddr();
        auto flags = std::cout.flags();
        std::cout << " buf=" << std::hex;
        for (uint32_t i = 0; i < 16u; ++i)
        {
            const uint32_t addr = (base + i) & PS2_RAM_MASK;
            std::cout << static_cast<uint32_t>(rdram[addr]);
            if (i + 1u < 16u)
            {
                std::cout << '.';
            }
        }
        std::cout.flags(flags);
    }

    uint8_t ps2PathWatchExtractByteFromWrite(uint32_t writeAddr,
                                             uint32_t watchAddr,
                                             uint64_t valueLo,
                                             uint64_t valueHi)
    {
        const uint32_t byteIndex = watchAddr - writeAddr;
        if (byteIndex < 8u)
        {
            return static_cast<uint8_t>((valueLo >> (byteIndex * 8u)) & 0xFFu);
        }
        return static_cast<uint8_t>((valueHi >> ((byteIndex - 8u) * 8u)) & 0xFFu);
    }
} // namespace

void ps2DiagSlotWatchReport(uint32_t writeAddr,
                            uint32_t size,
                            uint64_t valueLo,
                            uint64_t valueHi,
                            const char *op,
                            const R5900Context *ctx)
{
    const uint32_t watchPhys = g_ps2SlotWatchAddr.load(std::memory_order_relaxed) & PS2_RAM_MASK;
    const uint32_t logIndex = g_ps2SlotWatchLogCount.fetch_add(1, std::memory_order_relaxed);
    if (!ps2DiagLogBudget(std::cerr,
                          "[watch:slot-write]",
                          "PS2X_SLOT_WATCH_MAX_LOGS",
                          ps2SlotWatchMaxLogs(),
                          logIndex,
                          g_ps2SlotWatchTruncated))
    {
        return;
    }

    const uint32_t pc = ctx ? ctx->pc : 0u;
    const uint32_t ra = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)) : 0u;
    const uint32_t sp = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0)) : 0u;

    auto flags = std::cerr.flags();
    std::cerr << "[watch:slot-write] #" << (logIndex + 1u)
              << " op=" << op
              << " watch=0x" << std::hex << watchPhys
              << " addr=0x" << writeAddr
              << " size=0x" << size
              << " pc=0x" << pc
              << " ra=0x" << ra
              << " sp=0x" << sp
              << " vLo=0x" << valueLo;
    if (size > 8u)
    {
        std::cerr << " vHi=0x" << valueHi;
    }
    std::cerr.flags(flags);
    std::cerr << std::endl;
}

void ps2DiagPathWatchReport(const uint8_t *rdram,
                            uint32_t writeAddr,
                            uint32_t size,
                            uint64_t valueLo,
                            uint64_t valueHi,
                            const char *op,
                            const R5900Context *ctx)
{
    const uint32_t logIndex = g_ps2PathWatchLogCount.fetch_add(1, std::memory_order_relaxed);
    if (!ps2DiagLogBudget(std::cout,
                          "[watch:path-write]",
                          "PS2X_PATH_WATCH_MAX_LOGS",
                          ps2PathWatchMaxLogs(),
                          logIndex,
                          g_ps2PathWatchTruncated))
    {
        return;
    }

    const uint32_t watchAddr = ps2PathWatchPhysAddr();
    const bool touchesFirstByte = (watchAddr >= writeAddr) && (watchAddr < writeAddr + size);
    const uint8_t oldByte = rdram[watchAddr];
    const uint8_t newByte = touchesFirstByte
                                ? ps2PathWatchExtractByteFromWrite(writeAddr, watchAddr, valueLo, valueHi)
                                : oldByte;

    const uint32_t pc = ctx ? ctx->pc : 0u;
    const uint32_t ra = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)) : 0u;
    const uint32_t sp = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0)) : 0u;

    auto flags = std::cout.flags();
    std::cout << "[watch:path-write] #" << (logIndex + 1u)
              << " op=" << op
              << " addr=0x" << std::hex << writeAddr
              << " size=0x" << size
              << " pc=0x" << pc
              << " ra=0x" << ra
              << " sp=0x" << sp
              << " vLo=0x" << valueLo;
    if (size > 8u)
    {
        std::cout << " vHi=0x" << valueHi;
    }
    if (touchesFirstByte)
    {
        std::cout << " firstByte:" << static_cast<uint32_t>(oldByte)
                  << "->" << static_cast<uint32_t>(newByte);
        if (oldByte != 0u && newByte == 0u)
        {
            std::cout << " (ZEROED)";
        }
    }
    ps2PathWatchDumpPrefix(rdram);
    std::cout.flags(flags);
    std::cout << std::endl;
}

void ps2DiagPathRangeReport(const uint8_t *rdram,
                            uint32_t writeAddr,
                            uint32_t size,
                            const char *op,
                            const R5900Context *ctx)
{
    const uint32_t logIndex = g_ps2PathWatchLogCount.fetch_add(1, std::memory_order_relaxed);
    if (!ps2DiagLogBudget(std::cout,
                          "[watch:path-write]",
                          "PS2X_PATH_WATCH_MAX_LOGS",
                          ps2PathWatchMaxLogs(),
                          logIndex,
                          g_ps2PathWatchTruncated))
    {
        return;
    }

    const uint32_t pc = ctx ? ctx->pc : 0u;
    const uint32_t ra = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)) : 0u;
    const uint32_t sp = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0)) : 0u;
    const uint8_t firstByte = rdram[ps2PathWatchPhysAddr()];

    auto flags = std::cout.flags();
    std::cout << "[watch:path-range] #" << (logIndex + 1u)
              << " op=" << op
              << " addr=0x" << std::hex << writeAddr
              << " size=0x" << size
              << " pc=0x" << pc
              << " ra=0x" << ra
              << " sp=0x" << sp
              << " firstByte=" << static_cast<uint32_t>(firstByte);
    ps2PathWatchDumpPrefix(rdram);
    std::cout.flags(flags);
    std::cout << std::endl;
}

void ps2DiagAnnounceWatchState()
{
    const uint32_t slot = g_ps2SlotWatchAddr.load(std::memory_order_relaxed);
    const uint32_t pathBase = g_ps2PathWatchBase.load(std::memory_order_relaxed);
    const uint32_t pathBytes = g_ps2PathWatchBytes.load(std::memory_order_relaxed);
    const uint32_t slotBudget = ps2SlotWatchMaxLogs();
    const uint32_t pathBudget = ps2PathWatchMaxLogs();

    auto flags = std::cerr.flags();
    std::cerr << "[watch:state] slot-write=";
    if (slot == 0u)
    {
        std::cerr << "DISARMED";
    }
    else
    {
        std::cerr << "0x" << std::hex << (slot & PS2_RAM_MASK) << std::dec;
    }
    std::cerr << " budget=" << slotBudget << (slotBudget == 0u ? " (unlimited)" : "")
              << "; path-write=";
    if (pathBytes == 0u)
    {
        std::cerr << "DISARMED";
    }
    else
    {
        std::cerr << "0x" << std::hex << (pathBase & PS2_RAM_MASK)
                  << "+0x" << pathBytes << std::dec;
    }
    std::cerr << " budget=" << pathBudget << (pathBudget == 0u ? " (unlimited)" : "")
              << "; policy resolved in libps2_runtime, not inlined into the corpus"
              << std::endl;

    // The two VU0 rotation-basis probes were previously absent from this
    // banner, so a run that printed zero [vu0:exec] / [13cc70:vcallms] lines
    // could not be told apart from "gate never fired", "probe never armed"
    // and "budget exhausted". That ambiguity has produced three wrong
    // conclusions in this project, so both are announced here with their
    // resolved budgets AND the exact arming condition and disambiguation
    // rule. Truncation is already announced once per bucket through
    // ps2DiagLogBudget, so "budget exhausted" always leaves a TRUNCATED line.
    const uint32_t vu0ExecBaseline =
        ps2DiagEnvLimit("PS2X_VU0_EXEC_BASELINE_MAX_LOGS", 20u);
    const uint32_t vu0ExecRailed =
        ps2DiagEnvLimit("PS2X_VU0_EXEC_RAILED_MAX_LOGS", 500u);
    std::cerr << "[watch:state] vu0:exec=ARMED (runtime-side, always compiled in)"
              << " baseline-budget=" << vu0ExecBaseline
              << (vu0ExecBaseline == 0u ? " (unlimited)" : "")
              << " per-micro-address; railed-budget=" << vu0ExecRailed
              << (vu0ExecRailed == 0u ? " (unlimited)" : "")
              << "; fires on every VCALLMS that reaches the real VU0"
              << " interpreter. NO [vu0:exec] line AND no"
              << " [vu0:exec:*] TRUNCATED line means the gate never fired:"
              << " either no VCALLMS executed at all (cross-check"
              << " [vu0:census] totalCalls) or every VCALLMS took an early"
              << " out -- which prints [vu0:empty] when VU0 micro memory"
              << " never received an MPG upload." << std::endl;

    const uint32_t f13cc70Baseline =
        ps2DiagEnvLimit("PS2X_13CC70_BASELINE_MAX_LOGS", 8u);
    const uint32_t f13cc70Railed =
        ps2DiagEnvLimit("PS2X_13CC70_RAILED_MAX_LOGS", 500u);
    std::cerr << "[watch:state] 13cc70:vcallms budgets resolved here:"
              << " baseline-budget=" << f13cc70Baseline
              << (f13cc70Baseline == 0u ? " (unlimited)" : "")
              << "; railed-budget=" << f13cc70Railed
              << (f13cc70Railed == 0u ? " (unlimited)" : "")
              << ". This probe is CORPUS-side (emitted into"
              << " sub_0013CC70_0x13cc70.cpp), so libps2_runtime cannot"
              << " report whether it was compiled in -- only what budgets it"
              << " will use if it was. Disambiguation: sub_0013CC70's VCALLMS"
              << " targets micro@0x0, so if [vu0:exec] reports runs at"
              << " micro@0x0 while no [13cc70:vcallms] line appears, the"
              << " probe is NOT in this corpus build; if [vu0:exec] also"
              << " reports nothing at micro@0x0, the gate genuinely never"
              << " fired." << std::endl;
    std::cerr.flags(flags);
}

// ---------------------------------------------------------------------------
// ps2_watch — cheap permanent guest-memory transition watches.
// See runtime/ps2_guestwatch.h for the contract. Polled from the frame loop
// (~60 Hz); prints one line per observed change, sparsified after the first
// kVerboseChanges changes so per-frame counters stay visible but bounded.
// ---------------------------------------------------------------------------
namespace ps2_watch
{
    namespace
    {
        struct Watch
        {
            uint32_t addr = 0;
            uint32_t width = 4;
            const char *label = "";
            uint64_t lastValue = 0;
            bool primed = false;
            uint64_t changeCount = 0;
        };
        std::vector<Watch> g_watches;
        std::mutex g_watch_mutex;
        constexpr uint64_t kVerboseChanges = 16;
        constexpr uint64_t kSparseInterval = 600; // ~10 s at one change/frame

        uint64_t readValue(const uint8_t *rdram, uint32_t addr, uint32_t width)
        {
            const uint32_t phys = addr & (32u * 1024u * 1024u - 1u);
            uint64_t v = 0;
            std::memcpy(&v, rdram + phys, width);
            return v;
        }
    }

    void addWatch(uint32_t guestAddr, uint32_t byteWidth, const char *label)
    {
        if (byteWidth != 1 && byteWidth != 2 && byteWidth != 4)
        {
            byteWidth = 4;
        }
        std::lock_guard<std::mutex> lock(g_watch_mutex);
        g_watches.push_back(Watch{guestAddr, byteWidth, label, 0, false, 0});
    }

    void poll(const uint8_t *rdram)
    {
        if (!rdram)
        {
            return;
        }
        std::lock_guard<std::mutex> lock(g_watch_mutex);
        for (Watch &w : g_watches)
        {
            const uint64_t v = readValue(rdram, w.addr, w.width);
            if (!w.primed)
            {
                w.primed = true;
                w.lastValue = v;
                continue;
            }
            if (v == w.lastValue)
            {
                continue;
            }
            ++w.changeCount;
            if (w.changeCount <= kVerboseChanges ||
                (w.changeCount % kSparseInterval) == 0)
            {
                std::cout << "[watch] " << w.label
                          << " 0x" << std::hex << w.addr
                          << ": 0x" << w.lastValue << " -> 0x" << v
                          << std::dec << " (change #" << w.changeCount << ")"
                          << std::endl;
            }
            w.lastValue = v;
        }
    }
}

#define ELF_MAGIC 0x464C457F // "\x7FELF" in little endian
#define ET_EXEC 2            // Executable file
#define EM_MIPS 8            // MIPS architecture
#define PT_LOAD 1            // Loadable segment

static constexpr int FB_WIDTH = 640;
static constexpr int FB_HEIGHT = 512;
static constexpr int DEFAULT_DISPLAY_HEIGHT = 448;
static constexpr uint32_t DEFAULT_FB_SIZE = FB_WIDTH * FB_HEIGHT * 4;
static constexpr uint32_t DEFAULT_FB_ADDR = (PS2_RAM_SIZE - DEFAULT_FB_SIZE - 0x10000u);
#if defined(PLATFORM_VITA)
static constexpr int HOST_WINDOW_WIDTH = 960;
static constexpr int HOST_WINDOW_HEIGHT = 544;
#else
static constexpr int HOST_WINDOW_WIDTH = FB_WIDTH;
static constexpr int HOST_WINDOW_HEIGHT = DEFAULT_DISPLAY_HEIGHT;
#endif
struct ElfHeader
{
    uint32_t magic;
    uint8_t elf_class;
    uint8_t endianness;
    uint8_t version;
    uint8_t os_abi;
    uint8_t abi_version;
    uint8_t padding[7];
    uint16_t type;
    uint16_t machine;
    uint32_t version2;
    uint32_t entry;
    uint32_t phoff;
    uint32_t shoff;
    uint32_t flags;
    uint16_t ehsize;
    uint16_t phentsize;
    uint16_t phnum;
    uint16_t shentsize;
    uint16_t shnum;
    uint16_t shstrndx;
};

struct ProgramHeader
{
    uint32_t type;
    uint32_t offset;
    uint32_t vaddr;
    uint32_t paddr;
    uint32_t filesz;
    uint32_t memsz;
    uint32_t flags;
    uint32_t align;
};

namespace
{
    constexpr uint32_t kGuestHeapDefaultBase = 0x00100000u;
    constexpr uint32_t kGuestHeapDefaultAlignment = 16u;
    constexpr uint32_t kGuestHeapSafetyPad = 0x1000u;
    constexpr uint32_t kGuestHeapHardLimit = 0x01F00000u;

    // -----------------------------------------------------------------------
    // Async callback stack pool: [0x00080000, 0x00100000) -- KERNEL-RESERVED
    // memory, 512 KB, carved downward from kAsyncCallbackStackTop.
    //
    // Why here: the pool used to carve down from PS2_RAM_SIZE with floor
    // kGuestHeapHardLimit, so the FIRST reservation's stack top was
    // 0x1FFFFF0 -- inside the GUEST'S OWN main stack. DQ8's crt0 declares its
    // main stack explicitly via SetupThread(gp, stack=0x01F40000,
    // size=0xC0000) => $sp = 0x02000000, i.e. the game owns
    // [0x01F40000, 0x02000000). Host-dispatched guest callbacks (the
    // sceGsSyncVCallback chain, INTC handlers, alarms) therefore ran ON the
    // live main-thread stack, interleaved with it by the N=1 scheduler's
    // token handoff: each side's register spills corrupted the other's
    // frames (observed: the per-VSync user-callback pointer spilled by
    // sub_00160530 turning to float-bit garbage; $ra clobbers near
    // 0x1fff9f0; recover-pc storms with sp=0x1ffffe0).
    //
    // On real hardware these contexts run on KERNEL stacks in kernel-
    // reserved memory -- never on the interrupted user thread's stack. The
    // EE kernel owns phys [0, 0x00100000): this runtime's kernel-mirror
    // state sits below 0x00012000 (syscall table mirror at 0x11F80, probe
    // words at 0x2F0), the ELF loads at 0x00100000+, the guest heap at its
    // bss end, and all guest thread stacks are game-chosen addresses >=
    // 0x00100000. [0x00080000, 0x00100000) is untouched by both sides, so
    // callback stacks there are disjoint from ALL guest memory by
    // construction. NOTE: the m_asyncCallbackStackFloor default in
    // ps2_runtime.h (0x01FC0000) is dead code -- resetCPUState()/loadELF()
    // below always override it; align the header next time a header-touching
    // rebuild is scheduled (header edits recompile every generated unit).
    // -----------------------------------------------------------------------
    constexpr uint32_t kAsyncCallbackStackFloor = 0x00080000u;
    constexpr uint32_t kAsyncCallbackStackTop = 0x00100000u;

    constexpr uint32_t COP0_CAUSE_EXCCODE_MASK = 0x0000007Cu;
    constexpr uint32_t COP0_CAUSE_BD = 0x80000000u;
    constexpr uint32_t COP0_STATUS_EXL = 0x00000002u;
    constexpr uint32_t COP0_STATUS_BEV = 0x00400000u;
    constexpr uint32_t EXCEPTION_VECTOR_GENERAL = 0x80000080u;
    constexpr uint32_t EXCEPTION_VECTOR_TLB_REFILL = 0x80000000u;
    constexpr uint32_t EXCEPTION_VECTOR_BOOT = 0xBFC00200u;

    struct HostFrameProbePoint
    {
        uint32_t x;
        uint32_t y;
    };

    constexpr HostFrameProbePoint kGhostProbePoints[] = {
        {220u, 176u},
        {260u, 208u},
        {320u, 208u},
        {260u, 240u},
        {320u, 240u},
        {260u, 272u},
        {320u, 272u},
    };

    uint32_t sampleHostFramePixel(const std::vector<uint8_t> &pixels,
                                  uint32_t width,
                                  uint32_t height,
                                  uint32_t x,
                                  uint32_t y)
    {
        if (x >= width || y >= height)
        {
            return 0u;
        }

        const size_t offset = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4u;
        if (offset + 4u > pixels.size())
        {
            return 0u;
        }

        return static_cast<uint32_t>(pixels[offset + 0u]) |
               (static_cast<uint32_t>(pixels[offset + 1u]) << 8) |
               (static_cast<uint32_t>(pixels[offset + 2u]) << 16) |
               (static_cast<uint32_t>(pixels[offset + 3u]) << 24);
    }

    struct DispatchHistory
    {
        std::array<uint32_t, 64> pcs{};
        uint32_t next = 0u;
        bool wrapped = false;
    };

    // -----------------------------------------------------------------------
    // Per-GUEST-THREAD dispatch/recovery state.
    //
    // This used to be plain thread_local (one DispatchHistory + one recovery
    // budget per OS thread). Under the N=1 fiber scheduler ALL guest fibers
    // run on the single executor thread, so every fiber shared one dispatch
    // history, one 8192 recovery budget, and one first-bad-pc log: benign
    // misses from healthy threads consumed a sick thread's budget (a boot log
    // showed exactly 8193 not-found warnings), and the shared history let a
    // faulting fiber "recover" into a DIFFERENT fiber's recent pc (control-
    // flow transplant).
    //
    // Fix: key the state by guest thread id (g_currentThreadId) inside a
    // thread_local map. On the executor thread this yields one state per
    // fiber; host workers (irq/alarm, g_currentThreadId == -1) each get their
    // own isolated state because the map itself is thread_local.
    //
    // Note: entries for finished fibers persist until process exit (bounded
    // by the number of distinct tids, which is small). A cleaner design would
    // store this state in FiberContext and clear it on fiber teardown; the
    // map keyed by tid is the minimal correct version.
    // -----------------------------------------------------------------------
    struct DispatchRecoveryState
    {
        DispatchHistory history;
        uint32_t recoverCount = 0u;
        bool loggedContext = false;
    };

    DispatchRecoveryState &currentDispatchRecoveryState()
    {
        thread_local std::unordered_map<int, DispatchRecoveryState> s_states;
        thread_local int s_cachedTid = std::numeric_limits<int>::min();
        thread_local DispatchRecoveryState *s_cachedState = nullptr;

        const int tid = g_currentThreadId;
        if (s_cachedState == nullptr || tid != s_cachedTid)
        {
            // unordered_map is node-based: element addresses are stable across
            // rehash, so caching the pointer is safe.
            s_cachedState = &s_states[tid];
            s_cachedTid = tid;
        }
        return *s_cachedState;
    }

    void pushDispatchPc(uint32_t pc)
    {
        DispatchHistory &h = currentDispatchRecoveryState().history;
        h.pcs[h.next] = pc;
        h.next = (h.next + 1u) % static_cast<uint32_t>(h.pcs.size());
        if (h.next == 0u)
        {
            h.wrapped = true;
        }
    }

    std::string formatDispatchHistory()
    {
        const DispatchHistory &h = currentDispatchRecoveryState().history;
        const uint32_t count = h.wrapped ? static_cast<uint32_t>(h.pcs.size()) : h.next;
        if (count == 0u)
        {
            return "(empty)";
        }

        std::ostringstream oss;
        bool first = true;
        for (uint32_t i = 0u; i < count; ++i)
        {
            const uint32_t idx = (h.next + h.pcs.size() - count + i) % static_cast<uint32_t>(h.pcs.size());
            if (!first)
            {
                oss << " -> ";
            }
            first = false;
            oss << "0x" << std::hex << h.pcs[idx];
        }
        return oss.str();
    }

    uint32_t selectDispatchRecoveryPc(const PS2Runtime *runtime)
    {
        const DispatchHistory &h = currentDispatchRecoveryState().history;
        const uint32_t count = h.wrapped ? static_cast<uint32_t>(h.pcs.size()) : h.next;
        if (count == 0u)
        {
            return 0u;
        }

        uint32_t firstHigh = 0u;
        for (uint32_t step = 1u; step <= count; ++step)
        {
            const uint32_t idx = (h.next + h.pcs.size() - step) % static_cast<uint32_t>(h.pcs.size());
            const uint32_t pc = h.pcs[idx];
            if (pc < 0x00100000u)
            {
                continue;
            }
            if (runtime && !runtime->hasFunction(pc))
            {
                continue;
            }

            if (firstHigh == 0u)
            {
                firstHigh = pc;
                continue;
            }

            return pc;
        }

        return firstHigh;
    }

    uint32_t selectExceptionVector(const R5900Context *ctx, bool tlbRefill)
    {
        if (ctx->cop0_status & COP0_STATUS_BEV)
        {
            return EXCEPTION_VECTOR_BOOT;
        }
        return tlbRefill ? EXCEPTION_VECTOR_TLB_REFILL : EXCEPTION_VECTOR_GENERAL;
    }

    // ---- VU0 macro mode: R5900Context <-> VU1State marshalling -----------
    //
    // Ported from ran-j/PS2Recomp origin/main (merged PR #48, "Issue 11 vu0
    // macro"). Adapted in two places for this tree's local API:
    //   * our VU1State::vi is int32_t[16] where upstream's is int16_t[16];
    //   * our VU1State has no `top` member (VU0 has no XTOP anyway) but does
    //     carry an extra `xitop`.
    // R5900Context's whole vu0_* region is byte-identical to upstream's, so
    // nothing here touches the corpus ABI.

    void seedVu0IdleSuccess(R5900Context *ctx)
    {
        if (!ctx)
        {
            return;
        }

        ctx->vu0_clip_flags = 0;
        ctx->vu0_clip_flags2 = 0;
        ctx->vu0_mac_flags = 0;
        ctx->vu0_status = 0;
        ctx->vu0_q = 1.0f;
        ctx->vu0_vpu_stat = 0;
        ctx->vu0_vpu_stat2 = 0;
    }

    void copyVu0ContextToState(const R5900Context *ctx, VU1State &state)
    {
        std::memset(&state, 0, sizeof(state));

        for (uint32_t i = 0; i < 32u; ++i)
        {
            _mm_storeu_ps(state.vf[i], ctx->vu0_vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            // ctx->vi[i] is the architectural 16-bit VI register, stored as
            // uint16_t. VU1State::vi is int32_t (a deliberate divergence
            // from upstream's int16_t, kept for this fork's wider VU1
            // interpreter). Casting uint16_t -> int32_t DIRECTLY does not
            // sign-extend: it zero-extends, so a negative VI value like
            // 0x8000 (-32768) becomes +32768 instead of -32768. Upstream
            // avoids this by casting through int16_t first
            // (static_cast<int16_t>(ctx->vi[i])), which reinterprets the
            // bit pattern as signed before the (correctly sign-extending)
            // widen to int32_t. Route through int16_t here too so entry
            // state matches upstream and real hardware.
            state.vi[i] = static_cast<int32_t>(static_cast<int16_t>(ctx->vi[i]));
        }

        _mm_storeu_ps(state.acc, ctx->vu0_acc);
        state.q = ctx->vu0_q;
        state.p = ctx->vu0_p;
        state.i = ctx->vu0_i;
        state.pc = ctx->vu0_pc;
        state.mac = ctx->vu0_mac_flags;
        state.clip = ctx->vu0_clip_flags;
        state.status = ctx->vu0_status;
        state.itop = ctx->vu0_itop;
        state.xitop = ctx->vu0_xitop;

        // VF0 is hardwired to (0,0,0,1) and VI0 to 0 on real VU hardware.
        state.vf[0][0] = 0.0f;
        state.vf[0][1] = 0.0f;
        state.vf[0][2] = 0.0f;
        state.vf[0][3] = 1.0f;
        state.vi[0] = 0;
    }

    void copyVu0StateToContext(const VU1State &state, R5900Context *ctx)
    {
        for (uint32_t i = 0; i < 32u; ++i)
        {
            ctx->vu0_vf[i] = _mm_loadu_ps(state.vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            ctx->vi[i] = static_cast<uint16_t>(state.vi[i]);
        }

        ctx->vu0_acc = _mm_loadu_ps(state.acc);
        ctx->vu0_q = state.q;
        ctx->vu0_p = state.p;
        ctx->vu0_i = state.i;
        ctx->vu0_mac_flags = state.mac;
        ctx->vu0_clip_flags = state.clip;
        ctx->vu0_clip_flags2 = state.clip;
        ctx->vu0_status = static_cast<uint16_t>(state.status);
        ctx->vu0_itop = state.itop;
        ctx->vu0_pc = state.pc;
        ctx->vu0_tpc = state.pc;
        ctx->vu0_vpu_stat = 0;
        ctx->vu0_vpu_stat2 = 0;

        ctx->vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
        ctx->vi[0] = 0;
    }

    // ---- PS2X_VU0_TRACE=1: bounded VU0 entry/exit probe ------------------
    // First 20 calls to micro address 0 only: logs the rotation-vector input
    // (vf2) and the resulting matrix rows (vf20..vf23) as floats, so a run
    // gives direct evidence of whether the microprogram computes a sane
    // rotation matrix from a known input, independent of everything
    // downstream in the scene graph.
    bool vu0TraceOn()
    {
        static const bool on = []()
        {
            const char *e = std::getenv("PS2X_VU0_TRACE");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        return on;
    }

    void vu0TraceEntry(uint32_t startPC, const VU1State &state)
    {
        if (startPC != 0u)
        {
            return;
        }
        static std::atomic<uint32_t> s_count{0};
        static const uint32_t kMaxVu0TraceEntryLogs =
            ps2DiagEnvLimit("PS2X_VU0_TRACE_ENTRY_MAX_LOGS", 20u);
        static std::atomic<bool> s_vu0TraceEntryTruncated{false};
        const uint32_t n = s_count.fetch_add(1, std::memory_order_relaxed);
        if (!ps2DiagLogBudget(std::cerr,
                              "[vu0:trace:entry]",
                              "PS2X_VU0_TRACE_ENTRY_MAX_LOGS",
                              kMaxVu0TraceEntryLogs,
                              n,
                              s_vu0TraceEntryTruncated))
        {
            return;
        }
        std::cerr << "[vu0:trace] entry n=" << n
                  << " vf2=" << state.vf[2][0] << ',' << state.vf[2][1] << ','
                  << state.vf[2][2] << ',' << state.vf[2][3]
                  << " acc=" << state.acc[0] << ',' << state.acc[1] << ','
                  << state.acc[2] << ',' << state.acc[3]
                  << " q=" << state.q << " i=" << state.i
                  << std::endl;
    }

    void vu0TraceExit(uint32_t startPC, const VU1State &state)
    {
        if (startPC != 0u)
        {
            return;
        }
        static std::atomic<uint32_t> s_count{0};
        static const uint32_t kMaxVu0TraceExitLogs =
            ps2DiagEnvLimit("PS2X_VU0_TRACE_EXIT_MAX_LOGS", 20u);
        static std::atomic<bool> s_vu0TraceExitTruncated{false};
        const uint32_t n = s_count.fetch_add(1, std::memory_order_relaxed);
        if (!ps2DiagLogBudget(std::cerr,
                              "[vu0:trace:exit]",
                              "PS2X_VU0_TRACE_EXIT_MAX_LOGS",
                              kMaxVu0TraceExitLogs,
                              n,
                              s_vu0TraceExitTruncated))
        {
            return;
        }
        std::cerr << "[vu0:trace] exit  n=" << n
                  << " vf20=" << state.vf[20][0] << ',' << state.vf[20][1] << ','
                  << state.vf[20][2] << ',' << state.vf[20][3]
                  << " vf21=" << state.vf[21][0] << ',' << state.vf[21][1] << ','
                  << state.vf[21][2] << ',' << state.vf[21][3]
                  << " vf22=" << state.vf[22][0] << ',' << state.vf[22][1] << ','
                  << state.vf[22][2] << ',' << state.vf[22][3]
                  << " vf23=" << state.vf[23][0] << ',' << state.vf[23][1] << ','
                  << state.vf[23][2] << ',' << state.vf[23][3]
                  << std::endl;
    }

    // ---- vcallms-per-second census (always on, cheap) ---------------------
    // Quantifies how often executeVU0Microprogram actually runs the real
    // interpreter (not the seedVu0IdleSuccess early-outs), i.e. how often the
    // shared VU pipeline-timing globals get fenced away from VU1 mid-frame.
    // One atomic increment per call plus a rate-limited print; no locks.
    void vu0CensusNoteCall()
    {
        static std::atomic<uint64_t> s_total{0};
        static std::atomic<uint64_t> s_lastReportTotal{0};
        static std::atomic<uint64_t> s_lastReportMs{0};

        const uint64_t total = s_total.fetch_add(1, std::memory_order_relaxed) + 1;

        const auto now = std::chrono::steady_clock::now();
        const uint64_t nowMs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
        uint64_t lastMs = s_lastReportMs.load(std::memory_order_relaxed);
        if (nowMs - lastMs < 2000u)
        {
            return;
        }
        if (!s_lastReportMs.compare_exchange_strong(lastMs, nowMs, std::memory_order_relaxed))
        {
            return; // another thread just reported
        }
        const uint64_t lastTotal = s_lastReportTotal.exchange(total, std::memory_order_relaxed);
        const uint64_t deltaCalls = total - lastTotal;
        const uint64_t deltaMs = (lastMs == 0u) ? 0u : (nowMs - lastMs);
        const double perSec = (deltaMs > 0u) ? (static_cast<double>(deltaCalls) * 1000.0 / static_cast<double>(deltaMs)) : 0.0;
        std::cerr << "[vu0:census] totalCalls=" << total
                  << " deltaCalls=" << deltaCalls
                  << " callsPerSec=" << perSec << std::endl;
    }

    void raiseCop0Exception(R5900Context *ctx, uint32_t exceptionCode, bool tlbRefill = false)
    {
        if (ctx->in_delay_slot)
        {
            ctx->cop0_epc = ctx->branch_pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~COP0_CAUSE_EXCCODE_MASK) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK) |
                              COP0_CAUSE_BD;
        }
        else
        {
            ctx->cop0_epc = ctx->pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~(COP0_CAUSE_EXCCODE_MASK | COP0_CAUSE_BD)) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK);
        }

        ctx->cop0_status |= COP0_STATUS_EXL;
        ctx->pc = selectExceptionVector(ctx, tlbRefill);
        ctx->in_delay_slot = false;
    }

    std::filesystem::path normalizeAbsolutePath(const std::filesystem::path &path)
    {
        if (path.empty())
        {
            return {};
        }

#if defined(PLATFORM_VITA)
        const std::string generic = path.generic_string();
        const std::size_t colon = generic.find(':');
        if (colon != std::string::npos && colon != 0u)
        {
            const std::size_t slash = generic.find_first_of("/\\");
            if (slash == std::string::npos || colon < slash)
            {
                return path.lexically_normal();
            }
        }
#endif

        std::error_code ec;
        const std::filesystem::path absolute = std::filesystem::absolute(path, ec);
        if (ec)
        {
            return path.lexically_normal();
        }
        return absolute.lexically_normal();
    }

    PS2Runtime::IoPaths &runtimeIoPaths()
    {
        static PS2Runtime::IoPaths paths = []()
        {
            PS2Runtime::IoPaths defaults;
            std::error_code ec;
            const std::filesystem::path cwd = std::filesystem::current_path(ec);
            defaults.elfDirectory = ec ? std::filesystem::path(".") : cwd.lexically_normal();
            defaults.hostRoot = defaults.elfDirectory;
            defaults.cdRoot = defaults.elfDirectory;
            defaults.mcRoot = defaults.elfDirectory / "mc0";
            return defaults;
        }();

        return paths;
    }

    uint32_t readGuestU32Wrapped(const uint8_t *rdram, uint32_t addr)
    {
        if (!rdram)
        {
            return 0;
        }

        uint32_t value = 0;
        value |= static_cast<uint32_t>(rdram[(addr + 0u) & PS2_RAM_MASK]) << 0;
        value |= static_cast<uint32_t>(rdram[(addr + 1u) & PS2_RAM_MASK]) << 8;
        value |= static_cast<uint32_t>(rdram[(addr + 2u) & PS2_RAM_MASK]) << 16;
        value |= static_cast<uint32_t>(rdram[(addr + 3u) & PS2_RAM_MASK]) << 24;
        return value;
    }

    uint64_t readGuestU64Wrapped(const uint8_t *rdram, uint32_t addr)
    {
        const uint64_t lo = readGuestU32Wrapped(rdram, addr);
        const uint64_t hi = readGuestU32Wrapped(rdram, addr + 4u);
        return lo | (hi << 32);
    }

    uint32_t selectStackRecoveryPc(const uint8_t *rdram, const R5900Context *ctx, const PS2Runtime *runtime)
    {
        if (!rdram || !ctx || !runtime)
        {
            return 0u;
        }

        const uint32_t sp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0));
        constexpr uint32_t kScanBytes = 0x200u;

        for (uint32_t offset = 0u; offset < kScanBytes; offset += 8u)
        {
            const uint32_t slotAddr = sp + offset;
            const uint32_t ra32 = static_cast<uint32_t>(readGuestU64Wrapped(rdram, slotAddr));
            if (ra32 < 0x00100000u)
            {
                continue;
            }
            if (!runtime->hasFunction(ra32))
            {
                continue;
            }
            return ra32;
        }

        for (uint32_t offset = 0u; offset < kScanBytes; offset += 4u)
        {
            const uint32_t slotAddr = sp + offset;
            const uint32_t ra32 = readGuestU32Wrapped(rdram, slotAddr);
            if (ra32 < 0x00100000u)
            {
                continue;
            }
            if (!runtime->hasFunction(ra32))
            {
                continue;
            }
            return ra32;
        }

        return 0u;
    }

    std::string readGuestPrintableString(const uint8_t *rdram, uint32_t addr, size_t maxLen)
    {
        std::string out;
        if (!rdram || maxLen == 0)
        {
            return out;
        }

        out.reserve(std::min<size_t>(maxLen, 64));
        for (size_t i = 0; i < maxLen; ++i)
        {
            const char ch = static_cast<char>(rdram[(addr + static_cast<uint32_t>(i)) & PS2_RAM_MASK]);
            if (ch == '\0')
            {
                break;
            }
            if (ch >= 0x20 && ch < 0x7F)
            {
                out.push_back(ch);
            }
            else
            {
                out.push_back('.');
            }
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // PS2X_VU1_MEMDUMP diagnostic: env-gated, bounded dump of VU1 DATA memory
    // (16KB / 1024 quadwords) taken at MSCAL (microprogram start), BEFORE the
    // microprogram runs. This captures what was UPLOADED into VU1 memory, to
    // separate "wrong data arrived in VU1 memory" (upload path broken) from
    // "correct data arrived and was executed wrongly" (interpreter pipeline
    // model broken).
    //
    // OFF by default. A prior ad-hoc dump-every-MSCAL diagnostic wrote 3.43 GB
    // and delayed the run past its scripted input timing so the run never
    // reached FIELD. This instead takes a BOUNDED, time-spaced number of
    // snapshots and loudly announces the final tally (taken vs skipped) --
    // unannounced truncation is this project's most repeated failure mode.
    //
    // Env:
    //   PS2X_VU1_MEMDUMP=<path prefix>        off if unset/empty
    //   PS2X_VU1_MEMDUMP_COUNT=<n>            default 4
    //   PS2X_VU1_MEMDUMP_INTERVAL_MS=<ms>     min spacing between snapshots, default 60000
    //
    // Each snapshot is written to "<prefix>-<n>.bin", exactly PS2_VU1_DATA_SIZE
    // (16384) bytes.
    //
    // Called from BOTH microprogram-activation instructions VIF1 recognises
    // (MSCAL, which specifies a start address, and MSCNT, which resumes from
    // wherever VU1's PC already is) -- an early run of this diagnostic showed
    // ZERO MSCALs and ZERO snapshots for a 363s boot->FIELD run, which turned
    // out to mean DQ8 drives VU1 via MSCNT here, not MSCAL; instrumenting only
    // MSCAL would silently miss all of it. "kind" is "MSCAL" or "MSCNT" for
    // the log line only; startPC is the sentinel 0xFFFFFFFF for MSCNT since
    // that instruction carries no start address (it resumes in place).
    void maybeDumpVu1Memory(const uint8_t *vu1Data, uint32_t startPC, const char *kind)
    {
        static const char *prefix = std::getenv("PS2X_VU1_MEMDUMP");
        if (!prefix || !prefix[0])
        {
            return;
        }

        static const int maxDumps = []
        {
            if (const char *s = std::getenv("PS2X_VU1_MEMDUMP_COUNT"))
            {
                const int v = std::atoi(s);
                if (v > 0)
                {
                    return v;
                }
            }
            return 4;
        }();

        static const int64_t intervalMs = []
        {
            if (const char *s = std::getenv("PS2X_VU1_MEMDUMP_INTERVAL_MS"))
            {
                const long v = std::atol(s);
                if (v > 0)
                {
                    return static_cast<int64_t>(v);
                }
            }
            return static_cast<int64_t>(60000);
        }();

        static const auto s_t0 = std::chrono::steady_clock::now();
        static std::mutex s_mu;
        static int s_dumped = 0;
        static uint64_t s_skipped = 0;
        static int64_t s_nextDueMs = 0;
        static bool s_announcedCap = false;

        const auto now = std::chrono::steady_clock::now();
        const int64_t elapsedMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - s_t0).count();

        std::lock_guard<std::mutex> lk(s_mu);

        if (s_dumped >= maxDumps)
        {
            ++s_skipped;
            if (!s_announcedCap)
            {
                s_announcedCap = true;
                std::cerr << "[vu1:memdump] cap reached: " << s_dumped << "/" << maxDumps
                          << " snapshot(s) taken; ALL further MSCALs are now being SKIPPED"
                          << " (raise PS2X_VU1_MEMDUMP_COUNT to capture more)." << std::endl;
            }
            return;
        }

        if (elapsedMs < s_nextDueMs)
        {
            ++s_skipped;
            return;
        }

        char path[512];
        std::snprintf(path, sizeof(path), "%s-%d.bin", prefix, s_dumped);
        FILE *f = std::fopen(path, "wb");
        if (!f)
        {
            std::cerr << "[vu1:memdump] FAILED to open '" << path << "' for writing" << std::endl;
            return;
        }
        std::fwrite(vu1Data, 1, PS2_VU1_DATA_SIZE, f);
        std::fclose(f);

        ++s_dumped;
        s_nextDueMs = elapsedMs + intervalMs;

        std::cerr << "[vu1:memdump] snapshot " << s_dumped << "/" << maxDumps << " -> " << path
                   << "  kind=" << kind << "  startPC=0x" << std::hex << startPC << std::dec
                   << "  t=" << elapsedMs << "ms  (skipped so far: " << s_skipped << ")"
                   << std::endl;

        if (s_dumped >= maxDumps)
        {
            std::cerr << "[vu1:memdump] cap of " << maxDumps
                       << " snapshot(s) reached on this dump; further activations will be SKIPPED."
                       << std::endl;
        }
    }
}

static void UploadFrame(Texture2D &tex, PS2Runtime *rt, uint32_t &outWidth, uint32_t &outHeight)
{
    static uint64_t s_lastPresentationTick = std::numeric_limits<uint64_t>::max();
    static bool s_hasLatchedInitialFrame = false;
    static uint32_t s_lastDisplayFbp = std::numeric_limits<uint32_t>::max();
    static uint32_t s_lastSourceFbp = std::numeric_limits<uint32_t>::max();
    static bool s_lastPreferred = false;
    static uint32_t s_lastWidth = 0u;
    static uint32_t s_lastHeight = 0u;

    const uint64_t currentTick = ps2_syscalls::GetCurrentVSyncTick();
    if (!s_hasLatchedInitialFrame || currentTick != s_lastPresentationTick)
    {
        rt->gs().latchHostPresentationFrame();
        rt->gs().debugDumpFieldFramebuffers();
        s_lastPresentationTick = currentTick;
        s_hasLatchedInitialFrame = true;
    }

    std::vector<uint8_t> scratch;
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t displayFbp = 0u;
    uint32_t sourceFbp = 0u;
    bool usedPreferredDisplaySource = false;
    if (!rt->gs().copyLatchedHostPresentationFrame(scratch,
                                                   width,
                                                   height,
                                                   &displayFbp,
                                                   &sourceFbp,
                                                   &usedPreferredDisplaySource))
    {
        Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, MAGENTA);
        UpdateTexture(tex, blank.data);
        UnloadImage(blank);
        outWidth = FB_WIDTH;
        outHeight = DEFAULT_DISPLAY_HEIGHT;
        return;
    }

    PS2_IF_AGRESSIVE_LOGS({
        static uint32_t s_uploadDebugCount = 0u;
        if (s_uploadDebugCount < 128u ||
            displayFbp != s_lastDisplayFbp ||
            sourceFbp != s_lastSourceFbp ||
            usedPreferredDisplaySource != s_lastPreferred ||
            width != s_lastWidth ||
            height != s_lastHeight)
        {
            std::cout << "[frame:upload] idx=" << s_uploadDebugCount
                      << " tick=" << currentTick
                      << " displayFbp=" << displayFbp
                      << " sourceFbp=" << sourceFbp
                      << " size=" << width << "x" << height
                      << " preferred=" << static_cast<uint32_t>(usedPreferredDisplaySource ? 1u : 0u)
                      << std::endl;
        }
        static uint32_t s_probeDebugCount = 0u;
        if (s_probeDebugCount < 32u ||
            displayFbp != s_lastDisplayFbp ||
            sourceFbp != s_lastSourceFbp ||
            usedPreferredDisplaySource != s_lastPreferred)
        {
            std::cout << "[frame:probe] idx=" << s_probeDebugCount
                      << " tick=" << currentTick
                      << " displayFbp=" << displayFbp
                      << " sourceFbp=" << sourceFbp
                      << " preferred=" << static_cast<uint32_t>(usedPreferredDisplaySource ? 1u : 0u);
            for (const auto &probe : kGhostProbePoints)
            {
                if (probe.x >= width || probe.y >= height)
                {
                    continue;
                }

                const uint32_t pixel = sampleHostFramePixel(scratch, width, height, probe.x, probe.y);
                std::cout << " host[" << probe.x << "," << probe.y << "]=0x"
                          << std::hex << pixel << std::dec;
            }
            std::cout << std::endl;
            ++s_probeDebugCount;
        }
        ++s_uploadDebugCount;
    });
    s_lastDisplayFbp = displayFbp;
    s_lastSourceFbp = sourceFbp;
    s_lastPreferred = usedPreferredDisplaySource;
    s_lastWidth = width;
    s_lastHeight = height;

    std::vector<uint8_t> uploadBuffer(DEFAULT_FB_SIZE, 0u);
    if (!scratch.empty() && width != 0u && height != 0u)
    {
        const uint32_t copyWidth = std::min<uint32_t>(width, FB_WIDTH);
        const uint32_t copyHeight = std::min<uint32_t>(height, FB_HEIGHT);
        const size_t srcRowBytes = static_cast<size_t>(width) * 4u;
        const size_t dstRowBytes = static_cast<size_t>(FB_WIDTH) * 4u;
        const size_t copyRowBytes = static_cast<size_t>(copyWidth) * 4u;
        for (uint32_t y = 0; y < copyHeight; ++y)
        {
            const size_t srcOffset = static_cast<size_t>(y) * srcRowBytes;
            const size_t dstOffset = static_cast<size_t>(y) * dstRowBytes;
            if (srcOffset + copyRowBytes > scratch.size() ||
                dstOffset + copyRowBytes > uploadBuffer.size())
            {
                break;
            }
            std::memcpy(uploadBuffer.data() + dstOffset, scratch.data() + srcOffset, copyRowBytes);
        }
    }

    UpdateTexture(tex, uploadBuffer.data());
    outWidth = width;
    outHeight = height;
}

PS2Runtime::PS2Runtime()
{
    std::memset(&m_cpuContext, 0, sizeof(m_cpuContext));

    // R0 is always zero in MIPS
    m_cpuContext.r[0] = _mm_set1_epi32(0);

    // VU0 vf0 is hardwired on real hardware to the constant (x,y,z,w) =
    // (0,0,0,1). The memset above leaves it at (0,0,0,0); nothing in
    // recompiled VU0-macro-mode code writes vf0 (it is a read-only constant
    // register, same as MIPS $zero), so without this it silently stays zero
    // forever. Reads of vf0.w feed matrix-inverse cofactor divides used by
    // bone-palette/skinning math — an unpinned vf0 collapses every skinned
    // mesh to a point while rigid geometry (which never reads vf0) looks
    // fine. See dq8/reference/dc2-learnings/04-vu-interpreter-correctness.md.
    m_cpuContext.vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f); // lane3=w=1, lanes0-2=x,y,z=0

    // Stack pointer (SP) and global pointer (GP) will be set by the loaded ELF

    m_functionTable.clear();

    m_loadedModules.clear();
    m_guestHeapBlocks.clear();
    m_guestHeapBase = kGuestHeapDefaultBase;
    m_guestHeapEnd = kGuestHeapDefaultBase;
    m_guestHeapLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    m_guestHeapSuggestedBase = kGuestHeapDefaultBase;
    m_guestHeapConfigured = false;
    // Kernel-area pool, disjoint from all guest memory (see the layout
    // comment at kAsyncCallbackStackFloor). Was floor=kGuestHeapHardLimit /
    // top=PS2_RAM_SIZE, which put callback stacks inside the guest's own
    // main stack.
    m_asyncCallbackStackFloor = kAsyncCallbackStackFloor;
    m_asyncCallbackStackTop = kAsyncCallbackStackTop;
}

PS2Runtime::~PS2Runtime()
{
    try
    {
        requestStop();
        // Fiber pool is cleaned up by scheduler_shutdown() in run().
#if defined(PLATFORM_VITA)
        m_audioBackend.stopAll();
        m_audioBackend.setAudioReady(false);
#else
        if (IsAudioDeviceReady())
        {
            CloseAudioDevice();
            m_audioBackend.setAudioReady(false);
        }
#endif
        if (IsWindowReady())
        {
            CloseWindow();
        }

        m_loadedModules.clear();

        m_functionTable.clear();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: unknown" << std::endl;
    }
}

bool PS2Runtime::syncCoreSubsystems()
{
    uint8_t *const rdram = m_memory.getRDRAM();
    uint8_t *const gsVram = m_memory.getGSVRAM();
    if (!rdram || !gsVram)
    {
        return false;
    }

    if (m_boundRdram == rdram && m_boundGSVram == gsVram)
    {
        return true;
    }

    m_gs.init(gsVram, static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &m_memory.gs());
    m_gifArbiter.setProcessPacketFn([this](const uint8_t *data, uint32_t size)
                                    { m_gs.processGIFPacket(data, size); });
    m_memory.setGifArbiter(&m_gifArbiter);
    // DC2 spec 04 / G28: XTOP returns the VIF1 double-buffer **TOP**, XITOP
    // returns ITOP; conflating them makes a double-buffered VU1 program read
    // its input from the wrong half.
    //
    // TOP is latched from TOPS at microprogram activation, BEFORE the DBF flip
    // (ps2_vif1_interpreter.cpp latchVu1Activation, mirroring PCSX2
    // Vif_Codes.cpp vuExecMicro) -- it names the half VIF1 has just finished
    // unpacking into, which is what the program is about to read. This used to
    // pass TOPS *after* the flip, i.e. the half VIF1 will write NEXT, so every
    // batch transformed the other buffer.
    //
    // VU1State::xitop is the field carrying whatever XTOP reports (see
    // VU1Interpreter::execLower case 0x68); it is plumbed here rather than by
    // widening the MSCAL callback, which would touch ps2_memory.h /
    // ps2_runtime.h and rebuild every dq8/output TU.
    m_memory.setVu1MscalCallback([this](uint32_t startPC, uint32_t itop)
                                 { m_vu1.state().xitop = m_memory.vif1_regs.top & 0x3FFu;
                                   // Dump BEFORE execute: this is what arrived via upload,
                                   // not what the microprogram computes. See PS2X_VU1_MEMDUMP.
                                   maybeDumpVu1Memory(m_memory.getVU1Data(), startPC, "MSCAL");
                                   m_vu1.execute(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                 m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                 m_gs, &m_memory, startPC, itop, 65536); });
    m_memory.setVu1MscntCallback([this](uint32_t itop)
                                 { m_vu1.state().xitop = m_memory.vif1_regs.top & 0x3FFu;
                                   // See PS2X_VU1_MEMDUMP above: MSCNT is DQ8's other VU1
                                   // activation instruction and needs the same coverage.
                                   maybeDumpVu1Memory(m_memory.getVU1Data(), 0xFFFFFFFFu, "MSCNT");
                                   m_vu1.resume(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                m_gs, &m_memory, itop, 65536); });
    m_iop.init(rdram);
    m_iop.reset();
    m_vu1.reset();

    m_boundRdram = rdram;
    m_boundGSVram = gsVram;
    return true;
}

bool PS2Runtime::initialize(const char *title)
{
    try
    {
        if (!m_memory.initialize())
        {
            std::cerr << "Failed to initialize PS2 memory" << std::endl;
            return false;
        }

        if (!syncCoreSubsystems())
        {
            std::cerr << "Failed to bind runtime core subsystems" << std::endl;
            return false;
        }

#if defined(PLATFORM_VITA)
        InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, title); // raylib vita does not support audio
#else
        SetConfigFlags(FLAG_WINDOW_RESIZABLE);
        InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, title);
        InitAudioDevice();
        m_audioBackend.setAudioReady(IsAudioDeviceReady());
#endif
        SetTargetFPS(60);

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to initialize PS2 runtime: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Failed to initialize PS2 runtime: unknown exception" << std::endl;
    }

    return false;
}

bool PS2Runtime::loadELF(const std::string &elfPath)
{
    configureIoPathsFromElf(elfPath);

    std::ifstream file(elfPath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open ELF file: " << elfPath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    const std::streamoff fileSize = file.tellg();
    if (fileSize < static_cast<std::streamoff>(sizeof(ElfHeader)))
    {
        std::cerr << "ELF file is too small: " << elfPath << std::endl;
        return false;
    }
    file.seekg(0, std::ios::beg);

    ElfHeader header{};
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header)))
    {
        std::cerr << "Failed to read ELF header from: " << elfPath << std::endl;
        return false;
    }

    if (header.magic != ELF_MAGIC)
    {
        std::cerr << "Invalid ELF magic number" << std::endl;
        return false;
    }

    if (header.elf_class != 1u || header.endianness != 1u)
    {
        std::cerr << "Unsupported ELF format (expected 32-bit little-endian)." << std::endl;
        return false;
    }

    if (header.machine != EM_MIPS || header.type != ET_EXEC)
    {
        std::cerr << "Not a MIPS executable ELF file" << std::endl;
        return false;
    }

    if (header.phnum != 0u && header.phentsize < sizeof(ProgramHeader))
    {
        std::cerr << "Unsupported ELF program-header entry size: " << header.phentsize << std::endl;
        return false;
    }

    const uint64_t programHeaderTableEnd =
        static_cast<uint64_t>(header.phoff) +
        static_cast<uint64_t>(header.phnum) * static_cast<uint64_t>(header.phentsize);
    if (programHeaderTableEnd > static_cast<uint64_t>(fileSize))
    {
        std::cerr << "ELF program-header table is out of range." << std::endl;
        return false;
    }

    m_cpuContext.pc = header.entry;

    uint32_t maxLoadedRdramEnd = kGuestHeapDefaultBase;
    uint32_t moduleBase = std::numeric_limits<uint32_t>::max();
    uint32_t moduleEnd = 0u;
    bool loadedAnySegment = false;

    for (uint16_t i = 0; i < header.phnum; i++)
    {
        const uint64_t phOffset =
            static_cast<uint64_t>(header.phoff) +
            static_cast<uint64_t>(i) * static_cast<uint64_t>(header.phentsize);
        if (phOffset + sizeof(ProgramHeader) > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF program header " << i << " is out of range." << std::endl;
            return false;
        }

        ProgramHeader ph{};
        file.seekg(static_cast<std::streamoff>(phOffset), std::ios::beg);
        if (!file.read(reinterpret_cast<char *>(&ph), sizeof(ph)))
        {
            std::cerr << "Failed to read ELF program header " << i << std::endl;
            return false;
        }

        if (ph.type != PT_LOAD || ph.memsz == 0u)
        {
            continue;
        }

        if (ph.filesz > ph.memsz)
        {
            std::cerr << "ELF segment " << i << " has filesz > memsz." << std::endl;
            return false;
        }

        const uint64_t segmentFileEnd = static_cast<uint64_t>(ph.offset) + static_cast<uint64_t>(ph.filesz);
        if (segmentFileEnd > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF segment " << i << " exceeds file bounds." << std::endl;
            return false;
        }

        const bool scratch =
            ph.vaddr >= PS2_SCRATCHPAD_BASE &&
            ph.vaddr < (PS2_SCRATCHPAD_BASE + PS2_SCRATCHPAD_SIZE);

        uint32_t physAddr = 0u;
        try
        {
            physAddr = m_memory.translateAddress(ph.vaddr);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to translate ELF segment " << i
                      << " virtual address 0x" << std::hex << ph.vaddr
                      << std::dec << ": " << e.what() << std::endl;
            return false;
        }
        const uint64_t regionSize = scratch ? static_cast<uint64_t>(PS2_SCRATCHPAD_SIZE)
                                            : static_cast<uint64_t>(PS2_RAM_SIZE);
        const uint64_t segmentMemEnd = static_cast<uint64_t>(physAddr) + static_cast<uint64_t>(ph.memsz);
        if (segmentMemEnd > regionSize)
        {
            std::cerr << "ELF segment " << i << " exceeds "
                      << (scratch ? "scratchpad" : "RDRAM")
                      << " bounds (vaddr=0x" << std::hex << ph.vaddr
                      << " memsz=0x" << ph.memsz << std::dec << ")." << std::endl;
            return false;
        }

        uint8_t *destBase = scratch ? m_memory.getScratchpad() : m_memory.getRDRAM();
        if (!destBase)
        {
            std::cerr << "ELF segment " << i << " has no destination memory backing." << std::endl;
            return false;
        }

        uint8_t *dest = destBase + physAddr;
        if (ph.filesz > 0u)
        {
            file.seekg(static_cast<std::streamoff>(ph.offset), std::ios::beg);
            if (!file.read(reinterpret_cast<char *>(dest), ph.filesz))
            {
                std::cerr << "Failed to read ELF segment " << i << " payload." << std::endl;
                return false;
            }
        }

        if (ph.memsz > ph.filesz)
        {
            std::memset(dest + ph.filesz, 0, ph.memsz - ph.filesz);
        }

        RUNTIME_LOG("Loading segment: 0x" << std::hex << ph.vaddr
                                          << " - 0x" << (static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz))
                                          << " (filesz: 0x" << ph.filesz
                                          << ", memsz: 0x" << ph.memsz << ")"
                                          << std::dec << std::endl);

        if (!scratch)
        {
            maxLoadedRdramEnd = std::max(maxLoadedRdramEnd, static_cast<uint32_t>(segmentMemEnd));
        }

        if (ph.flags & 0x1u) // PF_X
        {
            const uint64_t execEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz);
            if (execEnd <= std::numeric_limits<uint32_t>::max())
            {
                m_memory.registerCodeRegion(ph.vaddr, static_cast<uint32_t>(execEnd));
            }
        }

        loadedAnySegment = true;
        moduleBase = std::min(moduleBase, ph.vaddr);
        const uint64_t segmentVirtualEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz);
        const uint32_t clampedVirtualEnd =
            (segmentVirtualEnd > std::numeric_limits<uint32_t>::max())
                ? std::numeric_limits<uint32_t>::max()
                : static_cast<uint32_t>(segmentVirtualEnd);
        moduleEnd = std::max(moduleEnd, clampedVirtualEnd);
    }

    if (!loadedAnySegment)
    {
        std::cerr << "ELF contains no loadable PT_LOAD segments." << std::endl;
        return false;
    }

    if (maxLoadedRdramEnd > PS2_RAM_SIZE)
    {
        maxLoadedRdramEnd = PS2_RAM_SIZE;
    }

    const uint32_t paddedEnd = (maxLoadedRdramEnd > (PS2_RAM_SIZE - kGuestHeapSafetyPad))
                                   ? PS2_RAM_SIZE
                                   : (maxLoadedRdramEnd + kGuestHeapSafetyPad);
    const uint32_t suggestedHeapBase = alignGuestHeapValue(paddedEnd, kGuestHeapDefaultAlignment);
    {
        std::lock_guard<std::mutex> lock(m_guestHeapMutex);
        if (!m_guestHeapConfigured)
        {
            const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
            m_guestHeapSuggestedBase = std::min(suggestedHeapBase, hardLimit);
            m_guestHeapBase = m_guestHeapSuggestedBase;
            m_guestHeapEnd = m_guestHeapSuggestedBase;
            m_guestHeapLimit = hardLimit;
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
        // Kernel-area pool (see layout comment at kAsyncCallbackStackFloor).
        // suggestedHeapBase no longer participates: the pool sits BELOW the
        // ELF load base, so the heap (which starts at the ELF's bss end) can
        // never reach it.
        m_asyncCallbackStackFloor = kAsyncCallbackStackFloor;
        m_asyncCallbackStackTop = kAsyncCallbackStackTop;
    }

    LoadedModule module;
    module.name = elfPath.substr(elfPath.find_last_of("/\\") + 1);
    module.baseAddress = (moduleBase == std::numeric_limits<uint32_t>::max()) ? 0x00100000u : moduleBase;
    module.size = (moduleEnd > module.baseAddress) ? static_cast<size_t>(moduleEnd - module.baseAddress) : 0u;
    module.active = true;

    m_loadedModules.push_back(module);

    ps2_game_overrides::applyMatching(*this, elfPath, m_cpuContext.pc);

    RUNTIME_LOG("ELF file loaded successfully. Entry point: 0x" << std::hex << m_cpuContext.pc << std::dec);
    return true;
}

const PS2Runtime::IoPaths &PS2Runtime::getIoPaths()
{
    return runtimeIoPaths();
}

void PS2Runtime::setIoPaths(const IoPaths &paths)
{
    IoPaths normalized = paths;
    normalized.elfPath = normalizeAbsolutePath(normalized.elfPath);
    normalized.elfDirectory = normalizeAbsolutePath(normalized.elfDirectory);
    normalized.hostRoot = normalizeAbsolutePath(normalized.hostRoot);
    normalized.cdRoot = normalizeAbsolutePath(normalized.cdRoot);
    normalized.mcRoot = normalizeAbsolutePath(normalized.mcRoot);
    normalized.cdImage = normalizeAbsolutePath(normalized.cdImage);

    if (normalized.elfDirectory.empty() && !normalized.elfPath.empty())
    {
        normalized.elfDirectory = normalized.elfPath.parent_path();
    }

    if (normalized.hostRoot.empty())
    {
        normalized.hostRoot = normalized.elfDirectory;
    }
    if (normalized.cdRoot.empty())
    {
        normalized.cdRoot = normalized.elfDirectory;
    }
    if (normalized.mcRoot.empty())
    {
        normalized.mcRoot = normalized.elfDirectory / "mc0";
    }

    runtimeIoPaths() = normalized;
}

void PS2Runtime::configureIoPathsFromElf(const std::string &elfPath)
{
    IoPaths paths = runtimeIoPaths();
    paths.elfPath = normalizeAbsolutePath(std::filesystem::path(elfPath));
    if (!paths.elfPath.empty())
    {
        paths.elfDirectory = paths.elfPath.parent_path();
    }

    if (!paths.elfDirectory.empty())
    {
        paths.hostRoot = paths.elfDirectory;
        paths.cdRoot = paths.elfDirectory;
        paths.mcRoot = paths.elfDirectory / "mc0";
    }

    setIoPaths(paths);
}

void PS2Runtime::registerFunction(uint32_t address, RecompiledFunction func)
{
    m_functionTable[address] = func;
}

bool PS2Runtime::hasFunction(uint32_t address) const
{
    auto it = m_functionTable.find(address);
    if (it != m_functionTable.end())
    {
        return true;
    }

    return false;
}

PS2Runtime::RecompiledFunction PS2Runtime::lookupFunction(uint32_t address)
{
    pushDispatchPc(address);

    auto it = m_functionTable.find(address);
    if (it != m_functionTable.end())
    {
        return it->second;
    }

    std::cerr << "Warning: Function at address 0x" << std::hex << address << std::dec << " not found" << std::endl;

    static RecompiledFunction defaultFunction = [](uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t ra = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)) : 0u;
        const uint32_t sp = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0)) : 0u;
        const uint32_t gp = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0)) : 0u;
        const uint32_t a0 = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[4], 0)) : 0u;
        const uint32_t a1 = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[5], 0)) : 0u;
        const uint32_t v0 = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[2], 0)) : 0u;
        const uint32_t v1 = ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[3], 0)) : 0u;

        if (ctx && runtime)
        {
            // Per-guest-thread recovery budget/log flag (see
            // currentDispatchRecoveryState): a sick fiber exhausting its 8192
            // budget no longer eats the budget of healthy fibers, and each
            // fiber gets its own [dispatch:first-bad-pc] context line.
            DispatchRecoveryState &rec = currentDispatchRecoveryState();
            uint32_t &s_recoverCount = rec.recoverCount;
            bool &s_loggedContext = rec.loggedContext;
            const uint32_t pc = ctx->pc;
            const bool hasPcFunction = runtime->hasFunction(pc);

            if (!hasPcFunction && s_recoverCount < 8192u)
            {
                if (!s_loggedContext)
                {
                    std::ostringstream stackDump;
                    if (rdram)
                    {
                        stackDump << " [stack]";
                        for (uint32_t off = 0u; off < 0x40u; off += 4u)
                        {
                            const uint32_t slot = readGuestU32Wrapped(rdram, sp + off);
                            stackDump << " +" << std::hex << off << "=0x" << slot;
                        }
                    }
                    std::cerr << "[dispatch:first-bad-pc] tid=" << std::dec << g_currentThreadId
                              << " bad=0x" << std::hex << pc
                              << " ra=0x" << ra
                              << " sp=0x" << sp
                              << " gp=0x" << gp
                              << " v0=0x" << v0
                              << " v1=0x" << v1
                              << " a0=0x" << a0
                              << " a1=0x" << a1
                              << " trace=" << formatDispatchHistory()
                              << stackDump.str()
                              << std::dec << std::endl;
                    s_loggedContext = true;
                }

                uint32_t recoveryPc = 0u;
                if (ra != 0u && runtime->hasFunction(ra))
                {
                    recoveryPc = ra;
                }

                if (recoveryPc == 0u)
                {
                    recoveryPc = selectStackRecoveryPc(rdram, ctx, runtime);
                }

                if (recoveryPc == 0u)
                {
                    recoveryPc = selectDispatchRecoveryPc(runtime);
                }

                if (recoveryPc != 0u && recoveryPc != pc)
                {
                    if (s_recoverCount < 256u)
                    {
                        std::cerr << "[dispatch:recover-pc] bad=0x" << std::hex << pc
                                  << " ra=0x" << ra
                                  << " fallback=0x" << recoveryPc
                                  << " sp=0x" << sp
                                  << std::dec << std::endl;
                    }
                    ++s_recoverCount;
                    ctx->pc = recoveryPc;
                    return;
                }
            }

            if (hasPcFunction)
            {
                s_recoverCount = 0u;
                s_loggedContext = false;
            }
            else if (pc < 0x00100000u && ra == pc && s_recoverCount < 4096u)
            {
                uint32_t recoveryPc = selectStackRecoveryPc(rdram, ctx, runtime);
                if (recoveryPc == 0u)
                {
                    recoveryPc = selectDispatchRecoveryPc(runtime);
                }
                if (recoveryPc != 0u && recoveryPc != pc)
                {
                    if (s_recoverCount < 128u)
                    {
                        std::cerr << "[dispatch:recover-low-pc] bad=0x" << std::hex << pc
                                  << " ra=0x" << ra
                                  << " fallback=0x" << recoveryPc
                                  << " sp=0x" << sp
                                  << std::dec << std::endl;
                    }
                    ++s_recoverCount;
                    ctx->pc = recoveryPc;
                    return;
                }
            }
        }

        std::ostringstream oss;
        oss << "Error: Called unimplemented function at address 0x" << std::hex << (ctx ? ctx->pc : 0u)
            << " ra=0x" << ra
            << " sp=0x" << sp
            << " gp=0x" << gp
            << " a0=0x" << a0
            << " hostTid=" << std::this_thread::get_id()
            << " pcTrace=" << formatDispatchHistory()
            << std::dec;

        static std::mutex s_defaultFnLogMutex;
        {
            std::lock_guard<std::mutex> lock(s_defaultFnLogMutex);
            std::cerr << oss.str() << std::endl;
        }

        runtime->requestStop();
    };

    return defaultFunction;
}

void PS2Runtime::SignalException(R5900Context *ctx, PS2Exception exception)
{
    if (exception == EXCEPTION_INTEGER_OVERFLOW)
    {
        HandleIntegerOverflow(ctx);
        return;
    }

    raiseCop0Exception(ctx, static_cast<uint32_t>(exception),
                       exception == EXCEPTION_TLB_REFILL);
}

void PS2Runtime::executeVU0Microprogram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    (void)rdram;

    // Real VU0 macro-mode execution, hand-ported from ran-j/PS2Recomp
    // origin/main (merged PR #48, "Issue 11 vu0 macro"). This function used to
    // be a no-op that seeded a few status flags and returned without writing a
    // single vf register, so DQ8 -- which builds its entire per-object
    // rotation matrix inside a VU0 microprogram -- read whatever stale
    // registers happened to be left over from the previous caller.
    //
    // Upstream holds the interpreter as a PS2Runtime data member (m_vu0). It
    // is a function-local static here instead: inserting a data member into
    // PS2Runtime would shift the layout the prebuilt corpus .so was compiled
    // against. VU0 state that must survive between calls lives in ctx, and the
    // micro/data memory lives in the VIF0 TU, so the interpreter object itself
    // is pure scratch and is safe to share.
    //
    // The address the recompiled VCALLMS passes is already a byte address
    // (imm15[8:0] << 3), which is why it is only masked to an instruction
    // boundary rather than scaled again.

    uint8_t *const vu0Code = ps2xVu0Code();
    uint8_t *const vu0Data = ps2xVu0Data();
    const uint32_t startPC = address & ~0x7u;

    if (!vu0Code || !vu0Data || startPC + 8u > PS2_VU0_CODE_SIZE)
    {
        seedVu0IdleSuccess(ctx);
        return;
    }

    // If no MPG has ever landed, micro memory is all zeroes and "executing" it
    // would just decode zero instructions. Say so once, loudly, because that
    // is the exact failure this port exists to remove and it must not be
    // mistaken for the microprogram running correctly.
    if (!ps2xVu0MicroLoaded())
    {
        static std::mutex s_vu0EmptyMutex;
        static bool s_vu0EmptyWarned = false;
        {
            std::lock_guard<std::mutex> lock(s_vu0EmptyMutex);
            if (!s_vu0EmptyWarned)
            {
                s_vu0EmptyWarned = true;
                std::cerr << "[vu0:empty] VCALLMS to micro @0x" << std::hex << startPC << std::dec
                          << " but VU0 micro memory has never received an MPG upload"
                          << " -- the VIF0 path is not delivering it" << std::endl;
            }
        }
        seedVu0IdleSuccess(ctx);
        return;
    }

    vu0CensusNoteCall();

    static std::mutex s_vu0RunMutex;
    static VU1Interpreter s_vu0;
    std::lock_guard<std::mutex> lock(s_vu0RunMutex);

    // The VU pipeline-timing model (s_vuCycle, the FMAC/STATUS/CLIP flag
    // pipe, Q/P EFU latency) lives in file-scope globals in ps2_vu1.cpp
    // that are shared by every VU1Interpreter instance -- including
    // PS2Runtime::m_vu1, the real VU1 geometry processor, whose state is
    // deliberately never reset across its many MSCAL/MSCNT kicks per
    // frame. s_vu0.reset() below unconditionally zeroes those globals
    // (vuPipeReset), which would silently drop whatever VU1 had pending
    // in flight at the moment this VCALLMS fires, and would then leave
    // VU0's OWN leftover pipe state for VU1's next kick to mis-adopt.
    // Save the shared globals now and restore them after this call so
    // VU0's use of them is invisible to VU1.
    VU1SharedPipelineSnapshot vu1PipelineSnapshot = s_vu0.saveSharedPipeline();

    s_vu0.reset();
    copyVu0ContextToState(ctx, s_vu0.state());
    // reset()'s vuPipeReset ran BEFORE copyVu0ContextToState populated
    // mac/clip/status from ctx, so it seeded the shadow flags (read by
    // FMAND/FMEQ/FMOR/...) from a still-zeroed state. Reseed now that the
    // real entry flags are in place, so shadow and architectural state
    // agree from cycle 0.
    s_vu0.resyncPipelineShadow();

    if (vu0TraceOn())
    {
        vu0TraceEntry(startPC, s_vu0.state());
    }

    s_vu0.execute(vu0Code, PS2_VU0_CODE_SIZE,
                  vu0Data, PS2_VU0_DATA_SIZE,
                  m_gs, &m_memory,
                  startPC, ctx->vu0_itop, 4096);

    if (vu0TraceOn())
    {
        vu0TraceExit(startPC, s_vu0.state());
    }

    copyVu0StateToContext(s_vu0.state(), ctx);
    s_vu0.restoreSharedPipeline(vu1PipelineSnapshot);

    // Per-micro-address run log, plus a railed detector that fires
    // independently of the run-count cap. The old "n < 3" cap made this
    // diagnostic silent about steady state -- exactly the picture the
    // world-matrix-corruption investigation needs (is the basis this
    // microprogram produces ever railed, and how often, across the whole
    // run, not just its first 3 calls). Both buckets are env-tunable
    // (0 = unlimited) and both announce truncation exactly once via the
    // shared budget machinery, so a capped run here can never be misread as
    // a run that saw nothing further -- that class of bug has already
    // produced wrong conclusions in this project.
    {
        static std::mutex s_vu0LogMutex;
        static std::unordered_map<uint32_t, int> s_vu0Ran;
        static std::atomic<uint32_t> s_vu0RailedCount{0};
        static std::atomic<bool> s_vu0BaselineTruncated{false};
        static std::atomic<bool> s_vu0RailedTruncated{false};
        static const uint32_t kBaselineMax =
            ps2DiagEnvLimit("PS2X_VU0_EXEC_BASELINE_MAX_LOGS", 20u);
        static const uint32_t kRailedMax =
            ps2DiagEnvLimit("PS2X_VU0_EXEC_RAILED_MAX_LOGS", 500u);

        float row4[4], row5[4], row6[4], row7[4];
        _mm_storeu_ps(row4, ctx->vu0_vf[4]);
        _mm_storeu_ps(row5, ctx->vu0_vf[5]);
        _mm_storeu_ps(row6, ctx->vu0_vf[6]);
        _mm_storeu_ps(row7, ctx->vu0_vf[7]);
        auto maxAbs4 = [](const float *f) -> float
        {
            float m = std::fabs(f[0]);
            for (int i = 1; i < 4; ++i)
                m = std::max(m, std::fabs(f[i]));
            return m;
        };
        const float maxBasis = std::max(std::max(maxAbs4(row4), maxAbs4(row5)),
                                         std::max(maxAbs4(row6), maxAbs4(row7)));
        const bool railed = maxBasis >= 1e18f;

        std::lock_guard<std::mutex> logLock(s_vu0LogMutex);
        int &n = s_vu0Ran[startPC];

        const bool logBaseline = ps2DiagLogBudget(std::cerr, "[vu0:exec:baseline]",
                                                    "PS2X_VU0_EXEC_BASELINE_MAX_LOGS",
                                                    kBaselineMax, static_cast<uint32_t>(n),
                                                    s_vu0BaselineTruncated);
        bool logRailed = false;
        if (railed)
        {
            const uint32_t railedIdx = s_vu0RailedCount.fetch_add(1, std::memory_order_relaxed);
            logRailed = ps2DiagLogBudget(std::cerr, "[vu0:exec:railed]",
                                          "PS2X_VU0_EXEC_RAILED_MAX_LOGS",
                                          kRailedMax, railedIdx, s_vu0RailedTruncated);
        }

        if (logBaseline || logRailed)
        {
            std::cerr << "[vu0:exec] micro@0x" << std::hex << startPC << std::dec
                      << " run#" << n
                      << (railed ? " RAILED" : " ok")
                      << " maxBasis=" << maxBasis
                      << " lastMpg=" << ps2xVu0LastMpgInstrCount() << "instr"
                      << " uploads=" << ps2xVu0MpgUploadCount()
                      << " vf4=" << row4[0] << ',' << row4[1] << ',' << row4[2] << ',' << row4[3]
                      << " vf5=" << row5[0] << ',' << row5[1] << ',' << row5[2] << ',' << row5[3]
                      << " vf6=" << row6[0] << ',' << row6[1] << ',' << row6[2] << ',' << row6[3]
                      << " vf7=" << row7[0] << ',' << row7[1] << ',' << row7[2] << ',' << row7[3]
                      << std::endl;
        }
        ++n;
    }
}

void PS2Runtime::vu0StartMicroProgram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    // VCALLMS and VCALLMSR both route here.
    executeVU0Microprogram(rdram, ctx, address);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx)
{
    handleSyscall(rdram, ctx, 0);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx, uint32_t encodedSyscallId)
{
    if (ctx->in_delay_slot)
    {
        throw std::runtime_error("Attempted to execute a syscall inside a branch delay slot! "
                                 "This breaks the atomic basic block model and is structurally unsupported by the emulator.");
    }

    const uint32_t syscallId = (encodedSyscallId != 0u)
                                   ? encodedSyscallId
                                   : getRegU32(ctx, 3); // $v1 / $3 is the EE kernel syscall number

    if (ps2_syscalls::dispatchNumericSyscall(syscallId, rdram, ctx, this))
    {
        return;
    }

    // God help you
    ps2_syscalls::TODO(rdram, ctx, this, encodedSyscallId);
}

void PS2Runtime::handleBreak(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_BREAKPOINT);
}

void PS2Runtime::handleTrap(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_TRAP);
}

void PS2Runtime::handleTLBR(uint8_t *rdram, R5900Context *ctx)
{
    uint32_t vpn = 0;
    uint32_t pfn = 0;
    uint32_t mask = 0;
    bool valid = false;

    const uint32_t index = ctx->cop0_index & 0x3Fu;
    if (!m_memory.tlbRead(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Preserve low ASID bits in EntryHi.
    ctx->cop0_entryhi = (ctx->cop0_entryhi & 0x00000FFFu) | (vpn & 0xFFFFF000u);
    ctx->cop0_entrylo0 = (ctx->cop0_entrylo0 & ~0x03FFFFC2u) |
                         ((pfn & 0x000FFFFFu) << 6) |
                         (valid ? 0x2u : 0u);
    ctx->cop0_pagemask = mask & 0x01FFE000u;
}

void PS2Runtime::handleTLBWI(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t index = ctx->cop0_index & 0x3Fu;
    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
    }
}

void PS2Runtime::handleTLBWR(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t entryCount = static_cast<uint32_t>(m_memory.tlbEntryCount());
    if (entryCount == 0)
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    const uint32_t wired = std::min(ctx->cop0_wired, entryCount - 1);
    uint32_t random = ctx->cop0_random % entryCount;
    if (random < wired)
    {
        random = wired;
    }

    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(random, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Keep COP0 bookkeeping in sync with the selected slot.
    ctx->cop0_index = (ctx->cop0_index & ~0x3Fu) | (random & 0x3Fu);
    ctx->cop0_random = (random <= wired) ? (entryCount - 1) : (random - 1);
}

void PS2Runtime::handleTLBP(uint8_t *rdram, R5900Context *ctx)
{
    const int32_t index = m_memory.tlbProbe(ctx->cop0_entryhi & 0xFFFFF000u);
    if (index >= 0)
    {
        ctx->cop0_index = (ctx->cop0_index & ~0x8000003Fu) |
                          (static_cast<uint32_t>(index) & 0x3Fu);
    }
    else
    {
        // MIPS sets probe failure bit (P) in Index[31].
        ctx->cop0_index |= 0x80000000u;
    }
}

void PS2Runtime::clearLLBit(R5900Context *ctx)
{
    // LL/SC reservation is tracked separately from COP0 Status.
    ctx->llbit = 0;
    ctx->lladdr = 0;
}

uint32_t PS2Runtime::alignGuestHeapValue(uint32_t value, uint32_t alignment)
{
    if (alignment == 0)
    {
        return value;
    }

    const uint32_t mask = alignment - 1u;
    if (value > (std::numeric_limits<uint32_t>::max() - mask))
    {
        return std::numeric_limits<uint32_t>::max();
    }
    return (value + mask) & ~mask;
}

bool PS2Runtime::isGuestHeapAlignmentValid(uint32_t alignment)
{
    return alignment != 0u && (alignment & (alignment - 1u)) == 0u;
}

uint32_t PS2Runtime::normalizeGuestHeapAlignment(uint32_t alignment)
{
    if (!isGuestHeapAlignmentValid(alignment))
    {
        return kGuestHeapDefaultAlignment;
    }
    return std::max(alignment, kGuestHeapDefaultAlignment);
}

uint32_t PS2Runtime::clampGuestHeapBase(uint32_t guestBase) const
{
    uint32_t normalized = guestBase;
    if (normalized >= PS2_RAM_SIZE)
    {
        normalized &= PS2_RAM_MASK;
    }
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    return std::min(normalized, hardLimit);
}

uint32_t PS2Runtime::clampGuestHeapLimit(uint32_t guestLimit) const
{
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    if (guestLimit == 0u || guestLimit > hardLimit)
    {
        return hardLimit;
    }
    return guestLimit;
}

void PS2Runtime::resetGuestHeapLocked(uint32_t guestBase, uint32_t guestLimit)
{
    uint32_t base = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    uint32_t limit = clampGuestHeapLimit(guestLimit);
    if (base == 0u)
    {
        const uint32_t fallbackBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
        base = alignGuestHeapValue(clampGuestHeapBase(fallbackBase), kGuestHeapDefaultAlignment);
    }

    if (limit <= base)
    {
        base = alignGuestHeapValue(clampGuestHeapBase(m_guestHeapSuggestedBase), kGuestHeapDefaultAlignment);
        limit = clampGuestHeapLimit(0u);
    }

    if (limit <= base)
    {
        base = 0u;
        limit = 0u;
    }

    m_guestHeapBlocks.clear();
    if (limit > base)
    {
        m_guestHeapBlocks.push_back({base, limit - base, true});
    }

    m_guestHeapBase = base;
    m_guestHeapEnd = base;
    m_guestHeapLimit = limit;
    m_guestHeapConfigured = true;
}

void PS2Runtime::ensureGuestHeapInitializedLocked()
{
    if (m_guestHeapConfigured)
    {
        return;
    }

    const uint32_t suggested = (m_guestHeapSuggestedBase == 0u) ? kGuestHeapDefaultBase : m_guestHeapSuggestedBase;
    resetGuestHeapLocked(suggested, clampGuestHeapLimit(0u));
}

int32_t PS2Runtime::findGuestHeapBlockIndexLocked(uint32_t guestAddr) const
{
    const uint32_t normalizedAddr = guestAddr & PS2_RAM_MASK;
    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock &block = m_guestHeapBlocks[i];
        if (!block.free && block.addr == normalizedAddr)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

uint32_t PS2Runtime::allocateGuestBlockLocked(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    if (size > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock block = m_guestHeapBlocks[i];
        if (!block.free)
        {
            continue;
        }

        const uint64_t blockStart = block.addr;
        const uint64_t blockEnd = blockStart + static_cast<uint64_t>(block.size);
        const uint32_t alignedAddr = alignGuestHeapValue(block.addr, normalizedAlignment);
        if (alignedAddr < block.addr)
        {
            continue;
        }

        const uint64_t alignedStart = alignedAddr;
        if (alignedStart > blockEnd)
        {
            continue;
        }

        const uint64_t allocEnd = alignedStart + static_cast<uint64_t>(allocSize);
        if (allocEnd > blockEnd)
        {
            continue;
        }

        const uint32_t prefixSize = static_cast<uint32_t>(alignedStart - blockStart);
        const uint32_t suffixSize = static_cast<uint32_t>(blockEnd - allocEnd);

        std::vector<GuestHeapBlock> replacement;
        replacement.reserve(3);
        if (prefixSize > 0u)
        {
            replacement.push_back({block.addr, prefixSize, true});
        }
        replacement.push_back({alignedAddr, allocSize, false});
        if (suffixSize > 0u)
        {
            replacement.push_back({static_cast<uint32_t>(allocEnd), suffixSize, true});
        }

        m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
        m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i),
                                 replacement.begin(),
                                 replacement.end());

        m_guestHeapEnd = std::max(m_guestHeapEnd, static_cast<uint32_t>(allocEnd));
        return alignedAddr;
    }

    return 0u;
}

void PS2Runtime::coalesceGuestHeapLocked()
{
    if (m_guestHeapBlocks.empty())
    {
        return;
    }

    size_t i = 1;
    while (i < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &prev = m_guestHeapBlocks[i - 1];
        GuestHeapBlock &curr = m_guestHeapBlocks[i];
        const uint64_t prevEnd = static_cast<uint64_t>(prev.addr) + static_cast<uint64_t>(prev.size);
        if (prev.free && curr.free && prevEnd == curr.addr)
        {
            prev.size += curr.size;
            m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        ++i;
    }
}

void PS2Runtime::freeGuestBlockLocked(uint32_t guestAddr)
{
    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return;
    }

    m_guestHeapBlocks[static_cast<size_t>(index)].free = true;
    coalesceGuestHeapLocked();
}

void PS2Runtime::configureGuestHeap(uint32_t guestBase, uint32_t guestLimit)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    uint32_t normalizedBase = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    if (normalizedBase == 0u)
    {
        normalizedBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
    }
    m_guestHeapSuggestedBase = normalizedBase;
    resetGuestHeapLocked(normalizedBase, guestLimit);
}

uint32_t PS2Runtime::guestMalloc(uint32_t size, uint32_t alignment)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    return allocateGuestBlockLocked(size, alignment);
}

uint32_t PS2Runtime::guestCalloc(uint32_t count, uint32_t size, uint32_t alignment)
{
    if (count == 0u || size == 0u)
    {
        return 0u;
    }
    if (count > (std::numeric_limits<uint32_t>::max() / size))
    {
        return 0u;
    }

    const uint32_t totalSize = count * size;
    const uint32_t guestAddr = guestMalloc(totalSize, alignment);
    if (guestAddr != 0u)
    {
        uint8_t *rdram = m_memory.getRDRAM();
        if (rdram)
        {
            uint32_t physAddr = guestAddr & PS2_RAM_MASK;
            if (physAddr + totalSize <= PS2_RAM_SIZE)
                std::memset(rdram + physAddr, 0, totalSize);
        }
    }

    return guestAddr;
}

uint32_t PS2Runtime::guestRealloc(uint32_t guestAddr, uint32_t newSize, uint32_t alignment)
{
    if (guestAddr == 0u)
    {
        return guestMalloc(newSize, alignment);
    }
    if (newSize == 0u)
    {
        guestFree(guestAddr);
        return 0u;
    }

    if (newSize > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t requestedSize = alignGuestHeapValue(newSize, kGuestHeapDefaultAlignment);

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();

    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return 0u;
    }

    const size_t blockIndex = static_cast<size_t>(index);
    const uint32_t oldAddr = m_guestHeapBlocks[blockIndex].addr;
    const uint32_t oldSize = m_guestHeapBlocks[blockIndex].size;

    if (requestedSize <= oldSize)
    {
        if (requestedSize < oldSize)
        {
            const uint32_t tailAddr = oldAddr + requestedSize;
            const uint32_t tailSize = oldSize - requestedSize;
            m_guestHeapBlocks[blockIndex].size = requestedSize;
            m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u),
                                     GuestHeapBlock{tailAddr, tailSize, true});
            coalesceGuestHeapLocked();
        }
        return oldAddr;
    }

    if (blockIndex + 1u < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &next = m_guestHeapBlocks[blockIndex + 1u];
        const uint64_t blockEnd = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].addr) +
                                  static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size);
        if (next.free && blockEnd == next.addr)
        {
            const uint64_t combined = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size) +
                                      static_cast<uint64_t>(next.size);
            if (combined >= requestedSize)
            {
                const uint32_t extraNeeded = requestedSize - m_guestHeapBlocks[blockIndex].size;
                m_guestHeapBlocks[blockIndex].size = requestedSize;
                if (next.size == extraNeeded)
                {
                    m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u));
                }
                else
                {
                    next.addr += extraNeeded;
                    next.size -= extraNeeded;
                }
                m_guestHeapEnd = std::max(m_guestHeapEnd, oldAddr + requestedSize);
                return oldAddr;
            }
        }
    }

    const uint32_t newAddr = allocateGuestBlockLocked(newSize, normalizedAlignment);
    if (newAddr == 0u)
    {
        return 0u;
    }

    uint8_t *rdram = m_memory.getRDRAM();
    if (rdram)
    {
        const uint32_t copyBytes = std::min(oldSize, newSize);
        uint32_t dstPhys = newAddr & PS2_RAM_MASK;
        uint32_t srcPhys = oldAddr & PS2_RAM_MASK;
        if (dstPhys + copyBytes <= PS2_RAM_SIZE && srcPhys + copyBytes <= PS2_RAM_SIZE)
            std::memmove(rdram + dstPhys, rdram + srcPhys, copyBytes);
    }

    freeGuestBlockLocked(oldAddr);
    return newAddr;
}

void PS2Runtime::guestFree(uint32_t guestAddr)
{
    if (guestAddr == 0u)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    freeGuestBlockLocked(guestAddr);
}

uint32_t PS2Runtime::guestHeapBase() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapBase : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapEnd() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapEnd : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapLimit() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapLimit : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::reserveAsyncCallbackStack(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
    uint32_t top = m_asyncCallbackStackTop;
    if (top > PS2_RAM_SIZE)
    {
        top = PS2_RAM_SIZE;
    }
    top &= ~(kGuestHeapDefaultAlignment - 1u);

    if (top <= allocSize)
    {
        return 0u;
    }

    uint32_t base = top - allocSize;
    base &= ~(normalizedAlignment - 1u);
    if (base < m_asyncCallbackStackFloor || base >= top)
    {
        return 0u;
    }

    m_asyncCallbackStackTop = base;
    // One line per reservation (a handful per boot): permanent evidence of
    // where host-dispatched callback stacks live, so any future overlap with
    // guest memory is visible in the boot log.
    std::cerr << "[async-stack] reserved [0x" << std::hex << base
              << ", 0x" << top << ") stackTop=0x" << (top - 0x10u)
              << std::dec << '\n';
    return top - 0x10u;
}

void PS2Runtime::dispatchLoop(uint8_t *rdram, R5900Context *ctx)
{
    uint32_t lastPc = std::numeric_limits<uint32_t>::max();
    uint32_t samePcCount = 0;
    constexpr uint32_t kSamePcYieldInterval = 0x4000u;

    while (!isStopRequested())
    {
        // Cooperative scheduling point: a guest loop that spins ACROSS function
        // dispatches (call/return chains, recover-pc storms) has its back-edge
        // HERE, not inside any recompiled function, so without this check such
        // a loop never reaches yield_point() and holds the guest token forever,
        // starving host workers (interrupt worker VBlank/INTC delivery) that
        // block in async_guest_begin(). Fast path is a counter test, so this is
        // as cheap as the checks the recompiler emits at intra-function
        // back-edges. Return value is irrelevant: whether or not we yielded,
        // ctx->pc is a clean function-boundary resume point.
        (void)shouldPreemptGuestExecution();

        const uint32_t pc = ctx->pc;

        if (pc == lastPc)
        {
            ++samePcCount;
            if ((samePcCount % kSamePcYieldInterval) == 0u)
            {
                PS2_IF_AGRESSIVE_LOGS({
                    RUNTIME_LOG("CPU is doing some work at PC 0x" << std::hex << pc << ". PC not updating.");
                });
                std::this_thread::yield();
            }
        }
        else
        {
            samePcCount = 0;
            lastPc = pc;
        }

        RecompiledFunction fn = lookupFunction(pc);
        const uint32_t dispatchedPc = pc;
        const uint32_t dispatchedRa = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0));

        // [DQ8 TOPO_W field-stall RCA, 2026-07-20] DIAGNOSTIC-ONLY, default-off,
        // read-only. Env-gated (DQ8_PC_SAMPLE) field-thread PC sampler: while
        // masterMode [0x3D2740]==0 (field), periodically print the dispatched
        // guest PC + this fiber's recent dispatch history. A guest thread stuck
        // in a cooperative spin-yield shows an identical, repeating history
        // across samples, pinning the exact spin site. Capped so the log stays
        // bounded. No effect on any code path when the env var is unset.
        {
            static const bool s_dq8PcSample = (std::getenv("DQ8_PC_SAMPLE") != nullptr);
            if (s_dq8PcSample)
            {
                const int16_t mm = *reinterpret_cast<const int16_t *>(
                    rdram + (0x3d2740u & PS2_RAM_MASK));
                if (mm == 0)
                {
                    static thread_local uint64_t s_ctr = 0;
                    static thread_local uint32_t s_printed = 0;
                    static const uint32_t kMaxPcSampleLogs =
                        ps2DiagEnvLimit("DQ8_PC_SAMPLE_MAX_LOGS", 1200u);
                    static std::atomic<bool> s_pcSampleTruncated{false};
                    if (((++s_ctr) % 0x20ull) == 0ull &&
                        ps2DiagLogBudget(std::cerr,
                                         "[dq8:pc-sample]",
                                         "DQ8_PC_SAMPLE_MAX_LOGS",
                                         kMaxPcSampleLogs,
                                         s_printed,
                                         s_pcSampleTruncated))
                    {
                        ++s_printed;
                        std::cerr << "[dq8:pc-sample] n=" << std::dec << s_ctr
                                  << " pc=0x" << std::hex << dispatchedPc
                                  << " ra=0x" << dispatchedRa
                                  << " hist=" << formatDispatchHistory()
                                  << std::dec << std::endl;
                    }
                }
            }
        }

        fn(rdram, ctx, this);

        if (ctx->pc == 0u)
        {
            const uint32_t ra = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0));
            const uint32_t sp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0));
            const uint32_t gp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0));
            std::cerr << "[dispatch:pc-zero] from=0x" << std::hex << dispatchedPc
                      << " fromRa=0x" << dispatchedRa
                      << " ra=0x" << ra
                      << " sp=0x" << sp
                      << " gp=0x" << gp
                      << " trace=" << formatDispatchHistory()
                      << std::dec << std::endl;

            // PC=0 means this guest thread returned (usually via jr $ra with RA=0).
            // Do not request a global runtime stop here: other guest threads may still run.
            break;
        }
    }
}

bool PS2Runtime::shouldPreemptGuestExecution()
{
    return ps2sched::yield_point();
}

uint8_t PS2Runtime::Load8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read8(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint16_t PS2Runtime::Load16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read16(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint32_t PS2Runtime::Load32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read32(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint64_t PS2Runtime::Load64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read64(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

__m128i PS2Runtime::Load128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read128(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return _mm_setzero_si128();
    }
}

void PS2Runtime::Store8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint8_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 1u, value, 0u, "WRITE8", ctx);
    try
    {
        m_memory.write8(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint16_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 2u, value, 0u, "WRITE16", ctx);
    try
    {
        m_memory.write16(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 4u, value, 0u, "WRITE32", ctx);
    try
    {
        m_memory.write32(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 8u, value, 0u, "WRITE64", ctx);
    try
    {
        m_memory.write64(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, __m128i value)
{
    alignas(16) uint64_t _parts[2];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(_parts), value);
    ps2TraceGuestWrite(rdram, vaddr, 16u, _parts[0], _parts[1], "WRITE128", ctx);
    try
    {
        m_memory.write128(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::requestStop()
{
    m_stopRequested.store(true, std::memory_order_relaxed);
    ps2_syscalls::notifyRuntimeStop();
}

void PS2Runtime::requestStopFlagOnly()
{
    m_stopRequested.store(true, std::memory_order_relaxed);
}

bool PS2Runtime::isStopRequested() const
{
    return m_stopRequested.load(std::memory_order_relaxed);
}

void PS2Runtime::HandleIntegerOverflow(R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_INTEGER_OVERFLOW);
}

// Trampoline pointer used by the scheduler stop callback (non-capturing lambda).
// Set in run() before scheduler_init(); cleared after scheduler_shutdown() returns.
static PS2Runtime* g_stopRuntime = nullptr;

void PS2Runtime::run()
{
    m_stopRequested.store(false, std::memory_order_relaxed);
    ps2_stubs::resetSifState();
    ps2_syscalls::resetSoundDriverRpcState();
    ps2_stubs::resetAudioStubState();
    ps2_stubs::resetGsSyncVCallbackState();
    ps2_stubs::resetMpegStubState();
    ps2_syscalls::initializeGuestKernelState(m_memory.getRDRAM());
    m_cpuContext.r[4] = _mm_setzero_si128();
    m_cpuContext.r[5] = _mm_setzero_si128();
    // Bootstrap $sp at top of RAM, as the hardware loader does; the guest's
    // crt0 immediately replaces it via SetupThread (DQ8: stack=0x01F40000,
    // size=0xC0000 => $sp=0x02000000). Callback stacks no longer live up
    // here (see kAsyncCallbackStackFloor), so this cannot collide with them.
    m_cpuContext.r[29] = _mm_set_epi64x(0, static_cast<int64_t>(PS2_RAM_SIZE - 0x10u));

    RUNTIME_LOG("Starting execution at address 0x" << std::hex << m_cpuContext.pc << std::dec);

    // Record what the forensic write watches are actually measuring. Emitted
    // here, after the game runner has had its chance to arm them, so a run's
    // log never leaves "no events" ambiguous between "disarmed", "budget
    // exhausted" and "genuinely never happened".
    ps2DiagAnnounceWatchState();

    // A blank image to use as a framebuffer
    Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, BLANK);
    Texture2D frameTex = LoadTextureFromImage(blank);
    UnloadImage(blank);

    // Initialize the fiber/pool scheduler.
    ps2sched::scheduler_init();
    g_stopRuntime = this;
    ps2sched::scheduler_set_stop_callback(+[]{ if (g_stopRuntime) g_stopRuntime->requestStopFlagOnly(); });

    // Create the main guest fiber (tid=1).
    uint8_t *rdram = m_memory.getRDRAM();
    {
        const uint32_t entry = m_cpuContext.pc;
        const uint32_t sp    = static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[29], 0));
        const uint32_t gp    = static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[28], 0));
        ps2sched::create_fiber(1, 1, entry, sp, gp, 0u, this, rdram);
    }

    ps2_syscalls::EnsureVSyncWorkerRunning(m_memory.getRDRAM(), this);

    // One-shot parked-thread diagnostics at ~30s and ~90s into the boot.
    // Permanent, low-noise (each fires exactly once per run): answers "where
    // is every guest thread parked" (pc/ra/state + which sema it's waiting
    // on, if any) without needing per-callsite instrumentation, and the 90s
    // shot shows whether anything moved since 30s. See PS2_PROJECT_STATE.md
    // §4 (2026-07-05) and §3.20 (2026-07-06, movie-EOF walk).
    const auto runStart = std::chrono::steady_clock::now();
    bool fiberDumpFired30 = false;
    bool fiberDumpFired90 = false;

    while (!isStopRequested())
    {
        if (!fiberDumpFired30 &&
            std::chrono::steady_clock::now() - runStart >= std::chrono::seconds(30))
        {
            fiberDumpFired30 = true;
            ps2sched::dump_all_fibers("30s-one-shot");
            ps2_syscalls::dumpSemaWaitLists("30s-one-shot");
        }
        if (!fiberDumpFired90 &&
            std::chrono::steady_clock::now() - runStart >= std::chrono::seconds(90))
        {
            fiberDumpFired90 = true;
            ps2sched::dump_all_fibers("90s-one-shot");
            ps2_syscalls::dumpSemaWaitLists("90s-one-shot");
        }

        // Guest-memory transition watches (runtime/ps2_guestwatch.h): a few
        // word reads per frame, prints only on value change. No-op when the
        // runner registered no watches.
        ps2_watch::poll(rdram);

        // Permanent low-noise GS-activity summary: one line when any of the
        // dma/gif/gs/vif counters FIRST becomes nonzero, then one line every
        // 600 frames (~10 s) while any counter is still changing. This is the
        // M0 "did the game start drawing" headline signal, previously only
        // visible under AGRESSIVE_LOGS builds.
        {
            static uint64_t s_frame = 0;
            static uint64_t lastDma = 0, lastGif = 0, lastGs = 0, lastVif = 0;
            static bool everActive = false;
            ++s_frame;
            const uint64_t curDma = m_memory.dmaStartCount();
            const uint64_t curGif = m_memory.gifCopyCount();
            const uint64_t curGs = m_memory.gsWriteCount();
            const uint64_t curVif = m_memory.vifWriteCount();
            const bool changed = (curDma != lastDma) || (curGif != lastGif) ||
                                 (curGs != lastGs) || (curVif != lastVif);
            const bool firstActivity =
                !everActive && (curDma | curGif | curGs | curVif) != 0u;
            if (firstActivity || (changed && (s_frame % 600u) == 0u))
            {
                everActive = true;
                const GSRegisters &gsr = m_memory.gs();
                std::cout << "[gs-activity] frame=" << s_frame
                          << " dma=" << curDma
                          << " gif=" << curGif
                          << " gsw=" << curGs
                          << " vif=" << curVif
                          << " prims=" << m_gs.drawStatPrims()
                          << " imgB=" << m_gs.drawStatImageBytes()
                          << std::hex
                          << " drawFbp=0x" << m_gs.drawStatLastFbp()
                          << " dispfb1=0x" << gsr.dispfb1
                          << " dispfb2=0x" << gsr.dispfb2
                          << " display1=0x" << gsr.display1
                          << std::dec
                          << (firstActivity ? " (FIRST ACTIVITY)" : "")
                          << std::endl;
                lastDma = curDma;
                lastGif = curGif;
                lastGs = curGs;
                lastVif = curVif;
            }
        }

        PS2_IF_AGRESSIVE_LOGS({
            static uint64_t tick = 0;
            tick++;
            if ((tick % 120) == 0)
            {
                uint64_t curDma = m_memory.dmaStartCount();
                uint64_t curGif = m_memory.gifCopyCount();
                uint64_t curGs = m_memory.gsWriteCount();
                uint64_t curVif = m_memory.vifWriteCount();
                const GSRegisters &gs = m_memory.gs();
                const int activeThreads = g_activeThreads.load(std::memory_order_relaxed);

                std::cout << "[run:tick] tick=" << tick
                          << " dispfb1=0x" << std::hex << gs.dispfb1
                          << " display1=0x" << gs.display1
                          << std::dec
                          << " activeThreads=" << activeThreads
                          << " dma=" << curDma
                          << " gif=" << curGif
                          << " gsw=" << curGs
                          << " vif=" << curVif
                          << std::endl;
            }
        });
        uint32_t presentWidth = FB_WIDTH;
        uint32_t presentHeight = DEFAULT_DISPLAY_HEIGHT;
        UploadFrame(frameTex, this, presentWidth, presentHeight);

        // Read-only, default-off probe for present-width band diagnosis.
        {
            static int s_probeWidth = -1;
            if (s_probeWidth < 0)
            {
                const char *e = std::getenv("DQ8_PROBE_PRESENT_W");
                s_probeWidth = (e && std::string(e) == "1") ? 1 : 0;
            }
            if (s_probeWidth > 0)
            {
                static uint32_t s_lastPW = 0u, s_lastPH = 0u;
                if (presentWidth != s_lastPW || presentHeight != s_lastPH)
                {
                    s_lastPW = presentWidth;
                    s_lastPH = presentHeight;
                    std::cerr << "[present:probe] presentWidth=" << presentWidth
                              << " presentHeight=" << presentHeight << std::endl;
                }
            }
        }

        BeginDrawing();
        ClearBackground(BLACK);
        const float srcWidth = static_cast<float>(std::max<uint32_t>(1u, presentWidth));
        const float srcHeight = static_cast<float>(std::max<uint32_t>(1u, presentHeight));
        const float screenWidth = static_cast<float>(GetScreenWidth());
        const float screenHeight = static_cast<float>(GetScreenHeight());
        // Present the guest's displayed frame to fill the window. The guest
        // frame always represents a full-screen 4:3 image that the PS2 scans
        // out across the whole display, regardless of its native pixel width
        // (e.g. DQ8's 512-wide display buffer, FRAME/DISPFB fbw=8, vs a
        // 640-wide one). We honor that by stretching the actual displayed
        // width (presentWidth, decoded from the guest DISPLAY/DISPFB registers
        // in UploadFrame) to fill the window -- exactly as 640-wide frames
        // already do -- instead of preserving the source's 1:1 pixel aspect,
        // which pillarboxed narrower buffers like 512 (the "black band").
        const Rectangle srcRect{0.0f, 0.0f, srcWidth, srcHeight};
        const Rectangle dstRect{0.0f, 0.0f, screenWidth, screenHeight};
        DrawTexturePro(frameTex, srcRect, dstRect, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
        EndDrawing();

        // DQ8_REC=1: capture distinct frames as PNGs under DQ8_REC_DIR/frames/
        {
            static int s_recEnabled = -1;
            static std::string s_recDir;
            static uint64_t s_frameTick = 0;
            static uint64_t s_lastHash = 0;
            static int s_recInterval = 30;

            if (s_recEnabled < 0) {
                const char* env = std::getenv("DQ8_REC");
                s_recEnabled = (env && std::string(env) == "1") ? 1 : 0;
                if (s_recEnabled) {
                    const char* dir = std::getenv("DQ8_REC_DIR");
                    s_recDir = dir ? dir : "/tmp/dq8diag/run";
                    const char* ivl = std::getenv("DQ8_REC_INTERVAL");
                    s_recInterval = ivl ? std::stoi(ivl) : 30;
                    std::filesystem::create_directories(s_recDir + "/frames");
                }
            }

            if (s_recEnabled > 0) {
                ++s_frameTick;
                if ((s_frameTick % static_cast<uint64_t>(s_recInterval)) == 0) {
                    Image img = LoadImageFromTexture(frameTex);
                    // Crop to the guest's displayed region so the capture
                    // reflects what is actually presented. The upload texture
                    // is padded to FB_WIDTH (640); the guest content is only
                    // presentWidth wide (512 for DQ8). Without this crop the
                    // diagnostic shows FB_WIDTH-presentWidth px of black
                    // padding on the right (the reported "band").
                    if (presentWidth > 0u && presentHeight > 0u &&
                        (static_cast<int>(presentWidth) < img.width ||
                         static_cast<int>(presentHeight) < img.height)) {
                        ImageCrop(&img, Rectangle{0.0f, 0.0f,
                                                  static_cast<float>(presentWidth),
                                                  static_cast<float>(presentHeight)});
                    }
                    // FNV-1a-64 hash over sampled pixels (every 16 bytes)
                    uint64_t h = 14695981039346656037ULL;
                    const uint8_t* pix = static_cast<const uint8_t*>(img.data);
                    const int total = img.width * img.height * 4;
                    for (int i = 0; i < total; i += 16)
                        h = (h ^ pix[i]) * 1099511628211ULL;
                    if (h != s_lastHash) {
                        s_lastHash = h;
                        char path[512];
                        std::snprintf(path, sizeof(path), "%s/frames/frame_%06llu.png",
                                      s_recDir.c_str(),
                                      static_cast<unsigned long long>(s_frameTick));
                        ExportImage(img, path);
                    }
                    UnloadImage(img);
                }
            }
        }

        if (WindowShouldClose())
        {
            RUNTIME_LOG("[run] window close requested, breaking out of loop");
            requestStop();
            break;
        }
    }

    requestStop();

    // Signal all guest fibers to stop and join the pool threads.
    ps2sched::scheduler_shutdown();
    ps2sched::scheduler_set_stop_callback(nullptr);
    g_stopRuntime = nullptr;

    UnloadTexture(frameTex);
    CloseWindow();
}
