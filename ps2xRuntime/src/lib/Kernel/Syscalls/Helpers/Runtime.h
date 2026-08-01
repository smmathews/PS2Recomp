// ThreadExitException is defined in ThreadExit.h (outside any anonymous namespace).
// ps2_scheduler.cpp includes ThreadExit.h directly to avoid pulling in the
// many static helper functions in this file.
#include "ThreadExit.h"
#include <chrono>
#include <thread>

// Bounded backoff for a borrowed host worker that hit a blocking syscall.
// Translates a non-fiber BlockResult into the correct token handling, then
// sleeps with exponential backoff (1us -> 1ms cap). After kMaxSpins iterations
// it logs ONCE and keeps sleeping at the cap so a self-deadlocked interrupt
// handler cannot busy-spin the CPU or starve the guest executor. State is held
// in a per-call counter object so each blocking site ramps independently.
struct NonFiberBackoff
{
    int spins = 0;
    std::chrono::microseconds delay{1};

    // Returns true if this worker should re-check its wait condition and may
    // loop again; the syscall's Mesa loop decides whether to actually re-check.
    void step(ps2sched::BlockResult br)
    {
        // Drop/reacquire the guest token only if we actually own it.
        if (br == ps2sched::BlockResult::NonFiberOwner)
        {
            ps2sched::async_guest_end();
            std::this_thread::sleep_for(delay);
            ps2sched::async_guest_begin();
        }
        else // NonFiberNoTok (or, defensively, any non-Parked non-fiber result)
        {
            std::this_thread::sleep_for(delay);
        }

        constexpr int kMaxSpins = 50;
        if (spins == kMaxSpins)
        {
            std::fprintf(stderr,
                "[ps2sched] WARNING: borrowed host worker has blocked on a guest "
                "condition for %d retries; capping backoff at 1ms (possible "
                "interrupt-context deadlock)\n", kMaxSpins);
        }
        if (spins < kMaxSpins)
        {
            ++spins;
            delay *= 2;
            if (delay > std::chrono::microseconds(1000))
                delay = std::chrono::microseconds(1000);
        }
    }
};

// One-shot: drop/reacquire token (if owned) and yield once. SleepThread for
// a borrowed worker has no wait-list to re-check, so it does not loop.
inline void nonFiberBlockBackoff(ps2sched::BlockResult br)
{
    if (br == ps2sched::BlockResult::NonFiberOwner)
    {
        ps2sched::async_guest_end();
        std::this_thread::yield();
        ps2sched::async_guest_begin();
    }
    else
    {
        std::this_thread::yield();
    }
}

// RAII guard that ensures async_guest_begin/async_guest_end are always paired,
// even on exception paths. Use this in IRQ/DMAC/alarm workers instead of bare calls.
struct AsyncGuestScope
{
    AsyncGuestScope()  { ps2sched::async_guest_begin(); }
    ~AsyncGuestScope() { ps2sched::async_guest_end();   }
    AsyncGuestScope(const AsyncGuestScope&) = delete;
    AsyncGuestScope& operator=(const AsyncGuestScope&) = delete;
};

static void throwIfTerminated(const std::shared_ptr<ThreadInfo> &info)
{
    if (info && info->terminated.load())
    {
        throw ThreadExitException();
    }
}

static void waitWhileSuspended(const std::shared_ptr<ThreadInfo> &info, PS2Runtime * /*runtime*/ = nullptr)
{
    if (!info) return;
    while (info->suspendCount > 0 && !info->terminated.load()) {
        // arm_park() before publishing to any wait-list so a concurrent
        // resume_fiber that fires between publish and block_current sees
        // wake_pending rather than missing the wakeup.
        ps2sched::arm_park();
        ps2sched::block_current();
    }
    if (info->terminated.load()) { throw ThreadExitException(); }
    if (info->suspendCount == 0) {
        std::lock_guard<std::mutex> lock(info->m);
        info->status = THS_RUN;
    }
}

static std::shared_ptr<ThreadInfo> lookupThreadInfo(int tid)
{
    std::lock_guard<std::mutex> lock(g_thread_map_mutex);
    auto it = g_threads.find(tid);
    if (it != g_threads.end())
    {
        return it->second;
    }
    return nullptr;
}

static std::shared_ptr<ThreadInfo> ensureCurrentThreadInfo(R5900Context *ctx)
{
    const int tid = g_currentThreadId;
    std::lock_guard<std::mutex> lock(g_thread_map_mutex);
    auto it = g_threads.find(tid);
    if (it != g_threads.end())
    {
        return it->second;
    }

    auto info = std::make_shared<ThreadInfo>();
    info->started = true;
    info->status = THS_RUN;
    info->currentPriority = info->priority;
    info->suspendCount = 0;
    if (ctx)
    {
        info->entry = ctx->pc;
        info->stack = getRegU32(ctx, 29);
        info->gp = getRegU32(ctx, 28);
    }
    info->waitType = TSW_NONE;
    info->waitId = 0;

    g_threads.emplace(tid, info);
    return info;
}

static std::shared_ptr<SemaInfo> lookupSemaInfo(int sid)
{
    std::lock_guard<std::mutex> lock(g_sema_map_mutex);
    auto it = g_semas.find(sid);
    if (it != g_semas.end())
    {
        return it->second;
    }
    return nullptr;
}

static std::shared_ptr<EventFlagInfo> lookupEventFlagInfo(int eid)
{
    std::lock_guard<std::mutex> lock(g_event_flag_map_mutex);
    auto it = g_eventFlags.find(eid);
    if (it != g_eventFlags.end())
    {
        return it->second;
    }
    return nullptr;
}

static void setRegU32(R5900Context *ctx, int reg, uint32_t value)
{
    if (!ctx || reg < 0 || reg > 31)
        return;
    SET_GPR_U32(ctx, reg, value);
}

static std::chrono::microseconds alarmTicksToDuration(uint16_t ticks)
{
    constexpr uint64_t kAlarmTickUsec = 64u; // Approximate EE H-SYNC tick period.
    const uint64_t clampedTicks = (ticks == 0u) ? 1u : static_cast<uint64_t>(ticks);
    return std::chrono::microseconds(clampedTicks * kAlarmTickUsec);
}


static void rpcCopyToRdram(uint8_t *rdram, uint32_t dst, uint32_t src, size_t size)
{
    if (!rdram || size == 0)
        return;

    constexpr size_t kMaxRpcTransferBytes = 1u * 1024u * 1024u;
    const size_t clampedSize = std::min(size, kMaxRpcTransferBytes);
    if (clampedSize != size)
    {
        static uint32_t warnCount = 0;
        if (warnCount < 8)
        {
            std::cerr << "[SifCallRpc] clamping copy size from " << size
                      << " to " << clampedSize
                      << " bytes (dst=0x" << std::hex << dst
                      << " src=0x" << src << std::dec << ")" << std::endl;
            ++warnCount;
        }
    }

    for (size_t i = 0; i < clampedSize; ++i)
    {
        const uint32_t dstAddr = dst + static_cast<uint32_t>(i);
        const uint32_t srcAddr = src + static_cast<uint32_t>(i);
        uint8_t *dstPtr = getMemPtr(rdram, dstAddr);
        const uint8_t *srcPtr = getConstMemPtr(rdram, srcAddr);
        if (!dstPtr || !srcPtr)
        {
            break;
        }
        *dstPtr = *srcPtr;
    }
}

static void rpcZeroRdram(uint8_t *rdram, uint32_t dst, size_t size)
{
    if (!rdram || size == 0)
        return;

    constexpr size_t kMaxRpcTransferBytes = 1u * 1024u * 1024u;
    const size_t clampedSize = std::min(size, kMaxRpcTransferBytes);
    if (clampedSize != size)
    {
        static uint32_t warnCount = 0;
        if (warnCount < 8)
        {
            std::cerr << "[SifCallRpc] clamping zero size from " << size
                      << " to " << clampedSize
                      << " bytes (dst=0x" << std::hex << dst << std::dec << ")" << std::endl;
            ++warnCount;
        }
    }

    for (size_t i = 0; i < clampedSize; ++i)
    {
        const uint32_t dstAddr = dst + static_cast<uint32_t>(i);
        uint8_t *dstPtr = getMemPtr(rdram, dstAddr);
        if (!dstPtr)
        {
            break;
        }
        *dstPtr = 0;
    }
}

static bool readStackU32(uint8_t *rdram, uint32_t sp, uint32_t offset, uint32_t &out)
{
    uint8_t *ptr = getMemPtr(rdram, sp + offset);
    if (!ptr)
        return false;
    out = *reinterpret_cast<uint32_t *>(ptr);
    return true;
}

// RpcInvokeExitReason, rpcInvokeExitReasonName, rpcInvokeFunction: moved to
// RpcInvoke.h (hoisted so Kernel/Stubs/MPEG.cpp's sceMpeg callback
// dispatcher can share the exact same trampoline instead of duplicating it;
// scempeg-callback-design.md §B.1).
#include "RpcInvoke.h"

static uint32_t rpcAllocPacketAddr(uint8_t *rdram)
{
    if (kRpcPacketPoolCount == 0)
        return 0;

    uint32_t slot = g_rpc_packet_index++ % kRpcPacketPoolCount;
    uint32_t addr = kRpcPacketPoolBase + (slot * kRpcPacketSize);
    rpcZeroRdram(rdram, addr, kRpcPacketSize);
    return addr;
}

static uint32_t rpcAllocServerAddr(uint8_t *rdram)
{
    if (kRpcServerPoolCount == 0)
        return 0;

    uint32_t slot = g_rpc_server_index++ % kRpcServerPoolCount;
    uint32_t addr = kRpcServerPoolBase + (slot * kRpcServerStride);
    rpcZeroRdram(rdram, addr, kRpcServerStride);
    return addr;
}

struct IrqHandlerInfo
{
    int id = 0;
    uint32_t cause = 0;
    uint32_t handler = 0;
    uint32_t arg = 0;
    uint32_t gp = 0;
    uint32_t sp = 0;
    bool enabled = true;
    int order = 0;
};

static std::unordered_map<int, IrqHandlerInfo> g_intcHandlers;
static std::unordered_map<int, IrqHandlerInfo> g_dmacHandlers;
static int g_nextIntcHandlerId = 1;
static int g_nextDmacHandlerId = 1;

static int g_intc_head_order = 0;
static int g_intc_tail_order = 1000;
static int g_dmac_head_order = 0;
static int g_dmac_tail_order = 1000;

inline std::string translatePs2Path(const char *ps2Path)
{
    if (!ps2Path || !*ps2Path)
    {
        return {};
    }

    std::string pathStr(ps2Path);
    std::string lower = toLowerAscii(pathStr);

    auto resolveWithBase = [&](const std::filesystem::path &base, const std::string &suffix) -> std::string
    {
        const std::string normalizedSuffix = normalizePs2PathSuffix(suffix);
        std::filesystem::path resolved = base;
        if (!normalizedSuffix.empty())
        {
            resolved /= std::filesystem::path(normalizedSuffix);
        }
        return resolved.lexically_normal().string();
    };

    if (lower.rfind("host0:", 0) == 0 || lower.rfind("host:", 0) == 0)
    {
        const std::size_t prefixLength = (lower.rfind("host0:", 0) == 0) ? 6 : 5;
        return resolveWithBase(getConfiguredHostRoot(), pathStr.substr(prefixLength));
    }

    if (lower.rfind("cdrom0:", 0) == 0 || lower.rfind("cdrom:", 0) == 0)
    {
        const std::size_t prefixLength = (lower.rfind("cdrom0:", 0) == 0) ? 7 : 6;
        return resolveWithBase(getConfiguredCdRoot(), pathStr.substr(prefixLength));
    }

    if (lower.rfind(kMc0Prefix, 0) == 0)
    {
        const std::size_t prefixLength = sizeof(kMc0Prefix) - 1;
        return resolveWithBase(getConfiguredMcRoot(), pathStr.substr(prefixLength));
    }

    if (!pathStr.empty() && (pathStr.front() == '/' || pathStr.front() == '\\'))
    {
        return resolveWithBase(getConfiguredCdRoot(), pathStr);
    }

    if (pathStr.size() > 1 && pathStr[1] == ':')
    {
        return pathStr;
    }

    return resolveWithBase(getConfiguredCdRoot(), pathStr);
}

static bool localtimeSafe(const std::time_t *t, std::tm *out)
{
#ifdef _WIN32
    return localtime_s(out, t) == 0;
#else
    return localtime_r(t, out) != nullptr;
#endif
}

static void encodePs2Time(std::time_t t, uint8_t out[8])
{
    std::tm tm{};
    if (!localtimeSafe(&t, &tm))
    {
        std::memset(out, 0, 8);
        return;
    }

    uint16_t year = static_cast<uint16_t>(tm.tm_year + 1900);
    out[0] = 0;
    out[1] = static_cast<uint8_t>(tm.tm_sec);
    out[2] = static_cast<uint8_t>(tm.tm_min);
    out[3] = static_cast<uint8_t>(tm.tm_hour);
    out[4] = static_cast<uint8_t>(tm.tm_mday);
    out[5] = static_cast<uint8_t>(tm.tm_mon + 1);
    out[6] = static_cast<uint8_t>(year & 0xFF);
    out[7] = static_cast<uint8_t>((year >> 8) & 0xFF);
}

static std::time_t fileTimeToTimeT(std::filesystem::file_time_type ft)
{
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        ft - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
    return std::chrono::system_clock::to_time_t(sctp);
}

static bool gmtimeSafe(const std::time_t *t, std::tm *out)
{
#ifdef _WIN32
    return gmtime_s(out, t) == 0;
#else
    return gmtime_r(t, out) != nullptr;
#endif
}

static int getTimezoneOffsetMinutes()
{
    std::time_t now = std::time(nullptr);
    std::tm local{};
    std::tm gmt{};
    if (!localtimeSafe(&now, &local) || !gmtimeSafe(&now, &gmt))
        return 0;

    std::time_t localTime = std::mktime(&local);
    std::time_t gmtTime = std::mktime(&gmt);
    if (localTime == static_cast<std::time_t>(-1) || gmtTime == static_cast<std::time_t>(-1))
        return 0;

    double diff = std::difftime(localTime, gmtTime);
    return static_cast<int>(diff / 60.0);
}

static uint32_t packOsdConfig(uint32_t spdifMode, uint32_t screenType, uint32_t videoOutput,
                              uint32_t japLanguage, uint32_t ps1drvConfig, uint32_t version,
                              uint32_t language, int timezoneOffset)
{
    uint32_t raw = 0;
    raw |= (spdifMode & 0x1) << 0;
    raw |= (screenType & 0x3) << 1;
    raw |= (videoOutput & 0x1) << 3;
    raw |= (japLanguage & 0x1) << 4;
    raw |= (ps1drvConfig & 0xFF) << 5;
    raw |= (version & 0x7) << 13;
    raw |= (language & 0x1F) << 16;
    raw |= (static_cast<uint32_t>(timezoneOffset) & 0x7FF) << 21;
    return raw;
}

static uint32_t packOsdConfig2(uint32_t format, uint32_t daylightSaving, uint32_t timeFormat,
                               uint32_t dateFormat, uint32_t version, uint32_t language)
{
    const uint32_t flags =
        ((daylightSaving & 0x1u) << 4) |
        ((timeFormat & 0x1u) << 5) |
        ((dateFormat & 0x3u) << 6);

    return (format & 0xFFu) |
           ((flags & 0xFFu) << 8) |
           ((version & 0xFFu) << 16) |
           ((language & 0xFFu) << 24);
}

static int decodeTimezoneOffset(uint32_t raw)
{
    int tz = static_cast<int>((raw >> 21) & 0x7FF);
    if (tz & 0x400)
        tz |= ~0x7FF;
    return tz;
}

static int clampTimezoneOffset(int tz)
{
    if (tz < -1024)
        return -1024;
    if (tz > 1023)
        return 1023;
    return tz;
}

static uint32_t sanitizeOsdConfigRaw(uint32_t raw)
{
    uint32_t spdifMode = raw & 0x1;
    uint32_t screenType = (raw >> 1) & 0x3;
    if (screenType > 2)
        screenType = 0;
    uint32_t videoOutput = (raw >> 3) & 0x1;
    uint32_t japLanguage = (raw >> 4) & 0x1;
    uint32_t ps1drvConfig = (raw >> 5) & 0xFF;
    uint32_t version = (raw >> 13) & 0x7;
    if (version > 2)
        version = 1;
    uint32_t language = (raw >> 16) & 0x1F;
    int tz = clampTimezoneOffset(decodeTimezoneOffset(raw));
    return packOsdConfig(spdifMode, screenType, videoOutput, japLanguage, ps1drvConfig, version, language, tz);
}

static uint32_t sanitizeOsdConfig2Raw(uint32_t raw)
{
    const uint32_t format = raw & 0xFFu;
    const uint32_t flags = (raw >> 8) & 0xFFu;
    const uint32_t daylightSaving = (flags >> 4) & 0x1u;
    const uint32_t timeFormat = (flags >> 5) & 0x1u;
    uint32_t dateFormat = (flags >> 6) & 0x3u;
    if (dateFormat > 2u)
        dateFormat = 0u;

    uint32_t version = (raw >> 16) & 0xFFu;
    if (version > 2u)
        version = 1u;

    const uint32_t language = (raw >> 24) & 0xFFu;
    return packOsdConfig2(format, daylightSaving, timeFormat, dateFormat, version, language);
}

static uint32_t syncOsdConfigRawVersionLanguage(uint32_t raw, uint32_t version, uint32_t language)
{
    raw &= ~((0x7u << 13) | (0x1Fu << 16));
    raw |= (version & 0x7u) << 13;
    raw |= (language & 0x1Fu) << 16;
    return sanitizeOsdConfigRaw(raw);
}

static uint32_t makeReadableOsdConfig2RawLocked()
{
    const uint32_t version = (g_osd_config_raw >> 13) & 0x7u;
    const uint32_t language = (g_osd_config_raw >> 16) & 0x1Fu;
    uint32_t raw = g_osd_config2_raw & 0x0000FFFFu;
    raw |= (version & 0xFFu) << 16;
    raw |= (language & 0xFFu) << 24;
    return sanitizeOsdConfig2Raw(raw);
}

static void ensureOsdConfigInitialized()
{
    std::lock_guard<std::mutex> lock(g_osd_mutex);
    if (g_osd_config_initialized)
        return;

    int tz = clampTimezoneOffset(getTimezoneOffsetMinutes());
    uint32_t spdifMode = 1;   // disabled
    uint32_t screenType = 0;  // 4:3
    uint32_t videoOutput = 0; // RGB
    uint32_t japLanguage = 1; // non-japanese
    uint32_t ps1drvConfig = 0;
    uint32_t version = 1;  // OSD2
    uint32_t language = 1; // English
    g_osd_config_raw = packOsdConfig(spdifMode, screenType, videoOutput, japLanguage, ps1drvConfig, version, language, tz);
    g_osd_config2_raw = packOsdConfig2(0, 0, 0, 0, version, language);
    g_osd_config_initialized = true;
}

static uint32_t allocTlsAddr(uint8_t *rdram)
{
    if (!rdram || kTlsPoolCount == 0)
        return 0;

    std::lock_guard<std::mutex> lock(g_tls_mutex);
    uint32_t slot = g_tls_index++ % kTlsPoolCount;
    uint32_t addr = kTlsPoolBase + (slot * kTlsBlockSize);
    rpcZeroRdram(rdram, addr, kTlsBlockSize);
    return addr;
}

static uint32_t allocBootModeAddr(uint8_t *rdram, size_t bytes)
{
    if (!rdram)
        return 0;

    size_t aligned = (bytes + 15u) & ~15u;
    if (g_bootmode_pool_offset + aligned > kBootModePoolBytes)
        return 0;

    uint32_t addr = kBootModePoolBase + g_bootmode_pool_offset;
    g_bootmode_pool_offset += static_cast<uint32_t>(aligned);
    rpcZeroRdram(rdram, addr, aligned);
    return addr;
}

static uint32_t createBootModeEntry(uint8_t *rdram, uint8_t id, uint16_t value, uint8_t lenField, const uint32_t *data, uint8_t dataCount)
{
    uint8_t allocCount = (dataCount == 0) ? 1 : dataCount;
    size_t bytes = static_cast<size_t>(1 + allocCount) * sizeof(uint32_t);
    uint32_t addr = allocBootModeAddr(rdram, bytes);
    if (!addr)
        return 0;

    uint32_t header = (static_cast<uint32_t>(lenField) << 24) |
                      (static_cast<uint32_t>(id) << 16) |
                      (static_cast<uint32_t>(value) & 0xFFFFu);

    uint32_t *dst = reinterpret_cast<uint32_t *>(getMemPtr(rdram, addr));
    if (!dst)
        return 0;

    dst[0] = header;
    for (uint8_t i = 0; i < allocCount; ++i)
    {
        dst[1 + i] = (data && i < dataCount) ? data[i] : 0;
    }

    return addr;
}

static void ensureBootModeTable(uint8_t *rdram)
{
    std::lock_guard<std::mutex> lock(g_bootmode_mutex);
    if (g_bootmode_initialized)
        return;

    g_bootmode_pool_offset = 0;
    g_bootmode_addresses.clear();

    const uint32_t boot3Data[1] = {0};
    const uint32_t boot5Data[1] = {0};

    g_bootmode_addresses[1] = createBootModeEntry(rdram, 1, 0, 0, nullptr, 0);
    g_bootmode_addresses[3] = createBootModeEntry(rdram, 3, 0, 1, boot3Data, 1);
    g_bootmode_addresses[4] = createBootModeEntry(rdram, 4, 0, 0, nullptr, 0);
    g_bootmode_addresses[5] = createBootModeEntry(rdram, 5, 0, 1, boot5Data, 1);
    g_bootmode_addresses[6] = createBootModeEntry(rdram, 6, 0, 0, nullptr, 0);
    g_bootmode_addresses[7] = createBootModeEntry(rdram, 7, 0, 0, nullptr, 0);

    g_bootmode_initialized = true;
}

