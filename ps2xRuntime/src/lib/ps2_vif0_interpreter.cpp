// VIF0 / VU0 micro-memory path.
//
// Hand-ported from ran-j/PS2Recomp origin/main (merged PR #48, "Issue 11 vu0
// macro"), adapted to this tree's local API. Upstream keeps the VU0 buffers
// and the pending-VIF0 queue as PS2Memory data members; here they are
// TU-globals so that no class layout changes and the prebuilt corpus .so stays
// pairable with a runtime-only rebuild. See runtime/ps2_vif0.h for the full
// ABI rationale.

#include "runtime/ps2_vif0.h"
#include "runtime/ps2_memory.h"
#include "ps2_runtime.h" // ps2DiagEnvLimit / ps2DiagLogBudget (shared budget machinery)

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

#include <emmintrin.h>

// Defined in ps2_gs_gpu.cpp: one process-wide monotonic epoch shared by every
// DQ8 probe, so timestamps taken in different TUs are directly comparable.
uint64_t dq8ProbeNowMs();

namespace
{
    // VIF opcodes. Kept local rather than shared with ps2_vif1_interpreter.cpp,
    // which declares its own copy in an unnamed enum at file scope.
    constexpr uint8_t kVif0Nop = 0x00;
    constexpr uint8_t kVif0Stcycl = 0x01;
    constexpr uint8_t kVif0Offset = 0x02;
    constexpr uint8_t kVif0Base = 0x03;
    constexpr uint8_t kVif0Itop = 0x04;
    constexpr uint8_t kVif0Stmod = 0x05;
    constexpr uint8_t kVif0Mskpath3 = 0x06;
    constexpr uint8_t kVif0Mark = 0x07;
    constexpr uint8_t kVif0Flushe = 0x10;
    constexpr uint8_t kVif0Flush = 0x11;
    constexpr uint8_t kVif0Flusha = 0x13;
    constexpr uint8_t kVif0Mscal = 0x14;
    constexpr uint8_t kVif0Mscnt = 0x17;
    constexpr uint8_t kVif0Mscalf = 0x15;
    constexpr uint8_t kVif0Stmask = 0x20;
    constexpr uint8_t kVif0Strow = 0x30;
    constexpr uint8_t kVif0Stcol = 0x31;
    constexpr uint8_t kVif0Mpg = 0x4A;

    // VU0 micro memory / data memory. One VU0 per machine, so one instance.
    alignas(16) uint8_t s_vu0Code[PS2_VU0_CODE_SIZE];
    alignas(16) uint8_t s_vu0Data[PS2_VU0_DATA_SIZE];

    std::atomic<uint64_t> s_mpgUploads{0};
    std::atomic<uint32_t> s_lastMpgInstrs{0};
    std::atomic<uint32_t> s_lastMpgDest{0};

    std::atomic<uint64_t> s_vif0Calls{0};
    std::atomic<uint64_t> s_vif0Unpacks{0};
    std::atomic<uint64_t> s_vif0Mscal{0};

    // VALUE-tested gate, never getenv()!=nullptr: this tree has a documented
    // history of presence-gate bugs where `FOO=0` silently enabled a probe.
    bool vif0TraceOn()
    {
        static const bool on = []()
        {
            const char *e = std::getenv("PS2X_VIF0_TRACE");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        return on;
    }

    struct Vif0Transfer
    {
        bool fromScratchpad = false;
        uint32_t srcAddr = 0;
        uint32_t qwc = 0;
        std::vector<uint8_t> chainData;
        // (chainData offset -> guest physical address of that segment's first
        // byte), increasing offset order. Mirrors
        // PS2Memory::PendingTransfer::chainSegMap; diagnostics only. See
        // g_vif0ChainSegMap / vif0ChainOffsetToGuest below.
        std::vector<std::pair<uint32_t, uint32_t>> chainSegMap;
    };

    std::mutex s_vif0QueueMutex;
    std::deque<Vif0Transfer> s_vif0Queue;
} // namespace

// Segment map of the VIF0 chain currently being interpreted. A TU-global
// rather than a PS2Memory member, for the same ABI reason as everything else
// in this file: adding a data member to PS2Memory shifts its layout and
// breaks the prebuilt corpus .so. Mirrors g_vif1ChainSegMap in
// ps2_vif0_interpreter.cpp's VIF1 sibling.
//
// Set immediately before a ps2xProcessVIF0Data(mem, const uint8_t*, uint32_t)
// call whose source is known, cleared immediately after. Empty means
// "unknown source" (e.g. a scratchpad-sourced segment, or no caller bothered
// to set it), in which case chainOffsetToGuest below returns the sentinel
// 0xFFFFFFFF rather than guessing.
namespace
{
    std::vector<std::pair<uint32_t, uint32_t>> g_vif0ChainSegMap;

    // Map a parse offset within the buffer just given to
    // ps2xProcessVIF0Data() back to the guest physical address those bytes
    // were DMA'd from. Diagnostics only: this is what lets the MPG-upload log
    // report which EE address (e.g. the 88-instruction program at 0x38188c vs
    // the 112-instruction one at 0x390750) an upload actually came from,
    // instead of only its destination in VU0 micro memory.
    uint32_t vif0ChainOffsetToGuest(uint32_t chainOffset)
    {
        const auto &m = g_vif0ChainSegMap;
        if (m.empty())
            return 0xFFFFFFFFu;
        size_t lo = 0, hi = m.size();
        while (lo < hi)
        {
            const size_t mid = lo + (hi - lo) / 2;
            if (m[mid].first <= chainOffset)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo == 0)
            return 0xFFFFFFFFu;
        const auto &seg = m[lo - 1];
        if (seg.second == 0xFFFFFFFFu)
            return 0xFFFFFFFFu; // scratchpad-sourced segment
        return seg.second + (chainOffset - seg.first);
    }
} // namespace

uint8_t *ps2xVu0Code() { return s_vu0Code; }
uint8_t *ps2xVu0Data() { return s_vu0Data; }
bool ps2xVu0MicroLoaded() { return s_mpgUploads.load(std::memory_order_relaxed) != 0u; }
uint64_t ps2xVu0MpgUploadCount() { return s_mpgUploads.load(std::memory_order_relaxed); }
uint32_t ps2xVu0LastMpgInstrCount() { return s_lastMpgInstrs.load(std::memory_order_relaxed); }
uint32_t ps2xVu0LastMpgDestAddr() { return s_lastMpgDest.load(std::memory_order_relaxed); }

void ps2xProcessVIF0Data(PS2Memory &mem, uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    // Direct (non-chain) transfer: the whole buffer is one contiguous segment
    // starting at srcPhys, so a one-entry map is exact, not a guess.
    g_vif0ChainSegMap.assign(1, std::make_pair(0u, srcPhys));
    ps2xProcessVIF0Data(mem, mem.m_rdram + srcPhys, sizeBytes);
    g_vif0ChainSegMap.clear();
}

void ps2xProcessVIF0Data(PS2Memory &mem, const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes == 0u)
        return;

    VIFRegisters &vif0_regs = mem.vif0_regs;
    s_vif0Calls.fetch_add(1, std::memory_order_relaxed);

    uint32_t pos = 0;
    while (pos + 4 <= sizeBytes)
    {
        uint32_t cmd = 0u;
        std::memcpy(&cmd, data + pos, sizeof(cmd));
        pos += 4u;

        const uint8_t opcode = static_cast<uint8_t>((cmd >> 24) & 0x7Fu);
        const uint16_t imm = static_cast<uint16_t>(cmd & 0xFFFFu);
        const uint8_t num = static_cast<uint8_t>((cmd >> 16) & 0xFFu);
        const bool irq = (cmd & 0x80000000u) != 0u;

        vif0_regs.code = cmd;
        vif0_regs.num = num;
        if (irq)
            vif0_regs.stat |= (1u << 11);

        if (opcode == kVif0Nop)
        {
            continue;
        }
        else if (opcode == kVif0Stcycl)
        {
            vif0_regs.cycle = imm;
            continue;
        }
        else if (opcode == kVif0Offset)
        {
            vif0_regs.ofst = imm & 0x3FFu;
            continue;
        }
        else if (opcode == kVif0Base)
        {
            vif0_regs.base = imm & 0x3FFu;
            continue;
        }
        else if (opcode == kVif0Itop)
        {
            vif0_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == kVif0Stmod)
        {
            vif0_regs.mode = imm & 3u;
            continue;
        }
        else if (opcode == kVif0Mskpath3)
        {
            continue;
        }
        else if (opcode == kVif0Mark)
        {
            vif0_regs.mark = imm;
            vif0_regs.stat |= (1u << 6);
            continue;
        }
        else if (opcode == kVif0Flushe || opcode == kVif0Flush || opcode == kVif0Flusha)
        {
            continue;
        }
        else if (opcode == kVif0Mscal || opcode == kVif0Mscalf || opcode == kVif0Mscnt)
        {
            // VIF0-initiated VU0 micro-mode start. DQ8 does not use this to run
            // the rotation-matrix program -- the recompiled EE code reaches it
            // through VCALLMS, which routes to
            // PS2Runtime::executeVU0Microprogram with a live R5900Context. The
            // VIF0 path has no context to run against, so record and log only.
            s_vif0Mscal.fetch_add(1, std::memory_order_relaxed);
            if (vif0TraceOn())
            {
                std::cerr << "[vif0:mscal] opcode=0x" << std::hex << static_cast<uint32_t>(opcode)
                          << " imm=0x" << imm << std::dec
                          << " (recorded, not executed: no R5900Context on the VIF0 path)"
                          << std::endl;
            }
            continue;
        }
        else if (opcode == kVif0Stmask)
        {
            if (pos + 4u > sizeBytes)
                break;
            std::memcpy(&vif0_regs.mask, data + pos, sizeof(vif0_regs.mask));
            pos += 4u;
            continue;
        }
        else if (opcode == kVif0Strow)
        {
            if (pos + 16u > sizeBytes)
                break;
            std::memcpy(vif0_regs.row, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == kVif0Stcol)
        {
            if (pos + 16u > sizeBytes)
                break;
            std::memcpy(vif0_regs.col, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == kVif0Mpg)
        {
            // MPG imm is a micro-memory address in units of 8 bytes (one
            // instruction pair). NUM==0 means 256 instructions.
            const uint32_t destAddr = static_cast<uint32_t>(imm & 0x1FFu) * 8u;
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            uint32_t copyBytes = 0u;
            if (destAddr < PS2_VU0_CODE_SIZE && mpgBytes > 0u)
            {
                copyBytes = mpgBytes;
                if (destAddr + copyBytes > PS2_VU0_CODE_SIZE)
                    copyBytes = PS2_VU0_CODE_SIZE - destAddr;
                if (pos + copyBytes <= sizeBytes)
                {
                    // Overwrite in place. This is what makes DQ8's hot-swap
                    // work without any special casing: the 88-instruction
                    // program from ELF 0x38188c and the 112-instruction one
                    // from 0x390750 both target micro address 0, and whichever
                    // landed last is simply what is in micro memory.
                    // Source EE address of the instruction body itself (not
                    // the 4-byte MPG vifcode ahead of it): `pos` at this point
                    // is exactly the chain offset the memcpy above reads from.
                    // Resolved via the chain segment map so a reassembled
                    // multi-tag chain still answers correctly instead of only
                    // the direct (non-chain) transfer path.
                    const uint32_t srcAddr = vif0ChainOffsetToGuest(pos);

                    std::memcpy(s_vu0Code + destAddr, data + pos, copyBytes);
                    const uint64_t n = s_mpgUploads.fetch_add(1, std::memory_order_relaxed);
                    s_lastMpgInstrs.store(instructionCount, std::memory_order_relaxed);
                    s_lastMpgDest.store(destAddr, std::memory_order_relaxed);

                    // "Did the VU0 microprogram ever actually arrive" (and
                    // from which EE address) is the single load-bearing fact
                    // this whole port turns on, so this is NOT capped at a
                    // handful of events by default -- a run that hot-swaps the
                    // program many times must show all of them, not just the
                    // first few. Env-tunable, 0 = unlimited; the budget
                    // machinery announces truncation exactly once so a capped
                    // run is never misread as one that saw nothing further.
                    static const uint32_t kMpgMaxLogs =
                        ps2DiagEnvLimit("PS2X_VU0_MPG_MAX_LOGS", 4096u);
                    static std::atomic<bool> s_mpgTruncated{false};
                    if (ps2DiagLogBudget(std::cerr, "[vu0:mpg]", "PS2X_VU0_MPG_MAX_LOGS",
                                          kMpgMaxLogs, static_cast<uint32_t>(n), s_mpgTruncated))
                    {
                        std::cerr << "[vu0:mpg] t=" << std::dec << dq8ProbeNowMs() << "ms upload#" << n
                                  << " dest=0x" << std::hex << destAddr
                                  << " src=0x" << srcAddr << std::dec
                                  << " instrs=" << instructionCount
                                  << " bytes=" << copyBytes
                                  << std::endl;
                    }
                }
            }

            pos += mpgBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else if ((opcode & 0x60u) == 0x60u)
        {
            // UNPACK into VU0 data memory.
            const uint8_t vn = static_cast<uint8_t>((opcode >> 2) & 0x3u);
            const uint8_t vl = static_cast<uint8_t>(opcode & 0x3u);
            const int components = static_cast<int>(vn) + 1;
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
                bitsPerComponent = (vn == 3u) ? 4 : 16;
                break;
            default:
                break;
            }
            const int bitsPerVector = (vl == 3u && vn == 3u) ? 16 : (components * bitsPerComponent);
            uint32_t bytesPerVector = static_cast<uint32_t>((bitsPerVector + 7) / 8);
            const uint32_t writeVectorCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            uint32_t cl = vif0_regs.cycle & 0xFFu;
            uint32_t wl = (vif0_regs.cycle >> 8) & 0xFFu;
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
            totalBytes = (totalBytes + 3u) & ~3u;

            if (pos + totalBytes <= sizeBytes && vl == 0u)
            {
                s_vif0Unpacks.fetch_add(1, std::memory_order_relaxed);
                uint32_t vuAddr = static_cast<uint32_t>(imm & 0x3FFu);
                if ((imm & 0x8000u) != 0u)
                    vuAddr = (vuAddr + (vif0_regs.tops & 0x3FFu)) & 0x3FFu;
                const uint8_t *srcBase = data + pos;
                uint32_t srcIndex = 0u;
                for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
                {
                    const uint32_t cyclePos = writeIndex % wl;
                    const bool sourceAvailable = (cl >= wl) || (cyclePos < cl);
                    uint32_t destVec = (cl >= wl) ? ((vuAddr + (writeIndex / wl) * cl + cyclePos) & 0x3FFu)
                                                  : ((vuAddr + writeIndex) & 0x3FFu);
                    const uint32_t destOff = destVec * 16u;
                    if (destOff + 16u > PS2_VU0_DATA_SIZE)
                    {
                        if (sourceAvailable && srcIndex < sourceVectorCount)
                            ++srcIndex;
                        continue;
                    }
                    if (!sourceAvailable || srcIndex >= sourceVectorCount)
                        continue;
                    const uint8_t *srcVec = srcBase + srcIndex * bytesPerVector;
                    ++srcIndex;
                    uint32_t lanes[4] = {0u, 0u, 0u, 0u};
                    std::memcpy(lanes, s_vu0Data + destOff, sizeof(lanes));
                    const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                    for (uint32_t c = 0; c < limit; ++c)
                    {
                        uint32_t scalar = 0u;
                        std::memcpy(&scalar, srcVec + c * 4u, sizeof(scalar));
                        lanes[c] = scalar;
                    }
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(s_vu0Data + destOff),
                                     _mm_loadu_si128(reinterpret_cast<const __m128i *>(lanes)));
                }
            }
            pos += totalBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else
        {
            break;
        }
    }
}

void ps2xEnqueueVif0Chain(std::vector<uint8_t> &&chainData,
                          std::vector<std::pair<uint32_t, uint32_t>> &&chainSegMap)
{
    if (chainData.empty())
        return;
    Vif0Transfer t;
    t.chainData = std::move(chainData);
    t.chainSegMap = std::move(chainSegMap);
    std::lock_guard<std::mutex> lock(s_vif0QueueMutex);
    s_vif0Queue.push_back(std::move(t));
}

void ps2xEnqueueVif0Transfer(bool fromScratchpad, uint32_t srcAddr, uint32_t qwc)
{
    if (qwc == 0u)
        return;
    Vif0Transfer t;
    t.fromScratchpad = fromScratchpad;
    t.srcAddr = srcAddr;
    t.qwc = qwc;
    std::lock_guard<std::mutex> lock(s_vif0QueueMutex);
    s_vif0Queue.push_back(std::move(t));
}

bool ps2xHasPendingVif0()
{
    std::lock_guard<std::mutex> lock(s_vif0QueueMutex);
    return !s_vif0Queue.empty();
}

void ps2xDrainVif0Transfers(PS2Memory &mem)
{
    std::deque<Vif0Transfer> pending;
    {
        std::lock_guard<std::mutex> lock(s_vif0QueueMutex);
        pending.swap(s_vif0Queue);
    }

    for (auto &p : pending)
    {
        if (!p.chainData.empty())
        {
            // Real reassembled-chain segment map: lets the MPG-upload log
            // resolve an upload's true EE source address even when the
            // upload's payload was coalesced from a DMAtag chain rather than
            // one contiguous transfer.
            g_vif0ChainSegMap = std::move(p.chainSegMap);
            ps2xProcessVIF0Data(mem, p.chainData.data(), static_cast<uint32_t>(p.chainData.size()));
            g_vif0ChainSegMap.clear();
            continue;
        }
        if (p.qwc == 0u)
            continue;

        uint32_t srcPhys = 0u;
        const uint64_t bytes64 = static_cast<uint64_t>(p.qwc) * 16ull;
        uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
        try
        {
            srcPhys = mem.translateAddress(p.srcAddr);
        }
        catch (const std::exception &)
        {
            continue;
        }

        if (p.fromScratchpad)
        {
            // No chain segment map for a raw scratchpad transfer -- clear so
            // a stale map from a previous (chain-path) iteration can never
            // leak in and misreport an unrelated EE address.
            g_vif0ChainSegMap.clear();
            uint32_t bytesLeft = sizeBytes;
            while (bytesLeft > 0)
            {
                if (srcPhys >= PS2_SCRATCHPAD_SIZE)
                    srcPhys = 0;
                uint32_t chunk = bytesLeft;
                if (srcPhys + chunk > PS2_SCRATCHPAD_SIZE)
                    chunk = PS2_SCRATCHPAD_SIZE - srcPhys;
                if (chunk == 0)
                    break;
                ps2xProcessVIF0Data(mem, mem.m_scratchpad + srcPhys, chunk);
                bytesLeft -= chunk;
                srcPhys += chunk;
            }
        }
        else
        {
            uint32_t bytesLeft = sizeBytes;
            while (bytesLeft > 0)
            {
                if (srcPhys >= PS2_RAM_SIZE)
                    srcPhys = 0;
                uint32_t chunk = bytesLeft;
                if (srcPhys + chunk > PS2_RAM_SIZE)
                    chunk = PS2_RAM_SIZE - srcPhys;
                if (chunk == 0)
                    break;
                ps2xProcessVIF0Data(mem, srcPhys, chunk);
                bytesLeft -= chunk;
                srcPhys += chunk;
            }
        }
    }
}
