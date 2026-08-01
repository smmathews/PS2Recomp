#pragma once

// rpcInvokeFunction, hoisted out of Runtime.h so it can be shared verbatim
// (not copy-pasted) by TUs outside Kernel/Syscalls -- currently Kernel/Stubs/
// MPEG.cpp's sceMpeg decode-event callback dispatcher (scempeg-callback-
// design.md §B.1). Runtime.h still owns the one call site inside
// Kernel/Syscalls (SyscallOverride dispatch); it just includes this header
// instead of defining its own copy.

#include "ps2_runtime.h"

#include <atomic>
#include <iostream>

enum class RpcInvokeExitReason
{
    Returned,
    NullPc,
    MissingFunction,
    StepLimit,
    SamePcLimit
};

inline const char *rpcInvokeExitReasonName(RpcInvokeExitReason reason)
{
    switch (reason)
    {
    case RpcInvokeExitReason::Returned:
        return "returned";
    case RpcInvokeExitReason::NullPc:
        return "null-pc";
    case RpcInvokeExitReason::MissingFunction:
        return "missing-function";
    case RpcInvokeExitReason::StepLimit:
        return "step-limit";
    case RpcInvokeExitReason::SamePcLimit:
        return "same-pc-limit";
    default:
        return "unknown";
    }
}

namespace ps2_rpc_invoke_detail
{
    inline void setRegU32(R5900Context *ctx, int reg, uint32_t value)
    {
        if (reg <= 0 || reg > 31)
            return;
        ctx->r[reg] = _mm_set_epi64x(0, static_cast<int64_t>(static_cast<int32_t>(value)));
    }
}

// Runs on the CALLING FIBER -- the guest thread that is already executing
// (an RPC handler, a syscall override, or -- per scempeg-callback-design.md
// -- a sceMpeg stub invoking a game-registered decode-event callback). Do NOT
// wrap calls to this in AsyncGuestScope: the calling fiber already holds the
// guest execution slot. Structurally immune to live-context clobber: it runs
// on a COPY of *ctx, on a private guest stack, with $ra pointed at a sentinel
// return address the interpreter loop stops on -- the caller's own registers
// and stack are never touched.
inline bool rpcInvokeFunction(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
                              uint32_t funcAddr, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t *outV0)
{
    if (!runtime || !ctx || !funcAddr || !runtime->hasFunction(funcAddr))
        return false;

    constexpr uint32_t kRpcInvokeStackSize = 0x4000u;
    constexpr uint32_t kRpcInvokeReturnSentinel = 0x00FFF000u;
    constexpr uint32_t kRpcInvokeMaxSteps = 0x8000u;

    R5900Context tmp = *ctx;
    ps2_rpc_invoke_detail::setRegU32(&tmp, 4, a0);
    ps2_rpc_invoke_detail::setRegU32(&tmp, 5, a1);
    ps2_rpc_invoke_detail::setRegU32(&tmp, 6, a2);
    ps2_rpc_invoke_detail::setRegU32(&tmp, 7, a3);

    thread_local uint32_t s_rpcInvokeStackBase = 0u;
    thread_local uint32_t s_rpcInvokeStackTop = 0u;
    if (s_rpcInvokeStackTop == 0u)
    {
        const uint32_t stackBase = runtime->guestMalloc(kRpcInvokeStackSize, 16u);
        if (stackBase != 0u)
        {
            s_rpcInvokeStackBase = stackBase;
            s_rpcInvokeStackTop = (stackBase + kRpcInvokeStackSize) & ~0xFu;
        }
    }
    if (s_rpcInvokeStackTop != 0u)
    {
        ps2_rpc_invoke_detail::setRegU32(&tmp, 29, s_rpcInvokeStackTop);
    }
    (void)s_rpcInvokeStackBase;

    ps2_rpc_invoke_detail::setRegU32(&tmp, 31, kRpcInvokeReturnSentinel);
    tmp.pc = funcAddr;

    uint32_t steps = 0u;
    uint32_t lastPc = 0xFFFFFFFFu;
    uint32_t samePcCount = 0u;
    RpcInvokeExitReason exitReason = RpcInvokeExitReason::MissingFunction;
    while (tmp.pc != 0u &&
           tmp.pc != kRpcInvokeReturnSentinel &&
           runtime->hasFunction(tmp.pc) &&
           steps < kRpcInvokeMaxSteps)
    {
        const uint32_t pc = tmp.pc;
        if (pc == lastPc)
        {
            ++samePcCount;
            if (samePcCount > 0x2000u)
            {
                exitReason = RpcInvokeExitReason::SamePcLimit;
                break;
            }
        }
        else
        {
            lastPc = pc;
            samePcCount = 0u;
        }

        PS2Runtime::RecompiledFunction func = runtime->lookupFunction(pc);
        func(rdram, &tmp, runtime);
        ++steps;
    }

    if (outV0)
    {
        *outV0 = getRegU32(&tmp, 2);
    }

    if (tmp.pc == kRpcInvokeReturnSentinel)
    {
        return true;
    }

    if (tmp.pc == 0u)
    {
        exitReason = RpcInvokeExitReason::NullPc;
    }
    else if (steps >= kRpcInvokeMaxSteps)
    {
        exitReason = RpcInvokeExitReason::StepLimit;
    }
    else if (!runtime->hasFunction(tmp.pc))
    {
        exitReason = RpcInvokeExitReason::MissingFunction;
    }

    static std::atomic<uint32_t> s_rpcInvokeFailureLogs{0u};
    constexpr uint32_t kMaxRpcInvokeFailureLogs = 64u;
    const uint32_t logIndex = s_rpcInvokeFailureLogs.fetch_add(1u, std::memory_order_relaxed);
    if (logIndex < kMaxRpcInvokeFailureLogs)
    {
        std::cerr << "[SyscallOverride:invoke-failed]"
                  << " func=0x" << std::hex << funcAddr
                  << " exitPc=0x" << tmp.pc
                  << " ra=0x" << getRegU32(&tmp, 31)
                  << std::dec
                  << " steps=" << steps
                  << " reason=" << rpcInvokeExitReasonName(exitReason)
                  << std::endl;
    }

    return false;
}
