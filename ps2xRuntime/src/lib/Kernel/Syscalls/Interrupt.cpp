#include "Common.h"
#include "Interrupt.h"
#include "ps2_log.h"
#include "Stubs/GS.h"
#include "ps2_fiber.h"

namespace ps2_syscalls
{
    namespace interrupt_state
    {
        constexpr uint32_t kIntcVblankStart = 2u;
        constexpr uint32_t kIntcVblankEnd = 3u;
        constexpr auto kVblankPeriod = std::chrono::microseconds(16667);
        constexpr int kMaxCatchupTicks = 4;

        std::mutex g_irq_handler_mutex;
        std::mutex g_irq_worker_mutex;
        std::condition_variable g_irq_worker_cv;
        std::mutex g_vsync_flag_mutex;
        std::vector<std::pair<int, uint64_t>> g_vsync_waitList;
        std::atomic<bool> g_irq_worker_stop{false};
        std::atomic<bool> g_irq_worker_running{false};
        std::thread g_irq_worker_thread; // joinable worker handle so stopInterruptWorker() can join it
        uint32_t g_enabled_intc_mask = 0xFFFFFFFFu;
        uint32_t g_enabled_dmac_mask = 0xFFFFFFFFu;
        uint64_t g_vsync_tick_counter = 0u;
        VSyncFlagRegistration g_vsync_registration{};

        // Level-triggered pending INTC causes raised by completed DMA
        // transfers (e.g. VIF1 kick -> INTC cause 5). Delivered by the irq
        // worker on its next tick — NOT synchronously at the raise site,
        // because the sce libdma protocol registers/enables the completion
        // handler AFTER the kick returns (kick; CreateSema; AddIntcHandler;
        // EnableIntc; wait) — a synchronous dispatch would fire before the
        // handler exists and be lost. A pending bit stays set until a
        // dispatch actually ran >= 1 handler for the cause (or it ages out),
        // covering the raise-vs-register race in the other direction too.
        std::atomic<uint32_t> g_pending_intc_causes{0u};
        uint32_t g_pending_intc_age[32] = {};
        constexpr uint32_t kPendingIntcMaxAgeTicks = 120u; // ~2 s @60 Hz
    }

    using namespace interrupt_state;

    static void writeGuestU32NoThrow(uint8_t *rdram, uint32_t addr, uint32_t value)
    {
        if (addr == 0u)
        {
            return;
        }

        uint8_t *dst = getMemPtr(rdram, addr);
        if (!dst)
        {
            return;
        }
        std::memcpy(dst, &value, sizeof(value));
    }

    static void writeGuestU64NoThrow(uint8_t *rdram, uint32_t addr, uint64_t value)
    {
        if (addr == 0u)
        {
            return;
        }

        uint8_t *dst = getMemPtr(rdram, addr);
        if (!dst)
        {
            return;
        }
        std::memcpy(dst, &value, sizeof(value));
    }

    static uint32_t readGuestU32NoThrow(uint8_t *rdram, uint32_t addr)
    {
        if (addr == 0u)
        {
            return 0u;
        }

        uint8_t *src = getMemPtr(rdram, addr);
        if (!src)
        {
            return 0u;
        }

        uint32_t value = 0u;
        std::memcpy(&value, src, sizeof(value));
        return value;
    }

    static uint32_t getAsyncHandlerStackTop(PS2Runtime *runtime)
    {
        constexpr uint32_t kAsyncHandlerStackSize = 0x4000u;
        // Failure fallback: top of the kernel-area callback pool, NOT
        // PS2_RAM_SIZE-0x10 -- that address is inside the guest's own main
        // stack (DQ8 SetupThread: [0x01F40000, 0x02000000)) and running a
        // handler there corrupts live guest frames. Only reachable if the
        // 512 KB pool is exhausted (32 x 16 KB) or runtime is null.
        constexpr uint32_t kFallbackStackTop = 0x00100000u - 0x10u;
        thread_local PS2Runtime *s_cachedRuntime = nullptr;
        thread_local uint32_t s_cachedStackTop = 0u;

        if (runtime == nullptr)
        {
            return kFallbackStackTop;
        }

        if (s_cachedRuntime != runtime || s_cachedStackTop == 0u)
        {
            s_cachedRuntime = runtime;
            s_cachedStackTop = runtime->reserveAsyncCallbackStack(kAsyncHandlerStackSize, 16u);
        }

        return (s_cachedStackTop != 0u) ? s_cachedStackTop : kFallbackStackTop;
    }

    // Returns the number of handlers actually dispatched (0 when the cause is
    // masked, no handler is registered, or a handler body is unavailable).
    static int dispatchIntcHandlersForCause(uint8_t *rdram, PS2Runtime *runtime, uint32_t cause)
    {
        if (!rdram || !runtime)
        {
            return 0;
        }

        std::vector<IrqHandlerInfo> handlers;
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            if (cause < 32u && (g_enabled_intc_mask & (1u << cause)) == 0u)
            {
                return 0;
            }

            handlers.reserve(g_intcHandlers.size());
            for (const auto &[id, info] : g_intcHandlers)
            {
                (void)id;
                if (!info.enabled)
                {
                    continue;
                }
                if (info.cause != cause)
                {
                    continue;
                }
                if (info.handler == 0u)
                {
                    continue;
                }
                handlers.push_back(info);
            }
            std::sort(handlers.begin(), handlers.end(), [](const IrqHandlerInfo &a, const IrqHandlerInfo &b)
                      { return a.order < b.order; });
        }

        int dispatched = 0;
        AsyncGuestScope guestScope; // token released on any exit path
        for (const IrqHandlerInfo &info : handlers)
        {
            if (!runtime->hasFunction(info.handler))
            {
                if (cause == kIntcVblankStart)
                {
                    PS2_IF_AGRESSIVE_LOGS({
                        static std::atomic<uint32_t> s_missingHandlerLogCount{0u};
                        static const uint32_t kMaxMissingHandlerLogs =
                            ps2DiagEnvLimit("PS2X_INTC_MISSING_MAX_LOGS", 32u);
                        static std::atomic<bool> s_missingHandlerTruncated{false};
                        const uint32_t logIndex = s_missingHandlerLogCount.fetch_add(1u, std::memory_order_relaxed);
                        if (ps2DiagLogBudget(std::cout,
                                             "[INTC:missing]",
                                             "PS2X_INTC_MISSING_MAX_LOGS",
                                             kMaxMissingHandlerLogs,
                                             logIndex,
                                             s_missingHandlerTruncated))
                        {
                            auto flags = std::cout.flags();
                            std::cout << "[INTC:missing] cause=" << cause
                                      << " handler=0x" << std::hex << info.handler
                                      << std::dec
                                      << " id=" << info.id
                                      << std::endl;
                            std::cout.flags(flags);
                        }
                    });
                }
                continue;
            }

            try
            {
                R5900Context irqCtx{};
                SET_GPR_U32(&irqCtx, 28, info.gp);
                SET_GPR_U32(&irqCtx, 29, getAsyncHandlerStackTop(runtime));
                SET_GPR_U32(&irqCtx, 31, 0u);
                SET_GPR_U32(&irqCtx, 4, cause);
                SET_GPR_U32(&irqCtx, 5, info.arg);
                SET_GPR_U32(&irqCtx, 6, 0u);
                SET_GPR_U32(&irqCtx, 7, 0u);
                irqCtx.pc = info.handler;

                ++dispatched;
                while (irqCtx.pc != 0u && runtime && !runtime->isStopRequested())
                {
                    PS2Runtime::RecompiledFunction step = runtime->lookupFunction(irqCtx.pc);
                    if (!step)
                    {
                        break;
                    }
                    // Interrupt handlers must be able to preempt a guest thread that is
                    // spinning on interrupt-produced state, such as a vblank counter.
                    step(rdram, &irqCtx, runtime);
                }
            }
            catch (const ThreadExitException &)
            {
            }
            catch (const std::exception &e)
            {
                static uint32_t warnCount = 0;
                if (warnCount < 8u)
                {
                    std::cerr << "[INTC] handler 0x" << std::hex << info.handler
                              << " threw exception: " << e.what() << std::dec << std::endl;
                    ++warnCount;
                }
            }
        }
        return dispatched;
    }

    // Record a level-triggered INTC event for asynchronous delivery by the
    // irq worker (next tick, <= one vblank of latency — hardware-plausible
    // DMA-completion timing). See g_pending_intc_causes for why delivery must
    // not be synchronous with the raise site. Safe from any thread.
    void raisePendingIntc(uint32_t cause)
    {
        if (cause >= 32u || cause == kIntcVblankStart || cause == kIntcVblankEnd)
        {
            return; // vblank causes are the worker's own periodic dispatches
        }
        const uint32_t bit = 1u << cause;
        const uint32_t prev = g_pending_intc_causes.fetch_or(bit, std::memory_order_acq_rel);
        if ((prev & bit) == 0u)
        {
            static std::atomic<uint32_t> s_raiseLog{0u};
            const uint32_t n = s_raiseLog.fetch_add(1u, std::memory_order_relaxed);
            if (n < 16u || (n % 256u) == 0u)
            {
                std::cout << "[INTC:raise] cause=" << cause << " (pending, n=" << n << ")" << std::endl;
            }
        }
    }

    // Drain pending non-vblank INTC causes (called from the irq worker each
    // tick). A cause stays pending until >= 1 handler actually ran, or it
    // ages out (raise-vs-AddIntcHandler registration race coverage).
    static void drainPendingIntc(uint8_t *rdram, PS2Runtime *runtime)
    {
        uint32_t pending = g_pending_intc_causes.load(std::memory_order_acquire);
        while (pending != 0u)
        {
            const uint32_t cause = static_cast<uint32_t>(__builtin_ctz(pending));
            const uint32_t bit = 1u << cause;
            pending &= ~bit;

            const int ran = dispatchIntcHandlersForCause(rdram, runtime, cause);
            if (ran > 0)
            {
                g_pending_intc_causes.fetch_and(~bit, std::memory_order_acq_rel);
                g_pending_intc_age[cause] = 0u;
                static std::atomic<uint32_t> s_deliverLog{0u};
                const uint32_t n = s_deliverLog.fetch_add(1u, std::memory_order_relaxed);
                if (n < 16u || (n % 256u) == 0u)
                {
                    std::cout << "[INTC:deliver] cause=" << cause
                              << " handlers=" << ran << " (n=" << n << ")" << std::endl;
                }
            }
            else if (++g_pending_intc_age[cause] > kPendingIntcMaxAgeTicks)
            {
                g_pending_intc_causes.fetch_and(~bit, std::memory_order_acq_rel);
                g_pending_intc_age[cause] = 0u;
                static std::atomic<uint32_t> s_dropLog{0u};
                const uint32_t n = s_dropLog.fetch_add(1u, std::memory_order_relaxed);
                if (n < 8u || (n % 64u) == 0u)
                {
                    std::cout << "[INTC:drop] cause=" << cause
                              << " aged out with no registered/enabled handler (n="
                              << n << ")" << std::endl;
                }
            }
        }
    }

    void dispatchDmacHandlersForCause(uint8_t *rdram, PS2Runtime *runtime, uint32_t cause)
    {
        if (!rdram || !runtime)
        {
            return;
        }

        std::vector<IrqHandlerInfo> handlers;
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            if (cause < 32u && (g_enabled_dmac_mask & (1u << cause)) == 0u)
            {
                return;
            }

            handlers.reserve(g_dmacHandlers.size());
            for (const auto &[id, info] : g_dmacHandlers)
            {
                (void)id;
                if (!info.enabled)
                {
                    continue;
                }
                if (info.cause != cause)
                {
                    continue;
                }
                if (info.handler == 0u)
                {
                    continue;
                }
                handlers.push_back(info);
            }
            std::sort(handlers.begin(), handlers.end(), [](const IrqHandlerInfo &a, const IrqHandlerInfo &b)
                      { return a.order < b.order; });
        }

        auto runHandlers = [&]()
        {
            for (const IrqHandlerInfo &info : handlers)
            {
                if (!runtime->hasFunction(info.handler))
                {
                    continue;
                }

                try
                {
                    R5900Context irqCtx{};
                    SET_GPR_U32(&irqCtx, 28, info.gp);
                    SET_GPR_U32(&irqCtx, 29, getAsyncHandlerStackTop(runtime));
                    SET_GPR_U32(&irqCtx, 31, 0u);
                    SET_GPR_U32(&irqCtx, 4, cause);
                    SET_GPR_U32(&irqCtx, 5, info.arg);
                    SET_GPR_U32(&irqCtx, 6, 0u);
                    SET_GPR_U32(&irqCtx, 7, 0u);
                    irqCtx.pc = info.handler;

                    while (irqCtx.pc != 0u && runtime && !runtime->isStopRequested())
                    {
                        PS2Runtime::RecompiledFunction step = runtime->lookupFunction(irqCtx.pc);
                        if (!step)
                        {
                            break;
                        }
                        step(rdram, &irqCtx, runtime);
                    }
                }
                catch (const ThreadExitException &)
                {
                }
                catch (const std::exception &e)
                {
                    static uint32_t warnCount = 0;
                    if (warnCount < 8u)
                    {
                        std::cerr << "[DMAC] handler 0x" << std::hex << info.handler
                                  << " threw exception: " << e.what() << std::dec << std::endl;
                        ++warnCount;
                    }
                }
            }
        };

        // Unlike dispatchIntcHandlersForCause (only ever called from the IRQ
        // worker host thread), this function is also reachable synchronously
        // from guest code: sceSifSetDma (Stubs/SIF.cpp) calls it inline while
        // servicing a guest syscall, i.e. while the calling fiber IS the guest
        // execution slot. AsyncGuestScope's async_guest_begin() aborts by
        // design if invoked from the guest executor thread (that guard exists
        // to catch host workers mistakenly running there) -- so only borrow
        // the token when this call is NOT already running on the guest thread.
        if (ps2sched::is_guest_thread())
        {
            runHandlers();
        }
        else
        {
            AsyncGuestScope guestScope; // token released on any exit path
            runHandlers();
        }
    }

    static uint64_t signalVSyncFlag(uint8_t *rdram)
    {
        VSyncFlagRegistration reg{};
        uint64_t tickValue = 0u;
        {
            std::lock_guard<std::mutex> lock(g_vsync_flag_mutex);
            reg = g_vsync_registration;
            tickValue = ++g_vsync_tick_counter;
        }

        // Wake all guest threads waiting for the next vsync tick.
        // Called from the IRQ worker (a non-guest host thread). Use the identity-
        // validated wakeup so a recycled tid cannot deliver this tick to the wrong
        // fiber: each entry carries the parking fiber's generation token.
        std::vector<std::pair<int, uint64_t>> vsyncWaiters;
        {
            std::lock_guard<std::mutex> lk(g_vsync_flag_mutex);
            vsyncWaiters.swap(g_vsync_waitList);
        }
        for (const auto &[tid, token] : vsyncWaiters)
        {
            ps2sched::enqueue_external_wakeup_validated(tid, token);
        }

        if (reg.flagAddr != 0u)
        {
            writeGuestU32NoThrow(rdram, reg.flagAddr, 1u);
        }
        if (reg.tickAddr != 0u)
        {
            writeGuestU64NoThrow(rdram, reg.tickAddr, tickValue);
        }
        return tickValue;
    }

    static void interruptWorkerMain(uint8_t *rdram, PS2Runtime *runtime)
    {
        g_currentThreadId = -1;
        std::cerr << "[irq-worker] interrupt worker started\n";

        using clock = std::chrono::steady_clock;
        const auto workerStart = clock::now();
        auto nextTick = clock::now() + kVblankPeriod;
        uint64_t totalTicks = 0u; // all VBlank ticks processed (incl. catch-up)

        // Cheap tick-health summary: the per-tick cadence log (n<3 || n%60==0)
        // only bounds wakeups, it does not prove sustained 60Hz delivery.
        // Log total ticks + wall time + effective Hz every 600 ticks (~10 s)
        // and once on worker exit.
        auto logTickSummary = [&](const char *tag)
        {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     clock::now() - workerStart)
                                     .count();
            const double seconds = static_cast<double>(elapsed) / 1000.0;
            const double hz = (seconds > 0.0) ? (static_cast<double>(totalTicks) / seconds) : 0.0;
            std::cerr << "[irq-worker] " << tag
                      << " ticks=" << totalTicks
                      << " elapsed=" << seconds << "s"
                      << " rate=" << hz << "Hz\n";
        };

        while (runtime != nullptr && !runtime->isStopRequested())
        {
            {
                std::unique_lock<std::mutex> lock(g_irq_worker_mutex);
                if (g_irq_worker_cv.wait_until(lock, nextTick, []()
                                               { return g_irq_worker_stop.load(std::memory_order_acquire); }))
                {
                    break;
                }
            }

            const auto now = clock::now();
            int ticksToProcess = 0;
            while (now >= nextTick && ticksToProcess < kMaxCatchupTicks)
            {
                ++ticksToProcess;
                nextTick += kVblankPeriod;
            }
            if (ticksToProcess == 0)
            {
                continue;
            }

            {
                static std::atomic<uint32_t> s_tickLog{0};
                uint32_t n = s_tickLog.fetch_add(1, std::memory_order_relaxed);
                if (n < 3u || (n % 60u) == 0u)
                    std::cerr << "[irq-worker] tick #" << n
                              << " waiters=" << ps2sched::host_token_waiters()
                              << "\n";
            }
            for (int i = 0; i < ticksToProcess; ++i)
            {
                const uint64_t tickValue = signalVSyncFlag(rdram);
                ps2_stubs::dispatchGsSyncVCallback(rdram, runtime, tickValue);
                dispatchIntcHandlersForCause(rdram, runtime, kIntcVblankStart);
                std::this_thread::sleep_for(std::chrono::microseconds(500));
                dispatchIntcHandlersForCause(rdram, runtime, kIntcVblankEnd);
                // Deferred DMA-completion interrupts (e.g. VIF1 -> cause 5),
                // raised by submitDmaSend and delivered here one tick later.
                drainPendingIntc(rdram, runtime);
            }

            const uint64_t before = totalTicks;
            totalTicks += static_cast<uint64_t>(ticksToProcess);
            if ((before / 600u) != (totalTicks / 600u))
            {
                logTickSummary("tick-summary");
            }
        }

        logTickSummary("exit-summary");
        g_irq_worker_running.store(false, std::memory_order_release);
        g_irq_worker_cv.notify_all();
    }

    static void ensureInterruptWorkerRunning(uint8_t *rdram, PS2Runtime *runtime)
    {
        if (!rdram || !runtime)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(g_irq_worker_mutex);
        if (g_irq_worker_running.load(std::memory_order_acquire))
        {
            return;
        }

        // Reap a previously-stopped worker thread before starting a new one.
        if (g_irq_worker_thread.joinable())
        {
            g_irq_worker_thread.join();
        }

        g_irq_worker_stop.store(false, std::memory_order_release);
        g_irq_worker_running.store(true, std::memory_order_release);
        try
        {
            g_irq_worker_thread = std::thread(interruptWorkerMain, rdram, runtime); // JOINABLE
        }
        catch (...)
        {
            g_irq_worker_running.store(false, std::memory_order_release);
        }
    }

    void EnsureVSyncWorkerRunning(uint8_t *rdram, PS2Runtime *runtime)
    {
        ensureInterruptWorkerRunning(rdram, runtime);
    }

    uint64_t GetCurrentVSyncTick()
    {
        std::lock_guard<std::mutex> lock(g_vsync_flag_mutex);
        return g_vsync_tick_counter;
    }

    void signalInterruptWorkerStop()
    {
        g_irq_worker_stop.store(true, std::memory_order_release);
        g_irq_worker_cv.notify_all();
    }

    void stopInterruptWorker()
    {
        g_irq_worker_stop.store(true, std::memory_order_release);
        g_irq_worker_cv.notify_all();

        // Actually JOIN the worker rather than a 500ms give-up. The worker
        // loop checks g_irq_worker_stop on its CV wait and in its while-condition,
        // so it exits promptly. We must NOT hold g_irq_worker_mutex while joining
        // (the worker takes that mutex on its CV wait — joining under it would deadlock).
        std::thread workerToJoin;
        {
            std::lock_guard<std::mutex> lock(g_irq_worker_mutex);
            if (g_irq_worker_thread.joinable())
            {
                workerToJoin = std::move(g_irq_worker_thread);
            }
        }
        if (workerToJoin.joinable())
        {
            workerToJoin.join();
        }

        // Wake any guest threads waiting on vsync during shutdown.
        std::vector<std::pair<int, uint64_t>> vsyncWaiters;
        {
            std::lock_guard<std::mutex> lk(g_vsync_flag_mutex);
            vsyncWaiters.swap(g_vsync_waitList);
        }
        for (const auto &[tid, token] : vsyncWaiters)
        {
            ps2sched::enqueue_external_wakeup_validated(tid, token);
        }
    }

    uint64_t WaitForNextVSyncTick(uint8_t *rdram, PS2Runtime *runtime)
    {
        ensureInterruptWorkerRunning(rdram, runtime);

        // Opaque identity of the fiber that is about to park. Non-fiber host
        // workers get token 0 and never publish to the wait-list.
        const uint64_t selfToken = ps2sched::current_fiber_token();
        const bool onFiber = (selfToken != 0u);

        if (onFiber)
        {
            // Publish under g_vsync_flag_mutex; arm_park after the lock is
            // released so g_sched_mutex is never nested under it.
            {
                std::lock_guard<std::mutex> lock(g_vsync_flag_mutex);
                g_vsync_waitList.emplace_back(g_currentThreadId, selfToken);
            }
            ps2sched::arm_park();
        }

        // Block the current fiber; signalVSyncFlag calls the validated wakeup
        // from the IRQ worker thread to wake us.
        const ps2sched::BlockResult br = ps2sched::block_current();

        // A borrowed host worker cannot park on the fiber scheduler. Drop the token
        // (only if owned) so the IRQ worker / fibers run, then return the
        // current tick. A fiber returning WokenInWindow means a tick arrived during
        // the parking window — also fine to return the current tick.
        if (br == ps2sched::BlockResult::NonFiberOwner ||
            br == ps2sched::BlockResult::NonFiberNoTok)
        {
            nonFiberBlockBackoff(br);
        }

        // A fiber woken from a real park (Parked) may have been woken by
        // scheduler_shutdown / TerminateThread rather than a vsync tick. If so,
        // unwind instead of returning a tick value. Mirrors WaitSema's terminate
        // check after wake. (Borrowed host workers have no ThreadInfo and never
        // reach Parked, so this is fiber-only.)
        if (onFiber && br == ps2sched::BlockResult::Parked)
        {
            std::shared_ptr<ThreadInfo> info = lookupThreadInfo(g_currentThreadId);
            if (info && info->terminated.load())
            {
                // Drop our wait-list entry before unwinding so a recycled tid
                // cannot inherit a stale token.
                {
                    std::lock_guard<std::mutex> clLock(g_vsync_flag_mutex);
                    auto &wl = g_vsync_waitList;
                    auto it = std::find_if(wl.begin(), wl.end(),
                                           [selfToken](const std::pair<int, uint64_t> &e)
                                           { return e.second == selfToken; });
                    if (it != wl.end()) wl.erase(it);
                }
                throw ThreadExitException();
            }
        }

        // If we were woken by something other than a vsync tick (shutdown,
        // TerminateThread, or a wakeup during the parking window), signalVSyncFlag
        // never drained us, so our entry is still queued. Remove it by fiber-token
        // identity (NOT by tid, which can recycle). A real vsync wake already
        // swapped us out, so this erase is a harmless no-op on that path. Non-fiber
        // callers never published, so there is nothing to erase.
        std::lock_guard<std::mutex> lock(g_vsync_flag_mutex);
        if (onFiber)
        {
            auto &wl = g_vsync_waitList;
            auto it = std::find_if(wl.begin(), wl.end(),
                                   [selfToken](const std::pair<int, uint64_t> &e)
                                   { return e.second == selfToken; });
            if (it != wl.end())
            {
                wl.erase(it);
            }
        }
        return g_vsync_tick_counter;
    }

    void WaitVSyncTick(uint8_t *rdram, PS2Runtime *runtime)
    {
        (void)WaitForNextVSyncTick(rdram, runtime);
    }

    void SetVSyncFlag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t flagAddr = getRegU32(ctx, 4);
        const uint32_t tickAddr = getRegU32(ctx, 5);

        {
            std::lock_guard<std::mutex> lock(g_vsync_flag_mutex);
            g_vsync_registration.flagAddr = flagAddr;
            g_vsync_registration.tickAddr = tickAddr;
        }

        writeGuestU32NoThrow(rdram, flagAddr, 0u);
        writeGuestU64NoThrow(rdram, tickAddr, 0u);
        ensureInterruptWorkerRunning(rdram, runtime);
        setReturnS32(ctx, KE_OK);
    }

    void EnableIntc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t cause = getRegU32(ctx, 4);
        if (cause < 32u)
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            g_enabled_intc_mask |= (1u << cause);
        }
        if (cause == kIntcVblankStart || cause == kIntcVblankEnd)
        {
            PS2_IF_AGRESSIVE_LOGS({
                static std::atomic<uint32_t> s_enableLogCount{0u};
                static const uint32_t kMaxEnableLogs =
                    ps2DiagEnvLimit("PS2X_INTC_ENABLE_MAX_LOGS", 32u);
                static std::atomic<bool> s_enableTruncated{false};
                const uint32_t logIndex = s_enableLogCount.fetch_add(1u, std::memory_order_relaxed);
                if (ps2DiagLogBudget(std::cout,
                                     "[EnableIntc]",
                                     "PS2X_INTC_ENABLE_MAX_LOGS",
                                     kMaxEnableLogs,
                                     logIndex,
                                     s_enableTruncated))
                {
                    RUNTIME_LOG("[EnableIntc] cause=" << cause);
                }
            });
        }
        setReturnS32(ctx, KE_OK);
    }

    void iEnableIntc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        EnableIntc(rdram, ctx, runtime);
    }

    void DisableIntc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t cause = getRegU32(ctx, 4);
        if (cause < 32u)
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            g_enabled_intc_mask &= ~(1u << cause);
        }
        if (cause == kIntcVblankStart || cause == kIntcVblankEnd)
        {
            PS2_IF_AGRESSIVE_LOGS({
                static std::atomic<uint32_t> s_disableLogCount{0u};
                static const uint32_t kMaxDisableLogs =
                    ps2DiagEnvLimit("PS2X_INTC_DISABLE_MAX_LOGS", 32u);
                static std::atomic<bool> s_disableTruncated{false};
                const uint32_t logIndex = s_disableLogCount.fetch_add(1u, std::memory_order_relaxed);
                if (ps2DiagLogBudget(std::cout,
                                     "[DisableIntc]",
                                     "PS2X_INTC_DISABLE_MAX_LOGS",
                                     kMaxDisableLogs,
                                     logIndex,
                                     s_disableTruncated))
                {
                    RUNTIME_LOG("[DisableIntc] cause=" << cause);
                }
            });
        }
        setReturnS32(ctx, KE_OK);
    }

    void iDisableIntc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        DisableIntc(rdram, ctx, runtime);
    }

    void AddIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        IrqHandlerInfo info{};
        info.cause = getRegU32(ctx, 4);
        info.handler = getRegU32(ctx, 5);
        uint32_t next = getRegU32(ctx, 6);
        info.arg = getRegU32(ctx, 7);
        info.gp = getRegU32(ctx, 28);
        info.sp = getRegU32(ctx, 29);
        info.enabled = true;

        int handlerId = 0;
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            info.order = (next == 0) ? --g_intc_head_order : ++g_intc_tail_order;
            handlerId = g_nextIntcHandlerId++;
            info.id = handlerId;
            g_intcHandlers[handlerId] = info;
        }

        if (info.cause == kIntcVblankStart)
        {
            PS2_IF_AGRESSIVE_LOGS({
                static std::atomic<uint32_t> s_addHandlerLogCount{0u};
                static const uint32_t kMaxAddHandlerLogs =
                    ps2DiagEnvLimit("PS2X_INTC_ADDHANDLER_MAX_LOGS", 32u);
                static std::atomic<bool> s_addHandlerTruncated{false};
                const uint32_t logIndex = s_addHandlerLogCount.fetch_add(1u, std::memory_order_relaxed);
                if (ps2DiagLogBudget(std::cout,
                                     "[AddIntcHandler]",
                                     "PS2X_INTC_ADDHANDLER_MAX_LOGS",
                                     kMaxAddHandlerLogs,
                                     logIndex,
                                     s_addHandlerTruncated))
                {
                    auto flags = std::cout.flags();
                    std::cout << "[AddIntcHandler] cause=" << info.cause
                              << " handler=0x" << std::hex << info.handler
                              << " arg=0x" << info.arg
                              << " gp=0x" << info.gp
                              << " sp=0x" << info.sp
                              << std::dec
                              << " id=" << handlerId
                              << std::endl;
                    std::cout.flags(flags);
                }
            });
        }

        ensureInterruptWorkerRunning(rdram, runtime);
        setReturnS32(ctx, handlerId);
    }

    void AddIntcHandler2(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        AddIntcHandler(rdram, ctx, runtime);
    }

    void RemoveIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t cause = getRegU32(ctx, 4);
        const int handlerId = static_cast<int>(getRegU32(ctx, 5));
        if (handlerId > 0)
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            auto it = g_intcHandlers.find(handlerId);
            if (it != g_intcHandlers.end() && it->second.cause == cause)
            {
                g_intcHandlers.erase(it);
            }
        }
        setReturnS32(ctx, KE_OK);
    }

    void AddDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        IrqHandlerInfo info{};
        info.cause = getRegU32(ctx, 4);
        info.handler = getRegU32(ctx, 5);
        uint32_t next = getRegU32(ctx, 6);
        info.arg = getRegU32(ctx, 7);
        info.gp = getRegU32(ctx, 28);
        info.sp = getRegU32(ctx, 29);
        info.enabled = true;

        int handlerId = 0;
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            info.order = (next == 0) ? --g_dmac_head_order : ++g_dmac_tail_order;
            handlerId = g_nextDmacHandlerId++;
            info.id = handlerId;
            g_dmacHandlers[handlerId] = info;
        }
        setReturnS32(ctx, handlerId);
    }

    void AddDmacHandler2(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        AddDmacHandler(rdram, ctx, runtime);
    }

    void RemoveDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t cause = getRegU32(ctx, 4);
        const int handlerId = static_cast<int>(getRegU32(ctx, 5));
        if (handlerId > 0)
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            auto it = g_dmacHandlers.find(handlerId);
            if (it != g_dmacHandlers.end() && it->second.cause == cause)
            {
                g_dmacHandlers.erase(it);
            }
        }
        setReturnS32(ctx, KE_OK);
    }

    void EnableIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const int handlerId = static_cast<int>(getRegU32(ctx, 5));
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            if (auto it = g_intcHandlers.find(handlerId); it != g_intcHandlers.end())
            {
                it->second.enabled = true;
            }
        }
        setReturnS32(ctx, KE_OK);
    }

    void DisableIntcHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const int handlerId = static_cast<int>(getRegU32(ctx, 5));
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            if (auto it = g_intcHandlers.find(handlerId); it != g_intcHandlers.end())
            {
                it->second.enabled = false;
            }
        }
        setReturnS32(ctx, KE_OK);
    }

    void EnableDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const int handlerId = static_cast<int>(getRegU32(ctx, 5));
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            if (auto it = g_dmacHandlers.find(handlerId); it != g_dmacHandlers.end())
            {
                it->second.enabled = true;
            }
        }
        setReturnS32(ctx, KE_OK);
    }

    void DisableDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const int handlerId = static_cast<int>(getRegU32(ctx, 5));
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            if (auto it = g_dmacHandlers.find(handlerId); it != g_dmacHandlers.end())
            {
                it->second.enabled = false;
            }
        }
        setReturnS32(ctx, KE_OK);
    }

    void EnableDmac(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t cause = getRegU32(ctx, 4);
        if (cause < 32u)
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            g_enabled_dmac_mask |= (1u << cause);
        }
        setReturnS32(ctx, KE_OK);
    }

    void iEnableDmac(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        EnableDmac(rdram, ctx, runtime);
    }

    void DisableDmac(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t cause = getRegU32(ctx, 4);
        if (cause < 32u)
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            g_enabled_dmac_mask &= ~(1u << cause);
        }
        setReturnS32(ctx, KE_OK);
    }

    void iDisableDmac(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        DisableDmac(rdram, ctx, runtime);
    }
}
