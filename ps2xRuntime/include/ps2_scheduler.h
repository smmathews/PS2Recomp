#ifndef PS2_SCHEDULER_H
#define PS2_SCHEDULER_H

// ---------------------------------------------------------------------------
// ps2_scheduler.h — public API for the N=1 fiber cooperative scheduler.
//
// Exactly one dedicated host OS thread (g_guest_thread, the "guest executor")
// runs all guest fibers. This eliminates cross-thread swapcontext UB structurally.
//
// No <semaphore>, <thread>, <mutex>, or other heavy headers are included here;
// this keeps the header safe for Vita (VitaSDK lacks C++20 <semaphore>) and
// keeps compile times low.
// ---------------------------------------------------------------------------

#include <cstdint>

// Guest thread identity. -1 means this host thread is not currently running a fiber.
// Defined in ps2_scheduler.cpp; extern-declared here and in State.h.
extern thread_local int g_currentThreadId;

class PS2Runtime;

namespace ps2sched
{
    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    // Initialise global scheduler state and create the single guest executor
    // thread. Call once before create_fiber().
    void scheduler_init();

    // Signal all fibers to terminate, join all pool threads, and free all
    // fibers. Safe to call from the main OS thread after the render loop exits.
    void scheduler_shutdown();

    // Register a callback that scheduler_shutdown() invokes once, just before
    // joining the guest executor thread. The runtime sets this to request a
    // stop so dispatchLoop's while(!isStopRequested()) exits between dispatched
    // functions. Pass nullptr to clear. May be called before scheduler_init().
    void scheduler_set_stop_callback(void (*fn)());

    // -----------------------------------------------------------------------
    // Thread lifecycle (called from Thread.cpp / Lifecycle.cpp)
    // -----------------------------------------------------------------------

    // Create a fiber for guest thread `tid` with the given CPU registers and
    // enqueue it as Ready. May THROW std::bad_alloc / std::runtime_error on
    // stack or fiber allocation failure. StartThread catches the throw and reports allocation failure.
    void create_fiber(int tid, int priority, uint32_t entry,
                      uint32_t sp, uint32_t gp, uint32_t arg,
                      PS2Runtime* rt, uint8_t* rdram);

    // Set terminateRequested on tid's fiber and wake it if blocked.
    void request_terminate(int tid);

    // Cooperatively wait (yielding) until tid's fiber has finished.
    // Must be called from a fiber (not from a host thread).
    void join_fiber(int tid);

    // Reorder the fiber in the run queue with its new priority.
    void update_priority(int tid, int newPriority);

    // -----------------------------------------------------------------------
    // Blocking / readiness (Sync.cpp, Thread.cpp, Interrupt.cpp)
    // -----------------------------------------------------------------------

    // Arm the park BEFORE publishing to an object wait-list. Sets the running
    // fiber's scheduler state to Blocked and clears wake_pending under
    // g_sched_mutex, so a waker that fires after the wait-list publish but before
    // block_current() sees state==Blocked and routes through wake_locked (setting
    // wake_pending) instead of dropping the wakeup. Acts on the fiber currently
    // running on the executor thread (tls_current_fiber); callers must only invoke
    // it from a fiber. No-op if called with no current fiber.
    void arm_park();

    // Result of block_current(). See block_current() below.
    enum class BlockResult
    {
        Parked,        // a fiber actually parked and was later resumed by a waker.
        WokenInWindow, // a fiber: a wakeup arrived in the arm/publish window; do
                       //   not park. Caller re-checks its wait condition.
        NonFiberOwner, // a borrowed host worker that HOLDS the guest token: caller
                       //   must async_guest_end()/yield/async_guest_begin(), then
                       //   re-check.
        NonFiberNoTok  // a borrowed host worker that does NOT hold the token:
                       //   caller just yields the host thread and re-checks; it
                       //   must NOT touch the guest token.
    };

    // Park the currently-running fiber. Caller MUST have called arm_park() first
    // and then published itself to the object wait-list. The fiber must release any
    // object mutexes BEFORE calling this. See BlockResult for the four outcomes.
    BlockResult block_current();

    // Wake a blocked fiber from within another fiber (from a guest thread).
    // Gated on suspendCount == 0.
    void make_ready(int tid);

    // Wake a blocked fiber from a non-fiber host thread (IRQ worker, vsync
    // worker, alarm worker, RPC worker). Gated on suspendCount == 0.
    void enqueue_external_wakeup(int tid);

    // Wake a blocked fiber from a non-fiber host thread, but ONLY if the fiber
    // currently mapped to `tid` is still the exact fiber identified by `token`
    // (from current_fiber_token()). The token encodes the fiber's generation, so
    // a recycled tid whose new fiber has a different generation will not match
    // and the stale wakeup is dropped. Validation happens under g_sched_mutex.
    void enqueue_external_wakeup_validated(int tid, uint64_t token);

    // Opaque identity token for the fiber currently running on this thread, or 0
    // if not on a fiber. Encodes generation + tid. Only ever compared for
    // equality / passed back to enqueue_external_wakeup_validated.
    uint64_t current_fiber_token();

    // Yield if a higher-priority fiber is ready. Enqueues self Ready BEFORE
    // yielding, so the fiber is never 'Running but off-queue'.
    void maybe_yield();

    // Suspend the current fiber (SuspendThread on self). Increments suspendCount.
    void suspend_self();

    // Suspend another fiber (SuspendThread on other). Increments suspendCount.
    void suspend_other(int tid);

    // Decrement the fiber's suspendCount and wake it if it reaches zero.
    // Called by ResumeThread after decrementing ThreadInfo::suspendCount.
    void resume_fiber(int tid);

    // Force the fiber's suspendCount to 0 and wake it if Blocked. Called by
    // ResumeThread when the PS2-visible ThreadInfo::suspendCount reaches 0, so
    // nested SuspendThread calls resolve in a single resume.
    void clear_suspend(int tid);

    // Rotate the equal-priority group in the run queue (RotateThreadReadyQueue).
    // If the CALLER is at `priority` it goes to the tail of its own group and
    // this call YIELDS (the running fiber is the conceptual head of its ready
    // queue on real hardware), provided another fiber at that priority or
    // better is runnable. Must therefore only be called from a fiber context
    // where yielding is legal (i.e. from the RotateThreadReadyQueue syscall).
    void rotate_ready_queue(int priority);

    // -----------------------------------------------------------------------
    // Async guest-code borrow
    // (called by IRQ worker / alarm worker / RPC worker host threads)
    //
    // Use the AsyncGuestScope RAII guard in Runtime.h instead of calling these
    // directly. MUST NOT be called from a fiber.
    // -----------------------------------------------------------------------

    // Acquire the "guest token": blocks until no fiber is executing guest code.
    void async_guest_begin();

    // Release the guest token.
    void async_guest_end();

    // -----------------------------------------------------------------------
    // Back-edge hook — called by PS2Runtime::shouldPreemptGuestExecution()
    // -----------------------------------------------------------------------

    // Sampled every 128 back-edges. Checks terminateRequested, suspendCount,
    // and priority; may ps2fiber_yield internally. Always returns false.
    bool yield_point();

    // Diagnostic: number of host workers currently blocked in async_guest_begin().
    int host_token_waiters();

    // True iff the calling OS thread is the single guest executor thread (the
    // thread that runs all guest fibers). Use this to decide whether a shared
    // dispatch helper (e.g. one invoked BOTH by a host worker via AsyncGuestScope
    // AND, in some call paths, synchronously from already-running guest code)
    // needs to borrow the guest token at all: a call already running on the
    // guest thread already owns the execution slot implicitly and must NOT
    // attempt async_guest_begin() (it aborts by design -- see async_guest_begin).
    bool is_guest_thread();

    // Diagnostic: log tid/priority/state/pc/ra for every known fiber (Fresh,
    // Ready, Running, Blocked, Exiting, Finished), one line per fiber, to
    // stdout. Safe to call from the host frame-loop thread (takes
    // g_sched_mutex briefly). Intended as a low-noise, permanent one-shot
    // diagnostic (e.g. triggered once ~30s into a boot) to answer "where is
    // each guest thread parked" without needing per-callsite instrumentation.
    void dump_all_fibers(const char *reasonTag);

} // namespace ps2sched

#endif // PS2_SCHEDULER_H
