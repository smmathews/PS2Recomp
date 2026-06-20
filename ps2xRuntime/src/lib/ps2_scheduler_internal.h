#ifndef PS2_SCHEDULER_INTERNAL_H
#define PS2_SCHEDULER_INTERNAL_H

// ---------------------------------------------------------------------------
// ps2_scheduler_internal.h — private implementation types for the
// N=1 fiber cooperative scheduler.
//
// NEVER include this from a public header. Include from ps2_scheduler.cpp
// and Kernel/Syscall files ONLY.
// ---------------------------------------------------------------------------

#include "ps2_scheduler.h" // public API (extern g_currentThreadId)
#include "ps2_fiber.h"     // PS2Fiber* (NOT libaco)
#include <ps2_runtime.h>   // R5900Context, PS2Runtime

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <cstdint>

// ---------------------------------------------------------------------------
// Per-fiber state
// ---------------------------------------------------------------------------

struct FiberContext
{
    int tid      = 0;
    int priority = 1; // lower number = higher priority (PS2 EE convention)

    // Monotonic per-process identity counter. Assigned once at construction.
    // Wait-lists that must survive tid recycling (vsync) store {tid, generation}
    // and validate the generation on wakeup, so a recycled tid whose new fiber
    // has a different generation cannot receive a stale wakeup. Pointer identity
    // is NOT used (FiberContext addresses can be reused by the allocator).
    uint32_t generation = 0;

    // Per-fiber saved CPU state. dispatchLoop() reads/writes this in-place.
    R5900Context cpu;

    // Fiber handle — owns the stack; freed by guest_executor_main on Finished.
    PS2Fiber* fiber = nullptr;

    // Scheduling state machine
    enum class State
    {
        Fresh,    // created, never resumed
        Ready,    // in g_run_queue
        Running,  // currently executing on g_guest_thread
        Blocked,  // parked (sema/event/sleep/suspend/vsync)
        Exiting,  // in fiber_trampoline exit hook; may yield, do NOT free/re-enqueue-as-Finished
        Finished  // fn returned / ThreadExitException caught
    };
    State state = State::Fresh;

    // Intrusive singly-linked list pointer (used by the run queue).
    FiberContext* next = nullptr;

    // Guards against duplicate run-queue insertion.
    bool in_run_queue = false;

    // Invariant: a waker that fires while this fiber is mid-park records the
    // wakeup here instead of enqueuing, because the fiber's `next` pointer is
    // still owned by the executing fiber until ps2fiber_yield returns. Set true
    // by a waker (make_ready /
    // enqueue_external_wakeup / resume_fiber) that fires while this fiber is
    // mid-park — i.e. it has set state=Blocked but ps2fiber_yield has not yet
    // returned control to the executor (detected via g_running_fiber == fc).
    // In that window the waker MUST NOT call enqueue_locked (fc->next is owned
    // by the still-executing fiber). Instead it sets wake_pending; the executor
    // re-enqueues the fiber in its post-resume code. arm_park() does NOT clear
    // this (a wakeup recorded between wait-list publish and arm_park must
    // survive); block_current() consumes it. Only ever read/written
    // under g_sched_mutex, so a plain bool is sufficient.
    bool wake_pending = false;

    // Set by request_terminate() (under g_sched_mutex). Read by yield_point on
    // the fiber's own execution (relaxed — executor is the only reader while the
    // fiber runs; writers wake it).
    std::atomic<bool> terminateRequested{false};

    // > 0 means the fiber must stay off the run queue (suspended). Set by
    // suspend_self/suspend_other; cleared by resume_fiber. All wakeup paths gate.
    std::atomic<int> suspendCount{0};

    // Set true by fiber_trampoline after the exit hook runs. Read by join_fiber.
    std::atomic<bool> finished{false};

    // Join priority-floor state. While this fiber is blocked in join_fiber
    // waiting on a lower-priority target, its scheduling priority is temporarily
    // lowered below the target so the target can run. joinFloorActive marks that
    // the lowering is in effect; joinSavedPriority holds the priority to restore on
    // join exit. ChangeThreadPriority (update_priority) writes joinSavedPriority
    // instead of fc->priority while the floor is active, so a concurrent
    // reprioritize is preserved across the join. All read/written under
    // g_sched_mutex.
    bool joinFloorActive = false;
    int  joinSavedPriority = 0;

    // Trampoline reads rt/rdram from here instead of smuggling them through guest registers.
    PS2Runtime* rt    = nullptr;
    uint8_t*    rdram = nullptr;

    // Free the owned fiber on destruction. The normal scheduler paths
    // (guest_executor_main Finished/dead branches, scheduler_shutdown) null out
    // `fiber` before the FiberContext is destroyed, so this destructor is a safety
    // net for early-exit paths in create_fiber: if the unique_ptr is
    // destroyed after ps2fiber_alloc but before the fiber is handed to the
    // scheduler, the stack/guard-page mapping is reclaimed instead of leaked.
    // ps2fiber_free is a no-op on nullptr.
    ~FiberContext()
    {
        if (fiber != nullptr)
        {
            ps2fiber_free(fiber);
            fiber = nullptr;
        }
    }

    // FiberContext owns a raw fiber; it must not be copied or moved (the map stores
    // it via unique_ptr and never copies it).
    FiberContext()
    {
        static std::atomic<uint32_t> s_nextGeneration{1};
        generation = s_nextGeneration.fetch_add(1, std::memory_order_relaxed);
    }
    FiberContext(const FiberContext&)            = delete;
    FiberContext& operator=(const FiberContext&) = delete;
};

// ---------------------------------------------------------------------------
// Globals (defined in ps2_scheduler.cpp; extern-declared here)
// ---------------------------------------------------------------------------

// The single global scheduler mutex. All queue operations, state transitions,
// and condition-variable waits are guarded by this.
extern std::mutex              g_sched_mutex;
extern std::condition_variable g_sched_cv;

// Intrusive singly-linked list; head is the highest-priority ready fiber.
// Access only under g_sched_mutex.
extern FiberContext* g_run_queue;

// The fiber currently executing guest code on g_guest_thread (or nullptr).
// Written under g_sched_mutex before/after ps2fiber_resume().
extern FiberContext* g_running_fiber;

// True while a host (non-fiber) thread holds the "guest token" for running
// handler code (INTC/DMAC/alarm/RPC). Serializes with g_running_fiber.
extern bool g_guest_token_held_by_host;

// Set to true during scheduler_shutdown to stop the executor loop.
extern std::atomic<bool> g_stop;

// All fiber contexts, keyed by tid.
extern std::unordered_map<int, std::unique_ptr<FiberContext>> g_fiber_map;

// The single guest executor thread (replaces old pool vector).
extern std::thread g_guest_thread;

// ---------------------------------------------------------------------------
// Thread-locals (defined in ps2_scheduler.cpp)
// ---------------------------------------------------------------------------

// Pointer to the FiberContext the executor is currently running.
// NULL when not on g_guest_thread or between fibers.
extern thread_local FiberContext* tls_current_fiber;

// True only on g_guest_thread.
extern thread_local bool tls_is_guest_thread;

// Back-edge counter sampled by yield_point().
extern thread_local uint32_t tls_backedge_counter;

// ---------------------------------------------------------------------------
// Intrusive priority-sorted queue helpers (callers must hold g_sched_mutex)
// ---------------------------------------------------------------------------

// Insert fc into the run queue in priority order (stable FIFO within same priority).
// No-op if fc->in_run_queue is true.
inline void enqueue_locked(FiberContext* fc)
{
    if (fc->in_run_queue) return; // idempotent
    fc->in_run_queue = true;
    fc->state = FiberContext::State::Ready;
    FiberContext** pp = &g_run_queue;
    // Lower priority number = higher priority. Stable: insert after existing same-prio nodes.
    while (*pp && (*pp)->priority <= fc->priority)
        pp = &(*pp)->next;
    fc->next = *pp;
    *pp = fc;
}

// Remove fc from the run queue. No-op if not present.
inline void remove_locked(FiberContext* fc)
{
    if (!fc->in_run_queue) return;
    FiberContext** pp = &g_run_queue;
    while (*pp && *pp != fc)
        pp = &(*pp)->next;
    if (*pp)
    {
        *pp = fc->next;
        fc->next = nullptr;
        fc->in_run_queue = false;
    }
}

// Pop and return the head of the run queue (highest priority), or nullptr.
inline FiberContext* pop_head_locked()
{
    FiberContext* fc = g_run_queue;
    if (!fc) return nullptr;
    g_run_queue = fc->next;
    fc->next = nullptr;
    fc->in_run_queue = false;
    return fc;
}

// Return the FiberContext for tid, or nullptr if not found.
// Caller should hold g_sched_mutex or ensure no concurrent modifications.
inline FiberContext* fiber_for(int tid)
{
    auto it = g_fiber_map.find(tid);
    return (it != g_fiber_map.end()) ? it->second.get() : nullptr;
}

// ---------------------------------------------------------------------------
// Fiber exit hook — called by fiber_trampoline after dispatchLoop returns.
// Set once by Thread.cpp during static initialization.
// ---------------------------------------------------------------------------
extern void (*g_fiber_exit_hook)(int tid, uint8_t* rdram, R5900Context* ctx, PS2Runtime* rt);

#endif // PS2_SCHEDULER_INTERNAL_H
