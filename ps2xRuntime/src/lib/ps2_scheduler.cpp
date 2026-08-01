// ps2_scheduler.cpp — N=1 fiber cooperative scheduler for PS2Recomp.
//
// Architecture:
//   - Exactly ONE dedicated host OS thread (g_guest_thread, the "guest executor")
//     runs all guest fibers. This eliminates cross-thread swapcontext UB
//     structurally — a ucontext_t is only ever saved and resumed on the one thread.
//   - Guest threads are mapped to FiberContext objects; each owns a PS2Fiber
//     (ucontext_t on POSIX with guard page; joinable pthread on Vita).
//   - Host threads (IRQ worker, alarm worker, RPC worker) that need to run guest
//     code acquire the guest token via AsyncGuestScope (RAII).
//   - Cooperative yield points sampled every 128 back-edges via yield_point().

#include "ps2_scheduler_internal.h"
#include "ps2_fiber.h"
#include <ps2_runtime.h>
#include <ps2_runtime_macros.h>
#include "Kernel/Syscalls/Helpers/ThreadExit.h" // canonical ThreadExitException
#include "Kernel/Syscalls/Interrupt.h" // stopInterruptWorker
#include "Kernel/Syscalls/Sync.h"      // stopAlarmWorker

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>
#include <tuple>
#include <vector>

// ---------------------------------------------------------------------------
// g_currentThreadId — -1 means this host thread is not currently running a fiber.
// ---------------------------------------------------------------------------
thread_local int g_currentThreadId = -1;

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
std::mutex              g_sched_mutex;
std::condition_variable g_sched_cv;
FiberContext*           g_run_queue               = nullptr;
FiberContext*           g_running_fiber            = nullptr;
bool                    g_guest_token_held_by_host = false;
// True on a host (non-fiber) worker thread that currently holds the guest token
// (acquired via async_guest_begin, released via async_guest_end). Used to assert
// the token is released by the same thread that acquired it, and so block_current
// can report whether the calling non-fiber worker owns the token. A thread_local
// bool replaces the (typically lock-based) std::atomic<std::thread::id> owner id.
static thread_local bool tls_holds_guest_token = false;
std::atomic<bool>       g_stop{false};
static std::atomic<int> g_host_token_waiters{0}; // host workers blocked in async_guest_begin()

std::unordered_map<int, std::unique_ptr<FiberContext>> g_fiber_map;
std::thread g_guest_thread;

// True only on the single guest executor thread (g_guest_thread). Set once at the
// top of guest_executor_main, before any fiber can be resumed. Read by
// ps2fiber_on_executor_thread(). A thread_local bool is lock-free and avoids the
// (typically lock-based) std::atomic<std::thread::id> on the context-switch hot
// path. Always correct: only the executor thread ever runs guest_executor_main.
static thread_local bool tls_is_executor_thread = false;

// ---------------------------------------------------------------------------
// Thread-locals
// ---------------------------------------------------------------------------
thread_local FiberContext* tls_current_fiber   = nullptr;
thread_local bool          tls_is_guest_thread = false;
thread_local uint32_t      tls_backedge_counter = 0;

// ---------------------------------------------------------------------------
// Fiber exit hook (set once by Thread.cpp)
// ---------------------------------------------------------------------------
void (*g_fiber_exit_hook)(int tid, uint8_t* rdram, R5900Context* ctx, PS2Runtime* rt) = nullptr;

// ---------------------------------------------------------------------------
// Stop callback — set by scheduler_set_stop_callback(); invoked by
// scheduler_shutdown() before joining g_guest_thread. Read/written under
// g_sched_mutex. nullptr-safe.
// ---------------------------------------------------------------------------
static void (*g_request_runtime_stop_fn)() = nullptr;

// ---------------------------------------------------------------------------
// Stack size per fiber
// ---------------------------------------------------------------------------
static constexpr size_t kFiberStackBytes = 1024 * 1024; // 1 MiB

// ---------------------------------------------------------------------------
// Fatal helper — terminate on invariant violation
// ---------------------------------------------------------------------------
#define SCHED_REQUIRE(cond, msg)                                               \
    do                                                                         \
    {                                                                          \
        if (!(cond))                                                           \
        {                                                                      \
            std::fprintf(stderr, "FATAL [ps2sched]: " msg "\n");              \
            std::terminate();                                                  \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// fiber_trampoline — entry point of every PS2Fiber.
//
// Runs on g_guest_thread (the guest executor). Catches ALL exceptions so the
// executor can continue scheduling other fibers after this one exits.
// ---------------------------------------------------------------------------
static void fiber_trampoline(void* arg)
{
    FiberContext* fc = static_cast<FiberContext*>(arg);
    SCHED_REQUIRE(fc != nullptr, "fiber_trampoline null arg");
    // guest_executor_main set tls_current_fiber on the executor thread before
    // resuming us (the N=1 / ucontext design runs all guest code on that one
    // thread).
    SCHED_REQUIRE(fc == tls_current_fiber, "fiber_trampoline fc mismatch");

    PS2Runtime* rt    = fc->rt;    // from struct, not smuggled GPRs
    uint8_t*    rdram = fc->rdram;

    try
    {
        rt->dispatchLoop(rdram, &fc->cpu);
    }
    catch (const ThreadExitException&)
    {
        // Cooperative termination — fall through.
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "[ps2sched] fiber tid=%d uncaught std::exception: %s\n",
                     fc->tid, e.what());
    }
    catch (...)
    {
        std::fprintf(stderr, "[ps2sched] fiber tid=%d uncaught unknown exception\n", fc->tid);
    }

    // Mark Exiting BEFORE running the exit hook. If a guest exit handler
    // makes a blocking syscall (block_current -> ps2fiber_yield), the executor's
    // post-resume code sees Exiting and RE-ENQUEUES the fiber (treated like a
    // cooperative yield) instead of freeing it — the trampoline still has a live
    // stack frame here. When the wakeup arrives the fiber resumes inside the
    // hook and continues. Only after the hook returns do we transition to
    // Finished, which is the executor's signal to free the fiber.
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        fc->state = FiberContext::State::Exiting;
    }

    if (g_fiber_exit_hook)
    {
        // The exit hook runs guest exit handlers (arbitrary recompiled code). If
        // one calls ExitThread it throws ThreadExitException; any such throw must
        // NOT cross the swapcontext boundary below (UB). Swallow it: the fiber is
        // already exiting.
        try
        {
            g_fiber_exit_hook(fc->tid, rdram, &fc->cpu, rt);
        }
        catch (const ThreadExitException&)
        {
            // Guest exit handler called ExitThread — normal exit completion.
        }
        catch (const std::exception& e)
        {
            std::fprintf(stderr,
                         "[ps2sched] fiber tid=%d exit hook threw std::exception: %s\n",
                         fc->tid, e.what());
        }
        catch (...)
        {
            std::fprintf(stderr,
                         "[ps2sched] fiber tid=%d exit hook threw unknown exception\n",
                         fc->tid);
        }
    }

    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        fc->state = FiberContext::State::Finished;
    }
    fc->finished.store(true, std::memory_order_release);
    g_sched_cv.notify_all(); // wake join_fiber waiters / executor

    ps2fiber_yield(); // back to guest_executor_main; never returns
    SCHED_REQUIRE(false, "fiber_trampoline resumed after Finished");
}

// ---------------------------------------------------------------------------
// guest_executor_main — the N=1 loop
// ---------------------------------------------------------------------------
static void guest_executor_main()
{
    // Publish our identity BEFORE anything can resume a fiber. ps2fiber_resume
    // (called below) asserts it runs on this thread via
    // ps2fiber_on_executor_thread(), which reads this TLS flag.
    tls_is_executor_thread = true;

    tls_is_guest_thread = true;

    std::unique_lock<std::mutex> lk(g_sched_mutex);
    // The wait predicate MUST be exactly (canQuit || canRun): an earlier
    // version short-circuited on bare g_stop, so with g_stop set while
    // (run_queue != null && waiters > 0) the predicate was instantly true but
    // neither the break nor the pop condition held -- the loop continue'd,
    // cv.wait saw the predicate already true and returned WITHOUT releasing
    // g_sched_mutex, and the executor livelocked (spinning under the lock,
    // also starving the very waiters it was gated on).
    auto canQuit = []
    {
        return g_stop && g_run_queue == nullptr;
    };
    auto canRun = []
    {
        return g_run_queue != nullptr &&
               g_running_fiber == nullptr &&
               !g_guest_token_held_by_host &&
               g_host_token_waiters.load(std::memory_order_relaxed) == 0;
    };
    while (true)
    {
        g_sched_cv.wait(lk, [&]
        {
            return canQuit() || canRun();
        });

        if (canQuit()) break;
        if (!canRun()) continue; // spurious wake; predicate re-checked under lock
        // NOTE on g_host_token_waiters gating: async_guest_begin()'s wakeup
        // predicate is (g_running_fiber == nullptr && !token_held), but the only
        // window where g_running_fiber is null is while THIS loop holds
        // g_sched_mutex between resumes. Without this gate the executor's own
        // cv.wait predicate is instantly true again (fiber re-enqueued by the
        // post-resume code), so it re-resumes the fiber without ever sleeping,
        // and a parked host worker (interrupt worker) can never win the token:
        // guest spins waiting for VBlank ticks the starved worker can't deliver.
        // Gating on waiters == 0 makes the executor genuinely sleep (releasing
        // the mutex) until the host worker takes and releases the token. This is
        // the executor-side half of the handoff whose fiber-side half is
        // yield_point() step 4 (yield when a host worker is parked).

        FiberContext* fc = pop_head_locked();
        SCHED_REQUIRE(fc != nullptr, "executor popped null with non-empty predicate");

        fc->state = FiberContext::State::Running;
        g_running_fiber = fc;
        // Guest code runs on THIS (executor) thread, so publish the guest
        // identity here before resuming.
        tls_current_fiber = fc;
        g_currentThreadId = fc->tid; // set BEFORE resume

        lk.unlock();
        ps2fiber_resume(fc->fiber); // runs until yield or finish
        lk.lock();

        tls_current_fiber = nullptr;
        g_currentThreadId = -1;           // -1 between fibers
        g_running_fiber = nullptr;        // cleared here, after resume returns.

        if (fc->state == FiberContext::State::Running ||
            fc->state == FiberContext::State::Exiting)
        {
            // Running: cooperative yield (maybe_yield / yield_point). The fiber
            //   normally enqueues itself before yielding; enqueue_locked is
            //   idempotent so this is a safe backstop.
            // Exiting: the exit hook yielded (e.g. a blocking syscall in a
            //   guest exit handler). It is NOT done yet — re-enqueue so it runs
            //   to completion. Do NOT free: fiber_trampoline still has a live
            //   stack frame.
            enqueue_locked(fc);
        }
        else if (fc->state == FiberContext::State::Blocked)
        {
            // A waker fired during the park window (between the fiber
            // setting state=Blocked and ps2fiber_yield returning here). It could
            // not enqueue safely (fc->next was owned by the running fiber), so it
            // set wake_pending instead. Honour it now — UNLESS the fiber was
            // suspended during the park window (a suspendCount > 0 means it
            // must stay off the run queue). The matching resume_fiber/clear_suspend
            // will wake it when the suspend is lifted.
            if (fc->wake_pending &&
                fc->suspendCount.load(std::memory_order_relaxed) == 0)
            {
                fc->wake_pending = false;
                enqueue_locked(fc); // sets state=Ready
            }
            // else: genuinely Blocked (or suspended); a future waker / resume
            // will enqueue it. Leave wake_pending as-is so a later resume can
            // honour it.
        }
        else if (fc->state == FiberContext::State::Finished)
        {
            // Erase the map entry while the lock is held so that a borrowed worker
            // racing in async_guest_begin cannot call StartThread on this tid and
            // insert a new FiberContext entry that we would then unconditionally clobber
            // after dropping and re-acquiring the lock. Moving out the unique_ptr
            // under the lock prevents the window between unlock and re-lock from
            // allowing tid reuse that would be silently destroyed.
            int tid = fc->tid;
            std::unique_ptr<FiberContext> dead;
            auto it = g_fiber_map.find(tid);
            if (it != g_fiber_map.end())
                dead = std::move(it->second);
            g_fiber_map.erase(tid);        // tid is now free for reuse under the lock
            PS2Fiber* deadFiber = dead ? dead->fiber : nullptr;
            if (dead) dead->fiber = nullptr; // suppress destructor double-free
            lk.unlock();
            ps2fiber_free(deadFiber);      // munmap outside the lock from a local
            // dead destructs here (FiberContext body) — fiber already freed above
            lk.lock();                     // re-acquire before next wait or break
        }
        // else Fresh: impossible after a resume.

        g_sched_cv.notify_all();
    }
    lk.unlock();
}

// ---------------------------------------------------------------------------
// wake_locked — race-safe wakeup. MUST be called with g_sched_mutex held and
// only when fc is genuinely runnable (Blocked + suspendCount gate satisfied by
// the caller). If fc is mid-park (set state=Blocked but ps2fiber_yield has not
// yet returned to the executor), g_running_fiber still points at fc: in that
// window we MUST NOT touch fc->next, so we set wake_pending and let the
// executor's post-resume code re-enqueue. Otherwise enqueue normally.
// ---------------------------------------------------------------------------
static void wake_locked(FiberContext* fc)
{
    // Called under g_sched_mutex. Idempotent: if already queued, nothing to do.
    if (fc->in_run_queue)
    {
        return;
    }

    if (g_running_fiber == fc)
    {
        // The fiber is still the running fiber. This covers two windows:
        //   1. mid-park: state==Blocked set (by arm_park or block_current) but
        //      ps2fiber_yield has not yet returned control to the executor.
        //   2. state==Blocked armed via arm_park(), the syscall published to
        //      an object wait-list and a waker fired before block_current().
        // In both cases fc->next is owned by the still-executing fiber, so we
        // MUST NOT enqueue_locked. Record the wakeup; block_current() (case 2)
        // or the executor post-resume code (case 1) honours wake_pending.
        fc->wake_pending = true;
    }
    else
    {
        // Fully parked (state==Blocked, not the running fiber): enqueue directly.
        enqueue_locked(fc);
    }
}

// ---------------------------------------------------------------------------
// scheduler_init
// ---------------------------------------------------------------------------
void ps2sched::scheduler_init()
{
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        g_run_queue                = nullptr;
        g_running_fiber            = nullptr;
        g_guest_token_held_by_host = false;
        g_stop.store(false, std::memory_order_relaxed);
        g_fiber_map.clear();
    }
    SCHED_REQUIRE(!g_guest_thread.joinable(), "scheduler_init while executor running");
    // The executor thread publishes tls_is_executor_thread itself at the top of
    // guest_executor_main, before any fiber resume.
    g_guest_thread = std::thread(guest_executor_main);
}

// ---------------------------------------------------------------------------
// scheduler_set_stop_callback
// ---------------------------------------------------------------------------
void ps2sched::scheduler_set_stop_callback(void (*fn)())
{
    std::lock_guard<std::mutex> lk(g_sched_mutex);
    g_request_runtime_stop_fn = fn;
}

// ---------------------------------------------------------------------------
// scheduler_shutdown
// ---------------------------------------------------------------------------
void ps2sched::scheduler_shutdown()
{
    // Stop and JOIN the host workers (IRQ/vsync + alarm) FIRST, before we
    // tear down the fiber map. Those workers call enqueue_external_wakeup() into
    // g_fiber_map; joining them here guarantees no such call can race the
    // teardown below, regardless of whether notifyRuntimeStop() already ran.
    // Both stop functions are idempotent (no-op if already stopped/joined).
    ps2_syscalls::stopInterruptWorker();
    ps2_syscalls::stopAlarmWorker();

    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        g_stop.store(true, std::memory_order_release);
        // Re-terminate every fiber so blocked ones wake and unwind.
        for (auto& [tid, fc] : g_fiber_map)
        {
            if (fc->state == FiberContext::State::Blocked)
            {
                fc->terminateRequested.store(true, std::memory_order_relaxed);
                fc->suspendCount.store(0, std::memory_order_relaxed);
                wake_locked(fc.get());
            }
            else if (fc->state == FiberContext::State::Exiting)
            {
                // Exiting fibers yielded inside their exit hook (e.g. a blocking
                // syscall in a guest exit handler). They are off-queue but still
                // alive with a live trampoline stack frame. Do NOT set
                // terminateRequested (the exit path is already underway and an
                // unwind here would abandon the hook); just re-enqueue so the
                // executor runs them to completion before the quit predicate
                // (g_stop && run_queue==nullptr) is satisfied.
                fc->suspendCount.store(0, std::memory_order_relaxed);
                wake_locked(fc.get());
            }
            else
            {
                // Fresh / Ready / Running / Finished: leave terminateRequested set
                // so a still-running fiber unwinds at its next yield_point.
                fc->terminateRequested.store(true, std::memory_order_relaxed);
            }
        }
    }
    g_sched_cv.notify_all();

    // Request the runtime stop so dispatchLoop's while(!isStopRequested()) exits
    // between dispatched functions. Invoked OUTSIDE g_sched_mutex: the callback
    // runs runtime.requestStopFlagOnly(), which sets its own atomic and must not
    // nest under g_sched_mutex.
    void (*stopFn)() = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        stopFn = g_request_runtime_stop_fn;
    }
    if (stopFn) stopFn();

    if (g_guest_thread.joinable()) g_guest_thread.join();

    // Executor has exited. No guest code runs now. Free any remaining fibers.
    std::lock_guard<std::mutex> lk(g_sched_mutex);
    for (auto& [tid, fc] : g_fiber_map)
    {
        if (fc->fiber)
        {
            ps2fiber_free(fc->fiber);
            fc->fiber = nullptr;
        }
    }
    g_fiber_map.clear();
    g_run_queue = nullptr;
}

// ---------------------------------------------------------------------------
// create_fiber — initialises CPU state and enqueues the fiber Ready; throws on failure.
// ---------------------------------------------------------------------------
void ps2sched::create_fiber(int tid, int priority, uint32_t entry,
                            uint32_t sp, uint32_t gp, uint32_t arg,
                            PS2Runtime* rt, uint8_t* rdram)
{
    // Refuse to create fibers once shutdown has begun. A fiber running its
    // terminate path can still reach StartThread -> create_fiber; if we allowed
    // it, the executor could partially run the new fiber during teardown and
    // leak its stack if it parks. Throwing here reuses StartThread's existing
    // allocation-failure path (it resets the thread to dormant and returns
    // KE_NO_MEMORY).
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        if (g_stop)
        {
            throw std::runtime_error("[ps2sched] create_fiber refused: scheduler stopping");
        }
    }

    // The R5900Context constructor zeroes all registers and sets the documented
    // reset defaults (cop0_random=47, vu0_q=1.0, vu0_vf[0]=(0,0,0,1) -- VF0 is
    // hardwired on real VU hardware and every guest fiber needs it pinned or
    // VU0-macro-mode skinning math silently collapses, see
    // dq8/reference/dc2-learnings/04-vu-interpreter-correctness.md).
    auto fc = std::make_unique<FiberContext>();
    fc->tid      = tid;
    fc->priority = priority;
    fc->rt       = rt;
    fc->rdram    = rdram;

    fc->cpu.pc = entry;
    SET_GPR_U32(&fc->cpu, 29, sp);  // $sp
    SET_GPR_U32(&fc->cpu, 28, gp);  // $gp
    SET_GPR_U32(&fc->cpu, 4,  arg); // $a0 (passed directly, not smuggled through other registers)
    SET_GPR_U32(&fc->cpu, 31, 0u);  // $ra = 0 (returns -> pc==0 -> dispatchLoop break)

    PS2Fiber* fiber = ps2fiber_alloc(fiber_trampoline, fc.get(), kFiberStackBytes);
    if (!fiber)
    {
        throw std::runtime_error("[ps2sched] fiber allocation failed");
    }
    fc->fiber = fiber;
    fc->state = FiberContext::State::Fresh;

    FiberContext* raw = fc.get();
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        g_fiber_map[tid] = std::move(fc);
        enqueue_locked(raw); // Fresh -> Ready
    }
    g_sched_cv.notify_all();
}

// ---------------------------------------------------------------------------
// request_terminate
// ---------------------------------------------------------------------------
void ps2sched::request_terminate(int tid)
{
    std::lock_guard<std::mutex> lk(g_sched_mutex);
    FiberContext* fc = fiber_for(tid);
    if (!fc) return;
    fc->terminateRequested.store(true, std::memory_order_relaxed);
    if (fc->state == FiberContext::State::Blocked)
    {
        fc->suspendCount.store(0, std::memory_order_relaxed); // so it can run to observe flag
        wake_locked(fc);
        g_sched_cv.notify_all();
    }
}

// ---------------------------------------------------------------------------
// join_fiber
// ---------------------------------------------------------------------------
void ps2sched::join_fiber(int tid)
{
    // Called from a fiber (TerminateThread of another tid). Cooperative: yield
    // until the target is Finished or gone from the map.
    SCHED_REQUIRE(ps2fiber_current() != nullptr, "join_fiber from non-fiber thread");

    FiberContext* self = tls_current_fiber;

    while (true)
    {
        bool done = false;
        {
            std::lock_guard<std::mutex> lk(g_sched_mutex);
            FiberContext* t = fiber_for(tid);
            done = (!t || t->finished.load(std::memory_order_acquire));
            if (!done && self && t)
            {
                // Per-iteration floor: ensure the target gets CPU before self
                // polls again. EQUAL priority (not target+1): the run queue is
                // stable FIFO within a priority level, so when self re-enqueues
                // on yield it lands BEHIND the queued target — the target still
                // runs first. A floor of target+1 deadlocked by starvation
                // (DQ8 M0, PS2_PROJECT_STATE §3.20): after the target fiber
                // died, ANOTHER runnable fiber at the target's priority (a
                // per-frame-woken worker) kept the queue non-empty at that
                // level forever, so the joiner — parked one level BELOW — never
                // ran again, never observed done, and never restored its
                // priority: TerminateThread hung forever. The target's priority
                // may change between iterations, so re-apply each pass.
                const int floor = t->priority;
                if (self->priority < floor)
                {
                    if (!self->joinFloorActive)
                    {
                        // First lowering: remember the real priority to restore.
                        self->joinFloorActive = true;
                        self->joinSavedPriority = self->priority;
                    }
                    // self is the running fiber here, so it is not in the queue;
                    // just set the field for the next enqueue.
                    self->priority = floor;
                }
            }
        }
        if (done) break;
        ps2fiber_yield(); // let the executor run the target; it will finish
    }

    // Restore the joiner's real priority. joinSavedPriority reflects the latest
    // value written by any concurrent ChangeThreadPriority (update_priority keeps
    // it current while the floor is active), so we never lose a reprioritize.
    if (self)
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        if (self->joinFloorActive)
        {
            const int restore = self->joinSavedPriority;
            self->joinFloorActive = false;
            if (self->priority != restore)
            {
                bool wasQueued = self->in_run_queue;
                if (wasQueued) remove_locked(self);
                self->priority = restore;
                if (wasQueued) enqueue_locked(self);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// update_priority
// ---------------------------------------------------------------------------
void ps2sched::update_priority(int tid, int newPriority)
{
    std::lock_guard<std::mutex> lk(g_sched_mutex);
    FiberContext* fc = fiber_for(tid);
    if (!fc) return;

    // If this fiber is currently inside join_fiber with a temporary priority floor
    // in effect, the game-requested priority must be recorded as the value to
    // restore on join exit, NOT written over the temporary floor. Otherwise either
    // join_fiber's restore would lose this change, or this write would let the
    // joiner outrun the target and spin the join.
    if (fc->joinFloorActive)
    {
        fc->joinSavedPriority = newPriority;
        return;
    }

    if (fc->priority == newPriority) return;
    bool wasQueued = fc->in_run_queue;
    if (wasQueued) remove_locked(fc);
    fc->priority = newPriority;
    if (wasQueued) enqueue_locked(fc);
    // If the target is the running fiber, its new priority takes effect at the next
    // yield_point (ChangeThreadPriority calls maybe_yield right after).
}

// ---------------------------------------------------------------------------
// arm_park — set state=Blocked and leaves any pending wakeup recorded by a
// waker intact (block_current consumes it). After this returns, the gates in
// make_ready / enqueue_external_wakeup see state==Blocked and route through
// wake_locked, which (because the fiber is still g_running_fiber) records the
// wakeup in wake_pending. block_current() then observes wake_pending and does
// not park, so no wakeup is lost in the publish window.
// ---------------------------------------------------------------------------
void ps2sched::arm_park()
{
    FiberContext* fc = tls_current_fiber;
    if (fc == nullptr)
    {
        // Borrowed host worker. There is no fiber to arm; block_current() returns
        // a non-fiber result from this same thread. No-op here. (Callers now gate
        // this call on ps2fiber_current() != nullptr, so reaching here from a
        // non-fiber should not happen, but stay defensive.)
        return;
    }
    std::lock_guard<std::mutex> lk(g_sched_mutex);
    // Do NOT clear wake_pending here. arm_park may run AFTER the fiber has been
    // published to an object wait-list (see WaitSema), so a wakeup recorded by a
    // waker between publish and arm_park must survive. block_current() consumes
    // wake_pending; if none is pending it confirms Blocked.
    fc->state = FiberContext::State::Blocked;
}

// ---------------------------------------------------------------------------
// block_current — honour a wakeup that arrived in the arm/publish window.
// Returns a BlockResult enum describing the four possible outcomes.
// ---------------------------------------------------------------------------
ps2sched::BlockResult ps2sched::block_current()
{
    FiberContext* fc = tls_current_fiber;
    if (fc == nullptr)
    {
        // A borrowed IRQ/alarm/RPC worker (running guest code under
        // AsyncGuestScope) called a blocking syscall. We cannot park a host
        // worker on the fiber scheduler. Report whether this worker actually holds
        // the guest token so the caller only drops/reacquires a token it owns.
        bool ownsToken;
        {
            std::lock_guard<std::mutex> lk(g_sched_mutex);
            ownsToken = g_guest_token_held_by_host && tls_holds_guest_token;
        }
        static std::atomic<uint32_t> s_nonFiberBlockLogs{0};
        if (s_nonFiberBlockLogs.fetch_add(1, std::memory_order_relaxed) < 32u)
        {
            std::fprintf(stderr,
                         "[ps2sched] WARNING: block_current() called from non-fiber "
                         "thread (tid=%d); not parking\n",
                         g_currentThreadId);
        }
        return ownsToken ? BlockResult::NonFiberOwner
                         : BlockResult::NonFiberNoTok;
    }
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        // Shutdown safety: a terminate-requested fiber must NEVER park again. If
        // it re-checks a Mesa condition during shutdown and tries to re-block,
        // return WokenInWindow so it falls through to its terminate check (the
        // syscall's terminated/throwIfTerminated path, or the next yield_point)
        // and unwinds. Without this, a Mesa re-block could re-enter Blocked with
        // no further shutdown wake and hang g_guest_thread.join().
        if (g_stop.load(std::memory_order_relaxed) && fc->terminateRequested.load(std::memory_order_relaxed))
        {
            fc->wake_pending = false;
            fc->state = FiberContext::State::Running; // never parked
            return BlockResult::WokenInWindow;
        }
        // A waker may have fired between arm_park() and here (after the syscall
        // published to the object wait-list). wake_locked recorded it in
        // wake_pending. If so, consume it and DO NOT park.
        if (fc->wake_pending)
        {
            fc->wake_pending = false;
            fc->state = FiberContext::State::Running; // we never actually parked
            return BlockResult::WokenInWindow;
        }
        // No wakeup yet. Confirm Blocked (arm_park already set it; this is a
        // harmless re-affirmation and also covers callers that did not arm).
        fc->wake_pending = false;
        fc->state = FiberContext::State::Blocked;
    }
    ps2fiber_yield(); // executor sees Blocked; re-enqueues iff wake_pending.
    // Resumes here when the executor pops fc again (state set to Running by it).
    return BlockResult::Parked;
}

// ---------------------------------------------------------------------------
// make_ready (suspendCount gate)
// ---------------------------------------------------------------------------
void ps2sched::make_ready(int tid)
{
    bool notify = false;
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        FiberContext* fc = fiber_for(tid);
        // No state==Blocked gate: a waker that fires in the publish/arm window
        // (state still Running, fiber still g_running_fiber) must reach
        // wake_locked so it can record wake_pending. wake_locked itself routes:
        //   in_run_queue        -> no-op (already queued)
        //   g_running_fiber==fc -> wake_pending=true (mid-park / publish window)
        //   else (Blocked)      -> enqueue_locked
        // The suspendCount gate stays: wake_locked does not check it, and a
        // suspended fiber must not be enqueued; resume_fiber/clear_suspend wakes
        // it when the suspend is lifted.
        if (fc && fc->suspendCount.load(std::memory_order_relaxed) == 0) // GATE
        {
            wake_locked(fc);
            notify = true;
        }
    }
    if (notify) g_sched_cv.notify_all();
}

// ---------------------------------------------------------------------------
// enqueue_external_wakeup (suspendCount gate)
// ---------------------------------------------------------------------------
void ps2sched::enqueue_external_wakeup(int tid)
{
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        FiberContext* fc = fiber_for(tid);
        // See make_ready: no state==Blocked gate so a waker in the publish/arm
        // window reaches wake_locked and records wake_pending. suspendCount gate
        // preserved because wake_locked does not check it.
        if (fc && fc->suspendCount.load(std::memory_order_relaxed) == 0) // GATE
        {
            wake_locked(fc);
        }
    }
    g_sched_cv.notify_all();
}

// ---------------------------------------------------------------------------
// make_fiber_token — encode {generation, tid} as a 64-bit token.
// ---------------------------------------------------------------------------
static inline uint64_t make_fiber_token(const FiberContext* fc)
{
    return (static_cast<uint64_t>(fc->generation) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(fc->tid));
}

// ---------------------------------------------------------------------------
// current_fiber_token — opaque identity of the currently-running fiber.
// ---------------------------------------------------------------------------
uint64_t ps2sched::current_fiber_token()
{
    FiberContext* fc = tls_current_fiber;
    return fc ? make_fiber_token(fc) : 0u;
}

// ---------------------------------------------------------------------------
// enqueue_external_wakeup_validated — like enqueue_external_wakeup but
// validates the fiber identity before waking, to guard against tid recycling.
// ---------------------------------------------------------------------------
void ps2sched::enqueue_external_wakeup_validated(int tid, uint64_t token)
{
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        FiberContext* fc = fiber_for(tid);
        // Identity check by {generation, tid} token, NOT by pointer. If `tid` was
        // recycled to a new fiber, that fiber has a different generation and the
        // token mismatches, so the stale wakeup is dropped.
        // Identity check by {generation, tid} token. No state==Blocked gate:
        // WaitForNextVSyncTick has the same publish/arm window as WaitSema, so a
        // tick that fires while the fiber is still Running must reach wake_locked.
        if (fc != nullptr &&
            token != 0u &&
            make_fiber_token(fc) == token &&
            fc->suspendCount.load(std::memory_order_relaxed) == 0)
        {
            wake_locked(fc);
        }
    }
    g_sched_cv.notify_all();
}

// ---------------------------------------------------------------------------
// maybe_yield — yield if a higher-priority fiber is ready.
// ---------------------------------------------------------------------------
void ps2sched::maybe_yield()
{
    FiberContext* fc = tls_current_fiber;
    if (!fc) return; // called from a host worker: no-op
    bool yield = false;
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        if (g_run_queue && g_run_queue->priority < fc->priority)
        {
            enqueue_locked(fc); // enqueue Ready before yielding
            yield = true;
        }
    }
    if (yield) ps2fiber_yield();
}

// ---------------------------------------------------------------------------
// suspend_self
// ---------------------------------------------------------------------------
void ps2sched::suspend_self()
{
    FiberContext* fc = tls_current_fiber;
    if (fc == nullptr)
    {
        // Borrowed host worker; cannot suspend a non-fiber. No-op + log.
        std::fprintf(stderr,
                     "[ps2sched] WARNING: suspend_self() called from non-fiber "
                     "thread (tid=%d); ignoring\n",
                     g_currentThreadId);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        fc->suspendCount.fetch_add(1, std::memory_order_relaxed);
        fc->wake_pending = false;           // arm the park (same as arm_park)
        fc->state = FiberContext::State::Blocked;
    }
    ps2fiber_yield(); // unconditional park; resume via resume_fiber/clear_suspend
}

// ---------------------------------------------------------------------------
// suspend_other
// ---------------------------------------------------------------------------
void ps2sched::suspend_other(int tid)
{
    std::lock_guard<std::mutex> lk(g_sched_mutex);
    FiberContext* fc = fiber_for(tid);
    if (!fc) return;
    fc->suspendCount.fetch_add(1, std::memory_order_relaxed);
    if (fc->state == FiberContext::State::Ready)
    {
        remove_locked(fc);                        // pull it out of the queue now
        fc->state = FiberContext::State::Blocked;
    }
    // If Blocked already: stays blocked (gate keeps it off-queue on wakeups).
    // Cannot be Running: only one fiber runs at a time (N=1 invariant).
}

// ---------------------------------------------------------------------------
// resume_fiber
// ---------------------------------------------------------------------------
void ps2sched::resume_fiber(int tid)
{
    bool notify = false;
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        FiberContext* fc = fiber_for(tid);
        if (!fc) return;
        int sc = fc->suspendCount.load(std::memory_order_relaxed);
        if (sc > 0) fc->suspendCount.store(sc - 1, std::memory_order_relaxed);
        if (fc->suspendCount.load(std::memory_order_relaxed) == 0 &&
            fc->state == FiberContext::State::Blocked)
        {
            wake_locked(fc); // suspendCount hit 0 -> runnable
            notify = true;
        }
    }
    if (notify) g_sched_cv.notify_all();
}

// ---------------------------------------------------------------------------
// clear_suspend — zero the scheduler gate and wake if Blocked
// ---------------------------------------------------------------------------
void ps2sched::clear_suspend(int tid)
{
    bool notify = false;
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        FiberContext* fc = fiber_for(tid);
        if (!fc) return;
        fc->suspendCount.store(0, std::memory_order_relaxed);
        if (fc->state == FiberContext::State::Blocked)
        {
            wake_locked(fc); // race-safe enqueue / wake_pending
            notify = true;
        }
    }
    if (notify) g_sched_cv.notify_all();
}

// ---------------------------------------------------------------------------
// rotate_ready_queue
// ---------------------------------------------------------------------------
void ps2sched::rotate_ready_queue(int priority)
{
    // The RUNNING fiber is not in g_run_queue (the executor pops it before
    // resuming). On real hardware the running thread IS the head of its
    // priority's ready queue, so RotateThreadReadyQueue(myPriority) moves the
    // CALLER to the tail of its group and reschedules — that is the whole point
    // of the syscall (a guest "yield to my equals" primitive; DQ8's fn_1a1050
    // is an infinite `RotateThreadReadyQueue(prio); b .` loop).
    //
    // Rotating only the queued nodes, as this function used to do, never moves
    // the caller, and the caller's maybe_yield() only preempts for a STRICTLY
    // higher-priority fiber — so an equal-priority yield loop monopolised the
    // N=1 cooperative CPU forever and starved every same-priority Ready fiber.
    FiberContext* self = tls_current_fiber;
    bool yieldSelf = false;
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);

        if (self != nullptr && self->priority == priority)
        {
            // Caller is the head of this group. Only actually round-robin if
            // somebody else can run at this priority or better; otherwise a
            // requeue+switch would just resume us and burn fiber switches.
            bool someoneElseRunnable = false;
            for (FiberContext* n = g_run_queue; n != nullptr; n = n->next)
            {
                if (n->priority > priority) break; // queue is priority-ordered
                if (n != self)
                {
                    someoneElseRunnable = true;
                    break;
                }
            }
            if (someoneElseRunnable)
            {
                enqueue_locked(self); // tail of our priority group, state=Ready
                yieldSelf = true;
            }
            // The remaining same-priority nodes keep their relative order, which
            // is exactly head->tail rotation with the caller as head.
        }
        else
        {
            // Rotating a group we are not part of: move that group's head node
            // to the tail. No reschedule of the caller.
            FiberContext** pp = &g_run_queue;
            while (*pp && (*pp)->priority != priority)
                pp = &(*pp)->next;
            if (!*pp) return;
            FiberContext* victim = *pp; // first node at this priority
            *pp = victim->next;
            victim->next = nullptr;
            FiberContext** ins = pp; // re-insert after the last node at this priority
            while (*ins && (*ins)->priority == priority)
                ins = &(*ins)->next;
            victim->next = *ins;
            *ins = victim;
        }
    }

    if (yieldSelf)
    {
        ps2fiber_yield(); // executor sees Ready+queued and reschedules us later
    }
}

// ---------------------------------------------------------------------------
// async_guest_begin — aborts if called from the guest executor thread (only host workers borrow the token).
// ---------------------------------------------------------------------------
void ps2sched::async_guest_begin()
{
    if (tls_is_guest_thread)
    {
        std::fprintf(stderr,
                     "FATAL [ps2sched]: async_guest_begin from guest executor thread\n");
        std::terminate();
    }
    g_host_token_waiters.fetch_add(1, std::memory_order_relaxed);
    std::unique_lock<std::mutex> lk(g_sched_mutex);
    g_sched_cv.wait(lk, []
    {
        return g_running_fiber == nullptr && !g_guest_token_held_by_host;
    });
    g_guest_token_held_by_host = true;
    tls_holds_guest_token = true;
    // g_currentThreadId stays -1 on this host worker thread.
    g_host_token_waiters.fetch_sub(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// async_guest_end
// ---------------------------------------------------------------------------
void ps2sched::async_guest_end()
{
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        // Only the worker that acquired the token may release it. A non-owner
        // reaching here means a blocking-syscall retry path called end() without
        // owning the token — a bug. Refuse to clear a token we do not own.
        SCHED_REQUIRE(g_guest_token_held_by_host,
                      "async_guest_end with token not held");
        SCHED_REQUIRE(tls_holds_guest_token,
                      "async_guest_end called by non-owner");
        g_guest_token_held_by_host = false;
        tls_holds_guest_token = false;
    }
    g_sched_cv.notify_all(); // wake the executor
}

// ---------------------------------------------------------------------------
// yield_point — sampled every 128 back-edges; checks terminate/suspend/priority.
// ---------------------------------------------------------------------------
bool ps2sched::yield_point()
{
    FiberContext* fc = tls_current_fiber;
    // Shutdown fast-path: force the unwind at the very next back-edge of a
    // terminate-requested fiber, without waiting for the 128-sample window.
    if (fc && g_stop.load(std::memory_order_relaxed) && fc->terminateRequested.load(std::memory_order_relaxed))
        throw ThreadExitException();

    if ((++tls_backedge_counter & 127u) != 0u) return false; // cheap fast path

    if (!fc) return false; // running under a host worker (AsyncGuestScope): no preempt

    // 1. Terminate request -> unwind THIS fiber's own stack.
    if (fc->terminateRequested.load(std::memory_order_relaxed))
    {
        throw ThreadExitException(); // caught in fiber_trampoline
    }

    // 2. Suspend request -> block self.
    if (fc->suspendCount.load(std::memory_order_relaxed) > 0)
    {
        block_current();
        return false;
    }

    // 3. Higher-priority fiber ready -> cooperative yield (enqueue Ready first).
    bool yield = false;
    {
        std::lock_guard<std::mutex> lk(g_sched_mutex);
        if (g_run_queue && g_run_queue->priority < fc->priority)
        {
            enqueue_locked(fc);
            yield = true;
        }
    }
    if (yield) ps2fiber_yield();

    // 4. If a host worker (interrupt worker) is parked waiting for the guest token,
    // cooperatively yield so it can run VSync/INTC handlers.
    if (fc && g_host_token_waiters.load(std::memory_order_relaxed) > 0) {
        ps2fiber_yield();   // returns to executor; executor re-enqueues us (state=Running)
        // After resuming, re-check terminate in case scheduler_shutdown() fired.
        if (g_stop.load(std::memory_order_relaxed) &&
            fc->terminateRequested.load(std::memory_order_relaxed))
            throw ThreadExitException();
        return true;
    }
    return false;
}

// Diagnostic accessor for g_host_token_waiters.
int ps2sched::host_token_waiters()
{
    return g_host_token_waiters.load(std::memory_order_relaxed);
}

// True only on the single guest executor thread. Mirrors the exact predicate
// async_guest_begin() uses to abort -- lets shared dispatch helpers that may be
// invoked either by a host worker (which must borrow the token) or, on some
// call paths, synchronously from already-running guest code (which already
// owns the execution slot) pick the correct behavior without duplicating that
// invariant.
bool ps2sched::is_guest_thread()
{
    return tls_is_guest_thread;
}

namespace
{
    const char *fiberStateName(FiberContext::State s)
    {
        switch (s)
        {
            case FiberContext::State::Fresh:    return "Fresh";
            case FiberContext::State::Ready:    return "Ready";
            case FiberContext::State::Running:  return "Running";
            case FiberContext::State::Blocked:  return "Blocked";
            case FiberContext::State::Exiting:  return "Exiting";
            case FiberContext::State::Finished: return "Finished";
        }
        return "?";
    }
}

// Diagnostic: dump every fiber's tid/priority/state/pc/ra. See ps2_scheduler.h.
void ps2sched::dump_all_fibers(const char *reasonTag)
{
    std::vector<std::tuple<int,int,FiberContext::State,uint32_t,uint32_t>> rows;
    {
        std::lock_guard<std::mutex> lock(g_sched_mutex);
        rows.reserve(g_fiber_map.size());
        for (const auto &[tid, fc] : g_fiber_map)
        {
            if (!fc) continue;
            const uint32_t pc = fc->cpu.pc;
            const uint32_t ra = getRegU32(&fc->cpu, 31);
            rows.emplace_back(tid, fc->priority, fc->state, pc, ra);
        }
    }
    std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) { return std::get<0>(a) < std::get<0>(b); });
    std::cout << "[dq8][fiberdump] reason=" << (reasonTag ? reasonTag : "?")
              << " count=" << rows.size() << std::endl;
    for (const auto &[tid, prio, state, pc, ra] : rows)
    {
        std::cout << "[dq8][fiberdump]   tid=" << tid
                  << " prio=" << prio
                  << " state=" << fiberStateName(state)
                  << " pc=0x" << std::hex << pc
                  << " ra=0x" << ra << std::dec
                  << std::endl;
    }
}

// Declared in ps2_fiber.h. Lets the fiber backend assert it only switches
// contexts on the registered executor thread without exposing the thread id.
bool ps2fiber_on_executor_thread()
{
    return tls_is_executor_thread;
}
