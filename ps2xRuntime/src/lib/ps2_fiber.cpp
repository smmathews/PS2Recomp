#include "ps2_fiber.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

#if !defined(PLATFORM_VITA)
// ============================================================================
// POSIX path — ucontext_t
// ============================================================================
#include <ucontext.h>
#include <sys/mman.h>
#include <unistd.h> // sysconf(_SC_PAGESIZE)

static_assert(sizeof(void*) <= 8, "makecontext pointer split requires 64-bit pointers");
static_assert(sizeof(unsigned int) == 4, "makecontext int args must be 32 bits");

// Abort if a fiber context switch is attempted off the guest executor thread.
// The whole design assumes exactly one thread ever saves/restores these
// ucontext_t values; doing it from another thread is undefined behaviour.
static inline void ps2fiber_require_executor_thread(const char* who)
{
    if (!ps2fiber_on_executor_thread())
    {
        std::fprintf(stderr, "FATAL: %s called off the guest executor thread\n", who);
        std::abort();
    }
}

// Per-guest-executor-thread state. These are thread_local but in practice
// only ever touched by the single g_guest_thread.
static thread_local ucontext_t tls_guest_main_ctx;
static thread_local PS2Fiber*  tls_current_fiber_ptr = nullptr;

struct PS2Fiber
{
    ucontext_t ctx;
    uint8_t*   map_base  = nullptr; // mmap base (guard page first)
    size_t     map_size  = 0;       // total mapping incl. guard page
    uint8_t*   stack     = nullptr; // usable stack region (after guard page)
    size_t     stackSize = 0;
    void     (*fn)(void*) = nullptr;
    void*      arg = nullptr;
};

// makecontext can only pass int-sized arguments portably. Split the 64-bit
// PS2Fiber* into two 32-bit halves and reassemble on entry. fn/arg are stored
// in the struct, so we only need to recover the PS2Fiber* here.
static void ps2fiber_trampoline(unsigned int self_hi, unsigned int self_lo)
{
    uintptr_t raw = (static_cast<uintptr_t>(self_hi) << 32) | static_cast<uintptr_t>(self_lo);
    PS2Fiber* self = reinterpret_cast<PS2Fiber*>(raw);
    self->fn(self->arg);
    // fn returned. Switch back to the guest executor main context. The executor
    // observes fiber state == Finished and frees it. We must NOT return from
    // this function (there is no uc_link); swap out unconditionally.
    swapcontext(&self->ctx, &tls_guest_main_ctx);
    // Unreachable.
    std::abort();
}

PS2Fiber* ps2fiber_alloc(void (*fn)(void*), void* arg, size_t stack_bytes)
{
    if (stack_bytes == 0) { std::fprintf(stderr, "ps2fiber_alloc: zero stack size\n"); return nullptr; }
    long pageQuery = sysconf(_SC_PAGESIZE);
    const size_t page = (pageQuery > 0) ? static_cast<size_t>(pageQuery) : 4096u;
    size_t usable = (stack_bytes + page - 1) & ~(page - 1);
    if (usable == 0) usable = 1024 * 1024; // unreachable after the check above
    size_t total = usable + page; // 1 guard page at low address

    void* base = mmap(nullptr, total, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED)
    {
        std::fprintf(stderr, "[ps2fiber] mmap failed\n");
        return nullptr;
    }
    if (mprotect(base, page, PROT_NONE) != 0) // guard page
    {
        munmap(base, total);
        std::fprintf(stderr, "[ps2fiber] mprotect guard failed\n");
        return nullptr;
    }

    PS2Fiber* f = new (std::nothrow) PS2Fiber();
    if (!f)
    {
        munmap(base, total);
        return nullptr;
    }

    f->map_base  = static_cast<uint8_t*>(base);
    f->map_size  = total;
    f->stack     = static_cast<uint8_t*>(base) + page; // usable region after guard
    f->stackSize = usable;
    f->fn  = fn;
    f->arg = arg;

    if (getcontext(&f->ctx) != 0)
    {
        munmap(base, total);
        delete f;
        return nullptr;
    }
    f->ctx.uc_stack.ss_sp   = f->stack;
    f->ctx.uc_stack.ss_size = f->stackSize;
    f->ctx.uc_link          = nullptr; // trampoline never returns via link

    uintptr_t raw = reinterpret_cast<uintptr_t>(f);
    makecontext(&f->ctx, reinterpret_cast<void(*)()>(ps2fiber_trampoline), 2,
                static_cast<unsigned int>(raw >> 32),
                static_cast<unsigned int>(raw & 0xFFFFFFFFu));
    return f;
}

void ps2fiber_free(PS2Fiber* f)
{
    if (!f) return;
    if (f->map_base) munmap(f->map_base, f->map_size);
    delete f;
}

void ps2fiber_resume(PS2Fiber* f)
{
    ps2fiber_require_executor_thread("ps2fiber_resume");
    PS2Fiber* prev = tls_current_fiber_ptr;
    tls_current_fiber_ptr = f;
    if (prev == nullptr)
    {
        // Resuming from the guest executor main context.
        swapcontext(&tls_guest_main_ctx, &f->ctx);
    }
    else
    {
        // Unreachable in the N=1 design: the executor only ever resumes a fiber
        // from the main context (prev == nullptr). A non-null prev would mean a
        // fiber resumed another fiber directly, which the scheduler never does.
        std::fprintf(stderr, "FATAL: nested ps2fiber_resume (prev != nullptr)\n");
        std::abort();
    }
    tls_current_fiber_ptr = prev;
}

void ps2fiber_yield()
{
    ps2fiber_require_executor_thread("ps2fiber_yield");
    PS2Fiber* self = tls_current_fiber_ptr;
    if (!self)
    {
        std::fprintf(stderr, "FATAL: ps2fiber_yield with no current fiber\n");
        std::abort();
    }
    tls_current_fiber_ptr = nullptr;
    swapcontext(&self->ctx, &tls_guest_main_ctx);
    tls_current_fiber_ptr = self;
}

PS2Fiber* ps2fiber_current()
{
    return tls_current_fiber_ptr;
}

bool ps2fiber_finished(PS2Fiber* /*f*/)
{
    // ucontext backend: no separate OS thread to outlive the PS2Fiber. The
    // executor relies on FiberContext::state == Finished instead.
    return false;
}

#else // PLATFORM_VITA
// ============================================================================
// Vita path — joinable pthread + sem_t
//
// NOT VALIDATED for the N=1 cooperative scheduler. The wake_pending / mid-park
// protocol in ps2_scheduler.cpp assumes exactly one host thread executes guest
// code at a time; the pthread backend runs each fiber on its own OS thread and
// the protocol's timing has not been re-proven for that concurrency model. The
// code below is retained for future work but must NOT compile until it is
// validated and tested.
// ============================================================================
#error "Vita fiber backend is not validated for the cooperative scheduler; do not enable PLATFORM_VITA until it is implemented and tested."
#include <pthread.h>
#include <semaphore.h>
#include "ps2_scheduler.h" // extern thread_local int g_currentThreadId

static thread_local PS2Fiber* tls_current_fiber_ptr = nullptr;

struct PS2Fiber
{
    pthread_t thread;
    sem_t     resume_sem; // posted by ps2fiber_resume; waited by fiber thread
    sem_t     yield_sem;  // posted by ps2fiber_yield/finish; waited by resume
    void    (*fn)(void*) = nullptr;
    void*     arg        = nullptr;
    bool      started    = false; // pthread created
    std::atomic<bool> finished{false}; // fn returned; written on the fiber thread and read on the executor, so must be atomic
    size_t    stackSize  = 0;
    // Guest thread id for this fiber. Set by the scheduler via
    // ps2fiber_set_tid() after alloc; published onto the fiber's own pthread in
    // ps2fiber_thread_main so blocking syscalls see the right g_currentThreadId.
    // Defaults to -1 until the scheduler assigns it.
    int       tid        = -1;
};

static void* ps2fiber_thread_main(void* raw)
{
    PS2Fiber* self = reinterpret_cast<PS2Fiber*>(raw);
    sem_wait(&self->resume_sem); // wait for first resume
    // Publish guest identity ON THE FIBER'S OWN THREAD. On Vita guest code
    // runs here, not on the executor thread, so blocking syscalls must read the
    // fiber's tid / current-fiber pointer from this thread's TLS.
    tls_current_fiber_ptr = self;
    g_currentThreadId = self->tid;
    self->fn(self->arg);
    self->finished.store(true, std::memory_order_release);
    g_currentThreadId = -1;
    tls_current_fiber_ptr = nullptr;
    sem_post(&self->yield_sem); // hand control back to executor; thread ends
    return nullptr;
}

PS2Fiber* ps2fiber_alloc(void (*fn)(void*), void* arg, size_t stack_bytes)
{
    PS2Fiber* f = new (std::nothrow) PS2Fiber();
    if (!f) return nullptr;
    f->fn        = fn;
    f->arg       = arg;
    f->stackSize = stack_bytes ? stack_bytes : (512 * 1024); // smaller on Vita
    sem_init(&f->resume_sem, 0, 0);
    sem_init(&f->yield_sem,  0, 0);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    // Joinable (not detached) so ps2fiber_free can join the thread and avoid a use-after-free.
    pthread_attr_setstacksize(&attr, f->stackSize);
    int rc = pthread_create(&f->thread, &attr, ps2fiber_thread_main, f);
    pthread_attr_destroy(&attr);
    if (rc != 0)
    {
        sem_destroy(&f->resume_sem);
        sem_destroy(&f->yield_sem);
        delete f;
        return nullptr;
    }
    f->started = true;
    return f;
}

void ps2fiber_set_tid(PS2Fiber* f, int tid)
{
    if (f) f->tid = tid;
}

void ps2fiber_resume(PS2Fiber* f)
{
    sem_post(&f->resume_sem); // wake fiber thread
    sem_wait(&f->yield_sem);  // block until it yields or finishes
}

void ps2fiber_yield()
{
    PS2Fiber* self = tls_current_fiber_ptr;
    if (!self) { std::abort(); }
    sem_post(&self->yield_sem);  // hand back to executor
    sem_wait(&self->resume_sem); // wait for next resume
}

PS2Fiber* ps2fiber_current()
{
    return tls_current_fiber_ptr;
}

bool ps2fiber_finished(PS2Fiber* f)
{
    // ps2fiber_thread_main sets finished=true just before the pthread exits.
    // The executor uses this to avoid resuming a dead fiber thread.
    return f && f->finished.load(std::memory_order_acquire);
}

void ps2fiber_free(PS2Fiber* f)
{
    if (!f) return;
    if (f->started && !f->finished.load(std::memory_order_acquire))
    {
        // Should not happen in normal flow (fibers run to completion before free).
        // Caller (scheduler) must only free Finished fibers.
        std::fprintf(stderr, "FATAL: ps2fiber_free on unfinished fiber\n");
        std::abort();
    }
    if (f->started) pthread_join(f->thread, nullptr); // joinable: no leak
    sem_destroy(&f->resume_sem);
    sem_destroy(&f->yield_sem);
    delete f;
}

#endif // PLATFORM_VITA
