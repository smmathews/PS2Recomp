#pragma once
#include <cstddef>

// Opaque fiber handle.
struct PS2Fiber;

// Allocate a fiber that will run fn(arg) on its own stack of stack_bytes.
// Returns nullptr on allocation failure (caller must handle — see create_fiber).
PS2Fiber* ps2fiber_alloc(void (*fn)(void*), void* arg, size_t stack_bytes);

#ifdef PLATFORM_VITA
// Vita-only: associate a guest thread id with a fiber. Published onto the
// fiber's own pthread (g_currentThreadId) in ps2fiber_thread_main so blocking
// syscalls park correctly. Call after ps2fiber_alloc, before first resume.
// The POSIX ucontext backend has no per-fiber OS thread and sets
// g_currentThreadId in guest_executor_main, so this does not exist there.
void ps2fiber_set_tid(PS2Fiber* f, int tid);
#endif

// Free the fiber's stack and struct. Must NOT be called while the fiber is
// running or is mid-yield (i.e. only after fn returned / it is Finished, or
// before it was ever resumed).
void      ps2fiber_free(PS2Fiber* f);

// Resume (switch TO) fiber f. MUST be called only from the guest executor
// thread (the thread that owns the "main" context). Returns when f yields or
// when f's fn returns.
void      ps2fiber_resume(PS2Fiber* f);

// Yield (switch FROM the current fiber back to the guest executor main context).
// MUST be called only from within a fiber running on the guest executor thread.
void      ps2fiber_yield();

// The fiber currently running on this thread, or nullptr if not on a fiber.
PS2Fiber* ps2fiber_current();

// True if the fiber's function has already returned. On the POSIX ucontext
// backend this is always false: a ucontext fiber has no separate OS thread,
// so it cannot be "dead" while its PS2Fiber still exists. The POSIX executor
// relies on FiberContext::state == Finished (set in the fiber trampoline) to
// detect completion rather than polling this function.
bool ps2fiber_finished(PS2Fiber* f);

// True when called on the registered guest executor thread. Defined in
// ps2_scheduler.cpp. Used by the POSIX backend to assert that fiber context
// switches only happen on the one executor thread (the N=1 invariant), which
// is what makes saving/restoring a ucontext_t safe.
bool ps2fiber_on_executor_thread();
