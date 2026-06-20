#include "Common.h"
#include "Interrupt.h"
#include "Lifecycle.h"
#include "Sync.h"

namespace ps2_syscalls
{
    using namespace interrupt_state;

    void notifyRuntimeStop()
    {
        stopInterruptWorker();
        {
            std::lock_guard<std::mutex> lock(g_irq_handler_mutex);
            g_intcHandlers.clear();
            g_dmacHandlers.clear();
            g_nextIntcHandlerId = 1;
            g_nextDmacHandlerId = 1;
            g_enabled_intc_mask = 0xFFFFFFFFu;
            g_enabled_dmac_mask = 0xFFFFFFFFu;
        }
        {
            std::lock_guard<std::mutex> lock(g_vsync_flag_mutex);
            g_vsync_registration = {};
            g_vsync_tick_counter = 0u;
        }

        std::vector<std::pair<int, std::shared_ptr<ThreadInfo>>> threads;
        threads.reserve(32);
        {
            std::lock_guard<std::mutex> lock(g_thread_map_mutex);
            for (const auto &entry : g_threads)
            {
                if (entry.second)
                {
                    threads.emplace_back(entry.first, entry.second);
                }
            }
            g_threads.clear();
            g_nextThreadId = 2; // Reserve id 1 for main thread.
        }
        // -1 is the "not a guest fiber" sentinel. notifyRuntimeStop runs on a
        // host thread (not the guest executor), which must never be mistaken for
        // a real guest thread id by arm_park / wait-list operations.
        g_currentThreadId = -1;

        for (const auto &[tid, threadInfo] : threads)
        {
            {
                std::lock_guard<std::mutex> lock(threadInfo->m);
                threadInfo->forceRelease = true;
                threadInfo->terminated = true;
            }
            ps2sched::request_terminate(tid);
        }

        std::vector<std::shared_ptr<SemaInfo>> semas;
        {
            std::lock_guard<std::mutex> lock(g_sema_map_mutex);
            semas.reserve(g_semas.size());
            for (const auto &entry : g_semas)
            {
                if (entry.second)
                {
                    semas.push_back(entry.second);
                }
            }
            g_semas.clear();
            g_nextSemaId = 1;
        }
        for (const auto &sema : semas)
        {
            std::vector<std::pair<int,uint64_t>> waiters;
            {
                std::lock_guard<std::mutex> lk(sema->m);
                waiters.swap(sema->waitList);
            }
            for (const auto& [tid, token] : waiters)
            {
                ps2sched::enqueue_external_wakeup_validated(tid, token);
            }
        }

        std::vector<std::shared_ptr<EventFlagInfo>> eventFlags;
        {
            std::lock_guard<std::mutex> lock(g_event_flag_map_mutex);
            eventFlags.reserve(g_eventFlags.size());
            for (const auto &entry : g_eventFlags)
            {
                if (entry.second)
                {
                    eventFlags.push_back(entry.second);
                }
            }
            g_eventFlags.clear();
            g_nextEventFlagId = 1;
        }
        for (const auto &eventFlag : eventFlags)
        {
            std::vector<std::pair<int,uint64_t>> waiters;
            {
                std::lock_guard<std::mutex> lk(eventFlag->m);
                waiters.swap(eventFlag->waitList);
            }
            for (const auto& [tid, token] : waiters)
            {
                ps2sched::enqueue_external_wakeup_validated(tid, token);
            }
        }

        // Stop and JOIN the alarm worker BEFORE clearing alarms, so no callback can
        // fire against rdram/runtime that is about to be destroyed.
        ps2_syscalls::stopAlarmWorker();
        {
            std::lock_guard<std::mutex> lock(g_alarm_mutex);
            g_alarms.clear();
        }
        g_alarm_cv.notify_all();

        {
            std::lock_guard<std::mutex> lock(g_exit_handler_mutex);
            g_exit_handlers.clear();
        }
        {
            std::lock_guard<std::mutex> lock(g_syscall_override_mutex);
            g_syscall_overrides.clear();
        }
    }

}
