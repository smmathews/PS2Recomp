#include "MiniTest.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/r5900_decoder.h"
#include "ps2recomp/types.h"
#include "ps2_runtime.h"
#include "runtime/ps2_memory.h"
#include "ps2_syscalls.h"
#include "ps2_stubs.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_gs_psmct32.h"
#include "ps2_runtime_macros.h"
#include "Stubs/MPEG.h"
#include "Stubs/Audio.h"
#include "Stubs/GS.h"
#include "Stubs/VU.h"
#include "Kernel/Syscalls/Interrupt.h"   // interrupt_state::g_vsync_waitList, WaitForNextVSyncTick, EnsureVSyncWorkerRunning, stopInterruptWorker

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "Kernel/Syscalls/Helpers/State.h"   // g_threads / g_thread_map_mutex

using namespace ps2recomp;
using namespace ps2_syscalls;

namespace
{
    constexpr uint32_t COP0_CAUSE_BD = 0x80000000u;
    constexpr uint32_t COP0_CAUSE_EXCCODE_MASK = 0x0000007Cu;
    constexpr uint32_t COP0_STATUS_EXL = 0x00000002u;
    constexpr uint32_t COP0_STATUS_BEV = 0x00400000u;
    constexpr uint32_t EXCEPTION_VECTOR_GENERAL = 0x80000080u;
    constexpr uint32_t EXCEPTION_VECTOR_BOOT = 0xBFC00200u;

    void setRegU32(R5900Context &ctx, int reg, uint32_t value)
    {
        ctx.r[reg] = _mm_set_epi64x(0, static_cast<int64_t>(value));
    }

    int32_t getRegS32(const R5900Context &ctx, int reg)
    {
        return static_cast<int32_t>(::getRegU32(&ctx, reg));
    }

    uint32_t makeVifCmd(uint8_t opcode, uint8_t num, uint16_t imm)
    {
        return (static_cast<uint32_t>(opcode) << 24) |
               (static_cast<uint32_t>(num) << 16) |
               static_cast<uint32_t>(imm);
    }

    bool hasSignedRdWrite(const std::string &generated, uint8_t rd)
    {
        if (rd == 0u)
        {
            return false;
        }

        const std::string needle = "SET_GPR_S32(ctx, " + std::to_string(rd) + ",";
        return generated.find(needle) != std::string::npos;
    }

    template <typename Predicate>
    bool waitUntil(Predicate pred, std::chrono::milliseconds timeout)
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (pred())
            {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return pred();
    }

    uint32_t frameOffsetBytes(uint32_t x, uint32_t y, uint32_t fbw)
    {
        return GSPSMCT32::addrPSMCT32(0u, (fbw != 0u) ? fbw : 1u, x, y);
    }

    void testRuntimeWorkerLoop(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!ctx || !runtime)
        {
            return;
        }

        // Keep touching guest memory so teardown races are easier to catch.
        (void)Ps2FastRead64(rdram, static_cast<uint32_t>(0x01FFFFF8u + (ctx->insn_count & 0x7u)));
        ++ctx->insn_count;

        if (runtime->isStopRequested())
        {
            ctx->pc = 0u;
            return;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    constexpr uint32_t kAsyncCounterAddr = 0x2400u;

    void testWaitForAsyncCounter(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        if (!rdram || !ctx)
        {
            return;
        }

        uint32_t counter = 0u;
        do
        {
            std::memcpy(&counter, rdram + kAsyncCounterAddr, sizeof(counter));
            if (counter == 0u)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } while (counter == 0u);

        ctx->pc = 0u;
    }

    void testSignalAsyncCounter(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        if (rdram)
        {
            const uint32_t counter = 1u;
            std::memcpy(rdram + kAsyncCounterAddr, &counter, sizeof(counter));
        }

        if (ctx)
        {
            ctx->pc = 0u;
        }
    }

    std::atomic<uint32_t> gAsyncCallbackObservedSp{0u};
    std::atomic<uint32_t> gAsyncCallbackObservedGp{0u};

    void testRecordAsyncCallbackStack(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        if (!ctx)
        {
            return;
        }

        gAsyncCallbackObservedSp.store(::getRegU32(ctx, 29), std::memory_order_release);
        gAsyncCallbackObservedGp.store(::getRegU32(ctx, 28), std::memory_order_release);
        ctx->pc = 0u;
    }

}

// ---------------------------------------------------------------------------
// Scheduler test helpers and per-test state
// ---------------------------------------------------------------------------

namespace
{
    // -----------------------------------------------------------------------
    // Shared state for scheduler step functions (indexed by test slot 0..2)
    // -----------------------------------------------------------------------
    constexpr uint32_t kSchedTestSemaIdAddr   = 0x3000u; // rdram offset: holds sid (int32)
    constexpr uint32_t kSchedDoneSemaIdAddr   = 0x3004u; // rdram offset: holds done_sid (int32)
    constexpr uint32_t kSchedLogBase          = 0x3010u; // rdram offset: int32[4] log slots
    constexpr uint32_t kSchedResultBase       = 0x3020u; // rdram offset: int32[4] result slots
    std::atomic<int32_t> gSchedSeqCounter{0};

    // Worker step function: WaitSema(sid from rdram), record result and sequence, signal done_sid.
    // In the fiber model this function IS the fiber body — it runs on a pool thread and
    // blocks cooperatively via WaitSema (which calls block_current() internally).
    void schedWorkerWaitAndSignal(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!rdram || !ctx)
        {
            if (ctx) ctx->pc = 0u;
            return;
        }

        // Read sema id from rdram
        int32_t sid = 0;
        std::memcpy(&sid, rdram + kSchedTestSemaIdAddr, sizeof(sid));
        int32_t doneSid = 0;
        std::memcpy(&doneSid, rdram + kSchedDoneSemaIdAddr, sizeof(doneSid));

        // WaitSema(sid) — blocks cooperatively if count==0
        R5900Context wCtx{};
        setRegU32(wCtx, 4, static_cast<uint32_t>(sid));
        ps2_syscalls::WaitSema(rdram, &wCtx, runtime);
        const int32_t waitResult = getRegS32(wCtx, 2);

        // Record: sequence index → result
        const int32_t seq = gSchedSeqCounter.fetch_add(1, std::memory_order_relaxed);
        if (seq >= 0 && seq < 4)
        {
            std::memcpy(rdram + kSchedResultBase + static_cast<uint32_t>(seq * 4), &waitResult, sizeof(waitResult));
            const int32_t tid = g_currentThreadId;
            std::memcpy(rdram + kSchedLogBase + static_cast<uint32_t>(seq * 4), &tid, sizeof(tid));
        }

        // Signal done_sid to wake the main test fiber
        if (doneSid > 0)
        {
            R5900Context sCtx{};
            setRegU32(sCtx, 4, static_cast<uint32_t>(doneSid));
            ps2_syscalls::SignalSema(rdram, &sCtx, runtime);
        }

        ctx->pc = 0u;
    }

    // Worker step function: just WaitSema(sid) indefinitely (for TerminateThread test).
    void schedWorkerBlockForever(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!rdram || !ctx)
        {
            if (ctx) ctx->pc = 0u;
            return;
        }

        int32_t sid = 0;
        std::memcpy(&sid, rdram + kSchedTestSemaIdAddr, sizeof(sid));

        R5900Context wCtx{};
        setRegU32(wCtx, 4, static_cast<uint32_t>(sid));
        ps2_syscalls::WaitSema(rdram, &wCtx, runtime);

        ctx->pc = 0u;
    }

    // Helper: create a semaphore via CreateSema (EE layout: count, max_count, init_count).
    // Returns the sema id or -1 on failure.
    int32_t createSchedSema(uint8_t *rdram, PS2Runtime *runtime, int initCount, int maxCount)
    {
        constexpr uint32_t kSemaParamAddr = 0x2F00u;
        const uint32_t params[6] = {
            0u,                              // count (unused by EE layout)
            static_cast<uint32_t>(maxCount), // max_count
            static_cast<uint32_t>(initCount),// init_count
            0u,                              // wait_threads
            0u,                              // attr
            0u,                              // option
        };
        std::memcpy(rdram + kSemaParamAddr, params, sizeof(params));

        R5900Context cCtx{};
        setRegU32(cCtx, 4, kSemaParamAddr);
        ps2_syscalls::CreateSema(rdram, &cCtx, runtime);
        return getRegS32(cCtx, 2);
    }

    // Helper: delete a semaphore.
    void deleteSchedSema(uint8_t *rdram, PS2Runtime *runtime, int32_t sid)
    {
        if (sid <= 0)
        {
            return;
        }
        R5900Context dCtx{};
        setRegU32(dCtx, 4, static_cast<uint32_t>(sid));
        ps2_syscalls::DeleteSema(rdram, &dCtx, runtime);
    }

    // Helper: create+start a worker thread and return its tid (or -1 on error).
    // In the fiber model, StartThread enqueues a fiber; the pool threads run it.
    int32_t startSchedWorker(uint8_t *rdram, PS2Runtime *runtime,
                              uint32_t entryAddr, int priority,
                              uint32_t stackAddr, uint32_t stackSize)
    {
        constexpr uint32_t kThreadParamAddr = 0x2E00u;
        const uint32_t threadParam[7] = {
            0u,          // attr
            entryAddr,   // entry
            stackAddr,   // stack
            stackSize,   // stack size
            0u,          // gp (0 = caller's gp)
            static_cast<uint32_t>(priority),
            0u,          // option
        };
        std::memcpy(rdram + kThreadParamAddr, threadParam, sizeof(threadParam));

        R5900Context createCtx{};
        setRegU32(createCtx, 4, kThreadParamAddr);
        ps2_syscalls::CreateThread(rdram, &createCtx, runtime);
        const int32_t tid = getRegS32(createCtx, 2);
        if (tid <= 0)
        {
            return -1;
        }

        R5900Context startCtx{};
        setRegU32(startCtx, 4, static_cast<uint32_t>(tid));
        setRegU32(startCtx, 5, 0u);
        ps2_syscalls::StartThread(rdram, &startCtx, runtime);
        if (getRegS32(startCtx, 2) != 0)
        {
            return -1;
        }
        return tid;
    }

} // anonymous namespace

void register_ps2_runtime_expansion_tests()
{
    MiniTest::Case("PS2RuntimeExpansion", [](TestCase &tc)
    {
        tc.Run("differential decoder/codegen gpr-write contract for MULT and DIV families", [](TestCase &t)
        {
            R5900Decoder decoder;
            CodeGenerator generator({}, {});

            const struct
            {
                const char *name;
                uint32_t raw;
            } cases[] = {
                {"MULT rd!=0", (OPCODE_SPECIAL << 26) | (4u << 21) | (5u << 16) | (3u << 11) | SPECIAL_MULT},
                {"MULT rd==0", (OPCODE_SPECIAL << 26) | (4u << 21) | (5u << 16) | (0u << 11) | SPECIAL_MULT},
                {"DIV rd!=0", (OPCODE_SPECIAL << 26) | (6u << 21) | (7u << 16) | (9u << 11) | SPECIAL_DIV},
                {"MMI MULT1 rd!=0", (OPCODE_MMI << 26) | (8u << 21) | (9u << 16) | (10u << 11) | MMI_MULT1},
                {"MMI DIV1 rd!=0", (OPCODE_MMI << 26) | (8u << 21) | (9u << 16) | (10u << 11) | MMI_DIV1},
            };

            for (size_t i = 0; i < std::size(cases); ++i)
            {
                const Instruction inst = decoder.decodeInstruction(0x1000u + static_cast<uint32_t>(i * 4u), cases[i].raw);
                const std::string generated = generator.translateInstruction(inst);
                const bool emittedRdWrite = hasSignedRdWrite(generated, inst.rd);

                t.Equals(emittedRdWrite, inst.modificationInfo.modifiesGPR,
                         std::string("decoder/codegen mismatch for ") + cases[i].name);
                t.IsTrue(inst.modificationInfo.modifiesControl,
                         std::string("HI/LO control side-effect missing for ") + cases[i].name);
            }
        });

        tc.Run("vblank intc handlers can preempt serialized guest execution", [](TestCase &t)
        {
            notifyRuntimeStop();

            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            constexpr uint32_t kBusyEntry = 0x160000u;
            constexpr uint32_t kIntcHandlerEntry = 0x170000u;

            runtime.registerFunction(kBusyEntry, &testWaitForAsyncCounter);
            runtime.registerFunction(kIntcHandlerEntry, &testSignalAsyncCounter);

            R5900Context addCtx{};
            setRegU32(addCtx, 4, 2u);
            setRegU32(addCtx, 5, kIntcHandlerEntry);
            setRegU32(addCtx, 6, 0u);
            setRegU32(addCtx, 7, 0u);
            AddIntcHandler(rdram.data(), &addCtx, &runtime);
            t.IsTrue(getRegS32(addCtx, 2) > 0, "AddIntcHandler should register a VBlank handler");

            R5900Context busyCtx{};
            busyCtx.pc = kBusyEntry;
            std::atomic<bool> workerDone{false};
            std::atomic<bool> workerThrew{false};

            std::thread worker([&]()
            {
                try
                {
                    runtime.dispatchLoop(rdram.data(), &busyCtx);
                }
                catch (...)
                {
                    workerThrew.store(true, std::memory_order_release);
                }
                workerDone.store(true, std::memory_order_release);
            });

            ps2_syscalls::EnsureVSyncWorkerRunning(rdram.data(), &runtime);

            const bool finished = waitUntil([&]()
            {
                return workerDone.load(std::memory_order_acquire);
            }, std::chrono::milliseconds(250));

            if (!finished)
            {
                // Do NOT self-heal with the expected value. Write a distinct sentinel so
                // the counter==1 assertion below fails loudly when the handler never fired.
                // We still release the spinning worker (it loops until counter != 0) so the
                // binary does not hang on a slow/flaky machine.
                const uint32_t sentinel = 999u;
                std::memcpy(rdram.data() + kAsyncCounterAddr, &sentinel, sizeof(sentinel));
            }

            if (worker.joinable())
            {
                worker.join();
            }

            runtime.requestStop();
            notifyRuntimeStop();

            uint32_t counter = 0u;
            std::memcpy(&counter, rdram.data() + kAsyncCounterAddr, sizeof(counter));

            t.IsFalse(workerThrew.load(std::memory_order_acquire),
                      "busy dispatch worker should not throw while VBlank handlers fire");
            t.IsTrue(finished,
                     "VBlank interrupt handlers should run even while a guest thread is spinning");
            t.Equals(counter, 1u, "VBlank handler should publish the awaited counter value");
        });

        tc.Run("GS async callbacks keep a dedicated stack when guest heap is exhausted", [](TestCase &t)
        {
            notifyRuntimeStop();

            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            constexpr uint32_t kCallbackEntry = 0x180000u;
            constexpr uint32_t kCallerGp = 0x0036A7F0u;
            constexpr uint32_t kCallerSp = 0x00123450u;
            constexpr uint32_t kAsyncStackFloor = 0x01F00000u;

            runtime.configureGuestHeap(kAsyncStackFloor, kAsyncStackFloor);
            runtime.registerFunction(kCallbackEntry, &testRecordAsyncCallbackStack);
            ps2_stubs::resetGsSyncVCallbackState();
            gAsyncCallbackObservedSp.store(0u, std::memory_order_release);
            gAsyncCallbackObservedGp.store(0u, std::memory_order_release);

            R5900Context registerCtx{};
            registerCtx.pc = 0x00101900u;
            setRegU32(registerCtx, 4, kCallbackEntry);
            setRegU32(registerCtx, 28, kCallerGp);
            setRegU32(registerCtx, 29, kCallerSp);
            ps2_stubs::sceGsSyncVCallback(rdram.data(), &registerCtx, &runtime);

            ps2_stubs::dispatchGsSyncVCallback(rdram.data(), &runtime, 1u);

            const uint32_t observedSp = gAsyncCallbackObservedSp.load(std::memory_order_acquire);
            const uint32_t observedGp = gAsyncCallbackObservedGp.load(std::memory_order_acquire);

            t.IsTrue(observedSp != 0u, "callback should execute");
            t.Equals(observedGp, kCallerGp, "callback should preserve the registered GP");
            t.IsTrue(observedSp != kCallerSp, "callback should not reuse the registering thread stack");
            t.IsTrue(observedSp >= kAsyncStackFloor,
                     "callback should switch to the reserved async stack pool");

            runtime.requestStop();
            notifyRuntimeStop();
        });

        tc.Run("MPEG init and callback stubs return success instead of TODO errors", [](TestCase &t)
        {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState();

            R5900Context initCtx{};
            ps2_stubs::sceMpegInit(rdram.data(), &initCtx, nullptr);
            t.Equals(getRegS32(initCtx, 2), 0,
                     "sceMpegInit should succeed so games can continue through movie setup");

            R5900Context addCtx0{};
            setRegU32(addCtx0, 4, 0x00123000u);
            setRegU32(addCtx0, 5, 1u);
            setRegU32(addCtx0, 6, 0x00124000u);
            setRegU32(addCtx0, 7, 0u);
            ps2_stubs::sceMpegAddCallback(rdram.data(), &addCtx0, nullptr);
            t.Equals(getRegS32(addCtx0, 2), 1,
                     "first sceMpegAddCallback should hand back a non-error callback handle");

            R5900Context addCtx1{};
            setRegU32(addCtx1, 4, 0x00123000u);
            setRegU32(addCtx1, 5, 2u);
            setRegU32(addCtx1, 6, 0x00124010u);
            setRegU32(addCtx1, 7, 0u);
            ps2_stubs::sceMpegAddCallback(rdram.data(), &addCtx1, nullptr);
            t.Equals(getRegS32(addCtx1, 2), 2,
                     "subsequent sceMpegAddCallback calls should keep succeeding");

            R5900Context reinitCtx{};
            ps2_stubs::sceMpegInit(rdram.data(), &reinitCtx, nullptr);

            R5900Context addAfterReinit{};
            setRegU32(addAfterReinit, 4, 0x00123000u);
            setRegU32(addAfterReinit, 5, 3u);
            setRegU32(addAfterReinit, 6, 0x00124020u);
            setRegU32(addAfterReinit, 7, 0u);
            ps2_stubs::sceMpegAddCallback(rdram.data(), &addAfterReinit, nullptr);
            t.Equals(getRegS32(addAfterReinit, 2), 1,
                     "sceMpegInit should reset MPEG callback bookkeeping between runs");
        });

        tc.Run("movie startup MPEG and audio stubs return safe progress values", [](TestCase &t)
        {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::clearMpegCompatLayout();
            ps2_stubs::resetMpegStubState();
            ps2_stubs::resetAudioStubState();

            R5900Context firstIsEndCtx{};
            setRegU32(firstIsEndCtx, 4, 0x00123000u);
            ps2_stubs::sceMpegIsEnd(rdram.data(), &firstIsEndCtx, nullptr);
            t.Equals(getRegS32(firstIsEndCtx, 2), 0,
                     "sceMpegIsEnd should allow one synthetic frame before reporting end");

            R5900Context demuxCtx{};
            setRegU32(demuxCtx, 4, 0x00123000u);
            setRegU32(demuxCtx, 5, 0x00400000u);
            setRegU32(demuxCtx, 6, 0x00004000u);
            setRegU32(demuxCtx, 7, 0x00410000u);
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &demuxCtx, nullptr);
            t.Equals(getRegS32(demuxCtx, 2), 0x4000,
                     "sceMpegDemuxPssRing should consume the provided input instead of trapping");

            R5900Context getPictureCtx{};
            setRegU32(getPictureCtx, 4, 0x00123000u);
            setRegU32(getPictureCtx, 5, 0x00124000u);
            setRegU32(getPictureCtx, 6, 440u);
            ps2_stubs::sceMpegGetPicture(rdram.data(), &getPictureCtx, nullptr);
            t.Equals(Ps2FastRead32(rdram.data(), 0x00123000u + 0x00u), 320u,
                     "sceMpegGetPicture should seed a safe movie width");
            t.Equals(Ps2FastRead32(rdram.data(), 0x00123000u + 0x04u), 240u,
                     "sceMpegGetPicture should seed a safe movie height");
            t.Equals(Ps2FastRead32(rdram.data(), 0x00123000u + 0x08u), 0u,
                     "first synthetic picture should preserve frameCount==0 for guest setup");

            R5900Context secondIsEndCtx{};
            setRegU32(secondIsEndCtx, 4, 0x00123000u);
            ps2_stubs::sceMpegIsEnd(rdram.data(), &secondIsEndCtx, nullptr);
            t.Equals(getRegS32(secondIsEndCtx, 2), 0,
                     "sceMpegIsEnd should keep the decode thread alive and let the guest stop playback");

            R5900Context remoteInitCtx{};
            ps2_stubs::sceSdRemoteInit(rdram.data(), &remoteInitCtx, nullptr);
            t.Equals(getRegS32(remoteInitCtx, 2), 0,
                     "sceSdRemoteInit should succeed so Veronica can set up movie audio");

            R5900Context blockTransCtx{};
            const uint32_t blockTransSp = 0x00100000u;
            setRegU32(blockTransCtx, 29, blockTransSp);
            setRegU32(blockTransCtx, 4, 1u);
            setRegU32(blockTransCtx, 5, 0x80E0u);
            setRegU32(blockTransCtx, 6, 1u);
            setRegU32(blockTransCtx, 7, 2u);
            std::memcpy(rdram.data() + blockTransSp + 0x10u, "\x40\x23\x01\x00", 4u);
            std::memcpy(rdram.data() + blockTransSp + 0x14u, "\x00\x30\x00\x00", 4u);
            std::memcpy(rdram.data() + blockTransSp + 0x18u, "\x40\x27\x01\x00", 4u);
            ps2_stubs::sceSdRemote(rdram.data(), &blockTransCtx, nullptr);
            t.Equals(getRegU32(&blockTransCtx, 2), 0x00012340u,
                     "sceSdRemote block transfer should publish the current IOP ring position");

            R5900Context statusCtx{};
            setRegU32(statusCtx, 29, blockTransSp);
            setRegU32(statusCtx, 4, 1u);
            setRegU32(statusCtx, 5, 0x80F0u);
            setRegU32(statusCtx, 6, 1u);
            setRegU32(statusCtx, 7, 0u);
            std::memset(rdram.data() + blockTransSp + 0x10u, 0, 12u);
            ps2_stubs::sceSdRemote(rdram.data(), &statusCtx, nullptr);
            t.Equals(getRegU32(&statusCtx, 2), 0x00012340u,
                     "sceSdRemote status polling should reuse the last configured transfer base");

            R5900Context setParamCtx{};
            setRegU32(setParamCtx, 29, blockTransSp);
            setRegU32(setParamCtx, 4, 1u);
            setRegU32(setParamCtx, 5, 0x8010u);
            setRegU32(setParamCtx, 6, 0x0F81u);
            setRegU32(setParamCtx, 7, 0u);
            ps2_stubs::sceSdRemote(rdram.data(), &setParamCtx, nullptr);
            t.Equals(getRegU32(&setParamCtx, 2), 0x00012340u,
                     "sceSdRemote set-param calls should not trap or disturb the movie audio state");
        });
         
        tc.Run("IPU init skips missing optional helper instead of dispatching the default trap", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};
            ctx.pc = 0x0010B470u;

            ps2_stubs::sceIpuInit(rdram.data(), &ctx, &runtime);

            t.IsFalse(runtime.isStopRequested(),
                      "sceIpuInit should tolerate the missing optional SetD4 helper");
            t.Equals(runtime.memory().read32(0x10002010u), 0x40000000u,
                     "sceIpuInit should still program IPU_CTRL");
            t.Equals(runtime.memory().read32(0x10002000u), 0u,
                     "sceIpuInit should leave IPU_CMD reset after initialization");
        });

        tc.Run("sprintf consumes EE varargs from a2 a3 t0 and preserves width formatting", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};

            constexpr uint32_t kDestAddr = 0x00002000u;
            constexpr uint32_t kFormatAddr = 0x00002100u;
            constexpr char kFormat[] = "rm_%1d%02d%1d.rdx";

            std::memcpy(rdram.data() + kFormatAddr, kFormat, sizeof(kFormat));
            setRegU32(ctx, 4, kDestAddr);
            setRegU32(ctx, 5, kFormatAddr);
            setRegU32(ctx, 6, 0u); // a2
            setRegU32(ctx, 7, 3u); // a3
            setRegU32(ctx, 8, 1u); // t0

            ps2_stubs::sprintf(rdram.data(), &ctx, &runtime);

            const std::string rendered(reinterpret_cast<const char *>(rdram.data() + kDestAddr));
            t.Equals(rendered, std::string("rm_0031.rdx"),
                     "sprintf should read the third variadic integer from t0 and honor %02d");
            t.Equals(getRegS32(ctx, 2), static_cast<int32_t>(rendered.size()),
                     "sprintf should return the rendered length");
        });

        tc.Run("multiply-add matrix writes rd only when R5900 requires it", [](TestCase &t)
        {
            R5900Decoder decoder;
            CodeGenerator generator({}, {});

            const struct
            {
                const char *name;
                uint32_t raw;
                bool expectedRdWrite;
            } cases[] = {
                {"MULTU rd!=0", (OPCODE_SPECIAL << 26) | (2u << 21) | (3u << 16) | (11u << 11) | SPECIAL_MULTU, true},
                {"MMI MADD rd!=0", (OPCODE_MMI << 26) | (2u << 21) | (3u << 16) | (12u << 11) | MMI_MADD, true},
                {"MMI MADDU rd!=0", (OPCODE_MMI << 26) | (2u << 21) | (3u << 16) | (13u << 11) | MMI_MADDU, true},
                {"MMI MADD1 rd!=0", (OPCODE_MMI << 26) | (2u << 21) | (3u << 16) | (14u << 11) | MMI_MADD1, true},
                {"MMI MADDU1 rd!=0", (OPCODE_MMI << 26) | (2u << 21) | (3u << 16) | (15u << 11) | MMI_MADDU1, true},
                {"MMI DIVU1 rd!=0", (OPCODE_MMI << 26) | (2u << 21) | (3u << 16) | (16u << 11) | MMI_DIVU1, false},
            };

            for (size_t i = 0; i < std::size(cases); ++i)
            {
                const Instruction inst = decoder.decodeInstruction(0x2000u + static_cast<uint32_t>(i * 4u), cases[i].raw);
                const std::string generated = generator.translateInstruction(inst);
                const bool emittedRdWrite = hasSignedRdWrite(generated, inst.rd);

                t.Equals(inst.modificationInfo.modifiesGPR, cases[i].expectedRdWrite,
                         std::string("decoder rd-write metadata mismatch for ") + cases[i].name);
                t.Equals(emittedRdWrite, cases[i].expectedRdWrite,
                         std::string("codegen rd-write mismatch for ") + cases[i].name);
            }
        });

        tc.Run("SignalException marks EPC and BD for delay-slot exceptions", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};

            ctx.pc = 0x2000u;
            ctx.branch_pc = 0x1FFCu;
            ctx.in_delay_slot = true;
            ctx.cop0_status = 0u;
            ctx.cop0_cause = 0u;

            runtime.SignalException(&ctx, EXCEPTION_ADDRESS_ERROR_LOAD);

            t.Equals(ctx.cop0_epc, 0x1FFCu, "delay-slot exception should capture branch_pc in EPC");
            t.IsTrue((ctx.cop0_cause & COP0_CAUSE_BD) != 0u, "delay-slot exception should set CAUSE.BD");
            t.Equals(ctx.cop0_cause & COP0_CAUSE_EXCCODE_MASK,
                     (static_cast<uint32_t>(EXCEPTION_ADDRESS_ERROR_LOAD) << 2) & COP0_CAUSE_EXCCODE_MASK,
                     "CAUSE.EXCCODE should match exception");
            t.IsTrue((ctx.cop0_status & COP0_STATUS_EXL) != 0u, "exception should set STATUS.EXL");
            t.Equals(ctx.pc, EXCEPTION_VECTOR_GENERAL, "exception should jump to general vector when BEV=0");
            t.IsFalse(ctx.in_delay_slot, "exception delivery should clear delay-slot state");
        });

        tc.Run("SignalException uses current pc without BD and honors BEV vector", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};

            ctx.pc = 0x3000u;
            ctx.in_delay_slot = false;
            ctx.cop0_status = COP0_STATUS_BEV;
            ctx.cop0_cause = COP0_CAUSE_BD;

            runtime.SignalException(&ctx, EXCEPTION_ADDRESS_ERROR_STORE);

            t.Equals(ctx.cop0_epc, 0x3000u, "non-delay exception should capture current pc in EPC");
            t.IsTrue((ctx.cop0_cause & COP0_CAUSE_BD) == 0u, "non-delay exception should clear CAUSE.BD");
            t.Equals(ctx.pc, EXCEPTION_VECTOR_BOOT, "BEV=1 should route exception to boot vector");
        });

        tc.Run("handleSyscall rejects invocation in delay slot", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};
            ctx.in_delay_slot = true;

            bool threw = false;
            try
            {
                runtime.handleSyscall(rdram.data(), &ctx, 0x3Cu);
            }
            catch (const std::runtime_error &)
            {
                threw = true;
            }

            t.IsTrue(threw, "syscall from delay slot should throw to preserve block atomicity");
        });

        tc.Run("VIF MSCAL and MSCNT toggle DBF and keep TOPS/ITOPS coherent", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            mem.vif1_regs.base = 4u;
            mem.vif1_regs.ofst = 2u;
            mem.vif1_regs.itop = 0x21u;
            mem.vif1_regs.stat &= ~(1u << 7); // DBF = 0

            uint32_t callbackPc = 0xFFFFFFFFu;
            uint32_t callbackItop = 0xFFFFFFFFu;
            uint32_t callbackCount = 0u;
            mem.setVu1MscalCallback([&](uint32_t startPC, uint32_t itop)
            {
                callbackPc = startPC;
                callbackItop = itop;
                callbackCount++;
            });

            const uint32_t mscal = makeVifCmd(0x14u, 0u, 3u); // start PC = 3 * 8
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&mscal), sizeof(mscal));

            t.Equals(callbackCount, 1u, "MSCAL should invoke VU1 callback exactly once");
            t.Equals(callbackPc, 24u, "MSCAL should pass startPC=imm*8");
            t.Equals(callbackItop, 0x21u, "MSCAL callback should receive current ITOP");
            t.Equals(mem.vif1_regs.itops, 0x21u, "MSCAL should latch ITOPS from ITOP");
            t.IsTrue((mem.vif1_regs.stat & (1u << 7)) != 0u, "MSCAL should toggle DBF on");
            t.Equals(mem.vif1_regs.tops, 6u, "DBF=1 should make TOPS=BASE+OFST");

            const uint32_t mscnt = makeVifCmd(0x17u, 0u, 0u);
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&mscnt), sizeof(mscnt));

            t.Equals(callbackCount, 1u, "MSCNT should not invoke MSCAL callback");
            t.IsTrue((mem.vif1_regs.stat & (1u << 7)) == 0u, "MSCNT should toggle DBF back off");
            t.Equals(mem.vif1_regs.tops, 4u, "DBF=0 should make TOPS=BASE");
            t.Equals(mem.vif1_regs.itops, 0x21u, "MSCNT should refresh ITOPS from ITOP");
        });

        tc.Run("GS sprite draw applies XYOFFSET and fully-outside scissor should not render", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            const uint64_t frame1 =
                (0ull << 0) |   // FBP
                (1ull << 16) |  // FBW
                (0ull << 24) |  // PSM CT32
                (0ull << 32);   // FBMSK
            gs.writeRegister(GS_REG_FRAME_1, frame1);

            // XYOFFSET=1,1 pixels (16.4 fixed point).
            const uint64_t xyoffset = (16ull) | (16ull << 32);
            gs.writeRegister(GS_REG_XYOFFSET_1, xyoffset);

            // Scissor initially includes pixel (1,1).
            const uint64_t scissorInside = (0ull) | (3ull << 16) | (0ull << 32) | (3ull << 48);
            gs.writeRegister(GS_REG_SCISSOR_1, scissorInside);

            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_SPRITE));
            gs.writeRegister(GS_REG_RGBAQ, 0xFF3214C8ull); // RGBA=(200,20,50,255)

            // With XYOFFSET=(1,1), vertex at (2,2) draws to pixel (1,1).
            const uint64_t xyz = (32ull) | (32ull << 16) | (0ull << 32);
            gs.writeRegister(GS_REG_XYZ2, xyz);
            gs.writeRegister(GS_REG_XYZ2, xyz);

            const uint32_t insideOff = frameOffsetBytes(1u, 1u, 1u);
            t.Equals(vram[insideOff + 0u], static_cast<uint8_t>(200u), "inside draw should write R");
            t.Equals(vram[insideOff + 1u], static_cast<uint8_t>(20u), "inside draw should write G");
            t.Equals(vram[insideOff + 2u], static_cast<uint8_t>(50u), "inside draw should write B");
            t.Equals(vram[insideOff + 3u], static_cast<uint8_t>(255u), "inside draw should write A");

            std::memset(vram.data(), 0, 1024u);

            // Move scissor so target pixel is fully outside.
            const uint64_t scissorOutside = (3ull) | (4ull << 16) | (3ull << 32) | (4ull << 48);
            gs.writeRegister(GS_REG_SCISSOR_1, scissorOutside);
            gs.writeRegister(GS_REG_XYZ2, xyz);
            gs.writeRegister(GS_REG_XYZ2, xyz);

            bool anyWrite = false;
            for (size_t i = 0; i < 1024u; ++i)
            {
                if (vram[i] != 0u)
                {
                    anyWrite = true;
                    break;
                }
            }
            t.IsFalse(anyWrite, "fully-outside sprite should not render any pixel");
        });

        tc.Run("GS alpha blend uses ALPHA register FIX factor", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            const uint64_t frame1 =
                (0ull << 0) |   // FBP
                (1ull << 16) |  // FBW
                (0ull << 24) |  // PSM CT32
                (0ull << 32);   // FBMSK
            gs.writeRegister(GS_REG_FRAME_1, frame1);
            gs.writeRegister(GS_REG_SCISSOR_1, (0ull) | (4ull << 16) | (0ull << 32) | (4ull << 48));
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);

            const uint32_t pxOff = frameOffsetBytes(1u, 1u, 1u);
            vram[pxOff + 0u] = 40u;
            vram[pxOff + 1u] = 40u;
            vram[pxOff + 2u] = 40u;
            vram[pxOff + 3u] = 255u;

            // ABE on sprite prim.
            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_SPRITE) | (1ull << 6));

            // ALPHA: (A-B)*FIX/128 + D
            // A=Cs(0), B=Cd(1), C=FIX(2), D=Cd(1), FIX=64.
            const uint64_t alpha = (0ull << 0) | (1ull << 2) | (2ull << 4) | (1ull << 6) | (64ull << 32);
            gs.writeRegister(GS_REG_ALPHA_1, alpha);
            gs.writeRegister(GS_REG_RGBAQ, 0xFFC8C8C8ull); // src RGB = 200

            const uint64_t xyz = (16ull) | (16ull << 16) | (0ull << 32); // pixel (1,1)
            gs.writeRegister(GS_REG_XYZ2, xyz);
            gs.writeRegister(GS_REG_XYZ2, xyz);

            // ((200 - 40) * 64 >> 7) + 40 = 120
            t.Equals(vram[pxOff + 0u], static_cast<uint8_t>(120u), "alpha blend should update R with FIX factor");
            t.Equals(vram[pxOff + 1u], static_cast<uint8_t>(120u), "alpha blend should update G with FIX factor");
            t.Equals(vram[pxOff + 2u], static_cast<uint8_t>(120u), "alpha blend should update B with FIX factor");
        });

        tc.Run("notifyRuntimeStop joins guest worker threads before teardown", [](TestCase &t)
        {
            notifyRuntimeStop();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            constexpr uint32_t kEntry = 0x250000u;
            constexpr uint32_t kThreadParamAddr = 0x2600u;
            const uint32_t threadParam[7] = {
                0u,          // attr
                kEntry,      // entry
                0x00100000u, // stack
                0x00000400u, // stack size
                0x00110000u, // gp
                8u,          // priority
                0u           // option
            };

            runtime.registerFunction(kEntry, &testRuntimeWorkerLoop);
            std::memcpy(rdram.data() + kThreadParamAddr, threadParam, sizeof(threadParam));

            R5900Context createCtx{};
            setRegU32(createCtx, 4, kThreadParamAddr);
            CreateThread(rdram.data(), &createCtx, &runtime);
            const int32_t tid = getRegS32(createCtx, 2);
            t.IsTrue(tid > 0, "CreateThread should succeed for teardown-join test");

            R5900Context startCtx{};
            setRegU32(startCtx, 4, static_cast<uint32_t>(tid));
            setRegU32(startCtx, 5, 0u);
            StartThread(rdram.data(), &startCtx, &runtime);
            t.Equals(getRegS32(startCtx, 2), KE_OK, "StartThread should launch worker");

            const bool started = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) > 0;
            }, std::chrono::milliseconds(500));
            t.IsTrue(started, "worker thread should become active");

            runtime.requestStop();
            const bool drained = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) == 0;
            }, std::chrono::milliseconds(2000));
            t.IsTrue(drained, "requestStop should drain all guest worker threads");

            notifyRuntimeStop();
        });

        tc.Run("Semaphore poll/signal remains stable under host-thread contention", [](TestCase &t)
        {
            notifyRuntimeStop();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            constexpr uint32_t kParamAddr = 0x2000u;
            const uint32_t semaParam[6] = {
                0u, // count
                1u, // max_count
                1u, // init_count
                0u, // wait_threads
                0u, // attr
                0u  // option
            };
            std::memcpy(rdram.data() + kParamAddr, semaParam, sizeof(semaParam));

            R5900Context createCtx{};
            setRegU32(createCtx, 4, kParamAddr);
            CreateSema(rdram.data(), &createCtx, &runtime);
            const int32_t sid = getRegS32(createCtx, 2);
            t.IsTrue(sid > 0, "CreateSema should return a valid sid");

            std::atomic<int32_t> pollOkCount{0};
            std::atomic<int32_t> signalOkCount{0};
            std::atomic<bool> pollerThrew{false};
            std::atomic<bool> signalerThrew{false};

            std::thread poller([&]()
            {
                try
                {
                    for (int i = 0; i < 64; ++i)
                    {
                        R5900Context pollCtx{};
                        setRegU32(pollCtx, 4, static_cast<uint32_t>(sid));
                        PollSema(rdram.data(), &pollCtx, &runtime);
                        if (getRegS32(pollCtx, 2) == KE_OK)
                        {
                            pollOkCount.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
                catch (...)
                {
                    pollerThrew.store(true, std::memory_order_release);
                }
            });

            std::thread signaler([&]()
            {
                try
                {
                    for (int i = 0; i < 64; ++i)
                    {
                        R5900Context signalCtx{};
                        setRegU32(signalCtx, 4, static_cast<uint32_t>(sid));
                        SignalSema(rdram.data(), &signalCtx, &runtime);
                        if (getRegS32(signalCtx, 2) == KE_OK)
                        {
                            signalOkCount.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
                catch (...)
                {
                    signalerThrew.store(true, std::memory_order_release);
                }
            });

            if (poller.joinable())
            {
                poller.join();
            }
            if (signaler.joinable())
            {
                signaler.join();
            }

            t.IsFalse(pollerThrew.load(std::memory_order_acquire),
                      "PollSema worker thread should not throw");
            t.IsFalse(signalerThrew.load(std::memory_order_acquire),
                      "SignalSema worker thread should not throw");
            t.IsTrue(pollOkCount.load(std::memory_order_relaxed) > 0,
                     "contended PollSema should observe at least one successful acquire");
            t.IsTrue(signalOkCount.load(std::memory_order_relaxed) > 0,
                     "contended SignalSema should observe successful releases");

            constexpr uint32_t kStatusAddr = 0x2100u;
            R5900Context referCtx{};
            setRegU32(referCtx, 4, static_cast<uint32_t>(sid));
            setRegU32(referCtx, 5, kStatusAddr);
            ReferSemaStatus(rdram.data(), &referCtx, &runtime);
            t.Equals(getRegS32(referCtx, 2), KE_OK, "ReferSemaStatus should succeed after contention");

            int32_t finalCount = 0;
            std::memcpy(&finalCount, rdram.data() + kStatusAddr + 0u, sizeof(finalCount));
            t.IsTrue(finalCount >= 0 && finalCount <= 1, "semaphore count should remain within [0, max_count]");

            runtime.requestStop();
            notifyRuntimeStop();
        });

        tc.Run("WaitEventFlag AND-mode is stable under concurrent setters", [](TestCase &t)
        {
            notifyRuntimeStop();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            constexpr uint32_t kEventParamAddr = 0x2400u;
            constexpr uint32_t kResBitsAddr = 0x2410u;
            const uint32_t eventParam[3] = {0u, 0u, 0u};
            std::memcpy(rdram.data() + kEventParamAddr, eventParam, sizeof(eventParam));

            R5900Context createCtx{};
            setRegU32(createCtx, 4, kEventParamAddr);
            CreateEventFlag(rdram.data(), &createCtx, &runtime);
            const int32_t eid = getRegS32(createCtx, 2);
            t.IsTrue(eid > 0, "CreateEventFlag should return a valid id");

            std::atomic<bool> waiterDone{false};
            std::atomic<int32_t> waiterRet{-9999};
            std::atomic<uint32_t> waiterBits{0u};
            std::atomic<bool> waiterThrew{false};
            std::atomic<bool> setterAThrew{false};
            std::atomic<bool> setterBThrew{false};

            std::thread waiter([&]()
            {
                try
                {
                    R5900Context waitCtx{};
                    setRegU32(waitCtx, 4, static_cast<uint32_t>(eid));
                    setRegU32(waitCtx, 5, 0x3u); // wait for bit0 and bit1 (AND mode)
                    setRegU32(waitCtx, 6, 0u);   // AND, no clear
                    setRegU32(waitCtx, 7, kResBitsAddr);
                    WaitEventFlag(rdram.data(), &waitCtx, &runtime);
                    waiterRet.store(getRegS32(waitCtx, 2), std::memory_order_relaxed);
                    uint32_t bits = 0u;
                    std::memcpy(&bits, rdram.data() + kResBitsAddr, sizeof(bits));
                    waiterBits.store(bits, std::memory_order_relaxed);
                }
                catch (...)
                {
                    waiterThrew.store(true, std::memory_order_release);
                }
                waiterDone.store(true, std::memory_order_release);
            });

            std::thread setterA([&]()
            {
                try
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    R5900Context setCtx{};
                    setRegU32(setCtx, 4, static_cast<uint32_t>(eid));
                    setRegU32(setCtx, 5, 0x1u);
                    SetEventFlag(rdram.data(), &setCtx, &runtime);
                }
                catch (...)
                {
                    setterAThrew.store(true, std::memory_order_release);
                }
            });

            std::thread setterB([&]()
            {
                try
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(15));
                    R5900Context setCtx{};
                    setRegU32(setCtx, 4, static_cast<uint32_t>(eid));
                    setRegU32(setCtx, 5, 0x2u);
                    SetEventFlag(rdram.data(), &setCtx, &runtime);
                }
                catch (...)
                {
                    setterBThrew.store(true, std::memory_order_release);
                }
            });

            const bool woke = waitUntil([&]()
            {
                return waiterDone.load(std::memory_order_acquire);
            }, std::chrono::milliseconds(500));

            if (setterA.joinable())
            {
                setterA.join();
            }
            if (setterB.joinable())
            {
                setterB.join();
            }
            if (waiter.joinable())
            {
                waiter.join();
            }

            t.IsFalse(waiterThrew.load(std::memory_order_acquire),
                      "WaitEventFlag waiter thread should not throw");
            t.IsFalse(setterAThrew.load(std::memory_order_acquire),
                      "SetEventFlag setterA thread should not throw");
            t.IsFalse(setterBThrew.load(std::memory_order_acquire),
                      "SetEventFlag setterB thread should not throw");
            t.IsTrue(woke, "WaitEventFlag AND waiter should wake after both bits are published");
            t.Equals(waiterRet.load(std::memory_order_relaxed), KE_OK, "WaitEventFlag should return KE_OK");
            t.IsTrue((waiterBits.load(std::memory_order_relaxed) & 0x3u) == 0x3u,
                     "WaitEventFlag result bits should include both concurrently-set bits");

            R5900Context deleteCtx{};
            setRegU32(deleteCtx, 4, static_cast<uint32_t>(eid));
            DeleteEventFlag(rdram.data(), &deleteCtx, &runtime);
            runtime.requestStop();
            notifyRuntimeStop();
        });

        tc.Run("sceVu0ApplyMatrix uses libvux matrix math with the imported EE ABI", [](TestCase &t)
        {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};

            constexpr uint32_t kOutAddr = 0x00100000u;
            constexpr uint32_t kMatrixAddr = 0x00100040u;
            constexpr uint32_t kSrcAddr = 0x00100080u;

            const float matrix[16] = {
                1.0f, 2.0f, 3.0f, 4.0f,
                5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f,
                13.0f, 14.0f, 15.0f, 16.0f,
            };
            const float src[4] = {1.0f, 2.0f, 3.0f, 1.0f};
            std::memcpy(rdram.data() + kMatrixAddr, matrix, sizeof(matrix));
            std::memcpy(rdram.data() + kSrcAddr, src, sizeof(src));

            setRegU32(ctx, 4, kOutAddr);
            setRegU32(ctx, 5, kMatrixAddr);
            setRegU32(ctx, 6, kSrcAddr);

            ps2_stubs::sceVu0ApplyMatrix(rdram.data(), &ctx, nullptr);

            float out[4]{};
            std::memcpy(out, rdram.data() + kOutAddr, sizeof(out));
            t.Equals(out[0], 51.0f, "sceVu0ApplyMatrix should compute X with libvux layout");
            t.Equals(out[1], 58.0f, "sceVu0ApplyMatrix should compute Y with libvux layout");
            t.Equals(out[2], 65.0f, "sceVu0ApplyMatrix should compute Z with libvux layout");
            t.Equals(out[3], 72.0f, "sceVu0ApplyMatrix should compute W with libvux layout");
            t.Equals(getRegS32(ctx, 2), 0, "sceVu0ApplyMatrix should report success");
        });

        tc.Run("sceVu0TransposeMatrix transposes a 4x4 matrix with dst/src ABI", [](TestCase &t)
        {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};

            constexpr uint32_t kDstAddr = 0x00100100u;
            constexpr uint32_t kSrcAddr = 0x00100140u;
            const float src[16] = {
                1.0f, 2.0f, 3.0f, 4.0f,
                5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f,
                13.0f, 14.0f, 15.0f, 16.0f,
            };
            std::memcpy(rdram.data() + kSrcAddr, src, sizeof(src));

            setRegU32(ctx, 4, kDstAddr);
            setRegU32(ctx, 5, kSrcAddr);

            ps2_stubs::sceVu0TransposeMatrix(rdram.data(), &ctx, nullptr);

            float out[16]{};
            std::memcpy(out, rdram.data() + kDstAddr, sizeof(out));
            t.Equals(out[0], 1.0f, "transpose should preserve [0][0]");
            t.Equals(out[1], 5.0f, "transpose should swap row 0 col 1");
            t.Equals(out[2], 9.0f, "transpose should swap row 0 col 2");
            t.Equals(out[3], 13.0f, "transpose should swap row 0 col 3");
            t.Equals(out[4], 2.0f, "transpose should swap row 1 col 0");
            t.Equals(out[5], 6.0f, "transpose should preserve [1][1]");
            t.Equals(out[10], 11.0f, "transpose should preserve [2][2]");
            t.Equals(out[12], 4.0f, "transpose should swap row 3 col 0");
            t.Equals(out[15], 16.0f, "transpose should preserve [3][3]");
            t.Equals(getRegS32(ctx, 2), 0, "sceVu0TransposeMatrix should report success");
        });

        tc.Run("sceVif1PkReset preserves the packet base pointer and clears open tag state", [](TestCase &t)
        {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};

            constexpr uint32_t kStateAddr = 0x00100200u;
            constexpr uint32_t kBaseAddr = 0x00101000u;

            setRegU32(ctx, 4, kStateAddr);
            setRegU32(ctx, 5, kBaseAddr);
            ps2_stubs::sceVif1PkInit(rdram.data(), &ctx, nullptr);

            const uint32_t dirtyCurrent = kBaseAddr + 0x40u;
            const uint32_t dirtyPending = 0x12345678u;
            const uint32_t dirtyDirectOpen = 0x00ABCDEFu;
            const uint32_t dirtyGifOpen = 0x00112233u;
            std::memcpy(rdram.data() + kStateAddr + 0u, &dirtyCurrent, sizeof(dirtyCurrent));
            std::memcpy(rdram.data() + kStateAddr + 8u, &dirtyPending, sizeof(dirtyPending));
            std::memcpy(rdram.data() + kStateAddr + 12u, &dirtyDirectOpen, sizeof(dirtyDirectOpen));
            std::memcpy(rdram.data() + kStateAddr + 20u, &dirtyGifOpen, sizeof(dirtyGifOpen));

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kStateAddr);
            ps2_stubs::sceVif1PkReset(rdram.data(), &ctx, nullptr);

            uint32_t current = 0u;
            uint32_t base = 0u;
            uint32_t pending = 0u;
            uint32_t directOpen = 0u;
            uint32_t gifOpen = 0u;
            std::memcpy(&current, rdram.data() + kStateAddr + 0u, sizeof(current));
            std::memcpy(&base, rdram.data() + kStateAddr + 4u, sizeof(base));
            std::memcpy(&pending, rdram.data() + kStateAddr + 8u, sizeof(pending));
            std::memcpy(&directOpen, rdram.data() + kStateAddr + 12u, sizeof(directOpen));
            std::memcpy(&gifOpen, rdram.data() + kStateAddr + 20u, sizeof(gifOpen));

            t.Equals(current, kBaseAddr, "sceVif1PkReset should restore current pointer to the packet base");
            t.Equals(base, kBaseAddr, "sceVif1PkReset should preserve the packet base pointer");
            t.Equals(pending, 0u, "sceVif1PkReset should clear pending count tracking");
            t.Equals(directOpen, 0u, "sceVif1PkReset should clear direct-code open state");
            t.Equals(gifOpen, 0u, "sceVif1PkReset should clear GIF-tag open state");
            t.Equals(::getRegU32(&ctx, 2), kBaseAddr, "sceVif1PkReset should return the packet base pointer");
        });

        tc.Run("sceVif1PkCloseDirectCode encodes DIRECT length in qwords", [](TestCase &t)
        {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};

            constexpr uint32_t kStateAddr = 0x00100400u;
            constexpr uint32_t kBaseAddr = 0x00102000u;

            setRegU32(ctx, 4, kStateAddr);
            setRegU32(ctx, 5, kBaseAddr);
            ps2_stubs::sceVif1PkInit(rdram.data(), &ctx, nullptr);

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kStateAddr);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::sceVif1PkCnt(rdram.data(), &ctx, nullptr);

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kStateAddr);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::sceVif1PkOpenDirectCode(rdram.data(), &ctx, nullptr);

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kStateAddr);
            setRegU32(ctx, 5, 4u); // reserve one qword worth of GIF payload
            ps2_stubs::sceVif1PkReserve(rdram.data(), &ctx, nullptr);

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kStateAddr);
            ps2_stubs::sceVif1PkCloseDirectCode(rdram.data(), &ctx, nullptr);

            uint32_t directCmd = 0u;
            std::memcpy(&directCmd, rdram.data() + kBaseAddr + 12u, sizeof(directCmd));
            t.Equals(directCmd, 0x50000001u, "sceVif1PkCloseDirectCode should store a 1-QW DIRECT length");
        });
    });
}

// ---------------------------------------------------------------------------
// Scheduler tests
//
// End-to-end blackbox tests: they run real guest fibers on the executor thread
// and verify outcomes through public syscall calls and atomic counters.
// scheduler_init() and scheduler_shutdown() bracket each test so the scheduler is
// freshly started and cleanly stopped each time.
// ---------------------------------------------------------------------------

void register_scheduler_tests()
{
    MiniTest::Case("Scheduler", [](TestCase &tc)
    {
        // A guest fiber blocking in WaitSema is woken by SignalSema from a
        // host thread, demonstrating the full cooperative block/wake cycle.
        // The worker fiber blocks on sid (count=0); the host signals sid and then
        // waits on done_sid (signalled by the worker after WaitSema returns).
        tc.Run("WaitSema blocks guest fiber and SignalSema from host wakes it", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();

            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00300000u, &schedWorkerWaitAndSignal);

            // Semaphore the worker will block on (initial count=0).
            const int32_t sid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(sid > 0, "WaitSema test: work sema should be created");

            // Done semaphore: worker signals this after WaitSema returns (maxCount=1).
            const int32_t doneSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(doneSid > 0, "WaitSema test: done sema should be created");

            if (sid <= 0 || doneSid <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, sid);
                deleteSchedSema(rdram.data(), &runtime, doneSid);
                ps2sched::scheduler_shutdown();
                runtime.requestStop();
                notifyRuntimeStop();
                return;
            }

            std::memcpy(rdram.data() + kSchedTestSemaIdAddr, &sid,     sizeof(sid));
            std::memcpy(rdram.data() + kSchedDoneSemaIdAddr, &doneSid, sizeof(doneSid));
            gSchedSeqCounter.store(0, std::memory_order_relaxed);

            // Start a worker fiber at priority 10.  Pool threads will pick it up and
            // it will call WaitSema(sid, count=0) → cooperative block.
            const int32_t workerTid = startSchedWorker(
                rdram.data(), &runtime,
                0x00300000u, 10,
                0x00400000u, 0x800u);
            t.IsTrue(workerTid > 0, "WaitSema test: worker StartThread should succeed");

            if (workerTid <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, sid);
                deleteSchedSema(rdram.data(), &runtime, doneSid);
                ps2sched::scheduler_shutdown();
                runtime.requestStop();
                notifyRuntimeStop();
                return;
            }

            // Wait until the worker blocks on sid (waitThreads == 1).
            const bool workerBlocked = waitUntil([&]()
            {
                constexpr uint32_t kStatusAddr = 0x2C00u;
                R5900Context refCtx{};
                setRegU32(refCtx, 4, static_cast<uint32_t>(sid));
                setRegU32(refCtx, 5, kStatusAddr);
                ps2_syscalls::ReferSemaStatus(rdram.data(), &refCtx, &runtime);
                int32_t waitThreads = 0;
                std::memcpy(&waitThreads, rdram.data() + kStatusAddr + 12u, sizeof(waitThreads));
                return waitThreads >= 1;
            }, std::chrono::milliseconds(500));

            t.IsTrue(workerBlocked, "WaitSema test: worker fiber should block on sid within 500ms");

            // Signal sid from the host thread — the pool will pick up the worker fiber.
            {
                R5900Context sigCtx{};
                setRegU32(sigCtx, 4, static_cast<uint32_t>(sid));
                ps2_syscalls::SignalSema(rdram.data(), &sigCtx, &runtime);
                t.Equals(getRegS32(sigCtx, 2), KE_OK, "WaitSema test: SignalSema(sid) should return KE_OK");
            }

            // Wait until the worker signals done_sid (i.e., its WaitSema returned).
            const bool workerDone = waitUntil([&]()
            {
                constexpr uint32_t kStatusAddr = 0x2C80u;
                R5900Context refCtx{};
                setRegU32(refCtx, 4, static_cast<uint32_t>(doneSid));
                setRegU32(refCtx, 5, kStatusAddr);
                ps2_syscalls::ReferSemaStatus(rdram.data(), &refCtx, &runtime);
                int32_t count = 0;
                std::memcpy(&count, rdram.data() + kStatusAddr + 8u, sizeof(count)); // init_count field
                return count >= 1;
            }, std::chrono::milliseconds(1000));

            t.IsTrue(workerDone, "WaitSema test: worker should signal done_sid after WaitSema(sid) returns");

            // Verify the worker's WaitSema(sid) returned KE_OK.
            {
                int32_t workerResult = -9999;
                std::memcpy(&workerResult, rdram.data() + kSchedResultBase, sizeof(workerResult));
                t.Equals(workerResult, KE_OK, "WaitSema(sid) in worker fiber should return KE_OK after SignalSema");
            }

            // Wait for the worker fiber to finish.
            const bool finished = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) <= 0;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(finished, "WaitSema test: worker fiber should finish within 1s");

            deleteSchedSema(rdram.data(), &runtime, sid);
            deleteSchedSema(rdram.data(), &runtime, doneSid);
            ps2sched::scheduler_shutdown();
            runtime.requestStop();
            notifyRuntimeStop();
        });

        // TerminateThread of a guest fiber blocked in WaitSema.
        // Verifies that request_terminate wakes the blocked fiber, it propagates
        // ThreadExitException, and g_activeThreads decrements.
        tc.Run("TerminateThread wakes a WaitSema-blocked fiber and cleans up", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();

            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00310000u, &schedWorkerBlockForever);

            // Semaphore the worker will block on indefinitely (count=0).
            const int32_t blockSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(blockSid > 0, "TerminateThread test: block sema should be created");

            if (blockSid <= 0)
            {
                ps2sched::scheduler_shutdown();
                runtime.requestStop();
                notifyRuntimeStop();
                return;
            }

            std::memcpy(rdram.data() + kSchedTestSemaIdAddr, &blockSid, sizeof(blockSid));

            const int32_t workerTid = startSchedWorker(
                rdram.data(), &runtime,
                0x00310000u, 10,
                0x00500000u, 0x800u);
            t.IsTrue(workerTid > 0, "TerminateThread test: worker fiber should start");

            if (workerTid <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, blockSid);
                ps2sched::scheduler_shutdown();
                runtime.requestStop();
                notifyRuntimeStop();
                return;
            }

            // Wait until the worker fiber is blocked on blockSid.
            const bool workerBlocked = waitUntil([&]()
            {
                constexpr uint32_t kStatusAddr = 0x2C00u;
                R5900Context refCtx{};
                setRegU32(refCtx, 4, static_cast<uint32_t>(blockSid));
                setRegU32(refCtx, 5, kStatusAddr);
                ps2_syscalls::ReferSemaStatus(rdram.data(), &refCtx, &runtime);
                int32_t waitThreads = 0;
                std::memcpy(&waitThreads, rdram.data() + kStatusAddr + 12u, sizeof(waitThreads));
                return waitThreads >= 1;
            }, std::chrono::milliseconds(500));

            t.IsTrue(workerBlocked, "TerminateThread test: worker should block on sema within 500ms");

            const int activeBeforeTerminate = g_activeThreads.load(std::memory_order_acquire);

            // Terminate the worker fiber.
            {
                R5900Context termCtx{};
                setRegU32(termCtx, 4, static_cast<uint32_t>(workerTid));
                ps2_syscalls::TerminateThread(rdram.data(), &termCtx, &runtime);
                t.Equals(getRegS32(termCtx, 2), KE_OK, "TerminateThread should return KE_OK");
            }

            // g_activeThreads should decrement as the fiber unwinds.
            const bool drained = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) < activeBeforeTerminate;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(drained, "TerminateThread: g_activeThreads should decrement after fiber exits");

            deleteSchedSema(rdram.data(), &runtime, blockSid);
            ps2sched::scheduler_shutdown();
            runtime.requestStop();
            notifyRuntimeStop();
        });

        // DeleteSema while N guest fibers are blocked in WaitSema.
        // Each waiter must receive KE_WAIT_DELETE; the sema id must be invalidated.
        // The fibers signal done_sid after receiving KE_WAIT_DELETE so we can wait
        // for all N completions from the host side.
        tc.Run("DeleteSema wakes all blocked fibers with KE_WAIT_DELETE", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();

            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00320000u, &schedWorkerWaitAndSignal);

            // Semaphore workers block on (count=0).
            const int32_t sid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(sid > 0, "DeleteSema test: block sema should be created");

            // Done semaphore: each worker signals it after WaitSema returns.
            // maxCount=3 so all 3 signals can accumulate.
            constexpr int kNumWorkers = 3;
            const int32_t doneSid = createSchedSema(rdram.data(), &runtime, 0, kNumWorkers);
            t.IsTrue(doneSid > 0, "DeleteSema test: done sema should be created");

            if (sid <= 0 || doneSid <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, sid);
                deleteSchedSema(rdram.data(), &runtime, doneSid);
                ps2sched::scheduler_shutdown();
                runtime.requestStop();
                notifyRuntimeStop();
                return;
            }

            std::memcpy(rdram.data() + kSchedTestSemaIdAddr, &sid,     sizeof(sid));
            std::memcpy(rdram.data() + kSchedDoneSemaIdAddr, &doneSid, sizeof(doneSid));
            gSchedSeqCounter.store(0, std::memory_order_relaxed);

            constexpr uint32_t kWorkerStackBase = 0x00600000u;
            constexpr uint32_t kWorkerStackSize = 0x800u;

            int32_t workerTids[kNumWorkers] = {-1, -1, -1};
            for (int i = 0; i < kNumWorkers; ++i)
            {
                const uint32_t stackAddr =
                    kWorkerStackBase + static_cast<uint32_t>(i) * kWorkerStackSize * 2u;
                workerTids[i] = startSchedWorker(
                    rdram.data(), &runtime,
                    0x00320000u, 10,
                    stackAddr, kWorkerStackSize);
                t.IsTrue(workerTids[i] > 0,
                         std::string("DeleteSema test: worker ") + std::to_string(i) + " should start");
            }

            // Wait until all N workers are blocked on sid (waitThreads == N).
            const bool allBlocked = waitUntil([&]()
            {
                constexpr uint32_t kStatusAddr = 0x2D00u;
                R5900Context refCtx{};
                setRegU32(refCtx, 4, static_cast<uint32_t>(sid));
                setRegU32(refCtx, 5, kStatusAddr);
                ps2_syscalls::ReferSemaStatus(rdram.data(), &refCtx, &runtime);
                int32_t waiters = 0;
                std::memcpy(&waiters, rdram.data() + kStatusAddr + 12u, sizeof(waiters));
                return waiters >= kNumWorkers;
            }, std::chrono::milliseconds(1000));

            t.IsTrue(allBlocked, "DeleteSema test: all workers should block on sid within 1s");

            // DeleteSema: marks deleted, wakes all N blocked fibers via make_ready.
            {
                R5900Context delCtx{};
                setRegU32(delCtx, 4, static_cast<uint32_t>(sid));
                ps2_syscalls::DeleteSema(rdram.data(), &delCtx, &runtime);
                t.Equals(getRegS32(delCtx, 2), KE_OK, "DeleteSema should return KE_OK");
            }

            // Wait until all N workers have signalled done_sid (count == N).
            const bool allDone = waitUntil([&]()
            {
                return gSchedSeqCounter.load(std::memory_order_relaxed) >= kNumWorkers;
            }, std::chrono::milliseconds(2000));

            t.IsTrue(allDone, "DeleteSema test: all worker fibers should complete within 2s");

            // Verify all workers received KE_WAIT_DELETE.
            for (int i = 0; i < kNumWorkers; ++i)
            {
                int32_t result = 0;
                std::memcpy(&result,
                            rdram.data() + kSchedResultBase + static_cast<uint32_t>(i * 4),
                            sizeof(result));
                t.Equals(result, KE_WAIT_DELETE,
                         std::string("DeleteSema: worker ") + std::to_string(i) +
                         " WaitSema should return KE_WAIT_DELETE");
            }

            // PollSema on the deleted id should return KE_UNKNOWN_SEMID.
            {
                R5900Context laterCtx{};
                setRegU32(laterCtx, 4, static_cast<uint32_t>(sid));
                ps2_syscalls::PollSema(rdram.data(), &laterCtx, &runtime);
                t.Equals(getRegS32(laterCtx, 2), KE_UNKNOWN_SEMID,
                         "PollSema on a deleted sema id should return KE_UNKNOWN_SEMID");
            }

            // Wait for all fibers to finish and clean up.
            const bool finished = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) <= 0;
            }, std::chrono::milliseconds(2000));
            t.IsTrue(finished, "DeleteSema test: all fibers should finish within 2s");

            deleteSchedSema(rdram.data(), &runtime, doneSid);
            ps2sched::scheduler_shutdown();
            runtime.requestStop();
            notifyRuntimeStop();
        });
    });
}

// ---------------------------------------------------------------------------
// Scheduler protocol tests — 19-test suite
// ---------------------------------------------------------------------------

namespace
{
    // -----------------------------------------------------------------------
    // RDRAM slot constants
    // -----------------------------------------------------------------------
    static constexpr uint32_t kTParamAddr       = 0x00004000u;
    static constexpr uint32_t kSParamAddr       = 0x00004020u;
    static constexpr uint32_t kEParamAddr       = 0x00004040u;
    static constexpr uint32_t kReferScratch     = 0x00004060u;
    static constexpr uint32_t kSlotWorkSid      = 0x00004100u;
    static constexpr uint32_t kSlotDoneSid      = 0x00004104u;
    static constexpr uint32_t kSlotGoSid        = 0x00004108u;
    static constexpr uint32_t kSlotEid          = 0x0000410Cu;
    static constexpr uint32_t kSlotTidParam     = 0x00004110u;
    static constexpr uint32_t kSlotResBits      = 0x00004114u;
    static constexpr uint32_t kRunLog           = 0x00004200u;
    static constexpr uint32_t kSeqAddr          = 0x00004240u;
    static constexpr uint32_t kResultBase       = 0x00004260u;
    static constexpr uint32_t kProgressCtr      = 0x00004280u;
    static constexpr uint32_t kStopFlag         = 0x00004284u;
    static constexpr uint32_t kStartedFlag      = 0x00004288u;
    static constexpr uint32_t kHostInGuestProbe = 0x0000428Cu;

    // Thread status constants (mirrors State.h)
    static constexpr int32_t kTHSRun         = 0x01;
    static constexpr int32_t kTHSWait        = 0x04;
    static constexpr int32_t kTHSSuspend     = 0x08;
    static constexpr int32_t kTHSWaitSuspend = 0x0c;
    static constexpr int32_t kTSWSleep       = 1;
    static constexpr int32_t kTSWSema        = 2;

    // -----------------------------------------------------------------------
    // Protocol-test helper: getSemaWaiters (reads waiters from ReferSemaStatus)
    // -----------------------------------------------------------------------
    int32_t getSemaWaiters(uint8_t *rdram, PS2Runtime *runtime, int32_t sid)
    {
        constexpr uint32_t kStatusScratch = 0x2C00u;
        R5900Context refCtx{};
        setRegU32(refCtx, 4, static_cast<uint32_t>(sid));
        setRegU32(refCtx, 5, kStatusScratch);
        ps2_syscalls::ReferSemaStatus(rdram, &refCtx, runtime);
        int32_t waiters = 0;
        std::memcpy(&waiters, rdram + kStatusScratch + 12u, sizeof(waiters));
        return waiters;
    }

    // -----------------------------------------------------------------------
    // Helper: getThreadStatus — reads fiber status+waitType via ReferThreadStatus
    // -----------------------------------------------------------------------
    static void getThreadStatus(std::vector<uint8_t> &rdram, PS2Runtime &runtime,
                                int tid, int32_t &outStatus, int32_t &outWaitType)
    {
        R5900Context ctx{};
        setRegU32(ctx, 4, static_cast<uint32_t>(tid));
        setRegU32(ctx, 5, kReferScratch);
        ps2_syscalls::ReferThreadStatus(rdram.data(), &ctx, &runtime);
        int32_t s = 0, w = 0;
        std::memcpy(&s, rdram.data() + kReferScratch + 0x00, 4);
        std::memcpy(&w, rdram.data() + kReferScratch + 0x24, 4);
        outStatus   = s;
        outWaitType = w;
    }

    // -----------------------------------------------------------------------
    // Helper: getEvfNumThreads — reads numThreads via ReferEventFlagStatus
    // -----------------------------------------------------------------------
    static int32_t getEvfNumThreads(std::vector<uint8_t> &rdram, PS2Runtime &runtime, int eid)
    {
        R5900Context ctx{};
        setRegU32(ctx, 4, static_cast<uint32_t>(eid));
        setRegU32(ctx, 5, kReferScratch);
        ps2_syscalls::ReferEventFlagStatus(rdram.data(), &ctx, &runtime);
        // Ps2EventFlagInfo: attr+0, option+4, initBits+8, currBits+12, numThreads+16
        int32_t num = 0;
        std::memcpy(&num, rdram.data() + kReferScratch + 16, 4);
        return num;
    }

    // -----------------------------------------------------------------------
    // Helper: getSemaCount — reads current count via ReferSemaStatus
    // -----------------------------------------------------------------------
    static int32_t getSemaCount(std::vector<uint8_t> &rdram, PS2Runtime *runtime, int sid)
    {
        R5900Context ctx{};
        setRegU32(ctx, 4, static_cast<uint32_t>(sid));
        setRegU32(ctx, 5, kReferScratch);
        ps2_syscalls::ReferSemaStatus(rdram.data(), &ctx, runtime);
        // ee_sema_t: count+0
        int32_t count = 0;
        std::memcpy(&count, rdram.data() + kReferScratch + 0, 4);
        return count;
    }

    // -----------------------------------------------------------------------
    // Helper: createSchedEvf — creates an event flag
    // -----------------------------------------------------------------------
    static int createSchedEvf(std::vector<uint8_t> &rdram, PS2Runtime *runtime,
                               uint32_t attr, uint32_t initBits)
    {
        uint32_t vals[3] = { attr, 0u, initBits };
        std::memcpy(rdram.data() + kEParamAddr, vals, 12);
        R5900Context ctx{};
        setRegU32(ctx, 4, kEParamAddr);
        ps2_syscalls::CreateEventFlag(rdram.data(), &ctx, runtime);
        return getRegS32(ctx, 2);
    }

    static void deleteSchedEvf(std::vector<uint8_t> &rdram, PS2Runtime *runtime, int eid)
    {
        R5900Context ctx{};
        setRegU32(ctx, 4, static_cast<uint32_t>(eid));
        ps2_syscalls::DeleteEventFlag(rdram.data(), &ctx, runtime);
    }

    // -----------------------------------------------------------------------
    // Helper: rdram write/read helpers
    // -----------------------------------------------------------------------
    static void rdramWrite32(std::vector<uint8_t> &rdram, uint32_t addr, uint32_t val)
    {
        std::memcpy(rdram.data() + addr, &val, 4);
    }

    static uint32_t rdramSeq(const std::vector<uint8_t> &rdram)
    {
        uint32_t v = 0;
        std::memcpy(&v, rdram.data() + kSeqAddr, 4);
        return v;
    }

    static void rdramSeqReset(std::vector<uint8_t> &rdram)
    {
        uint32_t z = 0;
        std::memcpy(rdram.data() + kSeqAddr, &z, 4);
    }

    // -----------------------------------------------------------------------
    // Step functions for the 19 protocol tests
    // All use rdram-resident kSeqAddr counter; safe under N=1.
    // -----------------------------------------------------------------------

    // stepWaitSemaRecordSignal: wait on workSid, record result+tid, signal doneSid
    static void stepWaitSemaRecordSignal(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t workSid = 0, doneSid = 0;
        std::memcpy(&workSid, rdram + kSlotWorkSid, 4);
        std::memcpy(&doneSid, rdram + kSlotDoneSid, 4);
        R5900Context sc{};
        setRegU32(sc, 4, static_cast<uint32_t>(workSid));
        ps2_syscalls::WaitSema(rdram, &sc, runtime);
        int32_t ret = getRegS32(sc, 2);
        uint32_t seq = 0;
        std::memcpy(&seq, rdram + kSeqAddr, 4);
        std::memcpy(rdram + kResultBase + seq * 4, &ret, 4);
        int32_t tid = g_currentThreadId;
        std::memcpy(rdram + kRunLog + seq * 4, &tid, 4);
        seq++;
        std::memcpy(rdram + kSeqAddr, &seq, 4);
        if (doneSid > 0)
        {
            R5900Context sc2{};
            setRegU32(sc2, 4, static_cast<uint32_t>(doneSid));
            ps2_syscalls::SignalSema(rdram, &sc2, runtime);
        }
        ctx->pc = 0u;
    }

    // stepWaitGateLogOrder: wait on goSid, log tid in run order
    static void stepWaitGateLogOrder(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t goSid = 0;
        std::memcpy(&goSid, rdram + kSlotGoSid, 4);
        R5900Context sc{};
        setRegU32(sc, 4, static_cast<uint32_t>(goSid));
        ps2_syscalls::WaitSema(rdram, &sc, runtime);
        uint32_t seq = 0;
        std::memcpy(&seq, rdram + kSeqAddr, 4);
        int32_t tid = g_currentThreadId;
        std::memcpy(rdram + kRunLog + seq * 4, &tid, 4);
        seq++;
        std::memcpy(rdram + kSeqAddr, &seq, 4);
        ctx->pc = 0u;
    }

    // stepSpinUntilStop: set kStartedFlag, spin yielding, log when stopped
    static void stepSpinUntilStop(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t one = 1u;
        std::memcpy(rdram + kStartedFlag, &one, 4);
        uint32_t stop = 0;
        while (true)
        {
            std::memcpy(&stop, rdram + kStopFlag, 4);
            if (stop) break;
            runtime->shouldPreemptGuestExecution();
        }
        uint32_t seq = 0;
        std::memcpy(&seq, rdram + kSeqAddr, 4);
        int32_t tid = g_currentThreadId;
        std::memcpy(rdram + kRunLog + seq * 4, &tid, 4);
        seq++;
        std::memcpy(rdram + kSeqAddr, &seq, 4);
        ctx->pc = 0u;
    }

    // stepProgressLoop: set kStartedFlag, increment kProgressCtr each iter, stop on kStopFlag
    static void stepProgressLoop(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t one = 1u;
        std::memcpy(rdram + kStartedFlag, &one, 4);
        uint32_t stop = 0;
        while (true)
        {
            std::memcpy(&stop, rdram + kStopFlag, 4);
            if (stop) break;
            uint32_t ctr = 0;
            std::memcpy(&ctr, rdram + kProgressCtr, 4);
            ctr++;
            std::memcpy(rdram + kProgressCtr, &ctr, 4);
            runtime->shouldPreemptGuestExecution();
        }
        ctx->pc = 0u;
    }

    // stepSleepRecordSignal: SleepThread, then record + signal doneSid
    static void stepSleepRecordSignal(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t doneSid = 0;
        std::memcpy(&doneSid, rdram + kSlotDoneSid, 4);
        R5900Context sc{};
        ps2_syscalls::SleepThread(rdram, &sc, runtime);
        uint32_t seq = 0;
        std::memcpy(&seq, rdram + kSeqAddr, 4);
        int32_t tid = g_currentThreadId;
        std::memcpy(rdram + kRunLog + seq * 4, &tid, 4);
        seq++;
        std::memcpy(rdram + kSeqAddr, &seq, 4);
        if (doneSid > 0)
        {
            R5900Context sc2{};
            setRegU32(sc2, 4, static_cast<uint32_t>(doneSid));
            ps2_syscalls::SignalSema(rdram, &sc2, runtime);
        }
        ctx->pc = 0u;
    }

    // stepWaitEvfAndRecord: WaitEventFlag(eid, 0x3, AND no-clear), record result+tid
    static void stepWaitEvfAndRecord(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t eid = 0, doneSid = 0;
        std::memcpy(&eid,    rdram + kSlotEid,    4);
        std::memcpy(&doneSid, rdram + kSlotDoneSid, 4);
        R5900Context sc{};
        setRegU32(sc, 4, static_cast<uint32_t>(eid));
        setRegU32(sc, 5, 0x3u);
        setRegU32(sc, 6, 0x0u);
        setRegU32(sc, 7, kSlotResBits);
        ps2_syscalls::WaitEventFlag(rdram, &sc, runtime);
        int32_t ret = getRegS32(sc, 2);
        uint32_t seq = 0;
        std::memcpy(&seq, rdram + kSeqAddr, 4);
        std::memcpy(rdram + kResultBase + seq * 4, &ret, 4);
        int32_t tid = g_currentThreadId;
        std::memcpy(rdram + kRunLog + seq * 4, &tid, 4);
        seq++;
        std::memcpy(rdram + kSeqAddr, &seq, 4);
        if (doneSid > 0)
        {
            R5900Context sc2{};
            setRegU32(sc2, 4, static_cast<uint32_t>(doneSid));
            ps2_syscalls::SignalSema(rdram, &sc2, runtime);
        }
        ctx->pc = 0u;
    }

    // stepChangePrioOfTargetThenLog: read tidX from kSlotTidParam, bump to prio=5, yield 500x, log self
    static void stepChangePrioOfTargetThenLog(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t targetTid = 0;
        std::memcpy(&targetTid, rdram + kSlotTidParam, 4);
        R5900Context sc{};
        setRegU32(sc, 4, static_cast<uint32_t>(targetTid));
        setRegU32(sc, 5, 5u);
        ps2_syscalls::ChangeThreadPriority(rdram, &sc, runtime);
        for (int i = 0; i < 500; ++i)
        {
            runtime->shouldPreemptGuestExecution();
        }
        uint32_t seq = 0;
        std::memcpy(&seq, rdram + kSeqAddr, 4);
        int32_t tid = g_currentThreadId;
        std::memcpy(rdram + kRunLog + seq * 4, &tid, 4);
        seq++;
        std::memcpy(rdram + kSeqAddr, &seq, 4);
        ctx->pc = 0u;
    }

    // stepSpinLogAtEntry: log tid at entry, spin until kStopFlag
    static void stepSpinLogAtEntry(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t seq = 0;
        std::memcpy(&seq, rdram + kSeqAddr, 4);
        int32_t tid = g_currentThreadId;
        std::memcpy(rdram + kRunLog + seq * 4, &tid, 4);
        seq++;
        std::memcpy(rdram + kSeqAddr, &seq, 4);
        uint32_t one = 1u;
        std::memcpy(rdram + kStartedFlag, &one, 4);
        uint32_t stop = 0;
        while (true)
        {
            std::memcpy(&stop, rdram + kStopFlag, 4);
            if (stop) break;
            runtime->shouldPreemptGuestExecution();
        }
        ctx->pc = 0u;
    }

    // stepBlockForever: WaitSema on workSid (never wakes normally)
    static void stepBlockForever(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t workSid = 0;
        std::memcpy(&workSid, rdram + kSlotWorkSid, 4);
        R5900Context sc{};
        setRegU32(sc, 4, static_cast<uint32_t>(workSid));
        ps2_syscalls::WaitSema(rdram, &sc, runtime);
        uint32_t seq = 0;
        std::memcpy(&seq, rdram + kSeqAddr, 4);
        int32_t ret = getRegS32(sc, 2);
        std::memcpy(rdram + kResultBase + seq * 4, &ret, 4);
        int32_t tid = g_currentThreadId;
        std::memcpy(rdram + kRunLog + seq * 4, &tid, 4);
        seq++;
        std::memcpy(rdram + kSeqAddr, &seq, 4);
        ctx->pc = 0u;
    }

    // stepHoldSilentExit: blocks on kSlotWorkSid then exits without logging
    static void stepHoldSilentExit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t holdSid = 0;
        std::memcpy(&holdSid, rdram + kSlotWorkSid, 4);
        R5900Context sc{};
        setRegU32(sc, 4, static_cast<uint32_t>(holdSid));
        ps2_syscalls::WaitSema(rdram, &sc, runtime);
        ctx->pc = 0u;
    }

    // stepTerminateTarget: reads target tid, calls TerminateThread on it, logs self
    static void stepTerminateTarget(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t targetTid = 0;
        std::memcpy(&targetTid, rdram + kSlotTidParam, 4);
        R5900Context sc{};
        setRegU32(sc, 4, static_cast<uint32_t>(targetTid));
        ps2_syscalls::TerminateThread(rdram, &sc, runtime);
        uint32_t seq = 0;
        std::memcpy(&seq, rdram + kSeqAddr, 4);
        int32_t tid = g_currentThreadId;
        std::memcpy(rdram + kRunLog + seq * 4, &tid, 4);
        seq++;
        std::memcpy(rdram + kSeqAddr, &seq, 4);
        ctx->pc = 0u;
    }

    // stepLogAndExit: log tid and exit immediately
    static void stepLogAndExit(uint8_t *rdram, R5900Context *ctx, PS2Runtime * /*runtime*/)
    {
        uint32_t seq = 0;
        std::memcpy(&seq, rdram + kSeqAddr, 4);
        int32_t tid = g_currentThreadId;
        std::memcpy(rdram + kRunLog + seq * 4, &tid, 4);
        seq++;
        std::memcpy(rdram + kSeqAddr, &seq, 4);
        ctx->pc = 0u;
    }

} // anonymous namespace

// ---------------------------------------------------------------------------
// 19-test scheduler protocol suite
// ---------------------------------------------------------------------------
void register_scheduler_protocol_tests()
{
    MiniTest::Case("SchedulerProtocol", [](TestCase &tc)
    {
        // ------------------------------------------------------------------
        // Test 1: Priority ordering — 3 fibers at priority 5, 10, 20
        // ------------------------------------------------------------------
        tc.Run("fibers run in priority order 5 then 10 then 20", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00600000u, &stepWaitGateLogOrder);

            const int32_t goSid = createSchedSema(rdram.data(), &runtime, 0, 3);
            t.IsTrue(goSid > 0, "T1: gate sema created");
            if (goSid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            {
                int32_t v = goSid;
                std::memcpy(rdram.data() + kSlotGoSid, &v, 4);
            }
            rdramSeqReset(rdram);
            std::memset(rdram.data() + kRunLog, 0, 32u);

            const int32_t tidC = startSchedWorker(rdram.data(), &runtime, 0x00600000u, 20, 0x00500000u, 0x2000u);
            const int32_t tidA = startSchedWorker(rdram.data(), &runtime, 0x00600000u,  5, 0x00502000u, 0x2000u);
            const int32_t tidB = startSchedWorker(rdram.data(), &runtime, 0x00600000u, 10, 0x00504000u, 0x2000u);
            t.IsTrue(tidA > 0, "T1: A started"); t.IsTrue(tidB > 0, "T1: B started"); t.IsTrue(tidC > 0, "T1: C started");

            const bool allBlocked = waitUntil([&]()
            {
                return getSemaWaiters(rdram.data(), &runtime, goSid) >= 3;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(allBlocked, "T1: all 3 blocked on gate sema");

            for (int i = 0; i < 3; ++i)
            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(goSid));
                ps2_syscalls::SignalSema(rdram.data(), &sc, &runtime);
            }

            const bool allDone = waitUntil([&](){ return rdramSeq(rdram) >= 3u; }, std::chrono::milliseconds(1000));
            t.IsTrue(allDone, "T1: all 3 completed");

            int32_t log[3] = {};
            std::memcpy(log, rdram.data() + kRunLog, 12);
            t.Equals(log[0], tidA, "T1: prio 5 runs first");
            t.Equals(log[1], tidB, "T1: prio 10 runs second");
            t.Equals(log[2], tidC, "T1: prio 20 runs last");

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            deleteSchedSema(rdram.data(), &runtime, goSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 2: FIFO within equal priority
        // ------------------------------------------------------------------
        tc.Run("equal-priority fibers run in FIFO creation order", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00610000u, &stepWaitGateLogOrder);

            const int32_t goSid = createSchedSema(rdram.data(), &runtime, 0, 2);
            t.IsTrue(goSid > 0, "T2: gate sema created");
            if (goSid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            {
                int32_t v = goSid;
                std::memcpy(rdram.data() + kSlotGoSid, &v, 4);
            }
            rdramSeqReset(rdram);
            std::memset(rdram.data() + kRunLog, 0, 16u);

            const int32_t tidA = startSchedWorker(rdram.data(), &runtime, 0x00610000u, 10, 0x00510000u, 0x2000u);
            const int32_t tidB = startSchedWorker(rdram.data(), &runtime, 0x00610000u, 10, 0x00512000u, 0x2000u);
            t.IsTrue(tidA > 0, "T2: A started"); t.IsTrue(tidB > 0, "T2: B started");

            const bool allBlocked = waitUntil([&]()
            {
                return getSemaWaiters(rdram.data(), &runtime, goSid) >= 2;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(allBlocked, "T2: both blocked on gate sema");

            for (int i = 0; i < 2; ++i)
            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(goSid));
                ps2_syscalls::SignalSema(rdram.data(), &sc, &runtime);
            }

            const bool allDone = waitUntil([&](){ return rdramSeq(rdram) >= 2u; }, std::chrono::milliseconds(1000));
            t.IsTrue(allDone, "T2: both completed");

            int32_t log[2] = {};
            std::memcpy(log, rdram.data() + kRunLog, 8);
            t.Equals(log[0], tidA, "T2: A (created first) runs first");
            t.Equals(log[1], tidB, "T2: B (created second) runs second");

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            deleteSchedSema(rdram.data(), &runtime, goSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 3: RotateThreadReadyQueue — race-free via async_guest_begin
        // ------------------------------------------------------------------
        tc.Run("RotateThreadReadyQueue reorders equal-priority ready fibers [A,B,C]->[B,C,A]", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00620000u, &stepLogAndExit);

            rdramSeqReset(rdram);
            std::memset(rdram.data() + kRunLog, 0, 32u);

            // Acquire guest token so executor cannot run fibers yet
            ps2sched::async_guest_begin();

            const int32_t tidA = startSchedWorker(rdram.data(), &runtime, 0x00620000u, 10, 0x00520000u, 0x2000u);
            const int32_t tidB = startSchedWorker(rdram.data(), &runtime, 0x00620000u, 10, 0x00522000u, 0x2000u);
            const int32_t tidC = startSchedWorker(rdram.data(), &runtime, 0x00620000u, 10, 0x00524000u, 0x2000u);
            t.IsTrue(tidA > 0, "T3: A started"); t.IsTrue(tidB > 0, "T3: B started"); t.IsTrue(tidC > 0, "T3: C started");

            // Rotate [A,B,C] -> [B,C,A] while executor is locked out
            {
                R5900Context sc{}; setRegU32(sc, 4, 10u);
                ps2_syscalls::RotateThreadReadyQueue(rdram.data(), &sc, &runtime);
            }

            // Release executor
            ps2sched::async_guest_end();

            const bool allDone = waitUntil([&](){ return rdramSeq(rdram) >= 3u; }, std::chrono::milliseconds(2000));
            t.IsTrue(allDone, "T3: all 3 fibers logged");

            int32_t log[3] = {};
            std::memcpy(log, rdram.data() + kRunLog, 12);
            t.Equals(log[0], tidB, "T3: B runs first after rotate");
            t.Equals(log[1], tidC, "T3: C runs second after rotate");
            t.Equals(log[2], tidA, "T3: A runs last (moved to tail)");

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 4: ChangeThreadPriority re-sorts Ready fiber
        // ------------------------------------------------------------------
        tc.Run("ChangeThreadPriority raises X above Y causing X to preempt Y", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00630000u, &stepChangePrioOfTargetThenLog);
            runtime.registerFunction(0x00638000u, &stepSpinLogAtEntry);

            rdramSeqReset(rdram);
            std::memset(rdram.data() + kRunLog, 0, 32u);
            rdramWrite32(rdram, kStartedFlag, 0u);
            rdramWrite32(rdram, kStopFlag, 0u);

            // Lock out executor
            ps2sched::async_guest_begin();

            const int32_t tidX = startSchedWorker(rdram.data(), &runtime, 0x00638000u, 20, 0x00532000u, 0x2000u);
            const int32_t tidY = startSchedWorker(rdram.data(), &runtime, 0x00630000u, 10, 0x00530000u, 0x2000u);
            t.IsTrue(tidX > 0, "T4: X started"); t.IsTrue(tidY > 0, "T4: Y started");

            {
                int32_t v = tidX;
                std::memcpy(rdram.data() + kSlotTidParam, &v, 4);
            }

            // Release executor: Y runs first (prio 10), calls ChangeThreadPriority(X, 5),
            // X bumped to prio=5 (higher than Y=10), Y yields 500x -> X preempts
            ps2sched::async_guest_end();

            // Wait for both to complete (Y logs after 500-yield loop; X logs at entry)
            const bool allDone = waitUntil([&](){ return rdramSeq(rdram) >= 2u; }, std::chrono::milliseconds(2000));
            t.IsTrue(allDone, "T4: both fibers completed");

            // Stop X's spin
            rdramWrite32(rdram, kStopFlag, 1u);
            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));

            int32_t log[2] = {};
            std::memcpy(log, rdram.data() + kRunLog, 8);
            t.Equals(log[0], tidX, "T4: X (bumped to prio=5) logs first");
            t.Equals(log[1], tidY, "T4: Y logs second (after yielding to X)");

            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 5: SuspendThread / ResumeThread
        // ------------------------------------------------------------------
        tc.Run("SuspendThread stops progress and ResumeThread restores it", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00640000u, &stepProgressLoop);

            rdramWrite32(rdram, kProgressCtr, 0u);
            rdramWrite32(rdram, kStartedFlag, 0u);
            rdramWrite32(rdram, kStopFlag, 0u);

            const int32_t tid = startSchedWorker(rdram.data(), &runtime, 0x00640000u, 10, 0x00540000u, 0x2000u);
            t.IsTrue(tid > 0, "T5: progress fiber started");
            if (tid <= 0)
            {
                rdramWrite32(rdram, kStopFlag, 1u);
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            const bool started = waitUntil([&]()
            {
                return Ps2FastRead32(rdram.data(), kStartedFlag) != 0u;
            }, std::chrono::milliseconds(500));
            t.IsTrue(started, "T5: fiber started");

            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::SuspendThread(rdram.data(), &sc, &runtime);
                t.Equals(getRegS32(sc, 2), KE_OK, "T5: SuspendThread returned KE_OK");
            }

            const bool suspended = waitUntil([&]()
            {
                int32_t status = 0, wt = 0;
                getThreadStatus(rdram, runtime, tid, status, wt);
                return status == kTHSSuspend;
            }, std::chrono::milliseconds(500));
            t.IsTrue(suspended, "T5: fiber reached THS_SUSPEND");

            // Verify progress halted
            const uint32_t cBefore = Ps2FastRead32(rdram.data(), kProgressCtr);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const uint32_t cAfter = Ps2FastRead32(rdram.data(), kProgressCtr);
            t.Equals(cBefore, cAfter, "T5: counter not advancing while suspended");

            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::ResumeThread(rdram.data(), &sc, &runtime);
                t.Equals(getRegS32(sc, 2), KE_OK, "T5: ResumeThread returned KE_OK");
            }

            const bool advanced = waitUntil([&]()
            {
                return Ps2FastRead32(rdram.data(), kProgressCtr) > cAfter;
            }, std::chrono::milliseconds(500));
            t.IsTrue(advanced, "T5: progress resumes after ResumeThread");

            rdramWrite32(rdram, kStopFlag, 1u);
            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 6: SuspendThread while Blocked gates make_ready
        // ------------------------------------------------------------------
        tc.Run("suspended fiber does not wake from SignalSema until ResumeThread", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00650000u, &stepWaitSemaRecordSignal);

            const int32_t workSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            const int32_t doneSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(workSid > 0, "T6: work sema created"); t.IsTrue(doneSid > 0, "T6: done sema created");
            if (workSid <= 0 || doneSid <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, workSid);
                deleteSchedSema(rdram.data(), &runtime, doneSid);
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            {
                int32_t w = workSid, d = doneSid;
                std::memcpy(rdram.data() + kSlotWorkSid, &w, 4);
                std::memcpy(rdram.data() + kSlotDoneSid, &d, 4);
            }
            rdramSeqReset(rdram);
            {
                int32_t bad = -9999;
                std::memcpy(rdram.data() + kResultBase, &bad, 4);
            }

            const int32_t tid = startSchedWorker(rdram.data(), &runtime, 0x00650000u, 10, 0x00550000u, 0x2000u);
            t.IsTrue(tid > 0, "T6: fiber started");

            const bool blocked = waitUntil([&]()
            {
                return getSemaWaiters(rdram.data(), &runtime, workSid) >= 1;
            }, std::chrono::milliseconds(500));
            t.IsTrue(blocked, "T6: fiber blocked on workSid");

            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::SuspendThread(rdram.data(), &sc, &runtime);
                t.Equals(getRegS32(sc, 2), KE_OK, "T6: SuspendThread returned KE_OK");
            }

            const bool isSuspended = waitUntil([&]()
            {
                int32_t status = 0, wt = 0;
                getThreadStatus(rdram, runtime, tid, status, wt);
                return status == kTHSWaitSuspend;
            }, std::chrono::milliseconds(500));
            t.IsTrue(isSuspended, "T6: fiber reached THS_WAITSUSPEND");

            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(workSid));
                ps2_syscalls::SignalSema(rdram.data(), &sc, &runtime);
                t.Equals(getRegS32(sc, 2), KE_OK, "T6: SignalSema returned KE_OK");
            }

            // Wait to confirm fiber did NOT consume it
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
            t.Equals(rdramSeq(rdram), 0u, "T6: fiber did not complete while suspended");
            t.Equals(getSemaCount(rdram, &runtime, workSid), 1, "T6: permit not consumed while suspended");

            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::ResumeThread(rdram.data(), &sc, &runtime);
                t.Equals(getRegS32(sc, 2), KE_OK, "T6: ResumeThread returned KE_OK");
            }

            const bool woke = waitUntil([&](){ return rdramSeq(rdram) >= 1u; }, std::chrono::milliseconds(1000));
            t.IsTrue(woke, "T6: fiber woke after ResumeThread");
            {
                int32_t ret = -9999;
                std::memcpy(&ret, rdram.data() + kResultBase, 4);
                t.Equals(ret, KE_OK, "T6: WaitSema returned KE_OK after resume");
            }

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            deleteSchedSema(rdram.data(), &runtime, workSid);
            deleteSchedSema(rdram.data(), &runtime, doneSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 7: SignalSema/PollSema race — Mesa loop correctness
        // ------------------------------------------------------------------
        tc.Run("SignalSema and PollSema race never over-consumes permits", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00660000u, &stepWaitSemaRecordSignal);

            const int32_t workSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            const int32_t doneSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(workSid > 0, "T7: work sema created"); t.IsTrue(doneSid > 0, "T7: done sema created");
            if (workSid <= 0 || doneSid <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, workSid);
                deleteSchedSema(rdram.data(), &runtime, doneSid);
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            {
                int32_t w = workSid, d = doneSid;
                std::memcpy(rdram.data() + kSlotWorkSid, &w, 4);
                std::memcpy(rdram.data() + kSlotDoneSid, &d, 4);
            }
            rdramSeqReset(rdram);
            {
                int32_t bad = -9999;
                std::memcpy(rdram.data() + kResultBase, &bad, 4);
            }

            const int32_t tid = startSchedWorker(rdram.data(), &runtime, 0x00660000u, 10, 0x00560000u, 0x2000u);
            t.IsTrue(tid > 0, "T7: waiter fiber started");

            const bool blocked = waitUntil([&]()
            {
                return getSemaWaiters(rdram.data(), &runtime, workSid) >= 1;
            }, std::chrono::milliseconds(500));
            t.IsTrue(blocked, "T7: waiter blocked on workSid");

            // Issue exactly ONE permit. Exactly one of {poller, waiter} must
            // consume it; the other must get nothing. Counting consumers from
            // observed results (not from the number of signals) is what lets this
            // test actually catch a double-consume.
            std::atomic<int> pollOK{0};

            std::thread poller([&]()
            {
                for (int i = 0; i < 200 && pollOK.load() == 0; ++i)
                {
                    R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(workSid));
                    ps2_syscalls::PollSema(rdram.data(), &sc, &runtime);
                    if (getRegS32(sc, 2) == KE_OK)
                        pollOK.store(1);
                }
            });

            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(workSid));
                ps2_syscalls::SignalSema(rdram.data(), &sc, &runtime);
            }

            poller.join();

            // If the poller stole the only permit, the waiter is still parked and
            // this single signal will never wake it. Release it WITHOUT producing a
            // permit so the test does not hang; it then records KE_RELEASE_WAIT
            // (not KE_OK), keeping the consumer count honest.
            if (pollOK.load() == 1)
            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::ReleaseWaitThread(rdram.data(), &sc, &runtime);
            }

            const bool waiterDone = waitUntil([&](){ return rdramSeq(rdram) >= 1u; }, std::chrono::milliseconds(2000));
            t.IsTrue(waiterDone, "T7: waiter fiber completed");

            int32_t waiterRet = -9999;
            std::memcpy(&waiterRet, rdram.data() + kResultBase, 4);
            const int waiterOK = (waiterRet == KE_OK) ? 1 : 0;

            // The single permit must have been consumed by exactly one party.
            const int consumers = pollOK.load() + waiterOK;
            t.Equals(consumers, 1, "T7: exactly one consumer of the single permit (no double-consume)");
            t.IsTrue(getSemaCount(rdram, &runtime, workSid) >= 0, "T7: sema count never negative");
            t.Equals(getSemaCount(rdram, &runtime, workSid), 0, "T7: permit fully consumed, none left over");

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            deleteSchedSema(rdram.data(), &runtime, workSid);
            deleteSchedSema(rdram.data(), &runtime, doneSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 8: AsyncGuestScope blocks executor while held
        // ------------------------------------------------------------------
        tc.Run("async_guest_begin blocks executor until async_guest_end", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00670000u, &stepProgressLoop);

            rdramWrite32(rdram, kStartedFlag, 0u);
            rdramWrite32(rdram, kStopFlag, 0u);
            rdramWrite32(rdram, kProgressCtr, 0u);

            ps2sched::async_guest_begin();

            const int32_t tid = startSchedWorker(rdram.data(), &runtime, 0x00670000u, 10, 0x00570000u, 0x2000u);
            t.IsTrue(tid > 0, "T8: fiber started");

            // Fiber must NOT have started while host holds token
            const uint32_t probe = Ps2FastRead32(rdram.data(), kStartedFlag);
            t.Equals(probe, 0u, "T8: no fiber runs while host holds guest token");

            ps2sched::async_guest_end();

            const bool started = waitUntil([&]()
            {
                return Ps2FastRead32(rdram.data(), kStartedFlag) != 0u;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(started, "T8: fiber starts after async_guest_end");

            rdramWrite32(rdram, kStopFlag, 1u);
            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 9: AsyncGuestScope RAII releases token on exception
        // ------------------------------------------------------------------
        tc.Run("AsyncGuestScope RAII releases token on exception path", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00680000u, &stepProgressLoop);

            rdramWrite32(rdram, kStartedFlag, 0u);
            rdramWrite32(rdram, kStopFlag, 0u);

            struct TestGuestScope {
                TestGuestScope()  { ps2sched::async_guest_begin(); }
                ~TestGuestScope() { ps2sched::async_guest_end(); }
            };

            try {
                TestGuestScope g;
                throw std::runtime_error("test exception");
            } catch (const std::exception &) { /* swallow */ }

            // Token should be released now
            const int32_t tid = startSchedWorker(rdram.data(), &runtime, 0x00680000u, 10, 0x00580000u, 0x2000u);
            t.IsTrue(tid > 0, "T9: fiber started");

            const bool started = waitUntil([&]()
            {
                return Ps2FastRead32(rdram.data(), kStartedFlag) != 0u;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(started, "T9: fiber starts after RAII scope released token on exception");

            rdramWrite32(rdram, kStopFlag, 1u);
            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 10: enqueue_external_wakeup from foreign thread wakes sleeping fiber
        // ------------------------------------------------------------------
        tc.Run("enqueue_external_wakeup from foreign thread wakes sleeping fiber", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00690000u, &stepSleepRecordSignal);

            const int32_t doneSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(doneSid > 0, "T10: done sema created");
            if (doneSid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            {
                int32_t d = doneSid;
                std::memcpy(rdram.data() + kSlotDoneSid, &d, 4);
            }
            rdramSeqReset(rdram);

            const int32_t tid = startSchedWorker(rdram.data(), &runtime, 0x00690000u, 10, 0x00590000u, 0x2000u);
            t.IsTrue(tid > 0, "T10: fiber started");

            const bool sleeping = waitUntil([&]()
            {
                int32_t status = 0, wt = 0;
                getThreadStatus(rdram, runtime, tid, status, wt);
                return status == kTHSWait && wt == kTSWSleep;
            }, std::chrono::milliseconds(500));
            t.IsTrue(sleeping, "T10: fiber is in SleepThread");

            std::thread foreign([&](){ ps2sched::enqueue_external_wakeup(tid); });
            foreign.join();

            const bool woke = waitUntil([&](){ return rdramSeq(rdram) >= 1u; }, std::chrono::milliseconds(1000));
            t.IsTrue(woke, "T10: fiber woke from enqueue_external_wakeup");
            {
                int32_t log0 = 0;
                std::memcpy(&log0, rdram.data() + kRunLog, 4);
                t.Equals(log0, tid, "T10: correct fiber woke");
            }

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            deleteSchedSema(rdram.data(), &runtime, doneSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 11: Suspended fiber does NOT wake on enqueue_external_wakeup
        // ------------------------------------------------------------------
        tc.Run("suspended sleeping fiber ignores enqueue_external_wakeup", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x006A0000u, &stepSleepRecordSignal);

            const int32_t doneSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(doneSid > 0, "T11: done sema created");
            if (doneSid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            {
                int32_t d = doneSid;
                std::memcpy(rdram.data() + kSlotDoneSid, &d, 4);
            }
            rdramSeqReset(rdram);

            const int32_t tid = startSchedWorker(rdram.data(), &runtime, 0x006A0000u, 10, 0x005A0000u, 0x2000u);
            t.IsTrue(tid > 0, "T11: fiber started");

            const bool sleeping = waitUntil([&]()
            {
                int32_t status = 0, wt = 0;
                getThreadStatus(rdram, runtime, tid, status, wt);
                return status == kTHSWait && wt == kTSWSleep;
            }, std::chrono::milliseconds(500));
            t.IsTrue(sleeping, "T11: fiber is sleeping");

            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::SuspendThread(rdram.data(), &sc, &runtime);
                t.Equals(getRegS32(sc, 2), KE_OK, "T11: SuspendThread returned KE_OK");
            }

            const bool isSuspended = waitUntil([&]()
            {
                int32_t status = 0, wt = 0;
                getThreadStatus(rdram, runtime, tid, status, wt);
                return status == kTHSWaitSuspend;
            }, std::chrono::milliseconds(500));
            t.IsTrue(isSuspended, "T11: fiber reached THS_WAITSUSPEND");

            std::thread foreign([&](){ ps2sched::enqueue_external_wakeup(tid); });
            foreign.join();

            std::this_thread::sleep_for(std::chrono::milliseconds(40));
            t.Equals(rdramSeq(rdram), 0u, "T11: suspended fiber did not wake from external wakeup");
            {
                int32_t status = 0, wt = 0;
                getThreadStatus(rdram, runtime, tid, status, wt);
                t.IsTrue(status == kTHSWaitSuspend || status == kTHSSuspend,
                         "T11: fiber remains suspended");
            }

            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::ResumeThread(rdram.data(), &sc, &runtime);
                t.Equals(getRegS32(sc, 2), KE_OK, "T11: ResumeThread returned KE_OK");
            }

            const bool woke = waitUntil([&](){ return rdramSeq(rdram) >= 1u; }, std::chrono::milliseconds(1000));
            t.IsTrue(woke, "T11: fiber woke after ResumeThread");

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            deleteSchedSema(rdram.data(), &runtime, doneSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 12: WaitSema basic round-trip
        // ------------------------------------------------------------------
        tc.Run("WaitSema returns KE_OK after SignalSema and consumes exactly one permit", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x006B0000u, &stepWaitSemaRecordSignal);

            const int32_t workSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            const int32_t doneSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(workSid > 0, "T12: work sema created"); t.IsTrue(doneSid > 0, "T12: done sema created");
            if (workSid <= 0 || doneSid <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, workSid);
                deleteSchedSema(rdram.data(), &runtime, doneSid);
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            {
                int32_t w = workSid, d = doneSid;
                std::memcpy(rdram.data() + kSlotWorkSid, &w, 4);
                std::memcpy(rdram.data() + kSlotDoneSid, &d, 4);
            }
            rdramSeqReset(rdram);
            {
                int32_t bad = -9999;
                std::memcpy(rdram.data() + kResultBase, &bad, 4);
            }

            const int32_t tid = startSchedWorker(rdram.data(), &runtime, 0x006B0000u, 10, 0x005B0000u, 0x2000u);
            t.IsTrue(tid > 0, "T12: fiber started");

            const bool blocked = waitUntil([&]()
            {
                return getSemaWaiters(rdram.data(), &runtime, workSid) >= 1;
            }, std::chrono::milliseconds(500));
            t.IsTrue(blocked, "T12: fiber blocked on workSid");

            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(workSid));
                ps2_syscalls::SignalSema(rdram.data(), &sc, &runtime);
            }

            const bool done = waitUntil([&](){ return rdramSeq(rdram) >= 1u; }, std::chrono::milliseconds(1000));
            t.IsTrue(done, "T12: fiber completed");

            {
                int32_t ret = -9999;
                std::memcpy(&ret, rdram.data() + kResultBase, 4);
                t.Equals(ret, KE_OK, "T12: WaitSema returned KE_OK after signal");
            }
            t.Equals(getSemaCount(rdram, &runtime, workSid), 0, "T12: permit consumed");

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            deleteSchedSema(rdram.data(), &runtime, workSid);
            deleteSchedSema(rdram.data(), &runtime, doneSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 13: SleepThread / WakeupThread round-trip
        // ------------------------------------------------------------------
        tc.Run("SleepThread blocks fiber and WakeupThread wakes it", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x006C0000u, &stepSleepRecordSignal);

            const int32_t doneSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(doneSid > 0, "T13: done sema created");
            if (doneSid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            {
                int32_t d = doneSid;
                std::memcpy(rdram.data() + kSlotDoneSid, &d, 4);
            }
            rdramSeqReset(rdram);

            const int32_t tid = startSchedWorker(rdram.data(), &runtime, 0x006C0000u, 10, 0x005C0000u, 0x2000u);
            t.IsTrue(tid > 0, "T13: fiber started");

            const bool sleeping = waitUntil([&]()
            {
                int32_t status = 0, wt = 0;
                getThreadStatus(rdram, runtime, tid, status, wt);
                return status == kTHSWait && wt == kTSWSleep;
            }, std::chrono::milliseconds(500));
            t.IsTrue(sleeping, "T13: fiber is in SleepThread");

            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::WakeupThread(rdram.data(), &sc, &runtime);
                t.Equals(getRegS32(sc, 2), KE_OK, "T13: WakeupThread returned KE_OK");
            }

            const bool woke = waitUntil([&](){ return rdramSeq(rdram) >= 1u; }, std::chrono::milliseconds(1000));
            t.IsTrue(woke, "T13: fiber woke after WakeupThread");
            {
                int32_t log0 = 0;
                std::memcpy(&log0, rdram.data() + kRunLog, 4);
                t.Equals(log0, tid, "T13: correct fiber woke");
            }

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            deleteSchedSema(rdram.data(), &runtime, doneSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 14: TerminateThread(other) via killer fiber
        // ------------------------------------------------------------------
        tc.Run("TerminateThread from killer fiber terminates blocked worker", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x006D0000u, &stepBlockForever);
            runtime.registerFunction(0x006D8000u, &stepTerminateTarget);

            const int32_t workSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(workSid > 0, "T14: work sema created");
            if (workSid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            {
                int32_t w = workSid;
                std::memcpy(rdram.data() + kSlotWorkSid, &w, 4);
            }
            rdramSeqReset(rdram);

            const int32_t workerTid = startSchedWorker(rdram.data(), &runtime, 0x006D0000u, 10, 0x005D0000u, 0x2000u);
            t.IsTrue(workerTid > 0, "T14: worker started");

            const bool blocked = waitUntil([&]()
            {
                return getSemaWaiters(rdram.data(), &runtime, workSid) >= 1;
            }, std::chrono::milliseconds(500));
            t.IsTrue(blocked, "T14: worker blocked on workSid");

            {
                int32_t v = workerTid;
                std::memcpy(rdram.data() + kSlotTidParam, &v, 4);
            }

            const int32_t killerTid = startSchedWorker(rdram.data(), &runtime, 0x006D8000u, 5, 0x005D2000u, 0x2000u);
            t.IsTrue(killerTid > 0, "T14: killer started");

            const bool allDone = waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(3000));
            t.IsTrue(allDone, "T14: both worker and killer finished");
            t.IsTrue(rdramSeq(rdram) >= 1u, "T14: killer logged after join_fiber returned");

            deleteSchedSema(rdram.data(), &runtime, workSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 15: scheduler_shutdown with blocked fibers
        // ------------------------------------------------------------------
        tc.Run("scheduler_shutdown terminates all blocked fibers within 5s", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x006E0000u, &stepBlockForever);

            const int32_t workSid = createSchedSema(rdram.data(), &runtime, 0, 3);
            t.IsTrue(workSid > 0, "T15: work sema created");
            if (workSid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            {
                int32_t w = workSid;
                std::memcpy(rdram.data() + kSlotWorkSid, &w, 4);
            }

            const int32_t tid1 = startSchedWorker(rdram.data(), &runtime, 0x006E0000u, 10, 0x005E0000u, 0x2000u);
            const int32_t tid2 = startSchedWorker(rdram.data(), &runtime, 0x006E0000u, 10, 0x005E2000u, 0x2000u);
            const int32_t tid3 = startSchedWorker(rdram.data(), &runtime, 0x006E0000u, 10, 0x005E4000u, 0x2000u);
            t.IsTrue(tid1 > 0, "T15: fiber 1 started");
            t.IsTrue(tid2 > 0, "T15: fiber 2 started");
            t.IsTrue(tid3 > 0, "T15: fiber 3 started");

            const bool allBlocked = waitUntil([&]()
            {
                return getSemaWaiters(rdram.data(), &runtime, workSid) >= 3;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(allBlocked, "T15: all 3 blocked on workSid");

            const auto t0 = std::chrono::steady_clock::now();
            ps2sched::scheduler_shutdown();
            const auto elapsed = std::chrono::steady_clock::now() - t0;

            t.IsTrue(elapsed < std::chrono::seconds(5), "T15: scheduler_shutdown returned within 5s");
            t.Equals(g_activeThreads.load(), 0, "T15: all fibers terminated");

            // Do NOT call scheduler_shutdown again
            deleteSchedSema(rdram.data(), &runtime, workSid);
            runtime.requestStop();
            notifyRuntimeStop();
            return;
        });

        // ------------------------------------------------------------------
        // Test 16: WaitEventFlag AND-mode
        // ------------------------------------------------------------------
        tc.Run("WaitEventFlag AND-mode re-blocks on partial bit set and completes on full set", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x006F0000u, &stepWaitEvfAndRecord);

            const int32_t eid = createSchedEvf(rdram, &runtime, 0u, 0u);
            const int32_t doneSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(eid > 0, "T16: event flag created"); t.IsTrue(doneSid > 0, "T16: done sema created");
            if (eid <= 0 || doneSid <= 0)
            {
                if (eid > 0) deleteSchedEvf(rdram, &runtime, eid);
                deleteSchedSema(rdram.data(), &runtime, doneSid);
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            {
                int32_t e = eid, d = doneSid;
                std::memcpy(rdram.data() + kSlotEid,    &e, 4);
                std::memcpy(rdram.data() + kSlotDoneSid, &d, 4);
            }
            rdramSeqReset(rdram);
            {
                int32_t bad = -9999;
                std::memcpy(rdram.data() + kResultBase, &bad, 4);
            }
            rdramWrite32(rdram, kSlotResBits, 0u);

            const int32_t tid = startSchedWorker(rdram.data(), &runtime, 0x006F0000u, 10, 0x005F0000u, 0x2000u);
            t.IsTrue(tid > 0, "T16: fiber started");

            const bool waiting = waitUntil([&]()
            {
                return getEvfNumThreads(rdram, runtime, eid) >= 1;
            }, std::chrono::milliseconds(500));
            t.IsTrue(waiting, "T16: fiber registered as EventFlag waiter");

            // Partial set — AND not satisfied
            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(eid)); setRegU32(sc, 5, 0x1u);
                ps2_syscalls::SetEventFlag(rdram.data(), &sc, &runtime);
            }

            // Fiber may briefly dequeue and re-block; wait for it to re-block
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            const bool reBlocked = waitUntil([&]()
            {
                return getEvfNumThreads(rdram, runtime, eid) >= 1;
            }, std::chrono::milliseconds(500));
            t.IsTrue(reBlocked, "T16: fiber re-blocked after partial bit set");
            t.Equals(rdramSeq(rdram), 0u, "T16: fiber did not complete on partial bit");

            // Complete set
            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(eid)); setRegU32(sc, 5, 0x2u);
                ps2_syscalls::SetEventFlag(rdram.data(), &sc, &runtime);
            }

            const bool completed = waitUntil([&](){ return rdramSeq(rdram) >= 1u; }, std::chrono::milliseconds(1000));
            t.IsTrue(completed, "T16: fiber completed after all AND bits set");

            {
                int32_t ret = -9999;
                std::memcpy(&ret, rdram.data() + kResultBase, 4);
                t.Equals(ret, KE_OK, "T16: WaitEventFlag returned KE_OK");
            }
            {
                uint32_t resBits = 0;
                std::memcpy(&resBits, rdram.data() + kSlotResBits, 4);
                t.IsTrue((resBits & 0x3u) == 0x3u, "T16: result bits include all waited bits");
            }

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            deleteSchedEvf(rdram, &runtime, eid);
            deleteSchedSema(rdram.data(), &runtime, doneSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 17: DeleteEventFlag wakes all waiters with KE_WAIT_DELETE
        // ------------------------------------------------------------------
        tc.Run("DeleteEventFlag wakes all waiters with KE_WAIT_DELETE", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00700000u, &stepWaitEvfAndRecord);

            // EA_MULTI = 0x2 allows multiple waiters
            const int32_t eid = createSchedEvf(rdram, &runtime, 0x2u, 0u);
            const int32_t doneSid = createSchedSema(rdram.data(), &runtime, 0, 3);
            t.IsTrue(eid > 0, "T17: event flag created"); t.IsTrue(doneSid > 0, "T17: done sema created");
            if (eid <= 0 || doneSid <= 0)
            {
                if (eid > 0) deleteSchedEvf(rdram, &runtime, eid);
                deleteSchedSema(rdram.data(), &runtime, doneSid);
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            {
                int32_t e = eid, d = doneSid;
                std::memcpy(rdram.data() + kSlotEid,    &e, 4);
                std::memcpy(rdram.data() + kSlotDoneSid, &d, 4);
            }
            rdramSeqReset(rdram);
            for (int i = 0; i < 3; ++i)
            {
                int32_t bad = -9999;
                std::memcpy(rdram.data() + kResultBase + static_cast<uint32_t>(i * 4), &bad, 4);
            }

            const int32_t tid1 = startSchedWorker(rdram.data(), &runtime, 0x00700000u, 10, 0x00480000u, 0x2000u);
            const int32_t tid2 = startSchedWorker(rdram.data(), &runtime, 0x00700000u, 10, 0x00482000u, 0x2000u);
            const int32_t tid3 = startSchedWorker(rdram.data(), &runtime, 0x00700000u, 10, 0x00484000u, 0x2000u);
            t.IsTrue(tid1 > 0, "T17: fiber 1 started"); t.IsTrue(tid2 > 0, "T17: fiber 2 started"); t.IsTrue(tid3 > 0, "T17: fiber 3 started");

            const bool allWaiting = waitUntil([&]()
            {
                return getEvfNumThreads(rdram, runtime, eid) >= 3;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(allWaiting, "T17: all 3 fibers waiting on event flag");

            deleteSchedEvf(rdram, &runtime, eid);

            const bool allDone = waitUntil([&](){ return rdramSeq(rdram) >= 3u; }, std::chrono::milliseconds(2000));
            t.IsTrue(allDone, "T17: all 3 fibers completed after DeleteEventFlag");

            for (int i = 0; i < 3; ++i)
            {
                int32_t ret = -9999;
                std::memcpy(&ret, rdram.data() + kResultBase + static_cast<uint32_t>(i * 4), 4);
                t.Equals(ret, KE_WAIT_DELETE,
                         std::string("T17: fiber ") + std::to_string(i) + " received KE_WAIT_DELETE");
            }

            // eid is now invalid
            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(eid)); setRegU32(sc, 5, kReferScratch);
                ps2_syscalls::ReferEventFlagStatus(rdram.data(), &sc, &runtime);
                t.IsTrue(getRegS32(sc, 2) < 0, "T17: deleted evf returns error on Refer");
            }

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(2000));
            deleteSchedSema(rdram.data(), &runtime, doneSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 18: DeleteSema wakes all blocked waiters with KE_WAIT_DELETE
        // ------------------------------------------------------------------
        tc.Run("DeleteSema wakes all blocked waiters with KE_WAIT_DELETE", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00710000u, &stepWaitSemaRecordSignal);

            const int32_t workSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            const int32_t doneSid = createSchedSema(rdram.data(), &runtime, 0, 3);
            t.IsTrue(workSid > 0, "T18: work sema created"); t.IsTrue(doneSid > 0, "T18: done sema created");
            if (workSid <= 0 || doneSid <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, workSid);
                deleteSchedSema(rdram.data(), &runtime, doneSid);
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            {
                int32_t w = workSid, d = doneSid;
                std::memcpy(rdram.data() + kSlotWorkSid, &w, 4);
                std::memcpy(rdram.data() + kSlotDoneSid, &d, 4);
            }
            rdramSeqReset(rdram);
            for (int i = 0; i < 3; ++i)
            {
                int32_t bad = -9999;
                std::memcpy(rdram.data() + kResultBase + static_cast<uint32_t>(i * 4), &bad, 4);
            }

            const int32_t tid1 = startSchedWorker(rdram.data(), &runtime, 0x00710000u, 10, 0x004A0000u, 0x2000u);
            const int32_t tid2 = startSchedWorker(rdram.data(), &runtime, 0x00710000u, 10, 0x004A2000u, 0x2000u);
            const int32_t tid3 = startSchedWorker(rdram.data(), &runtime, 0x00710000u, 10, 0x004A4000u, 0x2000u);
            t.IsTrue(tid1 > 0, "T18: fiber 1 started"); t.IsTrue(tid2 > 0, "T18: fiber 2 started"); t.IsTrue(tid3 > 0, "T18: fiber 3 started");

            const bool allBlocked = waitUntil([&]()
            {
                return getSemaWaiters(rdram.data(), &runtime, workSid) >= 3;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(allBlocked, "T18: all 3 blocked on workSid");

            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(workSid));
                ps2_syscalls::DeleteSema(rdram.data(), &sc, &runtime);
                t.Equals(getRegS32(sc, 2), KE_OK, "T18: DeleteSema returned KE_OK");
            }

            const bool allDone = waitUntil([&](){ return rdramSeq(rdram) >= 3u; }, std::chrono::milliseconds(2000));
            t.IsTrue(allDone, "T18: all 3 fibers completed after DeleteSema");

            for (int i = 0; i < 3; ++i)
            {
                int32_t ret = -9999;
                std::memcpy(&ret, rdram.data() + kResultBase + static_cast<uint32_t>(i * 4), 4);
                t.Equals(ret, KE_WAIT_DELETE,
                         std::string("T18: fiber ") + std::to_string(i) + " received KE_WAIT_DELETE");
            }

            // workSid is now invalid
            {
                R5900Context sc{}; setRegU32(sc, 4, static_cast<uint32_t>(workSid));
                ps2_syscalls::PollSema(rdram.data(), &sc, &runtime);
                t.IsTrue(getRegS32(sc, 2) < 0, "T18: deleted sema returns error on Poll");
            }

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(2000));
            deleteSchedSema(rdram.data(), &runtime, doneSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Test 19: Resource cleanup — g_activeThreads drains to 0 after each exit
        // ------------------------------------------------------------------
        tc.Run("g_activeThreads drains to 0 after each fiber exits", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00720000u, &stepLogAndExit);

            rdramSeqReset(rdram);
            std::memset(rdram.data() + kRunLog, 0, 32u);

            for (int i = 0; i < 4; ++i)
            {
                const int before = g_activeThreads.load();
                t.Equals(before, 0, std::string("T19: no leaked threads before fiber ") + std::to_string(i));

                const uint32_t stackAddr = 0x004B0000u + static_cast<uint32_t>(i) * 0x2000u;
                const int32_t tid = startSchedWorker(rdram.data(), &runtime, 0x00720000u, 10, stackAddr, 0x2000u);
                t.IsTrue(tid > 0, std::string("T19: fiber ") + std::to_string(i) + " started");

                const bool drained = waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
                t.IsTrue(drained, std::string("T19: fiber ") + std::to_string(i) + " exited and g_activeThreads returned to 0");
            }

            t.Equals(rdramSeq(rdram), 4u, "T19: all 4 fibers logged");

            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

    }); // MiniTest::Case
}

// ---------------------------------------------------------------------------
// Scheduler race tests — park/wake, borrowed-worker, alarm shutdown, exit-hook re-entry, nested suspend.
// ---------------------------------------------------------------------------

// Forward-declare alarm worker lifecycle functions from Sync.h (internal header).
// These are defined in the runtime TU; declared here to avoid including Sync.h.
namespace ps2_syscalls
{
    void ensureAlarmWorkerRunning();
    void stopAlarmWorker();
}

namespace
{
    // -----------------------------------------------------------------------
    // New RDRAM slot constants for the race suite (0x4300-0x43FF range)
    // -----------------------------------------------------------------------
    static constexpr uint32_t kRSlotWorkSid      = 0x00004300u; // int32 work sema id
    static constexpr uint32_t kRSlotDoneSid      = 0x00004304u; // int32 done sema id
    static constexpr uint32_t kRSlotProgress     = 0x00004308u; // uint32 progress counter
    static constexpr uint32_t kRSlotStop         = 0x0000430Cu; // uint32 stop flag
    static constexpr uint32_t kRSlotStarted      = 0x00004310u; // uint32 started flag
    static constexpr uint32_t kRSlotTidParam     = 0x00004314u; // int32 a tid passed to a fiber
    static constexpr uint32_t kRSlotExitRan      = 0x00004318u; // uint32 set by exit handler
    static constexpr uint32_t kRSlotResult       = 0x0000431Cu; // int32 a syscall return value
    static constexpr uint32_t kRSlotAlarmFired   = 0x00004320u; // uint32 set by alarm callback
    static constexpr uint32_t kRSlotEntryReached = 0x00004324u; // uint32 fiber body reached

    // Entry / stack base constants
    static constexpr uint32_t kREntryBase = 0x00730000u;
    static constexpr uint32_t kRStackBase = 0x004C0000u;
    static constexpr uint32_t kRStackSize = 0x2000u;

    // -----------------------------------------------------------------------
    // Step function: stepWaitSemaLoopRace
    // Loops 500x: each iteration WaitSema(workSid) then increments kRSlotProgress.
    // After the loop, signals doneSid once and exits.
    // -----------------------------------------------------------------------
    static void stepWaitSemaLoopRace(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t workSid = 0, doneSid = 0;
        std::memcpy(&workSid, rdram + kRSlotWorkSid, 4);
        std::memcpy(&doneSid, rdram + kRSlotDoneSid, 4);

        constexpr int kRounds = 500;
        for (int i = 0; i < kRounds; ++i)
        {
            R5900Context sc{};
            setRegU32(sc, 4, static_cast<uint32_t>(workSid));
            ps2_syscalls::WaitSema(rdram, &sc, runtime);
            // On terminate, WaitSema unwinds via ThreadExitException — acceptable.
            uint32_t ctr = 0;
            std::memcpy(&ctr, rdram + kRSlotProgress, 4);
            ctr++;
            std::memcpy(rdram + kRSlotProgress, &ctr, 4);
        }

        if (doneSid > 0)
        {
            R5900Context sc2{};
            setRegU32(sc2, 4, static_cast<uint32_t>(doneSid));
            ps2_syscalls::SignalSema(rdram, &sc2, runtime);
        }
        ctx->pc = 0u;
    }

    // -----------------------------------------------------------------------
    // Step function: stepMarkEntryAndExit (for R2)
    // Sets kRSlotEntryReached=1 and exits immediately.
    // -----------------------------------------------------------------------
    static void stepMarkEntryAndExit(uint8_t *rdram, R5900Context *ctx, PS2Runtime * /*runtime*/)
    {
        uint32_t one = 1u;
        std::memcpy(rdram + kRSlotEntryReached, &one, 4);
        ctx->pc = 0u;
    }

    // -----------------------------------------------------------------------
    // Step function: stepAlarmCallbackMark (for R3)
    // Called as alarm callback; sets kRSlotAlarmFired=1.
    // -----------------------------------------------------------------------
    static void stepAlarmCallbackMark(uint8_t *rdram, R5900Context *ctx, PS2Runtime * /*runtime*/)
    {
        uint32_t one = 1u;
        std::memcpy(rdram + kRSlotAlarmFired, &one, 4);
        ctx->pc = 0u;
    }

    // -----------------------------------------------------------------------
    // Step functions for the exit-hook re-entrancy test.
    //
    // stepExitHandlerBlocking: the exit handler — runs as guest code on the
    // same fiber during exit. Calls SleepThread to park while Exiting.
    // After WakeupThread delivers a wakeup, marks kRSlotExitRan=1 and returns.
    // -----------------------------------------------------------------------
    static void stepExitHandlerBlocking(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        R5900Context sc{};
        ps2_syscalls::SleepThread(rdram, &sc, runtime);
        uint32_t one = 1u;
        std::memcpy(rdram + kRSlotExitRan, &one, 4);
        ctx->pc = 0u;
    }

    // stepRegisterExitHandlerThenExit: the fiber body.
    // Registers stepExitHandlerBlocking as an exit handler (at kREntryBase+0x3100
    // == 0x00733100), signals kRSlotStarted=1, then returns so the fiber begins
    // exiting and on_fiber_exit runs the handler.
    static void stepRegisterExitHandlerThenExit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        // Handler entry must match what the test registered via registerFunction().
        static constexpr uint32_t kHandlerEntry = 0x00733100u;
        R5900Context rc{};
        setRegU32(rc, 4, kHandlerEntry); // func
        setRegU32(rc, 5, 0u);            // arg
        ps2_syscalls::RegisterExitHandler(rdram, &rc, runtime);

        uint32_t one = 1u;
        std::memcpy(rdram + kRSlotStarted, &one, 4);
        ctx->pc = 0u;
    }

} // anonymous namespace

void register_scheduler_race_tests()
{
    MiniTest::Case("SchedulerRace", [](TestCase &tc)
    {
        // ------------------------------------------------------------------
        // R1: Concurrent wakeups while a fiber parks must never lose a wakeup.
        // ------------------------------------------------------------------
        tc.Run("R1: concurrent wakeups while a fiber parks never lose a wakeup", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00730000u, &stepWaitSemaLoopRace);

            const int32_t workSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            const int32_t doneSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(workSid > 0 && doneSid > 0, "R1: semas created");
            if (workSid <= 0 || doneSid <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, workSid);
                deleteSchedSema(rdram.data(), &runtime, doneSid);
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            std::memcpy(rdram.data() + kRSlotWorkSid, &workSid, 4);
            std::memcpy(rdram.data() + kRSlotDoneSid, &doneSid, 4);
            {
                uint32_t zero = 0u;
                std::memcpy(rdram.data() + kRSlotProgress, &zero, 4);
            }

            const int32_t tid = startSchedWorker(rdram.data(), &runtime,
                                                 0x00730000u, 10, kRStackBase, kRStackSize);
            t.IsTrue(tid > 0, "R1: worker started");

            // Wait until the fiber is parked on workSid at least once.
            const bool blocked = waitUntil([&]()
            {
                return getSemaWaiters(rdram.data(), &runtime, workSid) >= 1;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(blocked, "R1: worker parked on workSid");

            constexpr int kRounds = 500;
            std::atomic<bool> stopWakers{false};

            // (b) mid-park hammer: redundant external wakeups to maximise park-window race rate.
            std::thread wakeStorm([&]()
            {
                while (!stopWakers.load(std::memory_order_acquire))
                {
                    ps2sched::enqueue_external_wakeup(tid); // safe no-op unless genuinely Blocked
                }
            });

            // (a) signaler: deliver exactly kRounds permits as fast as possible.
            // WorkSid has max==1, so retry on KE_SEMA_OVF until the permit lands.
            std::thread signaler([&]()
            {
                for (int i = 0; i < kRounds; ++i)
                {
                    R5900Context sc{};
                    setRegU32(sc, 4, static_cast<uint32_t>(workSid));
                    ps2_syscalls::SignalSema(rdram.data(), &sc, &runtime);
                    while (getRegS32(sc, 2) != KE_OK)
                    {
                        std::this_thread::yield();
                        R5900Context sc2{};
                        setRegU32(sc2, 4, static_cast<uint32_t>(workSid));
                        ps2_syscalls::SignalSema(rdram.data(), &sc2, &runtime);
                        sc = sc2;
                    }
                }
            });

            // Pass condition: progress reaches kRounds (no lost wakeup).
            const bool reached = waitUntil([&]()
            {
                uint32_t c = 0;
                std::memcpy(&c, rdram.data() + kRSlotProgress, 4);
                return c >= static_cast<uint32_t>(kRounds);
            }, std::chrono::milliseconds(5000));

            stopWakers.store(true, std::memory_order_release);
            if (signaler.joinable()) signaler.join();
            if (wakeStorm.joinable()) wakeStorm.join();

            t.IsTrue(reached, "R1: fiber completed all 500 park/wake rounds (no lost wakeup)");

            // Drain: fiber signals doneSid and exits.
            const bool finished = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) == 0;
            }, std::chrono::milliseconds(2000));
            t.IsTrue(finished, "R1: worker fiber exits cleanly (g_activeThreads==0)");

            deleteSchedSema(rdram.data(), &runtime, workSid);
            deleteSchedSema(rdram.data(), &runtime, doneSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // R2: blocking syscall from non-fiber AsyncGuestScope does not crash
        // ------------------------------------------------------------------
        tc.Run("R2: blocking syscall from non-fiber AsyncGuestScope does not crash", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00731000u, &stepMarkEntryAndExit);

            const int32_t blockSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(blockSid > 0, "R2: sema created");

            std::atomic<int32_t> waitRet{-9999};
            std::atomic<bool>    waitThrew{false};
            std::atomic<bool>    waitReturned{false};

            // Run the non-fiber blocking WaitSema on a background host thread so the
            // main thread can release it. block_current() is a no-op for non-fiber
            // contexts, so WaitSema's Mesa re-check loop spins until count > 0.
            std::thread guestBorrow([&]()
            {
                try
                {
                    ps2sched::async_guest_begin();
                    R5900Context sc{};
                    setRegU32(sc, 4, static_cast<uint32_t>(blockSid));
                    ps2_syscalls::WaitSema(rdram.data(), &sc, &runtime);
                    waitRet.store(getRegS32(sc, 2), std::memory_order_release);
                    ps2sched::async_guest_end();
                }
                catch (...)
                {
                    waitThrew.store(true, std::memory_order_release);
                    ps2sched::async_guest_end();
                }
                waitReturned.store(true, std::memory_order_release);
            });

            // Give the borrow thread a moment to enter WaitSema's spin loop,
            // then feed it a permit so the Mesa re-check returns KE_OK.
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            {
                R5900Context sc{};
                setRegU32(sc, 4, static_cast<uint32_t>(blockSid));
                ps2_syscalls::SignalSema(rdram.data(), &sc, &runtime);
            }

            const bool returned = waitUntil([&]()
            {
                return waitReturned.load(std::memory_order_acquire);
            }, std::chrono::milliseconds(2000));

            if (guestBorrow.joinable()) guestBorrow.join();

            t.IsTrue(returned, "R2: non-fiber WaitSema returned (no hang, no crash)");
            t.IsFalse(waitThrew.load(std::memory_order_acquire),
                      "R2: non-fiber WaitSema did not throw");
            t.Equals(waitRet.load(std::memory_order_acquire), KE_OK,
                     "R2: non-fiber WaitSema acquired the signalled permit and returned KE_OK");

            // Scheduler must still be healthy: a normal fiber runs to completion.
            {
                uint32_t zero = 0u;
                std::memcpy(rdram.data() + kRSlotEntryReached, &zero, 4);
            }
            const int32_t tid = startSchedWorker(rdram.data(), &runtime,
                                                 0x00731000u, 10,
                                                 kRStackBase + kRStackSize, kRStackSize);
            t.IsTrue(tid > 0, "R2: post-test fiber started");
            const bool ran = waitUntil([&]()
            {
                uint32_t v = 0;
                std::memcpy(&v, rdram.data() + kRSlotEntryReached, 4);
                return v == 1u;
            }, std::chrono::milliseconds(2000));
            t.IsTrue(ran, "R2: scheduler still healthy — fiber runs after the non-fiber block");

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(2000));
            deleteSchedSema(rdram.data(), &runtime, blockSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // R3: stopAlarmWorker joins promptly and no alarm fires after stop
        // ------------------------------------------------------------------
        tc.Run("R3: stopAlarmWorker joins promptly and no alarm fires after stop", [](TestCase &t)
        {
            notifyRuntimeStop(); // ensures any prior alarm worker is stopped+joined
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00732000u, &stepAlarmCallbackMark);
            {
                uint32_t zero = 0u;
                std::memcpy(rdram.data() + kRSlotAlarmFired, &zero, 4);
            }

            // Queue a near-immediate alarm (1 tick ≈ 64us); this starts the alarm worker.
            R5900Context sc{};
            setRegU32(sc, 4, 1u);           // ticks = 1
            setRegU32(sc, 5, 0x00732000u);  // handler entry
            setRegU32(sc, 6, 0u);           // arg
            ps2_syscalls::SetAlarm(rdram.data(), &sc, &runtime);
            const int32_t alarmId = getRegS32(sc, 2);
            t.IsTrue(alarmId > 0, "R3: SetAlarm queued an alarm and started the worker");

            // Immediately stop the worker — must return promptly (joins, does not spin).
            const auto t0 = std::chrono::steady_clock::now();
            ps2_syscalls::stopAlarmWorker();
            const auto elapsed = std::chrono::steady_clock::now() - t0;
            t.IsTrue(elapsed < std::chrono::seconds(1),
                     "R3: stopAlarmWorker returned within 1s (joined, did not hang)");

            // After a clean join, the callback must not fire (worker exited before rdram was touched).
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            uint32_t fired = 0;
            std::memcpy(&fired, rdram.data() + kRSlotAlarmFired, 4);
            t.Equals(fired, 0u, "R3: alarm callback did not fire after stopAlarmWorker()");

            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // R4: exit handler that blocks resumes and the fiber finishes cleanly
        // ------------------------------------------------------------------
        tc.Run("R4: exit handler that blocks resumes and the fiber finishes cleanly", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            // Body entry: 0x00733000; handler entry: 0x00733100
            // (stepRegisterExitHandlerThenExit has kHandlerEntry=0x00733100 hardcoded)
            runtime.registerFunction(0x00733000u, &stepRegisterExitHandlerThenExit);
            runtime.registerFunction(0x00733100u, &stepExitHandlerBlocking);

            {
                uint32_t zero = 0u;
                std::memcpy(rdram.data() + kRSlotStarted, &zero, 4);
                std::memcpy(rdram.data() + kRSlotExitRan, &zero, 4);
            }

            const int32_t tid = startSchedWorker(rdram.data(), &runtime,
                                                 0x00733000u, 10,
                                                 kRStackBase + 2u * kRStackSize, kRStackSize);
            t.IsTrue(tid > 0, "R4: worker started");

            // Wait until the body ran and registered the blocking exit handler.
            const bool started = waitUntil([&]()
            {
                uint32_t v = 0;
                std::memcpy(&v, rdram.data() + kRSlotStarted, 4);
                return v == 1u;
            }, std::chrono::milliseconds(2000));
            t.IsTrue(started, "R4: fiber body ran and registered the blocking exit handler");

            // Give the exit handler time to reach SleepThread and park.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            t.IsTrue(g_activeThreads.load(std::memory_order_acquire) >= 1,
                     "R4: fiber still alive while its exit handler is blocked (not freed early)");

            // Wake the sleeping exit handler from the host.
            {
                R5900Context sc{};
                setRegU32(sc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::WakeupThread(rdram.data(), &sc, &runtime);
            }

            // Handler should complete and the fiber should finish.
            const bool handlerDone = waitUntil([&]()
            {
                uint32_t v = 0;
                std::memcpy(&v, rdram.data() + kRSlotExitRan, 4);
                return v == 1u;
            }, std::chrono::milliseconds(2000));
            t.IsTrue(handlerDone, "R4: blocking exit handler resumed and ran to completion");

            const bool drained = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) == 0;
            }, std::chrono::milliseconds(2000));
            t.IsTrue(drained, "R4: fiber freed cleanly after exit handler finished (no double-free/leak)");

            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // R5: nested SuspendThread requires matching ResumeThread count to wake
        // ------------------------------------------------------------------
        tc.Run("R5: nested SuspendThread requires matching ResumeThread count to wake", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            // Reuse stepProgressLoop (protocol-namespace step fn); fresh registration.
            runtime.registerFunction(0x00734000u, &stepProgressLoop);

            // Reset the protocol-suite slots that stepProgressLoop uses.
            {
                uint32_t zero = 0u;
                std::memcpy(rdram.data() + kProgressCtr, &zero, 4);
                std::memcpy(rdram.data() + kStopFlag,    &zero, 4);
                std::memcpy(rdram.data() + kStartedFlag, &zero, 4);
            }

            const int32_t tid = startSchedWorker(rdram.data(), &runtime,
                                                 0x00734000u, 10,
                                                 kRStackBase + 3u * kRStackSize, kRStackSize);
            t.IsTrue(tid > 0, "R5: worker started");

            // Wait until the loop is making progress.
            const bool running = waitUntil([&]()
            {
                uint32_t c = 0;
                std::memcpy(&c, rdram.data() + kProgressCtr, 4);
                return c > 0u;
            }, std::chrono::milliseconds(2000));
            t.IsTrue(running, "R5: fiber is making progress before suspend");

            auto progress = [&]()
            {
                uint32_t c = 0;
                std::memcpy(&c, rdram.data() + kProgressCtr, 4);
                return c;
            };
            auto suspend = [&]()
            {
                R5900Context sc{};
                setRegU32(sc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::SuspendThread(rdram.data(), &sc, &runtime);
                return getRegS32(sc, 2);
            };
            auto resume = [&]()
            {
                R5900Context sc{};
                setRegU32(sc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::ResumeThread(rdram.data(), &sc, &runtime);
                return getRegS32(sc, 2);
            };

            // Suspend twice (nested).
            t.Equals(suspend(), KE_OK, "R5: first SuspendThread OK");
            t.Equals(suspend(), KE_OK, "R5: second SuspendThread OK");

            // Progress must halt. Sample, wait, sample again.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const uint32_t afterSuspend = progress();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            t.Equals(progress(), afterSuspend, "R5: progress halted while suspended (count=2)");

            // Resume once → still suspended (PS2 count 2->1).
            t.Equals(resume(), KE_OK, "R5: first ResumeThread OK");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            t.Equals(progress(), afterSuspend, "R5: progress still halted after one resume (count=1)");

            // Resume again → count reaches 0 → clear_suspend wakes the fiber.
            t.Equals(resume(), KE_OK, "R5: second ResumeThread OK");
            const uint32_t c1 = progress();
            const bool resumed = waitUntil([&](){ return progress() > c1; },
                                           std::chrono::milliseconds(1000));
            t.IsTrue(resumed, "R5: progress resumes after matching second resume (count=0)");

            // Stop the loop and drain.
            {
                uint32_t one = 1u;
                std::memcpy(rdram.data() + kStopFlag, &one, 4);
            }
            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(2000));

            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

    }); // MiniTest::Case("SchedulerRace")
}

// ---------------------------------------------------------------------------
// Scheduler stress tests — R6 sleep/wake-storm and R7 borrowed-worker WaitSema.
// ---------------------------------------------------------------------------

namespace
{
    // -----------------------------------------------------------------------
    // RDRAM slot constants for R6/R7 (0x4330–0x434F range)
    // -----------------------------------------------------------------------
    static constexpr uint32_t kR6Counter  = 0x00004330u; // R6: SleepThread iteration count (u32)
    static constexpr uint32_t kR6Done     = 0x00004334u; // R6: set to 1 when the loop completes (u32)
    static constexpr uint32_t kR7Started  = 0x00004338u; // R7: fiber sets to 1 when it begins running (u32)
    static constexpr uint32_t kR7Signalled = 0x0000433Cu; // R7: fiber sets to 1 just after SignalSema (u32)

    // stepSleepLoopN: SleepThread N times; bump kR6Counter each return; set kR6Done at end.
    static void stepSleepLoopN(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        constexpr uint32_t kN = 500u;
        for (uint32_t i = 0; i < kN; ++i)
        {
            R5900Context sc{};
            ps2_syscalls::SleepThread(rdram, &sc, runtime);
            uint32_t c = 0;
            std::memcpy(&c, rdram + kR6Counter, 4);
            c++;
            std::memcpy(rdram + kR6Counter, &c, 4);
        }
        uint32_t one = 1u;
        std::memcpy(rdram + kR6Done, &one, 4);
        ctx->pc = 0u;
    }

    // stepSignalAfterDelay: mark started, yield-spin, then SignalSema(workSid) and exit.
    // This fiber is the ONLY producer of the sema permit the borrowed host worker waits on.
    static void stepSignalAfterDelay(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t one = 1u;
        std::memcpy(rdram + kR7Started, &one, 4);

        for (int i = 0; i < 100; ++i)
        {
            runtime->shouldPreemptGuestExecution();
        }

        int32_t workSid = 0;
        std::memcpy(&workSid, rdram + kSlotWorkSid, 4);
        R5900Context sc{};
        setRegU32(sc, 4, static_cast<uint32_t>(workSid));
        ps2_syscalls::SignalSema(rdram, &sc, runtime);

        std::memcpy(rdram + kR7Signalled, &one, 4);
        ctx->pc = 0u;
    }

} // anonymous namespace

void register_scheduler_stress_tests()
{
    MiniTest::Case("SchedulerStress", [](TestCase &tc)
    {
        // ------------------------------------------------------------------
        // R6: SleepThread/WakeupThread no-count wake-storm
        // ------------------------------------------------------------------
        tc.Run("SleepThread/WakeupThread wake-storm never drops a no-count wake", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00740000u, &stepSleepLoopN);

            // Reset the shared slots.
            rdramWrite32(rdram, kR6Counter, 0u);
            rdramWrite32(rdram, kR6Done,    0u);

            // Launch the sleeping fiber. It loops SleepThread 500 times.
            const int32_t tid = startSchedWorker(rdram.data(), &runtime,
                                                 0x00740000u, 10,
                                                 0x004CA000u, 0x2000u);
            t.IsTrue(tid > 0, "R6: sleep-loop fiber should start");
            if (tid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // Tight wake-storm. We deliberately DO NOT wait for THS_WAIT before firing —
            // doing so would close the arm_park/publish race window the test exists to probe.
            // Fire many more wakeups than iterations: SleepThread re-sleeps immediately so
            // duplicate/early wakeups are harmless (no-count: an extra wake just bumps
            // wakeupCount and the next SleepThread consumes it). The point is that across
            // 500 iterations, many WakeupThread calls land in the pre-park window; not one
            // may be lost or the fiber hangs forever in SleepThread.
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2500);
            while (std::chrono::steady_clock::now() < deadline)
            {
                uint32_t done = 0;
                std::memcpy(&done, rdram.data() + kR6Done, 4);
                if (done) break;

                R5900Context wc{};
                setRegU32(wc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::WakeupThread(rdram.data(), &wc, &runtime);
                // No sleep, no THS_WAIT confirmation — keep the loop as tight as possible so
                // wakeups race against the fiber's arm_park -> block_current transition.
            }

            // The fiber must have completed all 500 iterations.
            const bool finished = waitUntil([&]()
            {
                uint32_t done = 0;
                std::memcpy(&done, rdram.data() + kR6Done, 4);
                return done == 1u;
            }, std::chrono::milliseconds(500));

            uint32_t counter = 0;
            std::memcpy(&counter, rdram.data() + kR6Counter, 4);

            t.IsTrue(finished, "R6: sleep-loop fiber should complete all 500 SleepThread iterations (no wake lost)");
            t.Equals(counter, 500u, "R6: fiber should have returned from SleepThread exactly 500 times");

            // Fiber should have exited and decremented g_activeThreads.
            const bool drained = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) <= 0;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(drained, "R6: sleep-loop fiber should exit and drain g_activeThreads");

            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // R7: borrowed-worker WaitSema, permit produced only by a fiber
        // ------------------------------------------------------------------
        tc.Run("borrowed-worker WaitSema unblocks via fiber-only SignalSema", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00748000u, &stepSignalAfterDelay);

            // count=0, max=1: the borrowed worker must block; only the fiber can produce a permit.
            const int32_t workSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(workSid > 0, "R7: work sema should be created");
            if (workSid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            rdramWrite32(rdram, kSlotWorkSid,  static_cast<uint32_t>(workSid));
            rdramWrite32(rdram, kR7Started,    0u);
            rdramWrite32(rdram, kR7Signalled,  0u);

            // Start the signalling fiber FIRST so it is enqueued Ready before the borrowed
            // worker grabs the token. The token-release loop handles either interleaving,
            // but enqueuing first keeps the common path deterministic.
            const int32_t fiberTid = startSchedWorker(rdram.data(), &runtime,
                                                      0x00748000u, 10,
                                                      0x004CC000u, 0x2000u);
            t.IsTrue(fiberTid > 0, "R7: signalling fiber should start");
            if (fiberTid <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, workSid);
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // Background host thread = a borrowed worker (IRQ-handler style). It acquires the
            // guest token, calls WaitSema on the count=0 sema, and records the result.
            std::atomic<bool> workerDone{false};
            std::atomic<int32_t> workerRet{-9999};
            std::atomic<bool> workerThrew{false};

            std::thread borrowedWorker([&]()
            {
                try
                {
                    g_currentThreadId = -1; // non-fiber host worker (matches IRQ/alarm workers)
                    ps2sched::async_guest_begin();   // acquire the guest token (AsyncGuestScope-equivalent)
                    R5900Context wc{};
                    setRegU32(wc, 4, static_cast<uint32_t>(workSid));
                    ps2_syscalls::WaitSema(rdram.data(), &wc, &runtime); // count=0 -> borrowed-worker block/retry path
                    workerRet.store(getRegS32(wc, 2), std::memory_order_release);
                    ps2sched::async_guest_end();     // release the guest token
                }
                catch (...)
                {
                    workerThrew.store(true, std::memory_order_release);
                    // Best-effort token release on the exception path so we never wedge the executor.
                    ps2sched::async_guest_end();
                }
                workerDone.store(true, std::memory_order_release);
            });

            // If the borrowed-worker path is present: WaitSema drops the token, the fiber runs,
            // signals workSid, the worker re-checks count, consumes the permit, returns KE_OK.
            // If absent: the worker spins holding the token, the fiber never runs,
            // and workerDone never becomes true -> this wait times out -> FAIL.
            const bool finished = waitUntil([&]()
            {
                return workerDone.load(std::memory_order_acquire);
            }, std::chrono::milliseconds(3000));

            if (!finished)
            {
                // Deadlock escape hatch so we don't hang the whole test binary on failure:
                // signal the sema from this (main) thread to release the borrowed worker,
                // then join. The assertion below still records the failure.
                R5900Context sc{};
                setRegU32(sc, 4, static_cast<uint32_t>(workSid));
                ps2_syscalls::SignalSema(rdram.data(), &sc, &runtime);
            }

            if (borrowedWorker.joinable())
            {
                borrowedWorker.join();
            }

            t.IsTrue(finished, "R7: borrowed-worker WaitSema should complete (the token is released so a fiber can signal)");
            t.IsFalse(workerThrew.load(std::memory_order_acquire), "R7: borrowed worker should not throw");
            t.Equals(workerRet.load(std::memory_order_acquire), KE_OK, "R7: borrowed-worker WaitSema should return KE_OK");

            // The permit must have come from the fiber, not from the deadlock escape hatch.
            uint32_t signalled = 0;
            std::memcpy(&signalled, rdram.data() + kR7Signalled, 4);
            t.Equals(signalled, 1u, "R7: the fiber (not the main thread) should have produced the permit");

            // Everything should drain.
            const bool drained = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) <= 0;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(drained, "R7: signalling fiber should exit and drain g_activeThreads");

            deleteSchedSema(rdram.data(), &runtime, workSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

    }); // MiniTest::Case("SchedulerStress")
}

// ---------------------------------------------------------------------------
// Scheduler supplement tests — S1 (vsync wait-list cleanup) and S2 (update_priority queued fiber).
// ---------------------------------------------------------------------------

namespace
{
    // -----------------------------------------------------------------------
    // New RDRAM slot constants for S1 (0x4350-0x4368 range)
    // -----------------------------------------------------------------------
    static constexpr uint32_t kS1SlotEntered = 0x00004350u; // uint32: set to 1 by stepVsyncWaitForever before first WaitForNextVSyncTick
    static constexpr uint32_t kS1SlotWorkSid = 0x00004354u; // int32: sema id fiber B blocks on (count 0)
    static constexpr uint32_t kS1SlotBWoke   = 0x00004358u; // uint32: set to 1 by fiber B if its WaitSema ever returns

    // stepVsyncWaitForever — fiber body for S1's target fiber A.
    // Marks kS1SlotEntered=1, then loops calling WaitForNextVSyncTick so it stays
    // parked in g_vsync_waitList until terminated.
    static void stepVsyncWaitForever(uint8_t *rdram, R5900Context * /*ctx*/, PS2Runtime *runtime)
    {
        uint32_t one = 1u;
        std::memcpy(rdram + kS1SlotEntered, &one, 4);
        for (;;)
        {
            ps2_syscalls::WaitForNextVSyncTick(rdram, runtime);
            // Belt-and-braces: observe terminate request promptly even if a tick
            // fires and wakes us before TerminateThread delivers the request.
            runtime->shouldPreemptGuestExecution();
        }
        // unreachable; ThreadExitException unwinds out.
    }

    // stepWaitSemaBait — fiber body for S1's fiber B.
    // Reads the sema id from kS1SlotWorkSid, blocks on it (count 0 = never wakes
    // normally). If it ever returns from WaitSema it sets kS1SlotBWoke=1.
    static void stepWaitSemaBait(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t workSid = 0;
        std::memcpy(&workSid, rdram + kS1SlotWorkSid, 4);
        R5900Context sc{};
        setRegU32(sc, 4, static_cast<uint32_t>(workSid));
        ps2_syscalls::WaitSema(rdram, &sc, runtime); // count 0 -> blocks
        uint32_t one = 1u;
        std::memcpy(rdram + kS1SlotBWoke, &one, 4); // reached only on spurious wakeup
        ctx->pc = 0u;
    }

} // anonymous namespace

void register_scheduler_vsync_priority_tests()
{
    MiniTest::Case("SchedulerVSyncAndPriority", [](TestCase &tc)
    {
        // ------------------------------------------------------------------
        // S1: vsync wait-list stale-tid cleanup
        // ------------------------------------------------------------------
        tc.Run("vsync wait-list drops a terminated waiter's tid", [](TestCase &t)
        {
            namespace istate = ps2_syscalls::interrupt_state;

            notifyRuntimeStop();
            ps2sched::scheduler_init();

            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00750000u, &stepVsyncWaitForever);
            runtime.registerFunction(0x00752000u, &stepWaitSemaBait);

            // Clear S1 slots.
            {
                uint32_t z = 0u;
                std::memcpy(rdram.data() + kS1SlotEntered, &z, 4);
                std::memcpy(rdram.data() + kS1SlotBWoke,   &z, 4);
            }

            // Sema that fiber B parks on (count 0, max 1).
            const int32_t workSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(workSid > 0, "S1: bait sema created");
            if (workSid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }
            {
                int32_t v = workSid;
                std::memcpy(rdram.data() + kS1SlotWorkSid, &v, 4);
            }

            // Start fiber A (the vsync waiter). WaitForNextVSyncTick calls
            // EnsureVSyncWorkerRunning internally.
            const int32_t tidA = startSchedWorker(
                rdram.data(), &runtime, 0x00750000u, 30, 0x004D0000u, 0x2000u);
            t.IsTrue(tidA > 0, "S1: vsync-waiter fiber A started");
            if (tidA <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, workSid);
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // Wait until A has set kS1SlotEntered AND its tid appears in g_vsync_waitList.
            const bool aQueued = waitUntil([&]()
            {
                uint32_t entered = 0u;
                std::memcpy(&entered, rdram.data() + kS1SlotEntered, 4);
                if (!entered) return false;
                std::lock_guard<std::mutex> lk(istate::g_vsync_flag_mutex);
                for (const auto &[tid, token] : istate::g_vsync_waitList)
                    if (tid == tidA) return true;
                return false;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(aQueued, "S1: fiber A is queued in g_vsync_waitList");

            // Sanity: confirm A's tid is currently in the list.
            {
                std::lock_guard<std::mutex> lk(istate::g_vsync_flag_mutex);
                bool found = false;
                for (const auto &[tid, token] : istate::g_vsync_waitList) if (tid == tidA) found = true;
                t.IsTrue(found, "S1: pre-terminate, A's tid is in the wait-list");
            }

            // Terminate fiber A. TerminateThread wakes A (request_terminate) and joins
            // it (join_fiber). A unwinds out of WaitForNextVSyncTick; the wake path
            // removes A's tid from g_vsync_waitList before signaling B.
            {
                R5900Context termCtx{};
                setRegU32(termCtx, 4, static_cast<uint32_t>(tidA));
                ps2_syscalls::TerminateThread(rdram.data(), &termCtx, &runtime);
                t.Equals(getRegS32(termCtx, 2), KE_OK, "S1: TerminateThread(A) returns KE_OK");
            }

            // Wait for A to finish (TerminateThread joined it, but g_activeThreads
            // decrement may trail slightly).
            const bool aGone = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) == 0;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(aGone, "S1: fiber A finished after TerminateThread");

            // A's tid must no longer be in g_vsync_waitList.
            {
                std::lock_guard<std::mutex> lk(istate::g_vsync_flag_mutex);
                bool stale = false;
                for (const auto &[tid, token] : istate::g_vsync_waitList) if (tid == tidA) stale = true;
                t.IsFalse(stale, "S1: terminated fiber A's tid was removed from g_vsync_waitList");
                t.IsTrue(istate::g_vsync_waitList.empty(),
                         "S1: g_vsync_waitList is empty after the sole waiter is terminated");
            }

            // SECONDARY ASSERTION (recycle tripwire): start fiber B on the same tid pool.
            // B blocks on a count-0 sema and must NOT be spuriously woken by a stale vsync
            // wakeup targeting A's (possibly recycled) tid.
            const int32_t tidB = startSchedWorker(
                rdram.data(), &runtime, 0x00752000u, 30, 0x004D2000u, 0x2000u);
            t.IsTrue(tidB > 0, "S1: bait fiber B started");

            // Wait until B is actually blocked on the sema.
            const bool bBlocked = waitUntil([&]()
            {
                return getSemaWaiters(rdram.data(), &runtime, workSid) >= 1;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(bBlocked, "S1: bait fiber B is blocked on the sema");

            // Ensure the vsync worker is running and let it tick several times
            // (~60 ms >= 2-3 real ticks at the 16.667 ms cadence).
            ps2_syscalls::EnsureVSyncWorkerRunning(rdram.data(), &runtime);
            std::this_thread::sleep_for(std::chrono::milliseconds(60));

            // B must NOT have been spuriously woken by a stale vsync wakeup.
            {
                uint32_t bWoke = 0u;
                std::memcpy(&bWoke, rdram.data() + kS1SlotBWoke, 4);
                t.Equals(bWoke, 0u,
                         "S1: bait fiber B was not spuriously woken by a stale vsync tid");
            }

            // Cleanup: signal the sema so B can unwind cleanly, then shut down.
            {
                R5900Context sigCtx{};
                setRegU32(sigCtx, 4, static_cast<uint32_t>(workSid));
                ps2_syscalls::SignalSema(rdram.data(), &sigCtx, &runtime);
            }
            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));

            deleteSchedSema(rdram.data(), &runtime, workSid);
            ps2sched::scheduler_shutdown(); // joins the vsync worker via stopInterruptWorker()
            runtime.requestStop();
            notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // S2: update_priority re-sorts a queued (Ready) fiber
        // ------------------------------------------------------------------
        tc.Run("ChangeThreadPriority re-sorts a queued (Ready) fiber to the front", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            // Reuse stepLogAndExit registered at a fresh entry address for S2.
            runtime.registerFunction(0x00758000u, &stepLogAndExit);

            rdramSeqReset(rdram);
            std::memset(rdram.data() + kRunLog, 0, 32u);

            // Freeze the executor so all three fibers sit Ready in the run queue and
            // none can run. update_priority must take its wasQueued==true branch
            // (remove_locked + enqueue_locked) for the queued A.
            ps2sched::async_guest_begin();

            // Start A(20), B(15), C(10). Queue order ascending by priority: [C(10), B(15), A(20)].
            const int32_t tidA = startSchedWorker(rdram.data(), &runtime, 0x00758000u, 20, 0x004D4000u, 0x2000u);
            const int32_t tidB = startSchedWorker(rdram.data(), &runtime, 0x00758000u, 15, 0x004D6000u, 0x2000u);
            const int32_t tidC = startSchedWorker(rdram.data(), &runtime, 0x00758000u, 10, 0x004D8000u, 0x2000u);
            t.IsTrue(tidA > 0, "S2: A started");
            t.IsTrue(tidB > 0, "S2: B started");
            t.IsTrue(tidC > 0, "S2: C started");

            // Change A's priority from 20 to 5 while A is queued Ready.
            // Expected new order: [A(5), C(10), B(15)].
            {
                R5900Context sc{};
                setRegU32(sc, 4, static_cast<uint32_t>(tidA));
                setRegU32(sc, 5, 5u);
                ps2_syscalls::ChangeThreadPriority(rdram.data(), &sc, &runtime);
                t.Equals(getRegS32(sc, 2), KE_OK, "S2: ChangeThreadPriority(A,5) returns KE_OK");
            }

            // Release the executor; fibers now run in the re-sorted priority order.
            ps2sched::async_guest_end();

            const bool allDone = waitUntil([&](){ return rdramSeq(rdram) >= 3u; },
                                           std::chrono::milliseconds(2000));
            t.IsTrue(allDone, "S2: all 3 fibers completed");

            int32_t log[3] = {};
            std::memcpy(log, rdram.data() + kRunLog, 12);
            t.Equals(log[0], tidA, "S2: A (re-sorted to prio 5) runs first");
            t.Equals(log[1], tidC, "S2: C (prio 10) runs second");
            t.Equals(log[2], tidB, "S2: B (prio 15) runs last");

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            ps2sched::scheduler_shutdown();
            runtime.requestStop();
            notifyRuntimeStop();
        });

    }); // MiniTest::Case("SchedulerVSyncAndPriority")
}

// ---------------------------------------------------------------------------
// Scheduler lifecycle tests — executor thread assertion, join-fiber priority
// stability, exit-hook completion, and multi-fiber shutdown ordering.
// ---------------------------------------------------------------------------

// Defined in ps2_scheduler.cpp (declared in the private header ps2_fiber.h that
// the test target cannot include). Linked from ps2_runtime.
bool ps2fiber_on_executor_thread();

namespace
{
    // ---- V6 RDRAM slot constants (0x4370-0x43B0) ----
    // U1
    static constexpr uint32_t kU1SlotFiberOnExec = 0x00004370u; // int32: 1 if fiber saw on_executor==true
    static constexpr uint32_t kU1SlotHostOnExec  = 0x00004374u; // int32: host thread's on_executor result

    // U2
    static constexpr uint32_t kU2SlotJoinerTid   = 0x00004378u; // int32: A's tid (joiner)
    static constexpr uint32_t kU2SlotTargetTid   = 0x0000437Cu; // int32: B's tid (join target)
    static constexpr uint32_t kU2SlotSpinning    = 0x00004380u; // uint32: B sets 1 after first yield (window open)
    static constexpr uint32_t kU2SlotJoinerPrio  = 0x00004384u; // int32: A's current_priority after join

    // U3
    static constexpr uint32_t kU3SlotWorkSid     = 0x00004388u; // int32: sema B blocks on (count 0)
    static constexpr uint32_t kU3SlotExitRan     = 0x0000438Cu; // uint32: set to 1 by B's exit handler

    // U4 (4 fibers: body-sentinel + hook-sentinel each)
    static constexpr uint32_t kU4SlotWorkSid     = 0x00004390u; // int32: sema all U4 fibers block on (count 0)
    static constexpr uint32_t kU4BodyBase        = 0x00004394u; // uint32[4]: body-reached sentinels
    static constexpr uint32_t kU4HookBase        = 0x000043A4u; // uint32[4]: exit-handler-ran sentinels
    static constexpr int      kU4FiberCount      = 4;

    // ---- V6 step functions ----

    // U1: runs on the executor thread; records whether ps2fiber_on_executor_thread() is true.
    static void stepRecordOnExecutor(uint8_t *rdram, R5900Context *ctx, PS2Runtime * /*runtime*/)
    {
        const int32_t onExec = ps2fiber_on_executor_thread() ? 1 : 0;
        std::memcpy(rdram + kU1SlotFiberOnExec, &onExec, 4);
        ctx->pc = 0u;
    }

    // U2 target (fiber B): yields 20 times, setting kU2SlotSpinning=1 after the first yield, then exits.
    static void stepU2TargetYieldLoop(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        for (int i = 0; i < 20; ++i)
        {
            runtime->shouldPreemptGuestExecution(); // cooperative yield point
            if (i == 0)
            {
                uint32_t one = 1u;
                std::memcpy(rdram + kU2SlotSpinning, &one, 4); // window is open
            }
        }
        ctx->pc = 0u;
    }

    // U2 joiner (fiber A): reads B's tid, calls TerminateThread(B), then reads its own current_priority.
    static void stepU2JoinerTerminateThenLog(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t targetTid = 0;
        std::memcpy(&targetTid, rdram + kU2SlotTargetTid, 4);

        R5900Context tc{};
        setRegU32(tc, 4, static_cast<uint32_t>(targetTid));
        ps2_syscalls::TerminateThread(rdram, &tc, runtime); // request_terminate + join_fiber

        // Read OUR OWN current_priority (TH_SELF = tid 0).
        R5900Context rc{};
        setRegU32(rc, 4, 0u);            // TH_SELF
        setRegU32(rc, 5, kReferScratch); // status struct dest
        ps2_syscalls::ReferThreadStatus(rdram, &rc, runtime);
        int32_t curPrio = 0;
        std::memcpy(&curPrio, rdram + kReferScratch + 0x18, 4); // current_priority @0x18
        std::memcpy(rdram + kU2SlotJoinerPrio, &curPrio, 4);

        ctx->pc = 0u;
    }

    // U3 exit handler: non-blocking; writes sentinel to kU3SlotExitRan.
    static void stepU3ExitHandlerMark(uint8_t *rdram, R5900Context *ctx, PS2Runtime * /*runtime*/)
    {
        uint32_t one = 1u;
        std::memcpy(rdram + kU3SlotExitRan, &one, 4);
        ctx->pc = 0u;
    }

    // U3 worker body: registers stepU3ExitHandlerMark as exit handler, then blocks on kU3SlotWorkSid.
    static void stepU3RegisterHandlerThenBlock(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        static constexpr uint32_t kU3HandlerEntry = 0x00778100u;
        R5900Context rc{};
        setRegU32(rc, 4, kU3HandlerEntry); // func
        setRegU32(rc, 5, 0u);              // arg
        ps2_syscalls::RegisterExitHandler(rdram, &rc, runtime);

        int32_t workSid = 0;
        std::memcpy(&workSid, rdram + kU3SlotWorkSid, 4);
        R5900Context sc{};
        setRegU32(sc, 4, static_cast<uint32_t>(workSid));
        ps2_syscalls::WaitSema(rdram, &sc, runtime); // blocks; never wakes normally
        ctx->pc = 0u;
    }

    // U4 exit handler: reads slot index from $a0 (handler arg), writes kU4HookBase+idx*4=1.
    static void stepU4ExitHandlerMark(uint8_t *rdram, R5900Context *ctx, PS2Runtime * /*runtime*/)
    {
        const uint32_t idx = ::getRegU32(ctx, 4); // $a0 == handler.arg (slot index)
        uint32_t one = 1u;
        if (idx < static_cast<uint32_t>(kU4FiberCount))
            std::memcpy(rdram + kU4HookBase + idx * 4u, &one, 4);
        ctx->pc = 0u;
    }

    // U4 worker body: reads slot index from $a0 (StartThread arg), writes body sentinel,
    // registers exit handler forwarding the same index, then blocks on kU4SlotWorkSid.
    static void stepU4RegisterHandlerThenBlock(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        static constexpr uint32_t kU4HandlerEntry = 0x00780100u;
        const uint32_t idx = ::getRegU32(ctx, 4); // $a0 == StartThread arg == slot index

        uint32_t one = 1u;
        if (idx < static_cast<uint32_t>(kU4FiberCount))
            std::memcpy(rdram + kU4BodyBase + idx * 4u, &one, 4);

        R5900Context rc{};
        setRegU32(rc, 4, kU4HandlerEntry); // func
        setRegU32(rc, 5, idx);             // arg -> forwarded to handler's $a0
        ps2_syscalls::RegisterExitHandler(rdram, &rc, runtime);

        int32_t workSid = 0;
        std::memcpy(&workSid, rdram + kU4SlotWorkSid, 4);
        R5900Context sc{};
        setRegU32(sc, 4, static_cast<uint32_t>(workSid));
        ps2_syscalls::WaitSema(rdram, &sc, runtime); // blocks; shutdown terminates it
        ctx->pc = 0u;
    }

} // anonymous namespace

void register_scheduler_lifecycle_tests()
{
    MiniTest::Case("SchedulerLifecycle", [](TestCase &tc)
    {
        // ------------------------------------------------------------------
        // U1: ps2fiber_on_executor_thread returns true on the executor, false elsewhere
        // ------------------------------------------------------------------
        tc.Run("U1: ps2fiber_on_executor_thread is true on the executor, false elsewhere", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00760000u, &stepRecordOnExecutor);

            // Sentinels: -1 means "not written yet" so we can tell the fiber actually ran.
            { int32_t init = -1; std::memcpy(rdram.data() + kU1SlotFiberOnExec, &init, 4); }
            { int32_t init = -1; std::memcpy(rdram.data() + kU1SlotHostOnExec,  &init, 4); }

            // Host-thread side: a plain std::thread is NOT the guest executor thread,
            // so ps2fiber_on_executor_thread() must return false there.
            std::thread hostProbe([&]()
            {
                const int32_t r = ps2fiber_on_executor_thread() ? 1 : 0;
                std::memcpy(rdram.data() + kU1SlotHostOnExec, &r, 4);
            });
            hostProbe.join();

            // Fiber side: the body runs on the guest executor thread.
            const int32_t tid = startSchedWorker(rdram.data(), &runtime,
                                                 0x00760000u, 10, 0x004D8000u, 0x2000u);
            t.IsTrue(tid > 0, "U1: probe fiber started");

            const bool fiberWrote = waitUntil([&]()
            {
                int32_t v = -1;
                std::memcpy(&v, rdram.data() + kU1SlotFiberOnExec, 4);
                return v != -1;
            }, std::chrono::milliseconds(2000));
            t.IsTrue(fiberWrote, "U1: probe fiber recorded its on-executor result");

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));

            int32_t fiberSaw = -1, hostSaw = -1;
            std::memcpy(&fiberSaw, rdram.data() + kU1SlotFiberOnExec, 4);
            std::memcpy(&hostSaw,  rdram.data() + kU1SlotHostOnExec,  4);

            t.Equals(fiberSaw, 1, "U1: fiber on the executor thread sees on_executor==true");
            t.Equals(hostSaw,  0, "U1: a non-executor host thread sees on_executor==false");

            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // U2: join_fiber restores the concurrently-changed joiner priority
        // ------------------------------------------------------------------
        tc.Run("U2: join_fiber restores the concurrently-changed joiner priority, not a stale snapshot", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00768000u, &stepU2JoinerTerminateThenLog);
            runtime.registerFunction(0x00770000u, &stepU2TargetYieldLoop);

            { uint32_t z = 0u;  std::memcpy(rdram.data() + kU2SlotSpinning,   &z, 4); }
            { int32_t neg = -1; std::memcpy(rdram.data() + kU2SlotJoinerPrio, &neg, 4); }

            // Start target B (prio 50) first so it exists when A joins it.
            const int32_t tidB = startSchedWorker(rdram.data(), &runtime, 0x00770000u, 50, 0x004E0000u, 0x2000u);
            t.IsTrue(tidB > 0, "U2: target B started");
            { int32_t v = tidB; std::memcpy(rdram.data() + kU2SlotTargetTid, &v, 4); }

            // Start joiner A (prio 60). A immediately calls TerminateThread(B) -> join_fiber(B).
            const int32_t tidA = startSchedWorker(rdram.data(), &runtime, 0x00768000u, 60, 0x004DC000u, 0x2000u);
            t.IsTrue(tidA > 0, "U2: joiner A started");
            { int32_t v = tidA; std::memcpy(rdram.data() + kU2SlotJoinerTid, &v, 4); }

            // Host thread: once B has yielded at least once (join window provably open),
            // fire ChangeThreadPriority(A, 10). This must land while A is inside join_fiber.
            std::thread reprio([&]()
            {
                const bool windowOpen = waitUntil([&]()
                {
                    uint32_t s = 0; std::memcpy(&s, rdram.data() + kU2SlotSpinning, 4);
                    return s == 1u;
                }, std::chrono::milliseconds(2000));
                if (!windowOpen) return;
                R5900Context sc{};
                setRegU32(sc, 4, static_cast<uint32_t>(tidA));
                setRegU32(sc, 5, 10u);
                ps2_syscalls::ChangeThreadPriority(rdram.data(), &sc, &runtime);
            });

            // Wait for both fibers to drain (A finishes after the join returns and it logs prio).
            const bool drained = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) == 0;
            }, std::chrono::milliseconds(4000));
            reprio.join();
            t.IsTrue(drained, "U2: both fibers finished");

            int32_t joinerPrio = -1;
            std::memcpy(&joinerPrio, rdram.data() + kU2SlotJoinerPrio, 4);

            // A's priority after the join must be the concurrently-set value, not the
            // stale original captured before the join began.
            t.Equals(joinerPrio, 10, "U2: joiner priority is the concurrently-set value (10), not stale (60)");

            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // U3: TerminateThread(other) runs the target fiber's exit handler to completion
        // ------------------------------------------------------------------
        tc.Run("U3: TerminateThread(other) runs the target fiber's exit handler to completion", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00778000u, &stepU3RegisterHandlerThenBlock);
            runtime.registerFunction(0x00778100u, &stepU3ExitHandlerMark);

            const int32_t workSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(workSid > 0, "U3: work sema created");
            if (workSid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }
            { int32_t v = workSid; std::memcpy(rdram.data() + kU3SlotWorkSid, &v, 4); }
            { uint32_t z = 0u;    std::memcpy(rdram.data() + kU3SlotExitRan,  &z, 4); }

            const int32_t workerTid = startSchedWorker(rdram.data(), &runtime, 0x00778000u, 10, 0x004E4000u, 0x2000u);
            t.IsTrue(workerTid > 0, "U3: worker started");

            const bool blocked = waitUntil([&]()
            {
                return getSemaWaiters(rdram.data(), &runtime, workSid) >= 1;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(blocked, "U3: worker blocked on workSid after registering its exit handler");

            // Terminate from the host: request_terminate wakes the worker, it unwinds via
            // ThreadExitException, fiber_trampoline runs the exit hook -> our handler.
            {
                R5900Context termCtx{};
                setRegU32(termCtx, 4, static_cast<uint32_t>(workerTid));
                ps2_syscalls::TerminateThread(rdram.data(), &termCtx, &runtime);
                t.Equals(getRegS32(termCtx, 2), KE_OK, "U3: TerminateThread returns KE_OK");
            }

            const bool drained = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) == 0;
            }, std::chrono::milliseconds(3000));
            t.IsTrue(drained, "U3: g_activeThreads reached 0 after termination");

            uint32_t exitRan = 0u;
            std::memcpy(&exitRan, rdram.data() + kU3SlotExitRan, 4);
            t.Equals(exitRan, 1u, "U3: terminated fiber's exit handler ran");

            deleteSchedSema(rdram.data(), &runtime, workSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // U4: scheduler_shutdown lets Exiting fibers complete their exit handlers
        // ------------------------------------------------------------------
        tc.Run("U4: scheduler_shutdown lets Exiting fibers complete their exit handlers", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x00780000u, &stepU4RegisterHandlerThenBlock);
            runtime.registerFunction(0x00780100u, &stepU4ExitHandlerMark);

            const int32_t workSid = createSchedSema(rdram.data(), &runtime, 0, kU4FiberCount);
            t.IsTrue(workSid > 0, "U4: work sema created");
            if (workSid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }
            { int32_t v = workSid; std::memcpy(rdram.data() + kU4SlotWorkSid, &v, 4); }
            std::memset(rdram.data() + kU4BodyBase, 0, kU4FiberCount * 4);
            std::memset(rdram.data() + kU4HookBase, 0, kU4FiberCount * 4);

            // Local worker-starter that forwards a per-fiber arg (slot index) in $a1.
            // Mirrors startSchedWorker but sets the StartThread arg.
            auto startU4Worker = [&](uint32_t entry, int prio, uint32_t stack, uint32_t arg) -> int32_t
            {
                constexpr uint32_t kThreadParamAddr = 0x2E00u;
                const uint32_t threadParam[7] = { 0u, entry, stack, 0x2000u, 0u,
                                                  static_cast<uint32_t>(prio), 0u };
                std::memcpy(rdram.data() + kThreadParamAddr, threadParam, sizeof(threadParam));
                R5900Context createCtx{};
                setRegU32(createCtx, 4, kThreadParamAddr);
                ps2_syscalls::CreateThread(rdram.data(), &createCtx, &runtime);
                const int32_t tid = getRegS32(createCtx, 2);
                if (tid <= 0) return -1;
                R5900Context startCtx{};
                setRegU32(startCtx, 4, static_cast<uint32_t>(tid));
                setRegU32(startCtx, 5, arg);
                ps2_syscalls::StartThread(rdram.data(), &startCtx, &runtime);
                return (getRegS32(startCtx, 2) == 0) ? tid : -1;
            };

            const uint32_t stacks[kU4FiberCount] = { 0x004E8000u, 0x004EC000u, 0x004F0000u, 0x004F4000u };
            int32_t tids[kU4FiberCount] = {};
            for (int i = 0; i < kU4FiberCount; ++i)
            {
                tids[i] = startU4Worker(0x00780000u, 10, stacks[i], static_cast<uint32_t>(i));
                t.IsTrue(tids[i] > 0, "U4: fiber started");
            }

            // Wait until all fibers reached their bodies AND are blocked on the sema.
            const bool allBlocked = waitUntil([&]()
            {
                return getSemaWaiters(rdram.data(), &runtime, workSid) >= kU4FiberCount;
            }, std::chrono::milliseconds(2000));
            t.IsTrue(allBlocked, "U4: all fibers blocked on workSid after registering handlers");

            // All bodies must have run before we check the exit-hook sentinels.
            for (int i = 0; i < kU4FiberCount; ++i)
            {
                uint32_t body = 0u;
                std::memcpy(&body, rdram.data() + kU4BodyBase + i * 4u, 4);
                t.Equals(body, 1u, "U4: fiber body ran (sentinel set) before shutdown");
            }

            const auto t0 = std::chrono::steady_clock::now();
            ps2sched::scheduler_shutdown();
            const auto elapsed = std::chrono::steady_clock::now() - t0;

            t.IsTrue(elapsed < std::chrono::seconds(5), "U4: scheduler_shutdown returned within 5s");
            t.Equals(g_activeThreads.load(), 0, "U4: all fibers terminated");

            // Every fiber's exit handler must have run to completion. A body sentinel set
            // without its hook sentinel means the Exiting fiber was abandoned mid-hook.
            for (int i = 0; i < kU4FiberCount; ++i)
            {
                uint32_t hook = 0u;
                std::memcpy(&hook, rdram.data() + kU4HookBase + i * 4u, 4);
                t.Equals(hook, 1u, "U4: this fiber's exit handler completed during shutdown");
            }

            // Do NOT call scheduler_shutdown again (matches T15).
            deleteSchedSema(rdram.data(), &runtime, workSid);
            runtime.requestStop();
            notifyRuntimeStop();
        });

    }); // MiniTest::Case("SchedulerLifecycle")
}

// ---------------------------------------------------------------------------
// Borrowed-worker suite: borrowed-worker WaitSema, exit handler calling
// ExitThread, and Mesa-loop reblock during shutdown.
// ---------------------------------------------------------------------------

// ExitThread is not declared elsewhere in this TU; forward-declare it (resolved
// at link time from the runtime, like the other ps2_syscalls:: entry points).
namespace ps2_syscalls
{
    void ExitThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
}

namespace
{
    // ---- W-suite RDRAM slots (0x43C0–0x43DF, all previously unused) ----
    // W1
    static constexpr uint32_t kW1WorkSid     = 0x000043C0u; // int32 : sema the host worker waits on (count 0)
    static constexpr uint32_t kW1Signalled   = 0x000043C4u; // uint32: fiber set 1 right after SignalSema
    // W2
    static constexpr uint32_t kW2Sentinel    = 0x000043C8u; // uint32: exit handler set 1 BEFORE calling ExitThread
    static constexpr uint32_t kW2BodyRan     = 0x000043CCu; // uint32: fiber body set 1 before returning
    // W3
    static constexpr uint32_t kW3WorkSid     = 0x000043D0u; // int32 : sema fiber A loops on (count stays 0)
    static constexpr uint32_t kW3ExitFlag    = 0x000043D4u; // uint32: reserved slot (see plan note)

    // W1 producer fiber: spin on a few yield points, SignalSema(workSid),
    // mark signalled, exit. Mirrors stepSignalAfterDelay (R7).
    static void stepW1SignalAfterYields(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        for (int i = 0; i < 5; ++i)
            runtime->shouldPreemptGuestExecution(); // cooperative yield x5

        int32_t workSid = 0;
        std::memcpy(&workSid, rdram + kW1WorkSid, 4);
        R5900Context sc{};
        setRegU32(sc, 4, static_cast<uint32_t>(workSid));
        ps2_syscalls::SignalSema(rdram, &sc, runtime);

        uint32_t one = 1u;
        std::memcpy(rdram + kW1Signalled, &one, 4);
        ctx->pc = 0u;
    }

    // W2 exit handler: write the sentinel FIRST (proves the handler ran), then call the
    // real ExitThread syscall, which throws ThreadExitException from inside the exit hook.
    // The trampoline wraps g_fiber_exit_hook in try/catch, so this must NOT crash the process.
    static void stepW2ExitHandlerCallsExitThread(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t one = 1u;
        std::memcpy(rdram + kW2Sentinel, &one, 4); // sentinel BEFORE the throw
        R5900Context ec{};
        ps2_syscalls::ExitThread(rdram, &ec, runtime); // throws ThreadExitException
        // unreachable
        ctx->pc = 0u;
    }

    // W2 fiber body: register the exit handler, mark body ran, return normally so the
    // trampoline drives on_fiber_exit -> our handler.
    static void stepW2RegisterHandlerThenReturn(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        static constexpr uint32_t kW2HandlerEntry = 0x007A8100u;
        R5900Context rc{};
        setRegU32(rc, 4, kW2HandlerEntry); // func
        setRegU32(rc, 5, 0u);              // arg
        ps2_syscalls::RegisterExitHandler(rdram, &rc, runtime);

        uint32_t one = 1u;
        std::memcpy(rdram + kW2BodyRan, &one, 4);
        ctx->pc = 0u; // return -> begin exiting -> hook runs
    }

    // W3 fiber A: Mesa loop on a count==0 sema. WaitSema blocks; on a normal wake it
    // re-checks count, finds 0, re-blocks. During shutdown block_current returns
    // immediately (g_stop + terminateRequested), WaitSema unwinds via ThreadExitException.
    static void stepW3MesaLoopOnSema(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t workSid = 0;
        std::memcpy(&workSid, rdram + kW3WorkSid, 4);
        for (;;)
        {
            R5900Context sc{};
            setRegU32(sc, 4, static_cast<uint32_t>(workSid));
            ps2_syscalls::WaitSema(rdram, &sc, runtime); // count==0 -> blocks
            // If WaitSema ever returns with KE_OK we'd consume a permit and loop; the
            // sema is never signalled, so the only way out is the terminate unwind
            // (ThreadExitException) during shutdown, which skips this point entirely.
            const int32_t r = getRegS32(sc, 2);
            if (r != KE_OK) break; // KE_WAIT_DELETE or similar -> stop looping
        }
        ctx->pc = 0u;
    }

    // W3 fiber B: keep the scheduler busy / hold a second active fiber so the sema is
    // never signalled. Simply blocks on the same sema (also count 0), so shutdown must
    // terminate it too.
    static void stepW3HoldResource(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t workSid = 0;
        std::memcpy(&workSid, rdram + kW3WorkSid, 4);
        R5900Context sc{};
        setRegU32(sc, 4, static_cast<uint32_t>(workSid));
        ps2_syscalls::WaitSema(rdram, &sc, runtime); // blocks; shutdown terminates it
        ctx->pc = 0u;
    }

} // anonymous namespace

void register_scheduler_borrowed_worker_tests()
{
    MiniTest::Case("SchedulerBorrowedWorker", [](TestCase &tc)
    {
        // ------------------------------------------------------------------
        // W1: borrowed host worker WaitSema — no deadlock, no g_threads[-1] entry
        // ------------------------------------------------------------------
        tc.Run("W1: borrowed-worker WaitSema succeeds and never creates g_threads[-1]", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x007A0000u, &stepW1SignalAfterYields);

            // count=0, max=1: a borrowed worker MUST block; only the fiber produces a permit.
            const int32_t workSid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(workSid > 0, "W1: work sema created");
            if (workSid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }
            rdramWrite32(rdram, kW1WorkSid,   static_cast<uint32_t>(workSid));
            rdramWrite32(rdram, kW1Signalled, 0u);

            // Producer fiber: enqueued Ready first so it can run as soon as the worker
            // releases the guest token inside its backoff.
            const int32_t fiberTid = startSchedWorker(rdram.data(), &runtime,
                                                      0x007A0000u, 10, 0x004FC000u, 0x2000u);
            t.IsTrue(fiberTid > 0, "W1: producer fiber started");
            if (fiberTid <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, workSid);
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            std::atomic<bool>    workerDone{false};
            std::atomic<int32_t> workerRet{-9999};
            std::atomic<bool>    workerThrew{false};

            // Borrowed host worker (IRQ/alarm style): tid -1, holds the guest token,
            // calls WaitSema on the count=0 sema. It backs off via NonFiberBackoff,
            // the fiber signals, and the Mesa re-check returns KE_OK.
            std::thread borrowedWorker([&]()
            {
                try
                {
                    g_currentThreadId = -1; // non-fiber host worker
                    ps2sched::async_guest_begin();
                    R5900Context wc{};
                    setRegU32(wc, 4, static_cast<uint32_t>(workSid));
                    ps2_syscalls::WaitSema(rdram.data(), &wc, &runtime);
                    workerRet.store(getRegS32(wc, 2), std::memory_order_release);
                    ps2sched::async_guest_end();
                }
                catch (...)
                {
                    workerThrew.store(true, std::memory_order_release);
                    ps2sched::async_guest_end();
                }
                workerDone.store(true, std::memory_order_release);
            });

            const bool finished = waitUntil([&]()
            {
                return workerDone.load(std::memory_order_acquire);
            }, std::chrono::milliseconds(3000));

            if (!finished)
            {
                // Escape hatch so a regression does not hang the whole binary; the
                // assertions below still record the failure.
                R5900Context sc{};
                setRegU32(sc, 4, static_cast<uint32_t>(workSid));
                ps2_syscalls::SignalSema(rdram.data(), &sc, &runtime);
            }
            if (borrowedWorker.joinable()) borrowedWorker.join();

            t.IsTrue(finished, "W1: borrowed-worker WaitSema completed (no deadlock)");
            t.IsFalse(workerThrew.load(std::memory_order_acquire), "W1: borrowed worker did not throw");
            t.Equals(workerRet.load(std::memory_order_acquire), KE_OK, "W1: WaitSema returned KE_OK");

            // The permit must have come from the fiber, not the escape hatch.
            uint32_t signalled = 0u;
            std::memcpy(&signalled, rdram.data() + kW1Signalled, 4);
            t.Equals(signalled, 1u, "W1: the producer fiber produced the permit");

            // No ThreadInfo must be created for the borrowed worker (tid -1).
            {
                std::lock_guard<std::mutex> lk(g_thread_map_mutex);
                t.Equals(static_cast<int>(g_threads.count(-1)), 0,
                         "W1: borrowed worker created no g_threads[-1] entry");
            }

            const bool drained = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) <= 0;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(drained, "W1: producer fiber drained g_activeThreads");

            deleteSchedSema(rdram.data(), &runtime, workSid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // W2: exit handler calls ExitThread — hook completes, no crash
        // ------------------------------------------------------------------
        tc.Run("W2: exit handler that calls ExitThread terminates the fiber cleanly", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x007A8000u, &stepW2RegisterHandlerThenReturn);
            runtime.registerFunction(0x007A8100u, &stepW2ExitHandlerCallsExitThread);

            rdramWrite32(rdram, kW2Sentinel, 0u);
            rdramWrite32(rdram, kW2BodyRan,  0u);

            const int32_t tid = startSchedWorker(rdram.data(), &runtime,
                                                 0x007A8000u, 10, 0x00500000u, 0x2000u);
            t.IsTrue(tid > 0, "W2: worker fiber started");
            if (tid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // The fiber runs its body (registers the handler, returns), then the trampoline
            // drives on_fiber_exit -> our handler -> ExitThread (throws). The exit hook
            // is wrapped in try/catch, so a throwing handler must not crash the process.
            const bool drained = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) == 0;
            }, std::chrono::milliseconds(3000));
            t.IsTrue(drained, "W2: fiber cleaned up despite ExitThread thrown from its exit handler");

            uint32_t bodyRan = 0u, sentinel = 0u;
            std::memcpy(&bodyRan,  rdram.data() + kW2BodyRan,  4);
            std::memcpy(&sentinel, rdram.data() + kW2Sentinel, 4);
            t.Equals(bodyRan,  1u, "W2: fiber body ran and registered the exit handler");
            t.Equals(sentinel, 1u, "W2: exit handler ran up to the ExitThread call (sentinel written)");

            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // W3: fiber that reblocks in a Mesa loop after shutdown wake -> no hang
        // ------------------------------------------------------------------
        tc.Run("W3: shutdown terminates a fiber that would reblock after being woken", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x007B0000u, &stepW3MesaLoopOnSema); // fiber A
            runtime.registerFunction(0x007B0100u, &stepW3HoldResource);   // fiber B

            // count=0, max=2: never signalled, so both fibers stay blocked until shutdown.
            const int32_t workSid = createSchedSema(rdram.data(), &runtime, 0, 2);
            t.IsTrue(workSid > 0, "W3: work sema created");
            if (workSid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }
            rdramWrite32(rdram, kW3WorkSid, static_cast<uint32_t>(workSid));

            const int32_t tidA = startSchedWorker(rdram.data(), &runtime,
                                                  0x007B0000u, 10, 0x00504000u, 0x2000u);
            const int32_t tidB = startSchedWorker(rdram.data(), &runtime,
                                                  0x007B0100u, 10, 0x00508000u, 0x2000u);
            t.IsTrue(tidA > 0 && tidB > 0, "W3: both fibers started");
            if (tidA <= 0 || tidB <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, workSid);
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // Wait until both fibers are parked on the sema (Mesa loop is armed).
            const bool blocked = waitUntil([&]()
            {
                return getSemaWaiters(rdram.data(), &runtime, workSid) >= 2;
            }, std::chrono::milliseconds(2000));
            t.IsTrue(blocked, "W3: both fibers blocked on the count-0 sema");

            // Shutdown. Fiber A's WaitSema loop, if woken, would re-check count (still 0)
            // and re-block. block_current() returns immediately under g_stop &&
            // terminateRequested, WaitSema observes terminated and throws, so the
            // Mesa loop exits and shutdown does NOT hang.
            const auto t0 = std::chrono::steady_clock::now();
            ps2sched::scheduler_shutdown();
            const auto elapsed = std::chrono::steady_clock::now() - t0;

            t.IsTrue(elapsed < std::chrono::seconds(2),
                     "W3: scheduler_shutdown returned within 2s");
            t.Equals(g_activeThreads.load(), 0, "W3: all fibers terminated");

            // Do NOT call scheduler_shutdown again (matches T15/U4).
            deleteSchedSema(rdram.data(), &runtime, workSid);
            runtime.requestStop();
            notifyRuntimeStop();
        });

    }); // MiniTest::Case("SchedulerBorrowedWorker")
}

// ---------------------------------------------------------------------------
// Scheduler window tests — publish/arm window and notify_all coverage (X1, X2)
// ---------------------------------------------------------------------------

namespace
{
    // -----------------------------------------------------------------------
    // RDRAM slot constants for the SchedulerWindow suite (0x43E0–0x43F8 range)
    // -----------------------------------------------------------------------
    static constexpr uint32_t kX1WorkSid    = 0x000043E0u; // sid the waiter blocks on
    static constexpr uint32_t kX1WakeCount  = 0x000043E4u; // # times the waiter fiber woke and re-checked
    static constexpr uint32_t kX1StopFlag   = 0x000043E8u; // host sets 1 to tell the waiter loop to exit
    static constexpr uint32_t kX1Exited     = 0x000043ECu; // waiter writes 1 just before ctx->pc = 0
    static constexpr uint32_t kX2WorkSid    = 0x000043F0u; // X2 sid the executor fiber blocks on
    static constexpr uint32_t kX2WakeCount  = 0x000043F4u; // X2 # times the fiber woke
    static constexpr uint32_t kX2Exited     = 0x000043F8u; // X2 waiter exit sentinel

    // -----------------------------------------------------------------------
    // X1 waiter: Mesa loop. WaitSema on kX1WorkSid; on each successful return
    // bump kX1WakeCount, then loop again until the host sets kX1StopFlag.
    // Writes kX1Exited=1 on the way out. Runs ON the executor thread (real fiber).
    // -----------------------------------------------------------------------
    static void stepX1WindowWaiter(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t sid = 0;
        std::memcpy(&sid, rdram + kX1WorkSid, sizeof(sid));

        for (;;)
        {
            uint32_t stop = 0u;
            std::memcpy(&stop, rdram + kX1StopFlag, sizeof(stop));
            if (stop != 0u)
            {
                break;
            }

            R5900Context wc{};
            setRegU32(wc, 4, static_cast<uint32_t>(sid));
            ps2_syscalls::WaitSema(rdram, &wc, runtime);
            const int32_t ret = getRegS32(wc, 2);

            // KE_WAIT_DELETE means the sema was deleted under us (shutdown path) — stop.
            if (ret == KE_WAIT_DELETE)
            {
                break;
            }
            if (ret == KE_OK)
            {
                uint32_t n = 0u;
                std::memcpy(&n, rdram + kX1WakeCount, sizeof(n));
                ++n;
                std::memcpy(rdram + kX1WakeCount, &n, sizeof(n));
            }
            // Loop: WaitSema again. count is back to 0 (we consumed the permit), so
            // we re-publish to the waitList and re-arm — re-entering the publish/arm window.
        }

        uint32_t done = 1u;
        std::memcpy(rdram + kX1Exited, &done, sizeof(done));
        ctx->pc = 0u;
    }

    // -----------------------------------------------------------------------
    // X2 fiber: Mesa loop on kX2WorkSid, bumping kX2WakeCount each successful
    // wake, up to 64 times. Writes kX2Exited=1 on the way out.
    // -----------------------------------------------------------------------
    static void stepX2ExecutorWaiter(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t sid = 0;
        std::memcpy(&sid, rdram + kX2WorkSid, sizeof(sid));

        for (uint32_t i = 0; i < 64u; ++i) // bounded: wake up to 64 times
        {
            R5900Context wc{};
            setRegU32(wc, 4, static_cast<uint32_t>(sid));
            ps2_syscalls::WaitSema(rdram, &wc, runtime);
            const int32_t ret = getRegS32(wc, 2);
            if (ret == KE_WAIT_DELETE)
            {
                break;
            }
            if (ret == KE_OK)
            {
                uint32_t n = 0u;
                std::memcpy(&n, rdram + kX2WakeCount, sizeof(n));
                ++n;
                std::memcpy(rdram + kX2WakeCount, &n, sizeof(n));
            }
        }

        uint32_t done = 1u;
        std::memcpy(rdram + kX2Exited, &done, sizeof(done));
        ctx->pc = 0u;
    }

} // anonymous namespace

// ---------------------------------------------------------------------------
// SchedulerWindow suite registration
// ---------------------------------------------------------------------------
void register_scheduler_window_tests()
{
    MiniTest::Case("SchedulerWindow", [](TestCase &tc)
    {
        // ------------------------------------------------------------------
        // X1: SignalSema in the publish/arm window is never lost
        // ------------------------------------------------------------------
        tc.Run("X1: SignalSema in the publish/arm window is never lost", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x007C0000u, &stepX1WindowWaiter);

            // count=0, maxCount=1: the waiter must block; each SignalSema does a real
            // 0->1 transition with a waiter pop and wakeup while the fiber is mid-park.
            const int32_t sid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(sid > 0, "X1: work sema created");
            if (sid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            rdramWrite32(rdram, kX1WorkSid,   static_cast<uint32_t>(sid));
            rdramWrite32(rdram, kX1WakeCount, 0u);
            rdramWrite32(rdram, kX1StopFlag,  0u);
            rdramWrite32(rdram, kX1Exited,    0u);

            const int32_t tid = startSchedWorker(rdram.data(), &runtime,
                                                 0x007C0000u, 10, 0x00508000u, 0x2000u);
            t.IsTrue(tid > 0, "X1: waiter fiber started");
            if (tid <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, sid);
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // NOTE: deliberately do NOT wait for getSemaWaiters(sid) >= 1. We start
            // hammering immediately so signals land inside the publish/arm window.
            std::atomic<bool> signalerThrew{false};
            std::atomic<uint32_t> signalsSent{0u};

            std::thread signaler([&]()
            {
                try
                {
                    const auto deadline =
                        std::chrono::steady_clock::now() + std::chrono::milliseconds(150);
                    uint32_t iters = 0u;
                    while (std::chrono::steady_clock::now() < deadline && iters < 200000u)
                    {
                        R5900Context sc{};
                        setRegU32(sc, 4, static_cast<uint32_t>(sid));
                        ps2_syscalls::SignalSema(rdram.data(), &sc, &runtime);
                        const int32_t sret = getRegS32(sc, 2);
                        signalsSent.fetch_add(1, std::memory_order_relaxed);

                        // If the permit is still outstanding (waiter has not consumed it),
                        // drain it so the next SignalSema is a fresh 0->1 transition.
                        if (sret == KE_OK)
                        {
                            R5900Context pc{};
                            setRegU32(pc, 4, static_cast<uint32_t>(sid));
                            ps2_syscalls::PollSema(rdram.data(), &pc, &runtime);
                        }
                        ++iters;
                    }
                }
                catch (...)
                {
                    signalerThrew.store(true, std::memory_order_release);
                }
            });

            // PRIMARY ASSERTION: the waiter must wake at least once. Under the broken
            // impl the first in-window signal is lost and the waiter parks forever, so
            // kX1WakeCount stays 0 and this times out (caught here, not a hang).
            const bool woke = waitUntil([&]()
            {
                uint32_t n = 0u;
                std::memcpy(&n, rdram.data() + kX1WakeCount, sizeof(n));
                return n >= 1u;
            }, std::chrono::milliseconds(3000));
            t.IsTrue(woke, "X1: waiter fiber woke at least once (no lost in-window wakeup)");

            if (signaler.joinable())
            {
                signaler.join();
            }
            t.IsFalse(signalerThrew.load(std::memory_order_acquire), "X1: signaler did not throw");
            t.IsTrue(signalsSent.load(std::memory_order_relaxed) > 0u, "X1: signaler ran");

            // Tell the waiter to exit its Mesa loop, then push one final permit so a
            // waiter parked between iterations wakes, sees the stop flag, and returns.
            rdramWrite32(rdram, kX1StopFlag, 1u);
            {
                R5900Context sc{};
                setRegU32(sc, 4, static_cast<uint32_t>(sid));
                ps2_syscalls::SignalSema(rdram.data(), &sc, &runtime);
            }

            const bool exited = waitUntil([&]()
            {
                uint32_t e = 0u;
                std::memcpy(&e, rdram.data() + kX1Exited, sizeof(e));
                return e == 1u;
            }, std::chrono::milliseconds(3000));
            t.IsTrue(exited, "X1: waiter fiber observed stop flag and exited");

            const bool drained = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) <= 0;
            }, std::chrono::milliseconds(3000));
            t.IsTrue(drained, "X1: waiter fiber drained g_activeThreads");

            // Reported for visibility; not asserted as a hard number (cooperative
            // scheduling means we cannot wake on every one of thousands of signals).
            {
                uint32_t n = 0u;
                std::memcpy(&n, rdram.data() + kX1WakeCount, sizeof(n));
                std::cout << "    [X1] wakes=" << n
                          << " signals=" << signalsSent.load(std::memory_order_relaxed)
                          << std::endl;
            }

            deleteSchedSema(rdram.data(), &runtime, sid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // X2: notify_all wakes the executor under host-worker CV contention
        // ------------------------------------------------------------------
        tc.Run("X2: notify_all wakes the executor under host-worker CV contention", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x007C8000u, &stepX2ExecutorWaiter);

            const int32_t sid = createSchedSema(rdram.data(), &runtime, 0, 1);
            t.IsTrue(sid > 0, "X2: work sema created");
            if (sid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }
            rdramWrite32(rdram, kX2WorkSid,   static_cast<uint32_t>(sid));
            rdramWrite32(rdram, kX2WakeCount, 0u);
            rdramWrite32(rdram, kX2Exited,    0u);

            const int32_t tid = startSchedWorker(rdram.data(), &runtime,
                                                 0x007C8000u, 10, 0x0050C000u, 0x2000u);
            t.IsTrue(tid > 0, "X2: executor fiber started");
            if (tid <= 0)
            {
                deleteSchedSema(rdram.data(), &runtime, sid);
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // Wait until the fiber is parked so the first signal targets a real waiter.
            waitUntil([&]() { return getSemaWaiters(rdram.data(), &runtime, sid) >= 1; },
                      std::chrono::milliseconds(1000));

            std::atomic<bool> stopWorkers{false};
            std::atomic<bool> workerThrew{false};

            // Host workers that contend on g_sched_cv via async_guest_begin/end.
            constexpr int kNumWorkers = 3;
            std::vector<std::thread> workers;
            workers.reserve(kNumWorkers);
            for (int w = 0; w < kNumWorkers; ++w)
            {
                workers.emplace_back([&]()
                {
                    g_currentThreadId = -1; // borrowed host worker
                    try
                    {
                        while (!stopWorkers.load(std::memory_order_acquire))
                        {
                            ps2sched::async_guest_begin();
                            ps2sched::async_guest_end();
                        }
                    }
                    catch (...)
                    {
                        workerThrew.store(true, std::memory_order_release);
                    }
                });
            }

            // Signaler: drive make_ready -> notify on g_sched_cv while workers contend.
            std::atomic<bool> signalerThrew{false};
            std::thread signaler([&]()
            {
                try
                {
                    for (uint32_t i = 0; i < 64u; ++i)
                    {
                        // Each successful wake must make it across the CV to the executor.
                        // Wait (bounded) until the fiber's wake count catches up so we
                        // exercise one notify at a time against live CV contention.
                        R5900Context sc{};
                        setRegU32(sc, 4, static_cast<uint32_t>(sid));
                        ps2_syscalls::SignalSema(rdram.data(), &sc, &runtime);

                        const uint32_t target = i + 1u;
                        const bool advanced = waitUntil([&]()
                        {
                            uint32_t n = 0u;
                            std::memcpy(&n, rdram.data() + kX2WakeCount, sizeof(n));
                            return n >= target;
                        }, std::chrono::milliseconds(200));
                        if (!advanced)
                        {
                            break; // executor was stranded — leave count short; asserted below
                        }
                    }
                }
                catch (...)
                {
                    signalerThrew.store(true, std::memory_order_release);
                }
            });

            if (signaler.joinable())
            {
                signaler.join();
            }
            stopWorkers.store(true, std::memory_order_release);
            for (auto &w : workers)
            {
                if (w.joinable())
                {
                    w.join();
                }
            }

            t.IsFalse(workerThrew.load(std::memory_order_acquire), "X2: host workers did not throw");
            t.IsFalse(signalerThrew.load(std::memory_order_acquire), "X2: signaler did not throw");

            // PRIMARY: every signal reached the executor and woke the fiber. Under a
            // notify_one regression a notification can be consumed by a parked host
            // worker instead of the executor, stranding the fiber; the wake count then
            // stalls below 64 and this fails. (Probabilistic: depends on the OS handing
            // a notify_one to a worker — see plan caveat.)
            {
                uint32_t n = 0u;
                std::memcpy(&n, rdram.data() + kX2WakeCount, sizeof(n));
                t.Equals(n, 64u, "X2: all 64 signals reached the executor fiber (notify_all)");
            }

            const bool exited = waitUntil([&]()
            {
                uint32_t e = 0u;
                std::memcpy(&e, rdram.data() + kX2Exited, sizeof(e));
                return e == 1u;
            }, std::chrono::milliseconds(3000));
            t.IsTrue(exited, "X2: executor fiber completed its 64-wake loop and exited");

            const bool drained = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) <= 0;
            }, std::chrono::milliseconds(3000));
            t.IsTrue(drained, "X2: executor fiber drained g_activeThreads");

            deleteSchedSema(rdram.data(), &runtime, sid);
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

    }); // MiniTest::Case("SchedulerWindow")
}

// ---------------------------------------------------------------------------
// Y-suite: four new scheduler correctness tests (Y1–Y4).
// RDRAM scratch block: 0x4400–0x44FF (does not overlap existing 0x4000–0x43FF).
// ---------------------------------------------------------------------------

namespace
{
    // -----------------------------------------------------------------------
    // RDRAM slot constants for the Y-suite
    // -----------------------------------------------------------------------

    // Y1: SleepThread Mesa loop
    static constexpr uint32_t kY1ExitedSleep = 0x00004400u; // uint32: 1 after SleepThread returns
    static constexpr uint32_t kY1SleepRet    = 0x00004404u; // int32 : SleepThread return value

    // Y4a: DeleteSema drain
    static constexpr uint32_t kY4WorkSid  = 0x00004408u; // int32 : sema id for Y4a waiters
    static constexpr uint32_t kY4RetBase  = 0x0000440Cu; // int32[4]: WaitSema return values per fiber

    // Y4b: stale-token rejection
    static constexpr uint32_t kY4bTokenLo = 0x00004420u; // uint32: low 32 bits of fiber token
    static constexpr uint32_t kY4bTokenHi = 0x00004424u; // uint32: high 32 bits of fiber token

    // -----------------------------------------------------------------------
    // Y1 step function: SleepThread, then record exit sentinel and return value
    // -----------------------------------------------------------------------
    static void stepY1SleepThenRecord(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        R5900Context sc{};
        ps2_syscalls::SleepThread(rdram, &sc, runtime); // blocks until a genuine WakeupThread
        // Reached only when SleepThread truly returns (consumes a wakeupCount permit).
        uint32_t one = 1u;
        std::memcpy(rdram + kY1ExitedSleep, &one, 4);
        int32_t ret = getRegS32(sc, 2);
        std::memcpy(rdram + kY1SleepRet, &ret, 4);
        ctx->pc = 0u;
    }

    // -----------------------------------------------------------------------
    // Y4a step function: WaitSema(sid from rdram), record return in an atomic slot
    // -----------------------------------------------------------------------
    static std::atomic<int> gY4SlotCounter{0};

    static void stepY4WaitSemaRecordRet(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        int32_t sid = 0;
        std::memcpy(&sid, rdram + kY4WorkSid, 4);

        R5900Context sc{};
        setRegU32(sc, 4, static_cast<uint32_t>(sid));
        ps2_syscalls::WaitSema(rdram, &sc, runtime); // blocks until sema is deleted
        int32_t ret = getRegS32(sc, 2);

        // Claim a result slot atomically.
        const int slot = gY4SlotCounter.fetch_add(1, std::memory_order_relaxed);
        if (slot >= 0 && slot < 4)
        {
            std::memcpy(rdram + kY4RetBase + static_cast<uint32_t>(slot) * 4u, &ret, 4);
        }

        ctx->pc = 0u;
    }

    // -----------------------------------------------------------------------
    // Y4b step function: publish current_fiber_token() into rdram, then exit
    // -----------------------------------------------------------------------
    static void stepY4bPublishTokenThenExit(uint8_t *rdram, R5900Context *ctx, PS2Runtime * /*runtime*/)
    {
        const uint64_t tok = ps2sched::current_fiber_token();
        const uint32_t lo = static_cast<uint32_t>(tok & 0xFFFFFFFFu);
        const uint32_t hi = static_cast<uint32_t>(tok >> 32u);
        std::memcpy(rdram + kY4bTokenLo, &lo, 4);
        std::memcpy(rdram + kY4bTokenHi, &hi, 4);
        ctx->pc = 0u;
    }

    // -----------------------------------------------------------------------
    // Z1: tid-reuse regression
    //
    // RDRAM slot: kZ1Done (0x00004450). Each fiber increments this on exit.
    // -----------------------------------------------------------------------
    static constexpr uint32_t kZ1Done = 0x00004450u; // uint32: incremented by each completing fiber

    static void stepZ1QuickExit(uint8_t *rdram, R5900Context *ctx, PS2Runtime * /*runtime*/)
    {
        uint32_t n = 0u;
        std::memcpy(&n, rdram + kZ1Done, 4);
        ++n;
        std::memcpy(rdram + kZ1Done, &n, 4);
        ctx->pc = 0u;
    }

} // anonymous namespace

// ---------------------------------------------------------------------------
// Y1 — SleepThread Mesa loop: ResumeThread must NOT count as a wakeup
// ---------------------------------------------------------------------------
void register_scheduler_sleep_resume_tests()
{
    MiniTest::Case("SchedulerSleepResume", [](TestCase &tc)
    {
        tc.Run("sleeping-and-suspended fiber re-parks after ResumeThread and only exits on WakeupThread", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x007D0000u, &stepY1SleepThenRecord);

            // Initialize sentinels so any premature write is detectable.
            {
                uint32_t z = 0u;
                std::memcpy(rdram.data() + kY1ExitedSleep, &z, 4);
                int32_t bad = -9999;
                std::memcpy(rdram.data() + kY1SleepRet, &bad, 4);
            }

            const int32_t tid = startSchedWorker(rdram.data(), &runtime,
                                                 0x007D0000u, 10, 0x00510000u, 0x2000u);
            t.IsTrue(tid > 0, "Y1: sleeping fiber started");
            if (tid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // Step 1: confirm the fiber is inside SleepThread (THS_WAIT / TSW_SLEEP).
            const bool sleeping = waitUntil([&]()
            {
                int32_t status = 0, wt = 0;
                getThreadStatus(rdram, runtime, tid, status, wt);
                return status == kTHSWait && wt == kTSWSleep;
            }, std::chrono::milliseconds(500));
            t.IsTrue(sleeping, "Y1: fiber entered SleepThread");
            if (!sleeping)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // Step 2: suspend the sleeping fiber; it must transition to THS_WAITSUSPEND.
            {
                R5900Context sc{};
                setRegU32(sc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::SuspendThread(rdram.data(), &sc, &runtime);
                t.Equals(getRegS32(sc, 2), KE_OK, "Y1: SuspendThread returned KE_OK");
            }

            const bool suspended = waitUntil([&]()
            {
                int32_t status = 0, wt = 0;
                getThreadStatus(rdram, runtime, tid, status, wt);
                return status == kTHSWaitSuspend;
            }, std::chrono::milliseconds(500));
            t.IsTrue(suspended, "Y1: fiber is sleeping AND suspended (THS_WAITSUSPEND)");

            // Step 3: ResumeThread clears the suspend gate but provides NO wakeupCount permit.
            // The Mesa loop inside SleepThread must re-park.
            {
                R5900Context sc{};
                setRegU32(sc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::ResumeThread(rdram.data(), &sc, &runtime);
                t.Equals(getRegS32(sc, 2), KE_OK, "Y1: ResumeThread returned KE_OK");
            }

            // Step 4: wait generously, then assert the fiber did NOT exit SleepThread.
            std::this_thread::sleep_for(std::chrono::milliseconds(60));

            {
                uint32_t exited = 0u;
                std::memcpy(&exited, rdram.data() + kY1ExitedSleep, 4);
                t.Equals(exited, 0u,
                         "Y1: fiber did not exit SleepThread after ResumeThread (no spurious wake)");
            }

            // Also confirm the fiber re-asserted THS_WAIT/TSW_SLEEP (re-parked in sleep).
            {
                int32_t status = 0, wt = 0;
                getThreadStatus(rdram, runtime, tid, status, wt);
                t.IsTrue(status == kTHSWait && wt == kTSWSleep,
                         "Y1: fiber re-parked in THS_SLEEP after ResumeThread");
            }

            // Step 5: a genuine WakeupThread must now release the fiber from SleepThread.
            {
                R5900Context sc{};
                setRegU32(sc, 4, static_cast<uint32_t>(tid));
                ps2_syscalls::WakeupThread(rdram.data(), &sc, &runtime);
                t.Equals(getRegS32(sc, 2), KE_OK, "Y1: WakeupThread returned KE_OK");
            }

            const bool woke = waitUntil([&]()
            {
                uint32_t exited = 0u;
                std::memcpy(&exited, rdram.data() + kY1ExitedSleep, 4);
                return exited == 1u;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(woke, "Y1: fiber exited SleepThread after WakeupThread");

            {
                int32_t sleepRet = -9999;
                std::memcpy(&sleepRet, rdram.data() + kY1SleepRet, 4);
                t.Equals(sleepRet, KE_OK, "Y1: SleepThread returned KE_OK on genuine wakeup");
            }

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

    }); // MiniTest::Case("SchedulerSleepResume")
}

// ---------------------------------------------------------------------------
// Y2 — scheduler_shutdown promptly unwinds a running, back-edge-bearing fiber
// ---------------------------------------------------------------------------
void register_scheduler_shutdown_clean_tests()
{
    MiniTest::Case("SchedulerShutdownClean", [](TestCase &tc)
    {
        tc.Run("scheduler_shutdown terminates a non-blocking running fiber promptly", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x007D8000u, &stepProgressLoop);

            // Initialize: kStopFlag is intentionally left 0 so the only exit is the shutdown unwind.
            rdramWrite32(rdram, kStartedFlag, 0u);
            rdramWrite32(rdram, kProgressCtr, 0u);
            rdramWrite32(rdram, kStopFlag,    0u);

            const int32_t tid = startSchedWorker(rdram.data(), &runtime,
                                                 0x007D8000u, 10, 0x00514000u, 0x2000u);
            t.IsTrue(tid > 0, "Y2: progress fiber started");
            if (tid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // Step 1: confirm the fiber is actively looping (Running, not blocked).
            const bool started = waitUntil([&]()
            {
                return Ps2FastRead32(rdram.data(), kStartedFlag) != 0u;
            }, std::chrono::milliseconds(500));
            t.IsTrue(started, "Y2: fiber entered its loop");

            const uint32_t ctr0 = Ps2FastRead32(rdram.data(), kProgressCtr);
            const bool progressing = waitUntil([&]()
            {
                return Ps2FastRead32(rdram.data(), kProgressCtr) > ctr0;
            }, std::chrono::milliseconds(500));
            t.IsTrue(progressing, "Y2: fiber is actively looping (progress counter advancing)");

            // Step 2: shut down and measure elapsed time.
            // kStopFlag remains 0; the only exit is the back-edge-triggered terminate throw.
            const auto t0 = std::chrono::steady_clock::now();
            ps2sched::scheduler_shutdown();
            const auto elapsed = std::chrono::steady_clock::now() - t0;

            // Step 3: shutdown must be prompt and complete.
            t.IsTrue(elapsed < std::chrono::milliseconds(500),
                     "Y2: scheduler_shutdown returned promptly (< 500 ms)");
            t.Equals(g_activeThreads.load(), 0,
                     "Y2: g_activeThreads is 0 after shutdown (fiber unwound)");

            // Do NOT call scheduler_shutdown again (mirrors T15/W3/U4).
            runtime.requestStop();
            notifyRuntimeStop();
        });

    }); // MiniTest::Case("SchedulerShutdownClean")
}

// ---------------------------------------------------------------------------
// Y3 — Borrowed-worker self-targeting syscalls return KE_ILLEGAL_THID, no g_threads[-1]
// ---------------------------------------------------------------------------
void register_scheduler_borrowed_guard_tests()
{
    MiniTest::Case("SchedulerBorrowedGuard", [](TestCase &tc)
    {
        tc.Run("self-targeting syscalls from a borrowed worker return KE_ILLEGAL_THID and do not corrupt g_threads", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            // Baseline: no -1 entry should exist at the start.
            {
                std::lock_guard<std::mutex> lk(g_thread_map_mutex);
                t.Equals(static_cast<int>(g_threads.count(-1)), 0,
                         "Y3: no g_threads[-1] entry at baseline");
            }

            // Each case: a self-targeting syscall (a0=0 means TH_SELF) called from a
            // borrowed host worker (g_currentThreadId == -1).
            struct SyscallCase
            {
                const char *name;
                void (*fn)(uint8_t *, R5900Context *, PS2Runtime *);
                uint32_t a0; // $a0
                uint32_t a1; // $a1
            };

            const SyscallCase cases[] = {
                { "SuspendThread(0)",          &ps2_syscalls::SuspendThread,          0u, 0u },
                { "ResumeThread(0)",           &ps2_syscalls::ResumeThread,           0u, 0u },
                { "ChangeThreadPriority(0)",   &ps2_syscalls::ChangeThreadPriority,   0u, 5u },
                { "CancelWakeupThread(0)",     &ps2_syscalls::CancelWakeupThread,     0u, 0u },
                { "RotateThreadReadyQueue(0)", &ps2_syscalls::RotateThreadReadyQueue,  0u, 0u },
                { "ReferThreadStatus(0)",      &ps2_syscalls::ReferThreadStatus,      0u, 0u },
                { "TerminateThread(0)",        &ps2_syscalls::TerminateThread,        0u, 0u },
            };

            for (const auto &c : cases)
            {
                std::atomic<int32_t> retVal{-1234};
                std::atomic<bool>    threw{false};

                std::thread worker([&]()
                {
                    try
                    {
                        g_currentThreadId = -1; // borrowed host worker: no PS2 thread identity
                        ps2sched::async_guest_begin();
                        R5900Context ctx{};
                        setRegU32(ctx, 4, c.a0); // $a0 = 0 (TH_SELF / self-target)
                        setRegU32(ctx, 5, c.a1); // $a1 = extra arg (e.g. new priority)
                        c.fn(rdram.data(), &ctx, &runtime);
                        retVal.store(getRegS32(ctx, 2), std::memory_order_release);
                        ps2sched::async_guest_end();
                    }
                    catch (...)
                    {
                        threw.store(true, std::memory_order_release);
                        ps2sched::async_guest_end();
                    }
                });
                worker.join();

                t.IsFalse(threw.load(std::memory_order_acquire),
                          std::string(c.name) + ": borrowed-worker syscall must not throw");
                t.Equals(retVal.load(std::memory_order_acquire), KE_ILLEGAL_THID,
                         std::string(c.name) + ": must return KE_ILLEGAL_THID for self-target from borrowed worker");
                {
                    std::lock_guard<std::mutex> lk(g_thread_map_mutex);
                    t.Equals(static_cast<int>(g_threads.count(-1)), 0,
                             std::string(c.name) + ": no g_threads[-1] entry created");
                }
            }

            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

    }); // MiniTest::Case("SchedulerBorrowedGuard")
}

// ---------------------------------------------------------------------------
// Y4 — Sema/event delete drains all waiters via generation-validated wakeups
// ---------------------------------------------------------------------------
void register_scheduler_sema_delete_tests()
{
    MiniTest::Case("SchedulerSemaDelete", [](TestCase &tc)
    {
        // ------------------------------------------------------------------
        // Y4a: N waiters on one sema all receive KE_WAIT_DELETE and exit cleanly
        // ------------------------------------------------------------------
        tc.Run("Y4a: DeleteSema wakes all N waiters with KE_WAIT_DELETE via pair-based drain", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x007E0000u, &stepY4WaitSemaRecordRet);

            constexpr int kN = 4;

            // Create the sema with init=0 so all N fibers block.
            const int32_t workSid = createSchedSema(rdram.data(), &runtime, 0, kN + 1);
            t.IsTrue(workSid > 0, "Y4a: work sema created");
            if (workSid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // Publish the sema id and reset result slots before starting fibers.
            {
                int32_t v = workSid;
                std::memcpy(rdram.data() + kY4WorkSid, &v, 4);
            }
            for (int i = 0; i < kN; ++i)
            {
                int32_t bad = -9999;
                std::memcpy(rdram.data() + kY4RetBase + static_cast<uint32_t>(i) * 4u, &bad, 4);
            }
            gY4SlotCounter.store(0, std::memory_order_relaxed);

            // Start N fibers, each of which calls WaitSema(workSid).
            constexpr uint32_t kStackBase = 0x00518000u;
            constexpr uint32_t kStackSize = 0x2000u;
            int32_t tids[kN] = {};
            for (int i = 0; i < kN; ++i)
            {
                const uint32_t stackAddr = kStackBase + static_cast<uint32_t>(i) * kStackSize;
                tids[i] = startSchedWorker(rdram.data(), &runtime,
                                           0x007E0000u, 10, stackAddr, kStackSize);
                t.IsTrue(tids[i] > 0, "Y4a: fiber started");
            }

            // Wait until all N fibers are blocked on the sema.
            const bool allBlocked = waitUntil([&]()
            {
                return getSemaWaiters(rdram.data(), &runtime, workSid) >= kN;
            }, std::chrono::milliseconds(2000));
            t.IsTrue(allBlocked, "Y4a: all N fibers blocked on the sema");
            if (!allBlocked)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // DeleteSema: must drain the pair-based waitList and wake all N with KE_WAIT_DELETE.
            {
                R5900Context dc{};
                setRegU32(dc, 4, static_cast<uint32_t>(workSid));
                ps2_syscalls::DeleteSema(rdram.data(), &dc, &runtime);
                t.Equals(getRegS32(dc, 2), KE_OK, "Y4a: DeleteSema returned KE_OK");
            }

            // All N fibers must finish.
            const bool drained = waitUntil([&]()
            {
                return g_activeThreads.load(std::memory_order_acquire) == 0;
            }, std::chrono::milliseconds(2000));
            t.IsTrue(drained, "Y4a: all N fibers exited after DeleteSema");

            // Every recorded return value must be KE_WAIT_DELETE.
            for (int i = 0; i < kN; ++i)
            {
                int32_t ret = -9999;
                std::memcpy(&ret, rdram.data() + kY4RetBase + static_cast<uint32_t>(i) * 4u, 4);
                t.Equals(ret, KE_WAIT_DELETE,
                         std::string("Y4a: slot ") + std::to_string(i) + " WaitSema returned KE_WAIT_DELETE");
            }

            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

        // ------------------------------------------------------------------
        // Y4b: stale generation token is rejected by enqueue_external_wakeup_validated
        // ------------------------------------------------------------------
        tc.Run("Y4b: enqueue_external_wakeup_validated with a wrong token does not wake the fiber", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x007E8000u, &stepY4bPublishTokenThenExit);
            runtime.registerFunction(0x007F0000u, &stepY1SleepThenRecord);

            // Phase 1: start a short-lived fiber that publishes its token and exits.
            {
                uint32_t z = 0u;
                std::memcpy(rdram.data() + kY4bTokenLo, &z, 4);
                std::memcpy(rdram.data() + kY4bTokenHi, &z, 4);
            }

            const int32_t tokenFiberTid = startSchedWorker(rdram.data(), &runtime,
                                                           0x007E8000u, 10, 0x00528000u, 0x2000u);
            t.IsTrue(tokenFiberTid > 0, "Y4b: token-publishing fiber started");
            if (tokenFiberTid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // Wait for the token fiber to write its token and exit.
            const bool tokenWritten = waitUntil([&]()
            {
                uint32_t lo = 0u, hi = 0u;
                std::memcpy(&lo, rdram.data() + kY4bTokenLo, 4);
                std::memcpy(&hi, rdram.data() + kY4bTokenHi, 4);
                return lo != 0u || hi != 0u;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(tokenWritten, "Y4b: token fiber published its token");

            waitUntil([&](){ return g_activeThreads.load() == 0; },
                      std::chrono::milliseconds(1000));

            uint32_t tokenLo = 0u, tokenHi = 0u;
            std::memcpy(&tokenLo, rdram.data() + kY4bTokenLo, 4);
            std::memcpy(&tokenHi, rdram.data() + kY4bTokenHi, 4);
            const uint64_t capturedToken =
                (static_cast<uint64_t>(tokenHi) << 32u) | static_cast<uint64_t>(tokenLo);

            // Phase 2: start a new sleeping fiber (reusing stepY1SleepThenRecord).
            // Get its CURRENT token, then fabricate a WRONG token by flipping a generation bit.
            {
                uint32_t z = 0u;
                std::memcpy(rdram.data() + kY1ExitedSleep, &z, 4);
                int32_t bad = -9999;
                std::memcpy(rdram.data() + kY1SleepRet, &bad, 4);
            }

            const int32_t sleeperTid = startSchedWorker(rdram.data(), &runtime,
                                                        0x007F0000u, 10, 0x0052C000u, 0x2000u);
            t.IsTrue(sleeperTid > 0, "Y4b: sleeper fiber started");
            if (sleeperTid <= 0)
            {
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // Wait until the sleeper is truly parked in SleepThread.
            const bool sleeperSleeping = waitUntil([&]()
            {
                int32_t status = 0, wt = 0;
                getThreadStatus(rdram, runtime, sleeperTid, status, wt);
                return status == kTHSWait && wt == kTSWSleep;
            }, std::chrono::milliseconds(500));
            t.IsTrue(sleeperSleeping, "Y4b: sleeper fiber is parked in SleepThread");
            if (!sleeperSleeping)
            {
                // Kick the sleeper out so we can clean up.
                R5900Context sc{};
                setRegU32(sc, 4, static_cast<uint32_t>(sleeperTid));
                ps2_syscalls::WakeupThread(rdram.data(), &sc, &runtime);
                waitUntil([&](){ return g_activeThreads.load() == 0; },
                          std::chrono::milliseconds(1000));
                ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop(); return;
            }

            // Fabricate a wrong token: flip one bit in the high word (generation part).
            // This guarantees a generation mismatch regardless of the actual token layout.
            const uint64_t wrongToken = capturedToken ^ (uint64_t(1u) << 32u);

            // Phase 3: fire the stale (wrong) token wake from a foreign host thread.
            // The scheduler must detect the generation mismatch and drop it.
            std::thread foreignWaker([&]()
            {
                ps2sched::enqueue_external_wakeup_validated(sleeperTid, wrongToken);
            });
            foreignWaker.join();

            // Phase 4: wait and assert the sleeper was NOT woken (stale token rejected).
            std::this_thread::sleep_for(std::chrono::milliseconds(60));
            {
                uint32_t exited = 0u;
                std::memcpy(&exited, rdram.data() + kY1ExitedSleep, 4);
                t.Equals(exited, 0u,
                         "Y4b: sleeper was not woken by the wrong-token wakeup (stale token rejected)");
            }
            {
                int32_t status = 0, wt = 0;
                getThreadStatus(rdram, runtime, sleeperTid, status, wt);
                t.IsTrue(status == kTHSWait && wt == kTSWSleep,
                         "Y4b: sleeper is still in THS_SLEEP after wrong-token wakeup");
            }

            // Sanity: a valid WakeupThread DOES reach the sleeper, proving it was genuinely
            // reachable and the stale wake was specifically rejected (not an inert no-op).
            {
                R5900Context sc{};
                setRegU32(sc, 4, static_cast<uint32_t>(sleeperTid));
                ps2_syscalls::WakeupThread(rdram.data(), &sc, &runtime);
                t.Equals(getRegS32(sc, 2), KE_OK, "Y4b: WakeupThread returned KE_OK");
            }

            const bool woke = waitUntil([&]()
            {
                uint32_t exited = 0u;
                std::memcpy(&exited, rdram.data() + kY1ExitedSleep, 4);
                return exited == 1u;
            }, std::chrono::milliseconds(1000));
            t.IsTrue(woke, "Y4b: sleeper woke normally after a valid WakeupThread");

            waitUntil([&](){ return g_activeThreads.load() == 0; }, std::chrono::milliseconds(1000));
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

    }); // MiniTest::Case("SchedulerSemaDelete")
}

// ---------------------------------------------------------------------------
// Z1 — Borrowed-worker StartThread concurrent with executor Finished teardown
//
// Regression for the use-after-free where g_fiber_map.erase(tid) after
// re-acquiring g_sched_mutex could destroy a freshly-created FiberContext if
// a borrowed worker recycled the tid during the munmap window between the
// unlock and re-lock in the Finished branch.  The fix erases the entry before
// dropping the lock so the tid is safe to reuse while the stack is being freed.
// ---------------------------------------------------------------------------
void register_scheduler_tid_reuse_tests()
{
    MiniTest::Case("SchedulerTidReuse", [](TestCase &tc)
    {
        // Z1: A borrowed host worker (g_currentThreadId == -1) creates and starts
        // fibers in a tight loop while the executor is simultaneously tearing down
        // just-exited fibers.  Each fiber increments kZ1Done and exits; the worker
        // deletes the dormant thread after each cycle so CreateThread can recycle
        // the tid.  If the executor ever erases a freshly-created FiberContext (the
        // old bug), the new fiber never runs and the final count falls short.
        tc.Run("Z1: borrowed worker StartThread during Finished teardown does not destroy new fiber", [](TestCase &t)
        {
            notifyRuntimeStop();
            ps2sched::scheduler_init();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            runtime.registerFunction(0x007B0000u, &stepZ1QuickExit);

            constexpr int kCycles = 30;
            std::atomic<bool> workerDone{false};
            std::atomic<bool> workerThrew{false};

            // All fibers share one stack (0x00510000, 0x2000 bytes); each cycle waits
            // for the previous fiber to exit before starting the next one.
            std::thread borrowedWorker([&]()
            {
                try
                {
                    g_currentThreadId = -1; // simulate IRQ/alarm borrowed host worker

                    for (int i = 0; i < kCycles; ++i)
                    {
                        // Create and start the next fiber while holding the guest token.
                        ps2sched::async_guest_begin();
                        const int32_t tid = startSchedWorker(rdram.data(), &runtime,
                                                             0x007B0000u, 32,
                                                             0x00510000u, 0x2000u);
                        ps2sched::async_guest_end();

                        if (tid <= 0)
                        {
                            break; // startSchedWorker failed — assertion below will catch it
                        }

                        // Wait for the executor to run the fiber to completion (outside
                        // the guest scope so the executor can hold the token).
                        const bool exited = waitUntil(
                            [&](){ return g_activeThreads.load() == 0; },
                            std::chrono::milliseconds(500));
                        if (!exited)
                        {
                            break; // timed out — assertion below will catch it
                        }

                        // Delete the dormant thread so CreateThread can reuse the tid
                        // on the next iteration, exercising the same-tid recycle path.
                        ps2sched::async_guest_begin();
                        R5900Context dc{};
                        setRegU32(dc, 4, static_cast<uint32_t>(tid));
                        ps2_syscalls::DeleteThread(rdram.data(), &dc, &runtime);
                        ps2sched::async_guest_end();
                    }
                }
                catch (...)
                {
                    workerThrew.store(true, std::memory_order_release);
                }
                workerDone.store(true, std::memory_order_release);
            });

            const bool done = waitUntil([&](){ return workerDone.load(); },
                                        std::chrono::milliseconds(20000));
            if (borrowedWorker.joinable())
            {
                borrowedWorker.join();
            }

            t.IsTrue(done,            "Z1: borrowed worker completed all cycles within 20s");
            t.IsFalse(workerThrew.load(), "Z1: borrowed worker did not throw");

            uint32_t completions = 0u;
            std::memcpy(&completions, rdram.data() + kZ1Done, 4);
            t.Equals(static_cast<int>(completions), kCycles,
                     "Z1: all fibers ran to completion (kZ1Done == kCycles)");

            waitUntil([&](){ return g_activeThreads.load() == 0; },
                      std::chrono::milliseconds(1000));
            ps2sched::scheduler_shutdown(); runtime.requestStop(); notifyRuntimeStop();
        });

    }); // MiniTest::Case("SchedulerTidReuse")
}
