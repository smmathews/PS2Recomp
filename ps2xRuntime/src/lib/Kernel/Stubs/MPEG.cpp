#include "Common.h"
#include "MPEG.h"
#include "MpegFfmpegDecoder.h"
#include "Syscalls/Helpers/RpcInvoke.h"

namespace ps2_stubs
{
    namespace
    {
        struct MpegRegisteredCallback
        {
            uint32_t type = 0u;
            uint32_t func = 0u;
            uint32_t data = 0u;
            uint32_t handle = 0u;
        };

        struct MpegPlaybackState
        {
            uint32_t picturesServed = 0u;
        };

        struct MpegStubState
        {
            bool initialized = false;
            uint32_t nextCallbackHandle = 1u;
            std::unordered_map<uint32_t, std::vector<MpegRegisteredCallback>> callbacksByMpeg;
            std::unordered_map<uint32_t, MpegPlaybackState> playbackByMpeg;
            PS2MpegCompatLayout compat;
        };

        std::mutex g_mpeg_stub_mutex;
        MpegStubState g_mpeg_stub_state;

        constexpr uint32_t kStubMovieWidth = 320u;
        constexpr uint32_t kStubMovieHeight = 240u;

        uint32_t mpegCompatSyntheticFrames(const PS2MpegCompatLayout &layout)
        {
            return layout.syntheticFramesBeforeEnd != 0u ? layout.syntheticFramesBeforeEnd : 1u;
        }

        MpegPlaybackState &getPlaybackState(uint32_t mpegAddr)
        {
            return g_mpeg_stub_state.playbackByMpeg[mpegAddr];
        }

        void resetMpegStubStateUnlocked()
        {
            const PS2MpegCompatLayout compat = g_mpeg_stub_state.compat;
            g_mpeg_stub_state.initialized = false;
            g_mpeg_stub_state.nextCallbackHandle = 1u;
            g_mpeg_stub_state.callbacksByMpeg.clear();
            g_mpeg_stub_state.playbackByMpeg.clear();
            g_mpeg_stub_state.compat = compat;
        }

        // -------------------------------------------------------------------
        // Phase 3+4 FMV bring-up (recomp2-local; dq8/PS2_PROJECT_STATE
        // §3.3x): synchronous host-side MPEG movie-decode session.
        //
        // Started by DQ8's cmd-0xC mailbox feeder (dq8/runner/
        // register_indirect.cpp's func_24D710 override) once it has read a
        // whole .MVI from the ISO and XOR-decrypted it with the fixed pad
        // (dq8/config/mvi_pad.h). Served frame-by-frame from
        // sceMpegGetPicture below, which is the only place that pulls from
        // it -- no background thread, no condition variable: the decode
        // step for "the next frame" runs synchronously inside the guest's
        // own sceMpegGetPicture call, so it can never race the N=1 fiber
        // scheduler or need a wakeup the scheduler doesn't know how to
        // deliver.
        struct MpegMovieSession
        {
            std::vector<uint8_t> videoEs;
            size_t esCursor = 0;
            std::unique_ptr<MpegFfmpegDecoder> decoder;
            std::deque<MpegDecodedFrame> frames;
            bool ended = false;

            // 2026-07-20 (Phase 4 metering, dq8 §3.31/§3.32): PTS-clock
            // pacing state for sceMpegGetPicture. baseVsyncTick is captured
            // the first time this session is asked for pacing (i.e. the
            // first sceMpegGetPicture call after the session started);
            // frame N is then "due" at baseVsyncTick +
            // floor(N*vsyncHz*fpsDen/fpsNum) -- see
            // reserveNextMovieFramePacingSlot() below.
            bool baseVsyncSet = false;
            uint64_t baseVsyncTick = 0u;
            uint64_t nextPacingFrameIndex = 0u;
            int fpsNum = 30000; // NTSC film-rate default (30000/1001 = 29.97 fps)
            int fpsDen = 1001;  // until the decoder reports a real rate (see below).
        };

        std::mutex g_movieSessionMutex;
        std::unique_ptr<MpegMovieSession> g_movieSession;
        uint32_t g_movieDoneByteAddr = 0u;

        // Vsync rate this metering assumes. DQ8 (NTSC) has been observed to
        // sustain 59.9-60 Hz on real boots (dq8/STATUS.md); a PAL title
        // would need 50 here. Kept as a single named constant rather than a
        // magic number in the due-tick formula below.
        constexpr uint64_t kMoviePacingVsyncHz = 60u;

        struct MoviePacingResult
        {
            bool active = false;
            uint64_t dueTick = 0u;
        };

        // Reserve this call's frame-pacing slot and report the vsync tick at
        // which it is "due" per the movie's own PTS clock. Takes+releases
        // g_movieSessionMutex internally and returns a plain value -- the
        // caller (sceMpegGetPicture) must NOT hold any lock while it then
        // parks on that tick (see the call site: WaitForNextVSyncTick parks
        // the pump fiber and only the IRQ worker thread wakes it; parking
        // while holding g_movieSessionMutex would deadlock the very thread
        // that needs to signal end-of-session cleanup elsewhere).
        //
        // due(N) is computed fresh from N every call (not accumulated frame
        // by frame), so integer-rounding never drifts across a long movie.
        MoviePacingResult reserveNextMovieFramePacingSlot()
        {
            std::lock_guard<std::mutex> lock(g_movieSessionMutex);
            if (!g_movieSession || g_movieSession->ended)
            {
                return {};
            }

            MpegMovieSession &session = *g_movieSession;
            const uint64_t nowTick = ps2_syscalls::GetCurrentVSyncTick();
            if (!session.baseVsyncSet)
            {
                session.baseVsyncSet = true;
                session.baseVsyncTick = nowTick;
                int num = 0;
                int den = 0;
                if (session.decoder && session.decoder->getFrameRate(num, den))
                {
                    session.fpsNum = num;
                    session.fpsDen = den;
                }
                std::cerr << "[MPEG] pacing: session base vsync tick=" << session.baseVsyncTick
                          << " fps=" << session.fpsNum << "/" << session.fpsDen << std::endl;
            }

            const uint64_t n = session.nextPacingFrameIndex++;
            const uint64_t dueTick = session.baseVsyncTick +
                (n * kMoviePacingVsyncHz * static_cast<uint64_t>(session.fpsDen)) /
                    static_cast<uint64_t>(session.fpsNum);
            return {true, dueTick};
        }

        enum class MovieFramePumpResult
        {
            NoSession, // no session active; caller should fall through to
                       // the pre-existing synthetic/RE:CV-compat path
            Frame,     // outFrame filled with the next decoded frame
            Ended,     // decoder drained on this call; session now retired
            Error,     // unrecoverable decoder-setup failure; session
                       // retired (scempeg-callback-design.md §C error path)
        };

        // Feed the decoder in bounded chunks (same shape as the Phase 2
        // offline validator, tools/mpeg_decode_test.cpp) until a frame is
        // available or the elementary stream is exhausted, flushing at true
        // end-of-stream. Pops and returns exactly one already-decoded frame
        // per call (frames beyond the first, if a chunk happened to decode
        // several at once, stay queued for the next call).
        MovieFramePumpResult pumpNextMovieFrame(MpegDecodedFrame &outFrame)
        {
            constexpr size_t kFeedChunk = 64u * 1024u;

            std::lock_guard<std::mutex> lock(g_movieSessionMutex);
            if (!g_movieSession)
            {
                return MovieFramePumpResult::NoSession;
            }
            if (g_movieSession->ended)
            {
                return MovieFramePumpResult::Ended;
            }

            MpegMovieSession &session = *g_movieSession;
            while (session.frames.empty() && session.esCursor < session.videoEs.size())
            {
                const size_t chunk = std::min(kFeedChunk, session.videoEs.size() - session.esCursor);
                if (!session.decoder->feed(session.videoEs.data() + session.esCursor, chunk, session.frames))
                {
                    // Unrecoverable decoder-setup failure (not a per-packet
                    // reject -- those are logged/skipped inside feed()).
                    // scempeg-callback-design.md §C: this is the only other
                    // trigger for the type-0 ("error") callback besides
                    // session-start failure.
                    session.ended = true;
                    return MovieFramePumpResult::Error;
                }
                session.esCursor += chunk;
            }
            if (session.frames.empty() && session.esCursor >= session.videoEs.size())
            {
                session.decoder->flush(session.frames);
            }
            if (session.frames.empty())
            {
                session.ended = true;
                return MovieFramePumpResult::Ended;
            }

            outFrame = std::move(session.frames.front());
            session.frames.pop_front();
            return MovieFramePumpResult::Frame;
        }

        void endMpegMovieSessionUnlocked()
        {
            g_movieSession.reset();
        }

        // Write one decoded RGBA frame into guest RDRAM in 16x16
        // macroblock-tiled, row-major order: tile (tileY,tileX) occupies
        // dst offset (tileY*(width/16)+tileX)*1024; within a tile, pixel
        // (px,py) sits at +(py*16+px)*4. Each 1024-byte tile is assembled in
        // a local scratch buffer first so the guest write is one contiguous
        // memcpy per tile instead of 256 separate getMemPtr() calls.
        void writeMpegFrameTiled(uint8_t *rdram, uint32_t dstAddr, const MpegDecodedFrame &frame)
        {
            const uint32_t dst = dstAddr & 0x0FFFFFFFu;
            const int width = frame.width;
            const int height = frame.height;
            if (width <= 0 || height <= 0 || (width % 16) != 0 || (height % 16) != 0)
            {
                std::cerr << "[MPEG] movie frame " << width << "x" << height
                          << " is not a multiple of the 16x16 macroblock tile size;"
                             " skipping guest write" << std::endl;
                return;
            }

            const int tilesX = width / 16;
            const int tilesY = height / 16;
            uint8_t tileBuf[16 * 16 * 4];
            for (int tileY = 0; tileY < tilesY; ++tileY)
            {
                for (int tileX = 0; tileX < tilesX; ++tileX)
                {
                    for (int py = 0; py < 16; ++py)
                    {
                        const uint8_t *srcRow = frame.rgba.data() +
                            (static_cast<size_t>(tileY * 16 + py) * static_cast<size_t>(width) +
                             static_cast<size_t>(tileX * 16)) * 4u;
                        uint8_t *dstRow = tileBuf + static_cast<size_t>(py) * 16u * 4u;
                        std::memcpy(dstRow, srcRow, 16u * 4u);
                        for (int px = 0; px < 16; ++px)
                        {
                            dstRow[px * 4 + 3] = 0x80u; // alpha; CT24 ignores it
                        }
                    }

                    // Macroblock tiles are laid out COLUMN-major in the picture
                    // buffer (all 16x16 tiles of tile-column 0 top-to-bottom,
                    // then column 1, ...), NOT row-major. Verified against the
                    // guest consumer: DQ8's movie texture-upload packet builder
                    // (func_19E5F0 @0x19e5f0) walks the buffer strictly
                    // sequentially (src += 0x400 per tile) with the tile-X index
                    // as its OUTER loop and tile-Y as the INNER one, emitting one
                    // 16x16 GS load-image block per tile. Row-major here put every
                    // tile in the wrong place: the decoded frame reached the
                    // screen as vertical stripes of correct-looking movie pixels
                    // (/tmp/dq8diag/20260720-S6-fix/frames/frame_000945.png).
                    const uint32_t tileOff = static_cast<uint32_t>(tileX * tilesY + tileY) * 1024u;
                    if (uint8_t *dstPtr = getMemPtr(rdram, dst + tileOff))
                    {
                        std::memcpy(dstPtr, tileBuf, sizeof(tileBuf));
                    }
                }
            }
        }

        // -------------------------------------------------------------
        // sceMpeg decode-event callback dispatch (scempeg-callback-design.md
        // §B/§C, 2026-07-23). DQ8 (and real libmpeg titles generally)
        // register up to 5 callback types {0,1,2,3,5} via sceMpegAddCallback
        // -- IPU/DMA-driver bookkeeping the game itself owns, which on real
        // HW fire from inside the library's GetPicture on the caller's own
        // thread. Our HLE decoder never invoked them (they were just
        // recorded); this is the generic, title-agnostic dispatch mechanism
        // that fires them faithfully, in-place, on the fiber that is already
        // executing sceMpegGetPicture.
        thread_local int s_mpegCbDepth = 0;

        // Lazily allocate the 0x20-byte guest scratch buffer the type-5
        // (PTS/DTS timestamp-pop) callback convention writes its two u64
        // results into. One buffer is enough: dispatch never recurses
        // (s_mpegCbDepth) and only ever runs on the single guest fiber
        // inside sceMpegGetPicture.
        uint32_t mpegCallbackScratchAddr(PS2Runtime *runtime)
        {
            static uint32_t s_scratch = 0u;
            if (s_scratch == 0u && runtime)
            {
                s_scratch = runtime->guestMalloc(0x20, 16u);
            }
            return s_scratch;
        }

        // Fire every registered callback of `type` for `mpegAddr`,
        // synchronously, on the CALLING guest fiber -- exactly the thread
        // real libmpeg runs decode-event callbacks on. Snapshot-then-release
        // g_mpeg_stub_mutex (§B.5: never held across rpcInvokeFunction,
        // which can re-enter sceMpeg stubs that take this same mutex).
        // Never call from the IRQ worker or any host thread (§B.3); the
        // depth guard (§B.4) makes a callback chain triggering further
        // dispatch a no-op rather than a reentrant call.
        uint32_t dispatchMpegCallbacks(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
                                        uint32_t mpegAddr, uint32_t type, uint32_t cbdataAddr)
        {
            if (s_mpegCbDepth > 0)
                return 1u;

            std::vector<MpegRegisteredCallback> snapshot;
            {
                std::lock_guard<std::mutex> lock(g_mpeg_stub_mutex);
                auto it = g_mpeg_stub_state.callbacksByMpeg.find(mpegAddr);
                if (it == g_mpeg_stub_state.callbacksByMpeg.end())
                    return 1u;
                for (const auto &cb : it->second)
                    if (cb.type == type)
                        snapshot.push_back(cb);
            }
            if (snapshot.empty())
                return 1u;

            static std::atomic<uint32_t> s_cbLogCount[6]{};
            static std::atomic<bool> s_cbLogTruncated[6]{};
            static const uint32_t kMaxCbLogs = ps2DiagEnvLimit("PS2X_MPEG_CB_MAX_LOGS", 40u);
            const uint32_t logBucket = (type < 6u) ? type : 5u;

            uint32_t v0 = 1u;
            ++s_mpegCbDepth;
            for (const auto &cb : snapshot)
            {
                uint32_t ret = 0u;
                const bool ok = rpcInvokeFunction(rdram, ctx, runtime, cb.func,
                                                  mpegAddr, cbdataAddr, cb.data, 0u, &ret);
                const uint32_t logIndex = s_cbLogCount[logBucket].fetch_add(1u, std::memory_order_relaxed);
                if (ps2DiagLogBudget(std::cerr,
                                     "[MPEG:cb]",
                                     "PS2X_MPEG_CB_MAX_LOGS",
                                     kMaxCbLogs,
                                     logIndex,
                                     s_cbLogTruncated[logBucket]))
                {
                    std::cerr << "[MPEG] cb type=" << type << " fn=0x" << std::hex << cb.func
                              << " ok=" << (ok ? 1 : 0) << " v0=0x" << ret << std::dec << std::endl;
                    if (type == 5u && cbdataAddr != 0u)
                    {
                        uint64_t pts = 0u;
                        uint64_t dts = 0u;
                        if (uint8_t *p = getMemPtr(rdram, cbdataAddr + 0x08u))
                            pts = *reinterpret_cast<uint64_t *>(p);
                        if (uint8_t *p = getMemPtr(rdram, cbdataAddr + 0x10u))
                            dts = *reinterpret_cast<uint64_t *>(p);
                        std::cerr << "[MPEG] cb type=5 pts/dts readback: "
                                  << static_cast<int64_t>(pts) << "/" << static_cast<int64_t>(dts)
                                  << std::endl;
                    }
                }
                if (ok)
                    v0 = ret;
            }
            --s_mpegCbDepth;
            return v0;
        }
    }

    bool startMpegMovieSession(std::vector<uint8_t> mviPlaintext)
    {
        std::vector<uint8_t> videoEs =
            demuxPssVideoElementaryStream(mviPlaintext.data(), mviPlaintext.size());
        if (videoEs.empty())
        {
            std::cerr << "[MPEG] movie session: demux produced 0 video ES bytes"
                         " (bad XOR pad or PSS layout?) -- not starting" << std::endl;
            return false;
        }

        std::lock_guard<std::mutex> lock(g_movieSessionMutex);
        g_movieSession = std::make_unique<MpegMovieSession>();
        g_movieSession->videoEs = std::move(videoEs);
        g_movieSession->decoder = std::make_unique<MpegFfmpegDecoder>();
        std::cerr << "[MPEG] movie session started: " << g_movieSession->videoEs.size()
                  << " bytes of MPEG-2 video ES" << std::endl;
        return true;
    }

    bool hasActiveMpegMovieSession()
    {
        std::lock_guard<std::mutex> lock(g_movieSessionMutex);
        return g_movieSession != nullptr && !g_movieSession->ended;
    }

    void setMpegMovieDoneByteAddr(uint32_t addr)
    {
        g_movieDoneByteAddr = addr;
    }

    void setMpegCompatLayout(const PS2MpegCompatLayout &layout)
    {
        std::lock_guard<std::mutex> lock(g_mpeg_stub_mutex);
        g_mpeg_stub_state.compat = layout;
    }

    void clearMpegCompatLayout()
    {
        std::lock_guard<std::mutex> lock(g_mpeg_stub_mutex);
        g_mpeg_stub_state.compat = {};
    }

    void resetMpegStubState()
    {
        std::lock_guard<std::mutex> lock(g_mpeg_stub_mutex);
        resetMpegStubStateUnlocked();
    }

    void sceMpegFlush(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegFlush", rdram, ctx, runtime);
    }

    void sceMpegAddBs(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegAddBs", rdram, ctx, runtime);
    }

    void sceMpegAddCallback(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;

        const uint32_t mpegAddr = getRegU32(ctx, 4);
        const uint32_t callbackType = getRegU32(ctx, 5);
        const uint32_t callbackFunc = getRegU32(ctx, 6);
        const uint32_t callbackData = getRegU32(ctx, 7);

        std::lock_guard<std::mutex> lock(g_mpeg_stub_mutex);
        g_mpeg_stub_state.initialized = true;
        (void)getPlaybackState(mpegAddr);

        const uint32_t handle = g_mpeg_stub_state.nextCallbackHandle++;
        g_mpeg_stub_state.callbacksByMpeg[mpegAddr].push_back(
            MpegRegisteredCallback{callbackType, callbackFunc, callbackData, handle});

        setReturnU32(ctx, handle);
    }

    void sceMpegAddStrCallback(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        setReturnU32(ctx, 0u);
    }

    void sceMpegClearRefBuff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)ctx;
        (void)runtime;
        static const uint32_t kRefGlobalAddrs[] = {
            0x171800u, 0x17180Cu, 0x171818u, 0x171804u, 0x171810u, 0x17181Cu};
        for (uint32_t addr : kRefGlobalAddrs)
        {
            uint8_t *p = getMemPtr(rdram, addr);
            if (!p)
                continue;
            uint32_t ptr = *reinterpret_cast<uint32_t *>(p);
            if (ptr != 0u)
            {
                uint8_t *q = getMemPtr(rdram, ptr + 0x28u);
                if (q)
                    *reinterpret_cast<uint32_t *>(q) = 0u;
            }
        }
        setReturnU32(ctx, 1u);
    }

    static void mpegGuestWrite32(uint8_t *rdram, uint32_t addr, uint32_t value)
    {
        if (uint8_t *p = getMemPtr(rdram, addr))
            *reinterpret_cast<uint32_t *>(p) = value;
    }
    static void mpegGuestWrite64(uint8_t *rdram, uint32_t addr, uint64_t value)
    {
        if (uint8_t *p = getMemPtr(rdram, addr))
        {
            *reinterpret_cast<uint32_t *>(p) = static_cast<uint32_t>(value);
            *reinterpret_cast<uint32_t *>(p + 4) = static_cast<uint32_t>(value >> 32);
        }
    }

    void sceMpegCreate(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t param_1 = getRegU32(ctx, 4); // a0
        const uint32_t param_2 = getRegU32(ctx, 5); // a1
        const uint32_t param_3 = getRegU32(ctx, 6); // a2

        const uint32_t uVar3 = (param_2 + 3u) & 0xFFFFFFFCu;
        const int32_t iVar2_signed = static_cast<int32_t>(param_3) - static_cast<int32_t>(uVar3 - param_2);

        if (iVar2_signed <= 0x117)
        {
            setReturnU32(ctx, 0u);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(g_mpeg_stub_mutex);
            getPlaybackState(param_1) = {};
        }

        const uint32_t puVar4 = uVar3 + 0x108u;
        const uint32_t innerSize = static_cast<uint32_t>(iVar2_signed) - 0x118u;

        mpegGuestWrite32(rdram, param_1 + 0x40, uVar3);

        const uint32_t a1_init = uVar3 + 0x118u;
        mpegGuestWrite32(rdram, puVar4 + 0x0, a1_init);
        mpegGuestWrite32(rdram, puVar4 + 0x4, innerSize);
        mpegGuestWrite32(rdram, puVar4 + 0x8, a1_init);
        mpegGuestWrite32(rdram, puVar4 + 0xC, a1_init);

        const uint32_t allocResult = runtime ? runtime->guestMalloc(0x600, 8u) : (uVar3 + 0x200u);
        mpegGuestWrite32(rdram, uVar3 + 0x44, allocResult);

        // param_1[0..2] = 0; param_1[4..0xe] = 0xffffffff/0 as per decompilation
        mpegGuestWrite32(rdram, param_1 + 0x00, 0);
        mpegGuestWrite32(rdram, param_1 + 0x04, 0);
        mpegGuestWrite32(rdram, param_1 + 0x08, 0);
        mpegGuestWrite64(rdram, param_1 + 0x10, 0xFFFFFFFFFFFFFFFFULL);
        mpegGuestWrite64(rdram, param_1 + 0x18, 0xFFFFFFFFFFFFFFFFULL);
        mpegGuestWrite64(rdram, param_1 + 0x20, 0);
        mpegGuestWrite64(rdram, param_1 + 0x28, 0xFFFFFFFFFFFFFFFFULL);
        mpegGuestWrite64(rdram, param_1 + 0x30, 0xFFFFFFFFFFFFFFFFULL);
        mpegGuestWrite64(rdram, param_1 + 0x38, 0);

        static const unsigned s_zeroOffsets[] = {
            0xB4, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC, 0xD0, 0xD4, 0xD8, 0xDC, 0xE0, 0xE4, 0xE8, 0xF8,
            0x0C, 0x14, 0x2C, 0x34, 0x3C,
            0x48, 0xFC, 0x100, 0x104, 0x70, 0x90, 0xAC};
        for (unsigned off : s_zeroOffsets)
            mpegGuestWrite32(rdram, uVar3 + off, 0u);
        mpegGuestWrite64(rdram, uVar3 + 0x78, 0);
        mpegGuestWrite64(rdram, uVar3 + 0x88, 0);

        mpegGuestWrite64(rdram, uVar3 + 0xF0, 0xFFFFFFFFFFFFFFFFULL);
        mpegGuestWrite32(rdram, uVar3 + 0x1C, 0x1209F8u);
        mpegGuestWrite32(rdram, uVar3 + 0x24, 0x120A08u);
        mpegGuestWrite32(rdram, uVar3 + 0xB0, 1u);
        mpegGuestWrite32(rdram, uVar3 + 0x9C, 0xFFFFFFFFu);
        mpegGuestWrite32(rdram, uVar3 + 0x80, 0xFFFFFFFFu);
        mpegGuestWrite32(rdram, uVar3 + 0x94, 0xFFFFFFFFu);
        mpegGuestWrite32(rdram, uVar3 + 0x98, 0xFFFFFFFFu);

        mpegGuestWrite32(rdram, 0x1717BCu, param_1);

        static const uint32_t s_refValues[] = {
            0x171A50u, 0x171C58u, 0x171CC0u, 0x171D28u, 0x171D90u,
            0x171AB8u, 0x171B20u, 0x171B88u, 0x171BF0u};
        for (unsigned i = 0; i < 9u; ++i)
            mpegGuestWrite32(rdram, 0x171800u + i * 4u, s_refValues[i]);

        uint32_t setDynamicRet = a1_init;
        if (uint8_t *p = getMemPtr(rdram, puVar4 + 8))
            setDynamicRet = *reinterpret_cast<uint32_t *>(p);
        mpegGuestWrite32(rdram, puVar4 + 12, setDynamicRet);

        setReturnU32(ctx, setDynamicRet);
    }

    void sceMpegDelete(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;

        const uint32_t mpegAddr = getRegU32(ctx, 4);
        std::lock_guard<std::mutex> lock(g_mpeg_stub_mutex);
        g_mpeg_stub_state.callbacksByMpeg.erase(mpegAddr);
        g_mpeg_stub_state.playbackByMpeg.erase(mpegAddr);
        setReturnU32(ctx, 0u);
    }

    void sceMpegDemuxPss(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegDemuxPss", rdram, ctx, runtime);
    }

    void sceMpegDemuxPssRing(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;

        const uint32_t availableBytes = getRegU32(ctx, 6);
        setReturnS32(ctx, static_cast<int32_t>(availableBytes));
    }

    void sceMpegDispCenterOffX(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegDispCenterOffX", rdram, ctx, runtime);
    }

    void sceMpegDispCenterOffY(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegDispCenterOffY", rdram, ctx, runtime);
    }

    void sceMpegDispHeight(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegDispHeight", rdram, ctx, runtime);
    }

    void sceMpegDispWidth(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegDispWidth", rdram, ctx, runtime);
    }

    void sceMpegGetDecodeMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegGetDecodeMode", rdram, ctx, runtime);
    }

    void sceMpegGetPicture(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t mpegAddr = getRegU32(ctx, 4);
        const uint32_t dstBufAddr = getRegU32(ctx, 5);

        // Diagnostic (recomp2-local, capped): confirm the guest video pump
        // actually reaches sceMpegGetPicture. The NoSession path below is
        // otherwise silent, so "zero [MPEG] logs" alone cannot distinguish
        // "GetPicture never called" from "called but no session".
        {
            static long s_getPicCalls = 0;
            if (s_getPicCalls < 20)
            {
                ++s_getPicCalls;
                std::cerr << "[MPEG] sceMpegGetPicture call #" << s_getPicCalls
                          << " mpeg=0x" << std::hex << mpegAddr
                          << " dst=0x" << dstBufAddr << std::dec
                          << " session=" << (hasActiveMpegMovieSession() ? "yes" : "no")
                          << std::endl;
            }
        }

        // sceMpeg decode-event callbacks (scempeg-callback-design.md §C):
        // type 2 ("stop-supply-DMA + capture position") fires once at
        // GetPicture entry, on real HW right before the library starts
        // taking the IPU for the next picture.
        if (hasActiveMpegMovieSession())
        {
            dispatchMpegCallbacks(rdram, ctx, runtime, mpegAddr, 2u, 0u);
        }

        // 2026-07-20 Phase 4 metering (GENERIC; upstream candidate along
        // with reserveNextMovieFramePacingSlot() above): park this fiber
        // until the movie's own PTS clock reaches this call's due vsync
        // tick, so the decode is metered to real playback speed instead of
        // draining the whole elementary stream across a handful of
        // host-synchronous calls (dq8 §3.31/§3.32 premature-EOF bug). NO
        // lock is held across the park -- reserveNextMovieFramePacingSlot()
        // already released g_movieSessionMutex before returning here;
        // WaitForNextVSyncTick parks the pump fiber and only the IRQ worker
        // thread (a real, separate host thread) wakes it on each real
        // vsync, so holding a lock across the park would risk a deadlock
        // against anything else that needs g_movieSessionMutex while this
        // fiber is parked.
        //
        // Type 3 ("background / keep-bitstream-flowing") reproduces
        // libmpeg's busy-wait-crank cadence: once per call if pacing is
        // active, plus once per vsync park -- no new timer/thread/interrupt
        // source, just riding the park this loop already does.
        {
            const MoviePacingResult pacing = reserveNextMovieFramePacingSlot();
            if (pacing.active)
            {
                dispatchMpegCallbacks(rdram, ctx, runtime, mpegAddr, 3u, 0u);
                while (ps2_syscalls::GetCurrentVSyncTick() < pacing.dueTick)
                {
                    (void)ps2_syscalls::WaitForNextVSyncTick(rdram, runtime);
                    dispatchMpegCallbacks(rdram, ctx, runtime, mpegAddr, 3u, 0u);
                }
            }
        }

        // Phase 3+4 FMV bring-up (recomp2-local): if a real movie-decode
        // session is active (started by DQ8's cmd-0xC feeder), pull+decode
        // the next frame synchronously and write it into guest RDRAM in
        // tiled macroblock order instead of the synthetic stub frame below.
        // NoSession (the common case for every game/path that never calls
        // startMpegMovieSession) falls through with the pre-existing
        // behavior completely unchanged.
        MpegDecodedFrame movieFrame;
        bool servedRealFrame = false;
        switch (pumpNextMovieFrame(movieFrame))
        {
        case MovieFramePumpResult::Frame:
        {
            writeMpegFrameTiled(rdram, dstBufAddr, movieFrame);
            servedRealFrame = true;
            static long s_frameLogCount = 0;
            static const uint32_t kMaxFrameLogs = ps2DiagEnvLimit("PS2X_MPEG_FRAME_MAX_LOGS", 200u);
            static std::atomic<bool> s_frameLogTruncated{false};
            if (ps2DiagLogBudget(std::cerr,
                                 "[MPEG:frame]",
                                 "PS2X_MPEG_FRAME_MAX_LOGS",
                                 kMaxFrameLogs,
                                 static_cast<uint32_t>(s_frameLogCount),
                                 s_frameLogTruncated))
            {
                ++s_frameLogCount;
                std::cerr << "[MPEG] movie session frame #" << s_frameLogCount << " "
                          << movieFrame.width << "x" << movieFrame.height
                          << " -> guest 0x" << std::hex << (dstBufAddr & 0x0FFFFFFFu)
                          << std::dec << std::endl;
            }

            // Type 5 ("give me this picture's PTS/DTS"): once per served
            // frame, after decode, before return (§C). cbdata is guest
            // scratch the callback forwarder fills in; zero it first so a
            // callback that only partially writes it can't leak a stale
            // timestamp into the read-back log.
            if (const uint32_t scratch = mpegCallbackScratchAddr(runtime))
            {
                if (uint8_t *p = getMemPtr(rdram, scratch))
                {
                    std::memset(p, 0, 0x20);
                }
                dispatchMpegCallbacks(rdram, ctx, runtime, mpegAddr, 5u, scratch);
            }
            break;
        }
        case MovieFramePumpResult::Ended:
        {
            std::cerr << "[MPEG] movie session reached EOF -- retiring session"
                      << (g_movieDoneByteAddr != 0u ? ", setting movie-done byte" : "")
                      << std::endl;
            if (g_movieDoneByteAddr != 0u)
            {
                if (uint8_t *p = getMemPtr(rdram, g_movieDoneByteAddr))
                {
                    *p = 1u;
                }
            }

            // Faithful libmpeg end-of-stream bookkeeping (§D.6): the
            // un-stubbed video-pump poll (func_1114E8) reads
            // *([mpegObj+0x40]+0x00) and takes its own EOF exit the moment
            // this word is nonzero. sceMpegCreate/sceMpegReset already own
            // this inner struct's other fields; this is the same object,
            // just the one word libmpeg itself would have set at EOF.
            if (uint8_t *base = getMemPtr(rdram, mpegAddr))
            {
                const uint32_t innerAddr = *reinterpret_cast<uint32_t *>(base + 0x40);
                if (uint8_t *inner = getMemPtr(rdram, innerAddr))
                {
                    *reinterpret_cast<uint32_t *>(inner + 0x00) = 1u;
                    std::cerr << "[MPEG] end-flag set, inner=0x" << std::hex << innerAddr
                              << std::dec << std::endl;
                }
            }

            std::lock_guard<std::mutex> lock(g_movieSessionMutex);
            endMpegMovieSessionUnlocked();
            break;
        }
        case MovieFramePumpResult::Error:
        {
            std::cerr << "[MPEG] movie session decoder error -- retiring session" << std::endl;
            dispatchMpegCallbacks(rdram, ctx, runtime, mpegAddr, 0u, 0u);
            std::lock_guard<std::mutex> lock(g_movieSessionMutex);
            endMpegMovieSessionUnlocked();
            break;
        }
        case MovieFramePumpResult::NoSession:
            break;
        }

        uint32_t picturesServed = 0u;
        PS2MpegCompatLayout compat{};
        {
            std::lock_guard<std::mutex> lock(g_mpeg_stub_mutex);
            MpegPlaybackState &playback = getPlaybackState(mpegAddr);
            mpegGuestWrite32(rdram, mpegAddr + 0x00u, kStubMovieWidth);
            mpegGuestWrite32(rdram, mpegAddr + 0x04u, kStubMovieHeight);
            mpegGuestWrite32(rdram, mpegAddr + 0x08u, playback.picturesServed);
            picturesServed = playback.picturesServed;
            compat = g_mpeg_stub_state.compat;
            playback.picturesServed += 1u;
        }

        if (servedRealFrame)
        {
            // Report REAL dims (overrides the stub write above) so the
            // guest pump's a2=(w*h)/256 and the 0xE0000 slot stride line up
            // with the frame we actually wrote.
            mpegGuestWrite32(rdram, mpegAddr + 0x00u, static_cast<uint32_t>(movieFrame.width));
            mpegGuestWrite32(rdram, mpegAddr + 0x04u, static_cast<uint32_t>(movieFrame.height));

            // 2026-07-20 (Checkpoint B): mpegObj+0x08 is the per-call
            // "picture-ready" status the guest video pump reads at
            // sub_0019E840 pc 0x19e910; `bnez` at 0x19e914 SKIPS the
            // func_19E5F0 GS texture-upload+draw whenever it is nonzero.
            // The generic path above writes the monotonic picturesServed
            // counter here (0,1,2,...), so every frame after the first
            // (counter!=0) had its upload skipped and never drew. Our
            // host decode is synchronous -- the picture IS ready the moment
            // GetPicture returns -- so write 0 ("ready, draw this frame")
            // for every served real frame. Faithful: 0 == picture available.
            mpegGuestWrite32(rdram, mpegAddr + 0x08u, 0u);
        }

        if (uint8_t *base = getMemPtr(rdram, mpegAddr))
        {
            const uint32_t iVar1 = *reinterpret_cast<uint32_t *>(base + 0x40);
            if (uint8_t *inner = getMemPtr(rdram, iVar1))
            {
                *reinterpret_cast<uint32_t *>(inner + 0xb0) = 1;
                *reinterpret_cast<uint32_t *>(inner + 0xd8) = (dstBufAddr & 0x0FFFFFFFu) | 0x20000000u;
                *reinterpret_cast<uint32_t *>(inner + 0xe4) = getRegU32(ctx, 6);
                *reinterpret_cast<uint32_t *>(inner + 0xdc) = 0;
                *reinterpret_cast<uint32_t *>(inner + 0xe0) = 0;
            }
        }

        if (!servedRealFrame &&
            compat.matchesMpegObject(mpegAddr) &&
            compat.hasFinishTargets() &&
            (picturesServed + 1u) >= mpegCompatSyntheticFrames(compat))
        {
            // No decoder yet: synthesize a safe frame so the guest can
            // initialize its movie presentation path, then mark playback finished.
            if (compat.videoStateAddr != 0u)
            {
                mpegGuestWrite32(rdram, compat.videoStateAddr, compat.finishedVideoStateValue);
            }
            if (compat.movieStateAddr != 0u)
            {
                mpegGuestWrite32(rdram, compat.movieStateAddr, compat.finishedMovieStateValue);
            }
        }

        // Type 1 ("restart-supply-DMA") fires once per call, after type 5,
        // before return (§C). Naturally gated: if this call's Ended case
        // just retired the session, hasActiveMpegMovieSession() is already
        // false and type 1 correctly does not fire on the terminating call.
        if (hasActiveMpegMovieSession())
        {
            dispatchMpegCallbacks(rdram, ctx, runtime, mpegAddr, 1u, 0u);
        }

        setReturnU32(ctx, 0u);
    }

    void sceMpegGetPictureRAW8(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegGetPictureRAW8", rdram, ctx, runtime);
    }

    void sceMpegGetPictureRAW8xy(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegGetPictureRAW8xy", rdram, ctx, runtime);
    }

    void sceMpegInit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;

        std::lock_guard<std::mutex> lock(g_mpeg_stub_mutex);
        resetMpegStubStateUnlocked();
        g_mpeg_stub_state.initialized = true;
        setReturnU32(ctx, 0u);
    }

    void sceMpegIsEnd(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        const uint32_t mpegAddr = getRegU32(ctx, 4);

        std::lock_guard<std::mutex> lock(g_mpeg_stub_mutex);
        g_mpeg_stub_state.initialized = true;
        const MpegPlaybackState &playback = getPlaybackState(mpegAddr);
        if (g_mpeg_stub_state.compat.matchesMpegObject(mpegAddr))
        {
            setReturnS32(ctx, playback.picturesServed >= mpegCompatSyntheticFrames(g_mpeg_stub_state.compat) ? 1 : 0);
            return;
        }

        // Generic fallback: keep decode threads alive until a game-specific path
        // decides to stop playback.
        setReturnS32(ctx, 0);
    }

    void sceMpegIsRefBuffEmpty(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegIsRefBuffEmpty", rdram, ctx, runtime);
    }

    void sceMpegReset(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t param_1 = getRegU32(ctx, 4);
        {
            std::lock_guard<std::mutex> lock(g_mpeg_stub_mutex);
            g_mpeg_stub_state.playbackByMpeg[param_1] = {};
        }
        uint8_t *base = getMemPtr(rdram, param_1);
        if (!base)
        {
            return;
        }
        uint32_t inner = *reinterpret_cast<uint32_t *>(base + 0x40);
        if (inner == 0u)
            return;
        mpegGuestWrite32(rdram, param_1 + 0x00u, 0u);
        mpegGuestWrite32(rdram, param_1 + 0x04u, 0u);
        mpegGuestWrite32(rdram, param_1 + 0x08u, 0u);
        mpegGuestWrite32(rdram, inner + 0x00, 0u);
        mpegGuestWrite32(rdram, inner + 0x04, 0u);
        mpegGuestWrite32(rdram, inner + 0x08, 0u);
        mpegGuestWrite32(rdram, param_1 + 0x08, 0u);
        mpegGuestWrite32(rdram, inner + 0x80, 0xFFFFFFFFu);
        mpegGuestWrite32(rdram, inner + 0xAC, 0u);
        mpegGuestWrite32(rdram, 0x171904u, 0u);
    }

    void sceMpegResetDefaultPtsGap(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegResetDefaultPtsGap", rdram, ctx, runtime);
    }

    void sceMpegSetDecodeMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegSetDecodeMode", rdram, ctx, runtime);
    }

    void sceMpegSetDefaultPtsGap(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegSetDefaultPtsGap", rdram, ctx, runtime);
    }

    void sceMpegSetImageBuff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceMpegSetImageBuff", rdram, ctx, runtime);
    }
}
