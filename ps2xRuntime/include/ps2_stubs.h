#pragma once

struct R5900Context;
class  PS2Runtime;

#include <cstdint>
#include <vector>
#include "ps2_call_list.h"
#include "runtime/ps2_memory.h"
#include "Stubs/Unimplemented.h"

struct PS2MpegCompatLayout
{
    uint32_t mpegObjectAddr = 0;
    uint32_t videoStateAddr = 0;
    uint32_t movieStateAddr = 0;
    uint32_t syntheticFramesBeforeEnd = 1u;
    uint32_t playingVideoStateValue = 0u;
    uint32_t playingMovieStateValue = 2u;
    uint32_t finishedVideoStateValue = 3u;
    uint32_t finishedMovieStateValue = 3u;

    [[nodiscard]] bool matchesMpegObject(uint32_t addr) const
    {
        return mpegObjectAddr != 0u && ((addr & PS2_RAM_MASK) == (mpegObjectAddr & PS2_RAM_MASK));
    }

    [[nodiscard]] bool hasFinishTargets() const
    {
        return videoStateAddr != 0u || movieStateAddr != 0u;
    }
};


namespace ps2_stubs
{
#define PS2_DECLARE_STUB(name) void name(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    PS2_STUB_LIST(PS2_DECLARE_STUB)
#undef PS2_DECLARE_STUB

    void resetSifState();

    void setMpegCompatLayout(const PS2MpegCompatLayout &layout);
    void clearMpegCompatLayout();

    // Phase 3+4 FMV bring-up (recomp2-local; dq8/PS2_PROJECT_STATE §3.3x):
    // synchronous host-side MPEG movie-decode session, implemented in
    // Kernel/Stubs/MPEG.cpp. Declared here (rather than in MPEG.h) so
    // callers outside the Kernel/Stubs tree -- e.g. dq8/runner/
    // register_indirect.cpp's cmd-0xC mailbox feeder -- can reach it with
    // just "ps2_stubs.h" (already included everywhere), matching the
    // existing setMpegCompatLayout()/clearMpegCompatLayout() pattern above.
    //
    // startMpegMovieSession: begin decoding an already XOR-decrypted, whole
    // .MVI file (MPEG-PS container bytes). Demuxes the video elementary
    // stream internally. Returns false (and starts nothing) if the demux
    // produces zero video ES bytes. Replaces/tears down any prior session.
    bool startMpegMovieSession(std::vector<uint8_t> mviPlaintext);
    // True while a movie-decode session is active and has not yet reached
    // end-of-stream.
    bool hasActiveMpegMovieSession();
    // Configure the guest address sceMpegGetPicture writes 1 to when the
    // active movie session's decoder drains (end-of-stream) -- the same
    // "movie done" byte the M0-era insta-EOF hack used to set unconditionally
    // (dq8/runner/register_indirect.cpp, [0x3d2a10]). 0 = disabled (no write).
    void setMpegMovieDoneByteAddr(uint32_t addr);
}
