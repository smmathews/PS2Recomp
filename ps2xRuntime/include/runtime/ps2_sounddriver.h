#ifndef PS2_SOUNDDRIVER_H
#define PS2_SOUNDDRIVER_H

#include <cstdint>

// ---------------------------------------------------------------------------
// PS2SoundDriverGameLayout — game-parameterized SoundDriver (SDRDRV-family)
// SIF-RPC HLE.
//
// Many PS2 titles ship a vendor sound driver on the IOP (SDRDRV, sdrdrv.irx,
// or a derivative) that the EE talks to through exactly one SIF-RPC service
// id.  The EE-side driver keeps its own state machines in EE RAM and only
// needs a small number of VALUES produced by the IOP side (stream-ready
// flags, SPU2 transfer-channel handles, completion flags).  This struct lets
// a game runner describe those guest addresses + the service id + the
// relevant RPC function numbers; the generic handler in RPC.cpp
// (handleSoundDriverRpcService) then answers the RPCs at the SIF boundary —
// the only layer the EE can observe — and the guest driver advances its own
// state machines naturally.
//
// Everything defaults to 0 = feature disabled.  No game-specific values live
// in the runtime; a runner fills this in and calls
// ps2_syscalls::setSoundDriverGameLayout() during startup.
//
// This is intentionally a separate header (not ps2_runtime.h): it is only
// needed by the runner's setup code and by RPC.cpp, and must not force a
// rebuild of recompiled-function translation units.
// ---------------------------------------------------------------------------
struct PS2SoundDriverGameLayout
{
    // The SIF-RPC service id the game binds for its sound driver
    // (sceSifBindRpc second argument), e.g. 0x80000701 for SDRDRV.
    uint32_t serviceSid = 0;

    // --- stream open (the "stream is ready to play" VALUE gate) ---
    // RPC function number of the stream-open subcommand. When it arrives, the
    // handler writes streamStateReadyValue to streamStateAddr and signals
    // nowait completion.
    uint32_t streamOpenFno = 0;
    uint32_t streamStateAddr = 0;       // guest address of the stream-state word
    uint32_t streamStateReadyValue = 0; // value meaning "ready/playing"

    // --- channel configuration (sequencer channel arming) ---
    // RPC function number of the channel-config subcommand. When it arrives,
    // the handler sets channelAllocFlagTableAddr[channel] = 1 so the EE-side
    // per-tick channel driver picks the channel object up and advances it
    // naturally. 0 = not yet known / disabled (the handler's default branch
    // logs unknown fnos so the observed protocol can be learned first).
    uint32_t channelConfigFno = 0;
    uint32_t channelObjectAddr = 0;         // channel-0 sequencer object
    uint32_t channelAllocFlagTableAddr = 0; // byte table, one flag per channel

    // --- stop / teardown ---
    // RPC function number of the stop/teardown subcommand. When it arrives,
    // the handler writes 1 to stopCompletionFlagAddr and re-publishes the
    // SPU2 handles. 0 = disabled.
    uint32_t stopFno = 0;
    uint32_t stopCompletionFlagAddr = 0;
    uint32_t streamDescriptorAddr = 0;

    // --- streamed-audio DMA completion cursor (movie / streamed-PCM) ---
    // Some titles stream audio to SPU2 through an IOP-side ring the EE only
    // observes via a few words: a byte CURSOR the IOP advances as it consumes
    // the ring, the stream buffer SIZE, and a running consumed-byte count. An
    // EE-side poster thread polls "(cursor >= bufSize)" to decide the buffer
    // drained, then gates the next stage (e.g. a movie's per-frame video
    // tick). With the IOP HLE'd there is NO EE-side producer of the cursor, so
    // a runner that HLEs the sound driver must model the DMA completion: when
    // the stream-DMA mailbox command posts (its EE-side dispatch index is
    // streamDmaCmdIndex; the runner recognizes it and calls
    // modelSoundDriverStreamDmaCompletion below), write cursor := bufSize and
    // add bufSize to the running byte count. All addrs 0 = disabled.
    //
    // NOTE the runner is responsible for GATING this on a genuinely-active
    // streaming session (e.g. a live movie decode) before invoking the model
    // hook — advancing the cursor for ordinary non-streaming sound-driver
    // traffic would falsely signal "buffer drained" to the poster.
    uint32_t streamCursorAddr = 0;    // IOP-advanced consume cursor
    uint32_t streamBufSizeAddr = 0;   // stream buffer size word
    uint32_t streamByteCountAddr = 0; // running consumed-byte count (optional)
    uint32_t streamDmaCmdIndex = 0;   // EE mailbox dispatch index for stream DMA

    // --- SPU2 transfer-channel handles ---
    // Written once at first contact with the service (the EE driver reads
    // them back to program SPU2 transfers). Up to two address/value pairs;
    // address 0 = slot unused.
    uint32_t spu2HandleAddr[2] = {0, 0};
    uint32_t spu2HandleValue[2] = {0, 0};

    // --- default (benign) completion ---
    // Any other fno on serviceSid completes benignly: recv[0] is set to
    // benignStatusValue (callers of the known-benign subcommands discard it;
    // the value matches what a driver returns when no audio device work was
    // required).
    uint32_t benignStatusValue = 0xffffff9bu;

    [[nodiscard]] bool enabled() const { return serviceSid != 0u; }
};

namespace ps2_syscalls
{
    // Install/replace the active game layout. Thread-safe; call any time
    // after runtime construction (typically right after loadELF()).
    void setSoundDriverGameLayout(const PS2SoundDriverGameLayout &layout);

    // Model an IOP SPU2 streaming-DMA completion for the installed game
    // layout: writes streamCursorAddr := [streamBufSizeAddr] and, if
    // configured, adds that size to [streamByteCountAddr]. The EE-side poster
    // that polls "(cursor >= bufSize)" then observes the buffer as drained and
    // advances. Address-agnostic — all constants come from the layout. Returns
    // true if it wrote (layout configures the cursor/size addrs and bufSize>0),
    // false if disabled or not yet configured. The CALLER must gate this on an
    // active streaming session (see the streamCursorAddr note above).
    bool modelSoundDriverStreamDmaCompletion(uint8_t *rdram);
}

#endif // PS2_SOUNDDRIVER_H
