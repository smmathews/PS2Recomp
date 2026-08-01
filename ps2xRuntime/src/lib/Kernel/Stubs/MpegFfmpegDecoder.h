#pragma once

// FMV decode blueprint, Phase 1+2 (recomp2-local; see dq8/PS2_PROJECT_STATE.md
// and the task that added this file for the full writeup).
//
// This header is deliberately FFmpeg-header-free (PIMPL) so it can be
// included from lightweight callers (e.g. the offline decode-test tool in
// ../../../../tools/mpeg_decode_test.cpp) without pulling <libavcodec/*.h>
// transitively.
//
// Ported from upstream PS2Recomp #120 (commit 8c8a97a,
// ps2xRuntime/src/lib/Kernel/Stubs/MPEG.cpp, class MpegFfmpegDecoder +
// the PSS/PES demux helpers in the same file's anonymous namespace), with
// ALL guest-RAM / ring-buffer / sceMpeg-callback-dispatch machinery
// stripped out — this file only knows about host byte buffers. That guest
// wiring is Phase 3's job (NOT done here — see the task's hard boundary:
// sceMpegGetPicture / the cmd-0xC EOF gate are UNCHANGED by this file).
//
// The existing 9 sceMpeg stubs in MPEG.cpp/MPEG.h are untouched; this is a
// new sibling file so Phase 3 can `#include "MpegFfmpegDecoder.h"` from
// MPEG.cpp later without having touched MPEG.cpp today.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

namespace ps2_stubs
{
    // One decoded video frame, converted to tightly-packed AV_PIX_FMT_RGBA
    // (width*height*4 bytes, row-major, no padding).
    struct MpegDecodedFrame
    {
        int width = 0;
        int height = 0;
        std::vector<uint8_t> rgba;
    };

    // Self-contained FFmpeg-backed MPEG-2 video elementary-stream decoder.
    // Feed() accepts raw MPEG-2 video ES bytes (already demuxed out of the
    // MPEG-PS container — see demuxPssVideoElementaryStream() below); it
    // internally runs them through av_parser_parse2() to find packet
    // boundaries, then avcodec_send_packet()/avcodec_receive_frame(), then
    // sws_scale() to AV_PIX_FMT_RGBA. Decoded frames are appended to the
    // caller-owned `frames` deque in decode order.
    class MpegFfmpegDecoder
    {
    public:
        MpegFfmpegDecoder();
        ~MpegFfmpegDecoder();

        MpegFfmpegDecoder(const MpegFfmpegDecoder &) = delete;
        MpegFfmpegDecoder &operator=(const MpegFfmpegDecoder &) = delete;

        // Feed a chunk of MPEG-2 video ES bytes. Returns false only on an
        // unrecoverable decoder-setup failure (e.g. codec/parser alloc
        // failed) — per-packet rejects are logged and skipped, not fatal.
        bool feed(const uint8_t *data, size_t size, std::deque<MpegDecodedFrame> &frames);

        // Drain any buffered frames at end-of-stream.
        bool flush(std::deque<MpegDecodedFrame> &frames);

        // Tear down decoder/parser/scaler state (also called by the dtor).
        void reset();

        // Count of packets rejected by the decoder/parser since construction
        // (av_parser_parse2 failures, avcodec_send_packet rejects,
        // avcodec_receive_frame failures). Used by the offline validation
        // entrypoint to report "0 errors" per the Phase 2 acceptance bar.
        uint32_t errorCount() const;

        // 2026-07-20 (Phase 3 metering): the decoded stream's frame rate, as
        // FFmpeg's AVCodecContext::framerate -- populated once the MPEG-2
        // sequence header has been parsed (i.e. not reliably available until
        // at least one frame has been decoded). Returns false (leaving
        // num/den untouched) if not yet known; callers should fall back to
        // a sane default (NTSC film-rate 30000/1001) in that case.
        bool getFrameRate(int &num, int &den) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };

    // Minimal MPEG-PS (Program Stream) demuxer for OFFLINE validation only.
    // Walks the full in-memory buffer `data` (already XOR-decrypted — see
    // dq8/config/mvi_pad.h) and extracts the MPEG-2 video elementary stream
    // (PES stream ids 0xE0-0xEF) payload bytes, concatenated in stream
    // order; audio (private_stream_1 / 0xC0-0xDF) and system/pack headers
    // are dropped. Ported from upstream #120's processPssBuffer() PES/pack
    // walking logic, simplified to operate on a complete buffer instead of
    // an incremental guest-fed ring (no guest addresses, no stream
    // callbacks — Phase 3 will need a streaming variant of this for the
    // live sceMpeg ring-buffer feed; this one is for the Phase 2 offline
    // decode-test tool).
    std::vector<uint8_t> demuxPssVideoElementaryStream(const uint8_t *data, size_t size);
}
