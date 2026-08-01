// mpeg_decode_test — Phase 2 OFFLINE validation entrypoint for the FMV decode
// blueprint (see dq8/PS2_PROJECT_STATE.md). Standalone binary, deliberately
// NOT linked into ps2_runtime / dq8_runner and NOT touching the guest
// sceMpeg* stubs (Kernel/Stubs/MPEG.cpp) — this only proves the ported
// MpegFfmpegDecoder + PSS demux (MpegFfmpegDecoder.h/.cpp, same dir tree)
// correctly decode a real DQ8 .MVI once XOR-decrypted with the recovered
// fixed pad (dq8/config/mvi_pad.h).
//
// Usage:
//   mpeg_decode_test <path/to/FILE.MVI> [maxFrames]
// or:
//   DQ8_DECODE_TEST=<path/to/FILE.MVI> mpeg_decode_test
//
// Dumps up to `maxFrames` (default 60) decoded frames as PNG to
// /tmp/dq8diag/20260720-decodetest/ (overridable via DQ8_DECODE_TEST_OUTDIR)
// and reports frame count / dimensions / decoder error count on stdout.

#include "../src/lib/Kernel/Stubs/MpegFfmpegDecoder.h"

extern "C"
{
#include <libavcodec/avcodec.h>
}

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// The 16 KB fixed DQ8 /MOVIE/*.MVI XOR pad, recovered + validated in the
// sibling dq8-recomp workspace (analysis/mvi-pad-recovery.md) and copied
// here verbatim. plaintext[i] = ciphertext[i] ^ kDq8MviPad[i % 0x4000].
#include "mvi_pad.h"

namespace
{
    std::vector<uint8_t> readFileBytes(const std::filesystem::path &path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
        {
            return {};
        }
        f.seekg(0, std::ios::end);
        const std::streamoff size = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<uint8_t> data(static_cast<size_t>(size));
        f.read(reinterpret_cast<char *>(data.data()), size);
        return data;
    }

    void xorDecryptInPlace(std::vector<uint8_t> &data)
    {
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = static_cast<uint8_t>(data[i] ^ kDq8MviPad[i % kDq8MviPadSize]);
        }
    }

    // Encode one RGBA frame as a real PNG file using FFmpeg's own PNG
    // encoder (AV_CODEC_ID_PNG) — avoids adding a separate PNG dependency
    // since ffmpeg is already linked.
    bool writePng(const ps2_stubs::MpegDecodedFrame &frame, const std::filesystem::path &outPath)
    {
        const AVCodec *pngCodec = avcodec_find_encoder(AV_CODEC_ID_PNG);
        if (!pngCodec)
        {
            std::cerr << "[decode-test] PNG encoder not found in this FFmpeg build" << std::endl;
            return false;
        }

        AVCodecContext *ctx = avcodec_alloc_context3(pngCodec);
        if (!ctx)
        {
            return false;
        }
        ctx->width = frame.width;
        ctx->height = frame.height;
        ctx->pix_fmt = AV_PIX_FMT_RGBA;
        ctx->time_base = {1, 25};

        bool ok = false;
        if (avcodec_open2(ctx, pngCodec, nullptr) >= 0)
        {
            AVFrame *avFrame = av_frame_alloc();
            avFrame->format = AV_PIX_FMT_RGBA;
            avFrame->width = frame.width;
            avFrame->height = frame.height;
            if (av_frame_get_buffer(avFrame, 0) >= 0)
            {
                const int rowBytes = frame.width * 4;
                for (int y = 0; y < frame.height; ++y)
                {
                    std::memcpy(avFrame->data[0] + static_cast<size_t>(y) * avFrame->linesize[0],
                                frame.rgba.data() + static_cast<size_t>(y) * rowBytes,
                                rowBytes);
                }

                AVPacket *pkt = av_packet_alloc();
                if (avcodec_send_frame(ctx, avFrame) >= 0 &&
                    avcodec_receive_packet(ctx, pkt) >= 0)
                {
                    std::ofstream out(outPath, std::ios::binary);
                    if (out)
                    {
                        out.write(reinterpret_cast<const char *>(pkt->data), pkt->size);
                        ok = out.good();
                    }
                }
                av_packet_free(&pkt);
            }
            av_frame_free(&avFrame);
        }
        avcodec_free_context(&ctx);
        return ok;
    }
}

int main(int argc, char **argv)
{
    std::string mviPath;
    size_t maxFrames = 60u;

    if (argc > 1)
    {
        mviPath = argv[1];
    }
    else if (const char *envPath = std::getenv("DQ8_DECODE_TEST"))
    {
        mviPath = envPath;
    }

    if (argc > 2)
    {
        maxFrames = static_cast<size_t>(std::strtoul(argv[2], nullptr, 10));
    }
    else if (const char *envFrames = std::getenv("DQ8_DECODE_TEST_FRAMES"))
    {
        maxFrames = static_cast<size_t>(std::strtoul(envFrames, nullptr, 10));
    }

    if (mviPath.empty())
    {
        std::cerr << "usage: mpeg_decode_test <path/to/FILE.MVI> [maxFrames]\n"
                     "   or: DQ8_DECODE_TEST=<path> mpeg_decode_test\n";
        return 2;
    }

    std::filesystem::path outDir = "/tmp/dq8diag/20260720-decodetest";
    if (const char *envOutDir = std::getenv("DQ8_DECODE_TEST_OUTDIR"))
    {
        outDir = envOutDir;
    }
    std::filesystem::create_directories(outDir);

    std::vector<uint8_t> cipher = readFileBytes(mviPath);
    if (cipher.empty())
    {
        std::cerr << "[decode-test] failed to read " << mviPath << std::endl;
        return 1;
    }
    std::cout << "[decode-test] read " << cipher.size() << " bytes from " << mviPath << std::endl;

    std::vector<uint8_t> plaintext = cipher;
    xorDecryptInPlace(plaintext);

    std::vector<uint8_t> videoEs = ps2_stubs::demuxPssVideoElementaryStream(plaintext.data(), plaintext.size());
    std::cout << "[decode-test] demuxed " << videoEs.size() << " bytes of MPEG-2 video ES" << std::endl;
    if (videoEs.empty())
    {
        std::cerr << "[decode-test] NOISE: demux produced zero video ES bytes — "
                     "XOR pad or PSS demux is wrong."
                  << std::endl;
        return 1;
    }

    ps2_stubs::MpegFfmpegDecoder decoder;
    std::deque<ps2_stubs::MpegDecodedFrame> frames;

    // Feed in bounded chunks and stop once we have enough frames for the
    // validation bar (default 60) — decoding an entire multi-minute movie
    // isn't necessary to prove the pipeline is correct.
    constexpr size_t kFeedChunk = 64u * 1024u;
    size_t offset = 0;
    bool feedOk = true;
    while (offset < videoEs.size() && frames.size() < maxFrames)
    {
        const size_t chunk = std::min(kFeedChunk, videoEs.size() - offset);
        feedOk = decoder.feed(videoEs.data() + offset, chunk, frames);
        offset += chunk;
        if (!feedOk)
        {
            break;
        }
    }
    // Only flush (drain) at true end-of-stream. Calling flush() after an
    // early stop (frames.size() reached maxFrames mid-GOP) forces the
    // decoder to terminate prediction early and yields a visibly smeared
    // final frame that is an artifact of THIS TEST's early-exit, not of the
    // ported decoder/demux — so skip it unless we actually consumed all of
    // the demuxed video ES.
    if (offset >= videoEs.size())
    {
        decoder.flush(frames);
    }

    const uint32_t errors = decoder.errorCount();
    std::cout << "[decode-test] feedOk=" << (feedOk ? "true" : "false")
              << " decodedFrames=" << frames.size()
              << " errorCount=" << errors << std::endl;

    if (frames.empty())
    {
        std::cerr << "[decode-test] NOISE: decoder produced zero frames." << std::endl;
        return 1;
    }

    std::cout << "[decode-test] first frame " << frames.front().width << "x" << frames.front().height
              << std::endl;

    size_t dumped = 0;
    for (size_t i = 0; i < frames.size(); ++i)
    {
        char name[64];
        std::snprintf(name, sizeof(name), "frame_%03zu.png", i);
        const std::filesystem::path outPath = outDir / name;
        if (writePng(frames[i], outPath))
        {
            ++dumped;
        }
        else
        {
            std::cerr << "[decode-test] failed to write " << outPath << std::endl;
        }
    }
    std::cout << "[decode-test] wrote " << dumped << " PNG(s) to " << outDir << std::endl;

    return (errors == 0u && dumped > 0u) ? 0 : 1;
}
