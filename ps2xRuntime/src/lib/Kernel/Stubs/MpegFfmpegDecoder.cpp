#include "MpegFfmpegDecoder.h"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/log.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>

// Ported from upstream PS2Recomp #120 (commit 8c8a97a,
// ps2xRuntime/src/lib/Kernel/Stubs/MPEG.cpp). See MpegFfmpegDecoder.h for
// what was deliberately left out (guest-RAM ties, ring buffer, callback
// dispatch — all Phase 3).

namespace ps2_stubs
{
    namespace
    {
        std::string ffmpegErrorString(int err)
        {
            std::array<char, AV_ERROR_MAX_STRING_SIZE> buffer{};
            if (av_strerror(err, buffer.data(), buffer.size()) < 0)
            {
                return "unknown FFmpeg error";
            }
            return std::string(buffer.data());
        }

        void configureFfmpegLogLevel()
        {
            static std::once_flag s_once;
            std::call_once(s_once, [] {
                av_log_set_level(AV_LOG_ERROR);
            });
        }

        // --- PSS/PES demux helpers (ported from upstream's anonymous
        // namespace in the same file; guest-address bookkeeping removed). ---

        constexpr uint8_t kMpegPackHeader = 0xBAu;
        constexpr uint8_t kMpegSystemHeader = 0xBBu;
        constexpr uint8_t kMpegProgramEnd = 0xB9u;
        constexpr size_t kStartCodeNotFound = std::numeric_limits<size_t>::max();

        size_t findStartCode(const uint8_t *data, size_t size, size_t from)
        {
            if (size < 4 || from >= size - 3u)
            {
                return kStartCodeNotFound;
            }
            for (size_t i = from; i + 3u < size; ++i)
            {
                if (data[i] == 0x00u && data[i + 1u] == 0x00u && data[i + 2u] == 0x01u)
                {
                    return i;
                }
            }
            return kStartCodeNotFound;
        }

        bool isVideoStreamId(uint8_t streamId)
        {
            return streamId >= 0xE0u && streamId <= 0xEFu;
        }

        bool isLengthPrefixedHeader(uint8_t streamId)
        {
            switch (streamId)
            {
            case kMpegSystemHeader:
            case 0xBCu: // program_stream_map
            case 0xBEu: // padding_stream
            case 0xBFu: // private_stream_2
            case 0xF0u: // ECM
            case 0xF1u: // EMM
            case 0xF2u: // DSMCC
            case 0xF8u: // ITU-T H.222.1 type E
            case 0xFFu: // program_stream_directory
                return true;
            default:
                return false;
            }
        }

        uint16_t readBe16(const uint8_t *p)
        {
            return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8u) |
                                         static_cast<uint16_t>(p[1]));
        }

        // packet points at the PES packet's own start code (00 00 01 <id>);
        // packetSize is the length of that packet (start code included).
        // Returns the offset (relative to `packet`) where the PES payload
        // begins, skipping the PES header / stuffing / PTS-DTS fields.
        size_t parsePesPayloadOffset(const uint8_t *packet, size_t packetSize)
        {
            if (!packet || packetSize <= 6u)
            {
                return packetSize;
            }

            size_t pos = 6u;
            if (packetSize >= 9u && (packet[pos] & 0xC0u) == 0x80u)
            {
                // MPEG-2 PES header: byte[6] top bits '10', byte[8] = PES_header_data_length.
                return std::min(packetSize, 9u + static_cast<size_t>(packet[pos + 2u]));
            }

            while (pos < packetSize && packet[pos] == 0xFFu)
            {
                ++pos;
            }

            if (pos + 1u < packetSize && (packet[pos] & 0xC0u) == 0x40u)
            {
                pos += 2u;
            }

            if (pos >= packetSize)
            {
                return packetSize;
            }

            const uint8_t flags = packet[pos];
            if ((flags & 0xF0u) == 0x20u)
            {
                pos += 5u;
            }
            else if ((flags & 0xF0u) == 0x30u)
            {
                pos += 10u;
            }
            else if (flags == 0x0Fu)
            {
                pos += 1u;
            }

            return std::min(packetSize, pos);
        }
    }

    std::vector<uint8_t> demuxPssVideoElementaryStream(const uint8_t *data, size_t size)
    {
        std::vector<uint8_t> out;
        if (!data || size < 4u)
        {
            return out;
        }

        size_t pos = 0u;
        while (true)
        {
            const size_t start = findStartCode(data, size, pos);
            if (start == kStartCodeNotFound || start + 3u >= size)
            {
                break;
            }

            const uint8_t streamId = data[start + 3u];

            if (streamId == kMpegProgramEnd)
            {
                break;
            }

            if (streamId == kMpegPackHeader)
            {
                if (start + 12u > size)
                {
                    break;
                }
                size_t packSize = 12u;
                if ((data[start + 4u] & 0xC0u) == 0x40u)
                {
                    if (start + 14u > size)
                    {
                        break;
                    }
                    packSize = 14u + static_cast<size_t>(data[start + 13u] & 0x07u);
                }
                if (start + packSize > size)
                {
                    break;
                }
                pos = start + packSize;
                continue;
            }

            if (start + 6u > size)
            {
                break;
            }

            const uint16_t packetLength = readBe16(data + start + 4u);

            if (isLengthPrefixedHeader(streamId))
            {
                const size_t packetEnd = start + 6u + static_cast<size_t>(packetLength);
                if (packetEnd > size)
                {
                    break;
                }
                pos = packetEnd;
                continue;
            }

            size_t packetEnd = 0u;
            if (packetLength != 0u)
            {
                packetEnd = start + 6u + static_cast<size_t>(packetLength);
                if (packetEnd > size)
                {
                    packetEnd = size;
                }
            }
            else
            {
                const size_t next = findStartCode(data, size, start + 6u);
                packetEnd = (next == kStartCodeNotFound) ? size : next;
            }

            if (isVideoStreamId(streamId))
            {
                const size_t payloadStart = start + parsePesPayloadOffset(data + start, packetEnd - start);
                if (payloadStart < packetEnd)
                {
                    out.insert(out.end(), data + payloadStart, data + packetEnd);
                }
            }
            // Audio (private_stream_1 / 0xC0-0xDF) and everything else is
            // dropped — Phase 2 offline validation is video-only per the
            // task's scope ("ignore audio for this phase").

            if (packetEnd <= start)
            {
                break; // safety: never spin without progress
            }
            pos = packetEnd;
        }

        return out;
    }

    // --- MpegFfmpegDecoder --------------------------------------------------

    struct MpegFfmpegDecoder::Impl
    {
        AVCodecParserContext *parser = nullptr;
        AVCodecContext *codecCtx = nullptr;
        AVFrame *frame = nullptr;
        AVPacket *packet = nullptr;
        SwsContext *swsCtx = nullptr;
        int swsWidth = 0;
        int swsHeight = 0;
        AVPixelFormat swsFormat = AV_PIX_FMT_NONE;
        bool initialized = false;
        bool drained = false;
        bool seenKeyframe = false;
        uint32_t errorCount = 0u;

        bool ensureInitialized()
        {
            if (initialized)
            {
                return true;
            }

            configureFfmpegLogLevel();

            const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_MPEG2VIDEO);
            if (!codec)
            {
                std::cerr << "[MPEG] FFmpeg MPEG-2 decoder not found." << std::endl;
                return false;
            }

            parser = av_parser_init(AV_CODEC_ID_MPEG2VIDEO);
            if (!parser)
            {
                std::cerr << "[MPEG] FFmpeg MPEG-video parser not found." << std::endl;
                return false;
            }

            codecCtx = avcodec_alloc_context3(codec);
            frame = av_frame_alloc();
            packet = av_packet_alloc();
            if (!codecCtx || !frame || !packet)
            {
                std::cerr << "[MPEG] failed to allocate FFmpeg decoder state." << std::endl;
                resetState();
                return false;
            }

            codecCtx->thread_count = 1;
            codecCtx->err_recognition = 0;
            const int ret = avcodec_open2(codecCtx, codec, nullptr);
            if (ret < 0)
            {
                std::cerr << "[MPEG] failed to open MPEG decoder: " << ffmpegErrorString(ret) << std::endl;
                resetState();
                return false;
            }

            initialized = true;
            drained = false;
            seenKeyframe = false;
            return true;
        }

        bool sendPacket(const uint8_t *data, size_t size, std::deque<MpegDecodedFrame> &frames)
        {
            if (!data || size == 0)
            {
                return true;
            }

            av_packet_unref(packet);
            const int allocRet = av_new_packet(packet, static_cast<int>(size));
            if (allocRet < 0)
            {
                std::cerr << "[MPEG] failed to allocate packet: " << ffmpegErrorString(allocRet) << std::endl;
                ++errorCount;
                return false;
            }
            std::memcpy(packet->data, data, size);

            int ret = avcodec_send_packet(codecCtx, packet);
            if (ret == AVERROR(EAGAIN))
            {
                if (!receiveFrames(frames))
                {
                    av_packet_unref(packet);
                    return false;
                }
                ret = avcodec_send_packet(codecCtx, packet);
            }
            av_packet_unref(packet);
            if (ret < 0 && ret != AVERROR(EAGAIN))
            {
                std::cerr << "[MPEG] decoder rejected packet, dropping: " << ffmpegErrorString(ret) << std::endl;
                ++errorCount;
                return true;
            }

            return receiveFrames(frames);
        }

        bool receiveFrames(std::deque<MpegDecodedFrame> &frames)
        {
            while (true)
            {
                const int ret = avcodec_receive_frame(codecCtx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                {
                    return true;
                }
                if (ret < 0)
                {
                    std::cerr << "[MPEG] decoder receive failed, dropping: " << ffmpegErrorString(ret) << std::endl;
                    ++errorCount;
                    return true;
                }

                if (!seenKeyframe)
                {
                    seenKeyframe = true;
                }

                if (!convertFrame(frames))
                {
                    av_frame_unref(frame);
                    ++errorCount;
                    return false;
                }
                av_frame_unref(frame);
            }
        }

        bool convertFrame(std::deque<MpegDecodedFrame> &frames)
        {
            const int width = frame->width;
            const int height = frame->height;
            const AVPixelFormat srcFormat = static_cast<AVPixelFormat>(frame->format);
            if (width <= 0 || height <= 0 || srcFormat == AV_PIX_FMT_NONE)
            {
                return false;
            }

            if (!swsCtx || swsWidth != width || swsHeight != height || swsFormat != srcFormat)
            {
                if (swsCtx)
                {
                    sws_freeContext(swsCtx);
                    swsCtx = nullptr;
                }
                swsCtx = sws_getContext(
                    width, height, srcFormat,
                    width, height, AV_PIX_FMT_RGBA,
                    SWS_BILINEAR, nullptr, nullptr, nullptr);
                if (!swsCtx)
                {
                    std::cerr << "[MPEG] failed to create FFmpeg scaler." << std::endl;
                    return false;
                }
                swsWidth = width;
                swsHeight = height;
                swsFormat = srcFormat;
            }

            MpegDecodedFrame decoded;
            decoded.width = width;
            decoded.height = height;
            decoded.rgba.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u);

            uint8_t *dstData[4] = {decoded.rgba.data(), nullptr, nullptr, nullptr};
            int dstLinesize[4] = {width * 4, 0, 0, 0};
            const int scaledRows = sws_scale(
                swsCtx, frame->data, frame->linesize, 0, height, dstData, dstLinesize);
            if (scaledRows <= 0)
            {
                std::cerr << "[MPEG] FFmpeg scaler produced no rows." << std::endl;
                return false;
            }

            frames.push_back(std::move(decoded));
            return true;
        }

        void resetState()
        {
            if (swsCtx)
            {
                sws_freeContext(swsCtx);
                swsCtx = nullptr;
            }
            if (frame)
            {
                av_frame_free(&frame);
            }
            if (packet)
            {
                av_packet_free(&packet);
            }
            if (codecCtx)
            {
                avcodec_free_context(&codecCtx);
            }
            if (parser)
            {
                av_parser_close(parser);
                parser = nullptr;
            }

            swsWidth = 0;
            swsHeight = 0;
            swsFormat = AV_PIX_FMT_NONE;
            initialized = false;
            drained = false;
        }
    };

    MpegFfmpegDecoder::MpegFfmpegDecoder() : m_impl(std::make_unique<Impl>()) {}

    MpegFfmpegDecoder::~MpegFfmpegDecoder()
    {
        reset();
    }

    bool MpegFfmpegDecoder::feed(const uint8_t *data, size_t size, std::deque<MpegDecodedFrame> &frames)
    {
        if (!data || size == 0)
        {
            return true;
        }
        if (!m_impl->ensureInitialized())
        {
            return false;
        }

        const uint8_t *cursor = data;
        size_t remaining = size;
        while (remaining > 0)
        {
            uint8_t *packetData = nullptr;
            int packetSize = 0;
            const int chunk = static_cast<int>(
                std::min<size_t>(remaining, static_cast<size_t>(std::numeric_limits<int>::max())));
            const int used = av_parser_parse2(
                m_impl->parser, m_impl->codecCtx, &packetData, &packetSize,
                cursor, chunk, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if (used < 0)
            {
                std::cerr << "[MPEG] parser failed: " << ffmpegErrorString(used) << std::endl;
                ++m_impl->errorCount;
                return false;
            }
            if (used == 0 && packetSize == 0)
            {
                break;
            }

            cursor += used;
            remaining -= static_cast<size_t>(used);

            if (packetSize > 0)
            {
                if (!m_impl->sendPacket(packetData, static_cast<size_t>(packetSize), frames))
                {
                    return false;
                }
            }
        }

        return true;
    }

    bool MpegFfmpegDecoder::flush(std::deque<MpegDecodedFrame> &frames)
    {
        if (!m_impl->initialized || m_impl->drained)
        {
            return true;
        }

        if (m_impl->parser)
        {
            uint8_t *packetData = nullptr;
            int packetSize = 0;
            const int used = av_parser_parse2(
                m_impl->parser, m_impl->codecCtx, &packetData, &packetSize,
                nullptr, 0, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            (void)used;
            if (packetSize > 0 && !m_impl->sendPacket(packetData, static_cast<size_t>(packetSize), frames))
            {
                return false;
            }
        }

        const int sendRet = avcodec_send_packet(m_impl->codecCtx, nullptr);
        if (sendRet < 0 && sendRet != AVERROR_EOF)
        {
            std::cerr << "[MPEG] decoder flush failed: " << ffmpegErrorString(sendRet) << std::endl;
            ++m_impl->errorCount;
            return false;
        }

        const bool ok = m_impl->receiveFrames(frames);
        m_impl->drained = true;
        return ok;
    }

    void MpegFfmpegDecoder::reset()
    {
        m_impl->resetState();
    }

    uint32_t MpegFfmpegDecoder::errorCount() const
    {
        return m_impl->errorCount;
    }

    bool MpegFfmpegDecoder::getFrameRate(int &num, int &den) const
    {
        if (!m_impl->initialized || !m_impl->codecCtx)
        {
            return false;
        }
        const AVRational fr = m_impl->codecCtx->framerate;
        if (fr.num <= 0 || fr.den <= 0)
        {
            return false;
        }
        num = fr.num;
        den = fr.den;
        return true;
    }
}
