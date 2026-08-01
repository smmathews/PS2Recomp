#include "runtime/ps2_gif_arbiter.h"
#include "runtime/ps2_gs_gpu.h"
#include "ps2_log.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>

namespace
{
    std::atomic<uint32_t> s_debugGifArbiterSubmitCount{0};
    std::atomic<uint32_t> s_debugGifArbiterDrainCount{0};

    const char *pathName(GifPathId id)
    {
        switch (id)
        {
        case GifPathId::Path1:
            return "path1";
        case GifPathId::Path2:
            return "path2";
        case GifPathId::Path3:
            return "path3";
        default:
            return "path?";
        }
    }

    // GsDump (PS2X_GS_DUMP, see ps2_gs_gpu.h/.cpp) records raw GIF path
    // traffic in PCSX2's own on-disk GIF_PATH numbering, which is NOT the
    // same as this runtime's internal GifPathId enum below. PCSX2:
    // Path1=0, Path2=1, Path3=2, Path1New=3. This runtime's VU1 XGKICK
    // traffic (GifPathId::Path1) is exactly the traffic a real PS2 submits
    // through the "new" VU1-to-GIF route, so it maps to PCSX2's Path1New (3)
    // -- matching the oracle DQ8 capture, where 100% of traffic is tagged
    // path byte 3.
    uint8_t gsDumpPathByte(GifPathId id)
    {
        switch (id)
        {
        case GifPathId::Path1:
            return 3u; // PCSX2 GIF_PATH_1_NEW (VU1 XGKICK)
        case GifPathId::Path2:
            return 1u; // PCSX2 GIF_PATH_2
        case GifPathId::Path3:
            return 2u; // PCSX2 GIF_PATH_3
        default:
            return 3u;
        }
    }
}

GifArbiter::GifArbiter(ProcessPacketFn processFn)
    : m_processFn(std::move(processFn))
{
}

bool GifArbiter::isImagePacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes < 16u)
        return false;

    uint64_t tagLo = 0;
    std::memcpy(&tagLo, data, sizeof(tagLo));
    const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
    return flg == 2u;
}

void GifArbiter::submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl)
{
    if (!data || sizeBytes < 16 || !m_processFn)
        return;

    const uint32_t debugIndex = s_debugGifArbiterSubmitCount.fetch_add(1, std::memory_order_relaxed);
    if (debugIndex < 96u)
    {
        uint64_t tagLo = 0;
        std::memcpy(&tagLo, data, sizeof(tagLo));
        const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
        const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
        uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
        if (nreg == 0u)
            nreg = 16u;
        RUNTIME_LOG("[gif:submit] idx=" << debugIndex
                                        << " path=" << pathName(pathId)
                                        << " size=" << sizeBytes
                                        << " nloop=" << nloop
                                        << " flg=" << static_cast<uint32_t>(flg)
                                        << " nreg=" << nreg
                                        << " directhl=" << static_cast<uint32_t>(path2DirectHl ? 1u : 0u)
                                        << std::endl);
    }

    GifArbiterPacket pkt;
    pkt.pathId = pathId;
    pkt.path2DirectHl = (pathId == GifPathId::Path2) && path2DirectHl;
    pkt.path3Image = (pathId == GifPathId::Path3) && isImagePacket(data, sizeBytes);
    pkt.data.resize(sizeBytes);
    std::memcpy(pkt.data.data(), data, sizeBytes);
    m_queue.push_back(std::move(pkt));
}

void GifArbiter::drain()
{
    if (!m_processFn)
        return;

    // Reentrancy-safe dispatch (M-T1 fix, 2026-07-18, generic): move the
    // current queue out before processing. m_processFn ultimately reaches
    // GS::processGIFPacket(), whose TRXDIR handling can synchronously
    // trigger a local-to-local/local-to-host copy that (through the same
    // DMA glue) re-enters submit()/drain(). Iterating m_queue in place
    // while m_processFn can append to and re-drain THAT SAME vector meant
    // a nested drain() re-walked entries the outer loop had already
    // processed -- observed as a texture/CLUT upload's own 16-byte GIFtag
    // reappearing a second time where its real pixel payload should be
    // (byte-identical duplicate, confirmed via VRAM dump). Draining a
    // locally-owned batch and leaving m_queue empty for the reentrant call
    // makes double-processing structurally impossible regardless of the
    // exact reentry path.
    std::vector<GifArbiterPacket> batch = std::move(m_queue);
    m_queue.clear();

    std::stable_sort(batch.begin(), batch.end(),
                     [](const GifArbiterPacket &a, const GifArbiterPacket &b)
                     {
                         // DIRECTHL cannot preempt PATH3 IMAGE transfers.
                         if (a.path2DirectHl != b.path2DirectHl || a.path3Image != b.path3Image)
                         {
                             if (a.path3Image && b.path2DirectHl)
                                 return true;
                             if (a.path2DirectHl && b.path3Image)
                                 return false;
                         }
                         return pathPriority(a.pathId) < pathPriority(b.pathId);
                     });

    for (size_t i = 0; i < batch.size(); ++i)
    {
        auto &pkt = batch[i];
        if (!pkt.data.empty())
        {
            const uint32_t debugIndex = s_debugGifArbiterDrainCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 4000u)
            {
                uint64_t tagLo = 0;
                std::memcpy(&tagLo, pkt.data.data(), sizeof(tagLo));
                const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
                const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
                uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
                if (nreg == 0u)
                    nreg = 16u;
                std::cout << "[gif:drain] idx=" << debugIndex
                                               << " path=" << pathName(pkt.pathId)
                                               << " size=" << pkt.data.size()
                                               << " nloop=" << nloop
                                               << " flg=" << static_cast<uint32_t>(flg)
                                               << " nreg=" << nreg
                                               << " directhl=" << static_cast<uint32_t>(pkt.path2DirectHl ? 1u : 0u)
                                               << " path3image=" << static_cast<uint32_t>(pkt.path3Image ? 1u : 0u)
                                               << std::endl;
            }
            if (GsDump::isEnabled())
            {
                GsDump::writeTransfer(gsDumpPathByte(pkt.pathId), pkt.data.data(),
                                       static_cast<uint32_t>(pkt.data.size()));
            }
            m_processFn(pkt.data.data(), static_cast<uint32_t>(pkt.data.size()));
        }
    }
    // NOTE: do not clear m_queue here -- a reentrant submit() during the
    // loop above (see the comment at the top of this function) may have
    // queued NEW packets into m_queue that a nested drain() either already
    // processed (leaving m_queue empty again) or hasn't yet (leaving them
    // correctly pending for the next drain() call). Clearing unconditionally
    // here would silently drop whichever case left m_queue non-empty.
}

uint8_t GifArbiter::pathPriority(GifPathId id)
{
    return static_cast<uint8_t>(id);
}
