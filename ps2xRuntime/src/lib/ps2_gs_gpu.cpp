#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_gs_common.h"
#include "runtime/ps2_gs_psmct16.h"
#include "runtime/ps2_gs_psmct32.h"
#include "runtime/ps2_gs_psmt4.h"
#include "runtime/ps2_gs_psmt8.h"
#include "ps2_log.h"
#include "ps2_syscalls.h"
#include "runtime/ps2_memory.h"
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>

// FMV-visibility RCA (2026-07-25). Milliseconds-since-GS-start of the most
// recent 512-wide CT32 IMAGE-mode upload -- i.e. the signature of an MPEG
// movie frame being streamed into VRAM as 16x16 tiles. Written by
// GS::processImageData, read by GS::debugDumpFieldFramebuffers (VRAM dump
// arming) and by the rasterizer's movie-quad probe, so both can scope
// themselves to "a movie is on screen right now" without a wall-clock guess
// or any cross-module header change. Definition lives here; the rasterizer
// declares it extern.
std::atomic<uint64_t> g_dq8MovieUploadMs{0};
std::atomic<uint32_t> g_dq8MovieUploadDbp{0};
// Incremented by the rasterizer's movie-quad probe (ps2_gs_rasterizer.cpp).
std::atomic<uint64_t> g_dq8MoviePrimsTotal{0};
std::atomic<uint64_t> g_dq8MoviePrimsTextured{0};

// One process-wide monotonic epoch shared by every FMV probe, so timestamps
// taken in different translation units / call sites are directly comparable.
uint64_t dq8ProbeNowMs()
{
    static const auto s_epoch = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - s_epoch)
            .count());
}

// ---------------------------------------------------------------------------
// GsDump (PS2X_GS_DUMP): raw GIF path traffic capture in a PCSX2
// GSdump-compatible on-disk format. See ps2_gs_gpu.h for the call surface.
//
// Format (deduced from the offline parser that already reads PCSX2 oracle
// .gs captures, gsparse2.py, against a known-good DQ8 dump with serial
// SLUS-21207 / crc 0xf4715852 that parses with zero bytes left over):
//
//   offset 0   u32  unused control word (parser never reads it)
//   offset 4   u32  "extra": byte length of the region starting at offset 8
//                   that the parser skips whole
//   offset 8   u32  hdr[0]  format/version tag
//   offset 12  u32  hdr[1]  size of a second skip region right after the
//                           "extra" region (VU1/VRAM freeze blob upstream;
//                           we emit none, so this is 0)
//   offset 16  u32  hdr[2]  size of this 9-field hdr array itself (36)
//   offset 20  u32  hdr[3]  serial string length
//   offset 24  u32  hdr[4]  crc
//   offset 28  u32  hdr[5]  presentation width
//   offset 32  u32  hdr[6]  presentation height
//   offset 36  u32  hdr[7]  size of (hdr array + serial string) = 36 + hdr[3]
//   offset 40  u32  hdr[8]  size of an upstream freeze-data blob; 0 here
//   offset 44  ...  serial string, hdr[3] bytes, no terminator
//   then       ...  hdr[8] bytes of freeze data (none, since hdr[8] == 0)
//   then       ...  hdr[1] bytes of the second skip region (none, since
//                   hdr[1] == 0)
//   then       8192 bytes: initial GS privileged-register snapshot; its
//                   CONTENT is never read by any offline tool downstream of
//                   the parser (only the offline scripts' xfers list
//                   matters), so we zero-fill it -- only the byte COUNT is
//                   load-bearing for the parser's fixed-offset framing.
//   then: packet stream to EOF, each packet self-delimiting:
//     u8 type
//       0 = Transfer: u8 path, u32 size (LE), then `size` raw bytes -- the
//           GIFtag + payload exactly as submitted to the GIF unit.
//       1 = VSync:    1 byte payload (not emitted by this writer)
//       2 = ReadFIFO2: 4 byte payload (not emitted by this writer)
//       3 = Registers: fixed-size 8192-byte snapshot (not emitted by this
//           writer)
//
// This writer only ever emits type-0 Transfer packets. That is a strict
// subset of the real format and round-trips through the same parser with
// zero bytes left over (verified with a standalone harness, see the DQ8
// investigation notes for this feature).
//
// Throughput (2026-07-31): the original writer called flush() after every
// single writeTransfer(), which is a syscall-per-packet and measured at
// 3.43 GB / 534,353 transfers in 1148s against a ~292s norm -- slow enough
// that a pad-input script's timed gates missed their windows and the run
// never reached FIELD. Fixed two ways, both below:
//   1. Buffering: a large filebuf (pubsetbuf, see kStreamBufferBytes) plus
//      an explicit flush policy (kFlushByteThreshold / kFlushIntervalMs)
//      checked only right after a whole packet has been written, so a
//      killed process still always leaves the file ending on a complete
//      packet boundary -- it can lose at most one flush interval's worth
//      of tail packets, never a torn one.
//   2. Path filter (PS2X_GS_DUMP_PATHS, env-gated, default = capture every
//      path so behaviour is unchanged unless set): the K3 analysis
//      consumer only needs Path1New (VU1 XGKICK traffic, on-disk byte 3);
//      a title-only run's path histogram was {1: 529851, 2: 4502} with
//      ZERO Path1New, i.e. essentially all 3.43 GB was traffic nobody
//      reads. Filtering happens BEFORE the packet is even built. A
//      filtered file is NEVER silently incomplete: the closing summary
//      always states whether a filter was active and, if so, the
//      kept/dropped counts per path, so a reader can never mistake a
//      filtered file for a full capture.
//
// `dumpPathByte` follows PCSX2's own on-disk GIF_PATH numbering, NOT this
// runtime's internal GifPathId enum (Path1=1/Path2=2/Path3=3 here, vs
// PCSX2's Path1=0/Path2=1/Path3=2/Path1New=3): the real oracle capture for
// DQ8 tags 100% of its VU1 XGKICK traffic as path byte 3 ("Path1New"), so
// callers must translate before calling writeTransfer -- see the mapping
// helper next to the GifArbiter::drain() call site in ps2_gif_arbiter.cpp.
namespace GsDump
{
    namespace
    {
        // Large filebuf attached via pubsetbuf() before open(), so the many
        // small writeTransfer() calls coalesce into few underlying write()
        // syscalls instead of one per packet.
        constexpr size_t kStreamBufferBytes = 4u * 1024u * 1024u; // 4 MiB

        // Explicit flush policy on top of the large filebuf. Checked only
        // right after a whole packet has already been written, never
        // mid-packet, so the on-disk file always ends on a complete packet
        // boundary even if the process is killed between flushes.
        constexpr uint64_t kFlushByteThreshold = 4u * 1024u * 1024u; // 4 MiB
        constexpr uint64_t kFlushIntervalMs = 2000u;                  // 2 s

        const char *dumpPathName(uint8_t b)
        {
            switch (b)
            {
            case 0: return "Path1";
            case 1: return "Path2";
            case 2: return "Path3";
            case 3: return "Path1New";
            default: return "Unknown";
            }
        }

        // Parses PS2X_GS_DUMP_PATHS: a comma/whitespace separated list of
        // on-disk GIF path bytes (0=Path1,1=Path2,2=Path3,3=Path1New) to
        // KEEP; every other byte is dropped before its packet is ever
        // built. Unset or empty -> returns false and leaves `allowed`
        // untouched (caller defaults it to all-true, i.e. unchanged legacy
        // "capture everything" behaviour). Out-of-range tokens are ignored.
        bool parsePathFilterEnv(bool (&allowed)[4])
        {
            const char *raw = std::getenv("PS2X_GS_DUMP_PATHS");
            if (!raw || !raw[0])
                return false;

            for (int i = 0; i < 4; ++i)
                allowed[i] = false;

            const char *p = raw;
            while (*p)
            {
                while (*p && !std::isdigit(static_cast<unsigned char>(*p)))
                    ++p;
                if (!*p)
                    break;
                char *end = nullptr;
                const long v = std::strtol(p, &end, 10);
                if (v >= 0 && v < 4)
                    allowed[static_cast<size_t>(v)] = true;
                p = (end != p) ? end : p + 1;
            }
            return true;
        }

        // Optional capture time window: PS2X_GS_DUMP_START_MS /
        // PS2X_GS_DUMP_END_MS, same monotonic epoch as dq8ProbeNowMs(), so a
        // capture can be scoped to e.g. the post-EOF FIELD window instead of
        // the whole run. Either bound may be set alone. Unset = no gating
        // (default, whole run captured).
        bool parseWindowEnv(int64_t &startMs, int64_t &endMs)
        {
            const char *s = std::getenv("PS2X_GS_DUMP_START_MS");
            const char *e = std::getenv("PS2X_GS_DUMP_END_MS");
            startMs = (s && s[0]) ? static_cast<int64_t>(std::strtoll(s, nullptr, 10)) : -1;
            endMs = (e && e[0]) ? static_cast<int64_t>(std::strtoll(e, nullptr, 10)) : -1;
            return (startMs >= 0) || (endMs >= 0);
        }

        struct State
        {
            std::mutex mutex;
            std::ofstream file;
            std::atomic<uint64_t> transfersWritten{0};
            std::atomic<uint64_t> bytesWritten{0};

            // Backing store for the ofstream's filebuf (see kStreamBufferBytes).
            std::vector<char> streamBuf = std::vector<char>(kStreamBufferBytes);
            uint64_t bytesSinceFlush = 0;
            uint64_t lastFlushMs = 0;

            // Path filter (PS2X_GS_DUMP_PATHS). Indexed by on-disk path
            // byte (0=Path1,1=Path2,2=Path3,3=Path1New).
            bool filterActive = false;
            bool pathAllowed[4] = {true, true, true, true};
            uint64_t keptByPath[4] = {0, 0, 0, 0};
            uint64_t droppedByPath[4] = {0, 0, 0, 0};

            // Optional time window (PS2X_GS_DUMP_START_MS/_END_MS).
            bool windowActive = false;
            int64_t windowStartMs = -1;
            int64_t windowEndMs = -1;
            uint64_t windowDropped = 0;
        };

        State &state()
        {
            static State s;
            return s;
        }

        void putU32(std::ofstream &f, uint32_t v)
        {
            f.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }

        // Caller holds state().mutex and has already verified state().file
        // is open.
        void writeHeaderLocked(std::ofstream &f)
        {
            static const char kSerial[10] = {'S', 'L', 'U', 'S', '-', '2', '1', '2', '0', '7'};
            static const uint32_t kSerialLen = static_cast<uint32_t>(sizeof(kSerial));
            static const uint32_t kHdrBytes = 9u * sizeof(uint32_t);
            static const uint32_t kCrc = 0xF4715852u; // DQ8 SLUS-21207
            static const uint32_t kWidth = 640u;
            static const uint32_t kHeight = 480u;

            putU32(f, 0xFFFFFFFFu);              // offset 0: unused control word
            putU32(f, kHdrBytes + kSerialLen);    // offset 4: "extra"
            putU32(f, 9u);                        // hdr[0]: format/version tag
            putU32(f, 0u);                        // hdr[1]: 2nd skip region size (none)
            putU32(f, kHdrBytes);                 // hdr[2]: size of this hdr array
            putU32(f, kSerialLen);                // hdr[3]: serial length
            putU32(f, kCrc);                       // hdr[4]: crc
            putU32(f, kWidth);                     // hdr[5]: width
            putU32(f, kHeight);                    // hdr[6]: height
            putU32(f, kHdrBytes + kSerialLen);    // hdr[7]: hdr + serial size
            putU32(f, 0u);                         // hdr[8]: freeze-data size (none)
            f.write(kSerial, sizeof(kSerial));
            static const std::vector<char> kRegPad(8192, 0);
            f.write(kRegPad.data(), static_cast<std::streamsize>(kRegPad.size()));
        }

        void shutdownLocked(State &s)
        {
            if (s.file.is_open())
            {
                s.file.flush();
                s.file.close();
                std::cerr << "[gsdump] closed: "
                          << s.transfersWritten.load(std::memory_order_relaxed) << " transfers, "
                          << s.bytesWritten.load(std::memory_order_relaxed) << " bytes"
                          << std::endl;

                // Announce filtering explicitly either way -- a reader must
                // never be able to mistake a filtered file for a complete
                // one just because this line was silent.
                if (s.filterActive)
                {
                    std::cerr << "[gsdump] FILTERED CAPTURE (PS2X_GS_DUMP_PATHS was set): "
                              << "this file does NOT contain every GIF path transferred this run"
                              << std::endl;
                    for (int i = 0; i < 4; ++i)
                    {
                        std::cerr << "[gsdump]   " << dumpPathName(static_cast<uint8_t>(i))
                                  << " (byte" << i << "): kept=" << s.keptByPath[i]
                                  << " dropped=" << s.droppedByPath[i]
                                  << (s.pathAllowed[i] ? " [captured]" : " [excluded]")
                                  << std::endl;
                    }
                }
                else
                {
                    std::cerr << "[gsdump] no path filter (PS2X_GS_DUMP_PATHS unset): "
                              << "all GIF paths captured" << std::endl;
                }

                if (s.windowActive)
                {
                    std::cerr << "[gsdump] time window active: [" << s.windowStartMs << ", "
                              << s.windowEndMs << "] ms since process start; "
                              << s.windowDropped << " transfers dropped as outside the window"
                              << std::endl;
                }
            }
        }

        extern "C" void gsDumpSignalHandler(int sig)
        {
            shutdown();
            std::signal(sig, SIG_DFL);
            std::raise(sig);
        }

        // Runs the env check + file open exactly once (C++11 magic-statics
        // guarantee); every later call anywhere in this TU just reads the
        // cached result. This is the single choke point for both init() and
        // isEnabled() so the file can never be opened twice.
        bool ensureInit()
        {
            static const bool enabled = []() -> bool
            {
                const char *path = std::getenv("PS2X_GS_DUMP");
                if (!path || !path[0])
                    return false;

                State &s = state();
                std::lock_guard<std::mutex> lock(s.mutex);

                s.filterActive = parsePathFilterEnv(s.pathAllowed);
                s.windowActive = parseWindowEnv(s.windowStartMs, s.windowEndMs);

                // Must be attached before open() to take effect.
                s.file.rdbuf()->pubsetbuf(s.streamBuf.data(),
                                          static_cast<std::streamsize>(s.streamBuf.size()));
                s.file.open(path, std::ios::binary | std::ios::trunc);
                if (!s.file.is_open())
                {
                    std::cerr << "[gsdump] failed to open '" << path
                              << "' for writing; dump disabled" << std::endl;
                    return false;
                }
                writeHeaderLocked(s.file);
                s.file.flush();
                s.lastFlushMs = dq8ProbeNowMs();
                std::atexit([]() { GsDump::shutdown(); });
                std::signal(SIGINT, gsDumpSignalHandler);
                std::signal(SIGTERM, gsDumpSignalHandler);
                std::cout << "[gsdump] enabled, writing GIF path traffic to '" << path << "'";
                if (s.filterActive)
                {
                    std::cout << " (PS2X_GS_DUMP_PATHS filter active, kept paths={";
                    bool first = true;
                    for (int i = 0; i < 4; ++i)
                    {
                        if (!s.pathAllowed[i])
                            continue;
                        if (!first)
                            std::cout << ",";
                        std::cout << dumpPathName(static_cast<uint8_t>(i));
                        first = false;
                    }
                    std::cout << "})";
                }
                if (s.windowActive)
                {
                    std::cout << " (time window [" << s.windowStartMs << ", " << s.windowEndMs
                              << "] ms since process start)";
                }
                std::cout << std::endl;
                return true;
            }();
            return enabled;
        }
    }

    void init()
    {
        ensureInit();
    }

    bool isEnabled()
    {
        return ensureInit();
    }

    void writeTransfer(uint8_t dumpPathByte, const uint8_t *data, uint32_t sizeBytes)
    {
        if (!ensureInit())
            return;
        if (!data || sizeBytes == 0u)
            return;

        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!s.file.is_open())
            return;

        // Filtering happens BEFORE the packet is built -- that is the
        // entire point, since the whole cost of a captured-but-unwanted
        // path was the write()+flush() this skips.
        if (s.windowActive)
        {
            const int64_t now = static_cast<int64_t>(dq8ProbeNowMs());
            const bool beforeStart = s.windowStartMs >= 0 && now < s.windowStartMs;
            const bool afterEnd = s.windowEndMs >= 0 && now > s.windowEndMs;
            if (beforeStart || afterEnd)
            {
                ++s.windowDropped;
                return;
            }
        }
        if (dumpPathByte < 4u && !s.pathAllowed[dumpPathByte])
        {
            ++s.droppedByPath[dumpPathByte];
            return;
        }
        if (dumpPathByte < 4u)
            ++s.keptByPath[dumpPathByte];

        // Build the whole packet (type + path + size + payload) in one
        // buffer and issue one write() call. flush() (below) is only ever
        // called right after this write() completes, so a process killed
        // between writeTransfer() calls always leaves a whole number of
        // complete packets on disk, never a truncated one.
        std::vector<char> packet;
        packet.reserve(6u + static_cast<size_t>(sizeBytes));
        packet.push_back(static_cast<char>(0)); // packet type 0 = Transfer
        packet.push_back(static_cast<char>(dumpPathByte));
        const uint32_t sz = sizeBytes;
        const char *szBytes = reinterpret_cast<const char *>(&sz);
        packet.insert(packet.end(), szBytes, szBytes + sizeof(sz));
        const char *payload = reinterpret_cast<const char *>(data);
        packet.insert(packet.end(), payload, payload + sizeBytes);

        s.file.write(packet.data(), static_cast<std::streamsize>(packet.size()));

        s.transfersWritten.fetch_add(1, std::memory_order_relaxed);
        s.bytesWritten.fetch_add(sizeBytes, std::memory_order_relaxed);

        // Flush on a byte/time threshold instead of every packet -- this is
        // the throughput fix (see the namespace-level comment above). Only
        // evaluated here, after a whole packet has already been written, so
        // the on-disk file always ends on a packet boundary.
        s.bytesSinceFlush += sizeBytes;
        const uint64_t now = dq8ProbeNowMs();
        if (s.bytesSinceFlush >= kFlushByteThreshold || (now - s.lastFlushMs) >= kFlushIntervalMs)
        {
            s.file.flush();
            s.bytesSinceFlush = 0;
            s.lastFlushMs = now;
        }
    }

    void shutdown()
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        shutdownLocked(s);
    }
}

// ---- emitted-vertex range census (PS2X_VTXSTAT=1) -------------------------
//
// "Does the geometry we emit actually span the screen in both axes?" XYZF2/
// XYZ2 carry X and Y as unsigned 12.4 fixed point biased by XYOFFSET, so the
// screen coordinate is (raw/16 - OF/16). A transform that has lost one axis
// shows up here immediately and unambiguously: X spreads over hundreds of
// pixels while Y is pinned to a couple of values, or every primitive is
// clamped to the 0 / 4095.9375 rails. Printed once every 5 s and reset, with
// the current draw FRAME base so in-engine geometry (fbp 0x000) can be told
// apart from the FMV blitter.
namespace
{
    bool gsVtxStatOn()
    {
        static const bool on = []()
        {
            const char *e = std::getenv("PS2X_VTXSTAT");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        return on;
    }

    struct GsVtxStat
    {
        uint64_t n = 0;
        float xmin = 1e30f, xmax = -1e30f;
        float ymin = 1e30f, ymax = -1e30f;
        double zmin = 1e300, zmax = -1e300;
        uint64_t xrail = 0, yrail = 0;
        uint64_t lastMs = 0;
    };

    // Two independent populations feeding the same [gs:vtx] census. Before
    // 2026-07-27 this note() was only ever called from GS::writeRegister's
    // ADREGLIST/A+D-driven XYZF2/XYZ2/XYZF3/XYZ3 handlers, so it silently
    // missed every PACKED-mode vertex (descriptors 0x04/0x05/0x0C/0x0D in
    // GS::writeRegisterPacked) -- which is how essentially all of DQ8's
    // VU1/PATH1 3D geometry actually arrives. That made the stat report look
    // like "no geometry" when in fact no geometry was being *watched*. Kept
    // as two separate accumulators (never merged) so a coverage gap like
    // this shows up as an all-zero population instead of silently vanishing
    // into someone else's range.
    enum class GsVtxPop
    {
        kAdRegList = 0, // XYZF2/XYZ2/XYZF3/XYZ3 via GS::writeRegister
        kPacked = 1,    // descriptors 0x04/0x05/0x0C/0x0D via GS::writeRegisterPacked
        kCount = 2,
    };

    GsVtxStat s_vtxStat[static_cast<size_t>(GsVtxPop::kCount)];

    const char *gsVtxPopName(GsVtxPop pop)
    {
        return pop == GsVtxPop::kPacked ? "packed" : "adreglist";
    }

    void gsVtxStatNote(GsVtxPop pop, float sx, float sy, double z, uint32_t fbp)
    {
        if (!gsVtxStatOn())
            return;
        GsVtxStat &s = s_vtxStat[static_cast<size_t>(pop)];
        ++s.n;
        if (sx < s.xmin) s.xmin = sx;
        if (sx > s.xmax) s.xmax = sx;
        if (sy < s.ymin) s.ymin = sy;
        if (sy > s.ymax) s.ymax = sy;
        if (z < s.zmin) s.zmin = z;
        if (z > s.zmax) s.zmax = z;
        if (sx <= 0.0f || sx >= 4095.9f) ++s.xrail;
        if (sy <= 0.0f || sy >= 4095.9f) ++s.yrail;

        const uint64_t now = dq8ProbeNowMs();
        if (s.lastMs == 0) { s.lastMs = now; return; }
        if (now - s.lastMs < 5000u) return;
        std::cerr << "[gs:vtx pop=" << gsVtxPopName(pop) << "] t=" << std::dec << now
                  << "ms n=" << s.n
                  << " fbp=0x" << std::hex << fbp << std::dec
                  << " x=[" << s.xmin << ',' << s.xmax << ']'
                  << " y=[" << s.ymin << ',' << s.ymax << ']'
                  << " z=[" << s.zmin << ',' << s.zmax << ']'
                  << " xrail=" << s.xrail << " yrail=" << s.yrail << std::endl;
        s = GsVtxStat{};
        s.lastMs = now;
    }
}

// ---- colour-by-draw-class probe (PS2X_COLOURPROBE=1) ----------------------
//
// 2026-07-27: a GIFtag REGS histogram found exactly two PACKED vertex-run
// classes -- one carrying ST+RGBAQ+XYZF2 per vertex (textured), one carrying
// a single RGBAQ then eight bare XYZF2 (no ST at all, ~73% of geometry) --
// and a separate sample of written pixels found source colours were
// overwhelmingly (0,0,0,0)/(0,0,0,128) instead of real scene colours. This
// probe tests the direct hypothesis: is class B painting black because its
// RGBAQ values themselves are black/near-black by construction, or is it
// texture-mapped with stale/absent texcoords (a different bug)? Bucketed by
// "did the enclosing GIFtag carry an ST register" rather than the full REGS
// signature, since that is the only distinction the hypothesis needs and it
// is cheap to compute once per tag in GS::processGIFPacket. Zero cost when
// PS2X_COLOURPROBE is unset.
namespace
{
    bool colourProbeOn()
    {
        static const bool on = []()
        {
            const char *e = std::getenv("PS2X_COLOURPROBE");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        return on;
    }

    // ---- ADC / vertex-queue window (PS2X_GS_ADC_WINDOW) -------------------
    //
    // GS::vertexKick used to `return` immediately on a NON-drawing kick
    // (PACKED XYZF2/XYZ2 with the ADC bit set, or an XYZF3/XYZ3 write). That
    // skips the queue maintenance at the bottom of the function, so for the
    // CONTINUOUS primitives (TRISTRIP/TRIFAN/LINESTRIP) the 3-entry sliding
    // window never advances past an ADC vertex: m_vtxCount just grows, later
    // vertices land in slots 3,4,5.. that GSRasterizer::drawTriangle never
    // reads (it reads m_vtxQueue[0..2] unconditionally), and the next drawing
    // kick emits a triangle built from stale vertices that straddle the
    // strip restart. On hardware the two ADC vertices at a strip restart are
    // exactly what flushes the previous strip out of the queue, so dropping
    // that maintenance splices the tail of one strip onto the head of the
    // next -- long thin triangles fanning between unrelated parts of a mesh.
    //
    // Gated so one binary can run both arms. OFF = historical behaviour.
    bool adcWindowFixOn()
    {
        static const bool on = []()
        {
            const char *e = std::getenv("PS2X_GS_ADC_WINDOW");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        return on;
    }

    // Unthrottled whole-run census of vertex kicks, split by primitive type
    // and by drawing/non-drawing, plus the count of non-drawing kicks that
    // land while the queue is already primed (m_vtxCount >= needed) -- those
    // are precisely the events where the historical early-return skipped
    // required window maintenance. There is no per-event log line and no
    // budget here, so nothing can be TRUNCATED; only the periodic summary is
    // printed. Single-threaded by construction, same as every other
    // file-scope probe accumulator in this TU.
    uint64_t s_vkKickDraw[8] = {0};
    uint64_t s_vkKickSkip[8] = {0};
    uint64_t s_vkDesync[8] = {0};
    uint64_t s_vkDrawn[8] = {0};
    uint64_t s_vkTotal = 0;

    // Set once per PACKED GIFtag (in GS::processGIFPacket, before the
    // vertex-run loop) to whether that tag's REGS list contains ST (desc
    // 0x02). Read back by writeRegisterPacked's RGBAQ (0x01) and vertex-kick
    // (0x04/0x05/0x0C/0x0D) handlers, which run later in the same tag under
    // the same GS::m_stateMutex lock -- single-threaded by construction, same
    // reasoning as every other file-scope probe accumulator in this TU.
    bool s_colourProbeTagHasST = true;

    const char *colourProbeClassName(bool hasST)
    {
        return hasST ? "hasST" : "noST";
    }

    // ---- RGBAQ value histogram, split by class -----------------------------
    struct RgbaHistEntry
    {
        uint32_t rgba = 0; // packed r<<24|g<<16|b<<8|a
        uint64_t count = 0;
    };
    constexpr size_t kRgbaHistCap = 64;

    struct RgbaqClassStat
    {
        std::vector<RgbaHistEntry> hist;
        uint64_t totalWrites = 0;
        uint64_t zeroRgbWrites = 0;
        uint64_t overflowWrites = 0;
    };
    RgbaqClassStat s_rgbaqStat[2]; // [0]=hasST [1]=noST
    uint64_t s_rgbaqHistLastPrintMs = 0;

    void rgbaqHistPrint()
    {
        std::cout << "[gs:rgbaq-hist] --- t=" << std::dec << dq8ProbeNowMs()
                   << "ms ---" << std::endl;
        for (int ci = 0; ci < 2; ++ci)
        {
            RgbaqClassStat &s = s_rgbaqStat[ci];
            std::vector<RgbaHistEntry> sorted = s.hist;
            std::sort(sorted.begin(), sorted.end(),
                      [](const RgbaHistEntry &a, const RgbaHistEntry &b)
                      { return a.count > b.count; });
            const double zeroFrac = s.totalWrites
                                         ? static_cast<double>(s.zeroRgbWrites) / static_cast<double>(s.totalWrites)
                                         : 0.0;
            std::cout << "[gs:rgbaq-hist] class=" << colourProbeClassName(ci == 0)
                       << " totalWrites=" << s.totalWrites
                       << " zeroRgbWrites=" << s.zeroRgbWrites
                       << " zeroRgbFrac=" << zeroFrac
                       << " distinct=" << s.hist.size()
                       << " overflowWrites=" << s.overflowWrites
                       << std::endl;
            const size_t topN = std::min<size_t>(8u, sorted.size());
            for (size_t i = 0; i < topN; ++i)
            {
                std::cout << "[gs:rgbaq-hist]   rgba=0x" << std::hex << sorted[i].rgba << std::dec
                           << " count=" << sorted[i].count << std::endl;
            }
        }
    }

    void rgbaqHistNote(bool hasST, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        RgbaqClassStat &s = s_rgbaqStat[hasST ? 0 : 1];
        ++s.totalWrites;
        if (r == 0u && g == 0u && b == 0u)
            ++s.zeroRgbWrites;

        const uint32_t key = (static_cast<uint32_t>(r) << 24) | (static_cast<uint32_t>(g) << 16) |
                              (static_cast<uint32_t>(b) << 8) | static_cast<uint32_t>(a);
        RgbaHistEntry *match = nullptr;
        for (auto &e : s.hist)
        {
            if (e.rgba == key)
            {
                match = &e;
                break;
            }
        }
        if (!match)
        {
            if (s.hist.size() < kRgbaHistCap)
            {
                s.hist.push_back(RgbaHistEntry{key, 0});
                match = &s.hist.back();
            }
            else
            {
                ++s.overflowWrites;
                match = nullptr;
            }
        }
        if (match)
            match->count += 1u;

        const uint64_t now = dq8ProbeNowMs();
        if (s_rgbaqHistLastPrintMs == 0u)
        {
            s_rgbaqHistLastPrintMs = now;
            return;
        }
        if (now - s_rgbaqHistLastPrintMs < 5000u)
            return;
        s_rgbaqHistLastPrintMs = now;
        rgbaqHistPrint();
    }

    // ---- PRIM/TEX0 draw-state sample, first 128 per class ------------------
    uint32_t s_drawStateSampled[2] = {0u, 0u}; // [0]=hasST [1]=noST
    std::atomic<bool> s_drawStateSampleTruncated[2]{};
    constexpr uint32_t kDrawStateSampleCapDefault = 128u;

    void drawStateSampleNote(bool hasST, const GSPrimReg &prim, const GSTex0Reg &tex0)
    {
        static const uint32_t kDrawStateSampleCap =
            ps2DiagEnvLimit("PS2X_DRAWSTATE_SAMPLE_MAX_LOGS", kDrawStateSampleCapDefault);
        uint32_t &sampled = s_drawStateSampled[hasST ? 0 : 1];
        if (!ps2DiagLogBudget(std::cout,
                             "[gs:drawstate]",
                             "PS2X_DRAWSTATE_SAMPLE_MAX_LOGS",
                             kDrawStateSampleCap,
                             sampled,
                             s_drawStateSampleTruncated[hasST ? 0 : 1]))
            return;
        const uint32_t idx = sampled++;
        std::cout << "[gs:drawstate] class=" << colourProbeClassName(hasST)
                   << " idx=" << idx
                   << " prim.type=" << static_cast<uint32_t>(prim.type)
                   << " tme=" << (prim.tme ? 1 : 0)
                   << " abe=" << (prim.abe ? 1 : 0)
                   << " fst=" << (prim.fst ? 1 : 0)
                   << " iip=" << (prim.iip ? 1 : 0);
        if (prim.tme)
        {
            std::cout << std::hex
                       << " tbp0=0x" << tex0.tbp0
                       << " cbp=0x" << tex0.cbp
                       << std::dec
                       << " tbw=" << static_cast<uint32_t>(tex0.tbw)
                       << " psm=0x" << std::hex << static_cast<uint32_t>(tex0.psm) << std::dec
                       << " tw=" << static_cast<uint32_t>(tex0.tw)
                       << " th=" << static_cast<uint32_t>(tex0.th)
                       << " cpsm=0x" << std::hex << static_cast<uint32_t>(tex0.cpsm) << std::dec
                       << " cld=" << static_cast<uint32_t>(tex0.cld);
        }
        std::cout << std::endl;
    }
}

namespace
{
    static constexpr uint32_t kDefaultDisplayWidth = 640u;
    static constexpr uint32_t kDefaultDisplayHeight = 448u;
    static constexpr uint32_t kHostFrameWidth = 640u;
    static constexpr uint32_t kHostFrameHeight = 512u;

    // Generic (title-agnostic) tracking of GS local-memory blocks that have
    // ever been referenced as a TEX0/TEX2 CLUT base pointer (CBP). Used only
    // to annotate diagnostic logs so we can tell, live, whether an upload or
    // local-to-local copy ever actually lands on a block a draw is about to
    // read its CLUT from -- without hardcoding any title-specific address.
    std::mutex &clutCbpMutex()
    {
        static std::mutex m;
        return m;
    }

    std::vector<uint32_t> &knownClutCbps()
    {
        static std::vector<uint32_t> v;
        return v;
    }

    void registerClutCbp(uint32_t cbp)
    {
        std::lock_guard<std::mutex> lock(clutCbpMutex());
        auto &v = knownClutCbps();
        if (std::find(v.begin(), v.end(), cbp) == v.end())
            v.push_back(cbp);
    }

    bool isKnownClutCbp(uint32_t cbp)
    {
        std::lock_guard<std::mutex> lock(clutCbpMutex());
        auto &v = knownClutCbps();
        return std::find(v.begin(), v.end(), cbp) != v.end();
    }

    uint16_t encodeFramePixelPSMCT16(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        return static_cast<uint16_t>(((r >> 3) & 0x1Fu) |
                                     (((g >> 3) & 0x1Fu) << 5) |
                                     (((b >> 3) & 0x1Fu) << 10) |
                                     ((a >= 0x40u) ? 0x8000u : 0u));
    }

    uint32_t addrPSMCT16Family(uint32_t basePtr, uint32_t width, uint8_t psm, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
            return GSPSMCT16::addrPSMCT16(basePtr, width, x, y);
        case GS_PSM_CT16S:
            return GSPSMCT16::addrPSMCT16S(basePtr, width, x, y);
        case GS_PSM_Z16:
            return GSPSMCT16::addrPSMZ16(basePtr, width, x, y);
        case GS_PSM_Z16S:
            return GSPSMCT16::addrPSMZ16S(basePtr, width, x, y);
        default:
            return 0u;
        }
    }

    static inline uint64_t loadLE64(const uint8_t *p)
    {
        uint64_t v;
        std::memcpy(&v, p, 8);
        return v;
    }

    void decodeDisplaySize(uint64_t display64, uint32_t &outWidth, uint32_t &outHeight)
    {
        const uint32_t dx = static_cast<uint32_t>((display64 >> 0) & 0x0FFFu);
        const uint32_t dy = static_cast<uint32_t>((display64 >> 12) & 0x07FFu);
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);

        outWidth = (dw + 1u) / (magh + 1u);
        outHeight = dh + 1u;

        if (outWidth < 64u || outHeight < 64u)
        {
            outWidth = kDefaultDisplayWidth;
            outHeight = kDefaultDisplayHeight;
        }

        outWidth = std::min<uint32_t>(outWidth, kHostFrameWidth);
        outHeight = std::min<uint32_t>(outHeight, kHostFrameHeight);
    }

    GSFrameReg decodeDisplayFrame(uint64_t dispfb64)
    {
        GSFrameReg frame{};
        frame.fbp = static_cast<uint32_t>(dispfb64 & 0x1FFu);
        frame.fbw = static_cast<uint32_t>((dispfb64 >> 9) & 0x3Fu);
        frame.psm = static_cast<uint8_t>((dispfb64 >> 15) & 0x1Fu);
        return frame;
    }

    struct GSDisplayReadOrigin
    {
        uint32_t x = 0u;
        uint32_t y = 0u;
    };

    GSDisplayReadOrigin decodeDisplayReadOrigin(uint64_t dispfb64)
    {
        GSDisplayReadOrigin origin{};
        origin.x = static_cast<uint32_t>((dispfb64 >> 32) & 0x7FFu);
        origin.y = static_cast<uint32_t>((dispfb64 >> 43) & 0x7FFu);
        return origin;
    }

    bool hasDisplaySetup(uint64_t display64, const GSFrameReg &frame)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);
        return frame.fbw != 0u || dw != 0u || dh != 0u || magh != 0u;
    }

    struct GSTransferTraversal
    {
        bool reverseX = false;
        bool reverseY = false;
    };

    GSTransferTraversal decodeTransferTraversal(uint8_t dir)
    {
        GSTransferTraversal traversal{};
        switch (dir & 0x3u)
        {
        case 1u:
            traversal.reverseY = true;
            break;
        case 2u:
            traversal.reverseX = true;
            break;
        case 3u:
            traversal.reverseX = true;
            traversal.reverseY = true;
            break;
        default:
            break;
        }
        return traversal;
    }

    uint32_t transferCoord(uint32_t start, uint32_t extent, uint32_t index, bool reverse)
    {
        if (reverse && extent != 0u)
        {
            return start + (extent - 1u - index);
        }
        return start + index;
    }

    struct GSPmodeState
    {
        bool enableCrt1 = false;
        bool enableCrt2 = false;
        bool mmod = false;
        bool amod = false;
        bool slbg = false;
        uint8_t alp = 0u;
    };

    GSPmodeState decodePmode(uint64_t pmode64)
    {
        GSPmodeState pmode{};
        pmode.enableCrt1 = (pmode64 & 0x1ull) != 0ull;
        pmode.enableCrt2 = (pmode64 & 0x2ull) != 0ull;
        pmode.mmod = ((pmode64 >> 5) & 0x1ull) != 0ull;
        pmode.amod = ((pmode64 >> 6) & 0x1ull) != 0ull;
        pmode.slbg = ((pmode64 >> 7) & 0x1ull) != 0ull;
        pmode.alp = static_cast<uint8_t>((pmode64 >> 8) & 0xFFu);
        return pmode;
    }

    struct GSSmode2State
    {
        bool interlaced = false;
        bool frameMode = true;
    };

    GSSmode2State decodeSMode2(uint64_t smode264)
    {
        GSSmode2State smode2{};
        smode2.interlaced = (smode264 & 0x1ull) != 0ull;
        smode2.frameMode = ((smode264 >> 1) & 0x1ull) != 0ull;
        return smode2;
    }

    void applyFieldPresentation(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height, bool oddField)
    {
        if (pixels.empty() || width == 0u || height < 2u)
        {
            return;
        }

        const std::vector<uint8_t> source = pixels;
        for (uint32_t y = 0; y < height; ++y)
        {
            uint32_t sourceY = ((y >> 1u) << 1u) + (oddField ? 1u : 0u);
            if (sourceY >= height)
            {
                sourceY = height - 1u;
            }

            const uint8_t *srcRow = source.data() + (sourceY * kHostFrameWidth * 4u);
            uint8_t *dstRow = pixels.data() + (y * kHostFrameWidth * 4u);
            std::memcpy(dstRow, srcRow, width * 4u);
        }
    }

    void normalizePresentationAlpha(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        if (pixels.empty() || width == 0u || height == 0u)
        {
            return;
        }

        for (uint32_t y = 0; y < height; ++y)
        {
            uint8_t *row = pixels.data() + (y * kHostFrameWidth * 4u);
            for (uint32_t x = 0; x < width; ++x)
            {
                row[x * 4u + 3u] = 255u;
            }
        }
    }

    uint8_t blendPresentationChannel(uint8_t src, uint8_t dst, uint32_t factor)
    {
        const int delta = static_cast<int>(src) - static_cast<int>(dst);
        return GSInternal::clampU8(static_cast<int>(dst) + ((delta * static_cast<int>(factor)) / 255));
    }

    uint32_t countNonBlackPixels(const std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        uint32_t count = 0u;
        for (uint32_t y = 0; y < height; ++y)
        {
            const uint8_t *row = pixels.data() + (y * kHostFrameWidth * 4u);
            for (uint32_t x = 0; x < width; ++x)
            {
                const uint8_t r = row[x * 4u + 0u];
                const uint8_t g = row[x * 4u + 1u];
                const uint8_t b = row[x * 4u + 2u];
                if (r != 0u || g != 0u || b != 0u)
                {
                    ++count;
                }
            }
        }
        return count;
    }

    bool clearFramebufferRect(uint8_t *vram,
                              uint32_t vramSize,
                              const GSContext &ctx,
                              uint32_t rgba)
    {
        if (!vram || vramSize == 0u || ctx.frame.fbw == 0u)
        {
            return false;
        }

        const uint32_t stride = GSInternal::fbStride(ctx.frame.fbw, ctx.frame.psm);
        if (stride == 0u)
        {
            return false;
        }

        const int x0 = std::max<int>(0, ctx.scissor.x0);
        const int x1 = std::max<int>(x0, ctx.scissor.x1);
        const int y0 = std::max<int>(0, ctx.scissor.y0);
        const int y1 = std::max<int>(y0, ctx.scissor.y1);
        const uint32_t base = ctx.frame.fbp * 8192u;

        uint8_t r = static_cast<uint8_t>(rgba & 0xFFu);
        uint8_t g = static_cast<uint8_t>((rgba >> 8) & 0xFFu);
        uint8_t b = static_cast<uint8_t>((rgba >> 16) & 0xFFu);
        uint8_t a = static_cast<uint8_t>((rgba >> 24) & 0xFFu);
        if ((ctx.fba & 0x1ull) != 0ull && ctx.frame.psm != GS_PSM_CT24)
        {
            a = static_cast<uint8_t>(a | 0x80u);
        }

        if (ctx.frame.psm == GS_PSM_CT32 || ctx.frame.psm == GS_PSM_CT24)
        {
            const uint32_t srcPixel =
                static_cast<uint32_t>(r) |
                (static_cast<uint32_t>(g) << 8) |
                (static_cast<uint32_t>(b) << 16) |
                (static_cast<uint32_t>(a) << 24);
            const uint32_t widthBlocks = (ctx.frame.fbw != 0u) ? ctx.frame.fbw : 1u;

            for (int y = y0; y <= y1; ++y)
            {
                for (int x = x0; x <= x1; ++x)
                {
                    const uint32_t off =
                        GSPSMCT32::addrPSMCT32(GSInternal::framePageBaseToBlock(ctx.frame.fbp),
                                               widthBlocks,
                                               static_cast<uint32_t>(x),
                                               static_cast<uint32_t>(y));
                    if (off + 4u > vramSize)
                    {
                        return true;
                    }

                    uint32_t pixel = srcPixel;
                    if (ctx.frame.fbmsk != 0u)
                    {
                        uint32_t existing = 0u;
                        std::memcpy(&existing, vram + off, sizeof(existing));
                        pixel = (pixel & ~ctx.frame.fbmsk) | (existing & ctx.frame.fbmsk);
                    }
                    std::memcpy(vram + off, &pixel, sizeof(pixel));
                }
            }
            return true;
        }

        if (ctx.frame.psm == GS_PSM_CT16 || ctx.frame.psm == GS_PSM_CT16S)
        {
            const uint16_t srcPixel = encodeFramePixelPSMCT16(r, g, b, a);
            const uint16_t mask = static_cast<uint16_t>(ctx.frame.fbmsk & 0xFFFFu);
            const uint32_t widthBlocks = (ctx.frame.fbw != 0u) ? ctx.frame.fbw : 1u;
            const uint32_t basePtr = GSInternal::framePageBaseToBlock(ctx.frame.fbp);

            for (int y = y0; y <= y1; ++y)
            {
                for (int x = x0; x <= x1; ++x)
                {
                    const uint32_t off = addrPSMCT16Family(basePtr,
                                                           widthBlocks,
                                                           ctx.frame.psm,
                                                           static_cast<uint32_t>(x),
                                                           static_cast<uint32_t>(y));
                    if (off + 2u > vramSize)
                    {
                        return true;
                    }

                    uint16_t pixel = srcPixel;
                    if (mask != 0u)
                    {
                        uint16_t existing = 0u;
                        std::memcpy(&existing, vram + off, sizeof(existing));
                        pixel = static_cast<uint16_t>((pixel & ~mask) | (existing & mask));
                    }
                    std::memcpy(vram + off, &pixel, sizeof(pixel));
                }
            }
            return true;
        }

        return false;
    }

    std::atomic<uint32_t> s_debugGifPacketCount{0};
    std::atomic<uint32_t> s_debugGsRegisterCount{0};
    std::atomic<uint32_t> s_debugGsPackedVertexCount{0};
    std::atomic<uint32_t> s_debugGsVertexKickCount{0};
    std::atomic<uint32_t> s_debugCopyRegCount{0};
    std::atomic<uint32_t> s_debugTexaWriteCount{0};
    std::atomic<uint32_t> s_debugCvFontUploadCount{0};
    std::atomic<uint32_t> s_debugLocalCopyCount{0};

    bool supportsFormatAwareLocalCopy(uint8_t psm)
    {
        switch (psm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        case GS_PSM_CT24:
        case GS_PSM_Z24:
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
        case GS_PSM_T8:
        case GS_PSM_T4:
        case GS_PSM_T8H:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
            return true;
        default:
            return false;
        }
    }

    uint32_t readTransferPixel(const uint8_t *vram,
                               uint32_t vramSize,
                               uint32_t basePtr,
                               uint8_t widthBlocks,
                               uint8_t psm,
                               uint32_t x,
                               uint32_t y)
    {
        const uint32_t width = (widthBlocks != 0u) ? static_cast<uint32_t>(widthBlocks) : 1u;
        const uint32_t base = basePtr * 256u;

        switch (psm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        {
            const uint32_t off = GSPSMCT32::addrPSMCT32(basePtr, width, x, y);
            if (off + 4u > vramSize)
                return 0u;
            uint32_t value = 0u;
            std::memcpy(&value, vram + off, sizeof(value));
            return value;
        }
        case GS_PSM_CT24:
        case GS_PSM_Z24:
        {
            const uint32_t off = GSPSMCT32::addrPSMCT32(basePtr, width, x, y);
            if (off + 3u > vramSize)
                return 0u;
            return static_cast<uint32_t>(vram[off + 0u]) |
                   (static_cast<uint32_t>(vram[off + 1u]) << 8) |
                   (static_cast<uint32_t>(vram[off + 2u]) << 16);
        }
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
        {
            const uint32_t off = addrPSMCT16Family(basePtr, width, psm, x, y);
            if (off + 2u > vramSize)
                return 0u;
            uint16_t value = 0u;
            std::memcpy(&value, vram + off, sizeof(value));
            return value;
        }
        case GS_PSM_T8:
        {
            const uint32_t off = GSPSMT8::addrPSMT8(basePtr, width, x, y);
            return (off < vramSize) ? vram[off] : 0u;
        }
        case GS_PSM_T4:
        {
            const uint32_t nibbleAddr = GSPSMT4::addrPSMT4(basePtr, width, x, y);
            const uint32_t byteOff = nibbleAddr >> 1;
            if (byteOff >= vramSize)
                return 0u;
            const int shift = static_cast<int>((nibbleAddr & 1u) << 2);
            return static_cast<uint32_t>((vram[byteOff] >> shift) & 0x0Fu);
        }
        // T8H/T4HL/T4HH share one CT32-addressed word: the index lives in
        // the alpha byte (T8H = bits 24-31, T4HL = bits 24-27, T4HH = bits
        // 28-31). Address it as a CT32 word, not the packed T4/T8 buffers.
        // Returned value keeps the index bits in their storage position
        // (masked, not shifted down) so it pairs directly with
        // writeTransferPixel()'s RMW below and with the position-masked
        // values processImageData() hands it after unpacking the packed
        // 4bpp/8bpp wire format (see the T4HL/T4HH/T8H branch there).
        case GS_PSM_T8H:
        {
            const uint32_t off = GSPSMCT32::addrPSMCT32(basePtr, width, x, y);
            if (off + 4u > vramSize)
                return 0u;
            uint32_t word = 0u;
            std::memcpy(&word, vram + off, sizeof(word));
            return word & 0xFF000000u;
        }
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
        {
            const uint32_t off = GSPSMCT32::addrPSMCT32(basePtr, width, x, y);
            if (off + 4u > vramSize)
                return 0u;
            uint32_t word = 0u;
            std::memcpy(&word, vram + off, sizeof(word));
            const uint32_t shift = (psm == GS_PSM_T4HH) ? 28u : 24u;
            return word & (0x0Fu << shift);
        }
        default:
            return 0u;
        }
    }

    void writeTransferPixel(uint8_t *vram,
                            uint32_t vramSize,
                            uint32_t basePtr,
                            uint8_t widthBlocks,
                            uint8_t psm,
                            uint32_t x,
                            uint32_t y,
                            uint32_t value)
    {
        const uint32_t width = (widthBlocks != 0u) ? static_cast<uint32_t>(widthBlocks) : 1u;
        const uint32_t base = basePtr * 256u;

        switch (psm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        {
            const uint32_t off = GSPSMCT32::addrPSMCT32(basePtr, width, x, y);
            if (off + 4u > vramSize)
                return;
            std::memcpy(vram + off, &value, sizeof(value));
            return;
        }
        case GS_PSM_CT24:
        case GS_PSM_Z24:
        {
            const uint32_t off = GSPSMCT32::addrPSMCT32(basePtr, width, x, y);
            if (off + 3u > vramSize)
                return;
            vram[off + 0u] = static_cast<uint8_t>(value & 0xFFu);
            vram[off + 1u] = static_cast<uint8_t>((value >> 8) & 0xFFu);
            vram[off + 2u] = static_cast<uint8_t>((value >> 16) & 0xFFu);
            return;
        }
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
        {
            const uint32_t off = addrPSMCT16Family(basePtr, width, psm, x, y);
            if (off + 2u > vramSize)
                return;
            const uint16_t value16 = static_cast<uint16_t>(value & 0xFFFFu);
            std::memcpy(vram + off, &value16, sizeof(value16));
            return;
        }
        case GS_PSM_T8:
        {
            const uint32_t off = GSPSMT8::addrPSMT8(basePtr, width, x, y);
            if (off < vramSize)
                vram[off] = static_cast<uint8_t>(value & 0xFFu);
            return;
        }
        case GS_PSM_T4:
        {
            const uint32_t nibbleAddr = GSPSMT4::addrPSMT4(basePtr, width, x, y);
            const uint32_t byteOff = nibbleAddr >> 1;
            if (byteOff >= vramSize)
                return;
            const uint8_t nibble = static_cast<uint8_t>(value & 0x0Fu);
            uint8_t &dst = vram[byteOff];
            if ((nibbleAddr & 1u) != 0u)
                dst = static_cast<uint8_t>((dst & 0x0Fu) | (nibble << 4));
            else
                dst = static_cast<uint8_t>((dst & 0xF0u) | nibble);
            return;
        }
        // T8H/T4HL/T4HH: read-modify-write the shared CT32 word, touching
        // only the plane's own bits (alpha byte for T8H, one nibble of the
        // alpha byte for T4HL/T4HH) so the other plane(s) packed into the
        // same word are preserved. `value` is expected to already carry its
        // index bits in storage position (see readTransferPixel() above and
        // processImageData()'s packed-4bpp/8bpp host upload path, which
        // shifts each unpacked nibble/byte into position before calling
        // here) -- it is masked here, not shifted.
        case GS_PSM_T8H:
        {
            const uint32_t off = GSPSMCT32::addrPSMCT32(basePtr, width, x, y);
            if (off + 4u > vramSize)
                return;
            uint32_t word = 0u;
            std::memcpy(&word, vram + off, sizeof(word));
            word = (word & 0x00FFFFFFu) | (value & 0xFF000000u);
            std::memcpy(vram + off, &word, sizeof(word));
            return;
        }
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
        {
            const uint32_t off = GSPSMCT32::addrPSMCT32(basePtr, width, x, y);
            if (off + 4u > vramSize)
                return;
            uint32_t word = 0u;
            std::memcpy(&word, vram + off, sizeof(word));
            const uint32_t mask = 0x0Fu << ((psm == GS_PSM_T4HH) ? 28u : 24u);
            word = (word & ~mask) | (value & mask);
            std::memcpy(vram + off, &word, sizeof(word));
            return;
        }
        default:
            return;
        }
    }
}

using namespace GSInternal;

GS::GS()
{
    reset();
}

void GS::init(uint8_t *vram, uint32_t vramSize, GSRegisters *privRegs)
{
    m_vram = vram;
    m_vramSize = vramSize;
    m_privRegs = privRegs;
    reset();
    GsDump::init(); // no-op unless PS2X_GS_DUMP is set; opens the dump file early
}

void GS::reset()
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    std::memset(m_ctx, 0, sizeof(m_ctx));
    m_prim = {};
    m_curR = 0x80;
    m_curG = 0x80;
    m_curB = 0x80;
    m_curA = 0x80;
    m_curQ = 1.0f;
    m_curS = 0.0f;
    m_curT = 0.0f;
    m_curU = 0;
    m_curV = 0;
    m_curFog = 0;
    m_prmodecont = true;
    m_pabe = false;
    m_texa = {0u, false, 0u};
    m_texclut = {0u, 0u, 0u};
    m_bitbltbuf = {};
    m_trxpos = {};
    m_trxreg = {};
    m_trxdir = 3;
    m_hwregX = 0;
    m_hwregY = 0;
    m_vtxCount = 0;
    m_vtxIndex = 0;
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0;
    m_preferredDisplaySourceFrame = {};
    m_preferredDisplayDestFbp = 0;
    m_hasPreferredDisplaySource = false;
    m_hostPresentationFrame.clear();
    m_hostPresentationWidth = 0u;
    m_hostPresentationHeight = 0u;
    m_hostPresentationDisplayFbp = 0u;
    m_hostPresentationSourceFbp = 0u;
    m_hostPresentationUsedPreferred = false;
    m_hasHostPresentationFrame = false;

    for (int i = 0; i < 2; ++i)
    {
        m_ctx[i].frame.fbw = 10;
        m_ctx[i].scissor = {0, 639, 0, 447};
        m_ctx[i].xyoffset = {0, 0};
    }
}

GSContext &GS::activeContext()
{
    return m_ctx[m_prim.ctxt ? 1 : 0];
}

void GS::snapshotVRAM()
{
    std::lock_guard<std::recursive_mutex> stateLock(m_stateMutex);
    if (!m_vram || m_vramSize == 0)
        return;
    std::lock_guard<std::mutex> lock(m_snapshotMutex);
    m_displaySnapshot.resize(m_vramSize);
    std::memcpy(m_displaySnapshot.data(), m_vram, m_vramSize);
}

const uint8_t *GS::lockDisplaySnapshot(uint32_t &outSize)
{
    m_snapshotMutex.lock();
    if (m_displaySnapshot.empty())
    {
        outSize = 0;
        return nullptr;
    }

    outSize = static_cast<uint32_t>(m_displaySnapshot.size());
    return m_displaySnapshot.data();
}

bool GS::getPreferredDisplaySource(GSFrameReg &outSource, uint32_t &outDestFbp) const
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    if (!m_hasPreferredDisplaySource)
    {
        outSource = {};
        outDestFbp = 0u;
        return false;
    }

    outSource = m_preferredDisplaySourceFrame;
    outDestFbp = m_preferredDisplayDestFbp;
    return true;
}

void GS::unlockDisplaySnapshot()
{
    m_snapshotMutex.unlock();
}

uint32_t GS::getLastDisplayBaseBytes() const
{
    return m_lastDisplayBaseBytes;
}

void GS::refreshDisplaySnapshot()
{
    snapshotVRAM();
}

bool GS::copyFrameToHostRgbaUnlocked(const GSFrameReg &frame,
                                     uint32_t width,
                                     uint32_t height,
                                     std::vector<uint8_t> &outPixels,
                                     bool preserveAlpha,
                                     bool useLocalMemoryLayout,
                                     bool frameBaseIsPages,
                                     uint32_t sourceOriginX,
                                     uint32_t sourceOriginY) const
{
    if (!m_vram || m_vramSize == 0u)
    {
        return false;
    }

    outPixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);

    const uint32_t baseBytes = frameBaseIsPages ? (frame.fbp * 8192u) : (frame.fbp * 256u);
    const uint32_t basePtr = frameBaseIsPages ? GSInternal::framePageBaseToBlock(frame.fbp) : frame.fbp;
    const uint32_t fbwBlocks = frame.fbw ? frame.fbw : (kHostFrameWidth / 64u);
    const uint32_t bytesPerPixel = (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S) ? 2u : 4u;
    const uint32_t strideBytes = fbwBlocks * 64u * bytesPerPixel;

    if (frame.psm == GS_PSM_CT32 || frame.psm == GS_PSM_CT24)
    {
        const uint32_t srcPixelBytes = (frame.psm == GS_PSM_CT24) ? 3u : 4u;
        if (useLocalMemoryLayout)
        {
            for (uint32_t y = 0; y < height; ++y)
            {
                uint8_t *dstRow = outPixels.data() + (y * kHostFrameWidth * 4u);
                for (uint32_t x = 0; x < width; ++x)
                {
                    const uint32_t srcX = sourceOriginX + x;
                    const uint32_t srcY = sourceOriginY + y;
                    const uint32_t srcOff = GSPSMCT32::addrPSMCT32(basePtr, fbwBlocks, srcX, srcY);
                    if (srcOff + srcPixelBytes > m_vramSize)
                    {
                        return false;
                    }

                    dstRow[x * 4u + 0u] = m_vram[srcOff + 0u];
                    dstRow[x * 4u + 1u] = m_vram[srcOff + 1u];
                    dstRow[x * 4u + 2u] = m_vram[srcOff + 2u];
                    dstRow[x * 4u + 3u] =
                        (preserveAlpha && frame.psm != GS_PSM_CT24) ? m_vram[srcOff + 3u] : 255u;
                }
            }
            return true;
        }

        for (uint32_t y = 0; y < height; ++y)
        {
            const uint32_t dstOff = y * kHostFrameWidth * 4u;
            uint8_t *dstRow = outPixels.data() + dstOff;
            for (uint32_t x = 0; x < width; ++x)
            {
                const uint32_t srcX = sourceOriginX + x;
                const uint32_t srcY = sourceOriginY + y;
                const uint32_t srcOff = baseBytes + (srcY * strideBytes) + (srcX * srcPixelBytes);
                if (srcOff + srcPixelBytes > m_vramSize)
                {
                    return false;
                }

                dstRow[x * 4u + 0u] = m_vram[srcOff + 0u];
                dstRow[x * 4u + 1u] = m_vram[srcOff + 1u];
                dstRow[x * 4u + 2u] = m_vram[srcOff + 2u];
                dstRow[x * 4u + 3u] =
                    (preserveAlpha && frame.psm != GS_PSM_CT24) ? m_vram[srcOff + 3u] : 255u;
            }
        }
        return true;
    }

    if (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S)
    {
        if (useLocalMemoryLayout)
        {
            for (uint32_t y = 0; y < height; ++y)
            {
                const uint32_t dstOff = y * kHostFrameWidth * 4u;
                uint8_t *dst = outPixels.data() + dstOff;
                for (uint32_t x = 0; x < width; ++x)
                {
                    const uint32_t srcX = sourceOriginX + x;
                    const uint32_t srcY = sourceOriginY + y;
                    const uint32_t srcOff = addrPSMCT16Family(basePtr, fbwBlocks, frame.psm, srcX, srcY);
                    if (srcOff + sizeof(uint16_t) > m_vramSize)
                    {
                        return false;
                    }

                    uint16_t pixel = 0u;
                    std::memcpy(&pixel, m_vram + srcOff, sizeof(pixel));
                    const uint32_t r = pixel & 31u;
                    const uint32_t g = (pixel >> 5) & 31u;
                    const uint32_t b = (pixel >> 10) & 31u;
                    dst[x * 4u + 0u] = static_cast<uint8_t>((r << 3) | (r >> 2));
                    dst[x * 4u + 1u] = static_cast<uint8_t>((g << 3) | (g >> 2));
                    dst[x * 4u + 2u] = static_cast<uint8_t>((b << 3) | (b >> 2));
                    dst[x * 4u + 3u] = preserveAlpha ? ((pixel & 0x8000u) ? 0x80u : 0x00u) : 255u;
                }
            }
            return true;
        }

        for (uint32_t y = 0; y < height; ++y)
        {
            const uint32_t dstOff = y * kHostFrameWidth * 4u;
            uint8_t *dst = outPixels.data() + dstOff;
            for (uint32_t x = 0; x < width; ++x)
            {
                const uint32_t srcX = sourceOriginX + x;
                const uint32_t srcY = sourceOriginY + y;
                const uint32_t srcOff = baseBytes + (srcY * strideBytes) + (srcX * 2u);
                if (srcOff + sizeof(uint16_t) > m_vramSize)
                {
                    return false;
                }

                uint16_t pixel = 0u;
                std::memcpy(&pixel, m_vram + srcOff, sizeof(pixel));
                const uint32_t r = pixel & 31u;
                const uint32_t g = (pixel >> 5) & 31u;
                const uint32_t b = (pixel >> 10) & 31u;
                dst[x * 4u + 0u] = static_cast<uint8_t>((r << 3) | (r >> 2));
                dst[x * 4u + 1u] = static_cast<uint8_t>((g << 3) | (g >> 2));
                dst[x * 4u + 2u] = static_cast<uint8_t>((b << 3) | (b >> 2));
                dst[x * 4u + 3u] = preserveAlpha ? ((pixel & 0x8000u) ? 0x80u : 0x00u) : 255u;
            }
        }
        return true;
    }

    return false;
}

void GS::latchHostPresentationFrame()
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);

    if (!m_privRegs || !m_vram || m_vramSize == 0u)
    {
        m_hostPresentationFrame.clear();
        m_hostPresentationWidth = 0u;
        m_hostPresentationHeight = 0u;
        m_hostPresentationDisplayFbp = 0u;
        m_hostPresentationSourceFbp = 0u;
        m_hostPresentationUsedPreferred = false;
        m_hasHostPresentationFrame = false;
        return;
    }

    const GSPmodeState pmode = decodePmode(m_privRegs->pmode);
    const GSSmode2State smode2 = decodeSMode2(m_privRegs->smode2);
    const bool applyFieldMode = smode2.interlaced && !smode2.frameMode;
    const bool oddField = (ps2_syscalls::GetCurrentVSyncTick() & 1ull) != 0ull;
    const GSFrameReg displayFrame1 = decodeDisplayFrame(m_privRegs->dispfb1);
    const GSFrameReg displayFrame2 = decodeDisplayFrame(m_privRegs->dispfb2);
    const GSDisplayReadOrigin displayOrigin1 = decodeDisplayReadOrigin(m_privRegs->dispfb1);
    const GSDisplayReadOrigin displayOrigin2 = decodeDisplayReadOrigin(m_privRegs->dispfb2);

    uint32_t width1 = 0u;
    uint32_t height1 = 0u;
    uint32_t width2 = 0u;
    uint32_t height2 = 0u;
    decodeDisplaySize(m_privRegs->display1, width1, height1);
    decodeDisplaySize(m_privRegs->display2, width2, height2);

    const bool validCrt1 = pmode.enableCrt1 && hasDisplaySetup(m_privRegs->display1, displayFrame1);
    const bool validCrt2 = pmode.enableCrt2 && hasDisplaySetup(m_privRegs->display2, displayFrame2);

    auto copyDisplaySource = [&](const GSFrameReg &displayFrame,
                                 const GSDisplayReadOrigin &displayOrigin,
                                 uint32_t width,
                                 uint32_t height,
                                 bool allowPreferred,
                                 bool preserveAlpha,
                                 GSFrameReg &selectedFrame,
                                 std::vector<uint8_t> &scratch,
                                 bool &usedPreferred) -> bool
    {
        selectedFrame = displayFrame;
        scratch.clear();
        usedPreferred = false;

        if (allowPreferred &&
            m_hasPreferredDisplaySource &&
            m_preferredDisplayDestFbp == displayFrame.fbp &&
            (m_preferredDisplaySourceFrame.fbw != 0u || m_preferredDisplaySourceFrame.fbp != displayFrame.fbp))
        {
            if (copyFrameToHostRgbaUnlocked(m_preferredDisplaySourceFrame,
                                            width,
                                            height,
                                            scratch,
                                            preserveAlpha,
                                            true,
                                            false,
                                            0u,
                                            0u))
            {
                selectedFrame = m_preferredDisplaySourceFrame;
                usedPreferred = true;
            }
        }

        if (scratch.empty() &&
            !copyFrameToHostRgbaUnlocked(displayFrame,
                                         width,
                                         height,
                                         scratch,
                                         preserveAlpha,
                                         true,
                                         true,
                                         displayOrigin.x,
                                         displayOrigin.y))
        {
            return false;
        }

        if (!usedPreferred && displayFrame.fbp == 0u && countNonBlackPixels(scratch, width, height) == 0u)
        {
            for (int contextIndex = 0; contextIndex < 2; ++contextIndex)
            {
                const GSFrameReg &candidate = m_ctx[contextIndex].frame;
                if (candidate.fbp == selectedFrame.fbp &&
                    candidate.fbw == selectedFrame.fbw &&
                    candidate.psm == selectedFrame.psm)
                {
                    continue;
                }

                std::vector<uint8_t> candidatePixels;
                if (!copyFrameToHostRgbaUnlocked(candidate,
                                                 width,
                                                 height,
                                                 candidatePixels,
                                                 preserveAlpha,
                                                 true,
                                                 true,
                                                 0u,
                                                 0u))
                {
                    continue;
                }

                if (countNonBlackPixels(candidatePixels, width, height) == 0u)
                {
                    continue;
                }

                selectedFrame = candidate;
                scratch.swap(candidatePixels);
                break;
            }
        }

        return true;
    };

    if (!validCrt1 && !validCrt2)
    {
        m_hostPresentationFrame.clear();
        m_hostPresentationWidth = 0u;
        m_hostPresentationHeight = 0u;
        m_hostPresentationDisplayFbp = 0u;
        m_hostPresentationSourceFbp = 0u;
        m_hostPresentationUsedPreferred = false;
        m_hasHostPresentationFrame = false;
        return;
    }

    if (validCrt1 && validCrt2)
    {
        GSFrameReg selectedFrame1{};
        GSFrameReg selectedFrame2{};
        std::vector<uint8_t> rc1;
        std::vector<uint8_t> rc2;
        bool usedPreferred1 = false;
        bool usedPreferred2 = false;

        const bool copiedCrt1 = copyDisplaySource(displayFrame1, displayOrigin1, width1, height1, false, true, selectedFrame1, rc1, usedPreferred1);
        const bool copiedCrt2 = copyDisplaySource(displayFrame2, displayOrigin2, width2, height2, false, true, selectedFrame2, rc2, usedPreferred2);

        if (copiedCrt1 && copiedCrt2)
        {
            const uint32_t width = std::max(width1, width2);
            const uint32_t height = std::max(height1, height2);
            const uint8_t bgR = static_cast<uint8_t>(m_privRegs->bgcolor & 0xFFu);
            const uint8_t bgG = static_cast<uint8_t>((m_privRegs->bgcolor >> 8) & 0xFFu);
            const uint8_t bgB = static_cast<uint8_t>((m_privRegs->bgcolor >> 16) & 0xFFu);
            const uint8_t bgA = pmode.alp;

            std::vector<uint8_t> merged(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
            for (uint32_t y = 0; y < height; ++y)
            {
                uint8_t *dstRow = merged.data() + (y * kHostFrameWidth * 4u);
                for (uint32_t x = 0; x < width; ++x)
                {
                    dstRow[x * 4u + 0u] = bgR;
                    dstRow[x * 4u + 1u] = bgG;
                    dstRow[x * 4u + 2u] = bgB;
                    dstRow[x * 4u + 3u] = bgA;
                }
            }

            if (!pmode.slbg)
            {
                for (uint32_t y = 0; y < height2; ++y)
                {
                    const uint8_t *srcRow = rc2.data() + (y * kHostFrameWidth * 4u);
                    uint8_t *dstRow = merged.data() + (y * kHostFrameWidth * 4u);
                    for (uint32_t x = 0; x < width2; ++x)
                    {
                        dstRow[x * 4u + 0u] = srcRow[x * 4u + 0u];
                        dstRow[x * 4u + 1u] = srcRow[x * 4u + 1u];
                        dstRow[x * 4u + 2u] = srcRow[x * 4u + 2u];
                        dstRow[x * 4u + 3u] = srcRow[x * 4u + 3u];
                    }
                }
            }

            for (uint32_t y = 0; y < height1; ++y)
            {
                const uint8_t *srcRow = rc1.data() + (y * kHostFrameWidth * 4u);
                uint8_t *dstRow = merged.data() + (y * kHostFrameWidth * 4u);
                for (uint32_t x = 0; x < width1; ++x)
                {
                    const uint8_t srcR = srcRow[x * 4u + 0u];
                    const uint8_t srcG = srcRow[x * 4u + 1u];
                    const uint8_t srcB = srcRow[x * 4u + 2u];
                    const uint8_t srcA = srcRow[x * 4u + 3u];
                    const uint8_t dstR = dstRow[x * 4u + 0u];
                    const uint8_t dstG = dstRow[x * 4u + 1u];
                    const uint8_t dstB = dstRow[x * 4u + 2u];
                    const uint8_t dstA = dstRow[x * 4u + 3u];
                    const uint32_t factor = pmode.mmod
                                                ? static_cast<uint32_t>(pmode.alp)
                                                : std::min<uint32_t>(255u, static_cast<uint32_t>(srcA) * 2u);

                    dstRow[x * 4u + 0u] = blendPresentationChannel(srcR, dstR, factor);
                    dstRow[x * 4u + 1u] = blendPresentationChannel(srcG, dstG, factor);
                    dstRow[x * 4u + 2u] = blendPresentationChannel(srcB, dstB, factor);
                    dstRow[x * 4u + 3u] = pmode.amod ? dstA : srcA;
                }
            }

            for (uint32_t y = 0; y < height; ++y)
            {
                uint8_t *row = merged.data() + (y * kHostFrameWidth * 4u);
                for (uint32_t x = 0; x < width; ++x)
                {
                    row[x * 4u + 3u] = 255u;
                }
            }

            if (applyFieldMode)
            {
                applyFieldPresentation(merged, width, height, oddField);
            }

            m_hostPresentationFrame.swap(merged);
            m_hostPresentationWidth = width;
            m_hostPresentationHeight = height;
            m_hostPresentationDisplayFbp = displayFrame1.fbp;
            m_hostPresentationSourceFbp = selectedFrame1.fbp;
            m_hostPresentationUsedPreferred = false;
            m_hasHostPresentationFrame = true;
            return;
        }
    }

    const GSFrameReg &displayFrame = validCrt1 ? displayFrame1 : displayFrame2;
    const uint32_t width = validCrt1 ? width1 : width2;
    const uint32_t height = validCrt1 ? height1 : height2;

    GSFrameReg selectedFrame = displayFrame;
    std::vector<uint8_t> scratch;
    bool usedPreferred = false;
    const GSDisplayReadOrigin &displayOrigin = validCrt1 ? displayOrigin1 : displayOrigin2;
    if (!copyDisplaySource(displayFrame, displayOrigin, width, height, true, false, selectedFrame, scratch, usedPreferred))
    {
        m_hostPresentationFrame.clear();
        m_hostPresentationWidth = 0u;
        m_hostPresentationHeight = 0u;
        m_hostPresentationDisplayFbp = displayFrame.fbp;
        m_hostPresentationSourceFbp = 0u;
        m_hostPresentationUsedPreferred = false;
        m_hasHostPresentationFrame = false;
        return;
    }

    if (applyFieldMode)
    {
        applyFieldPresentation(scratch, width, height, oddField);
    }

    normalizePresentationAlpha(scratch, width, height);

    m_hostPresentationFrame.swap(scratch);
    m_hostPresentationWidth = width;
    m_hostPresentationHeight = height;
    m_hostPresentationDisplayFbp = displayFrame.fbp;
    m_hostPresentationSourceFbp = selectedFrame.fbp;
    m_hostPresentationUsedPreferred = usedPreferred;
    m_hasHostPresentationFrame = true;
}

void GS::debugDumpFieldFramebuffers()
{
    static const char *s_dir = std::getenv("DQ8_VRAM_DUMP");
    if (!s_dir || !*s_dir)
        return;

    static const uint32_t s_every = []() {
        const char *e = std::getenv("DQ8_VRAM_DUMP_EVERY");
        return e ? std::max<uint32_t>(1u, static_cast<uint32_t>(std::strtoul(e, nullptr, 10))) : 60u;
    }();
    static const uint64_t s_afterMs = []() {
        const char *e = std::getenv("DQ8_VRAM_DUMP_AFTER_MS");
        return e ? static_cast<uint64_t>(std::strtoull(e, nullptr, 10)) : 0ull;
    }();
    const uint64_t elapsedMs = dq8ProbeNowMs();

    // FMV-visibility RCA (2026-07-25): a boot only reaches the field opening
    // movie ~10-15 wall minutes in, so a fixed DQ8_VRAM_DUMP_AFTER_MS is a
    // guess. DQ8_VRAM_DUMP_ON_MOVIE=1 instead arms the dumper *only* while a
    // movie is actively streaming tiles into VRAM (see g_dq8MovieUploadMs,
    // stamped by processImageData on 512-wide CT32 IMAGE uploads), which is
    // exactly the window of interest and costs nothing outside it.
    static const bool s_onMovie = []() {
        const char *e = std::getenv("DQ8_VRAM_DUMP_ON_MOVIE");
        return e && *e && *e != '0';
    }();
    if (s_onMovie)
    {
        const uint64_t lastUpload = g_dq8MovieUploadMs.load(std::memory_order_relaxed);
        if (lastUpload == 0ull || elapsedMs > lastUpload + 1000ull)
            return;
    }
    else if (elapsedMs < s_afterMs)
    {
        return;
    }

    static std::atomic<uint64_t> s_present{0};
    const uint64_t idx = s_present.fetch_add(1, std::memory_order_relaxed);
    if ((idx % s_every) != 0u)
        return;

    static std::atomic<uint32_t> s_sets{0};
    const uint32_t setNo = s_sets.fetch_add(1, std::memory_order_relaxed);
    if (setNo > 800u)
        return;

    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);

    // 0x150 == BITBLTBUF dbp 0x2a00 (the TITLE-mode movie surface) and
    // 0x170 == dbp 0x2e00 (the FIELD/TOPO movie surface): FRAME.FBP is in
    // 8 KiB units, BITBLTBUF.DBP in 256 B units, so dbp = fbp * 32. Dumping
    // both settles "is the decoded frame colourful in VRAM but never drawn"
    // versus "the decode/upload itself is black".
    const uint32_t fbps[] = {0x0u, 0x70u, 0x150u, 0x160u, 0x170u, 0x17cu, 0x1c0u, 0x1dcu};
    for (uint32_t fbp : fbps)
    {
        GSFrameReg fr{};
        fr.fbp = fbp;
        fr.fbw = 8u;
        fr.psm = GS_PSM_CT32;
        fr.fbmsk = 0u;

        std::vector<uint8_t> px;
        if (!copyFrameToHostRgbaUnlocked(fr, kHostFrameWidth, kHostFrameHeight, px, false, true, true, 0u, 0u))
            continue;

        uint32_t nonBlack = 0u;
        for (size_t i = 0; i + 3u < px.size(); i += 4u)
        {
            if (px[i] != 0u || px[i + 1u] != 0u || px[i + 2u] != 0u)
                ++nonBlack;
        }

        char path[512];
        std::snprintf(path, sizeof(path), "%s/set%03u_fbp%04x.ppm", s_dir, setNo, fbp);
        if (FILE *f = std::fopen(path, "wb"))
        {
            std::fprintf(f, "P6\n%u %u\n255\n", kHostFrameWidth, kHostFrameHeight);
            std::vector<uint8_t> rgb(static_cast<size_t>(kHostFrameWidth) * kHostFrameHeight * 3u);
            for (size_t p = 0, q = 0; p + 3u < px.size(); p += 4u, q += 3u)
            {
                rgb[q + 0u] = px[p + 0u];
                rgb[q + 1u] = px[p + 1u];
                rgb[q + 2u] = px[p + 2u];
            }
            std::fwrite(rgb.data(), 1u, rgb.size(), f);
            std::fclose(f);
        }

        std::cout << "[gs:vramdump] set=" << setNo
                  << " t=" << elapsedMs << "ms"
                  << " fbp=0x" << std::hex << fbp << std::dec
                  << " nonblack=" << nonBlack << "/" << (kHostFrameWidth * kHostFrameHeight)
                  << std::endl;
    }
}

bool GS::copyLatchedHostPresentationFrame(std::vector<uint8_t> &outPixels,
                                          uint32_t &outWidth,
                                          uint32_t &outHeight,
                                          uint32_t *outDisplayFbp,
                                          uint32_t *outSourceFbp,
                                          bool *outUsedPreferred) const
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    if (!m_hasHostPresentationFrame || m_hostPresentationFrame.empty())
    {
        outPixels.clear();
        outWidth = 0u;
        outHeight = 0u;
        if (outDisplayFbp)
            *outDisplayFbp = 0u;
        if (outSourceFbp)
            *outSourceFbp = 0u;
        if (outUsedPreferred)
            *outUsedPreferred = false;
        return false;
    }

    outWidth = m_hostPresentationWidth;
    outHeight = m_hostPresentationHeight;
    if (outDisplayFbp)
        *outDisplayFbp = m_hostPresentationDisplayFbp;
    if (outSourceFbp)
        *outSourceFbp = m_hostPresentationSourceFbp;
    if (outUsedPreferred)
        *outUsedPreferred = m_hostPresentationUsedPreferred;

    const size_t packedRowBytes = static_cast<size_t>(outWidth) * 4u;
    outPixels.assign(packedRowBytes * static_cast<size_t>(outHeight), 0u);
    if (outWidth != 0u && outHeight != 0u)
    {
        const size_t sourceRowBytes = static_cast<size_t>(kHostFrameWidth) * 4u;
        for (uint32_t y = 0; y < outHeight; ++y)
        {
            const size_t srcOffset = static_cast<size_t>(y) * sourceRowBytes;
            const size_t dstOffset = static_cast<size_t>(y) * packedRowBytes;
            if (srcOffset + packedRowBytes > m_hostPresentationFrame.size() ||
                dstOffset + packedRowBytes > outPixels.size())
            {
                outPixels.clear();
                outWidth = 0u;
                outHeight = 0u;
                if (outDisplayFbp)
                    *outDisplayFbp = 0u;
                if (outSourceFbp)
                    *outSourceFbp = 0u;
                if (outUsedPreferred)
                    *outUsedPreferred = false;
                return false;
            }

            std::memcpy(outPixels.data() + dstOffset,
                        m_hostPresentationFrame.data() + srcOffset,
                        packedRowBytes);
        }
    }
    return true;
}

namespace
{
    // Raw GIF-packet capture, so our emitted register/primitive stream can be
    // diffed against a PCSX2 .gs dump of the same scene using the same walker
    // (dq8-fmv-work/oracle/run3/gifwalk.py). Record framing deliberately
    // matches the transfer record inside a .gs dump:
    //     u8 kind(0) | u8 path | u32 len | len bytes
    // so the reader is a six-line loop.
    //
    // Gated by PS2X_GIFDUMP=<path>; PS2X_GIFDUMP_AFTER_MS delays the start
    // (the title screen alone is hundreds of MB) and PS2X_GIFDUMP_MAX_MB caps
    // the file (default 256). VALUE-tested env reads, never getenv()!=nullptr.
    std::FILE *gifDumpFile()
    {
        static std::FILE *f = []() -> std::FILE *
        {
            const char *p = std::getenv("PS2X_GIFDUMP");
            if (!p || p[0] == '\0')
                return nullptr;
            return std::fopen(p, "wb");
        }();
        return f;
    }

    uint64_t gifDumpAfterMs()
    {
        static const uint64_t v = []() -> uint64_t
        {
            const char *e = std::getenv("PS2X_GIFDUMP_AFTER_MS");
            return (e && e[0]) ? std::strtoull(e, nullptr, 10) : 0ull;
        }();
        return v;
    }

    uint64_t gifDumpMaxBytes()
    {
        static const uint64_t v = []() -> uint64_t
        {
            const char *e = std::getenv("PS2X_GIFDUMP_MAX_MB");
            const uint64_t mb = (e && e[0]) ? std::strtoull(e, nullptr, 10) : 256ull;
            return mb * 1024ull * 1024ull;
        }();
        return v;
    }

    std::atomic<uint64_t> s_gifDumpWritten{0};

    // Optional external arming: PS2X_GIFDUMP_TRIGGER=<path> makes the capture
    // record only while that file exists, re-checked at most 4x/second. A
    // wall-clock window is useless here because the boot-to-cutscene driver
    // takes 30-40 minutes with high variance; with a trigger file the window
    // can be opened the moment the frames show the scene under investigation.
    bool gifDumpArmed()
    {
        static const char *trig = []() -> const char *
        {
            const char *p = std::getenv("PS2X_GIFDUMP_TRIGGER");
            return (p && p[0]) ? p : nullptr;
        }();
        if (!trig)
            return true;
        static std::atomic<uint64_t> s_lastCheckMs{0};
        static std::atomic<bool> s_armed{false};
        const uint64_t now = dq8ProbeNowMs();
        uint64_t prev = s_lastCheckMs.load(std::memory_order_relaxed);
        if (now >= prev + 250ull &&
            s_lastCheckMs.compare_exchange_strong(prev, now, std::memory_order_relaxed))
        {
            std::FILE *t = std::fopen(trig, "rb");
            s_armed.store(t != nullptr, std::memory_order_relaxed);
            if (t)
                std::fclose(t);
        }
        return s_armed.load(std::memory_order_relaxed);
    }

    void gifDumpRecord(const uint8_t *data, uint32_t sizeBytes)
    {
        std::FILE *f = gifDumpFile();
        if (!f)
            return;
        if (dq8ProbeNowMs() < gifDumpAfterMs())
            return;
        if (!gifDumpArmed())
            return;
        if (s_gifDumpWritten.load(std::memory_order_relaxed) >= gifDumpMaxBytes())
            return;
        static std::mutex s_mtx;
        std::lock_guard<std::mutex> lk(s_mtx);
        const uint8_t kind = 0u;
        const uint8_t path = 3u;
        std::fwrite(&kind, 1, 1, f);
        std::fwrite(&path, 1, 1, f);
        std::fwrite(&sizeBytes, 4, 1, f);
        std::fwrite(data, 1, sizeBytes, f);
        s_gifDumpWritten.fetch_add(sizeBytes + 6u, std::memory_order_relaxed);
    }
}

void GS::processGIFPacket(const uint8_t *data, uint32_t sizeBytes)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    if (!data || sizeBytes < 16 || !m_vram)
        return;

    gifDumpRecord(data, sizeBytes);

    // M-T1 finding (2026-07-18, investigated, no code change needed here):
    // DQ8's texture/CLUT upload builder sends a tiny "declare" DMA burst
    // carrying ONLY the IMAGE GIFtag (EOP=1, 0 payload bytes in that same
    // transfer), then LATER a second, entirely separate, SELF-CONTAINED
    // burst carrying the identical tag followed by its FULL declared
    // payload (verified byte-for-byte via a temporary diagnostic:
    // incomingSize == owed-from-burst-1 + 16, and burst 2's own leading
    // qword == burst 1's tag, exactly, every single time). Burst 1 is a
    // harmless no-op (its 0-byte payload is correctly discarded by the
    // ordinary per-call tag parse below); burst 2 is a complete, ordinary
    // GIFtag+IMAGE packet that this SAME per-call parser already handles
    // correctly on its own -- exactly like the working font/CLUT upload
    // path (PS2_PROJECT_STATE.md §3.25). A cross-call "continuation" fix
    // was tried and reverted: treating burst 1's shortfall as a promise
    // that the NEXT buffer is pure payload with no tag of its own was
    // wrong and made things WORSE, consuming burst 2's own tag bytes as if
    // they were pixels. Investigation continues in ps2_gif_arbiter.cpp
    // (GifArbiter::drain()/submit()) and PS2_PROJECT_STATE.md's M-T1 entry.

    uint32_t offset = 0;
    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t packetIndex = s_debugGifPacketCount.fetch_add(1, std::memory_order_relaxed);
        if (packetIndex < 48u)
        {
            const uint64_t tagLo = loadLE64(data + offset);
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;
            RUNTIME_LOG("[gs:gif] idx=" << packetIndex
                                        << " size=" << sizeBytes
                                        << " nloop=" << nloop
                                        << " flg=" << static_cast<uint32_t>(flg)
                                        << " nreg=" << nreg
                                        << " ctx0fbp=" << m_ctx[0].frame.fbp
                                        << " ctx1fbp=" << m_ctx[1].frame.fbp
                                        << std::endl);
        }
    });

    {
        const uint64_t tagLo = loadLE64(data + offset);
        const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3);
        if (flg == GIF_FMT_PACKED)
        {
            m_hwregX = 0;
            m_hwregY = 0;
        }
    }

    while (offset + 16 <= sizeBytes)
    {
        uint64_t tagLo = loadLE64(data + offset);
        uint64_t tagHi = loadLE64(data + offset + 8);
        offset += 16;

        m_curQ = 1.0f;

        uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFF);
        uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3);
        uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xF);
        if (nreg == 0)
            nreg = 16;

        // Bounded permanent raw-GIFtag dump: the first N tags actually parsed
        // by this loop, with raw qwords. Diagnoses stream misalignment (e.g.
        // a PACKED A+D run consuming an IMAGE tag as data), which is
        // invisible from the per-register logs alone.
        {
            static std::atomic<uint64_t> s_tagCount{0};
            static const uint32_t kMaxGiftagDumpLogs = ps2DiagEnvLimit("PS2X_GIFTAG_DUMP_MAX_LOGS", 192u);
            static std::atomic<bool> s_giftagDumpTruncated{false};
            const uint64_t tagIdx = s_tagCount.fetch_add(1, std::memory_order_relaxed);
            if (ps2DiagLogBudget(std::cout,
                                 "[gs:giftag]",
                                 "PS2X_GIFTAG_DUMP_MAX_LOGS",
                                 kMaxGiftagDumpLogs,
                                 static_cast<uint32_t>(tagIdx),
                                 s_giftagDumpTruncated))
            {
                std::cout << "[gs:giftag] #" << tagIdx
                          << " off=" << (offset - 16u)
                          << " size=" << sizeBytes
                          << std::hex
                          << " lo=0x" << tagLo
                          << " hi=0x" << tagHi
                          << std::dec
                          << " nloop=" << nloop
                          << " eop=" << ((tagLo >> 15) & 1u)
                          << " flg=" << static_cast<uint32_t>(flg)
                          << " nreg=" << nreg
                          << std::endl;
            }
        }

        bool pre = ((tagLo >> 46) & 1) != 0;
        if (pre)
        {
            writeRegister(GS_REG_PRIM, (tagLo >> 47) & 0x7FF);
        }

        uint8_t regs[16];
        for (uint32_t i = 0; i < nreg; ++i)
            regs[i] = static_cast<uint8_t>((tagHi >> (i * 4)) & 0xF);

        if (flg == GIF_FMT_PACKED && colourProbeOn())
        {
            bool hasST = false;
            for (uint32_t i = 0; i < nreg; ++i)
            {
                if (regs[i] == 0x02u) // GS_REG_ST packed descriptor
                {
                    hasST = true;
                    break;
                }
            }
            s_colourProbeTagHasST = hasST;
        }

        if (flg == GIF_FMT_PACKED)
        {
            for (uint32_t loop = 0; loop < nloop; ++loop)
            {
                for (uint32_t r = 0; r < nreg; ++r)
                {
                    if (offset + 16 > sizeBytes)
                        return;
                    uint64_t lo = loadLE64(data + offset);
                    uint64_t hi = loadLE64(data + offset + 8);
                    offset += 16;
                    writeRegisterPacked(regs[r], lo, hi);
                }
            }
        }
        else if (flg == GIF_FMT_REGLIST)
        {
            for (uint32_t loop = 0; loop < nloop; ++loop)
            {
                for (uint32_t r = 0; r < nreg; ++r)
                {
                    if (offset + 8 > sizeBytes)
                        return;
                    writeRegister(regs[r], loadLE64(data + offset));
                    offset += 8;
                }
            }
            if ((nloop * nreg) & 1)
                offset += 8;
        }
        else if (flg == GIF_FMT_IMAGE)
        {
            uint32_t imageBytes = nloop * 16;
            if (offset + imageBytes > sizeBytes)
                imageBytes = sizeBytes - offset;

            // M-T1 diag (2026-07-18, generic, bounded, temporary): observe
            // declared-vs-available bytes for every real (dbw>=2, i.e.
            // texture/CLUT-shaped, not a 1-qword register-set BITBLTBUF)
            // IMAGE tag, to check whether LARGE texture-plane transfers
            // (dbw=4, ~229KB) follow the same "tiny declare-only burst,
            // then a later self-contained tag+full-payload burst" shape
            // already confirmed for the small CLUT transfers (16x16, 1KB),
            // or whether large transfers are genuinely split differently
            // (needing real cross-call continuation after all).
            {
                const uint32_t declaredBytes = nloop * 16u;
                const uint32_t availableBytes = (sizeBytes > offset) ? (sizeBytes - offset) : 0u;
                if (m_bitbltbuf.dbw >= 4u && availableBytes < declaredBytes)
                {
                    static std::atomic<uint64_t> s_shapeCount{0};
                    const uint64_t sn = s_shapeCount.fetch_add(1, std::memory_order_relaxed);
                    if (sn < 300u)
                    {
                        std::cout << "[gs:image-shape] #" << sn
                                  << std::hex << " dbp=0x" << m_bitbltbuf.dbp << std::dec
                                  << " dbw=" << static_cast<uint32_t>(m_bitbltbuf.dbw)
                                  << " rr=" << m_trxreg.rrw << "x" << m_trxreg.rrh
                                  << " nloop=" << nloop
                                  << " declaredBytes=" << declaredBytes
                                  << " packetSize=" << sizeBytes
                                  << " availableBytes=" << availableBytes
                                  << std::endl;
                    }
                }
            }

            if (nloop == 0u && m_trxreg.rrw != 0u && m_trxreg.rrh != 0u)
            {
                static std::atomic<uint64_t> s_zeroImageCount{0};
                const uint64_t zn = s_zeroImageCount.fetch_add(1, std::memory_order_relaxed);
                if (zn < 200u)
                {
                    const uint32_t availableBytes = (sizeBytes > offset) ? (sizeBytes - offset) : 0u;
                    std::cout << "[gs:zero-image] #" << zn << " kind=zero-nloop"
                              << std::hex
                              << " dbp=0x" << m_bitbltbuf.dbp
                              << " dpsm=0x" << static_cast<uint32_t>(m_bitbltbuf.dpsm)
                              << std::dec
                              << " dbw=" << static_cast<uint32_t>(m_bitbltbuf.dbw)
                              << " rr=" << m_trxreg.rrw << "x" << m_trxreg.rrh
                              << " tagOff=" << (offset - 16u)
                              << " packetSize=" << sizeBytes
                              << " availableBytes=" << availableBytes
                              << std::hex << " tagLo=0x" << tagLo << " tagHi=0x" << tagHi << std::dec
                              << std::endl;
                }
            }

            processImageData(data + offset, imageBytes);
            offset += imageBytes;
        }
    }
}

void GS::writeRegisterPacked(uint8_t regDesc, uint64_t lo, uint64_t hi)
{
    // Bounded permanent histogram of PACKED-mode register descriptors: shows
    // whether draw state (PRIM/RGBAQ/ST/UV/A+D) actually flows alongside the
    // vertex kicks. One line per 200k packed writes.
    {
        static std::atomic<uint64_t> s_packedCounts[16];
        static std::atomic<uint64_t> s_packedTotal{0};
        s_packedCounts[regDesc & 0xF].fetch_add(1, std::memory_order_relaxed);
        const uint64_t total = s_packedTotal.fetch_add(1, std::memory_order_relaxed);
        if ((total % 200000u) == 0u)
        {
            std::cout << "[gs:packed] total=" << total;
            for (int i = 0; i < 16; ++i)
            {
                const uint64_t c = s_packedCounts[i].load(std::memory_order_relaxed);
                if (c != 0u)
                {
                    std::cout << " r" << std::hex << i << std::dec << "=" << c;
                }
            }
            std::cout << std::endl;
        }
    }

    switch (regDesc)
    {
    case 0x00:
        writeRegister(GS_REG_PRIM, lo & 0x7FF);
        break;
    case 0x01:
        m_curR = static_cast<uint8_t>(lo & 0xFF);
        m_curG = static_cast<uint8_t>((lo >> 32) & 0xFF);
        m_curB = static_cast<uint8_t>(hi & 0xFF);
        m_curA = static_cast<uint8_t>((hi >> 32) & 0xFF);
        if (colourProbeOn())
            rgbaqHistNote(s_colourProbeTagHasST, m_curR, m_curG, m_curB, m_curA);
        break;
    case 0x02:
    {
        uint32_t sBits = static_cast<uint32_t>(lo & 0xFFFFFFFF);
        uint32_t tBits = static_cast<uint32_t>((lo >> 32) & 0xFFFFFFFF);
        uint32_t qBits = static_cast<uint32_t>(hi & 0xFFFFFFFF);
        std::memcpy(&m_curS, &sBits, 4);
        std::memcpy(&m_curT, &tBits, 4);
        std::memcpy(&m_curQ, &qBits, 4);
        if (m_curQ == 0.0f)
            m_curQ = 1.0f;
        break;
    }
    case 0x03:
        m_curU = static_cast<uint16_t>(lo & 0xFFFFu);
        m_curV = static_cast<uint16_t>((lo >> 32) & 0xFFFFu);
        break;
    case 0x04:
    {
        uint16_t x = static_cast<uint16_t>(lo & 0xFFFF);
        uint16_t y = static_cast<uint16_t>((lo >> 32) & 0xFFFF);
        uint32_t z = static_cast<uint32_t>((hi >> 4) & 0xFFFFFF);
        uint8_t f = static_cast<uint8_t>((hi >> 36) & 0xFF);
        bool adk = ((hi >> 47) & 1) != 0;
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyzf] idx=" << debugIndex
                                                    << " x=" << x
                                                    << " y=" << y
                                                    << " z=0x" << std::hex << z
                                                    << std::dec
                                                    << " fog=" << static_cast<uint32_t>(f)
                                                    << " kick=" << static_cast<uint32_t>(!adk ? 1u : 0u)
                                                    << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                    << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(x) / 16.0f;
        vtx.y = static_cast<float>(y) / 16.0f;
        vtx.z = static_cast<float>(z);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = f;
        gsVtxStatNote(GsVtxPop::kPacked,
                      vtx.x - activeContext().xyoffset.ofx / 16.0f,
                      vtx.y - activeContext().xyoffset.ofy / 16.0f,
                      vtx.z, activeContext().frame.fbp);
        if (colourProbeOn())
            drawStateSampleNote(s_colourProbeTagHasST, m_prim, activeContext().tex0);
        vertexKick(!adk);
        break;
    }
    case 0x05:
    {
        uint16_t x = static_cast<uint16_t>(lo & 0xFFFF);
        uint16_t y = static_cast<uint16_t>((lo >> 32) & 0xFFFF);
        uint32_t z = static_cast<uint32_t>(hi & 0xFFFFFFFF);
        bool adk = ((hi >> 47) & 1) != 0;
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyz] idx=" << debugIndex
                                                   << " x=" << x
                                                   << " y=" << y
                                                   << " z=0x" << std::hex << z
                                                   << std::dec
                                                   << " kick=" << static_cast<uint32_t>(!adk ? 1u : 0u)
                                                   << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                   << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(x) / 16.0f;
        vtx.y = static_cast<float>(y) / 16.0f;
        vtx.z = static_cast<float>(z);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = m_curFog;
        gsVtxStatNote(GsVtxPop::kPacked,
                      vtx.x - activeContext().xyoffset.ofx / 16.0f,
                      vtx.y - activeContext().xyoffset.ofy / 16.0f,
                      vtx.z, activeContext().frame.fbp);
        if (colourProbeOn())
            drawStateSampleNote(s_colourProbeTagHasST, m_prim, activeContext().tex0);
        vertexKick(!adk);
        break;
    }
    case 0x0A:
        m_curFog = static_cast<uint8_t>((hi >> 36) & 0xFF);
        break;
    case 0x0C:
    {
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyzf3] idx=" << debugIndex
                                                     << " x=" << static_cast<uint32_t>(lo & 0xFFFFu)
                                                     << " y=" << static_cast<uint32_t>((lo >> 32) & 0xFFFFu)
                                                     << " kick=0"
                                                     << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                     << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(lo & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((lo >> 32) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<float>((hi >> 4) & 0xFFFFFF);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = static_cast<uint8_t>((hi >> 36) & 0xFF);
        gsVtxStatNote(GsVtxPop::kPacked,
                      vtx.x - activeContext().xyoffset.ofx / 16.0f,
                      vtx.y - activeContext().xyoffset.ofy / 16.0f,
                      vtx.z, activeContext().frame.fbp);
        if (colourProbeOn())
            drawStateSampleNote(s_colourProbeTagHasST, m_prim, activeContext().tex0);
        vertexKick(false);
        break;
    }
    case 0x0D:
    {
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyz3] idx=" << debugIndex
                                                    << " x=" << static_cast<uint32_t>(lo & 0xFFFFu)
                                                    << " y=" << static_cast<uint32_t>((lo >> 32) & 0xFFFFu)
                                                    << " kick=0"
                                                    << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                    << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(lo & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((lo >> 32) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<float>(hi & 0xFFFFFFFF);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = m_curFog;
        gsVtxStatNote(GsVtxPop::kPacked,
                      vtx.x - activeContext().xyoffset.ofx / 16.0f,
                      vtx.y - activeContext().xyoffset.ofy / 16.0f,
                      vtx.z, activeContext().frame.fbp);
        if (colourProbeOn())
            drawStateSampleNote(s_colourProbeTagHasST, m_prim, activeContext().tex0);
        vertexKick(false);
        break;
    }
    case 0x0E:
    {
        uint8_t addr = static_cast<uint8_t>(hi & 0xFF);

        // M0 §3.24 diagnostic: bounded dump of every A+D (PACKED descriptor
        // 0xE) register write — addr byte + full 64-bit data qword — so a
        // single instrumented boot can show exactly which GS registers the
        // present/composite packet actually programs (and whether any addr
        // byte falls outside GS::writeRegister's known case list). First 32
        // unthrottled, then every 1000th, so a whole boot's A+D traffic is
        // sampled without flooding the log.
        {
            static std::atomic<uint64_t> s_adCount{0};
            const uint64_t n = s_adCount.fetch_add(1, std::memory_order_relaxed);
            if (n < 32u || (n % 1000u) == 0u)
            {
                const bool known =
                    addr == GS_REG_PRIM || addr == GS_REG_RGBAQ || addr == GS_REG_ST ||
                    addr == GS_REG_UV || addr == GS_REG_XYZF2 || addr == GS_REG_XYZ2 ||
                    addr == GS_REG_TEX0_1 || addr == GS_REG_TEX0_2 || addr == GS_REG_CLAMP_1 ||
                    addr == GS_REG_CLAMP_2 || addr == GS_REG_FOG || addr == GS_REG_XYZF3 ||
                    addr == GS_REG_XYZ3 || addr == GS_REG_AD || addr == GS_REG_TEX1_1 ||
                    addr == GS_REG_TEX1_2 || addr == GS_REG_TEX2_1 || addr == GS_REG_TEX2_2 ||
                    addr == GS_REG_XYOFFSET_1 || addr == GS_REG_XYOFFSET_2 ||
                    addr == GS_REG_PRMODECONT || addr == GS_REG_PRMODE ||
                    addr == GS_REG_TEXCLUT || addr == GS_REG_SCANMSK ||
                    addr == GS_REG_MIPTBP1_1 || addr == GS_REG_MIPTBP1_2 ||
                    addr == GS_REG_MIPTBP2_1 || addr == GS_REG_MIPTBP2_2 ||
                    addr == GS_REG_TEXA || addr == GS_REG_FOGCOL || addr == GS_REG_TEXFLUSH ||
                    addr == GS_REG_SCISSOR_1 || addr == GS_REG_SCISSOR_2 ||
                    addr == GS_REG_ALPHA_1 || addr == GS_REG_ALPHA_2 || addr == GS_REG_DIMX ||
                    addr == GS_REG_DTHE || addr == GS_REG_COLCLAMP || addr == GS_REG_TEST_1 ||
                    addr == GS_REG_TEST_2 || addr == GS_REG_PABE || addr == GS_REG_FBA_1 ||
                    addr == GS_REG_FBA_2 || addr == GS_REG_FRAME_1 || addr == GS_REG_FRAME_2 ||
                    addr == GS_REG_ZBUF_1 || addr == GS_REG_ZBUF_2 || addr == GS_REG_BITBLTBUF ||
                    addr == GS_REG_TRXPOS || addr == GS_REG_TRXREG || addr == GS_REG_TRXDIR ||
                    addr == GS_REG_HWREG || addr == GS_REG_SIGNAL || addr == GS_REG_FINISH ||
                    addr == GS_REG_LABEL || addr == 0x59 || addr == 0x5a || addr == 0x5b ||
                    addr == 0x5c || addr == 0x5f;
                std::cout << "[gs:ad] #" << n
                          << " addr=0x" << std::hex << static_cast<uint32_t>(addr)
                          << " data=0x" << lo
                          << std::dec
                          << (known ? "" : " UNKNOWN")
                          << std::endl;
            }
        }

        writeRegister(addr, lo);
        break;
    }
    case 0x0F:
        break;
    default:
        writeRegister(regDesc, lo);
        break;
    }
}

void GS::writeRegister(uint8_t regAddr, uint64_t value)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    const bool interestingReg =
        regAddr == GS_REG_PRIM ||
        regAddr == GS_REG_RGBAQ ||
        regAddr == GS_REG_ST ||
        regAddr == GS_REG_UV ||
        regAddr == GS_REG_XYZ2 ||
        regAddr == GS_REG_XYZ3 ||
        regAddr == GS_REG_XYZF2 ||
        regAddr == GS_REG_XYZF3 ||
        regAddr == GS_REG_TEX0_1 ||
        regAddr == GS_REG_TEX0_2 ||
        regAddr == GS_REG_TEX2_1 ||
        regAddr == GS_REG_TEX2_2 ||
        regAddr == GS_REG_TEXCLUT ||
        regAddr == GS_REG_TEXA ||
        regAddr == GS_REG_XYOFFSET_1 ||
        regAddr == GS_REG_XYOFFSET_2 ||
        regAddr == GS_REG_SCISSOR_1 ||
        regAddr == GS_REG_SCISSOR_2 ||
        regAddr == GS_REG_FRAME_1 ||
        regAddr == GS_REG_FRAME_2 ||
        regAddr == GS_REG_ALPHA_1 ||
        regAddr == GS_REG_ALPHA_2 ||
        regAddr == GS_REG_TEST_1 ||
        regAddr == GS_REG_TEST_2 ||
        regAddr == GS_REG_BITBLTBUF ||
        regAddr == GS_REG_TRXPOS ||
        regAddr == GS_REG_TRXREG ||
        regAddr == GS_REG_TRXDIR;

    PS2_IF_AGRESSIVE_LOGS({
        if (interestingReg)
        {
            const uint32_t debugIndex = s_debugGsRegisterCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 128u)
            {
                RUNTIME_LOG("[gs:reg] idx=" << debugIndex
                                            << " reg=0x" << std::hex << static_cast<uint32_t>(regAddr)
                                            << " value=0x" << value
                                            << std::dec
                                            << std::endl);
            }
        }
    });

    const bool isCopyRelevantReg =
        regAddr == GS_REG_PRIM ||
        regAddr == GS_REG_TEX0_2 ||
        regAddr == GS_REG_TEX1_2 ||
        regAddr == GS_REG_ALPHA_2 ||
        regAddr == GS_REG_TEST_2 ||
        regAddr == GS_REG_PABE ||
        regAddr == GS_REG_FRAME_2 ||
        regAddr == GS_REG_XYOFFSET_2 ||
        regAddr == GS_REG_SCISSOR_2;
    PS2_IF_AGRESSIVE_LOGS({
        if (isCopyRelevantReg &&
            s_debugCopyRegCount.fetch_add(1u, std::memory_order_relaxed) < 64u)
        {
            RUNTIME_LOG("[gs:copy-reg] reg=0x"
                        << std::hex << static_cast<uint32_t>(regAddr)
                        << " value=0x" << value
                        << std::dec
                        << " primCtxt=" << static_cast<uint32_t>(m_prim.ctxt)
                        << " ctx0fbp=" << m_ctx[0].frame.fbp
                        << " ctx1fbp=" << m_ctx[1].frame.fbp
                        << std::endl);
        }
    });

    switch (regAddr)
    {
    case GS_REG_PRIM:
    {
        // GS Users Manual semantics for PRMODECONT (AC): the primitive TYPE
        // field (bits 0-2) always comes from this PRIM write, but the
        // attribute bits (IIP/TME/FGE/ABE/AA1/FST/CTXT/FIX, bits 3-10) are
        // only sourced from PRIM when PRMODECONT=1 ("AC"). When
        // PRMODECONT=0, those attributes are supposed to be held fixed from
        // the last PRMODE write (see GS_REG_PRMODE below) — a PRIM write in
        // that mode only changes which primitive gets drawn next, not how.
        // Previously this handler unconditionally overwrote every attribute
        // bit from the PRIM value regardless of PRMODECONT, so any
        // PRMODECONT=0 caller that sets state via PRMODE once (e.g. TME=1
        // for a textured composite/present sprite) and then issues plain
        // PRIM writes to select primitive type per draw had its attributes
        // silently clobbered back to whatever bits happened to be packed
        // into the PRIM value (frequently 0) on every single draw.
        m_prim.type = static_cast<GSPrimType>(value & 0x7);
        if (m_prmodecont)
        {
            m_prim.iip = ((value >> 3) & 1) != 0;
            m_prim.tme = ((value >> 4) & 1) != 0;
            m_prim.fge = ((value >> 5) & 1) != 0;
            m_prim.abe = ((value >> 6) & 1) != 0;
            m_prim.aa1 = ((value >> 7) & 1) != 0;
            m_prim.fst = ((value >> 8) & 1) != 0;
            m_prim.ctxt = ((value >> 9) & 1) != 0;
            m_prim.fix = ((value >> 10) & 1) != 0;
        }
        m_vtxCount = 0;
        m_vtxIndex = 0;
        break;
    }
    case GS_REG_RGBAQ:
    {
        m_curR = static_cast<uint8_t>(value & 0xFF);
        m_curG = static_cast<uint8_t>((value >> 8) & 0xFF);
        m_curB = static_cast<uint8_t>((value >> 16) & 0xFF);
        m_curA = static_cast<uint8_t>((value >> 24) & 0xFF);
        uint32_t qBits = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);
        std::memcpy(&m_curQ, &qBits, 4);
        if (m_curQ == 0.0f)
            m_curQ = 1.0f;
        break;
    }
    case GS_REG_ST:
    {
        uint32_t sBits = static_cast<uint32_t>(value & 0xFFFFFFFF);
        uint32_t tBits = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);
        std::memcpy(&m_curS, &sBits, 4);
        std::memcpy(&m_curT, &tBits, 4);
        break;
    }
    case GS_REG_UV:
    {
        m_curU = static_cast<uint16_t>(value & 0xFFFFu);
        m_curV = static_cast<uint16_t>((value >> 16) & 0xFFFFu);
        break;
    }
    case GS_REG_XYZF2:
    case GS_REG_XYZF3:
    {
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(value & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((value >> 16) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<float>((value >> 32) & 0xFFFFFF);
        vtx.fog = static_cast<uint8_t>((value >> 56) & 0xFF);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        gsVtxStatNote(GsVtxPop::kAdRegList,
                      vtx.x - activeContext().xyoffset.ofx / 16.0f,
                      vtx.y - activeContext().xyoffset.ofy / 16.0f,
                      vtx.z, activeContext().frame.fbp);
        vertexKick(regAddr == GS_REG_XYZF2);
        break;
    }
    case GS_REG_XYZ2:
    case GS_REG_XYZ3:
    {
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(value & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((value >> 16) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<float>((value >> 32) & 0xFFFFFFFF);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = m_curFog;
        gsVtxStatNote(GsVtxPop::kAdRegList,
                      vtx.x - activeContext().xyoffset.ofx / 16.0f,
                      vtx.y - activeContext().xyoffset.ofy / 16.0f,
                      vtx.z, activeContext().frame.fbp);
        vertexKick(regAddr == GS_REG_XYZ2);
        break;
    }
    case GS_REG_TEX0_1:
    case GS_REG_TEX0_2:
    {
        int ci = (regAddr == GS_REG_TEX0_2) ? 1 : 0;
        auto &t = m_ctx[ci].tex0;
        t.tbp0 = static_cast<uint32_t>(value & 0x3FFF);
        t.tbw = static_cast<uint8_t>((value >> 14) & 0x3F);
        t.psm = static_cast<uint8_t>((value >> 20) & 0x3F);
        t.tw = static_cast<uint8_t>((value >> 26) & 0xF);
        t.th = static_cast<uint8_t>((value >> 30) & 0xF);
        t.tcc = static_cast<uint8_t>((value >> 34) & 0x1);
        t.tfx = static_cast<uint8_t>((value >> 35) & 0x3);
        t.cbp = static_cast<uint32_t>((value >> 37) & 0x3FFF);
        t.cpsm = static_cast<uint8_t>((value >> 51) & 0xF);
        t.csm = static_cast<uint8_t>((value >> 55) & 0x1);
        t.csa = static_cast<uint8_t>((value >> 56) & 0x1F);
        t.cld = static_cast<uint8_t>((value >> 61) & 0x7);
        registerClutCbp(t.cbp);

        // M-T1 diag (2026-07-18, generic, bounded, permanent): the §3.24
        // [gs:frame-change] probe only fires on FRAME.FBP transitions, so it
        // never shows which TEX0 programs a screen actually samples once the
        // draw target settles (the common case: many draws to the same FBP,
        // cycling TEX0). Log whenever the *content* of a context's TEX0
        // changes, independent of FBP. Unthrottled like [gs:frame-change] --
        // TEX0 writes are rare relative to per-vertex kicks.
        {
            static std::atomic<uint64_t> s_lastLoggedTex0[2]{0xFFFFFFFFFFFFFFFFull,
                                                              0xFFFFFFFFFFFFFFFFull};
            if (s_lastLoggedTex0[ci].exchange(value, std::memory_order_relaxed) != value)
            {
                std::cout << "[gs:tex0-change] ctx=" << ci
                          << std::hex
                          << " tbp0=0x" << t.tbp0
                          << " psm=0x" << static_cast<uint32_t>(t.psm)
                          << std::dec
                          << " tbw=" << static_cast<uint32_t>(t.tbw)
                          << " tw=" << static_cast<uint32_t>(t.tw)
                          << " th=" << static_cast<uint32_t>(t.th)
                          << std::hex
                          << " cbp=0x" << t.cbp
                          << " cpsm=0x" << static_cast<uint32_t>(t.cpsm)
                          << std::dec
                          << " csm=" << static_cast<uint32_t>(t.csm)
                          << " csa=" << static_cast<uint32_t>(t.csa)
                          << " cld=" << static_cast<uint32_t>(t.cld)
                          << std::endl;
            }
        }
        break;
    }
    case GS_REG_CLAMP_1:
    case GS_REG_CLAMP_2:
    {
        int ci = (regAddr == GS_REG_CLAMP_2) ? 1 : 0;
        m_ctx[ci].clamp = value;
        break;
    }
    case GS_REG_FOG:
        m_curFog = static_cast<uint8_t>((value >> 56) & 0xFF);
        break;
    case GS_REG_TEX1_1:
    case GS_REG_TEX1_2:
    {
        int ci = (regAddr == GS_REG_TEX1_2) ? 1 : 0;
        m_ctx[ci].tex1 = value;
        break;
    }
    case GS_REG_TEX2_1:
    case GS_REG_TEX2_2:
    {
        int ci = (regAddr == GS_REG_TEX2_2) ? 1 : 0;
        auto &t = m_ctx[ci].tex0;
        t.psm = static_cast<uint8_t>((value >> 20) & 0x3F);
        t.cbp = static_cast<uint32_t>((value >> 37) & 0x3FFF);
        t.cpsm = static_cast<uint8_t>((value >> 51) & 0xF);
        t.csm = static_cast<uint8_t>((value >> 55) & 0x1);
        t.csa = static_cast<uint8_t>((value >> 56) & 0x1F);
        t.cld = static_cast<uint8_t>((value >> 61) & 0x7);
        registerClutCbp(t.cbp);
        break;
    }
    case GS_REG_XYOFFSET_1:
    case GS_REG_XYOFFSET_2:
    {
        int ci = (regAddr == GS_REG_XYOFFSET_2) ? 1 : 0;
        m_ctx[ci].xyoffset.ofx = static_cast<uint16_t>(value & 0xFFFF);
        m_ctx[ci].xyoffset.ofy = static_cast<uint16_t>((value >> 32) & 0xFFFF);
        break;
    }
    case GS_REG_PRMODECONT:
        m_prmodecont = (value & 1) != 0;
        break;
    case GS_REG_PRMODE:
        if (!m_prmodecont)
        {
            m_prim.iip = ((value >> 3) & 1) != 0;
            m_prim.tme = ((value >> 4) & 1) != 0;
            m_prim.fge = ((value >> 5) & 1) != 0;
            m_prim.abe = ((value >> 6) & 1) != 0;
            m_prim.aa1 = ((value >> 7) & 1) != 0;
            m_prim.fst = ((value >> 8) & 1) != 0;
            m_prim.ctxt = ((value >> 9) & 1) != 0;
            m_prim.fix = ((value >> 10) & 1) != 0;
        }
        break;
    case GS_REG_TEXCLUT:
        m_texclut.cbw = static_cast<uint8_t>(value & 0x3Fu);
        m_texclut.cou = static_cast<uint8_t>((value >> 6) & 0x3Fu);
        m_texclut.cov = static_cast<uint16_t>((value >> 12) & 0x3FFu);
        break;
    case GS_REG_SCISSOR_1:
    case GS_REG_SCISSOR_2:
    {
        int ci = (regAddr == GS_REG_SCISSOR_2) ? 1 : 0;
        m_ctx[ci].scissor.x0 = static_cast<uint16_t>(value & 0x7FF);
        m_ctx[ci].scissor.x1 = static_cast<uint16_t>((value >> 16) & 0x7FF);
        m_ctx[ci].scissor.y0 = static_cast<uint16_t>((value >> 32) & 0x7FF);
        m_ctx[ci].scissor.y1 = static_cast<uint16_t>((value >> 48) & 0x7FF);
        break;
    }
    case GS_REG_ALPHA_1:
    case GS_REG_ALPHA_2:
    {
        int ci = (regAddr == GS_REG_ALPHA_2) ? 1 : 0;
        m_ctx[ci].alpha = value;
        break;
    }
    case GS_REG_TEST_1:
    case GS_REG_TEST_2:
    {
        int ci = (regAddr == GS_REG_TEST_2) ? 1 : 0;
        m_ctx[ci].test = value;
        break;
    }
    case GS_REG_FRAME_1:
    case GS_REG_FRAME_2:
    {
        int ci = (regAddr == GS_REG_FRAME_2) ? 1 : 0;
        m_ctx[ci].frame.fbp = static_cast<uint32_t>(value & 0x1FF);
        m_ctx[ci].frame.fbw = static_cast<uint32_t>((value >> 16) & 0x3F);
        m_ctx[ci].frame.psm = static_cast<uint8_t>((value >> 24) & 0x3F);
        m_ctx[ci].frame.fbmsk = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);
        break;
    }
    case GS_REG_ZBUF_1:
    case GS_REG_ZBUF_2:
    {
        int ci = (regAddr == GS_REG_ZBUF_2) ? 1 : 0;
        m_ctx[ci].zbuf = value;
        break;
    }
    case GS_REG_FBA_1:
    case GS_REG_FBA_2:
    {
        int ci = (regAddr == GS_REG_FBA_2) ? 1 : 0;
        m_ctx[ci].fba = value;
        break;
    }
    case GS_REG_BITBLTBUF:
    {
        m_bitbltbuf.sbp = static_cast<uint32_t>(value & 0x3FFF);
        m_bitbltbuf.sbw = static_cast<uint8_t>((value >> 16) & 0x3F);
        m_bitbltbuf.spsm = static_cast<uint8_t>((value >> 24) & 0x3F);
        m_bitbltbuf.dbp = static_cast<uint32_t>((value >> 32) & 0x3FFF);
        m_bitbltbuf.dbw = static_cast<uint8_t>((value >> 48) & 0x3F);
        m_bitbltbuf.dpsm = static_cast<uint8_t>((value >> 56) & 0x3F);
        break;
    }
    case GS_REG_TRXPOS:
    {
        m_trxpos.ssax = static_cast<uint16_t>(value & 0x7FF);
        m_trxpos.ssay = static_cast<uint16_t>((value >> 16) & 0x7FF);
        m_trxpos.dsax = static_cast<uint16_t>((value >> 32) & 0x7FF);
        m_trxpos.dsay = static_cast<uint16_t>((value >> 48) & 0x7FF);
        m_trxpos.dir = static_cast<uint8_t>((value >> 59) & 0x3);
        break;
    }
    case GS_REG_TRXREG:
    {
        m_trxreg.rrw = static_cast<uint16_t>(value & 0xFFF);
        m_trxreg.rrh = static_cast<uint16_t>((value >> 32) & 0xFFF);
        break;
    }
    case GS_REG_TRXDIR:
    {
        m_trxdir = static_cast<uint32_t>(value & 0x3);
        m_hwregX = 0;
        m_hwregY = 0;

        if (m_trxdir == 2 && m_vram)
        {
            performLocalToLocalTransfer();
        }
        else if (m_trxdir == 1 && m_vram)
        {
            performLocalToHostToBuffer();
        }
        break;
    }
    case GS_REG_HWREG:
    {
        uint8_t buf[8];
        std::memcpy(buf, &value, 8);
        processImageData(buf, 8);
        break;
    }
    case GS_REG_PABE:
        m_pabe = (value & 1u) != 0u;
        break;
    case GS_REG_TEXFLUSH:
    case GS_REG_SCANMSK:
    case GS_REG_FOGCOL:
    case GS_REG_DIMX:
    case GS_REG_DTHE:
    case GS_REG_COLCLAMP:
    case GS_REG_MIPTBP1_1:
    case GS_REG_MIPTBP1_2:
    case GS_REG_MIPTBP2_1:
    case GS_REG_MIPTBP2_2:
        break;
    case GS_REG_TEXA:
    {
        m_texa.ta0 = static_cast<uint8_t>(value & 0xFFu);
        m_texa.aem = ((value >> 15) & 0x1u) != 0u;
        m_texa.ta1 = static_cast<uint8_t>((value >> 32) & 0xFFu);
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t texaIndex = s_debugTexaWriteCount.fetch_add(1u, std::memory_order_relaxed);
            if (texaIndex < 24u)
            {
                RUNTIME_LOG("[gs:texa] idx=" << texaIndex
                                             << " value=0x" << std::hex << value
                                             << " ta0=0x" << ((value >> 0) & 0xFFu)
                                             << " aem=" << ((value >> 15) & 0x1u)
                                             << " ta1=0x" << ((value >> 32) & 0xFFu)
                                             << std::dec
                                             << std::endl);
            }
        });
        break;
    }
    case GS_REG_SIGNAL:
    {
        if (m_privRegs)
        {
            uint32_t id = static_cast<uint32_t>(value & 0xFFFFFFFF);
            uint32_t mask = static_cast<uint32_t>(value >> 32);
            uint32_t lo = static_cast<uint32_t>(m_privRegs->siglblid & 0xFFFFFFFF);
            lo = (lo & ~mask) | (id & mask);
            m_privRegs->siglblid = (m_privRegs->siglblid & 0xFFFFFFFF00000000ULL) | lo;
            m_privRegs->csr |= 0x1;
        }
        break;
    }
    case GS_REG_FINISH:
    {
        if (m_privRegs)
            m_privRegs->csr |= 0x2;
        break;
    }
    case GS_REG_LABEL:
    {
        if (m_privRegs)
        {
            uint32_t id = static_cast<uint32_t>(value & 0xFFFFFFFF);
            uint32_t mask = static_cast<uint32_t>(value >> 32);
            uint32_t hi = static_cast<uint32_t>(m_privRegs->siglblid >> 32);
            hi = (hi & ~mask) | (id & mask);
            m_privRegs->siglblid = (static_cast<uint64_t>(hi) << 32) | (m_privRegs->siglblid & 0xFFFFFFFF);
        }
        break;
    }
    case 0x59:
        if (m_privRegs)
            m_privRegs->dispfb1 = value;
        break;
    case 0x5a:
        if (m_privRegs)
            m_privRegs->display1 = value;
        break;
    case 0x5b:
        if (m_privRegs)
            m_privRegs->dispfb2 = value;
        break;
    case 0x5c:
        if (m_privRegs)
            m_privRegs->display2 = value;
        break;
    case 0x5f:
        if (m_privRegs)
            m_privRegs->bgcolor = value;
        break;
    default:
        break;
    }
}

void GS::performLocalToLocalTransfer()
{
    if (!m_vram)
        return;

    uint32_t sbp = m_bitbltbuf.sbp;
    uint8_t sbw = m_bitbltbuf.sbw;
    uint8_t spsm = m_bitbltbuf.spsm;
    uint32_t dbp = m_bitbltbuf.dbp;
    uint8_t dbw = m_bitbltbuf.dbw;
    uint8_t dpsm = m_bitbltbuf.dpsm;

    if (sbw == 0)
        sbw = 1;
    if (dbw == 0)
        dbw = 1;

    const uint32_t rrw = m_trxreg.rrw;
    const uint32_t rrh = m_trxreg.rrh;
    const uint32_t ssax = m_trxpos.ssax;
    const uint32_t ssay = m_trxpos.ssay;
    const uint32_t dsax = m_trxpos.dsax;
    const uint32_t dsay = m_trxpos.dsay;
    const GSTransferTraversal traversal = decodeTransferTraversal(m_trxpos.dir);
    const bool formatAware = (spsm == dpsm) && supportsFormatAwareLocalCopy(spsm);

    if (rrw == 0u || rrh == 0u)
    {
        return;
    }

    // Bounded permanent log for local->local blits (the composite-to-display
    // path some titles use instead of a DISPFB flip). First 16 verbose, then
    // every 600th.
    {
        static std::atomic<uint64_t> s_l2lCalls{0};
        const uint64_t n = s_l2lCalls.fetch_add(1, std::memory_order_relaxed);
        if (n < 16u || (n % 600u) == 0u)
        {
            std::cout << "[gs:l2l] #" << n
                      << std::hex
                      << " sbp=0x" << sbp << " dbp=0x" << dbp
                      << " spsm=0x" << static_cast<uint32_t>(spsm)
                      << " dpsm=0x" << static_cast<uint32_t>(dpsm)
                      << std::dec
                      << " rr=" << rrw << "x" << rrh
                      << " clutTarget=" << (isKnownClutCbp(dbp) ? 1 : 0)
                      << std::endl;
        }
    }

    PS2_IF_AGRESSIVE_LOGS({
        if ((spsm == GS_PSM_T4 || dpsm == GS_PSM_T4) &&
            s_debugLocalCopyCount.fetch_add(1u, std::memory_order_relaxed) < 96u)
        {
            RUNTIME_LOG("[gs:l2l] sbp=" << sbp
                                        << " dbp=" << dbp
                                        << " sbw=" << static_cast<uint32_t>(sbw)
                                        << " dbw=" << static_cast<uint32_t>(dbw)
                                        << " spsm=0x" << std::hex << static_cast<uint32_t>(spsm)
                                        << " dpsm=0x" << static_cast<uint32_t>(dpsm) << std::dec
                                        << " ss=(" << ssax << "," << ssay << ")"
                                        << " ds=(" << dsax << "," << dsay << ")"
                                        << " rr=(" << rrw << "," << rrh << ")"
                                        << " dir=" << static_cast<uint32_t>(m_trxpos.dir)
                                        << " formatAware=" << (formatAware ? 1 : 0) << std::endl);
        }
    });

    if (formatAware)
    {
        for (uint32_t row = 0; row < rrh; ++row)
        {
            const uint32_t srcY = transferCoord(ssay, rrh, row, traversal.reverseY);
            const uint32_t dstY = transferCoord(dsay, rrh, row, traversal.reverseY);
            for (uint32_t col = 0; col < rrw; ++col)
            {
                const uint32_t srcX = transferCoord(ssax, rrw, col, traversal.reverseX);
                const uint32_t dstX = transferCoord(dsax, rrw, col, traversal.reverseX);
                const uint32_t pixel =
                    readTransferPixel(m_vram, m_vramSize, sbp, sbw, spsm, srcX, srcY);
                writeTransferPixel(m_vram, m_vramSize, dbp, dbw, dpsm, dstX, dstY, pixel);
            }
        }
    }
    else
    {
        const uint32_t srcBase = sbp * 256u;
        const uint32_t dstBase = dbp * 256u;
        uint32_t srcBpp = bitsPerPixel(spsm) / 8u;
        uint32_t dstBpp = bitsPerPixel(dpsm) / 8u;
        if (srcBpp == 0)
            srcBpp = 4;
        if (dstBpp == 0)
            dstBpp = 4;
        const uint32_t srcStride = static_cast<uint32_t>(sbw) * 64u * srcBpp;
        const uint32_t dstStride = static_cast<uint32_t>(dbw) * 64u * dstBpp;
        const uint32_t copyBpp = (srcBpp < dstBpp) ? srcBpp : dstBpp;

        uint8_t pixelBytes[4] = {};
        for (uint32_t row = 0; row < rrh; ++row)
        {
            const uint32_t srcY = transferCoord(ssay, rrh, row, traversal.reverseY);
            const uint32_t dstY = transferCoord(dsay, rrh, row, traversal.reverseY);
            for (uint32_t col = 0; col < rrw; ++col)
            {
                const uint32_t srcX = transferCoord(ssax, rrw, col, traversal.reverseX);
                const uint32_t dstX = transferCoord(dsax, rrw, col, traversal.reverseX);
                const uint32_t srcOff = srcBase + srcY * srcStride + srcX * srcBpp;
                const uint32_t dstOff = dstBase + dstY * dstStride + dstX * dstBpp;
                if (srcOff + copyBpp > m_vramSize || dstOff + copyBpp > m_vramSize)
                {
                    continue;
                }

                std::memcpy(pixelBytes, m_vram + srcOff, copyBpp);
                std::memcpy(m_vram + dstOff, pixelBytes, copyBpp);
            }
        }
    }

    if (sbp == 0u && (dbp == 0u || dbp == 0x20u) && rrw >= 640u && rrh >= 512u)
    {
        m_lastDisplayBaseBytes = (dbp == 0x20u) ? 8192u : 0u;
        snapshotVRAM();
    }
}

void GS::vertexKick(bool drawing)
{
    ++m_vtxCount;
    ++m_vtxIndex;

    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t debugIndex = s_debugGsVertexKickCount.fetch_add(1, std::memory_order_relaxed);
        if (debugIndex < 96u)
        {
            RUNTIME_LOG("[gs:kick] idx=" << debugIndex
                                         << " drawing=" << static_cast<uint32_t>(drawing ? 1u : 0u)
                                         << " prim=" << static_cast<uint32_t>(m_prim.type)
                                         << " vtxCount=" << m_vtxCount
                                         << std::endl);
        }
    });

    int needed = 0;
    switch (m_prim.type)
    {
    case GS_PRIM_POINT:
        needed = 1;
        break;
    case GS_PRIM_LINE:
        needed = 2;
        break;
    case GS_PRIM_LINESTRIP:
        needed = 2;
        break;
    case GS_PRIM_TRIANGLE:
        needed = 3;
        break;
    case GS_PRIM_TRISTRIP:
        needed = 3;
        break;
    case GS_PRIM_TRIFAN:
        needed = 3;
        break;
    case GS_PRIM_SPRITE:
        needed = 2;
        break;
    default:
        return;
    }

    // ---- census (see adcWindowFixOn above) --------------------------------
    {
        const uint32_t t = static_cast<uint32_t>(m_prim.type) & 7u;
        if (drawing)
            ++s_vkKickDraw[t];
        else
        {
            ++s_vkKickSkip[t];
            if (m_vtxCount >= needed)
                ++s_vkDesync[t];
        }
        if ((++s_vkTotal % 2000000u) == 0u)
        {
            std::cout << "[gs:vkcensus] total=" << s_vkTotal << " adcFix="
                      << (adcWindowFixOn() ? 1 : 0);
            static const char *kNames[8] = {"point", "line", "linestrip", "tri",
                                            "tristrip", "trifan", "sprite", "bad"};
            for (uint32_t i = 0; i < 8u; ++i)
            {
                if ((s_vkKickDraw[i] | s_vkKickSkip[i]) == 0u)
                    continue;
                std::cout << ' ' << kNames[i] << "={draw=" << s_vkKickDraw[i]
                          << ",adcskip=" << s_vkKickSkip[i]
                          << ",desync=" << s_vkDesync[i]
                          << ",prims=" << s_vkDrawn[i] << '}';
            }
            std::cout << std::endl;
        }
    }

    if (m_vtxCount < needed)
        return;

    // Non-drawing kick: no primitive, but the window maintenance below still
    // has to run (that is the whole point of an ADC vertex). Historical
    // behaviour bailed out here instead.
    if (!drawing && !adcWindowFixOn())
        return;

    if (!drawing)
    {
        switch (m_prim.type)
        {
        case GS_PRIM_LINESTRIP:
            m_vtxQueue[0] = m_vtxQueue[1];
            m_vtxCount = 1;
            break;
        case GS_PRIM_TRISTRIP:
            m_vtxQueue[0] = m_vtxQueue[1];
            m_vtxQueue[1] = m_vtxQueue[2];
            m_vtxCount = 2;
            break;
        case GS_PRIM_TRIFAN:
            m_vtxQueue[1] = m_vtxQueue[2];
            m_vtxCount = 2;
            break;
        default:
            m_vtxCount = 0;
            break;
        }
        return;
    }

    ++s_vkDrawn[static_cast<uint32_t>(m_prim.type) & 7u];
    m_statPrims.fetch_add(1, std::memory_order_relaxed);
    GSContext &drawCtx = activeContext();
    m_statLastDrawFbp.store(drawCtx.frame.fbp, std::memory_order_relaxed);

    // M0 §3.24 diagnostic: log FRAME (draw target) + TEX0 (source texture)
    // whenever the draw-target FBP changes from the previous drawn batch.
    // Unthrottled (FBP changes should be rare relative to per-vertex kicks);
    // this is the direct answer to "does any executed batch ever target
    // fbp=0x70 (the displayed buffer)?" and, if so, what it samples from.
    {
        static std::atomic<uint32_t> s_lastLoggedFbp{0xFFFFFFFFu};
        const uint32_t fbpNow = drawCtx.frame.fbp;
        if (s_lastLoggedFbp.exchange(fbpNow, std::memory_order_relaxed) != fbpNow)
        {
            std::cout << "[gs:frame-change] prim=" << static_cast<uint32_t>(m_prim.type)
                       << " ctxt=" << static_cast<uint32_t>(m_prim.ctxt)
                       << std::hex
                       << " frame_fbp=0x" << fbpNow
                       << " frame_fbw=0x" << drawCtx.frame.fbw
                       << " frame_psm=0x" << static_cast<uint32_t>(drawCtx.frame.psm)
                       << " tex0_tbp=0x" << drawCtx.tex0.tbp0
                       << " tex0_psm=0x" << static_cast<uint32_t>(drawCtx.tex0.psm)
                       << " tex0_tw=0x" << drawCtx.tex0.tw
                       << " tex0_th=0x" << drawCtx.tex0.th
                       << " tme=" << (m_prim.tme ? 1u : 0u)
                       << " prmodecont=" << (m_prmodecont ? 1u : 0u)
                       << std::dec
                       << std::endl;
        }
    }

    m_rasterizer.drawPrimitive(this);

    switch (m_prim.type)
    {
    case GS_PRIM_LINE:
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_SPRITE:
    case GS_PRIM_POINT:
        m_vtxCount = 0;
        break;
    case GS_PRIM_LINESTRIP:
        m_vtxQueue[0] = m_vtxQueue[1];
        m_vtxCount = 1;
        break;
    case GS_PRIM_TRISTRIP:
        m_vtxQueue[0] = m_vtxQueue[1];
        m_vtxQueue[1] = m_vtxQueue[2];
        m_vtxCount = 2;
        break;
    case GS_PRIM_TRIFAN:
        m_vtxQueue[1] = m_vtxQueue[2];
        m_vtxCount = 2;
        break;
    default:
        m_vtxCount = 0;
        break;
    }
}

void GS::processImageData(const uint8_t *data, uint32_t sizeBytes)
{
    if (m_trxdir != 0 || !m_vram)
        return;

    m_statImageBytes.fetch_add(sizeBytes, std::memory_order_relaxed);

    // FMV-visibility RCA (2026-07-25): stamp "a movie frame is being streamed
    // into VRAM right now". The MPEG frame upload is the only traffic in this
    // game shaped as 16x16 CT32 tiles into a 512-pixel-wide (DBW=8) surface,
    // so this is a precise, title-agnostic movie-active edge. See
    // g_dq8MovieUploadMs at the top of this file.
    if (m_bitbltbuf.dbw == 8u && m_bitbltbuf.dpsm == GS_PSM_CT32 &&
        m_trxreg.rrw == 16u && m_trxreg.rrh == 16u)
    {
        const uint64_t nowMs = dq8ProbeNowMs();
        g_dq8MovieUploadMs.store(nowMs, std::memory_order_relaxed);
        g_dq8MovieUploadDbp.store(m_bitbltbuf.dbp, std::memory_order_relaxed);

        // Once-a-second census while a movie is live: primitives the guest
        // actually executed, and how many of them sampled the movie surface.
        // Emitted from HERE (the upload path, which keeps ticking) rather
        // than from the rasterizer, so that "zero primitives at all" is still
        // reported -- that is precisely the case under investigation.
        static const bool s_census = []() {
            const char *e = std::getenv("DQ8_MOVIEQUAD");
            return e && *e && *e != '0';
        }();
        if (s_census)
        {
            static std::atomic<uint64_t> s_lastReport{0};
            uint64_t prev = s_lastReport.load(std::memory_order_relaxed);
            if (nowMs >= prev + 1000ull &&
                s_lastReport.compare_exchange_strong(prev, nowMs, std::memory_order_relaxed))
            {
                std::cout << "[dq8:moviecensus] t=" << nowMs << "ms"
                          << " movieDbp=0x" << std::hex << m_bitbltbuf.dbp << std::dec
                          << " primsDuringMovie=" << g_dq8MoviePrimsTotal.load(std::memory_order_relaxed)
                          << " primsSamplingMovie=" << g_dq8MoviePrimsTextured.load(std::memory_order_relaxed)
                          << std::endl;
            }
        }
    }

    // Bounded destination log for IMAGE-mode uploads: where does host->local
    // data land (BITBLTBUF.DBP pages, DPSM, TRXREG size)? First 16 distinct
    // DBP/DPSM pairs verbose, then every 600th call. Answers "are these
    // texture uploads or direct-to-display writes" without a debugger. Also
    // flags whether this upload targets a block ever seen as a TEX0/TEX2
    // CLUT base pointer, to check live whether CLUTs actually get populated.
    {
        static std::atomic<uint64_t> s_imgCalls{0};
        static std::atomic<uint32_t> s_lastLoggedDbp{0xFFFFFFFFu};
        static std::atomic<uint32_t> s_lastLoggedDpsm{0xFFFFFFFFu};
        const uint64_t n = s_imgCalls.fetch_add(1, std::memory_order_relaxed);
        const uint32_t dbpNow = m_bitbltbuf.dbp;
        const uint32_t dpsmNow = m_bitbltbuf.dpsm;
        if (n < 16u || (n % 600u) == 0u ||
            dbpNow != s_lastLoggedDbp.load(std::memory_order_relaxed) ||
            dpsmNow != s_lastLoggedDpsm.load(std::memory_order_relaxed))
        {
            s_lastLoggedDbp.store(dbpNow, std::memory_order_relaxed);
            s_lastLoggedDpsm.store(dpsmNow, std::memory_order_relaxed);
            std::cout << "[gs:image] #" << n
                      << std::hex
                      << " dbp=0x" << dbpNow
                      << " dpsm=0x" << dpsmNow
                      << " dbw=" << std::dec << static_cast<uint32_t>(m_bitbltbuf.dbw)
                      << " rr=" << m_trxreg.rrw << "x" << m_trxreg.rrh
                      << " bytes=" << sizeBytes
                      << " clutTarget=" << (isKnownClutCbp(dbpNow) ? 1 : 0)
                      << std::endl;
        }
    }

    uint32_t dbp = m_bitbltbuf.dbp;
    uint8_t dbw = m_bitbltbuf.dbw;
    uint8_t dpsm = m_bitbltbuf.dpsm;

    if (dbw == 0)
        dbw = 1;
    uint32_t base = dbp * 256u;
    uint32_t bpp = bitsPerPixel(dpsm);
    uint32_t stridePixels = static_cast<uint32_t>(dbw) * 64u;

    uint32_t rrw = m_trxreg.rrw;
    uint32_t rrh = m_trxreg.rrh;
    uint32_t dsax = m_trxpos.dsax;
    uint32_t dsay = m_trxpos.dsay;

    if (dpsm == GS_PSM_T4HL || dpsm == GS_PSM_T4HH || dpsm == GS_PSM_T8H)
    {
        // GS "H" formats (T4HL/T4HH/T8H) share storage with a CT32 word but
        // the host->local WIRE transfer is still the packed 4bpp/8bpp stream
        // (PCSX2's GSLocalMemory reports trbpp=4 for T4HL/T4HH and trbpp=8
        // for T8H -- NOT 32; an earlier version of this fix believed
        // trbpp=32 based on a byte-count coincidence that actually proved
        // DQ8's engine over-transfers these uploads by 8x, not that the
        // format is 32bpp -- see PS2_PROJECT_STATE.md §3.27). Consume
        // nibbles (T4HL/T4HH) or bytes (T8H) at the packed density, and
        // place each index at its CT32 *storage* position (T4HL bits
        // 24-27, T4HH bits 28-31, T8H bits 24-31) via writeTransferPixel(),
        // which RMWs only this plane's own bits so a sibling plane already
        // resident in the same CT32 word (e.g. the other T4H glyph plane)
        // survives.
        //
        // Because DQ8 over-transfers 8x, the destination rect (TRXREG) is
        // full long before the source bytes run out; the trailing bytes
        // belong to the NEXT plane's staging buffer, not this one. Hard-stop
        // consuming the instant m_hwregY reaches rrh and silently discard
        // whatever's left in `data` -- mirrors the plain-T4 branch's
        // writeT4Nibble() early-return below, just applied to the whole
        // loop instead of a single nibble.
        const uint32_t widthBlocks = (dbw != 0) ? static_cast<uint32_t>(dbw) : 1u;

        if (dpsm == GS_PSM_T8H)
        {
            uint32_t offset = 0;
            while (offset < sizeBytes && m_hwregY < rrh)
            {
                const uint8_t srcByte = data[offset++];
                const uint32_t vx = dsax + m_hwregX;
                const uint32_t vy = dsay + m_hwregY;
                writeTransferPixel(m_vram, m_vramSize, dbp, static_cast<uint8_t>(widthBlocks), dpsm, vx, vy,
                                    static_cast<uint32_t>(srcByte) << 24);

                ++m_hwregX;
                if (m_hwregX >= rrw)
                {
                    m_hwregX = 0;
                    ++m_hwregY;
                }
            }
        }
        else
        {
            const uint32_t shift = (dpsm == GS_PSM_T4HH) ? 28u : 24u;

            auto writeHNibble = [&](uint8_t nibble)
            {
                if (m_hwregY >= rrh)
                    return;

                const uint32_t vx = dsax + m_hwregX;
                const uint32_t vy = dsay + m_hwregY;
                writeTransferPixel(m_vram, m_vramSize, dbp, static_cast<uint8_t>(widthBlocks), dpsm, vx, vy,
                                    (static_cast<uint32_t>(nibble & 0x0Fu)) << shift);

                ++m_hwregX;
                if (m_hwregX >= rrw)
                {
                    m_hwregX = 0;
                    ++m_hwregY;
                }
            };

            uint32_t offset = 0;
            while (offset < sizeBytes && m_hwregY < rrh)
            {
                const uint8_t srcByte = data[offset++];
                const uint32_t srcLo = srcByte & 0x0Fu;
                const uint32_t srcHi = (srcByte >> 4) & 0x0Fu;
                const uint32_t xBefore = m_hwregX;

                writeHNibble(static_cast<uint8_t>(srcLo));
                if ((xBefore + 1u) < rrw && m_hwregY < rrh)
                {
                    writeHNibble(static_cast<uint8_t>(srcHi));
                }
            }
        }
    }
    else if (bpp == 4)
    {
        uint32_t widthBlocks = (dbw != 0) ? static_cast<uint32_t>(dbw) : 1u;
        uint32_t offset = 0;

        // T4 image uploads can be split across multiple GIF IMAGE packets.
        // Keep advancing from the previous HWREG position instead of restarting at (0, 0).
        auto writeT4Nibble = [&](uint8_t nibble)
        {
            if (m_hwregY >= rrh)
                return;

            const uint32_t vx = dsax + m_hwregX;
            const uint32_t vy = dsay + m_hwregY;

            const uint32_t nibbleAddr = GSPSMT4::addrPSMT4(dbp, widthBlocks, vx, vy);
            const uint32_t byteOff = nibbleAddr >> 1;

            if (byteOff < m_vramSize)
            {
                const int shift = static_cast<int>((nibbleAddr & 1u) << 2);
                uint8_t &b = m_vram[byteOff];
                b = static_cast<uint8_t>((b & (0xF0u >> shift)) | ((nibble & 0x0Fu) << shift));
            }
            ++m_hwregX;
            if (m_hwregX >= rrw)
            {
                m_hwregX = 0;
                ++m_hwregY;
            }
        };

        while (offset < sizeBytes && m_hwregY < rrh)
        {
            const uint8_t srcByte = data[offset++];
            const uint32_t srcLo = srcByte & 0x0Fu;
            const uint32_t srcHi = (srcByte >> 4) & 0x0Fu;
            const uint32_t xBefore = m_hwregX;

            writeT4Nibble(static_cast<uint8_t>(srcLo));
            if ((xBefore + 1u) < rrw && m_hwregY < rrh)
            {
                writeT4Nibble(static_cast<uint8_t>(srcHi));
            }
        }
    }
    else if (dpsm == GS_PSM_T8)
    {
        uint32_t offset = 0;
        while (offset < sizeBytes && m_hwregY < rrh)
        {
            uint32_t pixelsLeft = rrw - m_hwregX;
            uint32_t pixelsToCopy = std::min<uint32_t>(pixelsLeft, sizeBytes - offset);
            if (pixelsToCopy == 0)
            {
                break;
            }

            for (uint32_t i = 0; i < pixelsToCopy; ++i)
            {
                const uint32_t vx = dsax + m_hwregX + i;
                const uint32_t vy = dsay + m_hwregY;
                const uint32_t dst = GSPSMT8::addrPSMT8(dbp, dbw, vx, vy);
                if (dst < m_vramSize)
                {
                    m_vram[dst] = data[offset + i];
                }
            }

            offset += pixelsToCopy;
            m_hwregX += pixelsToCopy;
            if (m_hwregX >= rrw)
            {
                m_hwregX = 0;
                ++m_hwregY;
            }
        }
    }
    else if (dpsm == GS_PSM_CT24 || dpsm == GS_PSM_Z24)
    {
        uint32_t transferBpp = 3;

        uint32_t offset = 0;
        while (offset < sizeBytes && m_hwregY < rrh)
        {
            uint32_t pixelsLeft = rrw - m_hwregX;
            uint32_t srcBytesLeft = pixelsLeft * transferBpp;
            uint32_t bytesAvail = sizeBytes - offset;
            uint32_t pixelsToCopy = pixelsLeft;
            if (srcBytesLeft > bytesAvail)
                pixelsToCopy = bytesAvail / transferBpp;

            if (pixelsToCopy == 0)
                break;

            if (pixelsToCopy > 0)
            {
                for (uint32_t p = 0; p < pixelsToCopy; ++p)
                {
                    const uint32_t vx = dsax + m_hwregX + p;
                    const uint32_t vy = dsay + m_hwregY;
                    const uint32_t dstOff = GSPSMCT32::addrPSMCT32(dbp, dbw, vx, vy);
                    if (dstOff + 4u <= m_vramSize)
                    {
                        m_vram[dstOff + 0u] = data[offset + p * 3u + 0u];
                        m_vram[dstOff + 1u] = data[offset + p * 3u + 1u];
                        m_vram[dstOff + 2u] = data[offset + p * 3u + 2u];
                        m_vram[dstOff + 3u] = 0x80u;
                    }
                }
            }

            offset += pixelsToCopy * transferBpp;
            m_hwregX += pixelsToCopy;
            if (m_hwregX >= rrw)
            {
                m_hwregX = 0;
                ++m_hwregY;
            }
        }
    }
    else
    {
        uint32_t bytesPerPixel = bpp / 8u;
        if (bytesPerPixel == 0)
            bytesPerPixel = 4;
        uint32_t strideBytes = stridePixels * bytesPerPixel;
        uint32_t rowBytes = rrw * bytesPerPixel;

        uint32_t offset = 0;
        while (offset < sizeBytes && m_hwregY < rrh)
        {
            uint32_t dstY = dsay + m_hwregY;
            uint32_t pixelsLeft = rrw - m_hwregX;
            uint32_t bytesLeft = pixelsLeft * bytesPerPixel;
            uint32_t bytesAvail = sizeBytes - offset;
            if (bytesLeft > bytesAvail)
                bytesLeft = (bytesAvail / bytesPerPixel) * bytesPerPixel;

            uint32_t pixelsCopied = bytesLeft / bytesPerPixel;
            if ((dpsm == GS_PSM_CT32 || dpsm == GS_PSM_Z32) && pixelsCopied > 0)
            {
                for (uint32_t p = 0; p < pixelsCopied; ++p)
                {
                    const uint32_t vx = dsax + m_hwregX + p;
                    const uint32_t vy = dstY;
                    const uint32_t dstOff = GSPSMCT32::addrPSMCT32(dbp, dbw, vx, vy);
                    if (dstOff + 4u <= m_vramSize)
                        std::memcpy(m_vram + dstOff, data + offset + p * 4u, 4u);
                }
            }
            else
            {
                uint32_t dstOff = base + dstY * strideBytes + (dsax + m_hwregX) * bytesPerPixel;
                if (dstOff + bytesLeft <= m_vramSize && bytesLeft > 0)
                    std::memcpy(m_vram + dstOff, data + offset, bytesLeft);
            }

            offset += bytesLeft;
            m_hwregX += pixelsCopied;
            if (m_hwregX >= rrw)
            {
                m_hwregX = 0;
                ++m_hwregY;
            }
        }
    }
}

void GS::performLocalToHostToBuffer()
{
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0;
    if (!m_vram)
        return;

    uint32_t sbp = m_bitbltbuf.sbp;
    uint8_t sbw = m_bitbltbuf.sbw;
    uint8_t spsm = m_bitbltbuf.spsm;

    if (sbw == 0)
        sbw = 1;
    uint32_t base = sbp * 256u;
    uint32_t bpp = bitsPerPixel(spsm);
    uint32_t stridePixels = static_cast<uint32_t>(sbw) * 64u;

    uint32_t rrw = m_trxreg.rrw;
    uint32_t rrh = m_trxreg.rrh;
    uint32_t ssax = m_trxpos.ssax;
    uint32_t ssay = m_trxpos.ssay;

    if (bpp == 4)
    {
        uint32_t rowBytes = (rrw + 1u) / 2u;
        if (rowBytes == 0)
            rowBytes = 1;
        m_localToHostBuffer.reserve(rowBytes * rrh);
        uint32_t widthBlocks = static_cast<uint32_t>(sbw);
        for (uint32_t y = 0; y < rrh; ++y)
        {
            for (uint32_t x = 0; x < rrw; ++x)
            {
                uint32_t vx = ssax + x;
                uint32_t vy = ssay + y;
                uint32_t nibbleAddr = GSPSMT4::addrPSMT4(sbp, widthBlocks, vx, vy);
                uint32_t byteOff = nibbleAddr >> 1;
                uint8_t nibble = 0;
                if (byteOff < m_vramSize)
                {
                    int shift = static_cast<int>((nibbleAddr & 1u) << 2);
                    nibble = static_cast<uint8_t>((m_vram[byteOff] >> shift) & 0x0Fu);
                }
                if (x & 1u)
                    m_localToHostBuffer.back() = static_cast<uint8_t>((m_localToHostBuffer.back() & 0x0Fu) | (nibble << 4));
                else
                    m_localToHostBuffer.push_back(nibble);
            }
        }
    }
    else if (spsm == GS_PSM_T8)
    {
        m_localToHostBuffer.reserve(rrw * rrh);
        for (uint32_t y = 0; y < rrh; ++y)
        {
            for (uint32_t x = 0; x < rrw; ++x)
            {
                const uint32_t src = GSPSMT8::addrPSMT8(sbp, sbw, ssax + x, ssay + y);
                m_localToHostBuffer.push_back((src < m_vramSize) ? m_vram[src] : 0u);
            }
        }
    }
    else if (spsm == GS_PSM_CT24 || spsm == GS_PSM_Z24)
    {
        uint32_t transferBpp = 3;
        m_localToHostBuffer.reserve(rrw * rrh * transferBpp);

        for (uint32_t y = 0; y < rrh; ++y)
        {
            for (uint32_t x = 0; x < rrw; ++x)
            {
                uint32_t srcOff = GSPSMCT32::addrPSMCT32(sbp, sbw, ssax + x, ssay + y);
                if (srcOff + 4 <= m_vramSize)
                {
                    m_localToHostBuffer.push_back(m_vram[srcOff + 0]);
                    m_localToHostBuffer.push_back(m_vram[srcOff + 1]);
                    m_localToHostBuffer.push_back(m_vram[srcOff + 2]);
                }
            }
        }
    }
    else
    {
        uint32_t bytesPerPixel = bpp / 8u;
        if (bytesPerPixel == 0)
            bytesPerPixel = 4;
        uint32_t strideBytes = stridePixels * bytesPerPixel;
        uint32_t rowBytes = rrw * bytesPerPixel;
        m_localToHostBuffer.reserve(rowBytes * rrh);

        for (uint32_t y = 0; y < rrh; ++y)
        {
            if (spsm == GS_PSM_CT32 || spsm == GS_PSM_Z32)
            {
                for (uint32_t x = 0; x < rrw; ++x)
                {
                    const uint32_t srcOff = GSPSMCT32::addrPSMCT32(sbp, sbw, ssax + x, ssay + y);
                    if (srcOff + 4u <= m_vramSize)
                    {
                        m_localToHostBuffer.push_back(m_vram[srcOff + 0u]);
                        m_localToHostBuffer.push_back(m_vram[srcOff + 1u]);
                        m_localToHostBuffer.push_back(m_vram[srcOff + 2u]);
                        m_localToHostBuffer.push_back(m_vram[srcOff + 3u]);
                    }
                }
            }
            else
            {
                uint32_t srcOff = base + (ssay + y) * strideBytes + ssax * bytesPerPixel;
                if (srcOff + rowBytes <= m_vramSize)
                {
                    for (uint32_t i = 0; i < rowBytes; ++i)
                        m_localToHostBuffer.push_back(m_vram[srcOff + i]);
                }
            }
        }
    }
}

bool GS::clearFramebufferContext(uint32_t contextIndex, uint32_t rgba)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    return clearFramebufferRect(m_vram, m_vramSize, m_ctx[(contextIndex != 0u) ? 1 : 0], rgba);
}

bool GS::clearActiveFramebuffer(uint32_t rgba)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    return clearFramebufferRect(m_vram, m_vramSize, activeContext(), rgba);
}

uint32_t GS::consumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    if (!dst || maxBytes == 0)
        return 0;
    size_t avail = m_localToHostBuffer.size() - m_localToHostReadPos;
    if (avail == 0)
        return 0;
    size_t toCopy = (avail < maxBytes) ? avail : static_cast<size_t>(maxBytes);
    std::memcpy(dst, m_localToHostBuffer.data() + m_localToHostReadPos, toCopy);
    m_localToHostReadPos += toCopy;
    return static_cast<uint32_t>(toCopy);
}
