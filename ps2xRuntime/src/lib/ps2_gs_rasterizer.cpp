#include "runtime/ps2_gs_rasterizer.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_gs_common.h"
#include "runtime/ps2_gs_psmct16.h"
#include "runtime/ps2_gs_psmct32.h"
#include "runtime/ps2_gs_psmt4.h"
#include "runtime/ps2_gs_psmt8.h"
#include "ps2_log.h"
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>

using namespace GSInternal;

namespace
{
    // Local twin of ps2DiagEnvLimit/ps2DiagLogBudget (ps2_runtime.h) for TUs
    // that do not include that header. A capped diagnostic that drops events
    // silently is worse than no diagnostic: the truncated tail reads as "the
    // event never happened again". See ps2_runtime.h for the canonical
    // version and rationale.
    uint32_t localDiagEnvLimit(const char *envName, uint32_t fallback)
    {
        const char *raw = std::getenv(envName);
        if (!raw || raw[0] == '\0')
        {
            return fallback;
        }
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(raw, &end, 0);
        if (end == raw)
        {
            return fallback;
        }
        return static_cast<uint32_t>(parsed);
    }

    bool localDiagLogBudget(std::ostream &os, const char *tag, const char *envName,
                            uint32_t limit, uint32_t logIndex,
                            std::atomic<bool> &truncationAnnounced)
    {
        if (limit == 0u || logIndex < limit)
        {
            return true;
        }
        bool expected = false;
        if (truncationAnnounced.compare_exchange_strong(expected, true, std::memory_order_relaxed))
        {
            os << tag << " TRUNCATED after " << limit
               << " events; further events are NOT logged (raise or disable with "
               << envName << "=<n>, 0 = unlimited)" << std::endl;
        }
        return false;
    }
}

// FMV-visibility RCA (2026-07-25): defined in ps2_gs_gpu.cpp. Last 512-wide
// CT32 16x16-tile IMAGE upload (the MPEG frame streaming signature) and its
// BITBLTBUF.DBP, so the movie-quad probe below can scope itself to real
// playback and know which VRAM base the current movie surface lives at.
extern std::atomic<uint64_t> g_dq8MovieUploadMs;
extern std::atomic<uint32_t> g_dq8MovieUploadDbp;

// Counts every primitive executed while a movie is streaming, split by
// "samples the movie surface" vs not. The periodic read-out lives in
// GS::processImageData so it still reports when the count is ZERO.
extern std::atomic<uint64_t> g_dq8MoviePrimsTotal;
extern std::atomic<uint64_t> g_dq8MoviePrimsTextured;

// Shared monotonic-ms clock, defined in ps2_gs_gpu.cpp (see its definition
// for the one-epoch-per-process rationale). Forward-declared here the same
// way ps2_vif1_interpreter.cpp / ps2_vu1.cpp already do.
uint64_t dq8ProbeNowMs();

namespace
{
    float fabsQ(float q)
    {
        return (std::fabs(q) > 1.0e-8f) ? q : 1.0f;
    }

    uint32_t decodePSMCT16(uint16_t pixel)
    {
        const uint32_t r = ((pixel >> 0) & 0x1Fu) << 3;
        const uint32_t g = ((pixel >> 5) & 0x1Fu) << 3;
        const uint32_t b = ((pixel >> 10) & 0x1Fu) << 3;
        const uint32_t a = (pixel & 0x8000u) ? 0x80u : 0u;
        return r | (g << 8) | (b << 16) | (a << 24);
    }

    uint32_t applyTexa(const GSTexaReg &texa, uint8_t psm, uint32_t texel)
    {
        if (psm == GS_PSM_CT32)
            return texel;

        const uint8_t r = static_cast<uint8_t>(texel & 0xFFu);
        const uint8_t g = static_cast<uint8_t>((texel >> 8) & 0xFFu);
        const uint8_t b = static_cast<uint8_t>((texel >> 16) & 0xFFu);
        const bool rgbZero = r == 0u && g == 0u && b == 0u;
        uint8_t a = static_cast<uint8_t>((texel >> 24) & 0xFFu);

        switch (psm)
        {
        case GS_PSM_CT24:
            a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            if ((a & 0x80u) != 0u)
                a = texa.ta1;
            else
                a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        default:
            break;
        }

        return (texel & 0x00FFFFFFu) | (static_cast<uint32_t>(a) << 24);
    }

    uint16_t encodePSMCT16(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
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

    std::atomic<uint32_t> s_debugPrimitiveCount{0};
    std::atomic<uint32_t> s_debugPixelCount{0};
    std::atomic<uint32_t> s_debugContext1PrimitiveCount{0};
    std::atomic<uint32_t> s_debugFbp150PixelCount{0};
    bool passesAlphaTest(uint64_t testReg, uint8_t alpha)
    {
        if ((testReg & 0x1u) == 0u)
            return true;

        const uint8_t atst = static_cast<uint8_t>((testReg >> 1) & 0x7u);
        const uint8_t aref = static_cast<uint8_t>((testReg >> 4) & 0xFFu);

        switch (atst)
        {
        case 0:
            return false;
        case 1:
            return true;
        case 2:
            return alpha < aref;
        case 3:
            return alpha <= aref;
        case 4:
            return alpha == aref;
        case 5:
            return alpha >= aref;
        case 6:
            return alpha > aref;
        case 7:
            return alpha != aref;
        default:
            return true;
        }
    }

    // Bridges writePixel's alpha/depth AFAIL decision (see writePixel body)
    // to the existing color-write code below, which only needs to know
    // whether to write color and whether to preserve destination alpha.
    // The AFAIL truth table itself is now evaluated once in writePixel,
    // jointly with the depth-write decision (SPEC 01), since AFAIL governs
    // both.
    struct AlphaTestResult
    {
        bool writeFramebuffer;
        bool preserveDestinationAlpha;
    };

    struct TextureCombineResult
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    TextureCombineResult combineTexture(const GSTex0Reg &tex,
                                        uint8_t vr,
                                        uint8_t vg,
                                        uint8_t vb,
                                        uint8_t va,
                                        uint8_t tr,
                                        uint8_t tg,
                                        uint8_t tb,
                                        uint8_t ta)
    {
        const bool textureHasAlpha = tex.tcc != 0u;
        TextureCombineResult out{tr, tg, tb, textureHasAlpha ? ta : va};

        switch (tex.tfx)
        {
        case 0: // MODULATE
            out.r = clampU8((tr * vr) >> 7);
            out.g = clampU8((tg * vg) >> 7);
            out.b = clampU8((tb * vb) >> 7);
            out.a = textureHasAlpha ? clampU8((ta * va) >> 7) : va;
            break;
        case 1: // DECAL
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        case 2: // HIGHLIGHT
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? clampU8(ta + va) : va;
            break;
        case 3: // HIGHLIGHT2
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? ta : va;
            break;
        default:
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        }

        return out;
    }

    uint32_t swizzleClutIndexCSM1(uint32_t index)
    {
        return (index & 0xE7u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
    }

    uint32_t resolveClutIndex(uint8_t index, uint8_t csm, uint8_t csa, uint8_t sourcePsm)
    {
        uint32_t clutIndex = static_cast<uint32_t>(index);

        // T4HL/T4HH are the "high nibble" 4bpp indexed formats packed into a
        // CT32-addressed word; they use the same 16-entry CLUT swizzle/CSA
        // rules as plain T4. T8H is the "high byte" 8bpp indexed format and
        // uses the same 256-entry CLUT rules as plain T8.
        if (sourcePsm == GS_PSM_T4 || sourcePsm == GS_PSM_T4HL || sourcePsm == GS_PSM_T4HH)
        {
            clutIndex = (static_cast<uint32_t>(csa) << 4u) | (clutIndex & 0x0Fu);
            if (csm == 0u)
                clutIndex = swizzleClutIndexCSM1(clutIndex);
        }
        else if ((sourcePsm == GS_PSM_T8 || sourcePsm == GS_PSM_T8H) && csm == 0u)
        {
            clutIndex = swizzleClutIndexCSM1(clutIndex);
        }

        return clutIndex;
    }

    bool tex1UsesLinearFilter(uint64_t tex1)
    {
        const uint8_t mmag = static_cast<uint8_t>((tex1 >> 5) & 0x1u);
        const uint8_t mmin = static_cast<uint8_t>((tex1 >> 6) & 0x7u);
        return mmag != 0u || mmin == 1u || (mmin & 0x4u) != 0u;
    }

    uint8_t lerpChannel(uint8_t c00, uint8_t c10, uint8_t c01, uint8_t c11, float fx, float fy)
    {
        const float top = static_cast<float>(c00) + (static_cast<float>(c10) - static_cast<float>(c00)) * fx;
        const float bottom = static_cast<float>(c01) + (static_cast<float>(c11) - static_cast<float>(c01)) * fx;
        return clampU8(static_cast<int>(std::lround(top + (bottom - top) * fy)));
    }
}

void GSRasterizer::drawPrimitive(GS *gs)
{
    const auto &ctx = gs->activeContext();

    // ---------------------------------------------------------------------
    // FMV-visibility RCA (2026-07-25, DQ8 opening movie TOPO.MVI is black).
    // Decoded MPEG frames land in VRAM as 16x16 CT32 tiles in a 512-wide
    // surface (title: BITBLTBUF.DBP 0x2a00, field: 0x2e00). The open question
    // is whether the game ever COMPOSITES that surface -- i.e. binds it as
    // TEX0 on a primitive -- and if so into which FRAME.FBP. This probe:
    //   * counts every primitive drawn while a movie is streaming, and
    //   * logs, in full, each primitive that samples the movie surface.
    // Both the working title/logo movie and the black field movie go through
    // it, so the two can be diffed from one boot. Env-gated (DQ8_MOVIEQUAD),
    // bounded, and completely inert otherwise.
    {
        static const bool s_on = []() {
            const char *e = std::getenv("DQ8_MOVIEQUAD");
            return e && *e && *e != '0';
        }();
        if (s_on)
        {
            const uint64_t lastUpload = g_dq8MovieUploadMs.load(std::memory_order_relaxed);
            const uint32_t movieDbp = g_dq8MovieUploadDbp.load(std::memory_order_relaxed);
            // TEX0.TBP0 and BITBLTBUF.DBP share the same 256-byte block unit.
            const bool samplesMovie =
                gs->m_prim.tme && movieDbp != 0u && ctx.tex0.tbp0 == movieDbp;
            if (lastUpload != 0ull)
            {
                g_dq8MoviePrimsTotal.fetch_add(1, std::memory_order_relaxed);
                if (samplesMovie)
                    g_dq8MoviePrimsTextured.fetch_add(1, std::memory_order_relaxed);
            }

            if (samplesMovie)
            {
                static std::atomic<uint64_t> s_hits{0};
                const uint64_t h = s_hits.fetch_add(1, std::memory_order_relaxed);
                if (h < 300u || (h % 600u) == 0u)
                {
                    const int ofx = ctx.xyoffset.ofx >> 4;
                    const int ofy = ctx.xyoffset.ofy >> 4;
                    const int nv = (gs->m_prim.type == GS_PRIM_SPRITE ||
                                    gs->m_prim.type == GS_PRIM_LINE ||
                                    gs->m_prim.type == GS_PRIM_LINESTRIP)
                                       ? 2
                                       : 3;
                    float minx = 1e9f, maxx = -1e9f, miny = 1e9f, maxy = -1e9f;
                    for (int i = 0; i < nv; ++i)
                    {
                        const GSVertex &v = gs->m_vtxQueue[i];
                        minx = std::min(minx, v.x - static_cast<float>(ofx));
                        maxx = std::max(maxx, v.x - static_cast<float>(ofx));
                        miny = std::min(miny, v.y - static_cast<float>(ofy));
                        maxy = std::max(maxy, v.y - static_cast<float>(ofy));
                    }
                    std::cout << "[dq8:moviequad] #" << h
                              << std::hex
                              << " tbp0=0x" << ctx.tex0.tbp0
                              << " tpsm=0x" << static_cast<uint32_t>(ctx.tex0.psm)
                              << std::dec
                              << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                              << " tw=" << static_cast<uint32_t>(ctx.tex0.tw)
                              << " th=" << static_cast<uint32_t>(ctx.tex0.th)
                              << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                              << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                              << std::hex
                              << " -> fbp=0x" << ctx.frame.fbp
                              << " fpsm=0x" << static_cast<uint32_t>(ctx.frame.psm)
                              << " fbmsk=0x" << ctx.frame.fbmsk
                              << " alpha=0x" << ctx.alpha
                              << " test=0x" << ctx.test
                              << std::dec
                              << " fbw=" << static_cast<uint32_t>(ctx.frame.fbw)
                              << " prim=" << static_cast<uint32_t>(gs->m_prim.type)
                              << " abe=" << (gs->m_prim.abe ? 1 : 0)
                              << " xy=(" << static_cast<int>(minx) << "," << static_cast<int>(miny)
                              << ")-(" << static_cast<int>(maxx) << "," << static_cast<int>(maxy) << ")"
                              << " scis=(" << ctx.scissor.x0 << "," << ctx.scissor.y0
                              << ")-(" << ctx.scissor.x1 << "," << ctx.scissor.y1 << ")"
                              << std::endl;
                }
            }

        }
    }

    // Bounded RCA draw-trace (env-gated, DQ8 black-field investigation).
    // Emits an ordered draw list for the two candidate FIELD targets
    // (fbp 0x0 = scene composite, 0x70 = CT24 present target) during a
    // wall-clock window, so we can see prim/tme/tex/xy-extent/blend/scissor
    // AND the depth state the game PROGRAMS (ZTE/ZTST/ZMSK) even though the
    // rasterizer does not honour it. No-op unless DQ8_DRAW_TRACE is set.
    {
        static const char *s_trace = std::getenv("DQ8_DRAW_TRACE");
        if (s_trace && *s_trace &&
            (ctx.frame.fbp == 0x0u || ctx.frame.fbp == 0x70u))
        {
            static const uint64_t s_afterMs = []() {
                const char *e = std::getenv("DQ8_DRAW_TRACE_AFTER_MS");
                return e ? static_cast<uint64_t>(std::strtoull(e, nullptr, 10)) : 0ull;
            }();
            static const uint64_t s_windowMs = []() {
                const char *e = std::getenv("DQ8_DRAW_TRACE_WINDOW_MS");
                return e ? static_cast<uint64_t>(std::strtoull(e, nullptr, 10)) : 1500ull;
            }();
            static const auto s_t0 = std::chrono::steady_clock::now();
            const uint64_t elapsedMs = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - s_t0)
                    .count());

            static std::atomic<uint64_t> s_lines{0};
            if (elapsedMs >= s_afterMs && elapsedMs < s_afterMs + s_windowMs &&
                s_lines.fetch_add(1, std::memory_order_relaxed) < 16000u)
            {
                const uint64_t test = ctx.test;
                const uint32_t ate = static_cast<uint32_t>(test & 1u);
                const uint32_t atst = static_cast<uint32_t>((test >> 1) & 7u);
                const uint32_t zte = static_cast<uint32_t>((test >> 16) & 1u);
                const uint32_t ztst = static_cast<uint32_t>((test >> 17) & 3u);
                const uint64_t zbuf = ctx.zbuf;
                const uint32_t zbp = static_cast<uint32_t>(zbuf & 0x1FFu);
                const uint32_t zpsm = static_cast<uint32_t>((zbuf >> 24) & 0xFu);
                const uint32_t zmsk = static_cast<uint32_t>((zbuf >> 32) & 1u);

                const int ofx = ctx.xyoffset.ofx >> 4;
                const int ofy = ctx.xyoffset.ofy >> 4;
                float minx = 1e9f, maxx = -1e9f, miny = 1e9f, maxy = -1e9f;
                float minz = 1e30f, maxz = -1e30f;
                const int nv = (gs->m_prim.type == GS_PRIM_SPRITE ||
                                gs->m_prim.type == GS_PRIM_LINE ||
                                gs->m_prim.type == GS_PRIM_LINESTRIP)
                                   ? 2
                                   : 3;
                for (int i = 0; i < nv; ++i)
                {
                    const GSVertex &v = gs->m_vtxQueue[i];
                    const float vx = v.x - static_cast<float>(ofx);
                    const float vy = v.y - static_cast<float>(ofy);
                    minx = std::min(minx, vx); maxx = std::max(maxx, vx);
                    miny = std::min(miny, vy); maxy = std::max(maxy, vy);
                    minz = std::min(minz, v.z); maxz = std::max(maxz, v.z);
                }

                std::cout << "[gs:drawtrace] fbp=0x" << std::hex << ctx.frame.fbp
                          << " fpsm=0x" << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                          << " prim=" << static_cast<uint32_t>(gs->m_prim.type)
                          << " tme=" << (gs->m_prim.tme ? 1 : 0)
                          << " abe=" << (gs->m_prim.abe ? 1 : 0)
                          << std::hex
                          << " alpha=0x" << ctx.alpha
                          << " tbp=0x" << ctx.tex0.tbp0
                          << " tbw=" << std::dec << static_cast<uint32_t>(ctx.tex0.tbw)
                          << " tpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                          << " ate=" << ate << " atst=" << atst
                          << " zte=" << zte << " ztst=" << ztst
                          << " zbp=0x" << std::hex << zbp << std::dec
                          << " zpsm=0x" << std::hex << zpsm << std::dec << " zmsk=" << zmsk
                          << " xy=(" << static_cast<int>(minx) << "," << static_cast<int>(miny)
                          << ")-(" << static_cast<int>(maxx) << "," << static_cast<int>(maxy) << ")"
                          << " z=[" << static_cast<long long>(minz) << "," << static_cast<long long>(maxz) << "]"
                          << " rgba=(" << static_cast<uint32_t>(gs->m_vtxQueue[nv - 1].r) << ","
                          << static_cast<uint32_t>(gs->m_vtxQueue[nv - 1].g) << ","
                          << static_cast<uint32_t>(gs->m_vtxQueue[nv - 1].b) << ","
                          << static_cast<uint32_t>(gs->m_vtxQueue[nv - 1].a) << ")"
                          << " scis=(" << ctx.scissor.x0 << "," << ctx.scissor.y0
                          << ")-(" << ctx.scissor.x1 << "," << ctx.scissor.y1 << ")"
                          << std::endl;
            }
        }
    }

    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t primitiveIndex = s_debugPrimitiveCount.fetch_add(1u, std::memory_order_relaxed);
        if (primitiveIndex < 64u)
        {
            std::cout << "[gs:prim] idx=" << primitiveIndex
                      << " type=" << static_cast<uint32_t>(gs->m_prim.type)
                      << " tme=" << static_cast<uint32_t>(gs->m_prim.tme)
                      << " abe=" << static_cast<uint32_t>(gs->m_prim.abe)
                      << " fst=" << static_cast<uint32_t>(gs->m_prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(gs->m_prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tw=" << static_cast<uint32_t>(ctx.tex0.tw)
                      << " th=" << static_cast<uint32_t>(ctx.tex0.th)
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(gs->m_texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(gs->m_texclut.cou)
                      << " cov=" << gs->m_texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec
                      << " v0=(" << gs->m_vtxQueue[0].x << "," << gs->m_vtxQueue[0].y << ")"
                      << " uv0=(" << (gs->m_vtxQueue[0].u >> 4) << "," << (gs->m_vtxQueue[0].v >> 4) << ")"
                      << " stq0=(" << gs->m_vtxQueue[0].s << "," << gs->m_vtxQueue[0].t << "," << gs->m_vtxQueue[0].q << ")"
                      << " v1=(" << gs->m_vtxQueue[1].x << "," << gs->m_vtxQueue[1].y << ")"
                      << " uv1=(" << (gs->m_vtxQueue[1].u >> 4) << "," << (gs->m_vtxQueue[1].v >> 4) << ")"
                      << " stq1=(" << gs->m_vtxQueue[1].s << "," << gs->m_vtxQueue[1].t << "," << gs->m_vtxQueue[1].q << ")"
                      << " v2=(" << gs->m_vtxQueue[2].x << "," << gs->m_vtxQueue[2].y << ")"
                      << " uv2=(" << (gs->m_vtxQueue[2].u >> 4) << "," << (gs->m_vtxQueue[2].v >> 4) << ")"
                      << " stq2=(" << gs->m_vtxQueue[2].s << "," << gs->m_vtxQueue[2].t << "," << gs->m_vtxQueue[2].q << ")"
                      << " rgba0=(" << static_cast<uint32_t>(gs->m_vtxQueue[0].r) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[0].g) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[0].b) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[0].a) << ")"
                      << " rgba1=(" << static_cast<uint32_t>(gs->m_vtxQueue[1].r) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[1].g) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[1].b) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[1].a) << ")"
                      << " rgba2=(" << static_cast<uint32_t>(gs->m_vtxQueue[2].r) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[2].g) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[2].b) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[2].a) << ")"
                      << std::endl;
        }
    });

    PS2_IF_AGRESSIVE_LOGS({
        if ((gs->m_prim.ctxt != 0u || ctx.frame.fbp == 150u) &&
            s_debugContext1PrimitiveCount.fetch_add(1u, std::memory_order_relaxed) < 32u)
        {
            std::cout << "[gs:copy-prim]"
                      << " type=" << static_cast<uint32_t>(gs->m_prim.type)
                      << " tme=" << static_cast<uint32_t>(gs->m_prim.tme)
                      << " abe=" << static_cast<uint32_t>(gs->m_prim.abe)
                      << " fst=" << static_cast<uint32_t>(gs->m_prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(gs->m_prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(gs->m_texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(gs->m_texclut.cou)
                      << " cov=" << gs->m_texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec << std::endl;
        }
    });

    if (gs->m_hasPreferredDisplaySource && ctx.frame.fbp == gs->m_preferredDisplayDestFbp)
    {
        gs->m_hasPreferredDisplaySource = false;
    }

    switch (gs->m_prim.type)
    {
    case GS_PRIM_SPRITE:
        drawSprite(gs);
        break;
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
        drawTriangle(gs);
        break;
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        drawLine(gs);
        break;
    case GS_PRIM_POINT:
    {
        const GSVertex &v = gs->m_vtxQueue[0];
        const auto &ctx = gs->activeContext();
        int px = static_cast<int>(v.x) - (ctx.xyoffset.ofx >> 4);
        int py = static_cast<int>(v.y) - (ctx.xyoffset.ofy >> 4);
        writePixel(gs, px, py, v.r, v.g, v.b, v.a, v.z);
        break;
    }
    default:
        break;
    }
}

namespace
{
    // SPEC 01 (dq8/reference/dc2-learnings/01-depth-testing-cpu-rasterizer.md)
    // kill switch: PS2X_DEPTH_TEST=0 disables the depth test/write path so a
    // regression can be bisected without a rebuild. Depth testing is ON by
    // DEFAULT (unset, empty, or any value other than exactly "0"), per the
    // spec's explicit "test the VALUE not mere presence" instruction -- this
    // codebase has other env flags that treat getenv()!=nullptr as "on",
    // which would make PS2X_DEPTH_TEST=0 silently stay enabled. Do not copy
    // that pattern here.
    bool depthTestKilled()
    {
        static const bool s_killed = []() {
            const char *e = std::getenv("PS2X_DEPTH_TEST");
            return e != nullptr && std::strcmp(e, "0") == 0;
        }();
        return s_killed;
    }

    // Clamp+truncate an interpolated Z (guest range up to unsigned 32-bit)
    // down to a plain uint32_t without invoking UB on negative/huge floats.
    uint32_t zFloatToU32(float zf)
    {
        if (zf <= 0.0f)
            return 0u;
        if (zf >= 4294967040.0f) // largest float exactly < 2^32
            return 0xFFFFFFFFu;
        return static_cast<uint32_t>(zf);
    }

    // ZBUF.PSM (bits 24-27 of the ZBUF register) is a raw 4-bit hardware
    // field with its OWN encoding -- 0=PSMZ32, 1=PSMZ24, 2=PSMZ16, 0xA=PSMZ16S
    // -- distinct from the 6-bit PSM codes FRAME/TEX0 use (GS_PSM_Z32=48 etc,
    // ps2_gs_gpu.h). Do not compare it against the GS_PSM_Z* enum constants;
    // switch on the raw value directly (spec 01 §2, "parse it per-draw").
    struct ZFormat
    {
        uint32_t bytesPerPixel;
        uint32_t mask;
    };

    ZFormat zFormatFor(uint32_t zpsmRaw)
    {
        switch (zpsmRaw)
        {
        case 2u:  // PSMZ16
        case 0xAu: // PSMZ16S
            return {2u, 0xFFFFu};
        case 1u: // PSMZ24 -- low 24 bits are Z, top byte reserved
            return {4u, 0x00FFFFFFu};
        case 0u: // PSMZ32
        default:
            return {4u, 0xFFFFFFFFu};
        }
    }

    uint32_t zAddrFor(uint32_t zpsmRaw, uint32_t block, uint32_t width, uint32_t x, uint32_t y)
    {
        switch (zpsmRaw)
        {
        case 2u:
            return GSPSMCT16::addrPSMZ16(block, width, x, y);
        case 0xAu:
            return GSPSMCT16::addrPSMZ16S(block, width, x, y);
        case 0u:
        case 1u:
        default:
            return GSPSMCT32::addrPSMCT32(block, width, x, y);
        }
    }

    bool depthComparePasses(uint32_t ztst, uint32_t newZ, uint32_t storedZ)
    {
        switch (ztst)
        {
        case 0: // NEVER
            return false;
        case 1: // ALWAYS
            return true;
        case 2: // GEQUAL -- PS2: larger Z is NEARER
            return newZ >= storedZ;
        case 3: // GREATER
            return newZ > storedZ;
        default:
            return true;
        }
    }

    // SPEC 03 (dq8/reference/dc2-learnings/03-nearplane-clip-homogeneous.md)
    // kill switch: PS2X_NEARPLANE_CLIP=0 disables near-plane/guard-band
    // triangle rejection. ON by DEFAULT (unset, empty, or any value other
    // than exactly "0"), per the spec's "test the VALUE not mere presence"
    // instruction (mirrors PS2X_DEPTH_TEST above) -- do not regress to a
    // getenv()!=nullptr presence check here.
    bool nearplaneClipDisabled()
    {
        static const bool s_disabled = []() {
            const char *e = std::getenv("PS2X_NEARPLANE_CLIP");
            return e != nullptr && std::strcmp(e, "0") == 0;
        }();
        return s_disabled;
    }

    // GSVertex.x/y/z reach the rasterizer already perspective-divided by the
    // upstream VU transform (real GS hardware has no vertex-position clip
    // pipeline of its own -- XYZ2/XYZ3 GIF registers are final screen-space
    // coordinates). GSVertex.q is the reciprocal-W that rode along with the
    // vertex purely for ST perspective-correct texture interpolation, but it
    // doubles as the only signal this rasterizer has for "how close to (or
    // behind) the camera was this vertex before the divide."
    //
    // DC2 (G101/G125/G126/G128, same Level-5 engine) found that once a
    // vertex is behind or extremely near the camera, its already-divided
    // screen X/Y has typically already been FTOI4-quantized/saturated by
    // the time it reaches the rasterizer -- so reconstructing clip-space
    // (clip = screen * W) and intersecting the near plane from that data
    // intersects from garbage endpoints (G128). Their final, shipped,
    // robust choice was NOT to attempt a precise per-edge clip: drop the
    // whole triangle when any vertex's q is at/behind the near-plane
    // threshold, rather than clip from corrupt data.
    //
    // We follow that same robust-over-precise choice. Note we only apply
    // G126's LOWER bound (q > qmin); the UPPER bound in DC2's checklist
    // (q <= 1/wNear) needs the camera's near-plane W, which is a VU/camera
    // constant this GS-level rasterizer has no visibility into -- omitted
    // deliberately rather than guessed.
    constexpr float kNearPlaneQMin = 1.0e-6f;

    // Real GS hardware has an implicit guard band: primitives far outside
    // the scissor are clipped/culled before they ever reach the pixel
    // pipeline. This software rasterizer has no such stage, so a triangle
    // with a vertex projected wildly off-screen (near-plane singularity,
    // missing strip-restart, or any other upstream corruption) paints a
    // full-width streak instead of vanishing the way HW would. DC2 (G89)
    // used a tunable margin outside the scissor, default 512px; we mirror
    // that default.
    constexpr float kGuardBandMarginPx = 512.0f;

    bool triangleNeedsNearOrGuardBandCull(float fx0, float fy0, float fx1, float fy1,
                                          float fx2, float fy2, float q0, float q1, float q2,
                                          const GSScissorReg &scissor)
    {
        // Near-plane: any vertex at/behind the camera -> drop (G101/G128).
        if (q0 <= kNearPlaneQMin || q1 <= kNearPlaneQMin || q2 <= kNearPlaneQMin)
            return true;

        // Guard-band: any vertex far outside the scissor -> drop (G89).
        const float loX = static_cast<float>(scissor.x0) - kGuardBandMarginPx;
        const float hiX = static_cast<float>(scissor.x1) + kGuardBandMarginPx;
        const float loY = static_cast<float>(scissor.y0) - kGuardBandMarginPx;
        const float hiY = static_cast<float>(scissor.y1) + kGuardBandMarginPx;

        if (fx0 < loX || fx0 > hiX || fy0 < loY || fy0 > hiY)
            return true;
        if (fx1 < loX || fx1 > hiX || fy1 < loY || fy1 > hiY)
            return true;
        if (fx2 < loX || fx2 > hiX || fy2 < loY || fy2 > hiY)
            return true;

        return false;
    }
}

void GSRasterizer::writePixel(GS *gs, int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a, float zf)
{
    const auto &ctx = gs->activeContext();
    if (x < ctx.scissor.x0 || x > ctx.scissor.x1 ||
        y < ctx.scissor.y0 || y > ctx.scissor.y1)
        return;

    // ---------------------------------------------------------------------
    // Alpha test -> outcome. Whether the depth test even runs, and whether a
    // pixel that fails alpha still forces a Z write, depends on this outcome
    // (TEST.AFAIL truth table) -- SPEC 01 "alpha-test interaction" / DC2 G240.
    //   AFAIL  | write framebuffer? | write depth?
    //   KEEP    no                   no
    //   FB_ONLY yes                  no
    //   ZB_ONLY no                   yes (forced, bypassing compare)
    //   RGB_ONLY rgb only            no
    // ---------------------------------------------------------------------
    const bool alphaPass = passesAlphaTest(ctx.test, a);
    bool writeColor;
    bool preserveDestAlpha = false;
    bool runDepthCompare = false; // normal ZTE/ZTST path (alpha passed)
    bool forceDepthWrite = false; // AFAIL=ZB_ONLY on alpha fail

    if (alphaPass)
    {
        writeColor = true;
        runDepthCompare = true;
    }
    else
    {
        switch (static_cast<uint8_t>((ctx.test >> 12) & 0x3u))
        {
        case 1: // FB_ONLY
            writeColor = true;
            break;
        case 3: // RGB_ONLY
            writeColor = true;
            preserveDestAlpha = true;
            break;
        case 2: // ZB_ONLY
            writeColor = false;
            forceDepthWrite = true;
            break;
        case 0: // KEEP
        default:
            writeColor = false;
            break;
        }
    }

    // ---------------------------------------------------------------------
    // Universal depth test/write (SPEC 01) -- honors the guest's real
    // TEST.ZTE/ZTST and ZBUF.ZBP/PSM/ZMSK against real VRAM for EVERY draw,
    // no per-screen scoping (DC2 G203: scoping is what broke other screens).
    // ---------------------------------------------------------------------
    const uint64_t zbufReg = ctx.zbuf;
    const bool zte = !depthTestKilled() && (((ctx.test >> 16) & 1u) != 0u);

    if (zte && (runDepthCompare || forceDepthWrite))
    {
        const uint32_t ztst = static_cast<uint32_t>((ctx.test >> 17) & 3u);
        const uint32_t zbp = static_cast<uint32_t>(zbufReg & 0x1FFu);
        const uint32_t zpsmRaw = static_cast<uint32_t>((zbufReg >> 24) & 0xFu);
        const bool zWritesEnabled = ((zbufReg >> 32) & 1u) == 0u; // ZMSK==0 => enabled

        const ZFormat zfmt = zFormatFor(zpsmRaw);
        const uint32_t zBlock = GSInternal::framePageBaseToBlock(zbp);
        // ZBUF has no width register of its own; the Z surface shares FRAME's
        // width, same convention DC2 used (spec 01 §4.2).
        const uint32_t zWidthBlocks = (ctx.frame.fbw != 0u) ? ctx.frame.fbw : 1u;
        const uint32_t zOff = zAddrFor(zpsmRaw, zBlock, zWidthBlocks, static_cast<uint32_t>(x), static_cast<uint32_t>(y));

        if (zOff + zfmt.bytesPerPixel <= gs->m_vramSize)
        {
            const uint32_t newZ = zFloatToU32(zf) & zfmt.mask;
            bool depthPass = true;

            if (runDepthCompare)
            {
                uint32_t storedZ = 0u;
                if (zfmt.bytesPerPixel == 2u)
                {
                    uint16_t packed = 0u;
                    std::memcpy(&packed, gs->m_vram + zOff, 2);
                    storedZ = packed;
                }
                else
                {
                    std::memcpy(&storedZ, gs->m_vram + zOff, 4);
                    storedZ &= zfmt.mask;
                }

                depthPass = depthComparePasses(ztst, newZ, storedZ);
                if (!depthPass)
                    writeColor = false; // depth-test failure discards the fragment
            }

            const bool doZWrite = zWritesEnabled &&
                                   ((runDepthCompare && depthPass) || forceDepthWrite);
            if (doZWrite)
            {
                if (zfmt.bytesPerPixel == 2u)
                {
                    uint16_t packed = static_cast<uint16_t>(newZ);
                    std::memcpy(gs->m_vram + zOff, &packed, 2);
                }
                else if (zpsmRaw == 1u)
                {
                    // PSMZ24: preserve the reserved top byte of the 32-bit word.
                    uint32_t existing = 0u;
                    std::memcpy(&existing, gs->m_vram + zOff, 4);
                    const uint32_t toWrite = (existing & 0xFF000000u) | newZ;
                    std::memcpy(gs->m_vram + zOff, &toWrite, 4);
                }
                else
                {
                    std::memcpy(gs->m_vram + zOff, &newZ, 4);
                }
            }
        }
    }

    if (!writeColor)
        return;

    const AlphaTestResult alphaTest{writeColor, preserveDestAlpha};

    const uint32_t widthBlocks = (ctx.frame.fbw != 0u) ? ctx.frame.fbw : 1u;
    const uint32_t bytesPerPixel =
        (ctx.frame.psm == GS_PSM_CT16 || ctx.frame.psm == GS_PSM_CT16S) ? 2u : 4u;

    uint32_t off = 0u;
    if (ctx.frame.psm == GS_PSM_CT32 || ctx.frame.psm == GS_PSM_CT24)
    {
        off = GSPSMCT32::addrPSMCT32(GSInternal::framePageBaseToBlock(ctx.frame.fbp),
                                     widthBlocks,
                                     static_cast<uint32_t>(x),
                                     static_cast<uint32_t>(y));
    }
    else
    {
        off = addrPSMCT16Family(GSInternal::framePageBaseToBlock(ctx.frame.fbp),
                                widthBlocks,
                                ctx.frame.psm,
                                static_cast<uint32_t>(x),
                                static_cast<uint32_t>(y));
    }

    if (off + bytesPerPixel > gs->m_vramSize)
        return;

    // Bounded permanent pixel-write sample: proves pixels actually land in
    // local memory (post scissor/alpha-test) and shows color + target. One
    // line per 2M pixel writes.
    {
        static std::atomic<uint64_t> s_pixelWrites{0};
        const uint64_t n = s_pixelWrites.fetch_add(1, std::memory_order_relaxed);
        if ((n % 2000000u) == 0u)
        {
            std::cout << "[gs:pixels] total=" << n
                      << " xy=(" << x << "," << y << ")"
                      << " rgba=(" << static_cast<uint32_t>(r) << ","
                      << static_cast<uint32_t>(g) << ","
                      << static_cast<uint32_t>(b) << ","
                      << static_cast<uint32_t>(a) << ")"
                      << std::hex
                      << " fbp=0x" << ctx.frame.fbp
                      << " psm=0x" << static_cast<uint32_t>(ctx.frame.psm)
                      << " tme=" << (gs->m_prim.tme ? 1 : 0)
                      << std::dec << std::endl;
        }
    }

    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t pixelIndex = s_debugPixelCount.fetch_add(1, std::memory_order_relaxed);
        if (pixelIndex < 32u)
        {
            std::cout << "[gs:pixel] idx=" << pixelIndex
                      << " xy=(" << x << "," << y << ")"
                      << " rgba=(" << static_cast<uint32_t>(r) << ","
                      << static_cast<uint32_t>(g) << ","
                      << static_cast<uint32_t>(b) << ","
                      << static_cast<uint32_t>(a) << ")"
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " off=0x" << std::hex << off << std::dec
                      << std::endl;
        }
    });

    PS2_IF_AGRESSIVE_LOGS({
        if (ctx.frame.fbp == 150u &&
            s_debugFbp150PixelCount.fetch_add(1u, std::memory_order_relaxed) < 32u)
        {
            std::cout << "[gs:fbp150-pixel]"
                      << " xy=(" << x << "," << y << ")"
                      << " rgba=(" << static_cast<uint32_t>(r) << ","
                      << static_cast<uint32_t>(g) << ","
                      << static_cast<uint32_t>(b) << ","
                      << static_cast<uint32_t>(a) << ")"
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " off=0x" << std::hex << off << std::dec << std::endl;
        }
    });

    const uint8_t srcR = r;
    const uint8_t srcG = g;
    const uint8_t srcB = b;

    if (gs->m_prim.abe)
    {
        uint32_t existing = 0u;
        if (bytesPerPixel == 2u)
        {
            uint16_t packed = 0u;
            std::memcpy(&packed, gs->m_vram + off, 2);
            existing = decodePSMCT16(packed);
        }
        else
        {
            std::memcpy(&existing, gs->m_vram + off, 4);
        }
        uint8_t dr = existing & 0xFF;
        uint8_t dg = (existing >> 8) & 0xFF;
        uint8_t db = (existing >> 16) & 0xFF;
        uint8_t da = (existing >> 24) & 0xFF;

        // PABE disables alpha blending when the source alpha MSB is clear.
        if (!(gs->m_pabe && (a & 0x80u) == 0u))
        {
            uint64_t alphaReg = ctx.alpha;
            uint8_t asel = alphaReg & 3;
            uint8_t bsel = (alphaReg >> 2) & 3;
            uint8_t csel = (alphaReg >> 4) & 3;
            uint8_t dsel = (alphaReg >> 6) & 3;
            uint8_t fix = static_cast<uint8_t>((alphaReg >> 32) & 0xFF);

            auto pickRGB = [&](uint8_t sel, int cs, int cd) -> int
            {
                if (sel == 0)
                    return cs;
                if (sel == 1)
                    return cd;
                return 0;
            };
            int cAlpha = (csel == 0) ? a : (csel == 1) ? da
                                                       : fix;

            r = clampU8(((pickRGB(asel, r, dr) - pickRGB(bsel, r, dr)) * cAlpha >> 7) + pickRGB(dsel, r, dr));
            g = clampU8(((pickRGB(asel, g, dg) - pickRGB(bsel, g, dg)) * cAlpha >> 7) + pickRGB(dsel, g, dg));
            b = clampU8(((pickRGB(asel, b, db) - pickRGB(bsel, b, db)) * cAlpha >> 7) + pickRGB(dsel, b, db));
        }
        else
        {
            r = srcR;
            g = srcG;
            b = srcB;
        }
    }

    uint32_t mask = ctx.frame.fbmsk;
    if (!alphaTest.preserveDestinationAlpha &&
        (ctx.fba & 0x1ull) != 0ull &&
        ctx.frame.psm != GS_PSM_CT24)
    {
        a = static_cast<uint8_t>(a | 0x80u);
    }

    if (bytesPerPixel == 2u)
    {
        uint16_t pixel = encodePSMCT16(r, g, b, a);
        if ((mask & 0xFFFFu) != 0u)
        {
            uint16_t existing = 0u;
            std::memcpy(&existing, gs->m_vram + off, 2);
            pixel = static_cast<uint16_t>((pixel & ~mask) | (existing & mask));
        }
        std::memcpy(gs->m_vram + off, &pixel, 2);
        return;
    }

    uint32_t pixel = static_cast<uint32_t>(r) | (static_cast<uint32_t>(g) << 8) | (static_cast<uint32_t>(b) << 16) | (static_cast<uint32_t>(a) << 24);

    if (mask != 0)
    {
        uint32_t existing;
        std::memcpy(&existing, gs->m_vram + off, 4);
        pixel = (pixel & ~mask) | (existing & mask);
    }

    if (alphaTest.preserveDestinationAlpha)
    {
        uint32_t existing = 0u;
        std::memcpy(&existing, gs->m_vram + off, 4);
        pixel = (pixel & 0x00FFFFFFu) | (existing & 0xFF000000u);
    }

    std::memcpy(gs->m_vram + off, &pixel, 4);
}

uint32_t GSRasterizer::readTexelPSMCT32(GS *gs, uint32_t tbp0, uint32_t tbw, int texU, int texV)
{
    if (tbw == 0)
        tbw = 1;
    uint32_t off = GSPSMCT32::addrPSMCT32(tbp0, tbw, static_cast<uint32_t>(texU), static_cast<uint32_t>(texV));
    if (off + 4 > gs->m_vramSize)
        return 0xFFFF00FFu;
    uint32_t texel;
    std::memcpy(&texel, gs->m_vram + off, 4);
    return texel;
}

uint32_t GSRasterizer::readTexelPSMCT16(GS *gs, uint32_t tbp0, uint32_t tbw, int texU, int texV)
{
    if (tbw == 0)
        tbw = 1;
    const uint8_t psm = gs->activeContext().tex0.psm;
    uint32_t off = addrPSMCT16Family(tbp0, tbw, psm, static_cast<uint32_t>(texU), static_cast<uint32_t>(texV));
    if (off + 2 > gs->m_vramSize)
        return 0xFFFF00FFu;
    uint16_t texel;
    std::memcpy(&texel, gs->m_vram + off, 2);
    return decodePSMCT16(texel);
}

uint32_t GSRasterizer::readTexelPSMT4(GS *gs, uint32_t tbp0, uint32_t tbw, int texU, int texV)
{
    if (tbw == 0)
        tbw = 1;
    uint32_t nibbleAddr = GSPSMT4::addrPSMT4(tbp0, tbw, static_cast<uint32_t>(texU), static_cast<uint32_t>(texV));
    uint32_t byteOff = nibbleAddr >> 1;
    if (byteOff >= gs->m_vramSize)
        return 0;
    uint8_t packed = gs->m_vram[byteOff];
    uint32_t shift = (nibbleAddr & 1u) << 2;
    uint32_t idx = (packed >> shift) & 0xFu;
    return idx;
}

// Bounded, format-agnostic diagnostic: the first time a given (cbp, cpsm,
// csa) CLUT is actually sampled, dump its first 16 entries as raw bytes.
// This is the cheapest way to answer "is the CLUT region populated when a
// draw first reads it, or is it still zero/stale local memory" without
// hardcoding any title-specific block address.
void GSRasterizer::logClutContentsOnce(GS *gs, uint32_t cbp, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm)
{
    static std::mutex s_mutex;
    static std::vector<uint32_t> s_seenKeys;
    static std::vector<uint32_t> s_lastContentHash;
    static uint32_t s_totalDumps = 0;
    constexpr uint32_t kMaxKeysDefault = 16u;
    constexpr uint32_t kMaxTotalDumpsDefault = 96u;
    static const uint32_t kMaxKeys = localDiagEnvLimit("PS2X_CLUT_DUMP_MAX_KEYS", kMaxKeysDefault);
    static const uint32_t kMaxTotalDumps = localDiagEnvLimit("PS2X_CLUT_DUMP_MAX_LOGS", kMaxTotalDumpsDefault);
    static std::atomic<bool> s_seenKeysTruncated{false};
    static std::atomic<bool> s_totalDumpsTruncated{false};

    const uint32_t key = (cbp << 8) | (static_cast<uint32_t>(cpsm) << 4) | static_cast<uint32_t>(csa & 0xFu);

    const uint32_t clutWidth = (gs->m_texclut.cbw != 0u) ? static_cast<uint32_t>(gs->m_texclut.cbw) : 1u;

    // M-T1 diag (2026-07-18): a fixed VRAM block (cbp) gets reprogrammed as
    // the CLUT for many different, unrelated screens over a boot (address
    // reuse is normal). The original "first sample of each key" dump only
    // ever shows whichever asset happened to claim that key FIRST -- for a
    // block reused later in the boot by a different screen, the dump is
    // stale and misleading (PS2_PROJECT_STATE.md §3.26/§3.27 already found
    // one such logging artifact). Re-dump (bounded, permanent, generic; no
    // game constants) whenever the actual 16-entry content differs from the
    // last dump for the same key, so a late-boot screen's real CLUT state is
    // observable too, not just whatever used the block first.
    uint32_t quickHash = 2166136261u;
    {
        for (uint32_t i = 0; i < 16u; ++i)
        {
            const uint32_t swizzled = (csm == 0u) ? swizzleClutIndexCSM1(i) : i;
            const uint32_t x = swizzled & 0x0Fu;
            const uint32_t y = swizzled >> 4;
            uint32_t rgba = 0u;
            if (cpsm == GS_PSM_CT32 || cpsm == GS_PSM_CT24)
            {
                const uint32_t off = GSPSMCT32::addrPSMCT32(cbp, clutWidth, x, y);
                if (off + 4u <= gs->m_vramSize)
                    std::memcpy(&rgba, gs->m_vram + off, 4);
            }
            else if (cpsm == GS_PSM_CT16 || cpsm == GS_PSM_CT16S)
            {
                const uint32_t off = addrPSMCT16Family(cbp, clutWidth, cpsm, x, y);
                if (off + 2u <= gs->m_vramSize)
                {
                    uint16_t c16 = 0u;
                    std::memcpy(&c16, gs->m_vram + off, 2);
                    rgba = decodePSMCT16(c16);
                }
            }
            quickHash = (quickHash ^ rgba) * 16777619u;
        }
    }

    std::lock_guard<std::mutex> lock(s_mutex);
    auto it = std::find(s_seenKeys.begin(), s_seenKeys.end(), key);
    if (it == s_seenKeys.end())
    {
        if (!localDiagLogBudget(std::cout, "[gs:clut-dump:keys]", "PS2X_CLUT_DUMP_MAX_KEYS",
                                kMaxKeys, static_cast<uint32_t>(s_seenKeys.size()), s_seenKeysTruncated))
            return;
        if (!localDiagLogBudget(std::cout, "[gs:clut-dump]", "PS2X_CLUT_DUMP_MAX_LOGS",
                                kMaxTotalDumps, s_totalDumps, s_totalDumpsTruncated))
            return;
        s_seenKeys.push_back(key);
        s_lastContentHash.push_back(quickHash);
    }
    else
    {
        const size_t idx = static_cast<size_t>(it - s_seenKeys.begin());
        if (s_lastContentHash[idx] == quickHash)
            return;
        if (!localDiagLogBudget(std::cout, "[gs:clut-dump]", "PS2X_CLUT_DUMP_MAX_LOGS",
                                kMaxTotalDumps, s_totalDumps, s_totalDumpsTruncated))
            return;
        s_lastContentHash[idx] = quickHash;
    }
    ++s_totalDumps;
    std::cout << "[gs:clut-dump] cbp=0x" << std::hex << cbp
              << " cpsm=0x" << static_cast<uint32_t>(cpsm)
              << " csm=" << std::dec << static_cast<uint32_t>(csm)
              << " csa=" << static_cast<uint32_t>(csa)
              << " sourcePsm=0x" << std::hex << static_cast<uint32_t>(sourcePsm) << std::dec
              << " cbw=" << clutWidth
              << " cou=" << static_cast<uint32_t>(gs->m_texclut.cou)
              << " cov=" << gs->m_texclut.cov
              << " entries=[";
    for (uint32_t i = 0; i < 16u; ++i)
    {
        // Real CLUT storage for CSM1 (csm==0) is swizzled the same way
        // resolveClutIndex() swizzles a live lookup index (entries 8-15
        // and 16-23 of the underlying 32-entry block trade places); dump
        // logical entry i through the same swizzle so this diagnostic
        // addresses the same bytes a real sample would. Without this, a
        // 16-entry CSM1 CLUT dump read entries 0-7 correctly but showed
        // entries 8-15 as whatever sits at (x=8..15, y=0) -- 8 entries
        // from the *next* CSA's block, not this one's.
        const uint32_t swizzled = (csm == 0u) ? swizzleClutIndexCSM1(i) : i;
        const uint32_t x = swizzled & 0x0Fu;
        const uint32_t y = swizzled >> 4;
        uint32_t rgba = 0xFFFF00FFu;
        if (cpsm == GS_PSM_CT32 || cpsm == GS_PSM_CT24)
        {
            const uint32_t off = GSPSMCT32::addrPSMCT32(cbp, clutWidth, x, y);
            if (off + 4u <= gs->m_vramSize)
                std::memcpy(&rgba, gs->m_vram + off, 4);
        }
        else if (cpsm == GS_PSM_CT16 || cpsm == GS_PSM_CT16S)
        {
            const uint32_t off = addrPSMCT16Family(cbp, clutWidth, cpsm, x, y);
            if (off + 2u <= gs->m_vramSize)
            {
                uint16_t c16 = 0u;
                std::memcpy(&c16, gs->m_vram + off, 2);
                rgba = decodePSMCT16(c16);
            }
        }
        std::cout << std::hex << "0x" << rgba << std::dec << (i + 1 < 16u ? "," : "");
    }
    std::cout << "]" << std::endl;
}

uint32_t GSRasterizer::lookupCLUT(GS *gs,
                                  uint8_t index,
                                  uint32_t cbp,
                                  uint8_t cpsm,
                                  uint8_t csm,
                                  uint8_t csa,
                                  uint8_t sourcePsm)
{
    logClutContentsOnce(gs, cbp, cpsm, csm, csa, sourcePsm);

    const uint32_t clutIndex = resolveClutIndex(index, csm, csa, sourcePsm);
    const uint32_t clutWidth = (gs->m_texclut.cbw != 0u) ? static_cast<uint32_t>(gs->m_texclut.cbw) : 1u;
    const uint32_t clutX = static_cast<uint32_t>(gs->m_texclut.cou) + (clutIndex & 0x0Fu);
    const uint32_t clutY = static_cast<uint32_t>(gs->m_texclut.cov) + (clutIndex >> 4);

    if (cpsm == GS_PSM_CT32 || cpsm == GS_PSM_CT24)
    {
        const uint32_t off = GSPSMCT32::addrPSMCT32(cbp, clutWidth, clutX, clutY);
        if (off + 4 > gs->m_vramSize)
            return 0xFFFF00FFu;
        uint32_t color;
        std::memcpy(&color, gs->m_vram + off, 4);
        return applyTexa(gs->m_texa, cpsm, color);
    }

    if (cpsm == GS_PSM_CT16 || cpsm == GS_PSM_CT16S)
    {
        uint32_t off = addrPSMCT16Family(cbp, clutWidth, cpsm, clutX, clutY);
        if (off + 2 > gs->m_vramSize)
            return 0xFFFF00FFu;
        uint16_t c16;
        std::memcpy(&c16, gs->m_vram + off, 2);
        return applyTexa(gs->m_texa, cpsm, decodePSMCT16(c16));
    }

    return 0xFFFF00FFu;
}

uint32_t GSRasterizer::sampleTexture(GS *gs, float s, float t, float q, uint16_t u, uint16_t v)
{
    const auto &ctx = gs->activeContext();
    const auto &tex = ctx.tex0;

    int texW = 1 << tex.tw;
    int texH = 1 << tex.th;

    float texUf, texVf;
    if (gs->m_prim.fst)
    {
        texUf = static_cast<float>(u) / 16.0f;
        texVf = static_cast<float>(v) / 16.0f;
    }
    else
    {
        const float invQ = 1.0f / fabsQ(q);
        texUf = s * invQ * static_cast<float>(texW);
        texVf = t * invQ * static_cast<float>(texH);
    }

    auto samplePoint = [&](int sampleU, int sampleV) -> uint32_t
    {
        sampleU = clampInt(sampleU, 0, texW - 1);
        sampleV = clampInt(sampleV, 0, texH - 1);

        if (tex.psm == GS_PSM_CT32 || tex.psm == GS_PSM_CT24)
            return applyTexa(gs->m_texa, tex.psm, readTexelPSMCT32(gs, tex.tbp0, tex.tbw, sampleU, sampleV));

        if (tex.psm == GS_PSM_CT16 || tex.psm == GS_PSM_CT16S)
            return applyTexa(gs->m_texa, tex.psm, readTexelPSMCT16(gs, tex.tbp0, tex.tbw, sampleU, sampleV));

        if (tex.psm == GS_PSM_T4)
        {
            uint32_t idx = readTexelPSMT4(gs, tex.tbp0, tex.tbw, sampleU, sampleV);
            return lookupCLUT(gs, static_cast<uint8_t>(idx), tex.cbp, tex.cpsm, tex.csm, tex.csa, tex.psm);
        }

        if (tex.psm == GS_PSM_T8)
        {
            if (tex.tbw == 0)
                return 0xFFFF00FFu;
            uint32_t off = GSPSMT8::addrPSMT8(tex.tbp0, tex.tbw, static_cast<uint32_t>(sampleU), static_cast<uint32_t>(sampleV));
            if (off >= gs->m_vramSize)
                return 0xFFFF00FFu;
            uint8_t idx = gs->m_vram[off];
            return lookupCLUT(gs, idx, tex.cbp, tex.cpsm, tex.csm, tex.csa, tex.psm);
        }

        // T4HL/T4HH/T8H are indexed formats packed into the alpha byte of a
        // CT32-addressed word: two independent 4bpp planes share one CT32
        // buffer (T4HL in bits 24-27, T4HH in bits 28-31), and T8H is an
        // 8bpp plane in bits 24-31. TBW addresses that shared CT32 buffer
        // directly (no doubling relative to the upload's DBW).
        if (tex.psm == GS_PSM_T4HL || tex.psm == GS_PSM_T4HH || tex.psm == GS_PSM_T8H)
        {
            if (tex.tbw == 0)
                return 0xFFFF00FFu;
            const uint32_t off = GSPSMCT32::addrPSMCT32(tex.tbp0, tex.tbw, static_cast<uint32_t>(sampleU), static_cast<uint32_t>(sampleV));
            if (off + 4u > gs->m_vramSize)
                return 0xFFFF00FFu;
            uint32_t word = 0u;
            std::memcpy(&word, gs->m_vram + off, 4u);

            uint8_t idx;
            if (tex.psm == GS_PSM_T8H)
                idx = static_cast<uint8_t>((word >> 24) & 0xFFu);
            else if (tex.psm == GS_PSM_T4HH)
                idx = static_cast<uint8_t>((word >> 28) & 0x0Fu);
            else // GS_PSM_T4HL
                idx = static_cast<uint8_t>((word >> 24) & 0x0Fu);

            return lookupCLUT(gs, idx, tex.cbp, tex.cpsm, tex.csm, tex.csa, tex.psm);
        }

        return 0xFFFF00FFu;
    };

    if (!tex1UsesLinearFilter(ctx.tex1))
    {
        return samplePoint(static_cast<int>(texUf), static_cast<int>(texVf));
    }

    const float sampleU = texUf - 0.5f;
    const float sampleV = texVf - 0.5f;
    const int u0 = static_cast<int>(std::floor(sampleU));
    const int v0 = static_cast<int>(std::floor(sampleV));
    const int u1 = u0 + 1;
    const int v1 = v0 + 1;
    const float fx = sampleU - static_cast<float>(u0);
    const float fy = sampleV - static_cast<float>(v0);

    const uint32_t c00 = samplePoint(u0, v0);
    const uint32_t c10 = samplePoint(u1, v0);
    const uint32_t c01 = samplePoint(u0, v1);
    const uint32_t c11 = samplePoint(u1, v1);

    const uint8_t r = lerpChannel(static_cast<uint8_t>(c00 & 0xFFu),
                                  static_cast<uint8_t>(c10 & 0xFFu),
                                  static_cast<uint8_t>(c01 & 0xFFu),
                                  static_cast<uint8_t>(c11 & 0xFFu),
                                  fx, fy);
    const uint8_t g = lerpChannel(static_cast<uint8_t>((c00 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 8) & 0xFFu),
                                  fx, fy);
    const uint8_t b = lerpChannel(static_cast<uint8_t>((c00 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 16) & 0xFFu),
                                  fx, fy);
    const uint8_t a = lerpChannel(static_cast<uint8_t>((c00 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 24) & 0xFFu),
                                  fx, fy);

    return static_cast<uint32_t>(r) |
           (static_cast<uint32_t>(g) << 8) |
           (static_cast<uint32_t>(b) << 16) |
           (static_cast<uint32_t>(a) << 24);
}

namespace
{
    // =========================================================================
    // DIAGNOSTIC PROBE ONLY -- NOT A RENDERING FIX (2026-07-27).
    //
    // FIELD's GS receives exactly two draw classes ([gs:rgbaq-hist],
    // [gs:drawstate], 2026-07-27): textured TRIANGLE_STRIP 3D geometry
    // ("hasST", tme=1, ~27% of vertices) and untextured alpha-blended
    // SPRITEs ("noST", tme=0, abe=1) whose RGBAQ writes are flat black
    // (RGB==0) 98.8% of the time, ~73% of vertices. Hypothesis under test:
    // the 3D scene IS being drawn correctly and is then being painted over
    // by this large volume of black sprites -- i.e. the game is stuck in a
    // fade-out that never completes. This probe, gated by
    // PS2X_SUPPRESS_BLACK_SPRITES=1 (default OFF), skips exactly the
    // primitives that match the hypothesis (PRIM=SPRITE, TME=0, vertex
    // colour RGB==0) so a human can look at the frame underneath and
    // confirm or refute it.
    //
    // This is a probe, not a fix: it throws away real draw calls
    // unconditionally rather than diagnosing *why* the fade / blend state
    // never resolves, and it must never be treated as a candidate for the
    // real render-correctness path. Zero cost when the env var is unset
    // (one relaxed bool check per sprite).
    bool blackSpriteSuppressOn()
    {
        static const bool s_on = []() {
            const char *e = std::getenv("PS2X_SUPPRESS_BLACK_SPRITES");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        return s_on;
    }

    std::atomic<uint64_t> g_blackSpriteSuppressedCount{0};
    std::atomic<uint64_t> g_blackSpriteDrawnCount{0};

    // Called for every sprite the probe evaluates (suppressed or not) so the
    // [gs:suppress] line always reports both sides of the ratio. Printed at
    // most once per 5s, mirroring the other periodic probes in this file.
    void blackSpriteSuppressNote(bool suppressed)
    {
        if (suppressed)
            g_blackSpriteSuppressedCount.fetch_add(1, std::memory_order_relaxed);
        else
            g_blackSpriteDrawnCount.fetch_add(1, std::memory_order_relaxed);

        static std::atomic<uint64_t> s_lastPrintMs{0};
        const uint64_t now = dq8ProbeNowMs();
        uint64_t last = s_lastPrintMs.load(std::memory_order_relaxed);
        if (last != 0ull && now - last < 5000ull)
            return;
        if (!s_lastPrintMs.compare_exchange_strong(last, now, std::memory_order_relaxed))
            return; // another call already printed this window
        std::cout << "[gs:suppress] DIAGNOSTIC PROBE (not a fix) t=" << std::dec << now
                   << "ms suppressed=" << g_blackSpriteSuppressedCount.load(std::memory_order_relaxed)
                   << " drawn=" << g_blackSpriteDrawnCount.load(std::memory_order_relaxed)
                   << std::endl;
    }
}

void GSRasterizer::drawSprite(GS *gs)
{
    const GSVertex &v0 = gs->m_vtxQueue[0];
    const GSVertex &v1 = gs->m_vtxQueue[1];

    // ---- DIAGNOSTIC PROBE (PS2X_SUPPRESS_BLACK_SPRITES=1) -- see the block
    // comment above blackSpriteSuppressOn(). NOT A FIX. Checked before any
    // clipping/blend/texture work so a suppressed primitive costs nothing
    // beyond the checks themselves. Sprite colour is flat-shaded from the
    // kick vertex (v1), matching how the real fill path below already reads
    // colour (see the SPEC 01 comment on v1.z a few lines down).
    if (blackSpriteSuppressOn())
    {
        const bool isBlackUntexturedSprite =
            !gs->m_prim.tme && v1.r == 0u && v1.g == 0u && v1.b == 0u;
        blackSpriteSuppressNote(isBlackUntexturedSprite);
        if (isBlackUntexturedSprite)
            return;
    }

    const auto &ctx = gs->activeContext();

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;

    if (x0 > x1)
        std::swap(x0, x1);
    if (y0 > y1)
        std::swap(y0, y1);

    const int unclippedX0 = x0;
    const int unclippedY0 = y0;
    const int spanX = std::max(1, x1 - x0);
    const int spanY = std::max(1, y1 - y0);
    const int unclippedX1 = unclippedX0 + spanX - 1;
    const int unclippedY1 = unclippedY0 + spanY - 1;

    // If the sprite rectangle is fully outside scissor, nothing should render.
    if (unclippedX1 < ctx.scissor.x0 || unclippedX0 > ctx.scissor.x1 ||
        unclippedY1 < ctx.scissor.y0 || unclippedY0 > ctx.scissor.y1)
    {
        // maybe a log here idk ?
        return;
    }

    const int drawX0 = clampInt(unclippedX0, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY0 = clampInt(unclippedY0, ctx.scissor.y0, ctx.scissor.y1);
    const int drawX1 = clampInt(unclippedX1, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY1 = clampInt(unclippedY1, ctx.scissor.y0, ctx.scissor.y1);

    const uint64_t alphaReg = ctx.alpha;
    const uint8_t alphaMode = static_cast<uint8_t>(alphaReg & 0xFFu);
    const uint8_t alphaFix = static_cast<uint8_t>((alphaReg >> 32) & 0xFFu);
    const bool looksLikeDisplayCopy =
        gs->m_prim.tme &&
        gs->m_prim.abe &&
        gs->m_prim.fst &&
        gs->m_prim.ctxt &&
        ctx.frame.fbp != ctx.tex0.tbp0 &&
        alphaMode == 0x64u &&
        (alphaFix == 0x60u || alphaFix == 0x80u) &&
        unclippedX0 <= 0 &&
        unclippedY0 <= 0 &&
        unclippedX1 >= 639 &&
        unclippedY1 >= 447;
    if (looksLikeDisplayCopy)
    {
        gs->m_preferredDisplaySourceFrame = {ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, 0u};
        gs->m_preferredDisplayDestFbp = ctx.frame.fbp;
        gs->m_hasPreferredDisplaySource = true;
    }

    uint8_t r = v1.r, g = v1.g, b = v1.b, a = v1.a;
    // Sprite Z is flat -- the kick vertex (v1) carries it, per SPEC 01 §4.2
    // (transcribed from the DC2 finding, G203's drawSprite fix).
    const float zf = v1.z;

    if (gs->m_prim.tme)
    {
        const auto &tex = ctx.tex0;
        int texW = 1 << tex.tw;
        int texH = 1 << tex.th;
        if (texW == 0)
            texW = 1;
        if (texH == 0)
            texH = 1;

        float u0f, v0f, u1f, v1f;
        if (gs->m_prim.fst)
        {
            u0f = static_cast<float>(v0.u >> 4);
            v0f = static_cast<float>(v0.v >> 4);
            u1f = static_cast<float>(v1.u >> 4);
            v1f = static_cast<float>(v1.v >> 4);
        }
        else
        {
            const float q0 = fabsQ(v0.q);
            const float q1 = fabsQ(v1.q);
            u0f = (v0.s / q0) * static_cast<float>(texW);
            v0f = (v0.t / q0) * static_cast<float>(texH);
            u1f = (v1.s / q1) * static_cast<float>(texW);
            v1f = (v1.t / q1) * static_cast<float>(texH);
        }

        float spriteW = static_cast<float>(spanX);
        float spriteH = static_cast<float>(spanY);
        if (spriteW < 1.0f)
            spriteW = 1.0f;
        if (spriteH < 1.0f)
            spriteH = 1.0f;

        for (int y = drawY0; y <= drawY1; ++y)
        {
            float ty = (static_cast<float>(y - unclippedY0) + 0.5f) / spriteH;
            float texVf = v0f + (v1f - v0f) * ty;

            for (int x = drawX0; x <= drawX1; ++x)
            {
                float tx = (static_cast<float>(x - unclippedX0) + 0.5f) / spriteW;
                float texUf = u0f + (u1f - u0f) * tx;
                uint32_t texel = 0xFFFF00FFu;
                if (gs->m_prim.fst)
                {
                    const uint16_t sampleU = static_cast<uint16_t>(clampInt(static_cast<int>(std::lround(texUf * 16.0f)), 0, 0xFFFF));
                    const uint16_t sampleV = static_cast<uint16_t>(clampInt(static_cast<int>(std::lround(texVf * 16.0f)), 0, 0xFFFF));
                    texel = sampleTexture(gs, 0.0f, 0.0f, 1.0f, sampleU, sampleV);
                }
                else
                {
                    texel = sampleTexture(gs,
                                          texUf / static_cast<float>(texW),
                                          texVf / static_cast<float>(texH),
                                          1.0f, 0u, 0u);
                }

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const TextureCombineResult color = combineTexture(tex, r, g, b, a, tr, tg, tb, ta);
                writePixel(gs, x, y, color.r, color.g, color.b, color.a, zf);
            }
        }
    }
    else
    {
        for (int y = drawY0; y <= drawY1; ++y)
            for (int x = drawX0; x <= drawX1; ++x)
                writePixel(gs, x, y, r, g, b, a, zf);
    }
}

void GSRasterizer::drawTriangle(GS *gs)
{
    const GSVertex &v0 = gs->m_vtxQueue[0];
    const GSVertex &v1 = gs->m_vtxQueue[1];
    const GSVertex &v2 = gs->m_vtxQueue[2];
    const auto &ctx = gs->activeContext();

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    float fx0 = v0.x - static_cast<float>(ofx);
    float fy0 = v0.y - static_cast<float>(ofy);
    float fx1 = v1.x - static_cast<float>(ofx);
    float fy1 = v1.y - static_cast<float>(ofy);
    float fx2 = v2.x - static_cast<float>(ofx);
    float fy2 = v2.y - static_cast<float>(ofy);

    if (!nearplaneClipDisabled() &&
        triangleNeedsNearOrGuardBandCull(fx0, fy0, fx1, fy1, fx2, fy2, v0.q, v1.q, v2.q, ctx.scissor))
    {
        return;
    }

    int minX = static_cast<int>(std::floor(std::min({fx0, fx1, fx2})));
    int maxX = static_cast<int>(std::ceil(std::max({fx0, fx1, fx2})));
    int minY = static_cast<int>(std::floor(std::min({fy0, fy1, fy2})));
    int maxY = static_cast<int>(std::ceil(std::max({fy0, fy1, fy2})));

    minX = clampInt(minX, ctx.scissor.x0, ctx.scissor.x1);
    maxX = clampInt(maxX, ctx.scissor.x0, ctx.scissor.x1);
    minY = clampInt(minY, ctx.scissor.y0, ctx.scissor.y1);
    maxY = clampInt(maxY, ctx.scissor.y0, ctx.scissor.y1);

    float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
    if (std::fabs(denom) < 0.001f)
        return;

    const float winding = (denom < 0.0f) ? -1.0f : 1.0f;
    const float invAbsDenom = 1.0f / std::fabs(denom);
    constexpr float kEdgeEpsilon = 1.0e-4f;

    for (int y = minY; y <= maxY; ++y)
    {
        float py = static_cast<float>(y) + 0.5f;
        for (int x = minX; x <= maxX; ++x)
        {
            float px = static_cast<float>(x) + 0.5f;

            float w0 = (((fy1 - fy2) * (px - fx2) + (fx2 - fx1) * (py - fy2)) * winding) * invAbsDenom;
            float w1 = (((fy2 - fy0) * (px - fx2) + (fx0 - fx2) * (py - fy2)) * winding) * invAbsDenom;
            float w2 = 1.0f - w0 - w1;

            if (w0 < -kEdgeEpsilon || w1 < -kEdgeEpsilon || w2 < -kEdgeEpsilon)
                continue;

            // Z interpolated linearly in screen space (barycentric), per
            // SPEC 01 §4.2 / DC2 G41.
            const float zf = v0.z * w0 + v1.z * w1 + v2.z * w2;

            uint8_t r, g, b, a;
            if (gs->m_prim.iip)
            {
                r = clampU8(static_cast<int>(v0.r * w0 + v1.r * w1 + v2.r * w2));
                g = clampU8(static_cast<int>(v0.g * w0 + v1.g * w1 + v2.g * w2));
                b = clampU8(static_cast<int>(v0.b * w0 + v1.b * w1 + v2.b * w2));
                a = clampU8(static_cast<int>(v0.a * w0 + v1.a * w1 + v2.a * w2));
            }
            else
            {
                r = v2.r;
                g = v2.g;
                b = v2.b;
                a = v2.a;
            }

            if (gs->m_prim.tme)
            {
                float is, it, iq;
                uint16_t iu, iv;
                if (gs->m_prim.fst)
                {
                    iu = static_cast<uint16_t>(v0.u * w0 + v1.u * w1 + v2.u * w2);
                    iv = static_cast<uint16_t>(v0.v * w0 + v1.v * w1 + v2.v * w2);
                    is = 0.0f;
                    it = 0.0f;
                    iq = 1.0f;
                }
                else
                {
                    const float invQ0 = 1.0f / fabsQ(v0.q);
                    const float invQ1 = 1.0f / fabsQ(v1.q);
                    const float invQ2 = 1.0f / fabsQ(v2.q);
                    const float sOverQ = (v0.s * invQ0) * w0 + (v1.s * invQ1) * w1 + (v2.s * invQ2) * w2;
                    const float tOverQ = (v0.t * invQ0) * w0 + (v1.t * invQ1) * w1 + (v2.t * invQ2) * w2;
                    const float invQ = invQ0 * w0 + invQ1 * w1 + invQ2 * w2;
                    iq = (std::fabs(invQ) > 1.0e-8f) ? (1.0f / invQ) : 1.0f;
                    is = sOverQ * iq;
                    it = tOverQ * iq;
                    iu = 0;
                    iv = 0;
                }

                uint32_t texel = sampleTexture(gs, is, it, iq, iu, iv);

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const auto &tex = ctx.tex0;
                const uint8_t shadeR = r;
                const uint8_t shadeG = g;
                const uint8_t shadeB = b;
                const uint8_t shadeA = a;
                const TextureCombineResult color = combineTexture(tex, shadeR, shadeG, shadeB, shadeA, tr, tg, tb, ta);

                r = color.r;
                g = color.g;
                b = color.b;
                a = color.a;
            }

            writePixel(gs, x, y, r, g, b, a, zf);
        }
    }
}

void GSRasterizer::drawLine(GS *gs)
{
    const GSVertex &v0 = gs->m_vtxQueue[0];
    const GSVertex &v1 = gs->m_vtxQueue[1];
    const auto &ctx = gs->activeContext();

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;

    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    int totalSteps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    if (totalSteps == 0)
        totalSteps = 1;
    int step = 0;

    for (;;)
    {
        float t = static_cast<float>(step) / static_cast<float>(totalSteps);
        uint8_t r, g, b, a;
        if (gs->m_prim.iip)
        {
            r = clampU8(static_cast<int>(v0.r + (v1.r - v0.r) * t));
            g = clampU8(static_cast<int>(v0.g + (v1.g - v0.g) * t));
            b = clampU8(static_cast<int>(v0.b + (v1.b - v0.b) * t));
            a = clampU8(static_cast<int>(v0.a + (v1.a - v0.a) * t));
        }
        else
        {
            r = v1.r;
            g = v1.g;
            b = v1.b;
            a = v1.a;
        }

        const float zf = v0.z + (v1.z - v0.z) * t;
        writePixel(gs, x0, y0, r, g, b, a, zf);

        if (x0 == x1 && y0 == y1)
            break;

        int e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
        ++step;
    }
}
