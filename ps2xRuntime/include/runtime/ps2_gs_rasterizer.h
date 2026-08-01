#ifndef PS2_GS_RASTERIZER_H
#define PS2_GS_RASTERIZER_H

#include <cstdint>

class GS;

class GSRasterizer
{
public:
    void drawPrimitive(GS *gs);
    // zf is the interpolated (or flat, for sprite/line/point) GS Z value for
    // this fragment, in the raw guest range (up to 32-bit unsigned) --
    // SPEC 01 depth testing (dq8/reference/dc2-learnings/01-*.md). Masked
    // and clamped internally to the active ZBUF.PSM's bit width.
    void writePixel(GS *gs, int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a, float zf);
    uint32_t sampleTexture(GS *gs, float s, float t, float q, uint16_t u, uint16_t v);
    uint32_t readTexelPSMCT32(GS *gs, uint32_t tbp0, uint32_t tbw, int texU, int texV);
    uint32_t readTexelPSMCT16(GS *gs, uint32_t tbp0, uint32_t tbw, int texU, int texV);
    uint32_t readTexelPSMT4(GS *gs, uint32_t tbp0, uint32_t tbw, int texU, int texV);
    uint32_t lookupCLUT(GS *gs, uint8_t index, uint32_t cbp, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm);

private:
    void drawSprite(GS *gs);
    void drawTriangle(GS *gs);
    void drawLine(GS *gs);
    static void logClutContentsOnce(GS *gs, uint32_t cbp, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm);
};

#endif
