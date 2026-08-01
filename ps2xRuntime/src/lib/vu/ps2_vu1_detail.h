#ifndef PS2_VU1_DETAIL_H
#define PS2_VU1_DETAIL_H

#include <cstdint>

// Instruction field extraction helpers
static inline uint8_t DEST(uint32_t i) { return (uint8_t)((i >> 21) & 0xF); }
static inline uint8_t FT(uint32_t i) { return (uint8_t)((i >> 16) & 0x1F); }
static inline uint8_t FS(uint32_t i) { return (uint8_t)((i >> 11) & 0x1F); }
static inline uint8_t FD(uint32_t i) { return (uint8_t)((i >> 6) & 0x1F); }
static inline uint8_t BC(uint32_t i) { return (uint8_t)(i & 0x3); }

// Lower instruction field helpers
static inline uint8_t LIT(uint32_t i) { return (uint8_t)((i >> 16) & 0x1F); }
static inline uint8_t LIS(uint32_t i) { return (uint8_t)((i >> 11) & 0x1F); }
static inline uint8_t LID(uint32_t i) { return (uint8_t)((i >> 6) & 0x1F); }
static inline uint8_t VIT(uint32_t i) { return (uint8_t)((i >> 16) & 0xF); }
static inline uint8_t VIS(uint32_t i) { return (uint8_t)((i >> 11) & 0xF); }
static inline uint8_t VID(uint32_t i) { return (uint8_t)((i >> 6) & 0xF); }
static inline int16_t IMM11(uint32_t i) { return (int16_t)(int32_t)((int32_t)(i << 21) >> 21); }
static inline int16_t IMM15(uint32_t i)
{
    uint32_t lo11 = i & 0x7FF;
    uint32_t hi4 = (i >> 21) & 0xF;
    uint32_t raw = (hi4 << 11) | lo11;
    return (int16_t)(int32_t)((int32_t)(raw << 17) >> 17);
}


static inline uint8_t vuUpperVfWriteReg(uint32_t upper)
{
    const uint8_t op = upper & 0x3Fu;
    const uint8_t dest = DEST(upper);
    const uint8_t ft = FT(upper);
    const uint8_t fd = FD(upper);

    if (dest == 0u)
        return 0u;

    if (op <= 0x2Fu)
        return fd;

    if (op >= 0x3Cu)
    {
        const uint8_t specialOp = static_cast<uint8_t>((upper & 0x3u) | ((upper >> 4) & 0x7Cu));
        switch (specialOp)
        {
        // Upper special ops that write a VF register use FT as destination.
        case 0x10: // ITOF0
        case 0x11: // ITOF4
        case 0x12: // ITOF12
        case 0x13: // ITOF15
        case 0x14: // FTOI0
        case 0x15: // FTOI4
        case 0x16: // FTOI12
        case 0x17: // FTOI15
        case 0x1D: // ABS
            return ft;
        default:
            return 0u; // ACC/NOP/CLIP/etc.
        }
    }

    return 0u;
}

static inline void vuSetRegBit(uint32_t &mask, uint8_t reg)
{
    if (reg != 0u && reg < 32u)
        mask |= (1u << reg);
}

static inline void vuLowerVfReadWriteMasks(uint32_t lower, uint32_t &readMask, uint32_t &writeMask)
{
    readMask = 0u;
    writeMask = 0u;

    if (lower == 0u || lower == 0x8000033Cu)
        return;

    const uint8_t opHi = static_cast<uint8_t>((lower >> 25) & 0x7Fu);
    const uint8_t it = LIT(lower);
    const uint8_t is = LIS(lower);

    if ((lower & 0x80000000u) != 0u)
    {
        const uint8_t funct = lower & 0x3Fu;
        if (funct >= 0x3Cu && funct <= 0x3Fu)
        {
            const uint8_t specialOp = static_cast<uint8_t>((lower & 0x3u) | ((lower >> 4) & 0x7Cu));
            switch (specialOp)
            {
            case 0x30: // MOVE
            case 0x31: // MR32
                vuSetRegBit(readMask, is);
                vuSetRegBit(writeMask, it);
                return;
            case 0x34: // LQI
            case 0x36: // LQD
                vuSetRegBit(writeMask, it);
                return;
            case 0x35: // SQI
            case 0x37: // SQD
                vuSetRegBit(readMask, is);
                return;
            case 0x38: // DIV
            case 0x3A: // RSQRT
                vuSetRegBit(readMask, is);
                vuSetRegBit(readMask, it);
                return;
            case 0x39: // SQRT
                vuSetRegBit(readMask, it);
                return;
            case 0x3C: // MTIR
            case 0x3E: // ILWR source base is integer, but field source is VF for MTIR only.
                if (specialOp == 0x3C)
                    vuSetRegBit(readMask, is);
                return;
            case 0x3D: // MFIR
            case 0x64: // MFP
                vuSetRegBit(writeMask, it);
                return;
            // EFU block (0x70-0x7E). Every one of these reads vf[is];
            // WAITP (0x7B) touches no VF register at all and is
            // intentionally left out of this list -- it stalls the whole
            // pair on P settling instead, which vuLowerIsWaitQOrP() below
            // handles.
            case 0x70: // ESADD
            case 0x71: // ERSADD
            case 0x72: // ELENG
            case 0x73: // ERLENG
            case 0x74: // EATANxy
            case 0x75: // EATANxz
            case 0x76: // ESUM
            case 0x78: // ESQRT
            case 0x79: // ERSQRT
            case 0x7A: // ERCPR
            case 0x7C: // ESIN
            case 0x7D: // EATAN
            case 0x7E: // EEXP
                vuSetRegBit(readMask, is);
                return;
            default:
                return;
            }
        }
        return;
    }

    switch (opHi)
    {
    case 0x00: // LQ
        vuSetRegBit(writeMask, it);
        return;
    case 0x01: // SQ
        vuSetRegBit(readMask, is);
        return;
    default:
        return;
    }
}

// WAITQ (lower special 0x3B) and WAITP (0x7B) touch no VF register at all,
// so vuLowerVfReadWriteMasks() correctly reports 0/0 for both and the
// write-overlap predicate below can never fire for them. But hardware
// stalls the WHOLE pair until Q/P settles -- a Q-consuming upper op (e.g.
// MULq) paired with WAITQ against a pending DIV/SQRT/RSQRT latency must see
// the value WAITQ forces, not whatever is still sitting in st.q. Running
// upper before lower (the default order) lets the upper read the stale
// value and only lets WAITQ force the settle immediately after.
//
// Hoisting WAITQ/WAITP ahead of the upper is always safe regardless of
// what the upper reads or writes: they have no VF side effect for the
// double-hazard snapshot/restore in run() to get wrong, so it degrades to
// "just run the lower first" -- which is exactly the ordering hardware's
// whole-pair stall requires.
static inline bool vuLowerIsWaitQOrP(uint32_t lower)
{
    if ((lower & 0x80000000u) == 0u)
        return false;
    const uint8_t funct = lower & 0x3Fu;
    if (funct < 0x3Cu || funct > 0x3Fu)
        return false;
    const uint8_t specialOp = static_cast<uint8_t>((lower & 0x3u) | ((lower >> 4) & 0x7Cu));
    return specialOp == 0x3Bu || specialOp == 0x7Bu; // WAITQ, WAITP
}

static inline bool vuLowerShouldRunBeforeUpper(uint32_t upper, uint32_t lower)
{
    if (vuLowerIsWaitQOrP(lower))
        return true;

    const uint8_t upperWrite = vuUpperVfWriteReg(upper);
    if (upperWrite == 0u)
        return false;

    uint32_t lowerReads = 0u;
    uint32_t lowerWrites = 0u;
    vuLowerVfReadWriteMasks(lower, lowerReads, lowerWrites);

    const uint32_t upperBit = (1u << upperWrite);
    return ((lowerReads | lowerWrites) & upperBit) != 0u;
}

#endif