#include "runtime/ps2_vu1.h"
#include "runtime/ps2_memory.h"
#include "ps2_vu1_detail.h"

#include <cstring>

VU1Interpreter::VU1Interpreter()
{
    reset();
}

void VU1Interpreter::reset()
{
    std::memset(&m_state, 0, sizeof(m_state));
    m_state.vf[0][3] = 1.0f; // VF0.w = 1.0
    m_state.q = 1.0f;
    vuPipeReset();
}

void VU1Interpreter::resyncPipelineShadow()
{
    // Re-seed the shadow flags from THIS instance's current (now fully
    // populated) m_state.mac/status/clip. reset() already called
    // vuPipeReset() once, but that ran before the caller copied real entry
    // state (e.g. copyVu0ContextToState) into m_state, so the shadow was
    // seeded from a still-zeroed state. Calling vuPipeReset() again here,
    // now that mac/status/clip hold the true entry values, makes the
    // shadow agree with the architectural flags a program actually reads
    // at PC 0.
    vuPipeReset();
}

float VU1Interpreter::broadcast(const float *vf, uint8_t bc)
{
    return vf[bc & 3];
}

// ---------------------------------------------------------------------
// VU pipeline model: MAC/STATUS/CLIP flags commit 4 cycles after the FMAC
// that produced them (matching PCSX2 _vuFMACflush); Q (FDIV/FSQRT/FRSQRT)
// and P (EFU) results commit after their own op-specific latency, forced
// early by WAITQ/WAITP. See ps2_vu1.h for why this is an instance member
// now instead of a TU-static shared between VU0 macro mode and VU1.
void VU1Interpreter::vuPipeReset()
{
    m_vuCycle = 0u;
    m_flagRd = 0u;
    m_flagCount = 0u;
    m_qPending = false;
    m_pPending = false;
    m_macShadow = m_state.mac;
    m_statusShadow = m_state.status;
    m_clipShadow = m_state.clip;
}

// Commit every pipeline entry whose result is architecturally visible at
// or before `now`.
void VU1Interpreter::vuPipeAdvance(uint64_t now)
{
    while (m_flagCount != 0u)
    {
        const FlagPipeEntry &e = m_flagPipe[m_flagRd];
        if (e.due > now)
            break;
        if (e.hasMacStatus)
        {
            m_state.mac = e.mac;
            m_state.status = e.status;
        }
        if (e.hasClip)
            m_state.clip = e.clip;
        m_flagRd = (m_flagRd + 1u) % kFlagPipeSlots;
        --m_flagCount;
    }
    if (m_qPending && m_qDue <= now)
    {
        m_state.q = m_qValue;
        m_qPending = false;
    }
    if (m_pPending && m_pDue <= now)
    {
        m_state.p = m_pValue;
        m_pPending = false;
    }
}

void VU1Interpreter::vuPipePushFlags(uint32_t mac, uint32_t status)
{
    if (m_flagCount == kFlagPipeSlots)
        vuPipeAdvance(m_flagPipe[m_flagRd].due); // ring full: retire the oldest
    const uint32_t wr = (m_flagRd + m_flagCount) % kFlagPipeSlots;
    m_flagPipe[wr] = FlagPipeEntry{m_vuCycle + 4u, mac, status, 0u, true, false};
    ++m_flagCount;
}

void VU1Interpreter::vuPipePushClip(uint32_t clip)
{
    if (m_flagCount == kFlagPipeSlots)
        vuPipeAdvance(m_flagPipe[m_flagRd].due);
    const uint32_t wr = (m_flagRd + m_flagCount) % kFlagPipeSlots;
    m_flagPipe[wr] = FlagPipeEntry{m_vuCycle + 4u, 0u, 0u, clip, false, true};
    ++m_flagCount;
}

// G200: issuing a second FDIV while one is in flight busy-stalls until the
// first latches, then starts the new one -- it must not drop the pending
// result.
void VU1Interpreter::vuPipeStartQ(float value, uint32_t latency)
{
    if (m_qPending)
    {
        if (m_qDue > m_vuCycle)
            m_vuCycle = m_qDue;
        vuPipeAdvance(m_vuCycle);
    }
    m_qPending = true;
    m_qDue = m_vuCycle + latency;
    m_qValue = value;
}

void VU1Interpreter::vuPipeStartP(float value, uint32_t latency)
{
    if (m_pPending)
    {
        if (m_pDue > m_vuCycle)
            m_vuCycle = m_pDue;
        vuPipeAdvance(m_vuCycle);
    }
    m_pPending = true;
    m_pDue = m_vuCycle + latency;
    m_pValue = value;
}

void VU1Interpreter::vuPipeWaitQ()
{
    if (m_qPending && m_qDue > m_vuCycle)
        m_vuCycle = m_qDue;
    vuPipeAdvance(m_vuCycle);
}

void VU1Interpreter::vuPipeWaitP()
{
    if (m_pPending && m_pDue > m_vuCycle)
        m_vuCycle = m_pDue;
    vuPipeAdvance(m_vuCycle);
}

// PCSX2 _vuFlushAll(): everything in flight becomes visible when the
// microprogram ends.
void VU1Interpreter::vuPipeFlushAll()
{
    uint64_t latest = m_vuCycle;
    for (uint32_t k = 0; k < m_flagCount; ++k)
    {
        const uint64_t due = m_flagPipe[(m_flagRd + k) % kFlagPipeSlots].due;
        if (due > latest)
            latest = due;
    }
    if (m_qPending && m_qDue > latest)
        latest = m_qDue;
    if (m_pPending && m_pDue > latest)
        latest = m_pDue;
    m_vuCycle = latest;
    vuPipeAdvance(latest);
}

void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
{
    // MAC / STATUS flag update. m_state.mac is declared and read by
    // FMAND/FMEQ/FMOR/FCAND but was never written anywhere, so it was
    // permanently 0 and every MAC-gated branch in Level-5's VU1 microcode
    // evaluated against the wrong constant.
    //
    // Only FMAC (upper) instructions set the flags. applyDest is also used
    // by lower ops (LQ/MOVE/MFIR/...) which must not touch them, so the
    // upper-op window is marked by m_vuInUpperOp.
    //
    // MAC layout (PS2 VU, 16 bits): O=15..12, U=11..8, S=7..4, Z=3..0, each
    // group ordered x,y,z,w from the high bit down. Lanes outside DEST read 0.
    //
    // The flag update is also where the VU's non-IEEE result conditioning
    // happens, and the conditioned value is what gets WRITTEN -- not just
    // flagged. PCSX2 VUops.cpp VU_MACx_UPDATE returns the rewritten bits:
    //     if (exp == 0)   { macflag = (macflag & ~0x1100) | 0x0001; return s; }
    //     if (exp == 255) { macflag = (macflag & ~0x1001) | 0x0100;
    //                       return s | 0x7f7fffff; }
    // i.e. an overflowing result becomes +/-Fmax and a denormal becomes
    // signed zero. This VU has no infinity and no NaN encoding, so leaving
    // a host Inf/NaN in vf or ACC hands the microprogram a value its own
    // arithmetic could never have produced, and it then spreads through
    // every subsequent FMAC, FMAND gate and vertex pack.
    float conditioned[4];
    if (m_vuInUpperOp)
    {
        uint32_t mac = 0u;
        for (int c = 0; c < 4; ++c)
        {
            conditioned[c] = result[c];
            if (!(dest & (0x8u >> c)))
                continue;
            const uint32_t bit = static_cast<uint32_t>(3 - c);
            const float v = result[c];
            uint32_t bits = 0u;
            std::memcpy(&bits, &v, sizeof(bits));
            if ((bits & 0x7FFFFFFFu) == 0u)
                mac |= (1u << bit); // Z
            if (bits & 0x80000000u)
                mac |= (1u << (4u + bit)); // S
            const uint32_t exp = (bits >> 23) & 0xFFu;
            if (exp == 0xFFu)
            {
                mac |= (1u << (12u + bit)); // O (VU saturates; inf/nan == overflow)
                const uint32_t sat = (bits & 0x80000000u) | 0x7F7FFFFFu;
                std::memcpy(&conditioned[c], &sat, sizeof(sat));
            }
            else if (exp == 0u && (bits & 0x7FFFFFu) != 0u)
            {
                mac |= (1u << (8u + bit)); // U (denormal)
                const uint32_t flushed = bits & 0x80000000u;
                std::memcpy(&conditioned[c], &flushed, sizeof(flushed));
            }
        }
        result = conditioned;
        // STATUS: bits 3..0 = current O,U,S,Z; bits 9..6 = sticky OS,US,SS,ZS.
        uint32_t cur = 0u;
        if (mac & 0x000Fu)
            cur |= 0x1u; // Z
        if (mac & 0x00F0u)
            cur |= 0x2u; // S
        if (mac & 0x0F00u)
            cur |= 0x4u; // U
        if (mac & 0xF000u)
            cur |= 0x8u; // O
        const uint32_t sticky = (m_statusShadow | (cur << 6)) & 0x3C0u;

        // Shadow updates immediately (successive FMACs chain off it); the
        // architectural MAC/STATUS become visible 4 cycles later, which is
        // what FMAND/FMEQ/FMOR/FCAND read.
        m_macShadow = mac;
        m_statusShadow = (m_statusShadow & ~0x3CFu) | cur | sticky;
        vuPipePushFlags(m_macShadow, m_statusShadow);
    }

    if (dest & 0x8)
        dst[0] = result[0]; // x
    if (dest & 0x4)
        dst[1] = result[1]; // y
    if (dest & 0x2)
        dst[2] = result[2]; // z
    if (dest & 0x1)
        dst[3] = result[3]; // w
}

void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const
{
    DecodedInstructionPair decoded;
    std::memcpy(&decoded.lower, vuCode + pc, sizeof(decoded.lower));
    std::memcpy(&decoded.upper, vuCode + pc + sizeof(decoded.lower), sizeof(decoded.upper));

    decoded.iBit = ((decoded.upper >> 31) & 1u) != 0u;
    decoded.eBit = ((decoded.upper >> 30) & 1u) != 0u;
    decoded.lowerBeforeUpper = !decoded.iBit && vuLowerShouldRunBeforeUpper(decoded.upper, decoded.lower);
    return decoded;
}

void VU1Interpreter::rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize,
                                             const PS2Memory *memory, uint64_t generation)
{
    const uint32_t pairCount = codeSize / 8u;
    m_decodedCodeCache.resize(pairCount);
    for (uint32_t i = 0; i < pairCount; ++i)
    {
        m_decodedCodeCache[i] = decodeInstructionPair(vuCode, i * 8u);
    }

    m_cachedVuCode = vuCode;
    m_cachedMemory = memory;
    m_cachedCodeSize = codeSize;
    m_cachedCodeGeneration = generation;
    m_decodedCodeCacheValid = true;
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::getDecodedInstructionPairForPc(const uint8_t *vuCode,
                                                                                      uint32_t codeSize,
                                                                                      PS2Memory *memory,
                                                                                      uint32_t pc)
{
    // Only 8-byte aligned VU instruction pairs can use the decode cache.
    if ((pc & 7u) != 0u)
    {
        return decodeInstructionPair(vuCode, pc);
    }

    const bool trackedVu1Code = vuCode == memory->getVU1Code();
    if (!trackedVu1Code)
    {
        return decodeInstructionPair(vuCode, pc);
    }

    const uint64_t generation = memory->getVU1CodeGeneration();
    const bool rebuild =
        !m_decodedCodeCacheValid ||
        m_cachedVuCode != vuCode ||
        m_cachedMemory != memory ||
        m_cachedCodeSize != codeSize ||
        m_cachedCodeGeneration != generation;

    if (rebuild)
    {
        rebuildDecodedCodeCache(vuCode, codeSize, memory, generation);
    }

    return m_decodedCodeCache[pc / 8u];
}

void VU1Interpreter::execute(uint8_t *vuCode, uint32_t codeSize,
                             uint8_t *vuData, uint32_t dataSize,
                             GS &gs, PS2Memory *memory,
                             uint32_t startPC, uint32_t top, uint32_t itop,
                             uint32_t maxCycles)
{
    m_state.pc = startPC & 0x3FFFu;
    m_state.ebit = false;
    m_state.top = top;
    m_state.itop = itop;
    m_state.branchPending = false;
    m_state.branchTarget = 0;
    m_state.branchDelay = 0;
    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::resume(uint8_t *vuCode, uint32_t codeSize,
                            uint8_t *vuData, uint32_t dataSize,
                            GS &gs, PS2Memory *memory,
                            uint32_t top, uint32_t itop, uint32_t maxCycles)
{
    m_state.ebit = false;
    m_state.top = top;
    m_state.itop = itop;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::run(uint8_t *vuCode, uint32_t codeSize,
                         uint8_t *vuData, uint32_t dataSize,
                         GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    for (uint32_t cycle = 0; cycle < maxCycles; ++cycle)
    {
        if (m_state.pc + 8 > codeSize)
            break;

        const DecodedInstructionPair decoded = getDecodedInstructionPairForPc(vuCode, codeSize, memory, m_state.pc);

        // One VU cycle per 64-bit instruction pair (VU1 is dual-issue), then
        // retire anything whose latency has expired -- PCSX2 calls
        // _vuTestPipes before each instruction for exactly this reason.
        ++m_vuCycle;
        vuPipeAdvance(m_vuCycle);

        // LOI is controlled by the upper I-bit.  The lower word is the float immediate.
        // DobieStation executes the upper instruction first, then commits lower into I.
        if (decoded.iBit)
        {
            // LOI is special: the upper instruction sees the old I value, then LOI loads I.
            m_vuInUpperOp = true;
            execUpper(decoded.upper);
            m_vuInUpperOp = false;
            std::memcpy(&m_state.i, &decoded.lower, sizeof(decoded.lower));
        }
        else if (decoded.lowerBeforeUpper)
        {
            // VU upper/lower execute as a pair.  If the upper op writes a VF register
            // that the lower op reads or also writes, Dobie runs the lower side first
            // so it observes the old VF value and the upper write has priority. WAITQ/
            // WAITP are hoisted here too (vuLowerIsWaitQOrP in ps2_vu1_detail.h) even
            // though they touch no VF register, because they stall the whole pair on
            // Q/P settling.
            //
            // That reordering alone mishandles one more case: if the upper op ALSO
            // reads (as fs/ft) a VF register that the lower op writes, neither
            // execution order is correct on its own -- the upper needs the pre-pair
            // value of that register, but running lower first just clobbered it.
            // Snapshot the upper op's own source registers before the lower op runs,
            // substitute the snapshot back in immediately before the upper op runs,
            // then restore the lower op's real result afterward -- unless the upper
            // op's own destination is that same register, in which case the upper's
            // write correctly wins and nothing is restored.
            const uint8_t upperFs = FS(decoded.upper);
            const uint8_t upperFt = FT(decoded.upper);
            const uint8_t upperWriteReg = vuUpperVfWriteReg(decoded.upper);

            uint32_t lowerReads = 0u;
            uint32_t lowerWrites = 0u;
            vuLowerVfReadWriteMasks(decoded.lower, lowerReads, lowerWrites);
            (void)lowerReads; // only the write mask matters for this snapshot

            const bool snapFs = upperFs != 0u && ((lowerWrites >> upperFs) & 1u) != 0u;
            const bool snapFt = upperFt != 0u && ((lowerWrites >> upperFt) & 1u) != 0u;

            float preFs[4], preFt[4];
            if (snapFs)
                std::memcpy(preFs, m_state.vf[upperFs], 16);
            if (snapFt)
                std::memcpy(preFt, m_state.vf[upperFt], 16);

            execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);

            float postLowerFs[4], postLowerFt[4];
            if (snapFs)
                std::memcpy(postLowerFs, m_state.vf[upperFs], 16);
            if (snapFt)
                std::memcpy(postLowerFt, m_state.vf[upperFt], 16);

            if (snapFs)
                std::memcpy(m_state.vf[upperFs], preFs, 16);
            if (snapFt)
                std::memcpy(m_state.vf[upperFt], preFt, 16);

            m_vuInUpperOp = true;
            execUpper(decoded.upper);
            m_vuInUpperOp = false;

            if (snapFs && upperFs != upperWriteReg)
                std::memcpy(m_state.vf[upperFs], postLowerFs, 16);
            if (snapFt && upperFt != upperWriteReg)
                std::memcpy(m_state.vf[upperFt], postLowerFt, 16);
        }
        else
        {
            m_vuInUpperOp = true;
            execUpper(decoded.upper);
            m_vuInUpperOp = false;
            execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
        }

        // Enforce VF0 invariant
        m_state.vf[0][0] = 0.0f;
        m_state.vf[0][1] = 0.0f;
        m_state.vf[0][2] = 0.0f;
        m_state.vf[0][3] = 1.0f;
        // Enforce VI0 invariant
        m_state.vi[0] = 0;

        uint32_t nextPC = m_state.pc + 8;
        if (nextPC >= codeSize)
            nextPC = 0;
        m_state.pc = nextPC;

        // VU branch/jump has a delay slot. Branch handlers set a pending target;
        // we execute one sequential instruction before committing the branch.
        if (m_state.branchPending)
        {
            if (m_state.branchDelay == 0)
            {
                m_state.pc = m_state.branchTarget & 0x3FFFu;
                m_state.branchPending = false;
            }
            else
            {
                --m_state.branchDelay;
            }
        }

        if (m_state.ebit)
            break;

        if (decoded.eBit)
            m_state.ebit = true;
    }

    // PCSX2 _vuFlushAll(): the FMAC-flag, FDIV(Q) and EFU(P) pipes all become
    // visible when the microprogram stops, so the next program (and any
    // EE-side reader of VU1 state) sees settled values.
    vuPipeFlushAll();
}
