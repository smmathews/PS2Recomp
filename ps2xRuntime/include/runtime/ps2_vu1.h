#ifndef PS2_VU1_H
#define PS2_VU1_H

#include <cstdint>
#include <vector>

class GS;
class PS2Memory;

struct VU1State
{
    float vf[32][4];
    int32_t vi[16];
    float acc[4];
    float q;
    float p;
    float i;
    uint32_t pc;
    uint32_t mac;
    uint32_t clip;
    uint32_t status;
    bool ebit;
    uint32_t top;  // VIF1 TOP visible to VU1 XTOP
    uint32_t itop; // VIF1 ITOP visible to VU1 XITOP

    bool branchPending;
    uint32_t branchTarget;
    uint32_t branchDelay;
};

class VU1Interpreter
{
public:
    VU1Interpreter();

    void reset();

    void execute(uint8_t *vuCode, uint32_t codeSize,
                 uint8_t *vuData, uint32_t dataSize,
                 GS &gs, PS2Memory *memory = nullptr,
                 uint32_t startPC = 0, uint32_t top = 0, uint32_t itop = 0,
                 uint32_t maxCycles = 65536);

    void resume(uint8_t *vuCode, uint32_t codeSize,
                uint8_t *vuData, uint32_t dataSize,
                GS &gs, PS2Memory *memory = nullptr,
                uint32_t top = 0, uint32_t itop = 0, uint32_t maxCycles = 65536);

    VU1State &state() { return m_state; }
    const VU1State &state() const { return m_state; }

    // Re-seed the shadow copies (m_macShadow/m_statusShadow/m_clipShadow)
    // that FMAND/FMEQ/FMOR/FCAND/... eventually read (through m_state.mac/
    // clip/status once the flag pipe commits them), from this instance's
    // CURRENT m_state.mac/status/clip. reset() seeds these shadows before
    // a caller (e.g. VU0 macro mode in PS2Runtime::executeVU0Microprogram)
    // has copied any real entry state into m_state, so entry shadow and
    // entry architectural flags can disagree for the first few cycles.
    // Call this once entry state (mac/clip/status) has actually been
    // populated, immediately before execute().
    void resyncPipelineShadow();

private:
    struct DecodedInstructionPair
    {
        uint32_t lower = 0;
        uint32_t upper = 0;
        bool iBit = false;
        bool eBit = false;
        bool lowerBeforeUpper = false;
    };

    VU1State m_state;
    std::vector<DecodedInstructionPair> m_decodedCodeCache;
    const uint8_t *m_cachedVuCode = nullptr;
    const PS2Memory *m_cachedMemory = nullptr;
    uint32_t m_cachedCodeSize = 0;
    uint64_t m_cachedCodeGeneration = 0;
    bool m_decodedCodeCacheValid = false;

    // ---------------------------------------------------------------------
    // VU pipeline-timing model: MAC/STATUS/CLIP flag latency and Q/P (FDIV/
    // EFU) result latency. Ported from ps2xRuntime main's file-scope statics
    // in ps2_vu1.cpp (s_vuInUpperOp, s_vuCycle, s_macShadow/s_statusShadow/
    // s_clipShadow, the FlagPipeEntry ring, s_qPending/s_pPending).
    //
    // On main these had to be free-standing TU statics because VU0 macro
    // mode and the real VU1 geometry processor are driven from two
    // different VU1Interpreter instances that shared one translation unit's
    // globals; a save/restore fence (VU1SharedPipelineSnapshot) existed
    // solely to stop one instance's use of those globals from clobbering
    // the other's in-flight state.
    //
    // dq8-base already gives VU0 macro mode and VU1 their own
    // VU1Interpreter instances (PS2Runtime::m_vu0 / m_vu1), so making this
    // state a plain instance member here removes the sharing the fence
    // existed to paper over -- there is nothing left to fence, and the
    // fence itself must not be ported (see resyncPipelineShadow() for the
    // one piece of that machinery that IS still needed: reset() runs
    // before a caller populates entry mac/clip/status, same as on main).
    struct FlagPipeEntry
    {
        uint64_t due = 0;
        uint32_t mac = 0;
        uint32_t status = 0;
        uint32_t clip = 0;
        bool hasMacStatus = false;
        bool hasClip = false;
    };
    static constexpr uint32_t kFlagPipeSlots = 8u;

    // True only while an FMAC (upper) instruction is executing -- gates the
    // MAC/STATUS flag block in applyDest() so lower ops that also route
    // through applyDest (LQ/MOVE/MR32/LQI/LQD/MFIR/MFP) don't touch flags.
    bool m_vuInUpperOp = false;

    uint64_t m_vuCycle = 0;
    uint32_t m_macShadow = 0;
    uint32_t m_statusShadow = 0;
    uint32_t m_clipShadow = 0;

    FlagPipeEntry m_flagPipe[kFlagPipeSlots];
    uint32_t m_flagRd = 0;
    uint32_t m_flagCount = 0;

    bool m_qPending = false;
    uint64_t m_qDue = 0;
    float m_qValue = 0.0f;

    bool m_pPending = false;
    uint64_t m_pDue = 0;
    float m_pValue = 0.0f;

    void vuPipeReset();
    void vuPipeAdvance(uint64_t now);
    void vuPipePushFlags(uint32_t mac, uint32_t status);
    void vuPipePushClip(uint32_t clip);
    void vuPipeStartQ(float value, uint32_t latency);
    void vuPipeStartP(float value, uint32_t latency);
    void vuPipeWaitQ();
    void vuPipeWaitP();
    void vuPipeFlushAll();

    void run(uint8_t *vuCode, uint32_t codeSize,
             uint8_t *vuData, uint32_t dataSize,
             GS &gs, PS2Memory *memory, uint32_t maxCycles);

    DecodedInstructionPair decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const;
    DecodedInstructionPair getDecodedInstructionPairForPc(const uint8_t *vuCode, uint32_t codeSize,
                                                          PS2Memory *memory, uint32_t pc);
    void rebuildDecodedCodeCache(const uint8_t *vuCode, uint32_t codeSize,
                                 const PS2Memory *memory, uint64_t generation);

    void execUpper(uint32_t instr);
    void execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr);

    void applyDest(float *dst, const float *result, uint8_t dest);
    void applyDestAcc(const float *result, uint8_t dest);
    float broadcast(const float *vf, uint8_t bc);
};

#endif
