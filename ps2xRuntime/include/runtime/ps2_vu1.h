#ifndef PS2_VU1_H
#define PS2_VU1_H

#include <cstdint>

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
    uint32_t itop;
    uint32_t xitop;
};

// Snapshot of the VU pipeline-timing model's file-scope globals in
// ps2_vu1.cpp (s_vuCycle, the FMAC/STATUS/CLIP flag-pipe ring, the Q/P
// EFU latency state). Those globals are shared by EVERY VU1Interpreter
// instance -- there are two live ones, PS2Runtime::m_vu1 (the real VU1
// geometry processor, driven continuously across many MSCAL/MSCNT calls
// with state that is deliberately never reset) and the VU0-macro-mode
// scratch interpreter in executeVU0Microprogram (freshly reset() on
// EVERY VCALLMS). Without save/restore around a VU0 call, VU0's reset()
// -- which zeroes s_vuCycle and drops any in-flight Q/flag-pipe entries
// -- silently destroys whatever VU1 had pending, and VU0's own run then
// leaves ITS pending entries in the same globals for VU1 to mis-adopt.
// This struct plus save/restoreSharedPipeline() let a caller fence one
// interpreter's use of the shared globals so it can't clobber another's.
// A plain free-standing struct (not a VU1State/VU1Interpreter member) so
// it costs no layout change to either -- both are corpus-ABI sensitive
// via PS2Runtime::m_vu1.
struct VU1SharedPipelineSnapshot
{
    unsigned char opaque[320];
};

class VU1Interpreter
{
public:
    VU1Interpreter();

    void reset();

    void execute(uint8_t *vuCode, uint32_t codeSize,
                 uint8_t *vuData, uint32_t dataSize,
                 GS &gs, PS2Memory *memory = nullptr,
                 uint32_t startPC = 0, uint32_t itop = 0,
                 uint32_t maxCycles = 65536);

    void resume(uint8_t *vuCode, uint32_t codeSize,
                uint8_t *vuData, uint32_t dataSize,
                GS &gs, PS2Memory *memory = nullptr,
                uint32_t itop = 0, uint32_t maxCycles = 65536);

    VU1State &state() { return m_state; }
    const VU1State &state() const { return m_state; }

    // Save/restore the shared pipeline-timing globals (see
    // VU1SharedPipelineSnapshot above). Call saveSharedPipeline() before
    // this interpreter touches them and restoreSharedPipeline() after,
    // to fence this instance's run from clobbering another instance's
    // in-flight state.
    VU1SharedPipelineSnapshot saveSharedPipeline() const;
    void restoreSharedPipeline(const VU1SharedPipelineSnapshot &snap);

    // Re-seed the shadow copies (s_macShadow/s_statusShadow/s_clipShadow)
    // that FMAND/FMEQ/FMOR/FCAND/... read, from this instance's CURRENT
    // m_state.mac/status/clip. reset() seeds these shadows from m_state
    // before the caller has copied any real entry state in (e.g. VU0's
    // ctx->vu0_mac_flags), so entry shadow and entry architectural flags
    // can disagree for the first few cycles. Call this once entry state
    // (mac/clip/status) has actually been populated, immediately before
    // execute().
    void resyncPipelineShadow();

private:
    VU1State m_state;

    void run(uint8_t *vuCode, uint32_t codeSize,
             uint8_t *vuData, uint32_t dataSize,
             GS &gs, PS2Memory *memory, uint32_t maxCycles);

    void execUpper(uint32_t instr);
    void execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr);

    void applyDest(float *dst, const float *result, uint8_t dest);
    void applyDestAcc(const float *result, uint8_t dest);
    float broadcast(const float *vf, uint8_t bc);
};

#endif
