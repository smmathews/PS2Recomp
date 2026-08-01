#ifndef PS2_VIF0_H
#define PS2_VIF0_H

#include <cstdint>
#include <utility>
#include <vector>

class PS2Memory;

// ---------------------------------------------------------------------------
// VIF0 / VU0 micro-memory support.
//
// This tree's PS2Memory has m_vu1Code/m_vu1Data but never had the VU0
// equivalents, and ps2_vif1_interpreter.cpp only ever implemented
// processVIF1Data. The consequence was that DQ8's VU0 microprogram upload
// (uploader sub_0013CD10, packet at guest 0x390740, DMAtag 0x10000043,
// VIFcode 0x4A700000 = MPG NUM=0x70 -> 112 instructions at micro address 0)
// was dropped on the floor: DMA channel 0 (0x10008000) was not routed
// anywhere, so nothing ever landed in VU0 micro memory, and
// PS2Runtime::executeVU0Microprogram had nothing it could have executed even
// if it had wanted to.
//
// ABI NOTE -- read before "tidying" this into PS2Memory.
// The VU0 micro/data buffers and the pending-VIF0 queue deliberately live as
// TU-globals here rather than as PS2Memory data members. The prebuilt game
// corpus .so is compiled against a fixed PS2Memory / R5900Context / PS2Runtime
// layout; inserting a data member into any of those shifts every subsequent
// field and segfaults instantly against an already-compiled corpus. Keeping
// this state in a TU-global changes no class layout at all, so the runtime can
// be rebuilt on its own and paired with the existing corpus. Member
// *functions* would have been safe too, but free functions keep the whole
// feature in one file that can be lifted out later.
//
// There is exactly one VU0 on the machine, so one set of globals is correct;
// this is shared state by nature, not an accident of the port.
// ---------------------------------------------------------------------------

// VU0 micro memory (4 KB instruction) and VU0 data memory (4 KB VU Mem).
uint8_t *ps2xVu0Code();
uint8_t *ps2xVu0Data();

// True once at least one MPG has landed in VU0 micro memory.
bool ps2xVu0MicroLoaded();

// Total MPG uploads and the instruction count / dest of the most recent one.
// DQ8 hot-swaps two programs into micro address 0 (112 instructions from ELF
// 0x390750 and 88 instructions from 0x38188c), so "which program is live" is
// answered by the most recent upload, never by the address. The MPG memcpy
// itself already gives that for free -- a later upload overwrites the earlier
// one in place, exactly as hardware does -- these counters are for logging.
uint64_t ps2xVu0MpgUploadCount();
uint32_t ps2xVu0LastMpgInstrCount();
uint32_t ps2xVu0LastMpgDestAddr();

// VIF0 vifcode parser. Handles NOP/STCYCL/ITOP/STMOD/MARK/FLUSH*/STMASK/
// STROW/STCOL/MPG/UNPACK, mirroring the VIF1 parser in this tree.
void ps2xProcessVIF0Data(PS2Memory &mem, const uint8_t *data, uint32_t sizeBytes);
void ps2xProcessVIF0Data(PS2Memory &mem, uint32_t srcPhys, uint32_t sizeBytes);

// Pending DMA channel-0 transfers, queued from PS2Memory's CHCR-write handler
// and drained from processPendingTransfers() alongside the GIF/VIF1 queues.
//
// chainSegMap (offset into chainData -> guest physical address of that
// segment's first byte, in increasing offset order) mirrors
// PS2Memory::PendingTransfer::chainSegMap for GIF/VIF1: diagnostics only, lets
// a later parse offset (e.g. an MPG upload's source bytes) be mapped back to
// the EE address it was DMA'd from instead of being reported as unknown.
void ps2xEnqueueVif0Chain(std::vector<uint8_t> &&chainData,
                          std::vector<std::pair<uint32_t, uint32_t>> &&chainSegMap);
void ps2xEnqueueVif0Transfer(bool fromScratchpad, uint32_t srcAddr, uint32_t qwc);
bool ps2xHasPendingVif0();
void ps2xDrainVif0Transfers(PS2Memory &mem);

#endif // PS2_VIF0_H
