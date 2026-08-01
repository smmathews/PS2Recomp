#ifndef PS2_IOP_H
#define PS2_IOP_H

#include <cstdint>
#include <functional>
#include <unordered_map>

class PS2Runtime;

constexpr uint32_t IOP_SID_SNDDRV_COMMAND = 0x00000000u;
constexpr uint32_t IOP_SID_SNDDRV_STATE = 0x00000001u;
constexpr uint32_t IOP_SID_LIBSD = 0x80000701u;

constexpr uint32_t IOP_RPC_SNDDRV_SUBMIT = 0x00000000u;
constexpr uint32_t IOP_RPC_SNDDRV_GET_STATUS_ADDR = 0x00000012u;
constexpr uint32_t IOP_RPC_SNDDRV_GET_ADDR_TABLE = 0x00000013u;

// Callback type for game-specific IOP RPC handlers.
// Return true if the RPC was handled (resultPtr must be set).
// Return false to fall through to the default unknown-RPC handler.
using IopRpcHandlerFn = std::function<bool(
    uint8_t *rdram,
    uint32_t sid, uint32_t rpcNum,
    uint32_t sendBufAddr, uint32_t sendSize,
    uint32_t recvBufAddr, uint32_t recvSize,
    uint32_t &resultPtr)>;

class ps2_iop
{
public:
    ps2_iop();
    ~ps2_iop() = default;

    void init(uint8_t *rdram);
    void reset();

    // Register a game-specific handler for a given IOP service ID (sid).
    // Only one handler per sid. Replaces any previously registered handler.
    void registerRpcHandler(uint32_t sid, IopRpcHandlerFn handler);

    bool handleRPC(PS2Runtime *runtime,
                   uint32_t sid, uint32_t rpcNum,
                   uint32_t sendBufAddr, uint32_t sendSize,
                   uint32_t recvBufAddr, uint32_t recvSize,
                   uint32_t &resultPtr,
                   bool &signalNowaitCompletion);

private:
    uint8_t *m_rdram = nullptr;
    std::unordered_map<uint32_t, IopRpcHandlerFn> m_handlers;
};

#endif
