#ifndef PS2_GUESTWATCH_H
#define PS2_GUESTWATCH_H

#include <cstdint>

// ---------------------------------------------------------------------------
// ps2_watch — cheap permanent guest-memory transition watches.
//
// A runner registers a handful of guest addresses whose VALUE TRANSITIONS
// matter for boot diagnosis (state-machine words, ready flags, heartbeat
// counters). The runtime's frame loop polls them (~60 Hz — a few word reads,
// effectively free) and prints ONE line per observed change:
//
//   [watch] <label> 0x<addr>: 0x<old> -> 0x<new> (change #N)
//
// Noise bound: after kVerboseChanges changes per watch, only every
// kSparseInterval-th change is printed (so a per-frame counter shows it is
// climbing without flooding the log).
//
// Separate header on purpose: included only by the runner's setup code and
// ps2_runtime.cpp — never by recompiled-function translation units.
// ---------------------------------------------------------------------------
namespace ps2_watch
{
    // byteWidth: 1, 2 or 4. Safe to call before the runtime starts running.
    void addWatch(uint32_t guestAddr, uint32_t byteWidth, const char *label);

    // Called by the runtime frame loop. No-op when no watches are registered.
    void poll(const uint8_t *rdram);
}

#endif // PS2_GUESTWATCH_H
