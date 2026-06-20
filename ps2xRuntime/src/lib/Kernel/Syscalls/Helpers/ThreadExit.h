#pragma once
#include <exception>

// Canonical ThreadExitException — outside any anonymous namespace so that
// ps2_scheduler.cpp can include this header and catch the same type.
// Previously this was inside namespace {} in Runtime.h which made the catch in
// fiber_trampoline a silent no-op (two distinct types with identical layout).
struct ThreadExitException final : public std::exception
{
    const char* what() const noexcept override
    {
        return "PS2 Thread Exit";
    }
};
