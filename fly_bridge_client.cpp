#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include "../runtime/botw_fly_bridge.h"

static void print_usage(const char *exe) {
    fprintf(stderr,
        "Usage: %s [buttons_hex [left_x left_y right_x right_y [target_frame [hold_frames]]]]\n"
        "Axes are floats in [-1,1]. Example: %s 0x100 0 0 0 0 0 1  (ZL for next frame)\n",
        exe, exe);
}

static bool read_exact(HANDLE pipe, void *buffer, DWORD size) {
    DWORD got = 0;
    if (!ReadFile(pipe, buffer, size, &got, nullptr)) {
        fprintf(stderr, "ReadFile failed: %lu\n", GetLastError());
        return false;
    }
    if (got != size) {
        fprintf(stderr, "Short read: %lu / %lu bytes\n", got, size);
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc > 8) { print_usage(argv[0]); return 2; }

    uint64_t buttons = argc > 1 ? _strtoui64(argv[1], nullptr, 0) : 0;
    float axes[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4 && argc > 2 + i; ++i) axes[i] = (float)strtod(argv[2 + i], nullptr);
    uint64_t target_frame = argc > 6 ? _strtoui64(argv[6], nullptr, 0) : 0;
    uint32_t hold_frames = argc > 7 ? (uint32_t)strtoul(argv[7], nullptr, 0) : 1;

    if (!WaitNamedPipeW(BOTW_FLY_PIPE_NAME, 5000)) {
        fprintf(stderr, "Bridge pipe not available (error %lu). Start Savage Wind with SAVAGEWIND_BOTW_BRIDGE=1.\n", GetLastError());
        return 1;
    }
    HANDLE pipe = CreateFileW(BOTW_FLY_PIPE_NAME, GENERIC_READ | GENERIC_WRITE, 0,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (pipe == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "CreateFileW failed: %lu\n", GetLastError());
        return 1;
    }
    DWORD mode = PIPE_READMODE_MESSAGE;
    if (!SetNamedPipeHandleState(pipe, &mode, nullptr, nullptr)) {
        fprintf(stderr, "SetNamedPipeHandleState failed: %lu\n", GetLastError());
        CloseHandle(pipe);
        return 1;
    }

    static uint64_t sequence = 1;
    BotwFlyCommandPacket command{};
    command.header.magic = BOTW_FLY_MAGIC;
    command.header.version = BOTW_FLY_PROTOCOL_VERSION;
    command.header.type = BOTW_FLY_MSG_COMMAND;
    command.header.payload_bytes = (uint32_t)(sizeof(command) - sizeof(command.header));
    command.header.sequence = sequence++;
    command.target_frame = target_frame;
    command.hold_frames = hold_frames;
    command.command_flags = BOTW_FLY_CMD_OVERRIDE;
    command.buttons = buttons & 0xFFFFull;
    command.left_x = axes[0];
    command.left_y = axes[1];
    command.right_x = axes[2];
    command.right_y = axes[3];

    DWORD sent = 0;
    if (!WriteFile(pipe, &command, sizeof(command), &sent, nullptr) || sent != sizeof(command)) {
        fprintf(stderr, "WriteFile command failed: %lu (%lu bytes)\n", GetLastError(), sent);
        CloseHandle(pipe);
        return 1;
    }

    BotwFlyTelemetryPacket telemetry{};
    bool applied = false;
    for (unsigned attempt = 0; attempt < 300; ++attempt) {
        if (!read_exact(pipe, &telemetry, sizeof(telemetry))) {
            CloseHandle(pipe);
            return 1;
        }
        if (telemetry.header.magic != BOTW_FLY_MAGIC || telemetry.header.version != BOTW_FLY_PROTOCOL_VERSION ||
            telemetry.header.type != BOTW_FLY_MSG_TELEMETRY) {
            fprintf(stderr, "Received an incompatible telemetry packet.\n");
            CloseHandle(pipe);
            return 1;
        }
        if (telemetry.ack_sequence == command.header.sequence &&
            telemetry.applied_sequence == command.header.sequence) {
            applied = true;
            break;
        }
    }
    if (!applied) fprintf(stderr, "Command was not confirmed as applied within 300 telemetry packets.\n");

    printf("ack=%llu applied=%llu status=%u frame=%llu valid=0x%02x\n",
           (unsigned long long)telemetry.ack_sequence,
           (unsigned long long)telemetry.applied_sequence,
           telemetry.ack_status,
           (unsigned long long)telemetry.frame_id,
           telemetry.valid_mask);
    if (telemetry.valid_mask & BOTW_FLY_VALID_POSITION) {
        printf("Link pos: %.3f %.3f %.3f\n", telemetry.position[0], telemetry.position[1], telemetry.position[2]);
    } else {
        printf("Link position not currently valid.\n");
    }
    if (telemetry.valid_mask & BOTW_FLY_VALID_VELOCITY) {
        printf("Link vel: %.3f %.3f %.3f\n", telemetry.velocity[0], telemetry.velocity[1], telemetry.velocity[2]);
    }
    if (telemetry.valid_mask & BOTW_FLY_VALID_ENEMIES)
        printf("Enemy samples: %u\n", telemetry.enemy_count);
    if (telemetry.valid_mask & BOTW_FLY_VALID_VITALS)
        printf("Health %.2f/%.2f  stamina %.2f/%.2f\n", telemetry.health_current,
               telemetry.health_max, telemetry.stamina_current, telemetry.stamina_max);
    if (telemetry.valid_mask & BOTW_FLY_VALID_EQUIPMENT)
        printf("Equipment IDs: weapon=%u shield=%u bow=%u\n", telemetry.equipped_weapon_id,
               telemetry.equipped_shield_id, telemetry.equipped_bow_id);
    if (telemetry.valid_mask & BOTW_FLY_VALID_AUDIO)
        printf("Audio flags: 0x%08x\n", telemetry.audio_flags);

    CloseHandle(pipe);
    return 0;
}
