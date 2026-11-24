#include "botw_fly_bridge.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <sddl.h>

namespace {
constexpr LONG kQueueSize = 64;
constexpr LONG kQueueMask = kQueueSize - 1;
constexpr DWORD kReconnectMs = 300;
constexpr DWORD kWaitMs = 5;

struct QueuedCommand {
    uint64_t target_frame;
    uint64_t sequence;
    uint64_t buttons;
    uint32_t hold_frames;
    uint32_t flags;
    int32_t lx, ly, rx, ry;
};

static_assert((kQueueSize & (kQueueSize - 1)) == 0, "queue size must be a power of two");
static_assert(sizeof(BotwFlyWireHeader) == 24, "wire header layout changed");
static_assert(sizeof(BotwFlyCommandPacket) == 64, "command wire layout changed");
static_assert(sizeof(BotwFlyEnemy) == 32, "enemy wire layout changed");
static_assert(sizeof(BotwFlyTelemetryPacket) == 640, "telemetry wire layout changed");

QueuedCommand g_queue[kQueueSize]{};
volatile LONG g_queue_head = 0; /* worker producer */
volatile LONG g_queue_tail = 0; /* game-thread consumer */

volatile LONG g_start_checked = 0;
volatile LONG g_running = 0;
volatile LONG64 g_ack_sequence = 0;
volatile LONG64 g_applied_sequence = 0;
volatile LONG g_ack_status = 0;
volatile LONG64 g_telemetry_revision = 1;
HANDLE g_worker = nullptr;
HANDLE g_stop_event = nullptr;
HANDLE g_telemetry_event = nullptr;
HANDLE volatile g_pipe_handle = nullptr;

SRWLOCK g_telemetry_lock = SRWLOCK_INIT;
BotwFlyTelemetryPacket g_telemetry{};
float g_previous_position[3]{};
uint64_t g_previous_timestamp_ns = 0;
bool g_previous_position_valid = false;

uint64_t g_current_frame = 0; /* game-thread only */
QueuedCommand g_pending{};
bool g_pending_valid = false;
BotwFlyInput g_active{};
bool g_active_valid = false;
uint64_t g_active_until_frame = 0;

static void signal_telemetry(void) {
    InterlockedIncrement64(&g_telemetry_revision);
    HANDLE event_handle = g_telemetry_event;
    if (event_handle) SetEvent(event_handle);
}

static bool read_env_enabled(void) {
    char value[16]{};
    DWORD n = GetEnvironmentVariableA("SAVAGEWIND_BOTW_BRIDGE", value, (DWORD)sizeof(value));
    if (!n || n >= sizeof(value)) return false;
    return _stricmp(value, "1") == 0 || _stricmp(value, "true") == 0 || _stricmp(value, "yes") == 0;
}

static int32_t axis_to_i16(float value) {
    if (!isfinite(value)) value = 0.0f;
    if (value > 1.0f) value = 1.0f;
    if (value < -1.0f) value = -1.0f;
    long rounded = lroundf(value * 32767.0f);
    if (rounded > 32767) rounded = 32767;
    if (rounded < -32767) rounded = -32767;
    return (int32_t)rounded;
}

static bool enqueue_command(const BotwFlyCommandPacket &packet) {
    const LONG head = InterlockedCompareExchange(&g_queue_head, 0, 0);
    const LONG next = (head + 1) & kQueueMask;
    const LONG tail = InterlockedCompareExchange(&g_queue_tail, 0, 0);
    if (next == tail) return false;

    QueuedCommand command{};
    command.target_frame = packet.target_frame;
    command.sequence = packet.header.sequence;
    command.buttons = packet.buttons & 0xFFFFull;
    command.hold_frames = packet.hold_frames < 1 ? 1 : packet.hold_frames > 120 ? 120 : packet.hold_frames;
    command.flags = packet.command_flags;
    command.lx = axis_to_i16(packet.left_x);
    command.ly = axis_to_i16(packet.left_y);
    command.rx = axis_to_i16(packet.right_x);
    command.ry = axis_to_i16(packet.right_y);
    if (!(command.flags & BOTW_FLY_CMD_OVERRIDE)) command.flags |= BOTW_FLY_CMD_OVERRIDE;

    g_queue[head] = command;
    MemoryBarrier();
    InterlockedExchange(&g_queue_head, next);
    return true;
}

static bool pop_command(QueuedCommand *out) {
    const LONG tail = InterlockedCompareExchange(&g_queue_tail, 0, 0);
    const LONG head = InterlockedCompareExchange(&g_queue_head, 0, 0);
    if (tail == head) return false;
    *out = g_queue[tail];
    MemoryBarrier();
    InterlockedExchange(&g_queue_tail, (tail + 1) & kQueueMask);
    return true;
}

static void handle_command_message(const unsigned char *bytes, DWORD length) {
    if (length != sizeof(BotwFlyCommandPacket)) {
        InterlockedExchange(&g_ack_status, 3);
        signal_telemetry();
        return;
    }
    BotwFlyCommandPacket packet{};
    memcpy(&packet, bytes, sizeof(packet));
    const bool valid = packet.header.magic == BOTW_FLY_MAGIC &&
        packet.header.version == BOTW_FLY_PROTOCOL_VERSION &&
        packet.header.type == BOTW_FLY_MSG_COMMAND &&
        packet.header.payload_bytes == sizeof(packet) - sizeof(packet.header) &&
        packet.header.sequence != 0;
    if (!valid) {
        InterlockedExchange(&g_ack_status, 3);
        signal_telemetry();
        return;
    }
    if (!enqueue_command(packet)) {
        InterlockedExchange64(&g_ack_sequence, (LONG64)packet.header.sequence);
        InterlockedExchange(&g_ack_status, 2);
    } else {
        InterlockedExchange64(&g_ack_sequence, (LONG64)packet.header.sequence);
        InterlockedExchange(&g_ack_status, 1);
    }
    signal_telemetry();
}

static bool telemetry_snapshot(BotwFlyTelemetryPacket *out, uint64_t *revision) {
    const LONG64 before = InterlockedCompareExchange64(&g_telemetry_revision, 0, 0);
    if (!TryAcquireSRWLockShared(&g_telemetry_lock)) return false;
    *out = g_telemetry;
    ReleaseSRWLockShared(&g_telemetry_lock);
    const LONG64 after = InterlockedCompareExchange64(&g_telemetry_revision, 0, 0);
    /* A publisher raced the copy. Drop this read; the signalled revision will be sampled again. */
    if (before != after) return false;
    *revision = (uint64_t)before;
    out->header.magic = BOTW_FLY_MAGIC;
    out->header.version = BOTW_FLY_PROTOCOL_VERSION;
    out->header.type = BOTW_FLY_MSG_TELEMETRY;
    out->header.payload_bytes = (uint32_t)(sizeof(*out) - sizeof(out->header));
    out->header.flags = 0;
    out->header.sequence = *revision;
    out->ack_sequence = (uint64_t)InterlockedCompareExchange64(&g_ack_sequence, 0, 0);
    out->applied_sequence = (uint64_t)InterlockedCompareExchange64(&g_applied_sequence, 0, 0);
    out->ack_status = (uint32_t)InterlockedCompareExchange(&g_ack_status, 0, 0);
    return true;
}

static bool make_security_attributes(SECURITY_ATTRIBUTES *sa, PSECURITY_DESCRIPTOR *descriptor) {
    /* Local interactive users + SYSTEM/admin; PIPE_REJECT_REMOTE_CLIENTS blocks remote SMB clients. */
    *descriptor = nullptr;
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
            L"D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GA;;;IU)", SDDL_REVISION_1,
            descriptor, nullptr)) return false;
    sa->nLength = sizeof(*sa);
    sa->lpSecurityDescriptor = *descriptor;
    sa->bInheritHandle = FALSE;
    return true;
}

static bool wait_overlapped(HANDLE pipe, OVERLAPPED *ov, DWORD *transferred) {
    HANDLE waits[2] = { g_stop_event, ov->hEvent };
    DWORD result = WaitForMultipleObjects(2, waits, FALSE, INFINITE);
    if (result == WAIT_OBJECT_0) {
        CancelIoEx(pipe, ov);
        return false;
    }
    if (result != WAIT_OBJECT_0 + 1) return false;
    DWORD ignored = 0;
    return GetOverlappedResult(pipe, ov, transferred ? transferred : &ignored, FALSE) != FALSE;
}

static bool serve_client(HANDLE pipe) {
    OVERLAPPED read_ov{};
    OVERLAPPED write_ov{};
    read_ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    write_ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!read_ov.hEvent || !write_ov.hEvent) {
        if (read_ov.hEvent) CloseHandle(read_ov.hEvent);
        if (write_ov.hEvent) CloseHandle(write_ov.hEvent);
        return false;
    }

    unsigned char read_buffer[sizeof(BotwFlyCommandPacket)]{};
    BotwFlyTelemetryPacket write_packet{};
    bool read_pending = false;
    bool write_pending = false;
    uint64_t last_sent_revision = 0;
    uint64_t pending_write_revision = 0;
    bool connected = true;

    while (connected && WaitForSingleObject(g_stop_event, 0) != WAIT_OBJECT_0) {
        if (!read_pending) {
            ResetEvent(read_ov.hEvent);
            DWORD bytes = 0;
            BOOL ok = ReadFile(pipe, read_buffer, sizeof(read_buffer), &bytes, &read_ov);
            if (ok) {
                if (bytes) handle_command_message(read_buffer, bytes);
            } else {
                DWORD error = GetLastError();
                if (error == ERROR_IO_PENDING) read_pending = true;
                else if (error == ERROR_MORE_DATA) connected = false;
                else connected = false;
            }
        }

        if (!write_pending) {
            uint64_t revision = 0;
            if (telemetry_snapshot(&write_packet, &revision) && revision != last_sent_revision) {
                ResetEvent(write_ov.hEvent);
                DWORD bytes = 0;
                BOOL ok = WriteFile(pipe, &write_packet, sizeof(write_packet), &bytes, &write_ov);
                if (ok) {
                    if (bytes == sizeof(write_packet)) last_sent_revision = revision;
                    else connected = false;
                } else {
                    DWORD error = GetLastError();
                    if (error == ERROR_IO_PENDING) {
                        write_pending = true;
                        pending_write_revision = revision;
                    } else {
                        connected = false;
                    }
                }
            }
        }

        HANDLE waits[4];
        DWORD count = 0;
        waits[count++] = g_stop_event;
        if (read_pending) waits[count++] = read_ov.hEvent;
        if (write_pending) waits[count++] = write_ov.hEvent;
        if (g_telemetry_event) waits[count++] = g_telemetry_event;
        DWORD result = WaitForMultipleObjects(count, waits, FALSE, kWaitMs);
        if (result == WAIT_OBJECT_0) break;

        DWORD index = result == WAIT_TIMEOUT ? MAXDWORD : result - WAIT_OBJECT_0;
        DWORD cursor = 1;
        if (read_pending) {
            if (index == cursor) {
                DWORD bytes = 0;
                if (GetOverlappedResult(pipe, &read_ov, &bytes, FALSE)) {
                    read_pending = false;
                    if (bytes) handle_command_message(read_buffer, bytes);
                } else {
                    connected = false;
                }
            }
            ++cursor;
        }
        if (write_pending) {
            if (index == cursor) {
                DWORD bytes = 0;
                if (GetOverlappedResult(pipe, &write_ov, &bytes, FALSE) && bytes == sizeof(write_packet)) {
                    last_sent_revision = pending_write_revision;
                    write_pending = false;
                } else {
                    connected = false;
                }
            }
            ++cursor;
        }
        /* telemetry event is auto-reset; the next loop snapshots the newest state */
    }

    if (read_pending) CancelIoEx(pipe, &read_ov);
    if (write_pending) CancelIoEx(pipe, &write_ov);
    CloseHandle(read_ov.hEvent);
    CloseHandle(write_ov.hEvent);
    return connected;
}

static DWORD WINAPI pipe_worker(void *) {
    InterlockedExchange(&g_running, 1);
    while (WaitForSingleObject(g_stop_event, 0) != WAIT_OBJECT_0) {
        SECURITY_ATTRIBUTES sa{};
        PSECURITY_DESCRIPTOR descriptor = nullptr;
        if (!make_security_attributes(&sa, &descriptor)) {
            WaitForSingleObject(g_stop_event, kReconnectMs);
            continue;
        }
        HANDLE pipe = CreateNamedPipeW(
            BOTW_FLY_PIPE_NAME,
            PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
            PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
            1, (DWORD)(sizeof(BotwFlyTelemetryPacket) * 4),
            (DWORD)(sizeof(BotwFlyCommandPacket) * 4), 0, &sa);
        LocalFree(descriptor);
        if (pipe == INVALID_HANDLE_VALUE) {
            WaitForSingleObject(g_stop_event, kReconnectMs);
            continue;
        }
        InterlockedExchangePointer((PVOID volatile *)&g_pipe_handle, pipe);

        OVERLAPPED connect_ov{};
        connect_ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!connect_ov.hEvent) {
            CloseHandle(pipe);
            InterlockedExchangePointer((PVOID volatile *)&g_pipe_handle, nullptr);
            continue;
        }
        BOOL connected = ConnectNamedPipe(pipe, &connect_ov);
        DWORD error = connected ? ERROR_SUCCESS : GetLastError();
        if (!connected && error == ERROR_IO_PENDING) {
            connected = wait_overlapped(pipe, &connect_ov, nullptr);
        } else if (!connected && error == ERROR_PIPE_CONNECTED) {
            connected = TRUE;
        } else if (!connected) {
            connected = FALSE;
        }
        CloseHandle(connect_ov.hEvent);

        if (connected && WaitForSingleObject(g_stop_event, 0) != WAIT_OBJECT_0) {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
            serve_client(pipe);
        }
        /* Do not FlushFileBuffers here: it can wait for a slow/disconnected client. */
        DisconnectNamedPipe(pipe);
        CloseHandle(pipe);
        InterlockedExchangePointer((PVOID volatile *)&g_pipe_handle, nullptr);
        if (WaitForSingleObject(g_stop_event, 0) != WAIT_OBJECT_0)
            WaitForSingleObject(g_stop_event, 50);
    }
    InterlockedExchange(&g_running, 0);
    return 0;
}

static void stop_server(void) {
    if (g_stop_event) SetEvent(g_stop_event);
    HANDLE pipe = (HANDLE)InterlockedExchangePointer((PVOID volatile *)&g_pipe_handle, nullptr);
    if (pipe && pipe != INVALID_HANDLE_VALUE) CancelIoEx(pipe, nullptr);
    if (g_worker) {
        WaitForSingleObject(g_worker, 1500);
        CloseHandle(g_worker);
        g_worker = nullptr;
    }
    if (g_telemetry_event) { CloseHandle(g_telemetry_event); g_telemetry_event = nullptr; }
    if (g_stop_event) { CloseHandle(g_stop_event); g_stop_event = nullptr; }
}

static void start_server_once(void) {
    if (InterlockedCompareExchange(&g_start_checked, 1, 0) != 0) return;
    if (!read_env_enabled()) return;
    g_stop_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    g_telemetry_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!g_stop_event || !g_telemetry_event) {
        if (g_stop_event) CloseHandle(g_stop_event), g_stop_event = nullptr;
        if (g_telemetry_event) CloseHandle(g_telemetry_event), g_telemetry_event = nullptr;
        return;
    }
    g_telemetry.header.magic = BOTW_FLY_MAGIC;
    g_telemetry.header.version = BOTW_FLY_PROTOCOL_VERSION;
    g_telemetry.header.type = BOTW_FLY_MSG_TELEMETRY;
    g_telemetry.header.payload_bytes = (uint32_t)(sizeof(g_telemetry) - sizeof(g_telemetry.header));
    g_worker = CreateThread(nullptr, 0, pipe_worker, nullptr, 0, nullptr);
    if (!g_worker) {
        CloseHandle(g_telemetry_event); g_telemetry_event = nullptr;
        CloseHandle(g_stop_event); g_stop_event = nullptr;
        return;
    }
    atexit(stop_server);
}

static bool try_lock_telemetry(void) {
    return TryAcquireSRWLockExclusive(&g_telemetry_lock) != FALSE;
}
static void unlock_telemetry(void) { ReleaseSRWLockExclusive(&g_telemetry_lock); }
} // namespace

extern "C" void botw_bridge_begin_frame(uint64_t frame) {
    start_server_once();
    if (!g_worker) return;
    g_current_frame = frame;

    if (g_active_valid && frame > g_active_until_frame) g_active_valid = false;

    for (;;) {
        QueuedCommand command{};
        if (g_pending_valid) {
            command = g_pending;
        } else if (!pop_command(&command)) {
            break;
        }
        if (command.target_frame && command.target_frame > frame) {
            g_pending = command;
            g_pending_valid = true;
            break;
        }
        g_pending_valid = false;
        g_active.sequence = command.sequence;
        g_active.buttons = command.buttons;
        g_active.flags = command.flags;
        g_active.left_x = command.lx;
        g_active.left_y = command.ly;
        g_active.right_x = command.rx;
        g_active.right_y = command.ry;
        g_active_valid = true;
        g_active_until_frame = frame + (uint64_t)command.hold_frames - 1ull;
        InterlockedExchange64(&g_applied_sequence, (LONG64)command.sequence);
        signal_telemetry();
    }

    if (try_lock_telemetry()) {
        g_telemetry.frame_id = frame;
        /* These values are frame samples, not persistent last-known values. */
        g_telemetry.valid_mask &= ~(BOTW_FLY_VALID_ENEMIES | BOTW_FLY_VALID_VITALS |
                                    BOTW_FLY_VALID_EQUIPMENT | BOTW_FLY_VALID_AUDIO);
        g_telemetry.enemy_count = 0;
        g_telemetry.audio_flags = 0;
        unlock_telemetry();
        signal_telemetry();
    }
}

extern "C" int botw_bridge_get_input(BotwFlyInput *out) {
    if (!out || !g_active_valid || !g_worker) return 0;
    if (g_current_frame > g_active_until_frame) {
        g_active_valid = false;
        return 0;
    }
    *out = g_active;
    return 1;
}

extern "C" void botw_bridge_reset_input(void) {
    g_active_valid = false;
    g_pending_valid = false;
    const LONG head = InterlockedCompareExchange(&g_queue_head, 0, 0);
    InterlockedExchange(&g_queue_tail, head);
}

extern "C" void botw_bridge_publish_link_state(uint64_t timestamp_ns, const float position[3], int valid) {
    if (!g_worker || !try_lock_telemetry()) return;
    if (!valid || !position) {
        g_telemetry.valid_mask &= ~(BOTW_FLY_VALID_POSITION | BOTW_FLY_VALID_VELOCITY);
        memset(g_telemetry.position, 0, sizeof(g_telemetry.position));
        memset(g_telemetry.velocity, 0, sizeof(g_telemetry.velocity));
        g_previous_position_valid = false;
        g_previous_timestamp_ns = 0;
    } else {
        bool finite = true;
        for (int i = 0; i < 3; ++i) finite = finite && isfinite(position[i]);
        if (!finite) {
            g_telemetry.valid_mask &= ~(BOTW_FLY_VALID_POSITION | BOTW_FLY_VALID_VELOCITY);
            g_previous_position_valid = false;
            g_previous_timestamp_ns = 0;
        } else {
            memcpy(g_telemetry.position, position, sizeof(g_telemetry.position));
            g_telemetry.valid_mask |= BOTW_FLY_VALID_POSITION;
            g_telemetry.timestamp_ns = timestamp_ns;
            if (g_previous_position_valid && timestamp_ns > g_previous_timestamp_ns) {
                const double dt = (double)(timestamp_ns - g_previous_timestamp_ns) * 1.0e-9;
                if (dt >= 0.0001 && dt <= 0.5) {
                    for (int i = 0; i < 3; ++i)
                        g_telemetry.velocity[i] = (float)(((double)position[i] - g_previous_position[i]) / dt);
                    g_telemetry.valid_mask |= BOTW_FLY_VALID_VELOCITY;
                } else {
                    g_telemetry.valid_mask &= ~BOTW_FLY_VALID_VELOCITY;
                }
            } else {
                g_telemetry.valid_mask &= ~BOTW_FLY_VALID_VELOCITY;
            }
            memcpy(g_previous_position, position, sizeof(g_previous_position));
            g_previous_timestamp_ns = timestamp_ns;
            g_previous_position_valid = true;
        }
    }
    unlock_telemetry();
    signal_telemetry();
}

extern "C" void botw_bridge_publish_vitals(float health_current, float health_max,
                                              float stamina_current, float stamina_max, int valid) {
    if (!g_worker || !try_lock_telemetry()) return;
    if (valid && isfinite(health_current) && isfinite(health_max) &&
        isfinite(stamina_current) && isfinite(stamina_max)) {
        g_telemetry.health_current = health_current;
        g_telemetry.health_max = health_max;
        g_telemetry.stamina_current = stamina_current;
        g_telemetry.stamina_max = stamina_max;
        g_telemetry.valid_mask |= BOTW_FLY_VALID_VITALS;
    } else {
        g_telemetry.valid_mask &= ~BOTW_FLY_VALID_VITALS;
    }
    unlock_telemetry();
    signal_telemetry();
}

extern "C" void botw_bridge_publish_equipment(uint32_t weapon_id, uint32_t shield_id,
                                                 uint32_t bow_id, int valid) {
    if (!g_worker || !try_lock_telemetry()) return;
    if (valid) {
        g_telemetry.equipped_weapon_id = weapon_id;
        g_telemetry.equipped_shield_id = shield_id;
        g_telemetry.equipped_bow_id = bow_id;
        g_telemetry.valid_mask |= BOTW_FLY_VALID_EQUIPMENT;
    } else {
        g_telemetry.valid_mask &= ~BOTW_FLY_VALID_EQUIPMENT;
    }
    unlock_telemetry();
    signal_telemetry();
}

extern "C" void botw_bridge_publish_enemies(const BotwFlyEnemy *enemies, uint32_t count, int valid) {
    if (!g_worker || !try_lock_telemetry()) return;
    if (valid && (enemies || count == 0)) {
        if (count > BOTW_FLY_MAX_ENEMIES) count = BOTW_FLY_MAX_ENEMIES;
        g_telemetry.enemy_count = count;
        if (count) memcpy(g_telemetry.enemies, enemies, sizeof(BotwFlyEnemy) * count);
        if (count < BOTW_FLY_MAX_ENEMIES)
            memset(g_telemetry.enemies + count, 0, sizeof(BotwFlyEnemy) * (BOTW_FLY_MAX_ENEMIES - count));
        g_telemetry.valid_mask |= BOTW_FLY_VALID_ENEMIES;
    } else {
        g_telemetry.enemy_count = 0;
        g_telemetry.valid_mask &= ~BOTW_FLY_VALID_ENEMIES;
    }
    unlock_telemetry();
    signal_telemetry();
}

extern "C" void botw_bridge_publish_audio(uint32_t audio_flags, int valid) {
    if (!g_worker || !try_lock_telemetry()) return;
    if (valid) {
        g_telemetry.audio_flags = audio_flags;
        g_telemetry.valid_mask |= BOTW_FLY_VALID_AUDIO;
    } else {
        g_telemetry.audio_flags = 0;
        g_telemetry.valid_mask &= ~BOTW_FLY_VALID_AUDIO;
    }
    unlock_telemetry();
    signal_telemetry();
}

#else /* non-Windows builds: retain the C ABI and compile as no-op stubs */
extern "C" void botw_bridge_begin_frame(uint64_t) {}
extern "C" int botw_bridge_get_input(BotwFlyInput *) { return 0; }
extern "C" void botw_bridge_reset_input(void) {}
extern "C" void botw_bridge_publish_link_state(uint64_t, const float *, int) {}
extern "C" void botw_bridge_publish_vitals(float, float, float, float, int) {}
extern "C" void botw_bridge_publish_equipment(uint32_t, uint32_t, uint32_t, int) {}
extern "C" void botw_bridge_publish_enemies(const BotwFlyEnemy *, uint32_t, int) {}
extern "C" void botw_bridge_publish_audio(uint32_t, int) {}
#endif
