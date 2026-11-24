#ifndef BOTW_FLY_BRIDGE_H
#define BOTW_FLY_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Local, little-endian x64 wire protocol. Version 1 uses fixed-size messages. */
#define BOTW_FLY_MAGIC 0x59465742u /* "BWFY" */
#define BOTW_FLY_PROTOCOL_VERSION 1u
#define BOTW_FLY_PIPE_NAME L"\\\\.\\pipe\\botw_fly_bridge"
#define BOTW_FLY_MAX_ENEMIES 16u

typedef enum BotwFlyMessageType {
    BOTW_FLY_MSG_COMMAND = 1,
    BOTW_FLY_MSG_TELEMETRY = 2
} BotwFlyMessageType;

typedef enum BotwFlyButton {
    BOTW_FLY_A = 0, BOTW_FLY_B, BOTW_FLY_X, BOTW_FLY_Y,
    BOTW_FLY_LSTICK, BOTW_FLY_RSTICK, BOTW_FLY_L, BOTW_FLY_R,
    BOTW_FLY_ZL, BOTW_FLY_ZR, BOTW_FLY_PLUS, BOTW_FLY_MINUS,
    BOTW_FLY_LEFT, BOTW_FLY_UP, BOTW_FLY_RIGHT, BOTW_FLY_DOWN
} BotwFlyButton;

#define BOTW_FLY_BUTTON_MASK(button) (1ull << (unsigned)(button))

enum {
    BOTW_FLY_CMD_OVERRIDE = 1u << 0, /* Replace host buttons/sticks, not merge. */
    BOTW_FLY_CMD_ALLOW_UI = 1u << 1  /* Explicitly permit injection in menus/focus loss. */
};

enum {
    BOTW_FLY_VALID_POSITION  = 1u << 0,
    BOTW_FLY_VALID_VELOCITY  = 1u << 1,
    BOTW_FLY_VALID_ENEMIES   = 1u << 2,
    BOTW_FLY_VALID_VITALS    = 1u << 3,
    BOTW_FLY_VALID_EQUIPMENT = 1u << 4,
    BOTW_FLY_VALID_AUDIO     = 1u << 5
};

#pragma pack(push, 1)
typedef struct BotwFlyWireHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t type;
    uint32_t payload_bytes;
    uint32_t flags;
    uint64_t sequence;
} BotwFlyWireHeader;

/* One client -> game command. Axes are normalized to [-1.0, +1.0]. */
typedef struct BotwFlyCommandPacket {
    BotwFlyWireHeader header;
    uint64_t target_frame; /* 0 = next input_frame; otherwise absolute input_frame id. */
    uint32_t hold_frames;  /* Clamped to [1, 120]. */
    uint32_t command_flags;
    uint64_t buttons;      /* Bits 0..15 match BotwFlyButton and Savage Wind's HLE enum. */
    float left_x;
    float left_y;
    float right_x;
    float right_y;
} BotwFlyCommandPacket;

typedef struct BotwFlyEnemy {
    uint32_t actor_type_hash; /* FNV-1a of actor type name; 0 if unknown. */
    uint32_t attack_state;    /* Game/version-specific enum chosen by the extractor. */
    float relative_x;
    float relative_y;
    float relative_z;
    float distance;
    int32_t animation_frame;  /* -1 if unavailable. */
    uint32_t flags;
} BotwFlyEnemy;

typedef struct BotwFlyTelemetryPacket {
    BotwFlyWireHeader header;
    uint64_t frame_id;
    uint64_t timestamp_ns;       /* Savage Wind monotonic clock domain. */
    uint64_t ack_sequence;        /* Last accepted command sequence. */
    uint64_t applied_sequence;    /* Last command consumed at input_frame boundary. */
    uint32_t valid_mask;
    uint32_t audio_flags;
    uint32_t enemy_count;
    uint32_t ack_status;          /* 0 none, 1 accepted, 2 queue full, 3 malformed. */
    float position[3];            /* Link world/local coordinates as exposed by PlayerInfo. */
    float velocity[3];            /* Derived from consecutive valid positions. */
    float health_current;
    float health_max;
    float stamina_current;
    float stamina_max;
    uint32_t equipped_weapon_id;
    uint32_t equipped_shield_id;
    uint32_t equipped_bow_id;
    uint32_t reserved;
    BotwFlyEnemy enemies[BOTW_FLY_MAX_ENEMIES];
} BotwFlyTelemetryPacket;
#pragma pack(pop)

/* HLE-side command state (host byte order; not sent directly over the pipe). */
typedef struct BotwFlyInput {
    uint64_t sequence;
    uint64_t buttons;
    uint32_t flags;
    int32_t left_x;
    int32_t left_y;
    int32_t right_x;
    int32_t right_y;
} BotwFlyInput;

/* Called at the start of input_frame(frame), on the game/runtime thread. */
void botw_bridge_begin_frame(uint64_t frame);
/* Returns 1 while an unexpired command is active; call from fill_state(). */
int botw_bridge_get_input(BotwFlyInput *out);
/* Release active input and discard queued commands on guest reboot/reset. */
void botw_bridge_reset_input(void);

/* Called on the presenting thread. Pass valid=0 to invalidate position/velocity. */
void botw_bridge_publish_link_state(uint64_t timestamp_ns, const float position[3], int valid);

/* Extension points for verified, game-version-specific extractors. Call each frame. */
void botw_bridge_publish_vitals(float health_current, float health_max,
                                float stamina_current, float stamina_max, int valid);
void botw_bridge_publish_equipment(uint32_t weapon_id, uint32_t shield_id,
                                   uint32_t bow_id, int valid);
void botw_bridge_publish_enemies(const BotwFlyEnemy *enemies, uint32_t count, int valid);
void botw_bridge_publish_audio(uint32_t audio_flags, int valid);

#ifdef __cplusplus
}
#endif

#endif /* BOTW_FLY_BRIDGE_H */
