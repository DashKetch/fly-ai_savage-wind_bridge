# BOTW Fly Bridge protocol quick reference

Pipe: `\\.\pipe\botw_fly_bridge`  
Mode: duplex, message type, local-only, v1 fixed-size packets.

## Command (`type = 1`, 64 bytes)

`BotwFlyCommandPacket` in `runtime/botw_fly_bridge.h`.

- `header.sequence`: monotonically increasing nonzero command ID.
- `target_frame`: 0 for next HLE input frame, otherwise absolute `input_frame(frame)` value.
- `hold_frames`: 1..120.
- `command_flags`: `BOTW_FLY_CMD_OVERRIDE` by default; optional `BOTW_FLY_CMD_ALLOW_UI`.
- `buttons`: bits 0..15 (`A B X Y LSTICK RSTICK L R ZL ZR PLUS MINUS LEFT UP RIGHT DOWN`).
- `left_x`, `left_y`, `right_x`, `right_y`: floats in [-1,1].

## Telemetry (`type = 2`, 640 bytes)

`BotwFlyTelemetryPacket` in `runtime/botw_fly_bridge.h`.

- `header.sequence`: telemetry revision, not command sequence.
- `frame_id`: last `input_frame(frame)` processed.
- `timestamp_ns`: monotonic timestamp of Link position sample.
- `ack_sequence`: command accepted into queue.
- `applied_sequence`: command reached the game-thread input boundary.
- `valid_mask`: bitfield; only read groups whose bit is set.
- `enemies[16]`: actor type hash, attack-state enum, relative XYZ, distance, animation frame, flags.
- `audio_flags`: extractor-defined bitset.

Do not read invalid fields. Enemy/vitals/equipment/audio fields are reserved for version-specific game extractors and are not populated by the base patch.
