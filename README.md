# Savage Wind BOTW Fly Bridge

A Windows-local IPC bridge for the Savage Wind Recomp runtime. It uses a duplex, message-mode Windows Named Pipe and a dedicated overlapped-I/O worker thread. The game's input path only drains a bounded SPSC command queue at `input_frame()` and copies a small controller state during the Npad HLE poll.

> **Status / scope:** the transport, command queue, Switch-style input override, and Link position/derived-velocity path are implemented. Enemy lists, health/stamina, equipment IDs, and sound-event flags are exposed as typed publisher APIs, but this bundle does **not** invent their BOTW 1.9.0 memory offsets or actor/audio hooks. Those channels remain invalid until a verified game-version-specific extractor calls the publisher functions. Check `valid_mask` before using any telemetry field.

## Files

- `runtime/botw_fly_bridge.h` — C ABI and fixed-size v1 wire protocol.
- `runtime/botw_fly_bridge.cpp` — Windows Named Pipe server, overlapped worker, bounded SPSC queue, telemetry snapshot, and non-Windows no-op stubs.
- `tools/apply_botw_fly_bridge.py` — exact-anchor integration patcher for Savage Wind's active `test` branch.
- `client/fly_bridge_client.cpp` — small Windows client for a one-command smoke test.

## Integration

Use the repository's active `test` branch, not an old release branch. The upstream contribution guide says development and pull requests target `test`.

1. Update your SavageWindRecomp checkout to the latest `test` branch.
2. Copy this bundle's `runtime/botw_fly_bridge.h` and `runtime/botw_fly_bridge.cpp` into the repository's `runtime/` folder.
3. Copy `tools/apply_botw_fly_bridge.py` into the repository's `tools/` folder.
4. From the repository root, run:

   ```powershell
   python tools\apply_botw_fly_bridge.py
   git diff --check
   git diff
   ```

5. Review the diff. The script modifies `Makefile`, `runtime/hle/input.c`, and `runtime/perfmon.c`. It refuses to apply if an anchor is missing, duplicated, or already patched.

The patcher adds `botw_fly_bridge.o` to the executable and `runtime-objs` targets. The existing Makefile's generic `runtime/%.cpp` rule compiles it; the Windows link flags already include `-ladvapi32` for SDDL security descriptor functions.

## Windows 11 build

Savage Wind's documented Windows toolchain is **MSYS2 UCRT64 + Clang/LLD**, not a stock MSVC project. The repository's Windows setup script installs the required SDL3, Vulkan, FFmpeg, and compiler dependencies.

From PowerShell at the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1
```

Then open an MSYS2 UCRT64 shell in the repository and run:

```bash
export PATH="/c/msys64/ucrt64/bin:/c/msys64/usr/bin:$PATH"
make runtime-objs
make -j
```

If this is a fresh setup without generated game code, follow Savage Wind's existing instructions for providing your own supported game files and running `make gen` once. This bridge does not contain Nintendo game data or keys.

Compile the sample client from the MSYS2 UCRT64 shell:

```bash
clang++ -std=c++17 -O2 -Iruntime client/fly_bridge_client.cpp -o build/fly_bridge_client.exe
```

Launch Savage Wind with the bridge enabled. In PowerShell:

```powershell
$env:SAVAGEWIND_BOTW_BRIDGE = "1"
.\build\savagewind.exe
```

The server is started lazily on the first runtime input frame. The endpoint is:

```text
\\.\pipe\botw_fly_bridge
```

One local client is accepted at a time. The pipe rejects remote clients and grants access to the local interactive-users group, Administrators, and SYSTEM.

## Smoke test

The sample client sends a one-frame neutral override by default, then reads telemetry until that sequence is acknowledged and applied. To test ZL (button bit 8):

```powershell
.\build\fly_bridge_client.exe 0x100 0 0 0 0 0 1
```

Button bits match Savage Wind's `BTN_*` ordering exactly:

| Bit | Button | Bit | Button |
|---:|---|---:|---|
| 0 | A | 8 | ZL |
| 1 | B | 9 | ZR |
| 2 | X | 10 | PLUS |
| 3 | Y | 11 | MINUS |
| 4 | L-stick click | 12 | LEFT |
| 5 | R-stick click | 13 | UP |
| 6 | L | 14 | RIGHT |
| 7 | R | 15 | DOWN |

The four stick values are normalized floats in `[-1.0, +1.0]`; Y-positive is up, matching Savage Wind's HLE axis convention. `target_frame=0` schedules the command at the next `input_frame` boundary. `hold_frames` is clamped to 1–120. Commands replace the current host input, rather than merging with it. UI/menu injection is disabled by default; set `BOTW_FLY_CMD_ALLOW_UI` explicitly to override that safeguard.

## Protocol v1

All messages are little-endian fixed-size binary records. They begin with a 24-byte `BotwFlyWireHeader`:

- `magic = 0x59465742` (`BWFY`)
- `version = 1`
- `type = 1` command, `type = 2` telemetry
- `payload_bytes` excludes the header
- `sequence` is nonzero for commands; telemetry has an independent monotonic revision

`BotwFlyCommandPacket` is 64 bytes total. It carries `target_frame`, `hold_frames`, command flags, a 16-bit button mask in a `uint64_t`, and four normalized stick floats.

`BotwFlyTelemetryPacket` is 640 bytes total. `valid_mask` determines which fields may be consumed:

- `BOTW_FLY_VALID_POSITION`: `position[3]`
- `BOTW_FLY_VALID_VELOCITY`: `velocity[3]` (derived from consecutive valid PlayerInfo positions and monotonic timestamps)
- `BOTW_FLY_VALID_ENEMIES`: `enemy_count` and `enemies[]`
- `BOTW_FLY_VALID_VITALS`: health/stamina current and maximum values
- `BOTW_FLY_VALID_EQUIPMENT`: equipped weapon/shield/bow IDs
- `BOTW_FLY_VALID_AUDIO`: `audio_flags`

`ack_sequence` means the server accepted the command into its queue. `applied_sequence` means the game thread consumed it at the input-frame boundary. Use the latter for frame-synchronous control. `ack_status` is `0` none, `1` accepted, `2` queue full, `3` malformed packet.

The command queue is bounded (63 usable entries). On overflow the new command is rejected and status 2 is reported. Future-targeted commands are held in order; submit commands with nondecreasing target frames so an earlier future command cannot block a later one.

## Telemetry hook coverage

### Already connected

Savage Wind's existing `perfmon_position()` safely checks runtime state, game heap bounds, and finite values before reading the game-maintained XYZ copy from `PlayerInfo + 0x88`. The patch invokes it from `perfmon_frame()` on the presenting thread, the same context the existing code documents as safe for that read. Velocity is estimated as `(position[n] - position[n-1]) / dt` in game-units per second. Position and velocity are invalidated during title/loading/event states when the existing probe declines to read.

### Extension hooks requiring verified offsets/functions

No verified public field mapping was found in the current runtime for nearby actor lists, enemy attack/animation state, current health/stamina values, equipped item IDs, or semantic audio events. The APIs below are intentionally available but require an extractor built against the local supported 1.9.0 game build:

```cpp
botw_bridge_publish_enemies(enemy_array, enemy_count, 1);
botw_bridge_publish_vitals(health, health_max, stamina, stamina_max, 1);
botw_bridge_publish_equipment(weapon_id, shield_id, bow_id, 1);
botw_bridge_publish_audio(audio_flags, 1);
```

Call them once per frame from the game/runtime thread after validating the corresponding game objects and pointers. Pass `valid=0` when the source is unavailable. `BotwFlyEnemy::actor_type_hash` should be FNV-1a over a stable actor type string, `attack_state` is an extractor-defined enum, and `animation_frame` is `-1` when unknown. Relative positions are enemy minus Link in the same coordinate basis as `position`.

Do not treat these fields as meaningful when their `valid_mask` bit is clear. In particular, a zero health value or zero enemy count is not evidence that the actual game value is zero unless the matching validity bit is set.

## Latency and scheduling

The pipe I/O is overlapped and runs on a worker thread; the game thread does not call `ReadFile`, `WriteFile`, wait on the pipe, or allocate per command. Queue transfer and controller-state copy are bounded. The telemetry snapshot uses a nonblocking SRW-lock attempt; if the worker is copying a packet at the exact moment a game-thread publisher runs, that telemetry sample is dropped rather than stalling the game thread.

**The <2 ms round-trip target is not a hard real-time guarantee.** Windows scheduling, client wake-up, and the next `input_frame` boundary determine actual latency. A command arriving just after a tick can wait almost one full game frame before `applied_sequence` advances (for example, up to roughly 16.7 ms at 60 Hz). Measure `ack_sequence` and `applied_sequence` timestamps in your client if you need empirical latency numbers. The transport itself is designed to avoid adding a synchronous pipe wait to the render/input path.

## Safety / operational notes

- Only connect to the local pipe; remote clients are rejected.
- Do not set `BOTW_FLY_CMD_ALLOW_UI` for autonomous play unless you intend the agent to operate menus.
- A zero-button command actively releases the injected controller state for its hold duration. Send zero buttons/sticks to release rather than relying on a client disconnect.
- Keep extraction code read-only. Validate guest pointers and heap ranges before dereferencing game memory.
- The bridge is a source modification for a local recompilation; it is not a game-data distribution package.
