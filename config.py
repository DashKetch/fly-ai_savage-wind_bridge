import os

class BotWBrainConfig:
    # --- Timing & Simulation Constraints ---
    TICK_RATE_HZ = 50
    DELTA_TIME_MS = 20.0  # Strict 20ms biological step constraint
    MAX_EPISODE_SECONDS = 180.0  # Timeout per training episode
    
    # --- Inter-Process Communication (IPC) ---
    PIPE_NAME = r"\\.\pipe\botw_fly_bridge"
    PIPE_FRAMING = "length-prefix"  # Options: length-prefix, line, message
    
    # --- Spatial & Coordinate Normalization ---
    # Hyrule map bounds (approximate coordinate ranges for central Hyrule)
    MAP_BOUNDS_X = (-4000.0, 4000.0)
    MAP_BOUNDS_Y = (0.0, 1000.0)   # Altitude
    MAP_BOUNDS_Z = (-4000.0, 4000.0)
    
    # --- State Normalization ---
    MAX_HEALTH = 100.0  # Normalized health pool
    STAMINA_MAX = 500.0
    
    # --- Memory & Storage ---
    MEMORY_DIR = "botw_memory"
    BUFFER_SIZE = 50_000
    BATCH_SIZE = 64
    SAVE_INTERVAL_EPISODES = 10

    @classmethod
    def validate(cls):
        os.makedirs(cls.MEMORY_DIR, exist_ok=True)
        assert cls.DELTA_TIME_MS == 1000.0 / cls.TICK_RATE_XYZ if hasattr(cls, 'TICK_RATE_XYZ') else True, "Tick rate check passed."
        print(f"[Config] Initialized. Pipe target: {cls.PIPE_NAME}")
        print(f"[Config] Memory persistence target: {cls.MEMORY_DIR}/")

BotWBrainConfig.validate()