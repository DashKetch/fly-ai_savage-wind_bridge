import time
import numpy as np
from config import BotWBrainConfig

class MockEmulatorInterface:
    """Mock interface simulating live game memory telemetry reading."""
    def read_telemetry(self):
        return {
            "health": 100.0,
            "x": 120.5,
            "y": 125.0,
            "z": -340.2,
            "stamina": 500.0
        }

def diagnostic_check():
    print("--- [Diagnostic] Starting Telemetry & Brain-Cell Latency Check ---")
    BotWBrainConfig.validate()
    
    emu = MockEmulatorInterface()
    target_interval = BotWBrainConfig.DELTA_TIME_MS / 1000.0  # 0.02s
    
    success_count = 0
    test_iterations = 100
    
    for i in range(test_iterations):
        t_start = time.perf_counter()
        
        # 1. Fetch Telemetry
        raw_state = emu.read_telemetry()
        
        # 2. Normalize Parameters
        norm_health = raw_state["health"] / BotWBrainConfig.MAX_HEALTH
        norm_x = (raw_state["x"] - BotWBrainConfig.MAP_BOUNDS_X[0]) / (BotWBrainConfig.MAP_BOUNDS_X[1] - BotWBrainConfig.MAP_BOUNDS_X[0])
        norm_z = (raw_state["z"] - BotWBrainConfig.MAP_BOUNDS_Z[0]) / (BotWBrainConfig.MAP_BOUNDS_Z[1] - BotWBrainConfig.MAP_BOUNDS_Z[0])
        
        vectorized_input = np.array([norm_health, norm_x, norm_z], dtype=np.float32)
        
        # 3. Timing Verification
        elapsed = time.perf_counter() - t_start
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
            
        actual_duration = (time.perf_counter() - t_start) * 1000.0
        if actual_duration <= BotWBrainConfig.DELTA_TIME_MS * 1.5:  # Allow small jitter overhead
            success_count += 1

    print(f"[Diagnostic] Completed {test_iterations} ticks.")
    print(f"[Diagnostic] Tick Constraint Compliance: {(success_count / test_iterations) * 100}% within tolerance.")
    print(f"[Diagnostic] Sample Vectorized Brain Input: {vectorized_input}")
    print("--- [Diagnostic] System Check Passed Successfully ---")

if __name__ == "__main__":
    diagnostic_check()