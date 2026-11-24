import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class EpisodeManager:
    def __init__(self, emulator_interface, config):
        self.emu = emulator_interface
        self.config = config
        self.episode_count = 0
        self.start_time = time.time()
        self.current_step = 0

    def reset_episode(self):
        logging.info(f"[EpisodeManager] Triggering reset for Episode {self.episode_count}...")
        
        # Execute automated reload / warp to test camp coordinates
        # (Assuming emulator hook provides safe state loading or coordinate injection)
        self.emu.load_savestate("test_camp_baseline")
        self.emu.reset_timers()
        
        self.start_time = time.time()
        self.current_step = 0
        self.episode_count += 1
        logging.info("[EpisodeManager] Reset complete. Agent re-spawned.")

    def check_termination(self, current_health, current_position):
        elapsed = time.time() - self.start_time
        self.current_step += 1

        # Check termination criteria
        is_dead = current_health <= 0.0
        is_timeout = elapsed >= self.config.MAX_EPISODE_SECONDS
        out_of_bounds = not (
            self.config.MAP_BOUNDS_X[0] <= current_position['x'] <= self.config.MAP_BOUNDS_X[1] and
            self.config.MAP_BOUNDS_Z[0] <= current_position['z'] <= self.config.MAP_BOUNDS_Z[1]
        )

        if is_dead or is_timeout or out_of_bounds:
            reason = "Death" if is_dead else ("Timeout" if is_timeout else "Out of Bounds")
            logging.info(f"[EpisodeManager] Termination triggered: {reason} (Step: {self.current_step})")
            self.reset_episode()
            return True
            
        return False