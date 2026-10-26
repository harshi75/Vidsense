"""Central configuration for the VidSense mini project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# --- synthetic video settings -------------------------------------------------
FRAME_SIZE = 32          # pixels (square, grayscale)
N_FRAMES = 16            # frames per clip
CLASSES = ["static", "horizontal", "vertical", "circular"]
GROUPS = ["bright_bg", "dark_bg"]   # nuisance attribute (lighting condition)

# --- dataset sizes ------------------------------------------------------------
N_TRAIN, N_VAL, N_TEST = 960, 240, 480
SPURIOUS_P = 0.90        # P(bright | class in {static, horizontal}) in train/val
                         # P(bright | class in {vertical, circular}) = 1 - SPURIOUS_P
SEEDS = [0, 1, 2]

# --- training -----------------------------------------------------------------
EPOCHS = 10
BATCH_SIZE = 32
LR = 2e-3
