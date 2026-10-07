import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"python"))
from azul_ai.web_game import main
if __name__ == "__main__":
    main()
