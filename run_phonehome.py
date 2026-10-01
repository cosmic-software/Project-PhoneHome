"""Entry point for headless Blender:

    blender --background --factory-startup --python run_phonehome.py -- [options]

Run with `-- --help` for the options.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phonehome.cli import main  # noqa: E402

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        # Blender keeps running after a --python script raises; exit with the script's code.
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if e.code not in (None, 0) and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
        sys.stdout.flush()
        os._exit(code)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
