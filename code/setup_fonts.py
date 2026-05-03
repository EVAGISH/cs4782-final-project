"""Download Source Serif 4, Inter, and JetBrains Mono into assets/fonts/.

Pulls static TTF files directly from the Google Fonts GitHub mirror so
plot_metrics.py can render figures with the correct typography even on a
machine where the fonts aren't installed system-wide. Run once:

    python code/setup_fonts.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


# (display_name, dest_filename, url) -- pulled from upstream Adobe / JetBrains
# / Google Fonts repos which actually serve these files at predictable paths.
FONTS = [
    ("Inter (variable)",
     "Inter[opsz,wght].ttf",
     "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf"),
    ("Inter Italic (variable)",
     "Inter-Italic[opsz,wght].ttf",
     "https://github.com/google/fonts/raw/main/ofl/inter/Inter-Italic%5Bopsz%2Cwght%5D.ttf"),

    ("Source Serif 4 Regular",
     "SourceSerif4-Regular.ttf",
     "https://github.com/adobe-fonts/source-serif/raw/release/TTF/SourceSerif4-Regular.ttf"),
    ("Source Serif 4 Bold",
     "SourceSerif4-Bold.ttf",
     "https://github.com/adobe-fonts/source-serif/raw/release/TTF/SourceSerif4-Bold.ttf"),
    ("Source Serif 4 Italic",
     "SourceSerif4-It.ttf",
     "https://github.com/adobe-fonts/source-serif/raw/release/TTF/SourceSerif4-It.ttf"),

    ("JetBrains Mono Regular",
     "JetBrainsMono-Regular.ttf",
     "https://github.com/JetBrains/JetBrainsMono/raw/master/fonts/ttf/JetBrainsMono-Regular.ttf"),
    ("JetBrains Mono Medium",
     "JetBrainsMono-Medium.ttf",
     "https://github.com/JetBrains/JetBrainsMono/raw/master/fonts/ttf/JetBrainsMono-Medium.ttf"),
    ("JetBrains Mono Bold",
     "JetBrainsMono-Bold.ttf",
     "https://github.com/JetBrains/JetBrainsMono/raw/master/fonts/ttf/JetBrainsMono-Bold.ttf"),
]


def main():
    out_dir = Path("assets/fonts")
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for display, fname, url in FONTS:
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [skip] {display}  ({fname} already present)")
            continue
        try:
            print(f"  [get ] {display}  -> {dest}")
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"  [FAIL] {display}: {e}")
            failed.append(display)

    if failed:
        print()
        print(f"WARNING: {len(failed)} fonts failed to download. "
              f"matplotlib will fall back to system fonts for those families.")
        sys.exit(1)
    else:
        print()
        print(f"Done. Drop more TTFs into {out_dir}/ if you want additional weights.")


if __name__ == "__main__":
    main()
