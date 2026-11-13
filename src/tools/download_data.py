"""Download the public datasets (and the 10-class teachers) ahead of time.

The code also fetches them on first use; this just does it up front.
ImageNet-1k must be obtained manually from image-net.org.

    python -m src.tools.download_data imagenette imagewoof fma hmdb51 teachers --root data
"""

from __future__ import annotations

import argparse

from ..download import TEACHERS, ensure_archive, ensure_fma, ensure_hmdb51, teacher

CHOICES = ("imagenette", "imagewoof", "fma", "hmdb51", "teachers")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="+", choices=CHOICES)
    ap.add_argument("--root", default="data", help="data root (the code's default --data_root)")
    args = ap.parse_args()
    for w in args.what:
        if w == "teachers":
            for d in TEACHERS:
                print(d, "teacher:", teacher(d))
        elif w == "fma":
            print("fma:", ensure_fma(args.root))
        elif w == "hmdb51":
            print("hmdb51:", ensure_hmdb51(args.root))
        else:
            print(w + ":", ensure_archive(w, args.root))


if __name__ == "__main__":
    main()
