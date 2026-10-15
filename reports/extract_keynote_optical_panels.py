"""Extract unmodified, annotated optical panels from a Keynote PDF export.

Usage: conda run -n base python reports/extract_keynote_optical_panels.py \
    --source /path/to/keynote-source.pdf --out-dir /path/to/article/reports/figures
Source: cape_cod_bay2.key, slides 2 and 4 (1920 x 1080 points).
Cropping removes slide text and embedded credits, retained in the caption.
The PDF crops are unmodified. PNG copies for Rustplotlib are rendered at
144 dpi, retaining the original image and annotations without enhancement.
"""
import argparse
import subprocess
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        (1, (1309, 380, 1920, 1080), 'optical_mechanicville_cropped.pdf'),
        (3, (1309, 180, 1920, 880), 'optical_dashcam_tynan_cropped.pdf'),
    ]
    for index, box, name in panels:
        page = PdfReader(args.source).pages[index]
        assert tuple(map(float, page.mediabox)) == (0., 0., 1920., 1080.)
        page.cropbox = RectangleObject(box)
        page.mediabox = RectangleObject(box)
        writer = PdfWriter()
        writer.add_page(page)
        with (args.out_dir / name).open('wb') as stream:
            writer.write(stream)
        print(args.out_dir / name)
        subprocess.run(['pdftoppm', '-r', '144', '-singlefile', '-png',
                        str(args.out_dir / name), str((args.out_dir / name).with_suffix(''))],
                       check=True)


if __name__ == '__main__':
    main()
