"""Draw the VankeDoodle logo as SVG.

The lettering is converted to outlines, so the SVG needs no fonts installed and
looks the same everywhere. The background is transparent, so the logo sits on
the app's own background colour. The outlines come from STIX General (SIL Open
Font License): upright for "Van-" and "Doodle", italic for k_e, the way a
variable is set in a scientific paper.
"""
import math
import os
import sys

import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.path import Path

# matplotlib ships the STIX fonts, so this runs anywhere matplotlib is installed
_FONTS = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
SERIF = os.path.join(_FONTS, "STIXGeneral.ttf")
STIX_ITALIC = os.path.join(_FONTS, "STIXGeneralItalic.ttf")
TEXT = "#D1D5DB"        # the app's text colour
ACCENT = "#00D2FF"      # the app's primary colour

def glyphs(text, font, size):
    tp = TextPath((0, 0), text, size=size, prop=FontProperties(fname=font))
    return tp

def to_d(path, dx, dy):
    """SVG path data for a matplotlib path, flipped to SVG's downward y."""
    out, verts, codes = [], path.vertices, path.codes
    i = 0
    while i < len(verts):
        c = codes[i]
        x, y = verts[i]
        if c == Path.MOVETO:
            out.append(f"M{dx + x:.2f} {dy - y:.2f}"); i += 1
        elif c == Path.LINETO:
            out.append(f"L{dx + x:.2f} {dy - y:.2f}"); i += 1
        elif c == Path.CURVE3:
            (x1, y1), (x2, y2) = verts[i], verts[i + 1]
            out.append(f"Q{dx + x1:.2f} {dy - y1:.2f} {dx + x2:.2f} {dy - y2:.2f}"); i += 2
        elif c == Path.CURVE4:
            (x1, y1), (x2, y2), (x3, y3) = verts[i], verts[i + 1], verts[i + 2]
            out.append(f"C{dx + x1:.2f} {dy - y1:.2f} {dx + x2:.2f} {dy - y2:.2f} {dx + x3:.2f} {dy - y3:.2f}"); i += 3
        elif c == Path.CLOSEPOLY:
            out.append("Z"); i += 1
        else:
            i += 1
    return " ".join(out)

def width(tp):
    return tp.get_extents().x1 if len(tp.vertices) else 0.0

def feather(x0, y0, length, rise, k=3.2, n=26, colour=ACCENT):
    """A quill whose shaft is an exponential decay curve, C(t) = e^(-k t): high on
    the left, falling to the right. Barbs sweep back from the shaft; the bare quill
    end is at the right."""
    def spine(s):
        return x0 + length * s, y0 - rise * math.exp(-k * s)
    pts = [spine(i / 80) for i in range(81)]
    d = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in pts)
    parts = [f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2.2" stroke-linecap="round"/>']
    for i in range(n):
        s = 0.04 + 0.70 * i / (n - 1)                 # vane along the first 3/4
        x, y = spine(s)
        # tangent and normal of the shaft at s
        dx, dy = length, rise * k * math.exp(-k * s)
        L = math.hypot(dx, dy); tx, ty = dx / L, dy / L
        nx, ny = -ty, tx
        w = 15 * math.sin(math.pi * min(1, (s + 0.05) / 0.8)) ** 0.7 + 2
        for side in (1, -1):
            # barbs angle back toward the tip of the plume (left)
            ex = x + side * nx * w - tx * w * 0.9
            ey = y + side * ny * w - ty * w * 0.9
            parts.append(f'<path d="M{x:.2f} {y:.2f} Q{(x + ex) / 2 - tx * 2:.2f} {(y + ey) / 2 - ty * 2:.2f} {ex:.2f} {ey:.2f}" '
                         f'fill="none" stroke="{colour}" stroke-width="1.1" stroke-opacity="0.55" stroke-linecap="round"/>')
    return "\n  ".join(parts)

def wordmark(size=64, baseline=100, x=120):
    van = glyphs("Van-", SERIF, size)
    k = glyphs("k", STIX_ITALIC, size * 1.02)
    e = glyphs("e", STIX_ITALIC, size * 0.55)
    doodle = glyphs(" Doodle", SERIF, size)
    parts, cx = [], x
    parts.append(f'<path d="{to_d(van, cx, baseline)}" fill="{TEXT}"/>'); cx += width(van) + 1
    parts.append(f'<path d="{to_d(k, cx, baseline)}" fill="{ACCENT}"/>'); cx += width(k) - 1
    parts.append(f'<path d="{to_d(e, cx, baseline + size * 0.16)}" fill="{ACCENT}"/>'); cx += width(e) + 2
    parts.append(f'<path d="{to_d(doodle, cx, baseline)}" fill="{TEXT}"/>'); cx += width(doodle)
    return "\n  ".join(parts), cx

def logo():
    body, right = wordmark()
    W, H = right + 14, 140
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H}" role="img" aria-label="VankeDoodle">
  <title>Van-kₑ Doodle</title>
  {feather(14, 112, 92, 84)}
  {body}
</svg>
'''

def icon():
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" role="img" aria-label="VankeDoodle">
  <title>Van-kₑ Doodle</title>
  {feather(12, 98, 96, 82)}
</svg>
'''

if __name__ == "__main__":
    # python assets/make_logo.py assets   (rewrites the two SVGs in place)
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    open(f"{out}/vankedoodle_logo.svg", "w").write(logo())
    open(f"{out}/vankedoodle_icon.svg", "w").write(icon())
