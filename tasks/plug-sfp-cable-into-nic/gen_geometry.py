"""Builds geometry.json (box tables for the NIC card with its SFP cages and for the SFP module) from the AIC SDF
colliders in assets/aic (Apache-2.0). Boxes are (centre xyz, size xyz) in each model's own frame, metres."""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "aic"
CAGE_PREFIX = "10099100-011lfc001"  # the dual SFP cage's collider boxes in the NIC model


def boxes(sdf: Path):
    out = []
    for col in ET.parse(sdf).getroot().iter("collision"):
        size = col.find(".//box/size")
        if size is None:
            continue  # the NIC's two small standoff cylinders are not needed
        pose = [float(v) for v in col.find("pose").text.split()]
        sx, sy, sz = (float(v) for v in size.text.split())
        if abs(abs(pose[3]) - math.pi / 2) < 0.01:  # rotated a quarter turn about x: y and z swap
            sy, sz = sz, sy
        out.append({"name": col.get("name"), "pos": [round(v, 6) for v in pose[:3]], "size": [round(v, 6) for v in (sx, sy, sz)]})
    return out


def extent(bs):
    lo = [min(b["pos"][i] - b["size"][i] / 2 for b in bs) for i in range(3)]
    hi = [max(b["pos"][i] + b["size"][i] / 2 for b in bs) for i in range(3)]
    return lo, hi


def main():
    module = boxes(ASSETS / "SFP Module" / "model.sdf")
    nic = boxes(ASSETS / "NIC Card" / "model.sdf")
    cage = [b for b in nic if b["name"].startswith(CAGE_PREFIX)]
    c_lo, c_hi = extent(cage)
    m_lo, m_hi = extent(module)
    body = [b for b in module if b["name"].startswith("body_collider")]
    b_lo, b_hi = extent(body)
    # the two cage bays: side walls are the thin tall boxes (x size ~1.1 mm); bays lie between consecutive walls
    walls = sorted(b["pos"][0] for b in cage if b["size"][0] < 0.002 and b["size"][2] > 0.01)
    bays = [{"x_min": walls[0] + 0.00055, "x_max": walls[1] - 0.00055}, {"x_min": walls[2] + 0.00055, "x_max": walls[3] - 0.00055}]
    floor_top = max(b["pos"][2] + b["size"][2] / 2 for b in cage if b["size"][2] < 0.0015 and b["pos"][2] < 0.005)
    roof_bot = min(b["pos"][2] - b["size"][2] / 2 for b in cage if b["size"][2] < 0.0015 and b["pos"][2] > 0.005)
    spec = {
        "units": "m",
        "source": "AIC SFP Module and NIC Card model.sdf colliders (Apache-2.0)",
        "nic": {"boxes": nic, "cage_open_y": round(c_lo[1], 5), "cage_back_y": round(c_hi[1], 5),
                "bays": [{k: round(v, 5) for k, v in bay.items()} for bay in bays],
                "cage_floor_z": round(floor_top, 5), "cage_roof_z": round(roof_bot, 5)},
        "module": {"boxes": module, "body_y": [round(b_lo[1], 5), round(b_hi[1], 5)], "extent_lo": [round(v, 5) for v in m_lo],
                   "extent_hi": [round(v, 5) for v in m_hi],
                   "note": "tip (the end that goes into the cage) is the -y end; +y end carries the latch and cable"},
    }
    (Path(__file__).parent / "geometry.json").write_text(json.dumps(spec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in spec["nic"].items() if k != "boxes"}, indent=1))
    print("module body y", spec["module"]["body_y"], "extent", spec["module"]["extent_lo"], spec["module"]["extent_hi"])


if __name__ == "__main__":
    main()
