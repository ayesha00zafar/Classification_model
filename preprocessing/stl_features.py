"""
stl_features.py  (v2 — clinical features)
------------------------------------------
Extracts clinically meaningful orthodontic measurements from STL pairs.

Key improvements over v1:
  - Overjet: horizontal distance between upper/lower incisor edges
  - Overbite: vertical overlap of upper over lower anteriors
  - Molar AP offset: anterior-posterior shift between upper/lower molars
  - Arch width discrepancy: upper vs lower width at molar and canine zones
  - Curve of Spee proxy: vertical curve of lower occlusal plane
  - Midline deviation: lateral offset of arch centroids
  - Cross-arch vertical gap: zone-by-zone vertical clearance
  - Arch symmetry: left vs right half comparison per arch

Total: 32 features
"""

import struct
import numpy as np


def _read_stl(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        buf = f.read(n * 50)
    triangles = []
    offset = 0
    for _ in range(n):
        offset += 12
        v1 = struct.unpack_from("<3f", buf, offset); offset += 12
        v2 = struct.unpack_from("<3f", buf, offset); offset += 12
        v3 = struct.unpack_from("<3f", buf, offset); offset += 12
        offset += 2
        triangles.append([v1, v2, v3])
    return np.array(triangles, dtype=np.float32)


def _verts(tri):
    return tri.reshape(-1, 3)


def _zone_verts(verts, x_min, x_max):
    """Return vertices in an X-zone (left/right/anterior regions)."""
    mask = (verts[:, 0] >= x_min) & (verts[:, 0] <= x_max)
    return verts[mask]


def _top_surface(verts, percentile=90):
    """Return vertices near the top (occlusal) surface of an arch."""
    z_thresh = np.percentile(verts[:, 2], percentile)
    return verts[verts[:, 2] >= z_thresh]


def _bottom_surface(verts, percentile=10):
    """Return vertices near the bottom surface of an arch."""
    z_thresh = np.percentile(verts[:, 2], percentile)
    return verts[verts[:, 2] <= z_thresh]


def extract_features(upper_path: str, lower_path: str) -> np.ndarray:
    """
    Extract 32 clinically-grounded features from upper+lower STL pair.

    Feature groups:
      [0-3]   Overjet & overbite proxies (AP and vertical relationships)
      [4-9]   Molar zone AP offsets (left/right separately — key for Class I/II/III)
      [10-15] Arch width discrepancies at 3 zones (crossbite indicator)
      [16-19] Midline and arch symmetry
      [20-25] Vertical gap per zone (anterior, premolar, molar — left & right)
      [26-31] Arch shape: curve of Spee, arch depth ratio, occlusal plane tilt
    """
    u_tri = _read_stl(upper_path)
    l_tri = _read_stl(lower_path)
    u_v   = _verts(u_tri)
    l_v   = _verts(l_tri)

    u_cx  = u_v[:, 0].mean()
    l_cx  = l_v[:, 0].mean()
    u_cy  = u_v[:, 1].mean()
    l_cy  = l_v[:, 1].mean()

    u_xr  = u_v[:, 0].max() - u_v[:, 0].min()
    l_xr  = l_v[:, 0].max() - l_v[:, 0].min()

    # Zone boundaries (relative to arch centre)
    ant_hw   = u_xr * 0.20   # anterior half-width
    pre_hw   = u_xr * 0.35   # premolar half-width
    mol_hw   = u_xr * 0.48   # molar half-width

    def zone_top_z(verts, x_min, x_max):
        z = _zone_verts(verts, x_min, x_max)
        return float(z[:, 2].max()) if len(z) > 10 else 0.0

    def zone_centroid_y(verts, x_min, x_max):
        z = _zone_verts(verts, x_min, x_max)
        return float(z[:, 1].mean()) if len(z) > 10 else 0.0

    def zone_centroid_x(verts, x_min, x_max):
        z = _zone_verts(verts, x_min, x_max)
        return float(z[:, 0].mean()) if len(z) > 10 else 0.0

    # ── [0-3] Overjet & overbite ──────────────────────────────────
    # Overjet: AP (Y-axis) gap at anterior zone
    u_ant_y = zone_centroid_y(u_v, u_cx - ant_hw, u_cx + ant_hw)
    l_ant_y = zone_centroid_y(l_v, l_cx - ant_hw, l_cx + ant_hw)
    overjet = u_ant_y - l_ant_y                        # +ve = upper forward

    # Overbite: vertical overlap at anterior zone
    u_ant_z = zone_top_z(u_v, u_cx - ant_hw, u_cx + ant_hw)
    l_ant_z = zone_top_z(l_v, l_cx - ant_hw, l_cx + ant_hw)
    overbite = u_ant_z - l_ant_z                       # +ve = upper higher (deep bite)

    # Global centroid offsets
    ap_offset  = float(u_cy - l_cy)                    # AP skeletal shift
    vert_offset= float(u_v[:, 2].mean() - l_v[:, 2].mean())

    # ── [4-9] Molar AP offsets (left/right separately) ────────────
    # RIGHT molar zone (positive X side)
    u_rmol_y = zone_centroid_y(u_v, u_cx + pre_hw, u_cx + mol_hw)
    l_rmol_y = zone_centroid_y(l_v, l_cx + pre_hw, l_cx + mol_hw)
    right_molar_ap = u_rmol_y - l_rmol_y               # Class I ≈ 0, Class II > 0, Class III < 0

    u_rmol_z = zone_top_z(u_v, u_cx + pre_hw, u_cx + mol_hw)
    l_rmol_z = zone_top_z(l_v, l_cx + pre_hw, l_cx + mol_hw)
    right_molar_vert = u_rmol_z - l_rmol_z

    # LEFT molar zone (negative X side)
    u_lmol_y = zone_centroid_y(u_v, u_cx - mol_hw, u_cx - pre_hw)
    l_lmol_y = zone_centroid_y(l_v, l_cx - mol_hw, l_cx - pre_hw)
    left_molar_ap = u_lmol_y - l_lmol_y

    u_lmol_z = zone_top_z(u_v, u_cx - mol_hw, u_cx - pre_hw)
    l_lmol_z = zone_top_z(l_v, l_cx - mol_hw, l_cx - pre_hw)
    left_molar_vert = u_lmol_z - l_lmol_z

    # AP asymmetry between left and right molars
    molar_ap_asym = right_molar_ap - left_molar_ap

    # Molar vertical asymmetry
    molar_vert_asym = right_molar_vert - left_molar_vert

    # ── [10-15] Arch width discrepancy ────────────────────────────
    def arch_width(verts, cx, hw):
        z = _zone_verts(verts, cx - hw, cx + hw)
        return float(z[:, 0].max() - z[:, 0].min()) if len(z) > 10 else 0.0

    u_ant_w  = arch_width(u_v, u_cx, ant_hw)
    l_ant_w  = arch_width(l_v, l_cx, ant_hw)
    u_pre_w  = arch_width(u_v, u_cx, pre_hw)
    l_pre_w  = arch_width(l_v, l_cx, pre_hw)
    u_mol_w  = arch_width(u_v, u_cx, mol_hw)
    l_mol_w  = arch_width(l_v, l_cx, mol_hw)

    ant_w_disc = u_ant_w - l_ant_w    # crossbite → lower wider than upper
    pre_w_disc = u_pre_w - l_pre_w
    mol_w_disc = u_mol_w - l_mol_w    # most important for transversal class

    # Width ratios
    ant_w_ratio = l_ant_w / (u_ant_w + 1e-6)
    pre_w_ratio = l_pre_w / (u_pre_w + 1e-6)
    mol_w_ratio = l_mol_w / (u_mol_w + 1e-6)

    # ── [16-19] Midline & symmetry ────────────────────────────────
    midline_dev = float(u_cx - l_cx)                   # lateral deviation

    def arch_symmetry(verts, cx):
        left_v  = verts[verts[:, 0] < cx]
        right_v = verts[verts[:, 0] > cx]
        if len(left_v) < 10 or len(right_v) < 10:
            return 0.0
        return float(abs(left_v[:, 1].mean() - right_v[:, 1].mean()))

    u_sym = arch_symmetry(u_v, u_cx)
    l_sym = arch_symmetry(l_v, l_cx)
    sym_diff = u_sym - l_sym

    # ── [20-25] Vertical gap per zone ─────────────────────────────
    def vert_gap(u_verts, l_verts, cx, x_min, x_max):
        u_z = _zone_verts(u_verts, x_min, x_max)
        l_z = _zone_verts(l_verts, x_min, x_max)
        if len(u_z) < 5 or len(l_z) < 5:
            return 0.0
        return float(u_z[:, 2].min() - l_z[:, 2].max())   # occlusal clearance

    gap_ant   = vert_gap(u_v, l_v, u_cx, u_cx-ant_hw,          u_cx+ant_hw)
    gap_pre_r = vert_gap(u_v, l_v, u_cx, u_cx+ant_hw,          u_cx+pre_hw)
    gap_pre_l = vert_gap(u_v, l_v, u_cx, u_cx-pre_hw,          u_cx-ant_hw)
    gap_mol_r = vert_gap(u_v, l_v, u_cx, u_cx+pre_hw,          u_cx+mol_hw)
    gap_mol_l = vert_gap(u_v, l_v, u_cx, u_cx-mol_hw,          u_cx-pre_hw)
    gap_asym  = gap_mol_r - gap_mol_l

    # ── [26-31] Arch shape features ───────────────────────────────
    # Curve of Spee proxy: Z variation along Y axis in lower arch
    l_top = _top_surface(l_v, percentile=85)
    if len(l_top) > 20:
        y_bins = np.linspace(l_top[:, 1].min(), l_top[:, 1].max(), 6)
        z_means = []
        for j in range(len(y_bins)-1):
            m = (l_top[:, 1] >= y_bins[j]) & (l_top[:, 1] < y_bins[j+1])
            if m.sum() > 3:
                z_means.append(l_top[m, 2].mean())
        curve_of_spee = float(np.std(z_means)) if len(z_means) > 2 else 0.0
    else:
        curve_of_spee = 0.0

    # Arch depth ratio
    u_depth = float(u_v[:, 1].max() - u_v[:, 1].min())
    l_depth = float(l_v[:, 1].max() - l_v[:, 1].min())
    depth_ratio = l_depth / (u_depth + 1e-6)

    # Occlusal plane tilt (Z difference left vs right molar top)
    occ_tilt = float(u_rmol_z - u_lmol_z)
    lower_occ_tilt = float(l_rmol_z - l_lmol_z)

    # Arch length (Y extent)
    u_len = float(u_v[:, 1].max() - u_v[:, 1].min())
    l_len = float(l_v[:, 1].max() - l_v[:, 1].min())
    arch_len_diff = u_len - l_len

    feats = np.array([
        # [0-3] Overjet / overbite
        overjet, overbite, ap_offset, vert_offset,
        # [4-9] Molar AP offsets
        right_molar_ap, right_molar_vert,
        left_molar_ap,  left_molar_vert,
        molar_ap_asym,  molar_vert_asym,
        # [10-15] Width discrepancy
        ant_w_disc, pre_w_disc, mol_w_disc,
        ant_w_ratio, pre_w_ratio, mol_w_ratio,
        # [16-19] Midline & symmetry
        midline_dev, u_sym, l_sym, sym_diff,
        # [20-25] Vertical gaps
        gap_ant, gap_pre_r, gap_pre_l, gap_mol_r, gap_mol_l, gap_asym,
        # [26-31] Arch shape
        curve_of_spee, depth_ratio, occ_tilt, lower_occ_tilt,
        arch_len_diff, float(u_xr - l_xr),
    ], dtype=np.float32)

    return feats   # shape: (32,)