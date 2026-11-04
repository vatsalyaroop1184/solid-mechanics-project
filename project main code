"""
TrussLab 3D - simulator for 3D pin-jointed truss bridges.

Run with:   streamlit run trusslab_3d.py

What it models
    * Sticks of solid circular cross-section, any material (editable material library).
    * Pin joints with 2 to N sticks each (N is a setting, default 8).
    * A thin road sheet that carries load only (weight, traffic, wind area) - it adds no stiffness.
    * Loads: self weight, road weight, traffic, wind (3D wind speed and/or 3D force on each stick),
      and point loads on joints.
    * Material properties are constants taken at the chosen temperature (no thermal stresses).

Coordinate system (right-handed):  X along the span, Y up, Z across the road (toward the viewer).

Method (APL104 topics only, no stiffness matrix / FEM)
    Equilibrium of every pin joint (method of joints)   -> member forces      (sum F = 0 in 3D)
    Axial stress and strain, Hooke's law                 -> sigma = F/A, eps = sigma/E
    Unit-load (virtual work) method, i.e. energy method  -> joint displacements
    Euler buckling of compression sticks                 -> P_cr = pi^2 E I / (K L)^2
    The method of joints needs a statically determinate truss: m + r = 3 j and a stable layout.
    Outside APL104: the wind drag formula (fluid drag) and the road-load sharing between joints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:  # shown as a friendly message in main()
    go = None

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

try:
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _v = (0, 0)
# Streamlit 1.50+ replaces use_container_width=True by width="stretch"
STRETCH = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

G_ACC = 9.81          # m/s^2
RHO_AIR = 1.225       # kg/m^3 (air at about 15 degC)

MAT_COLS = ["Material", "E (GPa)", "Density (kg/m3)", "Strength (MPa)"]
# Approximate textbook values near room temperature (about 20 degC). Edit them in the Materials table.
DEFAULT_MATERIALS = [
    ("Structural steel", 200.0, 7850.0, 250.0),
    ("High-strength steel", 205.0, 7850.0, 690.0),
    ("Stainless steel 304", 193.0, 8000.0, 215.0),
    ("Aluminium 6061-T6", 69.0, 2700.0, 276.0),
    ("Titanium Ti-6Al-4V", 114.0, 4430.0, 880.0),
    ("Cast iron (grey)", 100.0, 7200.0, 150.0),
    ("Copper", 117.0, 8960.0, 70.0),
    ("Brass", 100.0, 8500.0, 200.0),
    ("Timber (Douglas fir)", 12.0, 530.0, 40.0),
    ("Concrete C30", 30.0, 2400.0, 30.0),
]

J_COLS = ["Joint", "x (m)", "y (m)", "z (m)"]
S_COLS = ["Stick", "Joint A", "Joint B", "Diameter (mm)", "Material"]
SUP_COLS = ["Joint", "Fix X", "Fix Y", "Fix Z"]
DECK_COLS = ["Panel", "Joint 1", "Joint 2", "Joint 3", "Joint 4", "Thickness (mm)", "Material"]
JL_COLS = ["Joint", "Fx (kN)", "Fy (kN)", "Fz (kN)"]
SW_COLS = ["Stick", "Fx (N)", "Fy (N)", "Fz (N)"]

STANDARD_MODELS = ["Pratt", "Howe", "Warren (with verticals)", "Blank"]

C_RED = "#e5484d"
C_BLUE = "#2f7fd1"
C_GREEN = "#2a9d8f"
C_ORANGE = "#f08c00"
C_PURPLE = "#8e5bd6"
C_GREY = "#7f8c8d"


# ----------------------------------------------------------------------------
# Model data
# ----------------------------------------------------------------------------

@dataclass
class Truss:
    jid: np.ndarray                 # joint numbers (as typed by the user)
    xyz: np.ndarray                 # (j, 3) joint coordinates [m]
    sid: np.ndarray                 # stick numbers
    ia: np.ndarray                  # joint index of end A
    ib: np.ndarray                  # joint index of end B
    d: np.ndarray                   # diameters [m]
    mat: list                       # material name per stick
    E: np.ndarray                   # Young's modulus [Pa]
    rho: np.ndarray                 # density [kg/m3]
    sy: np.ndarray                  # strength [Pa]
    L: np.ndarray = field(default=None)    # lengths [m]
    e: np.ndarray = field(default=None)    # unit vectors A->B
    A: np.ndarray = field(default=None)    # areas [m2]
    sup: list = field(default_factory=list)      # (joint index, direction 0/1/2)
    deck: list = field(default_factory=list)     # dicts: idx, t, rho, name
    jload: np.ndarray = field(default=None)      # (j, 3) point loads [N]
    swind: np.ndarray = field(default=None)      # (m, 3) direct wind force per stick [N]

    def __post_init__(self):
        v = self.xyz[self.ib] - self.xyz[self.ia]
        self.L = np.linalg.norm(v, axis=1)
        self.e = v / self.L[:, None]
        self.A = np.pi * self.d**2 / 4.0


# ----------------------------------------------------------------------------
# Reading and checking the input tables
# ----------------------------------------------------------------------------

def _num(x):
    return pd.to_numeric(x, errors="coerce")


def _ints(series):
    v = _num(series).to_numpy(dtype=float)
    return v


def parse_tables(tb: dict, mats: pd.DataFrame, max_valence: int):
    """Turn the editable tables into a Truss. Returns (truss or None, errors, warnings)."""
    errors, warns = [], []

    # ---- materials
    lib = {}
    for n, (_, r) in enumerate(mats.dropna(how="all").iterrows(), start=1):
        name = str(r.get("Material", "")).strip()
        E, rho, sy = _num(r.get("E (GPa)")), _num(r.get("Density (kg/m3)")), _num(r.get("Strength (MPa)"))
        if not name or name.lower() == "nan":
            errors.append(f"Materials, row {n}: the name is missing.")
        elif pd.isna(E) or pd.isna(rho) or pd.isna(sy) or E <= 0 or rho <= 0 or sy <= 0:
            errors.append(f"Materials, '{name}': E, density and strength must all be positive numbers.")
        else:
            lib[name] = (E * 1e9, rho, sy * 1e6)

    # ---- joints
    jd = tb["joints"].dropna(how="all")
    if jd.empty:
        errors.append("Joints: the table is empty. Load a standard model or add joints.")
        return None, errors, warns
    ids = _ints(jd["Joint"])
    xyz = jd[["x (m)", "y (m)", "z (m)"]].apply(_num).to_numpy(dtype=float)
    bad = np.isnan(ids) | np.isnan(xyz).any(axis=1) | (np.abs(ids - np.round(ids)) > 1e-9)
    for k in np.where(bad)[0]:
        errors.append(f"Joints, row {k + 1}: the joint number must be a whole number and x, y, z must be filled in.")
    if bad.any():
        return None, errors, warns
    ids = np.round(ids).astype(int)
    dup = sorted({int(i) for i in ids if (ids == i).sum() > 1})
    if dup:
        errors.append(f"Joints: number(s) {dup} are used more than once.")
        return None, errors, warns
    pos = {int(i): k for k, i in enumerate(ids)}

    # ---- sticks
    sd = tb["sticks"].dropna(how="all")
    if sd.empty:
        errors.append("Sticks: the table is empty.")
        return None, errors, warns
    sid = _ints(sd["Stick"])
    ja, jb = _ints(sd["Joint A"]), _ints(sd["Joint B"])
    dia = _num(sd["Diameter (mm)"]).to_numpy(dtype=float)
    mname = [str(v).strip() for v in sd["Material"]]
    keep, S_ia, S_ib, S_d, S_E, S_rho, S_sy, S_mat, S_id = [], [], [], [], [], [], [], [], []
    seen_pairs, seen_ids = {}, set()
    for k in range(len(sd)):
        row = k + 1
        if np.isnan(sid[k]) or np.isnan(ja[k]) or np.isnan(jb[k]) or np.isnan(dia[k]):
            errors.append(f"Sticks, row {row}: stick number, both joints and the diameter must be filled in.")
            continue
        s, a, b = int(round(sid[k])), int(round(ja[k])), int(round(jb[k]))
        if s in seen_ids:
            errors.append(f"Sticks: number {s} is used more than once.")
            continue
        seen_ids.add(s)
        if a not in pos or b not in pos:
            errors.append(f"Stick {s}: joint {a if a not in pos else b} does not exist in the Joints table.")
            continue
        if a == b:
            errors.append(f"Stick {s}: both ends are joint {a}.")
            continue
        if dia[k] <= 0:
            errors.append(f"Stick {s}: the diameter must be positive.")
            continue
        if mname[k] not in lib:
            errors.append(f"Stick {s}: material '{mname[k]}' is not in the Materials table.")
            continue
        pair = tuple(sorted((a, b)))
        if pair in seen_pairs:
            errors.append(f"Sticks {seen_pairs[pair]} and {s} connect the same two joints ({pair[0]}, {pair[1]}).")
            continue
        seen_pairs[pair] = s
        if np.linalg.norm(xyz[pos[a]] - xyz[pos[b]]) < 1e-6:
            errors.append(f"Stick {s}: joints {a} and {b} are at the same place, so the stick has no length.")
            continue
        E, rho, sy = lib[mname[k]]
        S_id.append(s); S_ia.append(pos[a]); S_ib.append(pos[b]); S_d.append(dia[k] / 1000.0)
        S_E.append(E); S_rho.append(rho); S_sy.append(sy); S_mat.append(mname[k])
    if errors:
        return None, errors, warns

    j, m = len(ids), len(S_id)
    ia, ib = np.array(S_ia), np.array(S_ib)

    # ---- pivots: sticks per joint
    val = np.bincount(np.concatenate([ia, ib]), minlength=j)
    for k in range(j):
        if val[k] == 0:
            errors.append(f"Joint {ids[k]} is not connected to any stick. Delete it or connect it.")
        elif val[k] == 1:
            errors.append(f"Joint {ids[k]} has only 1 stick. A pivot needs 2 or more sticks (a single stick "
                          f"would be a loose end that carries no force).")
        elif val[k] > max_valence:
            errors.append(f"Joint {ids[k]} has {val[k]} sticks, more than the limit of {max_valence} per pivot. "
                          f"Reduce the sticks there or raise 'Max sticks per pivot' in the sidebar.")
    if errors:
        return None, errors, warns

    tr = Truss(jid=ids, xyz=xyz, sid=np.array(S_id), ia=ia, ib=ib, d=np.array(S_d), mat=S_mat,
               E=np.array(S_E), rho=np.array(S_rho), sy=np.array(S_sy))

    # ---- supports
    sup, sup_seen = [], set()
    for n, (_, r) in enumerate(tb["supports"].dropna(how="all").iterrows(), start=1):
        jn = _num(r["Joint"])
        if pd.isna(jn) or int(round(jn)) not in pos:
            errors.append(f"Supports, row {n}: joint {'?' if pd.isna(jn) else int(round(jn))} does not exist.")
            continue
        jn = int(round(jn))
        if jn in sup_seen:
            errors.append(f"Supports: joint {jn} appears more than once. Use one row per joint.")
            continue
        sup_seen.add(jn)
        for dname, col in enumerate(("Fix X", "Fix Y", "Fix Z")):
            if bool(r[col]) and not pd.isna(r[col]):
                sup.append((pos[jn], dname))
    tr.sup = sup
    if not sup:
        errors.append("Supports: no support is defined. Fix at least X, Y and Z at one joint.")

    # ---- road panels
    for n, (_, r) in enumerate(tb["deck"].dropna(how="all").iterrows(), start=1):
        jn = [_num(r[c]) for c in ("Joint 1", "Joint 2", "Joint 3", "Joint 4")]
        jn = [int(round(v)) for v in jn if not pd.isna(v)]
        t = _num(r["Thickness (mm)"])
        name = str(r["Material"]).strip()
        label = r["Panel"] if not pd.isna(r["Panel"]) else n
        if len(jn) < 3 or len(set(jn)) != len(jn):
            errors.append(f"Road panel {label}: give 3 or 4 different joints.")
            continue
        if any(v not in pos for v in jn):
            errors.append(f"Road panel {label}: a joint number does not exist.")
            continue
        if pd.isna(t) or t <= 0 or name not in lib:
            errors.append(f"Road panel {label}: thickness must be positive and the material must be in the table.")
            continue
        idx = [pos[v] for v in jn]
        pts = xyz[idx]
        nvec = np.cross(pts[2] - pts[0], pts[1] - pts[0]) if len(idx) == 3 else np.cross(pts[2] - pts[0], pts[3] - pts[1])
        if np.linalg.norm(nvec) < 1e-9:
            errors.append(f"Road panel {label}: the joints lie on a line, so the panel has no area.")
            continue
        if len(idx) == 4:
            nn = nvec / np.linalg.norm(nvec)
            off = abs(np.dot(pts[3] - pts[0], nn))
            if off > 0.02 * np.ptp(pts, axis=0).max():
                warns.append(f"Road panel {label}: the four joints are not in one plane (off by {off * 1000:.0f} mm).")
        tr.deck.append(dict(idx=idx, t=t / 1000.0, rho=lib[name][1], name=name, label=label))

    # ---- joint loads and direct stick wind forces
    tr.jload = np.zeros((j, 3))
    for n, (_, r) in enumerate(tb["jloads"].dropna(how="all").iterrows(), start=1):
        jn = _num(r["Joint"])
        if pd.isna(jn) or int(round(jn)) not in pos:
            errors.append(f"Joint loads, row {n}: joint {'?' if pd.isna(jn) else int(round(jn))} does not exist.")
            continue
        f = np.array([0.0 if pd.isna(_num(r[c])) else float(_num(r[c])) for c in ("Fx (kN)", "Fy (kN)", "Fz (kN)")])
        tr.jload[pos[int(round(jn))]] += f * 1000.0
    spos = {int(s): k for k, s in enumerate(tr.sid)}
    tr.swind = np.zeros((m, 3))
    for n, (_, r) in enumerate(tb["swind"].dropna(how="all").iterrows(), start=1):
        sn = _num(r["Stick"])
        if pd.isna(sn) or int(round(sn)) not in spos:
            errors.append(f"Stick wind forces, row {n}: stick {'?' if pd.isna(sn) else int(round(sn))} does not exist.")
            continue
        f = np.array([0.0 if pd.isna(_num(r[c])) else float(_num(r[c])) for c in ("Fx (N)", "Fy (N)", "Fz (N)")])
        tr.swind[spos[int(round(sn))]] += f
    if errors:
        return None, errors, warns
    return tr, errors, warns


# ----------------------------------------------------------------------------
# Loads
# ----------------------------------------------------------------------------

def deck_geometry(tr: Truss, panel):
    """Area, projected horizontal area and unit normal of a road panel."""
    pts = tr.xyz[panel["idx"]]
    nvec = np.cross(pts[1] - pts[0], pts[2] - pts[0]) if len(pts) == 3 else np.cross(pts[1] - pts[3], pts[2] - pts[0]) * -1.0
    if len(pts) == 4:
        nvec = np.cross(pts[2] - pts[0], pts[3] - pts[1])
    area = 0.5 * np.linalg.norm(nvec)
    return area, 0.5 * abs(nvec[1]), nvec / np.linalg.norm(nvec)


def compute_loads(tr: Truss, env: dict):
    """Return the joint load vector P (3j) [N] and the individual contributions."""
    j, m = len(tr.xyz), len(tr.ia)
    comp = {k: np.zeros((j, 3)) for k in ("Stick weight", "Road weight", "Traffic", "Wind on sticks (speed)",
                                          "Wind on road (speed)", "Wind on sticks (direct)", "Joint loads")}
    g = env["g"]
    if env["use_self"]:
        w = tr.rho * tr.A * tr.L * g
        np.add.at(comp["Stick weight"][:, 1], tr.ia, -w / 2.0)
        np.add.at(comp["Stick weight"][:, 1], tr.ib, -w / 2.0)
    for p in tr.deck:
        area, area_h, nrm = deck_geometry(tr, p)
        k = len(p["idx"])
        if env["use_road"]:
            comp["Road weight"][p["idx"], 1] += -p["rho"] * p["t"] * area * g / k
        if env["use_traffic"]:
            comp["Traffic"][p["idx"], 1] += -env["traffic_kpa"] * 1000.0 * area_h / k
        if env["use_wind"]:
            vn = float(np.dot(env["wind"], nrm))
            f = 0.5 * RHO_AIR * env["cd_deck"] * area * abs(vn) * vn * nrm
            comp["Wind on road (speed)"][p["idx"]] += f / k
    if env["use_wind"] and np.linalg.norm(env["wind"]) > 0:
        v = env["wind"]
        vn = v[None, :] - (tr.e @ v)[:, None] * tr.e            # wind component normal to each stick
        sp = np.linalg.norm(vn, axis=1)
        f = 0.5 * RHO_AIR * env["cd_stick"] * (tr.d * tr.L * sp)[:, None] * vn
        np.add.at(comp["Wind on sticks (speed)"], tr.ia, f / 2.0)
        np.add.at(comp["Wind on sticks (speed)"], tr.ib, f / 2.0)
    if env["use_direct"]:
        np.add.at(comp["Wind on sticks (direct)"], tr.ia, tr.swind / 2.0)
        np.add.at(comp["Wind on sticks (direct)"], tr.ib, tr.swind / 2.0)
    if env["use_joint"]:
        comp["Joint loads"] += tr.jload
    P = sum(comp.values())
    return P.reshape(-1), comp


# ----------------------------------------------------------------------------
# Solver: method of joints + unit-load method
# ----------------------------------------------------------------------------

def equilibrium_matrix(tr: Truss):
    """
    Joint equilibrium of the whole truss:  B x = -P,  x = [member forces F (tension +); support reactions R].
    Column of stick i:  +e_i at joint A (tension pulls A toward B),  -e_i at joint B.
    Column of a support component: +1 at that joint and direction.
    """
    j, m, r = len(tr.xyz), len(tr.ia), len(tr.sup)
    B = np.zeros((3 * j, m + r))
    for k in range(m):
        a, b = tr.ia[k], tr.ib[k]
        B[3 * a:3 * a + 3, k] += tr.e[k]
        B[3 * b:3 * b + 3, k] -= tr.e[k]
    for q, (jj, dd) in enumerate(tr.sup):
        B[3 * jj + dd, m + q] = 1.0
    return B


def diagnose(tr: Truss, B: np.ndarray):
    """Classify the truss for the method of joints (needs m + r = 3j and a non-singular B)."""
    j, m, r = len(tr.xyz), len(tr.ia), len(tr.sup)
    n_eq, n_unk = 3 * j, m + r
    U, S, Vt = np.linalg.svd(B)
    tol = 1e-9 * S.max()
    rank = int((S > tol).sum())
    info = dict(j=j, m=m, r=r, n_eq=n_eq, n_unk=n_unk, rank=rank, ok=False, message="",
                mech_joints=[], redundant_sticks=[], weak=False)
    if n_unk == n_eq and rank == n_eq:
        info["ok"] = True
        info["weak"] = bool(S.min() < 1e-5 * S.max())
        return info

    n_mech = n_eq - rank           # displacement modes that no stick or support resists
    n_red = n_unk - rank           # self-stress states (extra sticks or supports)
    parts = [f"Method of joints needs m + r = 3j exactly. Here m = {m} sticks, r = {r} support components, "
             f"j = {j} joints, so m + r = {n_unk} and 3j = {n_eq}."]
    if n_mech > 0:
        parts.append(f"The truss is unstable: it has {n_mech} way(s) to move without any stick resisting "
                     f"(a mechanism). Add sticks or supports where the marked joints can move.")
        modes = U[:, rank:]
        mag = np.sqrt((modes**2).reshape(j, 3, -1).sum(axis=(1, 2)))
        info["mech_joints"] = [int(tr.jid[k]) for k in np.argsort(-mag) if mag[k] > 0.1 * mag.max()][:12]
    if n_red > 0:
        parts.append(f"The truss is statically indeterminate: it has {n_red} more stick(s) or support component(s) "
                     f"than needed (redundant). Remove {n_red} stick(s) or support component(s) among the marked "
                     f"sticks, or the supports, so that m + r = 3j.")
        modes = Vt[rank:, :m]
        mag = np.sqrt((modes**2).sum(axis=0))
        if mag.max() > 1e-9:
            info["redundant_sticks"] = [int(tr.sid[k]) for k in np.argsort(-mag) if mag[k] > 0.1 * mag.max()][:15]
        sup_mag = np.sqrt((Vt[rank:, m:] ** 2).sum(axis=0)) if r else np.array([])
        if r and sup_mag.max() > 0.1 * max(mag.max(), sup_mag.max()):
            parts.append("Some support components are among the redundant ones.")
    info["message"] = " ".join(parts)
    return info


def solve_truss(tr: Truss, B: np.ndarray, P: np.ndarray, env: dict):
    """Solve a determinate truss. Returns a dict of results."""
    m, j = len(tr.ia), len(tr.xyz)
    x = np.linalg.solve(B, -P)
    F, R = x[:m], x[m:]
    resid = float(np.max(np.abs(B @ x + P)))

    # stress and strain (Hooke's law)
    sigma = F / tr.A
    strain = sigma / tr.E
    delta = F * tr.L / (tr.E * tr.A)                  # elongation of each stick

    # joint displacements by the unit-load (virtual work) method: B^T u = -[delta; 0]
    z = np.concatenate([delta, np.zeros(len(tr.sup))])
    u = np.linalg.solve(B.T, -z).reshape(j, 3)
    disp = np.linalg.norm(u, axis=1)

    # strength and Euler buckling
    K = env["K"]
    I = np.pi * tr.d**4 / 64.0
    Pcr = np.pi**2 * tr.E * I / (K * tr.L) ** 2
    cap_y = tr.sy * tr.A
    comp_ = F < 0
    cap = np.where(comp_, np.minimum(Pcr, cap_y), cap_y)
    mode = np.where(comp_ & (Pcr < cap_y), "buckling", "yield")
    absF = np.abs(F)
    with np.errstate(divide="ignore", invalid="ignore"):
        fs = np.where(absF > 1e-6, cap / absF, np.inf)
    util = np.where(np.isfinite(fs), 1.0 / fs, 0.0)
    slender = K * tr.L / (tr.d / 4.0)

    jm = int(np.argmax(disp))
    at_joint = np.where((tr.ia == jm) | (tr.ib == jm))[0]
    return dict(F=F, R=R, sigma=sigma, strain=strain, delta=delta, u=u, disp=disp, Pcr=Pcr, cap=cap,
                mode=mode, fs=fs, util=util, slender=slender, resid=resid, jmax=jm, at_joint=at_joint,
                imax_sigma=int(np.argmax(np.abs(sigma))), imax_strain=int(np.argmax(np.abs(strain))),
                imin_fs=int(np.argmin(fs)))


# ----------------------------------------------------------------------------
# Standard bridges (determinate 3D box trusses with a road on the bottom chords)
# ----------------------------------------------------------------------------

def standard_bridge(kind, n, p, H, W, dc, dv, dd, dx, mat, road_t, road_mat):
    """
    Two vertical side trusses joined by cross members and plane bracing, with a road on the bottom chords.
    One sway brace and four vertical bearings make it exactly determinate:  m + r = 3j.
    Joint order per station i: FB (front bottom), BB (back bottom), FT (front top), BT (back top).
    """
    tags = ("FB", "BB", "FT", "BT")
    yz = {"FB": (0.0, 0.0), "BB": (0.0, W), "FT": (H, 0.0), "BT": (H, W)}
    jn, rows = {}, []
    for i in range(n + 1):
        for t in tags:
            jn[(t, i)] = len(jn) + 1
            rows.append([jn[(t, i)], i * p, yz[t][0], yz[t][1]])
    joints = pd.DataFrame(rows, columns=J_COLS)
    st_rows = []

    def add(a, b, d):
        st_rows.append([len(st_rows) + 1, jn[a], jn[b], d, mat])

    for i in range(n):                                   # chords
        for t in tags:
            add((t, i), (t, i + 1), dc)
    for i in range(n + 1):                               # verticals and cross members
        add(("FB", i), ("FT", i), dv)
        add(("BB", i), ("BT", i), dv)
        add(("FB", i), ("BB", i), dx)
        add(("FT", i), ("BT", i), dx)
    for i in range(n):                                   # side diagonals
        left = i < n / 2.0
        for s in ("F", "B"):
            if kind == "Pratt":
                a, b = ((s + "T", i), (s + "B", i + 1)) if left else ((s + "T", i + 1), (s + "B", i))
            elif kind == "Howe":
                a, b = ((s + "B", i), (s + "T", i + 1)) if left else ((s + "B", i + 1), (s + "T", i))
            else:
                a, b = ((s + "T", i), (s + "B", i + 1)) if i % 2 == 0 else ((s + "B", i), (s + "T", i + 1))
            add(a, b, dd)
        add(("FB", i), ("BB", i + 1), dx)                # road-level bracing
        add(("FT", i), ("BT", i + 1), dx)                # top bracing
    add(("FB", 0), ("BT", 0), dx)                        # sway brace at the first end frame
    sticks = pd.DataFrame(st_rows, columns=S_COLS)

    f0, b0, fn, bn = jn[("FB", 0)], jn[("BB", 0)], jn[("FB", n)], jn[("BB", n)]
    supports = pd.DataFrame([[f0, True, True, True], [b0, False, True, False],
                             [fn, False, True, True], [bn, False, True, False]], columns=SUP_COLS)
    deck = pd.DataFrame([[i + 1, jn[("FB", i)], jn[("FB", i + 1)], jn[("BB", i + 1)], jn[("BB", i)], road_t, road_mat]
                         for i in range(n)], columns=DECK_COLS)
    return dict(joints=joints, sticks=sticks, supports=supports, deck=deck,
                jloads=pd.DataFrame(columns=JL_COLS), swind=pd.DataFrame(columns=SW_COLS))


def blank_tables():
    return dict(joints=pd.DataFrame(columns=J_COLS), sticks=pd.DataFrame(columns=S_COLS),
                supports=pd.DataFrame(columns=SUP_COLS), deck=pd.DataFrame(columns=DECK_COLS),
                jloads=pd.DataFrame(columns=JL_COLS), swind=pd.DataFrame(columns=SW_COLS))


# ----------------------------------------------------------------------------
# 3D figure
# ----------------------------------------------------------------------------

def P3(a):
    """Physical (X, Y, Z) -> plot coordinates (right-handed scene with Y up)."""
    a = np.asarray(a, dtype=float)
    return np.stack([a[..., 0], -a[..., 2], a[..., 1]], axis=-1)


def seg_xyz(pa, pb):
    n = len(pa)
    out = np.full((3 * n, 3), np.nan)
    out[0::3], out[1::3] = pa, pb
    return out[:, 0], out[:, 1], out[:, 2]


def seg_values(v):
    out = np.full(3 * len(v), np.nan)
    out[0::3] = v
    out[1::3] = v
    return out


class Arrows:
    """Arrows drawn with lines (shaft + four head strokes) in physical coordinates."""

    def __init__(self, color, name):
        self.color, self.name, self.pts, self.labels = color, name, [], []

    def add(self, tail, tip, label=None, head=0.0):
        tail, tip = np.asarray(tail, float), np.asarray(tip, float)
        self.pts.append((tail, tip))
        d = tip - tail
        n = np.linalg.norm(d)
        if n == 0:
            return
        d = d / n
        ref = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        p1 = np.cross(d, ref)
        p1 /= np.linalg.norm(p1)
        p2 = np.cross(d, p1)
        for q in (p1, -p1, p2, -p2):
            self.pts.append((tip, tip - d * head + q * 0.4 * head))
        if label:
            self.labels.append((tail, label))

    def trace(self):
        if not self.pts:
            return []
        a = np.array([p[0] for p in self.pts])
        b = np.array([p[1] for p in self.pts])
        x, y, z = seg_xyz(P3(a), P3(b))
        out = [go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=self.color, width=6),
                            name=self.name, hoverinfo="skip")]
        if self.labels:
            lp = P3(np.array([q for q, _ in self.labels]))
            out.append(go.Scatter3d(x=lp[:, 0], y=lp[:, 1], z=lp[:, 2], mode="text", text=[t for _, t in self.labels],
                                    textfont=dict(color=self.color, size=12), showlegend=False, hoverinfo="skip"))
        return out


def build_figure(tr: Truss, res, diag, env, color_by, exag_pct, true_scale, show_ghost, show_deck,
                 show_supports, show_loads, show_numbers, comp=None):
    xyz = tr.xyz
    span = float(np.ptp(xyz, axis=0).max()) or 1.0
    j, m = len(xyz), len(tr.ia)
    fig = go.Figure()

    if res is not None:
        u = res["u"]
        umax = float(res["disp"].max())
        if true_scale:
            sc = 1.0
        else:
            sc = (exag_pct / 100.0) * 0.12 * span / umax if umax > 1e-15 else 1.0
        pos = xyz + sc * u
    else:
        sc, pos = 1.0, xyz

    # undeformed (ghost) outline
    if show_ghost and res is not None:
        x, y, z = seg_xyz(P3(xyz[tr.ia]), P3(xyz[tr.ib]))
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="rgba(120,130,140,0.45)", width=2),
                                   name="Original position", hoverinfo="skip"))

    # road panels
    if show_deck and tr.deck:
        vx, tris = [], []
        for p in tr.deck:
            base = len(vx)
            vx.extend(pos[p["idx"]].tolist())
            tris.append([base, base + 1, base + 2])
            if len(p["idx"]) == 4:
                tris.append([base, base + 2, base + 3])
        vp = P3(np.array(vx))
        tri = np.array(tris)
        fig.add_trace(go.Mesh3d(x=vp[:, 0], y=vp[:, 1], z=vp[:, 2], i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                                color="#8d99a6", opacity=0.45, name="Road sheet", showlegend=True, hoverinfo="skip"))

    # sticks
    pa, pb = P3(pos[tr.ia]), P3(pos[tr.ib])
    x, y, z = seg_xyz(pa, pb)
    if res is None:
        colour = np.full(m, 0.0)
        cmin, cmax, cs, title = 0.0, 1.0, [[0, "#4b5d6e"], [1, "#4b5d6e"]], ""
        if diag is not None and diag["redundant_sticks"]:
            red = np.isin(tr.sid, diag["redundant_sticks"])
            colour = red.astype(float)
            cs, cmin, cmax = [[0, "#4b5d6e"], [1, C_PURPLE]], 0.0, 1.0
        show_bar = False
    else:
        show_bar = True
        if color_by == "Axial stress σ":
            colour = res["sigma"] / 1e6
            lim = max(float(np.abs(colour).max()), 1e-9)
            cmin, cmax, cs, title = -lim, lim, "RdBu_r", "σ (MPa)  red = tension"
        elif color_by == "Axial strain ε":
            colour = res["strain"] * 1e6
            lim = max(float(np.abs(colour).max()), 1e-9)
            cmin, cmax, cs, title = -lim, lim, "RdBu_r", "ε (µε)  red = tension"
        elif color_by == "Utilisation (load / capacity)":
            colour = res["util"]
            cmin, cmax, cs, title = 0.0, max(1.0, float(colour.max())), "Turbo", "load / capacity"
        else:
            colour = 0.5 * (res["disp"][tr.ia] + res["disp"][tr.ib]) * 1000.0
            cmin, cmax, cs, title = 0.0, max(float(colour.max()), 1e-9), "Turbo", "displacement (mm)"
    line = dict(color=seg_values(colour), colorscale=cs, cmin=cmin, cmax=cmax, width=7)
    if show_bar:
        line["colorbar"] = dict(title=dict(text=title, side="right"), thickness=14, len=0.7)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=line, name="Sticks", hoverinfo="skip",
                               showlegend=False))

    # hover information at the middle of each stick
    mid = P3(0.5 * (pos[tr.ia] + pos[tr.ib]))
    if res is not None:
        hover = [
            f"<b>Stick {tr.sid[k]}</b> (joints {tr.jid[tr.ia[k]]}–{tr.jid[tr.ib[k]]})<br>"
            f"{tr.mat[k]}, Ø {tr.d[k] * 1000:.1f} mm, L = {tr.L[k]:.3f} m<br>"
            f"Force {res['F'][k] / 1000:.2f} kN ({'tension' if res['F'][k] >= 0 else 'compression'})<br>"
            f"σ = {res['sigma'][k] / 1e6:.2f} MPa, ε = {res['strain'][k] * 1e6:.1f} µε<br>"
            f"Factor of safety {('∞' if not np.isfinite(res['fs'][k]) else f'{res['fs'][k]:.2f}')} ({res['mode'][k]})"
            for k in range(m)]
    else:
        hover = [f"Stick {tr.sid[k]}: joints {tr.jid[tr.ia[k]]}–{tr.jid[tr.ib[k]]}, L = {tr.L[k]:.3f} m" for k in range(m)]
    fig.add_trace(go.Scatter3d(x=mid[:, 0], y=mid[:, 1], z=mid[:, 2], mode="markers",
                               marker=dict(size=6, color="rgba(0,0,0,0.02)"), hovertext=hover, hoverinfo="text",
                               showlegend=False))

    # joints
    jp = P3(pos)
    if res is not None:
        jh = [f"<b>Joint {tr.jid[k]}</b><br>u = ({res['u'][k, 0] * 1000:.2f}, {res['u'][k, 1] * 1000:.2f}, "
              f"{res['u'][k, 2] * 1000:.2f}) mm<br>|u| = {res['disp'][k] * 1000:.2f} mm" for k in range(j)]
    else:
        jh = [f"Joint {tr.jid[k]} at ({xyz[k, 0]:.2f}, {xyz[k, 1]:.2f}, {xyz[k, 2]:.2f}) m" for k in range(j)]
    mech = set(diag["mech_joints"]) if (diag is not None and not diag["ok"]) else set()
    jcol = [C_ORANGE if int(tr.jid[k]) in mech else "#1f2a36" for k in range(j)]
    fig.add_trace(go.Scatter3d(x=jp[:, 0], y=jp[:, 1], z=jp[:, 2],
                               mode="markers+text" if show_numbers else "markers",
                               text=[str(int(v)) for v in tr.jid] if show_numbers else None,
                               textposition="top center", textfont=dict(size=11),
                               marker=dict(size=[7 if c == C_ORANGE else 3.5 for c in jcol], color=jcol),
                               hovertext=jh, hoverinfo="text", name="Joints", showlegend=False))

    # largest displacement
    if res is not None:
        k = res["jmax"]
        fig.add_trace(go.Scatter3d(x=[jp[k, 0]], y=[jp[k, 1]], z=[jp[k, 2]], mode="markers+text",
                                   marker=dict(size=11, color="#ffd400", symbol="diamond",
                                               line=dict(color="black", width=2)),
                                   text=[f"max displacement {res['disp'][k] * 1000:.2f} mm (joint {tr.jid[k]})"],
                                   textposition="top center", textfont=dict(size=13, color="#b08900"),
                                   name="Max displacement", hoverinfo="skip"))
        s = res["imax_sigma"]
        xs_, ys_, zs_ = seg_xyz(pa[s:s + 1], pb[s:s + 1])
        fig.add_trace(go.Scatter3d(x=xs_, y=ys_, z=zs_, mode="lines", line=dict(color="black", width=12),
                                   name=f"Max stress (stick {tr.sid[s]})", hoverinfo="skip"))

    # supports
    if show_supports and tr.sup:
        sj = sorted({jj for jj, _ in tr.sup})
        cnt = {jj: sum(1 for a, _ in tr.sup if a == jj) for jj in sj}
        dirs = {jj: "".join("XYZ"[d] for a, d in tr.sup if a == jj) for jj in sj}
        sp_ = P3(pos[sj] - np.array([0.0, 0.04 * span, 0.0]))
        fig.add_trace(go.Scatter3d(x=sp_[:, 0], y=sp_[:, 1], z=sp_[:, 2], mode="markers",
                                   marker=dict(size=7, color=C_GREEN, symbol=["square" if cnt[jj] == 3 else "diamond" for jj in sj],
                                               line=dict(color="white", width=1)),
                                   hovertext=[f"Support at joint {tr.jid[jj]}: fixes {dirs[jj]}" for jj in sj],
                                   hoverinfo="text", name="Supports"))

    # loads
    if show_loads:
        head = 0.025 * span
        L_arrow = 0.10 * span
        arrows = Arrows(C_RED, "Joint loads")
        for k in range(j):
            f = tr.jload[k] if env["use_joint"] else np.zeros(3)
            n = np.linalg.norm(f)
            if n > 0:
                d = f / n
                arrows.add(pos[k] - d * L_arrow, pos[k], f"{n / 1000:.3g} kN", head)
        for tr_ in arrows.trace():
            fig.add_trace(tr_)
        if env["use_wind"] and np.linalg.norm(env["wind"]) > 0:
            v = env["wind"]
            d = v / np.linalg.norm(v)
            c = 0.5 * (xyz.min(axis=0) + xyz.max(axis=0))
            wa = Arrows(C_BLUE, "Wind")
            for off in (-0.25, 0.0, 0.25):
                perp = np.cross(d, [0.0, 1.0, 0.0])
                perp = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 1e-9 else np.array([1.0, 0.0, 0.0])
                base = c - d * 0.55 * span + perp * off * span + np.array([0.0, 0.25 * span, 0.0])
                wa.add(base, base + d * 0.25 * span, f"{np.linalg.norm(v):.3g} m/s" if off == 0.0 else None, head)
            for t_ in wa.trace():
                fig.add_trace(t_)

    fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=0.02, x=0.01),
        scene=dict(aspectmode="data",
                   xaxis=dict(title="X  (span, m)"), yaxis=dict(title="Z", showticklabels=False),
                   zaxis=dict(title="Y  (up, m)"),
                   camera=dict(eye=dict(x=1.35, y=-1.9, z=0.9))),
    )
    return fig, sc


# ----------------------------------------------------------------------------
# Interface helpers
# ----------------------------------------------------------------------------

CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1500px;}
.hero {
    background: linear-gradient(120deg, #1b2a41 0%, #324a5f 50%, #c9a227 120%);
    padding: 1.3rem 1.8rem; border-radius: 16px; margin-bottom: 1rem;
    box-shadow: 0 6px 22px rgba(20, 30, 45, 0.28);
}
.hero h1 {margin: 0; padding: 0; font-size: 2rem; color: #ffffff; letter-spacing: .3px;}
.hero p {margin: .35rem 0 0 0; color: rgba(255,255,255,.88); font-size: 1rem;}
.chips {margin-top: .7rem;}
.chip {
    display: inline-block; padding: .18rem .7rem; margin: 0 .4rem .3rem 0;
    border-radius: 999px; background: rgba(255,255,255,.16); color: #fff;
    font-size: .82rem; border: 1px solid rgba(255,255,255,.28);
}
div[data-testid="stMetric"] {
    background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.22);
    border-radius: 12px; padding: .7rem .9rem;
}
div[data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .8;}
.stTabs [data-baseweb="tab-list"] {gap: .3rem;}
.stTabs [data-baseweb="tab"] {font-weight: 600; padding: .5rem 1.1rem;}
.sectiontitle {font-size: 1.05rem; font-weight: 700; margin: 1rem 0 .3rem 0;}
</style>
"""


def fmt(v, unit="", digits=4):
    return f"{v:.{digits}g}{(' ' + unit) if unit else ''}"


def apply_template(kind, n, p, H, W, dc, dv, dd, dx, mat, road_t, road_mat):
    st.session_state["tables"] = (blank_tables() if kind == "Blank" else
                                  standard_bridge(kind.split(" ")[0], int(n), p, H, W, dc, dv, dd, dx, mat, road_t, road_mat))
    st.session_state["ver"] = st.session_state.get("ver", 0) + 1


def _load_clicked():
    s = st.session_state
    apply_template(s["tpl_kind"], s["tpl_n"], s["tpl_p"], s["tpl_H"], s["tpl_W"], s["tpl_dc"], s["tpl_dv"],
                   s["tpl_dd"], s["tpl_dx"], s["tpl_mat"], s["tpl_rt"], s["tpl_rm"])


def sidebar_settings():
    sb = st.sidebar
    sb.markdown("## ⚙️ Settings")
    max_val = sb.number_input("Max sticks per pivot", min_value=3, max_value=24, value=8, step=1,
                              help="Every joint must have 2 up to this many sticks. A stable, determinate 3D "
                                   "bridge needs about 6 sticks per joint on average; set 4 to test a small design "
                                   "against the original 2-3-4 rule.")
    fs_req = sb.number_input("Required factor of safety", min_value=1.0, max_value=10.0, value=1.5, step=0.1,
                             help="A stick passes when its capacity / force is at least this value.")
    K = sb.number_input("Effective length factor K", min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                        help="Pin joints at both ends: K = 1 (Euler buckling P_cr = π²EI/(KL)²).")
    temp = sb.number_input("Temperature of the material properties (°C)", min_value=-100.0, max_value=1000.0,
                           value=20.0, step=5.0,
                           help="Material properties are constants at this temperature. The value is only a label: "
                                "the app does not change E or strength with temperature and calculates no thermal "
                                "stress. Enter properties for this temperature in the Materials table.")
    sb.caption("Coordinates: X along the span, Y up, Z across the road. Loads in global axes: "
               "downward is negative Y.")
    return dict(max_val=int(max_val), fs_req=float(fs_req), K=float(K), temp=float(temp))


def table_editor(label, key, df, ver, config, help_text=None):
    if help_text:
        st.caption(help_text)
    return st.data_editor(df, key=f"{key}_{ver}", num_rows="dynamic", hide_index=True, column_config=config,
                          **STRETCH)


def tab_environment(S):
    st.markdown('<div class="sectiontitle">Load cases included</div>', unsafe_allow_html=True)
    c = st.columns(6)
    use_self = c[0].checkbox("Stick weight", value=True)
    use_road = c[1].checkbox("Road weight", value=True)
    use_traffic = c[2].checkbox("Traffic on road", value=True)
    use_wind = c[3].checkbox("Wind (speed)", value=True)
    use_direct = c[4].checkbox("Wind (force on sticks)", value=True)
    use_joint = c[5].checkbox("Joint loads", value=True)

    a, b = st.columns(2)
    with a:
        st.markdown('<div class="sectiontitle">Gravity and traffic</div>', unsafe_allow_html=True)
        g = st.number_input("Gravity g (m/s²)", min_value=0.0, max_value=30.0, value=G_ACC, step=0.01)
        traffic = st.number_input("Traffic load on the road (kPa = kN/m²)", min_value=0.0, max_value=100.0,
                                  value=5.0, step=0.5,
                                  help="Downward pressure on the horizontal area of every road panel, shared equally "
                                       "between the joints of the panel. 5 kPa is about a crowd of pedestrians.")
    with b:
        st.markdown('<div class="sectiontitle">Wind speed (3D)</div>', unsafe_allow_html=True)
        st.caption("Air velocity in global axes. Wind pushes each stick with drag F = ½ρ·Cd·d·L·|v⊥|·v⊥, where v⊥ is "
                   "the wind component perpendicular to the stick. Shielding of one truss by another is ignored.")
        w1, w2, w3 = st.columns(3)
        vx = w1.number_input("vx (m/s)", value=0.0, step=1.0)
        vy = w2.number_input("vy (m/s)", value=0.0, step=1.0)
        vz = w3.number_input("vz (m/s)", value=25.0, step=1.0, help="Positive Z blows across the road toward the viewer.")
        d1, d2 = st.columns(2)
        cd_stick = d1.number_input("Drag coefficient, sticks", min_value=0.1, max_value=3.0, value=1.2, step=0.1)
        cd_deck = d2.number_input("Drag coefficient, road sheet", min_value=0.1, max_value=3.0, value=1.3, step=0.1)

    env = dict(use_self=use_self, use_road=use_road, use_traffic=use_traffic, use_wind=use_wind,
               use_direct=use_direct, use_joint=use_joint, g=float(g), traffic_kpa=float(traffic),
               wind=np.array([vx, vy, vz], dtype=float), cd_stick=float(cd_stick), cd_deck=float(cd_deck),
               K=S["K"])

    st.markdown('<div class="sectiontitle">Materials (properties at the chosen temperature)</div>', unsafe_allow_html=True)
    st.caption(f"Constant properties taken at {S['temp']:g} °C. The values are approximate textbook numbers near room "
               f"temperature: edit them, or add your own material, for any other temperature. Strength is the yield "
               f"(or allowable) stress used for both tension and compression.")
    if "mats" not in st.session_state:
        st.session_state["mats"] = pd.DataFrame(DEFAULT_MATERIALS, columns=MAT_COLS)
    mats = st.data_editor(st.session_state["mats"], key="mats_editor", num_rows="dynamic", hide_index=True,
                          **STRETCH,
                          column_config={
                              "Material": st.column_config.TextColumn("Material", required=True),
                              "E (GPa)": st.column_config.NumberColumn("E (GPa)", format="%.1f", min_value=0.001),
                              "Density (kg/m3)": st.column_config.NumberColumn("Density (kg/m3)", format="%.0f", min_value=1.0),
                              "Strength (MPa)": st.column_config.NumberColumn("Strength (MPa)", format="%.1f", min_value=0.001),
                          })

    ver = st.session_state["ver"]
    T = st.session_state["tables"]
    st.markdown('<div class="sectiontitle">Point loads on joints</div>', unsafe_allow_html=True)
    st.caption("Force on a joint in global axes, in kN. A downward load has a negative Fy.")
    jloads = st.data_editor(T["jloads"], key=f"ed_jloads_{ver}", num_rows="dynamic", hide_index=True,
                            **STRETCH,
                            column_config={"Joint": st.column_config.NumberColumn("Joint", format="%d", step=1),
                                           "Fx (kN)": st.column_config.NumberColumn(format="%.3f"),
                                           "Fy (kN)": st.column_config.NumberColumn(format="%.3f"),
                                           "Fz (kN)": st.column_config.NumberColumn(format="%.3f")})
    st.markdown('<div class="sectiontitle">Wind force put directly on sticks</div>', unsafe_allow_html=True)
    st.caption("Total 3D wind force on one stick, in N, in global axes. Half goes to each end joint. "
               "This is added to the wind-speed loads when both are switched on.")
    swind = st.data_editor(T["swind"], key=f"ed_swind_{ver}", num_rows="dynamic", hide_index=True,
                           **STRETCH,
                           column_config={"Stick": st.column_config.NumberColumn("Stick", format="%d", step=1),
                                          "Fx (N)": st.column_config.NumberColumn(format="%.2f"),
                                          "Fy (N)": st.column_config.NumberColumn(format="%.2f"),
                                          "Fz (N)": st.column_config.NumberColumn(format="%.2f")})
    return env, mats, jloads, swind


def tab_build(mat_names):
    ver = st.session_state["ver"]
    T = st.session_state["tables"]
    st.markdown('<div class="sectiontitle">Start from a standard bridge</div>', unsafe_allow_html=True)
    st.caption("Standard models are 3D box trusses: two side trusses, cross members and bracing, with the road on the "
               "bottom chords. They are exactly determinate (m + r = 3j), so the method of joints solves them. "
               "Loading a model replaces the tables below; after that you can edit any joint, stick or support.")
    c = st.columns([1.3, 1, 1, 1, 1])
    c[0].selectbox("Model", STANDARD_MODELS, key="tpl_kind")
    c[1].number_input("Bays", min_value=2, max_value=12, value=4, step=1, key="tpl_n")
    c[2].number_input("Bay length (m)", min_value=0.5, max_value=20.0, value=3.0, step=0.5, key="tpl_p")
    c[3].number_input("Height (m)", min_value=0.5, max_value=15.0, value=3.0, step=0.5, key="tpl_H")
    c[4].number_input("Width (m)", min_value=0.5, max_value=15.0, value=4.0, step=0.5, key="tpl_W")
    c = st.columns(6)
    c[0].number_input("Chords Ø (mm)", min_value=2.0, max_value=1000.0, value=120.0, step=5.0, key="tpl_dc")
    c[1].number_input("Verticals Ø (mm)", min_value=2.0, max_value=1000.0, value=90.0, step=5.0, key="tpl_dv")
    c[2].number_input("Diagonals Ø (mm)", min_value=2.0, max_value=1000.0, value=90.0, step=5.0, key="tpl_dd")
    c[3].number_input("Cross/bracing Ø (mm)", min_value=2.0, max_value=1000.0, value=70.0, step=5.0, key="tpl_dx")
    c[4].selectbox("Stick material", mat_names, index=0, key="tpl_mat")
    c[5].number_input("Road thickness (mm)", min_value=0.5, max_value=500.0, value=20.0, step=1.0, key="tpl_rt")
    c = st.columns([1, 3])
    c[0].selectbox("Road material", mat_names, index=min(len(mat_names) - 1, 0), key="tpl_rm")
    c[1].markdown("&nbsp;")
    st.button("Load this model", type="primary", on_click=_load_clicked)

    jmax = max(int(T["joints"]["Joint"].max()) if len(T["joints"]) else 1, 1)
    st.markdown('<div class="sectiontitle">Joints (pivots)</div>', unsafe_allow_html=True)
    joints = table_editor("Joints", "ed_joints", T["joints"], ver, {
        "Joint": st.column_config.NumberColumn("Joint", format="%d", step=1),
        "x (m)": st.column_config.NumberColumn(format="%.3f"),
        "y (m)": st.column_config.NumberColumn(format="%.3f"),
        "z (m)": st.column_config.NumberColumn(format="%.3f")},
        "One row per pivot. Joint numbers must be whole numbers and unique.")
    st.markdown('<div class="sectiontitle">Sticks (circular cross-section)</div>', unsafe_allow_html=True)
    sticks = table_editor("Sticks", "ed_sticks", T["sticks"], ver, {
        "Stick": st.column_config.NumberColumn("Stick", format="%d", step=1),
        "Joint A": st.column_config.NumberColumn("Joint A", format="%d", step=1),
        "Joint B": st.column_config.NumberColumn("Joint B", format="%d", step=1),
        "Diameter (mm)": st.column_config.NumberColumn("Diameter (mm)", format="%.1f", min_value=0.1),
        "Material": st.column_config.SelectboxColumn("Material", options=mat_names, required=True)},
        "Solid circular sticks pinned at both ends. Each stick carries axial force only.")
    st.markdown('<div class="sectiontitle">Supports</div>', unsafe_allow_html=True)
    supports = table_editor("Supports", "ed_supports", T["supports"], ver, {
        "Joint": st.column_config.NumberColumn("Joint", format="%d", step=1),
        "Fix X": st.column_config.CheckboxColumn("Fix X"),
        "Fix Y": st.column_config.CheckboxColumn("Fix Y"),
        "Fix Z": st.column_config.CheckboxColumn("Fix Z")},
        "Tick the directions a support holds. A stable bridge needs at least 6 held components in total; "
        "m + r must equal 3j.")
    st.markdown('<div class="sectiontitle">Road sheet panels</div>', unsafe_allow_html=True)
    deck = table_editor("Road", "ed_deck", T["deck"], ver, {
        "Panel": st.column_config.NumberColumn("Panel", format="%d", step=1),
        "Joint 1": st.column_config.NumberColumn("Joint 1", format="%d", step=1),
        "Joint 2": st.column_config.NumberColumn("Joint 2", format="%d", step=1),
        "Joint 3": st.column_config.NumberColumn("Joint 3", format="%d", step=1),
        "Joint 4": st.column_config.NumberColumn("Joint 4 (optional)", format="%d", step=1),
        "Thickness (mm)": st.column_config.NumberColumn("Thickness (mm)", format="%.1f", min_value=0.1),
        "Material": st.column_config.SelectboxColumn("Material", options=mat_names, required=True)},
        "Each panel is a thin sheet on 3 or 4 joints, listed in order around the panel. It adds weight, takes the "
        "traffic and wind loads and shares them equally with its joints; it adds no stiffness.")
    return dict(joints=joints, sticks=sticks, supports=supports, deck=deck)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="TrussLab 3D", page_icon="🌉", layout="wide")
    if go is None:
        st.error("The **plotly** package is not installed in this environment. Add a `requirements.txt` next to the "
                 "app with the lines `streamlit`, `plotly`, `numpy`, `pandas`, then reboot the app. "
                 "Locally, run `pip install plotly`.")
        st.stop()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <h1>🌉 TrussLab 3D</h1>
          <p>Build a 3D truss bridge from circular sticks, load it with weight, traffic and wind, and see where it
          moves and how strongly each stick is stressed.</p>
          <div class="chips">
            <span class="chip">Method of joints</span><span class="chip">Any material</span>
            <span class="chip">3D wind</span><span class="chip">Euler buckling</span>
            <span class="chip">Max displacement, strain, stress</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    S = sidebar_settings()

    if "ver" not in st.session_state:
        st.session_state["ver"] = 0
    if "tables" not in st.session_state:
        apply_template("Pratt", 4, 3.0, 3.0, 4.0, 120.0, 90.0, 90.0, 70.0, DEFAULT_MATERIALS[0][0], 20.0,
                       DEFAULT_MATERIALS[-1][0])

    tab_b, tab_e, tab_r, tab_3, tab_n = st.tabs(
        ["① Build", "② Loads & materials", "③ Results", "④ 3D Simulation", "⑤ Theory & notes"])

    with tab_e:
        env, mats, jloads, swind = tab_environment(S)
    mat_names = [str(v).strip() for v in mats["Material"].dropna() if str(v).strip()] or [DEFAULT_MATERIALS[0][0]]
    with tab_b:
        tb = tab_build(mat_names)
    tb["jloads"], tb["swind"] = jloads, swind

    truss, errors, warns = parse_tables(tb, mats, S["max_val"])
    if errors:
        msg = st.container()
        for e in errors:
            msg.error(e)
        for w in warns:
            msg.warning(w)
        for tab in (tab_r, tab_3):
            with tab:
                st.info("Fix the problems listed above to run the simulation.")
        with tab_n:
            theory_tab(S)
        return

    for w in warns:
        st.warning(w)
    if not truss.sup:
        st.error("Supports: no support is defined. Fix at least X, Y and Z at one joint.")
        return

    B = equilibrium_matrix(truss)
    diag = diagnose(truss, B)
    status = (f"**{len(truss.jid)}** joints  ·  **{len(truss.sid)}** sticks  ·  **{len(truss.sup)}** support components  ·  "
              f"m + r = **{diag['n_unk']}**, 3j = **{diag['n_eq']}**")
    P, comp = compute_loads(truss, env)
    res = None
    if diag["ok"]:
        try:
            res = solve_truss(truss, B, P, env)
        except np.linalg.LinAlgError:
            diag["ok"] = False
            diag["message"] = "The truss cannot be solved: the joint equations are singular (unstable layout)."

    with tab_b:
        if diag["ok"]:
            st.success(status + "  ·  statically determinate and stable: the method of joints can solve it.")
            if diag["weak"]:
                st.warning("The layout is nearly unstable (some sticks are almost in a mechanism), so forces and "
                           "displacements will be very large.")
        else:
            st.error(status + "\n\n" + diag["message"] +
                     (f"\n\nJoints that can move: {diag['mech_joints']}." if diag["mech_joints"] else "") +
                     (f"\n\nSticks in the redundant set: {diag['redundant_sticks']}." if diag["redundant_sticks"] else ""))

    with tab_r:
        if res is None:
            st.error(diag["message"])
            st.info("The 3D tab shows the geometry; joints that can move are orange and redundant sticks purple.")
        else:
            results_tab(truss, res, comp, P, S, env)

    with tab_3:
        c1, c2, c3 = st.columns([1.6, 1.6, 2.4])
        color_by = c1.radio("Colour sticks by", ["Axial stress σ", "Axial strain ε", "Utilisation (load / capacity)",
                                                 "Joint displacement"], disabled=res is None)
        exag = c2.slider("Deformation exaggeration (%)", 1, 100, 30, disabled=res is None,
                         help="100% draws the largest joint displacement as 12% of the bridge size.")
        with c3:
            true_scale = st.checkbox("Show true scale (no exaggeration)", value=False, disabled=res is None)
            d1, d2, d3 = st.columns(3)
            show_ghost = d1.checkbox("Original position", value=True, disabled=res is None)
            show_deck = d2.checkbox("Road sheet", value=True)
            show_supports = d3.checkbox("Supports", value=True)
            e1, e2 = st.columns(2)
            show_loads = e1.checkbox("Loads and wind", value=True)
            show_numbers = e2.checkbox("Joint numbers", value=False)
        fig, sc = build_figure(truss, res, diag, env, color_by, exag, true_scale, show_ghost, show_deck,
                               show_supports, show_loads, show_numbers, comp)
        st.plotly_chart(fig, **STRETCH)
        if res is not None:
            k = res["jmax"]
            st.caption(f"Displacements are drawn ×{sc:,.4g}. The yellow diamond marks the joint that moves furthest from "
                       f"its original position (joint {truss.jid[k]}, {res['disp'][k] * 1000:.2f} mm); the thick black "
                       f"stick has the largest stress. Hover over a stick or joint for details.")

    with tab_n:
        theory_tab(S)


def results_tab(tr: Truss, res: dict, comp: dict, P: np.ndarray, S: dict, env: dict):
    k = res["jmax"]
    sj = res["at_joint"]
    s_here = sj[int(np.argmax(np.abs(res["sigma"][sj])))] if len(sj) else res["imax_sigma"]
    is_ = res["imax_sigma"]
    fs_min = res["fs"][res["imin_fs"]]
    all_ok = bool(np.all((res["fs"] >= S["fs_req"]) | ~np.isfinite(res["fs"])))

    if all_ok:
        st.success(f"All sticks pass: the lowest factor of safety is "
                   f"{'∞' if not np.isfinite(fs_min) else f'{fs_min:.2f}'} (required {S['fs_req']:g}).")
    else:
        bad = [int(tr.sid[i]) for i in np.where(res["fs"] < S["fs_req"])[0]]
        st.error(f"{len(bad)} stick(s) fail the required factor of safety of {S['fs_req']:g}: sticks {bad[:20]}"
                 f"{' …' if len(bad) > 20 else ''}. The lowest factor of safety is {fs_min:.2f} "
                 f"(stick {tr.sid[res['imin_fs']]}, {res['mode'][res['imin_fs']]}).")

    st.markdown('<div class="sectiontitle">Largest displacement from the original position</div>', unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Max displacement", fmt(res["disp"][k] * 1000, "mm"), help=f"Joint {tr.jid[k]}")
    c[1].metric("At joint", f"{tr.jid[k]}", help=f"Original position: x = {tr.xyz[k, 0]:.3f}, y = {tr.xyz[k, 1]:.3f}, "
                                                 f"z = {tr.xyz[k, 2]:.3f} m")
    c[2].metric("Components (x, y, z)", f"{res['u'][k, 0] * 1000:.2f}, {res['u'][k, 1] * 1000:.2f}, {res['u'][k, 2] * 1000:.2f} mm")
    c[3].metric("Joint position now", f"{tr.xyz[k, 0] + res['u'][k, 0]:.4f}, {tr.xyz[k, 1] + res['u'][k, 1]:.4f}, "
                                      f"{tr.xyz[k, 2] + res['u'][k, 2]:.4f} m")
    st.markdown(f"**Strain and stress at joint {tr.jid[k]}** (the sticks that meet there):")
    here = pd.DataFrame({
        "Stick": tr.sid[sj], "Force (kN)": res["F"][sj] / 1000.0, "Stress σ (MPa)": res["sigma"][sj] / 1e6,
        "Strain ε (µε)": res["strain"][sj] * 1e6, "Strain (%)": res["strain"][sj] * 100.0,
        "Factor of safety": res["fs"][sj]})
    st.dataframe(here.style.format({"Force (kN)": "{:,.3f}", "Stress σ (MPa)": "{:,.3f}", "Strain ε (µε)": "{:,.1f}",
                                    "Strain (%)": "{:,.5f}", "Factor of safety": "{:,.2f}"}), hide_index=True)
    st.caption(f"Largest stress among them: stick {tr.sid[s_here]}, σ = {res['sigma'][s_here] / 1e6:.3f} MPa "
               f"({'tension' if res['sigma'][s_here] >= 0 else 'compression'}), ε = {res['strain'][s_here] * 1e6:.1f} µε.")

    st.markdown('<div class="sectiontitle">Whole bridge</div>', unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Max stress |σ|", fmt(abs(res["sigma"][is_]) / 1e6, "MPa"),
                help=f"Stick {tr.sid[is_]}, {'tension' if res['sigma'][is_] >= 0 else 'compression'}")
    c[1].metric("In stick", f"{tr.sid[is_]}", help=f"Joints {tr.jid[tr.ia[is_]]}–{tr.jid[tr.ib[is_]]}")
    c[2].metric("Max strain |ε|", fmt(abs(res["strain"][res["imax_strain"]]) * 1e6, "µε"),
                help=f"Stick {tr.sid[res['imax_strain']]}")
    c[3].metric("Lowest factor of safety", "∞" if not np.isfinite(fs_min) else f"{fs_min:.2f}",
                help=f"Stick {tr.sid[res['imin_fs']]}, governed by {res['mode'][res['imin_fs']]}")
    mass_sticks = float((tr.rho * tr.A * tr.L).sum())
    mass_road = float(sum(p["rho"] * p["t"] * deck_geometry(tr, p)[0] for p in tr.deck))
    c = st.columns(4)
    c[0].metric("Stick mass", fmt(mass_sticks, "kg"))
    c[1].metric("Road sheet mass", fmt(mass_road, "kg"))
    c[2].metric("Total load, vertical", fmt(P.reshape(-1, 3)[:, 1].sum() / 1000, "kN"))
    c[3].metric("Total load, horizontal", fmt(float(np.hypot(P.reshape(-1, 3)[:, 0].sum(), P.reshape(-1, 3)[:, 2].sum())) / 1000, "kN"))

    st.markdown('<div class="sectiontitle">Sticks</div>', unsafe_allow_html=True)
    df = pd.DataFrame({
        "Stick": tr.sid, "Joints": [f"{tr.jid[a]}–{tr.jid[b]}" for a, b in zip(tr.ia, tr.ib)],
        "Material": tr.mat, "Ø (mm)": tr.d * 1000, "L (m)": tr.L, "Force (kN)": res["F"] / 1000,
        "Type": np.where(res["F"] >= 0, "tension", "compression"), "σ (MPa)": res["sigma"] / 1e6,
        "ε (µε)": res["strain"] * 1e6, "Elongation (mm)": res["delta"] * 1000,
        "Capacity (kN)": res["cap"] / 1000, "Governed by": res["mode"], "Slenderness KL/r": res["slender"],
        "Factor of safety": res["fs"],
        "Status": np.where(~np.isfinite(res["fs"]) | (res["fs"] >= S["fs_req"]), "OK", "FAIL")})
    st.dataframe(df.style.format({"Ø (mm)": "{:.1f}", "L (m)": "{:.3f}", "Force (kN)": "{:,.3f}", "σ (MPa)": "{:,.3f}",
                                  "ε (µε)": "{:,.1f}", "Elongation (mm)": "{:,.4f}", "Capacity (kN)": "{:,.1f}",
                                  "Slenderness KL/r": "{:,.0f}", "Factor of safety": "{:,.2f}"}),
                 hide_index=True, height=360)
    st.download_button("Download stick results (CSV)", df.to_csv(index=False), "stick_results.csv", "text/csv")

    st.markdown('<div class="sectiontitle">Joint displacements and support reactions</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        dj = pd.DataFrame({"Joint": tr.jid, "ux (mm)": res["u"][:, 0] * 1000, "uy (mm)": res["u"][:, 1] * 1000,
                           "uz (mm)": res["u"][:, 2] * 1000, "|u| (mm)": res["disp"] * 1000})
        st.dataframe(dj.style.format({c: "{:,.3f}" for c in dj.columns if c != "Joint"}), hide_index=True, height=300)
    with b:
        R = res["R"]
        rows = {}
        for q, (jj, dd) in enumerate(tr.sup):
            rows.setdefault(int(tr.jid[jj]), [0.0, 0.0, 0.0])[dd] = R[q] / 1000.0
        rr = pd.DataFrame([[jj, *v] for jj, v in rows.items()], columns=["Joint", "Rx (kN)", "Ry (kN)", "Rz (kN)"])
        st.dataframe(rr.style.format({c: "{:,.3f}" for c in rr.columns if c != "Joint"}), hide_index=True)
        tot = P.reshape(-1, 3).sum(axis=0) / 1000.0
        st.caption(f"Sum of reactions: ({R_sum(tr, R)[0]:.3f}, {R_sum(tr, R)[1]:.3f}, {R_sum(tr, R)[2]:.3f}) kN. "
                   f"Sum of loads: ({tot[0]:.3f}, {tot[1]:.3f}, {tot[2]:.3f}) kN. "
                   f"Largest joint-equilibrium error: {res['resid']:.2e} N.")

    st.markdown('<div class="sectiontitle">Load summary (resultant force of each load case, kN)</div>', unsafe_allow_html=True)
    ls = pd.DataFrame([[name, *(v.sum(axis=0) / 1000.0)] for name, v in comp.items()],
                      columns=["Load case", "Fx", "Fy", "Fz"])
    st.dataframe(ls.style.format({"Fx": "{:,.3f}", "Fy": "{:,.3f}", "Fz": "{:,.3f}"}), hide_index=True)


def R_sum(tr: Truss, R: np.ndarray):
    s = np.zeros(3)
    for q, (_, dd) in enumerate(tr.sup):
        s[dd] += R[q] / 1000.0
    return s


def theory_tab(S: dict):
    st.markdown(
        """
#### Method (APL104 topics)

| Step | Idea | Lecture |
|---|---|---|
| Sticks pinned at both ends, loads at joints | Each stick is a two-force member: it carries axial force only | L5–6 |
| Member forces | **Method of joints**: the sum of forces on every pin is zero in x, y and z. All joints are written together as **B·x = −P** (x = stick forces and support reactions) and solved | L5–6 |
| Stress and strain | σ = F/A and ε = σ/E (Hooke's law); elongation Δ = F·L/(E·A) | L11–12, L15–16 |
| Joint displacements | **Unit-load (virtual-work) method**: δ = Σ f·F·L/(E·A), where f is the stick force caused by a unit load at the joint. It is the same as joining the stick elongations with zero movement at the supports | L29–30 |
| Buckling | Compression sticks: **Euler load** P_cr = π²·E·I/(K·L)², I = π·d⁴/64 | L28 |
| Strength | Capacity = σ_y·A in tension; min(σ_y·A, P_cr) in compression. Factor of safety = capacity / force | L31–32 (failure) |

#### Why the truss must be determinate
Method of joints gives 3j equations for m stick forces and r support reactions. It works when **m + r = 3j** and the
layout is stable. If m + r > 3j the truss has redundant sticks or supports and needs more than equilibrium
(compatibility with stiffness); if the layout is unstable it is a mechanism. The app reports which case you have and marks the
joints that can move or the sticks that are redundant. A stable determinate 3D truss needs about 6 sticks per joint on
average, which is why the pivot limit is a setting.

#### Not covered by the APL104 page
- **Wind drag**: F = ½·ρ·Cd·d·L·|v⊥|·v⊥ per stick and ½·ρ·Cd·A·|vₙ|·vₙ on the road sheet (fluid drag).
- **Road load sharing**: weight, traffic and wind on a road panel are shared equally between its joints.
- Solving the joint equations numerically (a linear-algebra routine).

#### Assumptions
- Linear-elastic material and small displacements; solid circular sticks with constant properties at one temperature
  (no thermal stress, no temperature change of E or strength).
- Frictionless pins; loads act only at joints; each stick's weight and wind load is split half-and-half between its ends.
- The road sheet is a load carrier only. It adds no stiffness and its edges do not resist wind in-plane.
- Wind is a steady (static) load. Gusts, vortex shedding, shielding between trusses and fatigue are not modelled.
- Euler buckling uses the effective-length factor K from the sidebar; joint connectors have no weight or size.
- Educational tool: not a substitute for code-compliant design.
        """)
    st.caption(f"Current settings: required factor of safety {S['fs_req']:g}, K = {S['K']:g}, "
               f"max sticks per pivot {S['max_val']}, properties at {S['temp']:g} °C.")


if __name__ == "__main__":
    main()
