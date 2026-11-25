"""Node values of a linear field: MIKE IO get_node_centered_data vs a port of DHI.Mesh MeshNodeInterpolation.

DHI.Mesh: https://github.com/DHI/DHI.Mesh/blob/c23cd612827ad0e2cfd0d2823774442149ddc7f2/src/DHI.Mesh/MeshNodeInterpolation.cs
Usage (from the mikeio repo root): python dhimesh_compare.py [out.png]
"""
import sys, warnings
import numpy as np
import mikeio

warnings.simplefilter("ignore")


def dhi_mesh_weights(nc, ec, node_elements, allow_extrapolation):
    """Port of MeshNodeInterpolation.SetupNodeInterpolation, one Node at a time."""
    out = []
    for n, it in enumerate(node_elements):
        d = ec[it, :2] - nc[n, :2]
        w = np.zeros(len(it))
        tot = 0.0
        if len(it) >= 3:
            Ixx, Iyy = (d**2).sum(0)
            Ixy = (d[:, 0] * d[:, 1]).sum()
            Rx, Ry = d.sum(0)
            lam = Ixx * Iyy - Ixy**2
            if lam > 1e-10 * (Ixx * Iyy):
                lx = (Ixy * Ry - Iyy * Rx) / lam
                ly = (Ixy * Rx - Ixx * Ry) / lam
                w = 1.0 + lx * d[:, 0] + ly * d[:, 1]
                if not allow_extrapolation:
                    w = w.clip(0, 2)
                tot = w.sum()
        if tot <= 1e-10:
            w = 1 / np.hypot(d[:, 0], d[:, 1])
            tot = w.sum()
        out.append(w / tot if tot != 0 else np.zeros(len(it)))
    return out


def node_elements(table, n_nodes):
    ne = [[] for _ in range(n_nodes)]
    for e, nodes in enumerate(table):
        for n in nodes:
            ne[n].append(e)
    return [np.array(x) for x in ne]


def dhi_mesh(g, values, allow_extrapolation):
    table = g._conforming_element_table
    table = g.element_table if table is None else table
    ne = node_elements(table, g.n_nodes)
    ws = dhi_mesh_weights(g.node_coordinates, g.element_coordinates, ne, allow_extrapolation)
    return np.array([(w * values[it]).sum() for w, it in zip(ws, ne)])


def linear(x, y):
    return 2 * x + 3 * y + 10


FILES = {
    "Submesh1 locally refined": "tests/testdata/submesh/Dfsu2D_CustomBlock6_Submesh1_LocallyRefined.dfsu",
    "oresundHD_run1": "tests/testdata/oresundHD_run1.dfsu",
    "HD2D": "tests/testdata/HD2D.dfsu",
}

def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mikeio.spatial._FM_plot as P

    current = P._get_node_centered_data
    g = mikeio.open(FILES["Submesh1 locally refined"]).geometry
    nc, ec = g.node_coordinates, g.element_coordinates

    def ported(nodes, table, centres, data, extrapolate=True, conforming=None):
        return dhi_mesh(g, data, True)

    fields = {"2x+3y+10": linear(ec[:, 0], ec[:, 1]), "x² + y²": ec[:, 0] ** 2 + ec[:, 1] ** 2}
    hang = np.array([10, 12])
    fig, axs = plt.subplots(2, 2, figsize=(13, 11))
    for r, (name, v) in enumerate(fields.items()):
        for c, (lbl, fn) in enumerate([("MIKE IO", current), ("DHI.Mesh port, AllowExtrapolation=true", ported)]):
            P._get_node_centered_data = fn
            ax = axs[r, c]
            mikeio.DataArray(v, geometry=g, item=name).plot.contourf(ax=ax, levels=30, show_mesh=True, add_colorbar=False)
            ax.plot(nc[hang, 0], nc[hang, 1], "ro")
            ax.set_title(f"{name}: {lbl}")
    P._get_node_centered_data = current
    fig.tight_layout()
    fig.savefig(out, dpi=80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot(sys.argv[1])
    print("max |error| of 2x + 3y + 10 at Nodes (field range in brackets)")
    for name, fn in FILES.items():
        g = mikeio.open(fn).geometry
        nc, ec = g.node_coordinates, g.element_coordinates
        x0, y0 = nc[:, 0].min(), nc[:, 1].min()
        v = linear(ec[:, 0] - x0, ec[:, 1] - y0)
        truth = linear(nc[:, 0] - x0, nc[:, 1] - y0)
        internal = ~np.isin(np.arange(g.n_nodes), g._get_boundary_faces())  # codes can be 0 on the boundary
        hanging = np.zeros(g.n_nodes, bool)
        if g._conforming_element_table is not None:
            for a, b in zip(g.element_table, g._conforming_element_table):
                hanging[np.setdiff1d(b, a)] = True
        rows = {
            "MIKE IO": g.get_node_centered_data(v),
            "DHI.Mesh, AllowExtrapolation=true": dhi_mesh(g, v, True),
            "DHI.Mesh, default (clipped)": dhi_mesh(g, v, False),
        }
        print(f"\n{name} ({g.n_elements} Elements, range {np.ptp(truth):.4g})")
        for lbl, z in rows.items():
            e = np.abs(z - truth)
            msg = f"  {lbl:36s} internal {e[internal & ~hanging].max():9.3g}   boundary {e[~internal].max():9.3g}"
            if hanging.any():
                msg += f"   hanging {e[hanging].max():9.3g}"
            print(msg)
