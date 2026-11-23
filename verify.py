"""Before/after evidence for locally refined dfsu. Usage: verify.py <tag> <outdir>"""
import sys, warnings, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mikeio
from mikecore.DfsuFile import DfsuFile

warnings.simplefilter("ignore")
tag, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
fn = "tests/testdata/submesh/Dfsu2D_CustomBlock6_Submesh1_LocallyRefined.dfsu"
ds = mikeio.read(fn)
g = ds.geometry
print(f"[{tag}] exteriors:", len(g.boundary_polygons.exteriors), "interiors:", len(g.boundary_polygons.interiors))
ds.to_dfs(f"{out}/copy.dfsu")
print(f"[{tag}] submesh type in/out:", DfsuFile.Open(fn).SubMeshType, DfsuFile.Open(f"{out}/copy.dfsu").SubMeshType)
ec = g.element_coordinates
f = lambda x, y: 2 * x + 3 * y + 10
lin = f(ec[:, 0], ec[:, 1])
da = mikeio.DataArray(lin, geometry=g, item="linear")
nc = g.node_coordinates
zn = g.get_node_centered_data(lin)
err = np.abs(zn - f(nc[:, 0], nc[:, 1]))
hang = np.array([10, 12])
interior = np.where(g.codes == 0)[0]
others = np.setdiff1d(interior, hang)
print(f"[{tag}] node error at hanging nodes 11,13:", np.round(err[hang], 2), " max at other internal nodes:", round(err[others].max(), 2))
print(f"[{tag}] points on interior line inside:", g.contains(nc[hang, :2]).tolist())
for pt in ["contour", "contourf", "shaded"]:
    try:
        da.plot(plot_type=pt, n_refinements=1)
        print(f"[{tag}] {pt} n_refinements=1: OK")
    except Exception as e:
        print(f"[{tag}] {pt} n_refinements=1: {type(e).__name__}: {e}")
    plt.close("all")

fig, axs = plt.subplots(1, 4, figsize=(22, 5))
ax = axs[0]
for p in g.boundary_polygons.exteriors:
    ax.fill(*p.xy.T, alpha=0.3); ax.plot(*p.xy.T)
ax.set_title(f"boundary_polygons: {len(g.boundary_polygons.exteriors)} exterior(s)")
g.plot.outline(ax=axs[1]); axs[1].set_title("plot.outline()")
da.plot.contour(ax=axs[2], levels=20); axs[2].set_title("contour of 2x+3y+10")
sc = axs[3].scatter(nc[:, 0], nc[:, 1], c=err, cmap="Reds", vmin=0, vmax=35)
fig.colorbar(sc, ax=axs[3]); axs[3].set_title("|node value error|")
for a in axs:
    a.plot(nc[hang, 0], nc[hang, 1], "o", mfc="none", mec="r", ms=12)
    a.set_aspect("equal")
fig.suptitle(tag); fig.tight_layout(); fig.savefig(f"{out}/{tag}-panels.png", dpi=80)
plt.close("all")
fig, ax = plt.subplots(figsize=(7, 6))
da.plot.contourf(ax=ax, levels=40, show_mesh=True)
ax.plot(nc[hang, 0], nc[hang, 1], "ro")
x0, y0 = nc[hang, 0].mean(), nc[hang, 1].mean()
d = 1.2 * np.ptp(nc[hang, 0]) + 1.2 * np.ptp(nc[hang, 1])
ax.set_xlim(x0 - d, x0 + d); ax.set_ylim(y0 - d, y0 + d)
ax.set_title(f"{tag}: contourf zoom"); fig.savefig(f"{out}/{tag}-contourf-zoom.png", dpi=80)
