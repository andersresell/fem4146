import numpy as np
import matplotlib.pyplot as plt
import signal
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import fem_utils
from matplotlib import cm


def triangulate_quad_mesh(ind, element_type):
    triangles = []
    outline = []
    for e in range(ind.shape[0]):
        if element_type == fem_utils.TYPE_HEX8:
            quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (3, 2, 6, 7), (3, 0, 4, 7), (1, 2, 6, 5)]
            lines = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        elif element_type == fem_utils.TYPE_HEX27:
            #only defining quads by outer nodes
            quads = [(0, 2, 8, 6), (18, 20, 26, 24), (6, 0, 18, 24), (2, 8, 26, 20), (0, 2, 20, 18), (8, 6, 24, 26)]
            lines = [(0, 2), (2, 8), (8, 6), (6, 0), (18, 20), (20, 26), (26, 24), (24, 18), (0, 18), (2, 20), (8, 26),
                     (6, 24)]
        elif element_type == fem_utils.TYPE_HEX64:
            #only defining quads by outer nodes
            quads = [(0, 3, 15, 12), (48, 51, 63, 60), (12, 0, 48, 60), (3, 15, 63, 51), (0, 3, 51, 48),
                     (15, 12, 60, 63)]
            lines = [(0, 3), (3, 15), (15, 12), (12, 0), (48, 51), (51, 63), (63, 60), (60, 48), (0, 48), (3, 51),
                     (15, 63), (12, 60)]
        else:
            assert False
        ind_e = ind[e, :]
        for quad in quads:
            triangles.append([ind_e[quad[0]], ind_e[quad[1]], ind_e[quad[2]]])
            triangles.append([ind_e[quad[0]], ind_e[quad[2]], ind_e[quad[3]]])
        for line in lines:
            outline.append([ind_e[line[0]], ind_e[line[1]]])

    return np.array(triangles, dtype=int), np.array(outline, dtype=int)


from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt


def plot_3d_mesh(nodes, r, triangles, outlines, scale=1, R=np.zeros(0), show_nodes=True, element_type=-1):
    nN = nodes.shape[0]
    assert len(r) == 3 * nN
    r = r.reshape(-1, 3)
    nodes_defo = nodes + scale * r
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    disp_mag = np.linalg.norm(r, axis=1)

    # Normalize the displacement values (w) to range [0, 1] for the colormap
    norm = Normalize(vmin=np.min(disp_mag), vmax=np.max(disp_mag))
    cmap = cm.viridis  # Choose a colormap

    # Create faces for the triangles and assign colors based on displacement
    faces = [[nodes_defo[triangle[i]] for i in range(3)] for triangle in triangles]
    colors = [cmap(norm(np.mean(disp_mag[triangle])))
              for triangle in triangles]  # Color based on avg displacement per triangle

    # Add the face collection with the assigned colors
    collection = Poly3DCollection(faces, facecolors=colors, alpha=1.0)
    ax.add_collection3d(collection)

    if show_nodes:
        # Scatter plot for nodes
        ax.scatter(nodes_defo[:, 0], nodes_defo[:, 1], nodes_defo[:, 2], c='r', s=0.5, zorder=-1)  # plots the nodes
        for i in range(nodes_defo.shape[0]):
            ax.text(nodes_defo[i, 0], nodes_defo[i, 1], nodes_defo[i, 2], str(i), color="blue", fontsize=8, zorder=10)

    # Plot outlines (edges of the mesh)
    x_outlines = np.zeros(2)
    y_outlines = np.zeros(2)
    z_outlines = np.zeros(2)
    for line in outlines:
        assert len(line) == 2
        for i in range(2):
            x_outlines[i] = nodes_defo[line[i]][0]
            y_outlines[i] = nodes_defo[line[i]][1]
            z_outlines[i] = nodes_defo[line[i]][2]
        ax.plot(np.array(x_outlines), np.array(y_outlines), np.array(z_outlines), color="k")

    # Adjust axis limits
    x_min = np.min(nodes[:, 0])
    x_max = np.max(nodes[:, 0])
    y_min = np.min(nodes[:, 1])
    y_max = np.max(nodes[:, 1])
    z_min = np.min(nodes[:, 2])
    z_max = np.max(nodes[:, 2])
    DOMAIN_SCALE = 1.3
    L_domain = DOMAIN_SCALE * max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_avg = (x_min + x_max) / 2
    y_avg = (y_min + y_max) / 2
    z_avg = (z_min + z_max) / 2
    ax.set_xlim(x_avg - L_domain / 2, x_avg + L_domain / 2)
    ax.set_ylim(y_avg - L_domain / 2, y_avg + L_domain / 2)
    ax.set_zlim(z_avg - L_domain / 2, z_avg + L_domain / 2)

    # Plot applied forces
    if len(R) > 0:
        assert len(R) == 3 * nN
        max_val = np.max(np.abs(R))
        arrow_scale_ratio = (L_domain / 20) / max_val

        for i in range(nN):
            Ri = R[3 * i:3 * i + 3]
            if np.linalg.norm(Ri) > 0:
                start = nodes_defo[i]
                dir = arrow_scale_ratio * Ri
                ax.quiver(start[0], start[1], start[2], dir[0], dir[1], dir[2], color='r')

    # Add colorbar to show displacement
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10)

    NUM_TICKS = 8
    tickvals = np.linspace(np.min(disp_mag), np.max(disp_mag), NUM_TICKS)
    ticklabels = []
    for tickval in tickvals:
        ticklabels.append("{:.5f}".format(tickval))

    cbar.set_ticks(tickvals)
    cbar.set_ticklabels(ticklabels)
    cbar.set_label("Displacement magnitude")

    # Axis labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    title = ""
    if element_type in fem_utils.element_type_to_str:
        title += fem_utils.element_type_to_str[element_type] + ", "
    title += "Displacement scale = " + str(scale)
    plt.title(title)
