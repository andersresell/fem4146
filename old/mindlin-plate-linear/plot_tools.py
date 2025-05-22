import numpy as np
import matplotlib.pyplot as plt
import signal
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import element_utils
from matplotlib import cm


def triangulate_quad_mesh(ind, element_type):
    triangles = []
    outline = []
    for e in range(ind.shape[0]):
        if element_type == element_utils.TYPE_Q4 or element_type == element_utils.TYPE_Q4R:
            quads = [(0, 1, 2, 3)]
            lines = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif element_type == element_utils.TYPE_Q9 or element_type == element_utils.TYPE_Q9R:
            quads = [(0, 4, 8, 7), (4, 1, 5, 8), (8, 5, 2, 6), (7, 8, 6, 3)]
            lines = [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0)]
        elif element_type == element_utils.TYPE_Q16:
            quads = [(0, 4, 12, 11), (4, 5, 13, 12), (5, 1, 6, 13), (11, 12, 15, 10), (12, 13, 14, 15), (13, 6, 7, 14),
                     (10, 15, 9, 3), (15, 14, 8, 9), (14, 7, 2, 8)]
            lines = [(0, 4), (4, 5), (5, 1), (1, 6), (6, 7), (7, 2), (2, 8), (8, 9), (9, 3), (3, 10), (10, 11), (11, 0)]
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


def plot_3d_mesh(nodes, w, triangles, outlines, scale=1, Rw=np.zeros(0), show_node_labels=True, element_type=-1):
    assert nodes.shape[0] == len(w)
    nodes_3d = np.c_[nodes, w * scale]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normalize the displacement values (w) to range [0, 1] for the colormap
    norm = Normalize(vmin=min(w), vmax=max(w))
    cmap = cm.viridis  # Choose a colormap

    # Create faces for the triangles and assign colors based on displacement
    faces = [[nodes_3d[triangle[i]] for i in range(3)] for triangle in triangles]
    colors = [cmap(norm(np.mean(w[triangle])))
              for triangle in triangles]  # Color based on avg displacement per triangle

    # Add the face collection with the assigned colors
    collection = Poly3DCollection(faces, facecolors=colors, alpha=0.5)
    ax.add_collection3d(collection)

    # Scatter plot for nodes
    ax.scatter(nodes_3d[:, 0], nodes_3d[:, 1], nodes_3d[:, 2], c='r', s=0.5, zorder=10)  # plots the nodes

    if show_node_labels:
        for i in range(nodes_3d.shape[0]):
            ax.text(nodes_3d[i, 0], nodes_3d[i, 1], nodes_3d[i, 2], str(i), color="blue", fontsize=8)

    # Plot outlines (edges of the mesh)
    x_outlines = np.zeros(2)
    y_outlines = np.zeros(2)
    z_outlines = np.zeros(2)
    for line in outlines:
        assert len(line) == 2
        for i in range(2):
            x_outlines[i] = nodes_3d[line[i]][0]
            y_outlines[i] = nodes_3d[line[i]][1]
            z_outlines[i] = nodes_3d[line[i]][2]
        ax.plot(np.array(x_outlines), np.array(y_outlines), np.array(z_outlines), color="k")

    # Adjust axis limits
    x_min = np.min(nodes[:, 0])
    x_max = np.max(nodes[:, 0])
    y_min = np.min(nodes[:, 1])
    y_max = np.max(nodes[:, 1])
    DOMAIN_SCALE = 1.3
    L_domain = DOMAIN_SCALE * max(x_max - x_min, y_max - y_min)
    x_avg = (x_min + x_max) / 2
    y_avg = (y_min + y_max) / 2
    ax.set_xlim(x_avg - L_domain / 2, x_avg + L_domain / 2)
    ax.set_ylim(y_avg - L_domain / 2, y_avg + L_domain / 2)
    ax.set_zlim(-L_domain / 2, L_domain / 2)

    # Plot applied forces
    if len(Rw) > 0:
        max_val = np.max(np.abs(Rw))
        arrow_scale_ratio = (L_domain / 20) / max_val

        for i in range(len(Rw)):
            if abs(Rw[i]) > 0:
                start = nodes_3d[i]
                dir = arrow_scale_ratio * np.array([0, 0, Rw[i]])
                ax.quiver(start[0], start[1], start[2], dir[0], dir[1], dir[2], color='r')

    # Add colorbar to show displacement
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10)

    NUM_TICKS = 8
    tickvals = np.linspace(np.min(w), np.max(w), NUM_TICKS)
    ticklabels = []
    for tickval in tickvals:
        ticklabels.append("{:.5f}".format(tickval))

    cbar.set_ticks(tickvals)
    cbar.set_ticklabels(ticklabels)
    cbar.set_label("Vertical Displacement")

    # Axis labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    title = ""
    if element_type in element_utils.element_type_to_str:
        title += element_utils.element_type_to_str[element_type] + ", "
    title += "Displacement scale = " + str(scale)
    plt.title(title)
