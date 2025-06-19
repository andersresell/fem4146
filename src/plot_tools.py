import numpy as np
import matplotlib.pyplot as plt
import signal
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from element_utils import *
from matplotlib import cm
from mesh import Mesh
from utils import Config
from solver_data import *

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, TextBox
from matplotlib.axes import Axes


def triangulate_quad_mesh(elements, element_type):
    triangles_glob = []
    outline_glob = []
    for e in range(elements.shape[0]):
        if element_type == ELEMENT_TYPE_Q4 or element_type == ELEMENT_TYPE_Q4R:
            triangles = []
            quads = [(0, 1, 2, 3)]
            lines = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif element_type == ELEMENT_TYPE_Q8 or element_type == ELEMENT_TYPE_Q8R:
            triangles = [(0, 4, 7), (1, 5, 4), (2, 6, 5), (3, 7, 6)]
            quads = [(4, 5, 6, 7)]
            lines = [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0)]
        elif element_type == ELEMENT_TYPE_Q9 or element_type == ELEMENT_TYPE_Q9R:
            triangles = []
            quads = [(0, 4, 8, 7), (4, 1, 5, 8), (8, 5, 2, 6), (7, 8, 6, 3)]
            lines = [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0)]
            # quads = [(0, 4, 8, 7), (4, 1, 5, 8), (8, 5, 2, 6), (7, 8, 6, 3)]
            # lines = [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0)]
        elif element_type == ELEMENT_TYPE_Q16:
            triangles = []
            quads = [(0, 4, 12, 11), (4, 5, 13, 12), (5, 1, 6, 13), (11, 12, 15, 10), (12, 13, 14, 15), (13, 6, 7, 14),
                     (10, 15, 9, 3), (15, 14, 8, 9), (14, 7, 2, 8)]
            lines = [(0, 4), (4, 5), (5, 1), (1, 6), (6, 7), (7, 2), (2, 8), (8, 9), (9, 3), (3, 10), (10, 11), (11, 0)]
        else:
            assert False
        element = elements[e, :]
        for tri in triangles:
            triangles_glob.append([element[tri[0]], element[tri[1]], element[tri[2]]])
        for quad in quads:
            triangles_glob.append([element[quad[0]], element[quad[1]], element[quad[2]]])
            triangles_glob.append([element[quad[0]], element[quad[2]], element[quad[3]]])
        for line in lines:
            outline_glob.append([element[line[0]], element[line[1]]])

    return np.array(triangles_glob, dtype=int), np.array(outline_glob, dtype=int)


def get_mesh_bounds(nodes):
    x_min = np.min(nodes[:, 0])
    x_max = np.max(nodes[:, 0])
    y_min = np.min(nodes[:, 1])
    y_max = np.max(nodes[:, 1])
    return x_min, x_max, y_min, y_max


class Plot:

    def __init__(self, config: Config, mesh: Mesh, solver_data: SolverData):
        self.fig, self.ax = plt.subplots(figsize=(12, 10), dpi=100)
        self.cbar = None

        plt.subplots_adjust(left=0.25, bottom=0.25)

        slider_disp_scale = TextBox(plt.axes([0.25, 0.1, 0.65, 0.03]), 'Disp scale', initial=config.disp_scaling)
        slider_node_scale = Slider(plt.axes([0.25, 0.05, 0.65, 0.03]), 'Node size', 0, 10, valinit=config.node_scale)
        radio_contour_type = RadioButtons(plt.axes([0.05, 0.7, 0.15, 0.05]), ['Displacement', 'Stress'])
        check_show_node_labels = CheckButtons(plt.axes([0.05, 0.5, 0.15, 0.05]), ['Show node labels'], [True])

        def update(val):
            try:
                config.disp_scaling = float(slider_disp_scale.text)
            except ValueError:
                print("Invalid input for displacement scaling. Using default value.")
                config.disp_scaling = 1.0

            config.contour_type = radio_contour_type.value_selected
            config.node_scale = slider_node_scale.val
            config.show_node_labels = check_show_node_labels.get_status()[0]

            self.ax.cla()
            self.plot_2d(config, mesh, solver_data)
            self.fig.canvas.draw_idle()

        slider_disp_scale.on_submit(update)
        slider_node_scale.on_changed(update)
        radio_contour_type.on_clicked(update)
        check_show_node_labels.on_clicked(update)

        update(None)  # initial draw
        plt.show()

    def plot_2d(self, config: Config, mesh: Mesh, solver_data: SolverData):

        triangles, outlines = triangulate_quad_mesh(mesh.elements, config.element_type)

        nN = mesh.get_nN()
        nE = mesh.get_nE()
        # elements = mesh.elements
        nodes = mesh.nodes

        if config.problem_type == PROBLEM_TYPE_PLANE_STRESS:
            u, v = unpack_solution(config, mesh, solver_data.r)
            disp_mag = np.sqrt(u**2 + v**2)
        else:
            assert False  #FIX
            assert config.problem_type == PROBLEM_TYPE_PLATE
            w, thetax, thetay = solver_data.unpack_solution(config, mesh, solver_data)
            disp_mag = w
        #  u[:] = np.sin(nodes[:, 0])

        x = nodes[:, 0] + u * config.disp_scaling
        y = nodes[:, 1] + v * config.disp_scaling

        #====================================================================
        # Plot mesh body
        #====================================================================
        tris = tri.Triangulation(x, y, triangles)
        tpc = self.ax.tripcolor(tris, disp_mag, shading='gouraud')  #, cmap='viridis')

        fig = self.ax.get_figure()

        if self.cbar is None:
            self.cbar = fig.colorbar(tpc, ax=self.ax)

        self.cbar.update_normal(tpc)
        self.cbar.set_label(r"$u^{\mathrm{mag}} = \sqrt{u^2 + v^2}$")

        #====================================================================
        # Plot outlines
        #====================================================================
        segments = [[[x[i], y[i]], [x[j], y[j]]] for i, j in outlines]
        line_collection = LineCollection(segments, colors='black', linewidths=0.5)
        self.ax.add_collection(line_collection)

        #====================================================================
        # Plot nodes
        #====================================================================
        SCALE_NODES = 10

        x_min, x_max, y_min, y_max = get_mesh_bounds(nodes)
        L_domain = max(x_max - x_min, y_max - y_min)
        domain_scale = L_domain / np.sqrt(nN)
        self.ax.scatter(x, y, color="black", s=config.node_scale * domain_scale * SCALE_NODES, zorder=10)

        SCALE_NODELABELS = 50  #should make it independent of figure size
        if config.show_node_labels:
            for i in range(nN):
                self.ax.text(x[i],
                             y[i],
                             str(i),
                             fontsize=config.node_scale * domain_scale * SCALE_NODELABELS,
                             color="black",
                             ha="left",
                             va="bottom",
                             zorder=20)

        #====================================================================
        # Plot applied forces
        #====================================================================
        if config.problem_type == PROBLEM_TYPE_PLANE_STRESS and config.plot_external_forces:
            Rx, Ry = unpack_solution(config, mesh, solver_data.R)
            self.plot_arrows(x, y, Rx, Ry, "External Forces", "red", domain_scale)

        self.ax.set_aspect('equal')
        self.ax.set_xlabel(r"$x$")
        self.ax.set_ylabel(r"$y$")

    def plot_arrows(self, x, y, field_x, field_y, label, color, domain_scale):
        ARROW_SCALE = 0.0000001
        ARROW_WIDTH = 0.01
        scale = ARROW_SCALE * domain_scale
        self.ax.quiver(x, y, field_x * scale, field_y * scale, color=color,
                       width=ARROW_WIDTH * domain_scale)  #, scale=ARROW_SCALE * domain_scale,wi)
        import matplotlib.lines as mlines
        # Create a proxy handle for the legend
        arrow_proxy = mlines.Line2D([], [],
                                    color=color,
                                    marker=r'$\rightarrow$',
                                    linestyle='None',
                                    markersize=10,
                                    label=label)
        self.ax.legend(handles=[arrow_proxy])
        self.ax.axis('equal')
