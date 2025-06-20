import numpy as np
import matplotlib.pyplot as plt
import signal
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from element_utils import *
from matplotlib import cm
from mesh import Mesh, QuadElementTraits
from utils import Config
from solver_data import *
import element_stiffness

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, TextBox
from matplotlib.axes import Axes


def triangulate_quad_mesh(elements, element_type):
    triangles_glob = []
    outline_glob = []
    nNl = element_type_to_nNl[element_type]
    nE = elements.shape[0]
    for e in range(nE):
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
        i_last = e * nNl
        #Triangles vertices are local to the element
        for tri in triangles:
            triangles_glob.append([i_last + tri[0], i_last + tri[1], i_last + tri[2]])
        for quad in quads:
            triangles_glob.append([i_last + quad[0], i_last + quad[1], i_last + quad[2]])
            triangles_glob.append([i_last + quad[0], i_last + quad[2], i_last + quad[3]])
        #Line vertices are global
        for line in lines:
            outline_glob.append([element[line[0]], element[line[1]]])
        # for tri in triangles:
        #     triangles_glob.append([element[tri[0]], element[tri[1]], element[tri[2]]])
        # for quad in quads:
        #     triangles_glob.append([element[quad[0]], element[quad[1]], element[quad[2]]])
        #     triangles_glob.append([element[quad[0]], element[quad[2]], element[quad[3]]])
        # for line in lines:
        #     outline_glob.append([element[line[0]], element[line[1]]])

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

        slider_disp_scale = TextBox(plt.axes([0.25, 0.1, 0.65, 0.03]), 'disp scale', initial=config.disp_scaling)
        slider_node_scale = Slider(plt.axes([0.25, 0.05, 0.65, 0.03]), 'node size', 0, 10, valinit=config.node_scale)
        radio_contour_type = RadioButtons(plt.axes([0.05, 0.7, 0.15, 0.05]), ['disp', 'stress'])
        radio_disp_component = RadioButtons(plt.axes([0.05, 0.6, 0.15, 0.05]), ['mag', 'x', 'y'])
        radio_stress_component = RadioButtons(plt.axes([0.05, 0.5, 0.15, 0.065]), ['mises', 'xx', 'yy', 'xy'])
        check_show_node_labels = CheckButtons(plt.axes([0.05, 0.4, 0.15, 0.05]), ['show node labels'],
                                              [config.show_node_labels])

        def update(val):
            try:
                config.disp_scaling = float(slider_disp_scale.text)
            except ValueError:
                print("Invalid input for displacement scaling. Using default value.")
                config.disp_scaling = 1.0

            config.contour_type = radio_contour_type.value_selected
            config.disp_component = radio_disp_component.value_selected
            config.stress_component = radio_stress_component.value_selected
            config.node_scale = slider_node_scale.val
            config.show_node_labels = check_show_node_labels.get_status()[0]

            self.ax.cla()
            self.plot_2d(config, mesh, solver_data)
            self.fig.canvas.draw_idle()

        slider_disp_scale.on_submit(update)
        slider_node_scale.on_changed(update)
        radio_contour_type.on_clicked(update)
        radio_disp_component.on_clicked(update)
        radio_stress_component.on_clicked(update)
        check_show_node_labels.on_clicked(update)

        update(None)  # initial draw
        plt.show()

    def plot_2d(self, config: Config, mesh: Mesh, solver_data: SolverData):

        triangles, outlines = triangulate_quad_mesh(mesh.elements, config.element_type)

        nN = mesh.get_nN()
        nNl = element_type_to_nNl[config.element_type]
        nE = mesh.get_nE()
        # elements = mesh.elements
        nodes = mesh.nodes
        u, v = unpack_solution(config, mesh, solver_data.r)

        #Global coordinates
        x = nodes[:, 0] + u * config.disp_scaling
        y = nodes[:, 1] + v * config.disp_scaling
        #Coorinades local to the element
        xE = np.zeros(nE * nNl)
        yE = np.zeros(nE * nNl)
        uE = np.zeros(nE * nNl)
        vE = np.zeros(nE * nNl)
        for e in range(nE):
            for il in range(nNl):
                xE[e * nNl + il] = x[mesh.elements[e, il]]
                yE[e * nNl + il] = y[mesh.elements[e, il]]
                uE[e * nNl + il] = u[mesh.elements[e, il]]
                vE[e * nNl + il] = v[mesh.elements[e, il]]

        #====================================================================
        # Calculate vertex scalar field (determines the color of the mesh)
        #====================================================================
        if config.problem_type == PROBLEM_TYPE_PLANE_STRESS:
            if config.contour_type == "disp":
                if config.disp_component == "mag":
                    vertex_scalar_E = np.sqrt(uE**2 + vE**2)
                elif config.disp_component == "x":
                    vertex_scalar_E = uE
                elif config.disp_component == "y":
                    vertex_scalar_E = vE
                else:
                    print(f"Unknown displacement component form plotting: {config.disp_component}")
                    exit(1)
            else:
                assert config.contour_type == "stress"
                vertex_scalar_E = self.calculate_vertex_scalar_from_stress_plane_stress_problem(
                    config, mesh, solver_data)
                # vertex_scalar_E = np.zeros(nE * nNl)
                # vertex_scalar_E[:] = 1

        else:
            assert False  #FIXME
            assert config.problem_type == PROBLEM_TYPE_PLATE
            w, thetax, thetay = solver_data.unpack_solution(config, mesh, solver_data)
            disp_mag = w

        #====================================================================
        # Plot mesh body
        #====================================================================

        tris = tri.Triangulation(xE, yE, triangles)
        tpc = self.ax.tripcolor(tris, vertex_scalar_E, shading='gouraud')  #, cmap='viridis')

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

    def calculate_vertex_scalar_from_stress_plane_stress_problem(self, config: Config, mesh: Mesh,
                                                                 solver_data: SolverData):

        nNl = element_type_to_nNl[config.element_type]
        nE = mesh.get_nE()
        nodes = mesh.nodes
        element_type = config.element_type
        vertex_scalar_E = np.zeros(nE * nNl)
        E = config.E
        nu = config.nu
        element_traits = QuadElementTraits(element_type)

        elements = mesh.elements
        r = solver_data.r
        u, v = unpack_solution(config, mesh, r)

        arr_sigma = element_stiffness.calc_stress_plane_stress(config, mesh, nodes, u, v, element_traits.xi_eta)

        for e in range(nE):
            for il in range(nNl):
                # coord_x = mesh.nodes[elements[e, :], 0]
                # coord_y = mesh.nodes[elements[e, :], 1]

                # #fmt: off
                # D = E / (1 - nu**2) * np.array([[1,     nu,     0],
                #                                 [nu,    1,      0],
                #                                 [0,     0,  (1 - nu) / 2]])
                # #fmt: on

                #====================================================================
                # Will first compute the stress at the Gauss points
                #====================================================================
                # nGauss = shape_functions.element_type_to_nGauss_1D[element_type]
                # arr_xi = shape_functions.get_arr_xi(nGauss)
                # arr_w = shape_functions.get_arr_w(nGauss)

                # arr_sigma = np.zeros(nGauss**2, 3 )
                # r_e = np.zeros(nNl * 2)
                # r_e[0::2] = u[elements[e, :]]
                # r_e[1::2] = v[elements[e, :]]

                # for i in range(nGauss):
                #     for j in range(nGauss):
                #         xi = arr_xi[i]
                #         eta = arr_xi[j]
                #         N = shape_functions.calc_N(xi, eta, element_type)
                #         dNdx, dNdy = shape_functions.calc_dNdx_dNdy(xi, eta, coord_x, coord_y, element_type)

                #         B = np.zeros((3, 2 * nNl))
                #         for k in range(nNl):
                #             #fmt: off
                #             B[:, 2 * k:2 * k + 2] = np.array([[dNdx[k],     0],
                #                                               [0,       dNdy[k]],
                #                                               [dNdy[k], dNdx[k]]])
                #             #fmt: on

                #         arr_sigma[i*nGauss+j] =  D @ B @ r_e

                #====================================================================
                # Will now extrapolate the stress to the local vertices
                # (used for rendering)
                #====================================================================
                # for il in range(nNl):
                #     xi,eta  = QuadElementTraits.xi_eta[il]
                #     N = shape_functions.ca
                # vertex_scalar_E[e*nNl+il] =

                #====================================================================
                # Will first try to calculate the stress consistently. Might
                # add more methods later.
                #====================================================================
                for il in range(nNl):
                    assert arr_sigma.shape == (nE * nNl, 3)
                    sigma = arr_sigma[e * nNl + il]
                    sxx = sigma[0]
                    syy = sigma[1]
                    sxy = sigma[2]
                    if config.stress_component == "mises":
                        val = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
                    elif config.stress_component == "xx":
                        val = sxx
                    elif config.stress_component == "yy":
                        val = syy
                    else:
                        assert config.stress_component == "xy"
                        val = sxy
                    vertex_scalar_E[e * nNl + il] = val
        return vertex_scalar_E
