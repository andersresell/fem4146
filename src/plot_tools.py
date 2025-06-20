import numpy as np
import matplotlib.pyplot as plt
import signal
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.linalg
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


def get_L_domain(nodes):
    x_min, x_max, y_min, y_max = get_mesh_bounds(nodes)
    return max(x_max - x_min, y_max - y_min)


class Plot:

    def __init__(self, config: Config, mesh: Mesh, solver_data: SolverData):
        self.fig, self.ax = plt.subplots(figsize=(12, 10), dpi=100)
        self.cbar = None

        plt.subplots_adjust(left=0.3, bottom=0.25)

        VSPACE_BETWEEN_WIDGETS = 0.04
        self.curr_vspace_left = 0.9

        def get_vspace_left(height):
            vspace = self.curr_vspace_left - height / 2
            assert self.curr_vspace_left > 0.1
            self.curr_vspace_left -= (height + VSPACE_BETWEEN_WIDGETS)
            return vspace

        self.curr_vspace_bottom_index = 0
        vspaces_bottom = np.linspace(0.15, 0.05, 4)

        def get_new_vspace_bottom():
            assert self.curr_vspace_bottom_index < len(vspaces_bottom)
            vspace = vspaces_bottom[self.curr_vspace_bottom_index]
            self.curr_vspace_bottom_index += 1
            return vspace

        radio_contour_type = RadioButtons(plt.axes([0.05, get_vspace_left(0.05), 0.15, 0.05]), ['disp', 'stress'])
        radio_contour_type.ax.set_title('Contour type:')
        radio_disp_component = RadioButtons(plt.axes([0.05, get_vspace_left(0.05), 0.15, 0.05]), ['mag', 'u', 'v'])
        radio_disp_component.ax.set_title('Disp component:')
        radio_stress_component = RadioButtons(plt.axes([0.05, get_vspace_left(0.065), 0.15, 0.065]),
                                              ['mises', 'xx', 'yy', 'xy'])
        radio_stress_component.ax.set_title('Stress component:')
        radio_arrows = RadioButtons(plt.axes([0.05, get_vspace_left(0.065), 0.15, 0.065]),
                                    ['none', 'disp', 'external forces', 'reaction forces'])
        radio_arrows.ax.set_title('Arrow type:')

        check_buttons = CheckButtons(
            plt.axes([0.05, get_vspace_left(0.05), 0.15,
                      0.05]), ['show node labels', 'specify contour limits', 'show mesh', 'show boundary conditions'],
            [config.show_node_labels, config.specify_contour_limits, config.show_mesh, config.show_bcs])
        check_buttons.ax.set_title('Options:')

        box_contour_min = TextBox(plt.axes([0.1, get_vspace_left(0.03), 0.04, 0.03]),
                                  'contour lim min',
                                  initial=config.contour_min)
        box_contour_max = TextBox(plt.axes([0.1, get_vspace_left(0.03), 0.04, 0.03]),
                                  'contour lim max',
                                  initial=config.contour_max)
        box_disp_scale = TextBox(plt.axes([0.1, get_vspace_left(0.03), 0.04, 0.03]),
                                 'disp scale',
                                 initial=config.disp_scaling)

        slider_node_scale = Slider(plt.axes([0.25, get_new_vspace_bottom(), 0.50, 0.03]),
                                   'node size',
                                   0,
                                   10,
                                   valinit=config.node_scale)
        slider_arrow_scale = Slider(plt.axes([0.25, get_new_vspace_bottom(), 0.50, 0.03]),
                                    'arrow size',
                                    0,
                                    10,
                                    valinit=config.arrow_scale)

        def update(val):
            try:
                config.disp_scaling = float(box_disp_scale.text)
            except ValueError:
                print("Invalid input for displacement scaling. Using default value.")
                config.disp_scaling = 1.0

            config.contour_type = radio_contour_type.value_selected
            config.disp_component = radio_disp_component.value_selected
            config.stress_component = radio_stress_component.value_selected
            config.arrow_type = radio_arrows.value_selected
            config.node_scale = slider_node_scale.val
            config.arrow_scale = slider_arrow_scale.val
            config.show_node_labels = check_buttons.get_status()[0]
            config.specify_contour_limits = check_buttons.get_status()[1]
            config.show_mesh = check_buttons.get_status()[2]
            config.show_bcs = check_buttons.get_status()[3]

            try:
                config.contour_min = float(box_contour_min.text)
            except ValueError:
                print("Invalid input for contour min. Using default value.")
                config.contour_min = 0.0

            try:
                config.contour_max = float(box_contour_max.text)
            except ValueError:
                print("Invalid input for contour max. Using default value.")
                config.contour_max = 1.0

            self.ax.cla()
            self.plot_2d(config, mesh, solver_data)
            self.fig.canvas.draw_idle()

        box_disp_scale.on_submit(update)
        slider_node_scale.on_changed(update)
        slider_arrow_scale.on_changed(update)
        radio_contour_type.on_clicked(update)
        radio_disp_component.on_clicked(update)
        radio_stress_component.on_clicked(update)
        radio_arrows.on_clicked(update)
        check_buttons.on_clicked(update)
        box_contour_min.on_submit(update)
        box_contour_max.on_submit(update)

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
                elif config.disp_component == "u":
                    vertex_scalar_E = uE
                elif config.disp_component == "v":
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
        if config.specify_contour_limits:
            vmin = config.contour_min
            vmax = config.contour_max
        else:
            vmin = np.min(vertex_scalar_E)
            vmax = np.max(vertex_scalar_E)

        tpc = self.ax.tripcolor(tris, vertex_scalar_E, shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)

        fig = self.ax.get_figure()

        if self.cbar is None:
            self.cbar = fig.colorbar(tpc, ax=self.ax)
        self.cbar.update_normal(tpc)
        vmin, vmax = tpc.get_clim()

        # Force uniformly spaced colorbar ticks
        NUM_TICKS = 10
        ticks = np.linspace(vmin, vmax, NUM_TICKS)
        self.cbar.set_ticks(ticks)

        assert config.problem_type == PROBLEM_TYPE_PLANE_STRESS  #FIX IF USING PLATE PROBLEM
        if config.contour_type == "disp":
            if config.disp_component == "mag":
                self.cbar.set_label(r"$u^{\mathrm{mag}} = \sqrt{u^2 + v^2}$")
            elif config.disp_component == "x":
                self.cbar.set_label(r"$u$")
            elif config.disp_component == "y":
                self.cbar.set_label(r"$v$")
            else:
                print(f"Error: Unknown displacement component for plotting: {config.disp_component}")
        elif config.contour_type == "stress":
            if config.stress_component == "mises":
                self.cbar.set_label(r"$\sigma^{\mathrm{VM}}$")
            elif config.stress_component == "xx":
                self.cbar.set_label(r"$\sigma_{\mathrm{xx}}$")
            elif config.stress_component == "yy":
                self.cbar.set_label(r"$\sigma_{\mathrm{yy}}$")
            elif config.stress_component == "xy":
                self.cbar.set_label(r"$\sigma_{\mathrm{xy}}$")
            else:
                print(f"Error: Unknown stress component for plotting: {config.stress_component}")
        else:
            print(f"Error: Unknown contour type for plotting: {config.contour_type}")

        #====================================================================
        # Plot outlines
        #====================================================================
        if config.show_mesh:
            segments = [[[x[i], y[i]], [x[j], y[j]]] for i, j in outlines]
            line_collection = LineCollection(segments, colors='black', linewidths=0.5)
            self.ax.add_collection(line_collection)

        #====================================================================
        # Plot nodes
        #====================================================================
        SCALE_NODES = 10

        SCALE_DOMAIN = 20
        domain_scale = SCALE_DOMAIN / np.sqrt(nN)
        if config.show_mesh:
            self.ax.scatter(x, y, color="black", s=config.node_scale * domain_scale * SCALE_NODES, zorder=10)

        SCALE_NODELABELS = 3
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
        # Plot arrow fields
        #====================================================================
        if config.arrow_type != "none":
            if config.problem_type == PROBLEM_TYPE_PLANE_STRESS:
                if config.arrow_type == "disp":
                    self.plot_arrows(x, y, u, v, "disp", "blue", domain_scale, config.arrow_scale)
                elif config.arrow_type == "external forces":
                    Rx_ext, Ry_ext = unpack_solution(config, mesh, solver_data.R_ext)
                    self.plot_arrows(x, y, Rx_ext, Ry_ext, "external forces", "red", domain_scale, config.arrow_scale)
                elif config.arrow_type == "reaction forces":
                    R_rea = solver_data.R_int - solver_data.R_ext
                    Rx_rea, Ry_rea = unpack_solution(config, mesh, R_rea)
                    self.plot_arrows(x, y, Rx_rea, Ry_rea, "reaction forces", "green", domain_scale, config.arrow_scale)
                else:
                    print(f"Unknown arrow type for plotting: {config.arrow_type}")
                    exit(1)
            else:
                assert False  #FIXME

        #====================================================================
        # Plot boundary conditions
        #====================================================================
        if config.show_bcs:
            assert config.problem_type == PROBLEM_TYPE_PLANE_STRESS  # FIXME
            L_domain = get_L_domain(mesh.nodes)
            s = L_domain * 0.02 * domain_scale
            segments = []
            for bc in config.bcs:
                nodeIDs_constrained = mesh.node_sets[bc.node_set_name]
                for I in nodeIDs_constrained:
                    x0, y0 = x[I], y[I]
                    if bc.dof == DOF_U:
                        segments.append([[x0 - s, y0], [x0 + s, y0]])
                    elif bc.dof == DOF_V:
                        segments.append([[x0, y0 - s], [x0, y0 + s]])
                    else:
                        raise ValueError("Unsupported DOF in BC")
            bc_lines = LineCollection(segments, colors='brown', linewidths=5 * domain_scale)
            self.ax.add_collection(bc_lines)

        self.ax.set_aspect('equal')
        self.ax.set_xlabel(r"$x$")
        self.ax.set_ylabel(r"$y$")

    def plot_arrows(self, x, y, field_x, field_y, label, color, domain_scale, user_scale):
        ARROW_LENGTH_SCALE = 0.05
        ARROW_WIDTH_SCALE = 0.001
        max_vector_norm = np.max(np.sqrt(field_x**2 + field_y**2))
        #handle normalization in a non singular way
        field_x_normalized = field_x / (max_vector_norm + 1e-10)  #ensure safe division
        field_y_normalized = field_y / (max_vector_norm + 1e-10)  #ensure safe division
        self.ax.quiver(
            x,
            y,
            ARROW_LENGTH_SCALE * user_scale * field_x_normalized,
            ARROW_LENGTH_SCALE * user_scale * field_y_normalized,
            color=color,
            scale=1,  #this scale thing is weird, increasing it makes arrows smaller
            width=ARROW_WIDTH_SCALE * domain_scale)
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
