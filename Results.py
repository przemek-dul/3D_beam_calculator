import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import plotly.graph_objects as go
import plotly
import mplcursors
import sys, os
import plotly.offline
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication
import matplotlib as mpl


class Results:
    def __init__(self, model):
        self.model = model
        self.density = 10
        self.node_x = np.array([])
        self.node_y = np.array([])
        self.node_z = np.array([])

        self.x = np.array([])
        self.y = np.array([])
        self.z = np.array([])

        self.ux = np.array([])
        self.uy = np.array([])
        self.uz = np.array([])
        self.fx = np.array([])
        self.fy = np.array([])
        self.fz = np.array([])

        self.calculate_density()
        self.calculate_nodes_xyz()
        self.calculate_displacements(self.density)

        self.to_scale = None
        self.scale = 1
        self.basic_dim = None
        self.calculate_to_scale()

        self.plotly_figs = []
        self.mpl_figs = []

    def calculate_density(self):
        self.density = int(-0.28 * len(self.model.elements)+37.82)
        if self.density < 4:
            self.density = 4

    def calculate_to_scale(self):

        s1 = max(max(self.node_x), max(self.node_y), max(self.node_z))
        s2 = min(min(self.node_x), min(self.node_y), min(self.node_z))

        max_u = max(max(np.abs(self.ux)), max(np.abs(self.uy)), max(np.abs(self.uz)))

        self.scale = (s1 - s2) / (max_u * 12)

        if self.scale < 1:
            self.scale = 1

        self.to_scale = (s1 - s2) / 4
        self.basic_dim = (s1 - s2) / 5

    def calculate_nodes_xyz(self):
        for node in self.model.nodes:
            self.node_x = np.append(self.node_x, node.x)
            self.node_y = np.append(self.node_y, node.y)
            self.node_z = np.append(self.node_z, node.z)

    def calculate_displacements(self, density):

        for element in self.model.elements:
            us, fs, ds = self.shape_func(element, density)

            self.ux = np.append(self.ux, us[0])
            self.uy = np.append(self.uy, us[1])
            self.uz = np.append(self.uz, us[2])
            self.fx = np.append(self.fx, fs[0])
            self.fy = np.append(self.fy, fs[1])
            self.fz = np.append(self.fz, fs[2])

            self.x = np.append(self.x, ds[0])
            self.y = np.append(self.y, ds[1])
            self.z = np.append(self.z, ds[2])

    def shape_func(self, element, num):
        x_v = np.linspace(0, element.L, num)

        u_x = np.array([])
        u_y = np.array([])
        u_z = np.array([])
        f_x = np.array([])
        f_y = np.array([])
        f_z = np.array([])

        c_matrix = element.local_c_matrix

        N9 = (c_matrix[6, 0] - c_matrix[0, 0]) / element.L
        N10 = c_matrix[0, 0]

        for x in x_v:
            N1, N2, N3, N4, N5, N6, N7, N8 = element.shape_func(x, 'uz')
            u_yy = N1 * c_matrix[1, 0] + N2 * c_matrix[5, 0] + N3 * c_matrix[7, 0] + N4 * c_matrix[11, 0]
            f_zz = N5 * c_matrix[1, 0] + N6 * c_matrix[5, 0] + N7 * c_matrix[7, 0] + N8 * c_matrix[11, 0]

            N1, N2, N3, N4, N5, N6, N7, N8 = element.shape_func(x, 'uz')
            u_zz = N1 * c_matrix[2, 0] + N2 * c_matrix[4, 0] + N3 * c_matrix[8, 0] + N4 * c_matrix[10, 0]
            f_yy = N5 * c_matrix[2, 0] + N6 * c_matrix[4, 0] + N7 * c_matrix[8, 0] + N8 * c_matrix[10, 0]

            u_xx = N9 * x + N10

            l_matrix = np.matrix([[u_xx], [u_yy], [u_zz], [0], [f_yy], [f_zz]])
            g_matrix = np.dot(element.t_matrix[0:6, 0:6], l_matrix)

            u_x = np.append(u_x, g_matrix[0, 0])
            u_y = np.append(u_y, g_matrix[1, 0])
            u_z = np.append(u_z, g_matrix[2, 0])
            f_x = np.append(f_x, g_matrix[3, 0])
            f_y = np.append(f_y, g_matrix[4, 0])
            f_z = np.append(f_z, g_matrix[5, 0])

        x = np.linspace(element.node1.x, element.node2.x, num)
        y = np.linspace(element.node1.y, element.node2.y, num)
        z = np.linspace(element.node1.z, element.node2.z, num)

        return [u_x, u_y, u_z], [f_x, f_y, f_z], [x, y, z]

    def basic_plot(self, scale, value, plane):
        fig, ax = plt.subplots(figsize=(8, 6))
        points = None
        horizontal_text = ''
        vertical_text = ''

        if plane == 'xy' or plane == 'yx':
            points = np.array([self.x + scale * self.ux, self.y + scale * self.uy]).T.reshape(-1, 1, 2)
            horizontal_text = 'X axis'
            vertical_text = 'Y axis'
        elif plane == 'yz' or plane == 'zy':
            points = np.array([self.z + scale * self.uz, self.y + scale * self.uy]).T.reshape(-1, 1, 2)
            horizontal_text = 'z axis'
            vertical_text = 'Y axis'
        elif plane == 'xz' or plane == 'zx':
            points = np.array([self.z + scale * self.uz, self.x + scale * self.ux]).T.reshape(-1, 1, 2)
            horizontal_text = 'Z axis'
            vertical_text = 'X axis'

        ax.set_ylabel(vertical_text)
        ax.set_xlabel(horizontal_text)

        cmin = min(value)
        cmax = max(value)

        if cmin != cmax:
            cmap = plt.get_cmap('jet')
        else:
            single_color = plt.get_cmap('jet')(0.5)
            cmap = mpl.colors.ListedColormap([single_color])

        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(cmin, cmax)
        ind = 0
        for i in range(0, len(self.model.elements)):
            lc = LineCollection(segments[self.density * i:self.density * i + (self.density-1)],
                                cmap=cmap, norm=norm, label=str(ind))
            lc.set_array(value[self.density * i:self.density * i + (self.density-1)])
            lc.set_linewidth(4)
            line = ax.add_collection(lc)
            ind += 1

        cbar = fig.colorbar(line, ax=ax)

        cursor = mplcursors.cursor(ax, hover=True)
        cursor.connect('add', lambda sel: self.show_cbar_value(sel, value))

        cbar.ax.text(1.05, 1.05, f"Max: {self.value_format(cmax)}", transform=cbar.ax.transAxes, va='bottom',
                     ha='left')
        cbar.ax.set_xlabel(f"Min: {self.value_format(cmin)}", rotation=0, ha='left')

        if cmin == cmax:
            cbar.set_ticks(np.linspace(cmin, cmin, num=1))

        return fig, ax, cbar

    def plot_undeformed(self, ax, plane):
        if plane == 'xy' or plane == 'yx':
            for i in range(0, len(self.model.elements)):
                ax.plot(self.x[self.density * i:self.density * i + self.density],
                        self.y[self.density * i:self.density * i + self.density],
                        linestyle='dashed', c='gray', linewidth=2)
        elif plane == 'yz' or plane == 'zy':
            for i in range(0, len(self.model.elements)):
                ax.plot(self.z[self.density * i:self.density * i + self.density ],
                        self.y[self.density * i:self.density * i + self.density],
                        linestyle='dashed', c='gray', linewidth=2)

        elif plane == 'xz' or plane == 'zx':
            for i in range(0, len(self.model.elements)):
                ax.plot(self.z[self.density * i:self.density * i + self.density],
                        self.x[self.density * i:self.density * i + self.density],
                        linestyle='dashed', c='gray', linewidth=2)
        return ax

    def plot_nodes(self, ax, plane, scale):
        if plane == 'xy' or plane == 'yx':
            for node in self.model.nodes:
                ax.scatter([node.x + scale * node.ux], [node.y + scale * node.uy], color='gray', marker='o', zorder=9,
                           s=1)
                ax.text(node.x + scale * node.ux + 0.03 * self.basic_dim,
                        node.y + scale * node.uy + 0.03 * self.basic_dim, node.index)

            if plane == 'yz' or plane == 'zy':
                for node in self.model.nodes:
                    ax.scatter([node.z + scale * node.uz], [node.y + scale * node.uy], color='gray', marker='o',
                               zorder=9, s=1)
                    ax.text(node.z + scale * node.uz + 0.03 * self.basic_dim,
                            node.y + scale * node.uy + 0.03 * self.basic_dim, node.index)

            if plane == 'xz' or plane == 'zx':
                for node in self.model.nodes:
                    ax.scatter([node.z + scale * node.uz], [node.x + scale * node.ux], color='gray', marker='o',
                               zorder=9, s=1)
                    ax.text(node.z + scale * node.uz + 0.03 * self.basic_dim,
                            node.x + scale * node.ux + 0.03 * self.basic_dim, node.index)

            return ax

    def plot_points(self, ax, plane):
        if plane == 'xy' or plane == 'yx':
            for line in self.model.geometry:
                ax.scatter([line.point1.x, line.point2.x], [line.point1.y, line.point2.y], color='black', marker='*',
                           zorder=10)
                ax.text(line.point1.x - 0.1 * self.basic_dim, line.point1.y - 0.1 * self.basic_dim, line.point1.index)
                ax.text(line.point2.x - 0.1 * self.basic_dim, line.point2.y - 0.1 * self.basic_dim, line.point2.index)

            if plane == 'yz' or plane == 'zy':
                for line in self.model.geometry:
                    ax.scatter([line.point1.z, line.point2.z], [line.point1.y, line.point2.y], color='black',
                               marker='*', zorder=10)
                    ax.text(line.point1.z - 0.1 * self.basic_dim, line.point1.y - 0.1 * self.basic_dim,
                            line.point1.index)
                    ax.text(line.point2.z - 0.1 * self.basic_dim, line.point2.y - 0.1 * self.basic_dim,
                            line.point2.index)

            if plane == 'xz' or plane == 'zx':
                for line in self.model.geometry:
                    ax.scatter([line.point1.z, line.point2.z], [line.point1.x, line.point2.x], color='black',
                               marker='*', zorder=10)
                    ax.text(line.point1.z - 0.1 * self.basic_dim, line.point1.x - 0.1 * self.basic_dim,
                            line.point1.index)
                    ax.text(line.point2.z - 0.1 * self.basic_dim, line.point2.x - 0.1 * self.basic_dim,
                            line.point2.index)
        return ax

    def show_cbar_value(self, sel, values):
        if type(sel.artist) == LineCollection:
            ind = int(sel.index[0])
            line_ind = int(sel.artist.get_label())
            f_ind = line_ind * self.density + ind
            val = self.value_format(values[f_ind])
            sel.annotation.set_text(f'Value:{val}')

    def value_format(self, value):
        if 1000 > abs(value) > 0.001 or value == 0:
            return round(value, 5)
        else:
            return f'{value:.4e}'

    def basic_plot_3d(self, scale, value):

        cmin = min(value)
        cmax = max(value)
        if cmin != cmax:
            tickvals = np.linspace(cmin, cmax, 5)
            ticktext = [self.value_format(x) for x in tickvals]
            ticktext[0] = f'Min:{self.value_format(cmin)}'
            ticktext[-1] = f'Min:{self.value_format(cmax)}'
            color = 'jet'
        else:
            tickvals = np.linspace(cmin, cmax, 1)
            ticktext = [self.value_format(cmin)]
            color = [[0, 'green'], [1, 'green']]

        layout = go.Layout(scene=dict(
                               xaxis=dict(showgrid=False, showline=False, showticklabels=False, backgroundcolor='white',
                                          title=''),
                               yaxis=dict(showgrid=False, showline=False, showticklabels=False, backgroundcolor='white',
                                          title=''),
                               zaxis=dict(showgrid=False, showline=False, showticklabels=False, backgroundcolor='white',
                                          title=''),
                               bgcolor='white', aspectmode='data'))

        fig = go.Figure(layout=layout)

        for i in range(0, len(self.model.elements)):
            fig.add_trace(go.Scatter3d(
                x=self.x[self.density * i:self.density * i + self.density]
                  + scale * self.ux[self.density * i:self.density * i + self.density],
                y=self.y[self.density * i:self.density * i + self.density]
                  + scale * self.uy[self.density * i:self.density * i + self.density],
                z=self.z[self.density * i:self.density * i + self.density]
                  + scale * self.uz[self.density * i:self.density * i + self.density],
                mode='lines',
                text=[f'Value: {self.value_format(x)}' for x in value[self.density * i:self.density * i + self.density]],
                hoverinfo='text',
                showlegend=False,
                line=dict(
                    color=value[self.density * i:self.density * i + self.density],
                    width=14,
                    colorscale=color,
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(
                        title='',
                        ticks='outside',
                        tickvals=tickvals,
                        ticktext=ticktext
                    ))))
        xs = np.linspace(0.1, 0.9, 10)
        ys = [-0.02, 0.02]
        fig.add_shape(type="line", x0=0.1, y0=0, x1=0.9, y1=0, line=dict(color="black", width=3))
        for x in xs:
            fig.add_shape(type="line", x0=x, y0=ys[0], x1=x, y1=ys[1], line=dict(color="black", width=3))
            fig.add_annotation(xref="paper", yref="paper", x=x, y=0.03, text="2", showarrow=False,
                               font=dict(size=12, color="black"), )


        return fig

    def plot_axis_system(self, fig):
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, self.basic_dim], mode='lines', showlegend=False,
                                   line=dict(color='blue', width=4, showscale=False,)))
        fig.add_cone(x=[0], y=[0], z=[self.basic_dim], u=[0], v=[0], w=[0.5 * self.basic_dim], anchor='tip',
                     colorscale=[[0, "blue"], [1, "blue"]], showscale=False, )
        t1 = dict(x=0, y=0, z=1.2 * self.basic_dim, text='Z', showarrow=False, font=dict(color='black', size=12))

        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, self.basic_dim], z=[0, 0], mode='lines', showlegend=False,
                                   line=dict(color='blue', width=4, showscale=False)))
        fig.add_cone(x=[0], y=[self.basic_dim], z=[0], u=[0], v=[0.5 * self.basic_dim], w=[0], anchor='tip',
                     colorscale=[[0, "blue"], [1, "blue"]], showscale=False, )
        t2 = dict(x=0, y=1.2 * self.basic_dim, z=0, text='Y', showarrow=False, font=dict(color='black', size=12))

        fig.add_trace(go.Scatter3d(x=[0, self.basic_dim], y=[0, 0], z=[0, 0], mode='lines', showlegend=False,
                                   line=dict(color='blue', width=4, showscale=False)))
        fig.add_cone(x=[self.basic_dim], y=[0], z=[0], u=[0.5 * self.basic_dim], v=[0], w=[0], anchor='tip',
                     colorscale=[[0, "blue"], [1, "blue"]], showscale=False, )
        t3 = dict(x=1.2 * self.basic_dim, y=0, z=0, text='X', showarrow=False, font=dict(color='black', size=12))

        fig.update_layout(scene=dict(annotations=[t1, t2, t3]))

        return fig

    def plot_undeformed_3d(self, fig):
        for i in range(0, len(self.model.elements)):
            fig.add_trace(go.Scatter3d(
                x=self.x[self.density * i:self.density * i + self.density],
                y=self.y[self.density * i:self.density * i + self.density],
                z=self.z[self.density * i:self.density * i + self.density],
                mode='lines',
                hoverinfo='text',
                line=dict(
                    color='gray',
                    width=5,),
                showlegend=False
            ))

        return fig

    def plot_nodes_3d(self, fig, scale):
        for node in self.model.nodes:
            fig.add_trace(go.Scatter3d(
                x=[node.x + scale * node.ux],
                y=[node.y + scale * node.uy],
                z=[node.z + scale * node.uz],
                mode='markers+text',
                hoverinfo='skip',
                text=[node.index],
                showlegend=False,
                marker=dict(
                    size=5,
                    symbol='square',
                    color='gray',
                )))
        fig.data = fig.data[::-1]
        return fig

    def plot_points_3d(self, fig):
        fig.data = fig.data[::-1]
        for line in self.model.geometry:
            fig.add_trace(go.Scatter3d(
                x=[line.point1.x, line.point2.x],
                y=[line.point1.y, line.point2.y],
                z=[line.point1.z, line.point2.z],
                mode='markers+text',
                hoverinfo='skip',
                text=[line.point1.index, line.point2.index],
                showlegend=False,
                marker=dict(
                    size=5,
                    symbol='diamond',
                    color='black',
                )))
        fig.data = fig.data[::-1]
        return fig

    def nodal_deformation(self, option, scale='auto', plane='xy', show_undeformed=False,
                          show_points=False, show_nodes=False):

        if scale == 'auto':
            scale = self.scale

        value = None
        title = ''
        option = option.lower()

        if option == 'total':
            value = np.sqrt(self.ux ** 2 + self.uy ** 2)
            title = 'Total nodal deformation'
        elif option == 'ux':
            value = self.ux
            title = 'Ux nodal deformation'
        elif option == 'uy':
            value = self.uy
            title = 'Uy nodal deformation'
        elif option == 'uz':
            value = self.uz
            title = 'Uz nodal deformation'
        elif option == 'fx':
            value = self.fx
            title = 'fx nodal angle'
        elif option == 'fy':
            value = self.fy
            title = 'fy nodal angle'
        elif option == 'fz':
            value = self.fz
            title = 'fz nodal angle'

        fig, ax, cbar = self.basic_plot(scale, value, plane)
        ax.set_title(title)

        if show_undeformed:
            ax = self.plot_undeformed(ax, plane)
        if show_points:
            ax = self.plot_points(ax, plane)
        if show_nodes:
            ax = self.plot_nodes(ax, plane, scale)

        ax.axis('equal')

        self.mpl_figs.append(fig)

        return fig

    def element_stress(self, option, scale='auto', plane='xy', show_undeformed=False, show_points=False, show_nodes=False):

        option = option.upper()

        if scale == 'auto':
            scale = self.scale

        value = np.array([])
        title = ''

        if option == 'S1':
            title = 'Normal stress due to stretch'
            for element in self.model.elements:
                value = np.append(value, element.S1 * np.ones(self.density))
        elif option == 'T1':
            title = 'Maximum shear stress due to torsion'
            for element in self.model.elements:
                value = np.append(value, element.T1 * np.ones(self.density))
        elif option == 'S2':
            title = 'Maximum Normal stress due to bending in local y-direction'
            for element in self.model.elements:
                value = np.append(value, element.S2 * np.ones(self.density))
        elif option == 'T2':
            title = 'Shear stress due to bending in local y-direction'
            for element in self.model.elements:
                value = np.append(value, element.T2 * np.ones(self.density))
        elif option == 'S3':
            title = 'Maximum Normal stress due to local bending in z-direction'
            for element in self.model.elements:
                value = np.append(value, element.S3 * np.ones(self.density))
        elif option == 'T3':
            title = 'Shear stress due to bending in local z-direction'
            for element in self.model.elements:
                value = np.append(value, element.T3 * np.ones(self.density))
        elif option == 'TOTAL':
            title = 'von Mises stress'
            for element in self.model.elements:
                value = np.append(value, element.von_Mises_stress * np.ones(self.density))

        fig, ax, cbar = self.basic_plot(scale, value, plane)
        ax.set_title(title)

        if show_undeformed:
            ax = self.plot_undeformed(ax, plane)
        if show_points:
            ax = self.plot_points(ax, plane)
        if show_nodes:
            ax = self.plot_nodes(ax, plane, scale)

        ax.axis('equal')

        self.mpl_figs.append(fig)

        return fig

    def nodal_deformation_3d(self, option, scale='auto',  show_undeformed=False, show_axis_system=True,
                             show_nodes=False, show_points=False):
        if scale == 'auto':
            scale = self.scale
        value = None
        title = ''
        option = option.lower()

        if option == 'total':
            value = np.sqrt(self.ux ** 2 + self.uy ** 2)
            title = 'Total nodal deformation'
        elif option == 'ux':
            value = self.ux
            title = 'Ux nodal deformation'
        elif option == 'uy':
            value = self.uy
            title = 'Uy nodal deformation'
        elif option == 'uz':
            value = self.uz
            title = 'Uz nodal deformation'
        elif option == 'fx':
            value = self.fx
            title = 'fx nodal angle'
        elif option == 'fy':
            value = self.fy
            title = 'fy nodal angle'
        elif option == 'fz':
            value = self.fz
            title = 'fz nodal angle'

        fig = self.basic_plot_3d(scale, value)
        fig.update_layout(title=title)

        if show_axis_system:
            fig = self.plot_axis_system(fig)
        if show_undeformed:
            fig = self.plot_undeformed_3d(fig)
        if show_nodes:
            fig = self.plot_nodes_3d(fig, scale)
        if show_points:
            if show_nodes:
                fig.data = fig.data[::-1]
            fig = self.plot_points_3d(fig)

        self.plotly_figs.append(fig)

        return fig

    def element_stress_3d(self, option, scale='auto', show_undeformed=False, show_axis_system=True, show_points=False,
                       show_nodes=False):
        option = option.upper()

        if scale == 'auto':
            scale = self.scale

        value = np.array([])
        title = ''

        if option == 'S1':
            title = 'Normal stress due to stretch'
            for element in self.model.elements:
                value = np.append(value, element.S1 * np.ones(self.density))
        elif option == 'T1':
            title = 'Maximum shear stress due to torsion'
            for element in self.model.elements:
                value = np.append(value, element.T1 * np.ones(self.density))
        elif option == 'S2':
            title = 'Maximum Normal stress due to bending in local y-direction'
            for element in self.model.elements:
                value = np.append(value, element.S2 * np.ones(self.density))
        elif option == 'T2':
            title = 'Shear stress due to bending in local y-direction'
            for element in self.model.elements:
                value = np.append(value, element.T2 * np.ones(self.density))
        elif option == 'S3':
            title = 'Maximum Normal stress due to local bending in z-direction'
            for element in self.model.elements:
                value = np.append(value, element.S3 * np.ones(self.density))
        elif option == 'T3':
            title = 'Shear stress due to bending in local z-direction'
            for element in self.model.elements:
                value = np.append(value, element.T3 * np.ones(self.density))
        elif option == 'TOTAL':
            title = 'von Mises stress'
            for element in self.model.elements:
                value = np.append(value, element.von_Mises_stress * np.ones(self.density))

        fig = self.basic_plot_3d(scale, value)
        fig.update_layout(title=title)

        if show_axis_system:
            fig = self.plot_axis_system(fig)
        if show_undeformed:
            fig = self.plot_undeformed_3d(fig)
        if show_nodes:
            fig = self.plot_nodes_3d(fig, scale)
        if show_points:
            if show_nodes:
                fig.data = fig.data[::-1]
            fig = self.plot_points_3d(fig)

        self.plotly_figs.append(fig)

        return fig

    def force_torque_plot(self, option, plane='xy'):

        option = option.lower()
        values = None
        title = ''

        if option == 'my':
            values = [element.M_Y for element in self.model.elements]
            title = 'Bending moment - local z axis'
        elif option == 'mz':
            values = [element.M_Z for element in self.model.elements]
            title = 'Bending moment - local y axis'
        elif option == 'fy':
            values = [element.F_Y for element in self.model.elements]
            title = 'Shear force - local y direction'
        elif option == 'fz':
            values = [element.F_Z for element in self.model.elements]
            title = 'Shear force - local z direction'
        elif option == 'fx':
            values = [element.F_S for element in self.model.elements]
            title = 'Strain force - local x direction'
        elif option == 'ft':
            values = [element.F_S for element in self.model.elements]
            title = ' Shear force due to torsion - local x axis'

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(title)

        cmin = min(values)
        cmax = max(values)

        norm = plt.Normalize(cmin, cmax)

        if cmin == cmax:
            single_color = plt.get_cmap('jet')(0.5)  # Choose a single color
            cmap = mpl.colors.ListedColormap([single_color])
            norm = plt.Normalize(cmin, cmax)
        else:
            cmap = plt.get_cmap('jet')
            norm = plt.Normalize(cmin, cmax)

        for element in self.model.elements:
            value = None
            if option == 'my':
                value = element.M_Y
            elif option == 'mz':
                value = element.M_Z
            elif option == 'fy':
                value = element.F_Y
            elif option == 'fz':
                value = element.F_Z

            color = cmap(norm(value))

            value = value * self.to_scale / (max(np.abs(values)))

            g1_matrix = np.dot(element.t_matrix[0:2, 0:2], np.array([[0], [value]]))
            g2_matrix = np.dot(element.t_matrix[6:8, 6:8], np.array([[0], [value]]))

            xx = [element.node1.x + g1_matrix[0, 0], element.node2.x + g2_matrix[0, 0]]
            yy = [element.node1.y + g1_matrix[1, 0], element.node2.y + g2_matrix[1, 0]]

            x = [element.node1.x, element.node2.x]
            y = [element.node1.y, element.node2.y]

            ax.plot(xx, yy, c=color, alpha=0.75)
            ax.plot(x, y, linestyle='dashed', c='black', linewidth=2)
            polygon = patches.Polygon(np.column_stack([xx + x[::-1], yy + y[::-1]]), closed=True, facecolor=color,
                                      edgecolor='none', alpha=0.75)
            ax.add_patch(polygon)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(values)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.text(1.05, 1.05, f"Max: {round(np.max(values), 7)}", transform=cbar.ax.transAxes, va='bottom',
                     ha='left')
        cbar.ax.set_xlabel(f"Min: {round(np.min(values), 7)}", rotation=0, ha='left')
        if cmin == cmax:
            cbar.set_ticks(np.linspace(cmin, cmax, num=1))
        ax.axis('equal')

        self.mpl_figs.append(fig)

        return fig

    def evaluate_all_results(self):
        app = QApplication(sys.argv)
        index = 1
        windows = []
        for fig in self.plotly_figs:
            plotly.offline.plot(fig, filename=f'fig{index}.html', auto_open=False)
            web = QWebEngineView()
            file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'fig{index}.html'))
            web.load(QUrl.fromLocalFile(file_path))
            web.setWindowTitle('Results graph')
            web.setGeometry(200, 200, 800, 600)
            web.page().runJavaScript("""var graphDiv = document.getElementById('myDiv');

var N = 40,
    x = d3.range(N),
    y = d3.range(N).map( d3.random.normal() ),
    data = [ { x:x, y:y } ];
    layout = { title:'Click-drag to zoom' };

Plotly.newPlot(graphDiv, data, layout);

graphDiv.on('plotly_relayout',
    function(eventdata){
        alert( 'ZOOM!' + '\n\n' +
            'Event data:' + '\n' +
             JSON.stringify(eventdata) + '\n\n' +
            'x-axis start:' + eventdata['xaxis.range[0]'] + '\n' +
            'x-axis end:' + eventdata['xaxis.range[1]'] );
    });""")
            windows.append(web)
            index += 1
        for window in windows:
            window.show()

        plt.show()

        if len(self.mpl_figs) > 0:
            print('xd')
            return None
        else:
            sys.exit(app.exec_())
