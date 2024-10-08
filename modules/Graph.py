import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import mplcursors
import matplotlib as mpl
import matplotlib.patches as patches
import plotly.graph_objects as go
from loguru import logger


def show_cbar_value(sel, values):
    # shows values of matplotlib results by hover of them by mouse cursor (for basic plots)
    if type(sel.artist) == LineCollection:
        ind = int(sel.index[0])
        line_ind = int(sel.artist.get_label())
        val = values[line_ind, ind]
        val = value_format(val)
        sel.annotation.set_text(f'Value:{val}')


def show_cbar_value_bar(sel, values):
    # shows values of matplotlib results by hover of them by mouse cursor (for bar plots)
    sel.annotation.set_text(f'None')
    if type(sel.artist) == matplotlib.lines.Line2D:
        line_ind = sel.artist.get_label()
        if line_ind[0:6] != '_child':
            line_ind = line_ind.split(',')

            id1 = int(line_ind[0])
            id2 = int(line_ind[1])
            val = values[id1, id2]
            val = value_format(val)
            sel.annotation.set_text(f'Value:{val}')
        else:
            sel.annotation.set_text(f'None')


def value_format(value):
    if 1000 > abs(value) > 0.001 or value == 0:
        return round(value, 5)
    else:
        return f'{value:.4e}'


class Graph2d:
    # matplotlib figure object
    def __init__(self, displacement_vector, points_vector, values, scale, indexes, points_key, cursor):
        self.points_vector = points_vector  # points for which deformations were calculated
        self.displacement_vector = displacement_vector  # used to display deflection lines for all types of results
        self.values = values  # values to plot
        self.indexes = indexes  # plane vector
        self.points_key = points_key  # used to plot model construction points
        self.cursor = cursor  # object used to show values by hover

        self.scale = scale  # factor by which the deformation results are multiplied
        self.basic_dim = None  # value used to nicely set numbering on the chart

        self.calculate_vars()

    def calculate_vars(self):
        """
        automatic calculation of scale, basic on system dimensions and max value of deformations
         - if user did not define it
        """
        #
        s1 = np.amax(self.points_vector)
        s2 = np.amin(self.points_vector)

        max_u = np.amax(np.abs(self.displacement_vector[:, 0:3]))

        if self.scale == 'auto':
            if max_u != 0:
                self.scale = (s1 - s2) / (max_u * 12)
            else:
                self.scale = 1

            if self.scale < 1:
                self.scale = 1

        self.basic_dim = (s1 - s2) / 80

    def basic_plot(self):
        logger.warning("""Matplotlib plots have not been optimized yet.
         It is recommended to use plotly plots, even for 2d systems""")

        fig, ax = plt.subplots(figsize=(8, 6))

        cmin = np.amin(self.values)
        cmax = np.amax(self.values)

        if cmin != cmax:
            cmap = plt.get_cmap('jet')
        else:
            single_color = plt.get_cmap('jet')(0.5)
            cmap = mpl.colors.ListedColormap([single_color])

        norm = plt.Normalize(cmin, cmax)
        ind = 0

        # plotting each element separately
        for i in range(0, len(self.displacement_vector)):

            in_x = self.points_vector[i, self.indexes[0]].flatten() + self.scale * self.displacement_vector[
                i, self.indexes[0]].flatten()
            in_y = self.points_vector[i, self.indexes[1]].flatten() + self.scale * self.displacement_vector[
                i, self.indexes[1]].flatten()

            in_value = self.values[i]

            """
            In matplotlib each line can be assigned one value. However, the input data has two values at the end of
            the line for deformation values. Therefore, the input data for deformations must be reduced.
            """
            if len(self.values[i]) == len(self.points_vector[i]):
                in_value = np.array([np.mean(self.values[i, n:n + 2]) for n in range(0, len(self.values[i]) - 1)])

            # plot deflection line
            points = np.array([in_x, in_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cmap, norm=norm, label=str(ind))
            # assign colors to lines based on values
            lc.set_array(in_value)
            lc.set_linewidth(5.5)

            line = ax.add_collection(lc)
            ind += 1

        # scale bar definition for the chart
        cbar = fig.colorbar(line, ax=ax)
        cbar.ax.text(1.05, 1.05, f"Max: {value_format(cmax)}", transform=cbar.ax.transAxes, va='bottom', ha='left')
        cbar.ax.set_xlabel(f"Min: {value_format(cmin)}", rotation=0, ha='left')

        if cmin == cmax:
            cbar.set_ticks(np.linspace(cmin, cmin, num=1))

        return fig, ax, cbar

    def get_fig(self, show_undeformed=False, show_points=False, show_nodes=False):
        fig, ax, cbar = self.basic_plot()

        if show_undeformed:
            ax = self.plot_undeformed(ax)
        if show_points:
            ax = self.plot_points(ax)
        if show_nodes:
            ax = self.plot_nodes(ax)

        if self.cursor:
            logger.warning("""Cursor for matplotlib plots is very performance intensive.
             It is not recommended to use it for meshes with lots of elements""")

            cursor = mplcursors.cursor(ax, hover=True)
            cursor.connect('add', lambda sel: show_cbar_value(sel, self.values))

        ax.axis('equal')
        self.clear_data()

        return fig, ax

    def plot_undeformed(self, ax):
        # draws the system in an undeformed state
        for element in self.points_vector:
            ax.plot(element[self.indexes[0], [0, -1]], element[self.indexes[1], [0, -1],], c='gray', linewidth=2)
        return ax

    def plot_nodes(self, ax):
        # marks all created nodes on the chart along with their numbering
        for i in range(0, len(self.displacement_vector)):
            in_x = self.points_vector[i, self.indexes[0], [0, -1]] + self.scale * self.displacement_vector[
                i, self.indexes[0], [0, -1]]
            in_y = self.points_vector[i, self.indexes[1], [0, -1]] + self.scale * self.displacement_vector[
                i, self.indexes[1], [0, -1]]

            ax.scatter(in_x, in_y, color='gray', marker='s', zorder=9, s=15)

            ax.text(in_x[0] + self.basic_dim, in_y[0] + self.basic_dim, i)
            ax.text(in_x[1] + self.basic_dim, in_y[1] + self.basic_dim, i + 1)

        return ax

    def plot_points(self, ax):
        # marks all the points that make up the system's structure on the chart
        for point in self.points_key:
            in_x = [point['coordinates'][self.indexes[0]]]
            in_y = [point['coordinates'][self.indexes[1]]]

            ax.scatter(in_x, in_y, color='black', marker='*', zorder=10)
            ax.text(in_x + (-3 * self.basic_dim), in_y + (-3 * self.basic_dim), point['index'])

        return ax

    def clear_data(self):
        self.points_vector = None
        self.displacement_vector = None


class Bar_2d:
    # other style of previous graph - bar matplotlib plot
    def __init__(self, values, points, indexes, cursor):
        self.values = values
        self.points = points
        self.indexes = indexes
        self.cursor = cursor

        self.to_scale = None
        self.calculate_vars()

    def calculate_vars(self):
        s1 = np.amax(self.points)
        s2 = np.amin(self.points)

        self.to_scale = (s1 - s2) / 4

    def get_fig(self):
        logger.warning("""Matplotlib plots have not been optimized yet.
         It is recommended to use plotly plots, even for 2d systems""")

        logger.warning("""Bar plots have not been optimized yet, for both types of plots.
         It is recommended to use normal 3d results""")

        fig, ax = plt.subplots(figsize=(8, 6))

        amax = np.amax(np.abs(self.values))
        cmin = np.amin(self.values)
        cmax = np.amax(self.values)

        if cmin == cmax:
            single_color = plt.get_cmap('jet')(0.5)
            cmap = mpl.colors.ListedColormap([single_color])
            norm = plt.Normalize(cmin, cmax)
        else:
            cmap = plt.get_cmap('jet')
            norm = plt.Normalize(cmin, cmax)

        for i in range(0, len(self.points)):
            in_values = self.values[i]
            in_points = self.points[i]

            # calculating the angle at which the element is located relative to the global system
            if in_points[self.indexes[0], 1] - in_points[self.indexes[0], 0] != 0:
                m = (in_points[self.indexes[1], 1] - in_points[self.indexes[1], 0]) / (
                        in_points[self.indexes[0], 1] - in_points[self.indexes[0], 0])

                angle = -np.arctan(m)

            elif in_points[self.indexes[1], 1] > in_points[self.indexes[1], 0]:
                angle = np.pi / 2
            else:
                angle = -np.pi / 2

            """
            In matplotlib each line can be assigned one value. However, the input data has two values at the end of
            the line for deformation values. Therefore, the input data for deformations must be reduced.
            """
            if len(self.values[i]) == len(self.points[i]):
                in_values = np.array([np.mean(self.values[i, n:n + 2]) for n in range(0, len(self.values[i]) - 1)])

            # drawing a bar for each element separately
            for j in range(0, len(in_points[0]) - 1):

                value = in_values[j]
                color = cmap(norm(value))

                # scaling the size of the bar based on the dimensions of the system
                if value != 0:
                    value = value * self.to_scale / amax

                # in this case there is no drawing of deflection lines
                # coordinates of element nodes
                x = [in_points[self.indexes[0], j], in_points[self.indexes[0], j + 1]]
                y = [in_points[self.indexes[1], j], in_points[self.indexes[1], j + 1]]

                # coordinates of bar
                xx = [in_points[self.indexes[0], j] + value * np.sin(angle),
                      in_points[self.indexes[0], j + 1] + value * np.sin(angle)]
                yy = [in_points[self.indexes[1], j] + value * np.cos(angle),
                      in_points[self.indexes[1], j + 1] + value * np.cos(angle)]

                ind = str(i) + ',' + str(j)
                ax.plot(xx, yy, c=color, alpha=0.75, label=ind)
                ax.plot(x, y, linestyle='dashed', c='black', linewidth=1)
                polygon = patches.Polygon(np.column_stack([xx + x[::-1], yy + y[::-1]]), closed=True, facecolor=color,
                                          edgecolor='none', alpha=0.75)
                ax.add_patch(polygon)

        # scale bar definition for the chart
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(self.values)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.text(1.05, 1.05, f"Max: {value_format(np.max(self.values))}", transform=cbar.ax.transAxes, va='bottom',
                     ha='left')
        cbar.ax.set_xlabel(f"Min: {value_format(np.min(self.values))}", rotation=0, ha='left')
        if cmin == cmax:
            cbar.set_ticks(np.linspace(cmin, cmax, num=1))

        if self.cursor:
            logger.warning("""Cursor for matplotlib plots is very performance intensive.
             It is not recommended to use it for meshes with lots of elements""")

            cursor = mplcursors.cursor(ax, hover=True)
            cursor.connect('add', lambda sel: show_cbar_value_bar(sel, self.values))

        ax.axis('equal')

        self.points = None

        return fig, ax


class Graph3d(Graph2d):
    # plotly figure object
    def __init__(self, displacement_vector, points_vector, values, scale, points_key):
        super().__init__(displacement_vector, points_vector, values, scale, None, points_key, None)

    def basic_plot(self):
        cmin = np.amin(self.values)
        cmax = np.amax(self.values)

        # customization of bar scale appearance
        if cmin != cmax:
            tickvals = np.linspace(cmin, cmax, 5)
            ticktext = [value_format(x) for x in tickvals]
            ticktext[0] = f'Min:{value_format(cmin)}'
            ticktext[-1] = f'Max:{value_format(cmax)}'
            color = 'jet'
        else:
            tickvals = np.linspace(cmin, cmax, 1)
            ticktext = [value_format(cmin)]
            color = [[0, 'green'], [1, 'green']]

        # customization of graph appearance
        layout = go.Layout(scene=dict(
            xaxis=dict(showgrid=True, showline=True, showbackground=False, backgroundcolor='rgb(211, 211, 211)',
                       title='', zerolinewidth=5, gridcolor='black'),
            yaxis=dict(showgrid=True, showline=True, showbackground=False, backgroundcolor='rgb(211, 211, 211)',
                       title='', zerolinewidth=5, gridcolor='black'),
            zaxis=dict(showgrid=True, showline=True, showbackground=False, backgroundcolor='rgb(211, 211, 211)',
                       title='', zerolinewidth=5, gridcolor='black'),
            bgcolor='white', aspectmode='data'), margin=dict(t=30, r=0, l=20, b=10))

        fig = go.Figure(layout=layout)
        fig = self.set_plane(fig)
        fig.update_scenes(camera_projection_type="orthographic")

        # plotting each element separately
        for i in range(0, len(self.displacement_vector)):
            in_x = self.points_vector[i, 0].flatten() + self.scale * self.displacement_vector[i, 0].flatten()
            in_y = self.points_vector[i, 1].flatten() + self.scale * self.displacement_vector[i, 1].flatten()
            in_z = self.points_vector[i, 2].flatten() + self.scale * self.displacement_vector[i, 2].flatten()

            in_value = self.values[i]

            """
            In plotly (in opposite to matplotlib) each line can be assigned two value at the end and at the beginning.
            However, the input data has one value for element for stress and force data. Therefore, the input data for
            stress and force must be extended.
            """
            if len(self.values[i]) != len(self.points_vector[i]):
                in_value = np.array([np.mean(self.values[i, n:n + 2]) for n in range(0, len(self.values[i]) - 1)])
                in_value = np.insert(in_value, 0, self.values[i, 0])
                in_value = np.append(in_value, self.values[i, -1])

            # drawing a deflection line and assigning it colors depending on the values
            fig.add_trace(go.Scatter3d(
                x=in_x, y=in_y, z=in_z,
                mode='lines',
                text=[f'Value: {value_format(x)}' for x in self.values[i]],
                hoverinfo='text',
                showlegend=False,
                line=dict(
                    color=in_value,
                    width=20,
                    colorscale=color,
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(
                        title='',
                        ticks='outside',
                        tickvals=tickvals,
                        ticktext=ticktext))))
        return fig

    def plot_undeformed(self, fig):
        # draws the system in an undeformed state
        for element in self.points_vector:
            fig.add_trace(go.Scatter3d(
                x=element[0, [0, -1]],
                y=element[1, [0, -1]],
                z=element[2, [0, -1]],
                mode='lines',
                hoverinfo='text',
                line=dict(
                    color='gray',
                    width=5, ),
                showlegend=False))
        fig.data = fig.data[::-1]
        return fig

    def plot_nodes(self, fig):
        # marks all created nodes on the chart along with their numbering
        for i in range(0, len(self.displacement_vector)):
            in_x = self.points_vector[i, 0, [0, -1]] + self.scale * self.displacement_vector[i, 0, [0, -1]]
            in_y = self.points_vector[i, 1, [0, -1]] + self.scale * self.displacement_vector[i, 1, [0, -1]]
            in_z = self.points_vector[i, 2, [0, -1]] + self.scale * self.displacement_vector[i, 2, [0, -1]]

            fig.add_trace(go.Scatter3d(
                x=in_x, y=in_y, z=in_z,
                mode='markers+text',
                hoverinfo='skip',
                text=[i, i + 1],
                showlegend=False,
                marker=dict(
                    size=5,
                    symbol='square',
                    color='gray')))
        fig.data = fig.data[::-1]

        return fig

    def plot_points(self, fig):
        # marks all the points that make up the system's structure on the chart
        fig.data = fig.data[::-1]

        for point in self.points_key:
            in_x = [point['coordinates'][0]]
            in_y = [point['coordinates'][1]]
            in_z = [point['coordinates'][2]]

            fig.add_trace(go.Scatter3d(
                x=in_x, y=in_y, z=in_z,
                mode='markers+text',
                hoverinfo='skip',
                text=[point['index']],
                showlegend=False,
                marker=dict(
                    size=5,
                    symbol='diamond',
                    color='black')))
        fig.data = fig.data[::-1]

        return fig

    def set_plane(self, fig):
        # if the system is oriented in one global plane, the default view is oriented normal to that plane
        x = False
        y = False
        z = False
        for point in self.points_key:
            if point['coordinates'][0] != 0:
                x = True
            if point['coordinates'][1] != 0:
                y = True
            if point['coordinates'][2] != 0:
                z = True

        if not x:
            fig.update_layout(scene=dict(camera=dict(eye=dict(x=1, y=0, z=0), up=dict(x=0, y=1, z=0))))
        if not y:
            fig.update_layout(scene=dict(camera=dict(eye=dict(x=0, y=1, z=0), up=dict(x=0, y=0, z=1))))
        if not z:
            fig.update_layout(scene=dict(camera=dict(eye=dict(x=0, y=0, z=1), up=dict(x=0, y=1, z=0))))

        return fig

    def get_fig(self, show_undeformed=False, show_points=False, show_nodes=False):
        fig = self.basic_plot()

        if show_undeformed:
            fig = self.plot_undeformed(fig)
        if show_points:
            fig = self.plot_points(fig)
        if show_nodes:
            fig = self.plot_nodes(fig)

        self.clear_data()

        return fig


class Bar_3d(Bar_2d):
    # other style of previous graph - bar matplotlib plot
    def __init__(self, values, points, elements, points_key):
        super().__init__(values, points, None, None)
        self.values = values
        self.points = points
        self.elements = elements
        self.points_key = points_key

    def set_plane(self, fig):
        # if the system is oriented in one global plane, the default view is oriented normal to that plane
        x = False
        y = False
        z = False
        for point in self.points_key:
            if point['coordinates'][0] != 0:
                x = True
            if point['coordinates'][1] != 0:
                y = True
            if point['coordinates'][2] != 0:
                z = True

        if not x:
            fig.update_layout(scene=dict(camera=dict(eye=dict(x=1, y=0, z=0), up=dict(x=0, y=1, z=0))))
        if not y:
            fig.update_layout(scene=dict(camera=dict(eye=dict(x=0, y=1, z=0), up=dict(x=0, y=0, z=1))))
        if not z:
            fig.update_layout(scene=dict(camera=dict(eye=dict(x=0, y=0, z=1), up=dict(x=0, y=1, z=0))))

        return fig

    def get_fig(self):
        logger.warning("""Bar plots have not been optimized yet, for both types of plots.
                                It is recommended to use normal 3d results""")

        # customization of graph appearance
        layout = go.Layout(scene=dict(
            xaxis=dict(showgrid=True, showline=True, showbackground=False, backgroundcolor='rgb(211, 211, 211)',
                       title='', zerolinewidth=5, gridcolor='black'),
            yaxis=dict(showgrid=True, showline=True, showbackground=False, backgroundcolor='rgb(211, 211, 211)',
                       title='', zerolinewidth=5, gridcolor='black'),
            zaxis=dict(showgrid=True, showline=True, showbackground=False, backgroundcolor='rgb(211, 211, 211)',
                       title='', zerolinewidth=5, gridcolor='black'),
            bgcolor='white', aspectmode='data'), margin=dict(t=30, r=0, l=20, b=10))

        fig = go.Figure(layout=layout)

        amax = np.amax(np.abs(self.values))
        cmin = np.amin(self.values)
        cmax = np.amax(self.values)

        # customization of bar scale appearance
        if cmin != cmax:
            tickvals = np.linspace(cmin, cmax, 5)
            ticktext = [value_format(x) for x in tickvals]
            ticktext[0] = f'Min:{value_format(cmin)}'
            ticktext[-1] = f'Max:{value_format(cmax)}'
            color = 'jet'

            cmap = plt.get_cmap('jet')
            norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
            scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        else:
            tickvals = np.linspace(cmin, cmax, 1)
            ticktext = [value_format(cmin)]
            color = [[0, 'green'], [1, 'green']]
            scalar_mappable = None

        for i in range(0, len(self.points)):
            # drawing a bar for each element separately
            in_values = self.values[i]
            in_points = self.points[i]

            # the slope of the line is calculated using the element transformation matrix
            t_matrix = self.elements[i].t_matrix[0:3, 0:3]

            """
            In plotly (in opposite to matplotlib) each line can be assigned two value at the end and at the beginning.
            However, the input data has one value for element for stress and force data. Therefore, the input data for
            stress and force must be extended.
            """
            if len(self.values[i]) == len(self.points[i]):
                in_values = np.array([np.mean(self.values[i, n:n + 2]) for n in range(0, len(self.values[i]))])

            for j in range(0, len(in_points[0]) - 1):
                value = in_values[j]

                # scaling the size of the bar based on the dimensions of the system
                if value != 0:
                    value = value * self.to_scale / amax

                # bar coordinates in the global system
                local_value_vector = np.array([0, value, 0])
                global_value_vector = np.dot(np.linalg.inv(t_matrix), local_value_vector)

                # in this case there is no drawing of deflection lines
                # coordinates of element nodes
                x = [in_points[0, j], in_points[0, j + 1]]
                y = [in_points[1, j], in_points[1, j + 1]]
                z = [in_points[2, j], in_points[2, j + 1]]

                # coordinates of bar
                xx = [in_points[0, j] + global_value_vector[0], in_points[0, j + 1] + global_value_vector[0]]
                yy = [in_points[1, j] + global_value_vector[1], in_points[1, j + 1] + global_value_vector[1]]
                zz = [in_points[2, j] + global_value_vector[2], in_points[2, j + 1] + global_value_vector[2]]

                # line of value
                fig.add_trace(go.Scatter3d(
                    x=xx, y=yy, z=zz,
                    mode='lines',
                    text=[f'Value: {value_format(in_values[j])}', f'Value: {value_format(in_values[j])}'],
                    hoverinfo='text',
                    showlegend=False,
                    line=dict(
                        color=[in_values[j], in_values[j]],
                        width=8,
                        colorscale=color,
                        cmin=cmin,
                        cmax=cmax,
                        colorbar=dict(
                            title='',
                            ticks='outside',
                            tickvals=tickvals,
                            ticktext=ticktext))))

                # line along element
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    hoverinfo='text',
                    line=dict(
                        color='black',
                        width=7, ),
                    showlegend=False))

                # set color to mesh
                if scalar_mappable is not None:
                    rgba_color = scalar_mappable.to_rgba(in_values[j])
                    rgb = [int(255 * x) for x in rgba_color[0:3]]
                    in_color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]},0.5)"
                else:
                    in_color = "green"

                mesh_x = [x[0], x[-1], xx[0], xx[-1]]
                mesh_y = [y[0], y[-1], yy[0], yy[-1]]
                mesh_z = [z[0], z[-1], zz[0], zz[-1]]

                # Define the connectivity of the rectangle
                i = [0, 1, 2, 0]
                j = [1, 2, 3, 0]
                k = [2, 3, 0, 1]

                # mesh between plotted value line and line along element
                fig.add_mesh3d(
                    x=mesh_x,
                    y=mesh_y,
                    z=mesh_z,
                    i=i,
                    j=j,
                    k=k,
                    color=in_color,
                    hoverinfo='none')

        self.points = None
        self.elements = None

        fig = self.set_plane(fig)
        fig.update_scenes(camera_projection_type="orthographic")

        return fig


class Section_graph:
    def __init__(self, element, values, line, length):
        self.element = element
        self.values = values
        self.line = line
        self.length = length

    def get_fig(self):
        section = self.element.section
        self.values = section.mask(self.values)
        indicator = f"\n Line index = {self.line.index}||Length = {self.length}"
        cmin = np.amin(self.values)
        cmax = np.amax(self.values)

        if cmin != cmax:
            cmap = plt.get_cmap('jet')
        else:
            single_color = plt.get_cmap('jet')(0.5)
            cmap = mpl.colors.ListedColormap([single_color])

        norm = plt.Normalize(cmin, cmax)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pcolormesh(section.z_points, section.y_points, self.values, cmap=cmap, norm=norm)

        ax.axhline(y=section.origin_point[1], color='k')
        ax.axvline(x=section.origin_point[0], color='k')
        ax.set_ylabel('Y')
        ax.set_xlabel('Z' + indicator)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(self.values)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.text(1.05, 1.05, f"Max: {value_format(cmax)}", transform=cbar.ax.transAxes, va='bottom', ha='left')
        cbar.ax.set_xlabel(f"Min: {value_format(cmin)}", rotation=0, ha='left')

        if cmin == cmax:
            cbar.set_ticks(np.linspace(cmin, cmin, num=1))
        ax.axis('equal')

        return fig, ax

