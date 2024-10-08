import matplotlib.pyplot as plt
import plotly
import sys
import os
import plotly.offline
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication
from Model import Static
from modules.Graph import Graph2d, Bar_2d, Graph3d, Bar_3d, Section_graph
import numpy as np
import plotly.graph_objects as go
from Line import Line
from loguru import logger


class Static_results:
    def __init__(self, model: Static):
        self._model = model

        self._plotly_figs = []  # list of created matplotlib graphs
        self._mpl_figs = []  # list of created plotly graphs

    def _check_input(self):
        if type(self._model) != Static:
            raise TypeError("argument - model must be Static, Modal or Harmonic")
        elif not self._model.solved:
            raise AttributeError("passed model is not solved")

    # resolution defines number of points per element to approximate values by shape functions
    def _get_resolution(self, value, option='default'):
        # automatic calculate resolution if user did not define it
        max_resolution = 100
        min_resolution = 3
        if value != 'auto':
            return value
        else:
            q_e = len(self._model.elements)
            if q_e < 30:
                resolution = max_resolution
            elif q_e < 1000:
                resolution = int(max_resolution - (max_resolution - min_resolution) * (q_e - 10) / 40)
            else:
                resolution = min_resolution

            if option == 'total':
                resolution = int(resolution/8)
            if resolution < 3:
                resolution = 3

            return resolution

    def _get_data(self, option, resolution, data_type, index=0):
        # returns deformations, stress and forces for all elements in analysis
        displacement_vector, points_vector = self._model.get_elements_disp(resolution, index=index)
        option = option.lower()

        disp_key = {'ux': {'index': 0, 'tittle': 'Ux deformation'}, 'uy': {'index': 1, 'tittle': 'Uy deformation'},
                    'uz': {'index': 2, 'tittle': 'Uz deformation'}, 'rotx': {'index': 3, 'tittle': 'Rot_x angle'},
                    'roty': {'index': 4, 'tittle': 'Rot_y angle'}, 'rotz': {'index': 5, 'tittle': 'Rot_z angle'},
                    'total_disp': {'index': 6, 'tittle': 'Total deformation'},
                    'total_rot': {'index': 7, 'tittle': 'Total rotation'}}

        stress_key = {'nx': {'index': 0, 'tittle': 'Normal stress due to stretch'},
                      'sy': {'index': 1, 'tittle': 'Maximum Shear stress due to bending in local y-direction'},
                      'sz': {'index': 2, 'tittle': 'Maximum Shear stress due to bending in local z-direction'},
                      'st': {'index': 3, 'tittle': 'Maximum shear stress due to torsion'},
                      'ny': {'index': 4, 'tittle': 'Maximum Normal stress due to bending in local y-direction'},
                      'nz': {'index': 5, 'tittle': 'Maximum Normal stress due to bending in local z-direction'},
                      'total': {'index': 6, 'tittle': 'Maximum von Misses Stress'}}

        force_key = {'fx': {'index': 0, 'tittle': 'Strain force - local x direction'},
                     'fy': {'index': 1, 'tittle': 'Shear force - local y direction'},
                     'fz': {'index': 2, 'tittle': 'Shear force - local z direction'},
                     'mx': {'index': 3, 'tittle': 'Torsion moment - local x axis'},
                     'my': {'index': 4, 'tittle': 'Bending moment - local y axis'},
                     'mz': {'index': 5, 'tittle': 'Bending moment - local z axis'}}

        if data_type == 'deformation':
            main_key = disp_key

            values = displacement_vector[:, main_key[option]['index'], :]

        elif data_type == 'stress':
            main_key = stress_key

            if main_key[option]['index'] != 6:
                values, _ = self._model.get_elements_stress_force(resolution, index=index)
                values = values[:, main_key[option]['index'], :]
            else:
                logger.warning("""Von Mises stress for torsion of non-circular cross section gives invalid results!""")
                values = self._model.get_vMs(resolution, index)

        else:
            main_key = force_key

            _, values = self._model.get_elements_stress_force(resolution, index=index)
            values = values[:, main_key[option]['index'], :]

        tittle = main_key[option]['tittle']

        return displacement_vector, points_vector, values, tittle

    def _get_plane_index(self, plane):
        # returns indexes that are used to plot results in matplotlib 2d graphs and axis titles
        plane = plane.lower()
        indexes = None
        x_axis = None
        y_axis = None

        if plane == 'xy' or plane == 'yx':
            indexes = (0, 1)
            x_axis = 'X axis'
            y_axis = 'Y axis'
        elif plane == 'yz' or plane == 'zy':
            indexes = (1, 2)
            x_axis = 'Y axis'
            y_axis = 'Z axis'
        elif plane == 'xz' or plane == 'zx':
            indexes = (0, 2)
            x_axis = 'X axis'
            y_axis = 'Z axis'

        return indexes, x_axis, y_axis

    def _check_fig_input(self, data_type, option, resolution, scale=None, plane=None, show_undeformed=None,
                         show_points=None, show_nodes=None, cursor=None):
        if type(option) != str:
            raise TypeError('argument - option must be str')

        option = option.lower()

        if data_type == 'deformation':
            if option not in ('ux', 'uy', 'uz', 'rotx', 'roty', 'rotz', 'total_disp', 'total_rot'):
                raise ValueError("""argument - must take one of the following values:
                 'ux', 'uy', 'uz', 'rotx', 'roty', 'rotz', 'total_disp', 'total_rot'""")

        elif data_type == 'stress':
            if option not in ('nx', 'sy', 'sz', 'st', 'ny', 'nz', 'total'):
                raise ValueError("""argument - must take one of the following values:
                 'nx', 'sy', 'sz', 'st', 'ny', 'nz', 'total""")

        elif data_type == 'force':
            if option not in ('fx', 'fy', 'fz', 'mx', 'my', 'mz'):
                raise ValueError("""argument - must take one of the following values:
                        'fx', 'fy', 'fz', 'mx', 'my', 'mz'""")

        if type(resolution) != str and type(resolution) != int:
            raise TypeError("argument - resolution must be INT or 'auto'")
        elif type(resolution) == str and resolution != 'auto':
            raise ValueError("argument - resolution must be INT or 'auto'")
        elif type(resolution) == int and resolution < 2:
            raise ValueError("argument - resolution must be greater or equal to 3")

        if scale is not None:
            if type(scale) != str and type(scale) != float:
                raise TypeError("argument - scale must be FLOAT or 'auto'")
            elif type(scale) == str and scale != 'auto':
                raise ValueError("argument - scale must be FLOAT or 'auto'")
            elif type(scale) == float and scale <= 0:
                raise ValueError("argument scale must be greater than 0")

        if plane is not None:
            if type(plane) != str:
                raise TypeError("argument - plane must be STR")
            else:
                plane = plane.lower()
                if plane not in ('xy', 'yx', 'yz', 'zy', 'xz', 'zx'):
                    raise ValueError("""argument - plane must take one of the following values:
                     'xy', 'yx', 'yz', 'zy', 'xz', 'zx'""")

        if show_undeformed is not None:
            if type(show_undeformed) != bool:
                raise TypeError("argument - show_undeformed must be BOOL")

        if show_points is not None:
            if type(show_points) != bool:
                raise TypeError("argument - show_points must be BOOL")

        if show_nodes is not None:
            if type(show_nodes) != bool:
                raise TypeError("argument - show_nodes must be BOOL")

        if cursor is not None:
            if type(cursor) != bool:
                raise TypeError("argument - cursor must be BOOL type")

    def _basic_2d_results(self, option, scale, plane, show_undeformed, show_points, show_nodes, resolution, data_type,
                          cursor):
        # mother function for deformation, stress and forces graphs for matplotlib basic plot
        self._check_fig_input(data_type=data_type, option=option, scale=scale, plane=plane, resolution=resolution,
                              show_undeformed=show_undeformed, show_points=show_points, show_nodes=show_nodes,
                              cursor=cursor)

        resolution = self._get_resolution(resolution, option)
        indexes, x_axis, y_axis = self._get_plane_index(plane)

        disp_vector, points_vector, value, title = self._get_data(option, resolution, data_type)

        points_key = self._model.get_points()

        fig_obj = Graph2d(disp_vector, points_vector, value, scale, indexes, points_key, cursor)

        fig, ax = fig_obj.get_fig(show_points=show_points, show_nodes=show_nodes, show_undeformed=show_undeformed)

        ax.set_title(title)
        ax.set_ylabel(y_axis)
        ax.set_xlabel(x_axis)

        self._mpl_figs.append(fig)

        return fig

    def _basic_3d_results(self, option, scale, show_undeformed, show_points, show_nodes, resolution, data_type):
        self._check_fig_input(data_type=data_type, option=option, resolution=resolution, scale=scale,
                              show_undeformed=show_undeformed, show_points=show_points, show_nodes=show_nodes)
        # mother function for deformation, stress and forces graphs for plotly basic plot
        resolution = self._get_resolution(resolution, option)
        disp_vector, points_vector, value, title = self._get_data(option, resolution, data_type)

        points_key = self._model.get_points()

        fig_obj = Graph3d(disp_vector, points_vector, value, scale, points_key)

        fig = fig_obj.get_fig(show_points=show_points, show_nodes=show_nodes, show_undeformed=show_undeformed)
        fig.update_layout(title=title)

        self._plotly_figs.append(fig)

        return fig

    def deformation_2d(self, option: str, scale: float = 'auto', plane: str = 'xy', show_undeformed: bool = False,
                       show_points: bool = False, show_nodes: bool = False, resolution: int = 'auto',
                       cursor: bool = False) -> plt.figure:

        return self._basic_2d_results(option, scale, plane, show_undeformed, show_points, show_nodes, resolution,
                                      'deformation', cursor)

    def deformation_3d(self, option: str, scale: float = 'auto', show_undeformed: bool = False,
                       show_points: bool = False, show_nodes: bool = False, resolution: int = 'auto') -> go.Figure:

        return self._basic_3d_results(option, scale, show_undeformed, show_points, show_nodes, resolution,
                                      'deformation')

    def stress_2d(self, option: str, scale: float = 'auto', plane: str = 'xy', show_undeformed: bool = False,
                  show_points: bool = False, show_nodes: bool = False, resolution: int = 'auto',
                  cursor: bool = False) -> plt.figure:

        return self._basic_2d_results(option, scale, plane, show_undeformed, show_points, show_nodes, resolution,
                                      'stress', cursor)

    def stress_3d(self, option: str, scale: float = 'auto', show_undeformed: bool = False,
                  show_points: bool = False, show_nodes: bool = False, resolution: int = 'auto') -> go.Figure:

        return self._basic_3d_results(option, scale, show_undeformed, show_points, show_nodes, resolution, 'stress')

    def force_2d(self, option: str, scale: float = 'auto', plane: str = 'xy', show_undeformed: bool = False,
                 show_points: bool = False, show_nodes: bool = False, resolution: int = 'auto',
                 cursor: bool = False) -> plt.figure:

        return self._basic_2d_results(option, scale, plane, show_undeformed, show_points, show_nodes, resolution,
                                      'force', cursor)

    def force_3d(self, option: str, scale: float = 'auto', show_undeformed: bool = False,
                 show_points: bool = False, show_nodes: bool = False, resolution: int = 'auto') -> go.Figure:

        return self._basic_3d_results(option, scale, show_undeformed, show_points, show_nodes, resolution, 'force')

    def _basic_2d_bar_results(self, option, plane, resolution, data_type, cursor):
        # mother function for deformation, stress and forces graphs for matplotlib bar plot
        self._check_fig_input(data_type=data_type, option=option, resolution=resolution, plane=plane, cursor=cursor)

        resolution = self._get_resolution(resolution, option)

        indexes, x_axis, y_axis = self._get_plane_index(plane)

        disp_vector, points_vector, value, title = self._get_data(option, resolution, data_type)

        fig_obj = Bar_2d(value, points_vector, indexes, cursor)

        fig, ax = fig_obj.get_fig()

        ax.set_title(title)
        ax.set_ylabel(y_axis)
        ax.set_xlabel(x_axis)

        self._mpl_figs.append(fig)

        return fig

    def _basic_3d_bar_results(self, option, resolution, data_type):
        # mother function for deformation, stress and forces graphs for plotly bar plot
        self._check_fig_input(data_type=data_type, option=option, resolution=resolution)

        resolution = self._get_resolution(resolution, option)

        disp_vector, points_vector, value, title = self._get_data(option, resolution, data_type)
        points_key = self._model.get_points()
        fig_obj = Bar_3d(value, points_vector, self._model.elements, points_key)

        fig = fig_obj.get_fig()
        fig.update_layout(title=title)

        self._plotly_figs.append(fig)

        return fig

    def bar_deformation_2d(self, option: str, plane: str = 'xy', resolution: int = 'auto', cursor: bool = False)\
            -> plt.figure:
        return self._basic_2d_bar_results(option, plane, resolution, 'deformation', cursor)

    def bar_deformation_3d(self, option: str, resolution: int = 'auto') -> go.Figure:
        return self._basic_3d_bar_results(option, resolution, 'deformation')

    def bar_stress_2d(self, option: str, plane: str = 'xy', resolution: int = 'auto', cursor: bool = False)\
            -> plt.figure:
        return self._basic_2d_bar_results(option, plane, resolution, 'stress', cursor)

    def bar_stress_3d(self, option: str, resolution: int = 'auto') -> go.Figure:
        return self._basic_3d_bar_results(option, resolution, 'stress')

    def bar_force_2d(self, option: str, plane: str = 'xy', resolution: int = 'auto', cursor: bool = False)\
            -> plt.figure:
        return self._basic_2d_bar_results(option, plane, resolution, 'force', cursor)

    def bar_force_3d(self, option: str, resolution: int = 'auto') -> go.Figure:
        return self._basic_3d_bar_results(option, resolution, 'force')

    def evaluate_all_results(self):
        # shows all returned figures in separated windows
        app = QApplication(sys.argv)  # qt5 application
        index = 1
        windows = []
        for fig in self._plotly_figs:
            # save plotly figure in temp directory
            file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'temp/fig{index}.html'))
            plotly.offline.plot(fig, filename=file_path, auto_open=False)
            web = QWebEngineView()  # set web browser to qt application
            web.load(QUrl.fromLocalFile(file_path))  # load saved figure to application
            web.setWindowTitle('Results graph')
            web.setGeometry(200, 200, 800, 600)
            windows.append(web)
            index += 1
        for window in windows:
            window.show()  # show all plotly figures

        plt.show()  # show all matplotlib figures
        if len(self._plotly_figs) > 0:
            app.exec_()
        for i in range(1, index):
            file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'temp/fig{i}.html'))
            os.remove(file_path)  # remove all saved figures from temp directory after close all windows

        if len(self._mpl_figs) > 0:
            return None
        else:
            sys.exit()  # shutdown by sys if user created any plotly figures

    def _check_max_value_input(self, option, lines, data_type):
        deformation_key = ('ux', 'uy', 'uz', 'rotx', 'roty', 'rotz', 'total_disp', 'total_rot')
        stress_key = ('nx', 'sy', 'sz', 'st', 'ny', 'nz')
        force_key = ('fx', 'fy', 'fz', 'mx', 'my', 'mz')

        if data_type == 'deformation':
            main_key = deformation_key
        elif data_type == 'stress':
            main_key = stress_key
        else:
            main_key = force_key

        if type(option) != str:
            raise TypeError('argument - option must be a STRING')
        else:
            option = option.lower()
            if option not in main_key:
                raise ValueError(f"argument - option must take one of the following values:{main_key}")

        if lines is not None and type(lines) != list and type(lines) != np.ndarray:
            raise TypeError("argument - lines must be LIST or None")
        elif lines is not None:
            for element in lines:
                if type(element) != Line:
                    raise TypeError('elements of the lines must be Line')
                elif len(element.elements_index) == 0:
                    raise AttributeError('one or more of passed lines are not meshed')

    def _max_value(self, option, lines, data_type):
        # mother function for max value of deformation, stress or force. Returns max value for defined list lines.
        # if lines are not defined, function returns max value for whole system
        self._check_max_value_input(option, lines, data_type)
        option = option.lower()

        disp_key = {'ux': 0, 'uy': 1, 'uz': 2, 'rotx': 3, 'roty': 4, 'rotz': 5, 'total_disp': 6, 'total_rot': 7}
        stress_key = {'nx': 0, 'sy': 1, 'sz': 2, 'st': 3, 'ny': 4, 'nz': 5}
        force_key = {'fx': 0, 'fy': 1, 'fz': 2, 'mx': 3, 'my': 4, 'mz': 5}

        res = self._get_resolution('auto')

        if lines is None:
            lines = self._model.mesh.lines

        if data_type == 'deformation':
            main_key = disp_key
            max_value = self._model.elements[lines[0].elements_index[0]].get_max_displacements(res)[main_key[option]]
        elif data_type == 'stress':
            main_key = stress_key
            max_value = self._model.elements[lines[0].elements_index[0]].get_max_stress(res)[main_key[option]]
        else:
            main_key = force_key
            max_value = self._model.elements[lines[0].elements_index[0]].get_max_force(res)[main_key[option]]

        for line in lines:
            for index in line.elements_index:
                element = self._model.elements[index]

                if data_type == 'deformation':
                    value = element.get_max_displacements(2)[main_key[option]]
                elif data_type == 'stress':
                    value = element.get_max_stress(2)[main_key[option]]
                else:
                    value = element.get_max_force(2)[main_key[option]]

                if value > max_value:
                    max_value = value

        return max_value[0]

    def max_displacements(self, option: str, lines: list = None) -> float:
        return self._max_value(option, lines, 'deformation')

    def max_stress(self, option: str, lines: list = None) -> float:
        return self._max_value(option, lines, 'stress')

    def max_force(self, option: str, lines: list = None) -> float:
        return self._max_value(option, lines, 'force')

    def residuals_at_bc_points(self) -> dict:
        # returns residuals forces and moments for points, where displacement boundary conditions were defined
        output = {}
        resolution = self._get_resolution('auto')
        for point in self._model.displacement_points:
            node_index = point.node_number - 1
            node = self._model.nodes[node_index]
            total_force = np.zeros((6))
            for index in node.elements_index:
                id = -1
                element = self._model.elements[index]
                if element.node1.index == node.index:
                    id = 0

                _, force_vector = element.get_stress_force_vector(resolution)
                force_vector = force_vector[:, id]

                t_matrix = np.linalg.inv(element.t_matrix[0:6, 0:6])
                local_force = np.dot(t_matrix, force_vector)

                total_force -= local_force

            output[f"Point-index: {point.index}-node_number: {point.node_number}"] = {'Fx': total_force[0],
                                                                                      'Fy': total_force[1],
                                                                                      'Fz': total_force[2],
                                                                                      'Mx': total_force[3],
                                                                                      'My': total_force[4],
                                                                                      'Mz': total_force[5]}
        return output

    def section_stress(self, option: str, line: Line, length: float, resolution: int = 'auto') -> plt.figure:

        # check input
        stress_key = ('nx', 'sy', 'sz', 'st', 'ny', 'nz', 'total')
        if type(option) != str:
            raise TypeError('argument - option must be a STRING')
        else:
            option = option.lower()
            if option not in stress_key:
                raise ValueError(f"argument - option must take one of the following values:{stress_key}")

        if type(line) != Line:
            raise TypeError("argument - line must be Line")

        if type(length) != float and type(length) != int:
            raise TypeError("argument - length must be a FLOAT or INT")
        elif length > line.len or length < 0:
            raise ValueError(f"argument - length must grater than 0 and less than Line's length")

        stress_key = {'nx': {'index': 0, 'tittle': 'Normal stress due to stretch'},
                      'sy': {'index': 1, 'tittle': 'Shear stress due to bending in local y-direction'},
                      'sz': {'index': 2, 'tittle': 'Shear stress due to bending in local z-direction'},
                      'st': {'index': 3, 'tittle': 'Shear stress due to torsion'},
                      'ny': {'index': 4, 'tittle': 'Normal stress due to bending in local y-direction'},
                      'nz': {'index': 5, 'tittle': 'Normal stress due to bending in local z-direction'},
                      'total': {'index': 6, 'tittle': 'Von Misses Stress'}}
        if type(resolution) != str and type(resolution) != int:
            raise TypeError("argument - resolution must be INT or 'auto'")
        elif type(resolution) == str and resolution != 'auto':
            raise ValueError("argument - resolution must be INT or 'auto'")
        elif type(resolution) == int and resolution < 2:
            raise ValueError("argument - resolution must be greater or equal to 3")

        resolution = self._get_resolution(resolution)

        in_length = 0
        element = None

        for i in line.elements_index:
            element = self._model.elements[i]
            in_length += element.L
            if in_length >= length:
                break
        out_length = in_length - length
        title = stress_key[option]['tittle']
        values = element.get_section_stresses(out_length, resolution)

        if option == 'total' and not element.section.circular and np.amax(values[3]**2) > 0.01:
            logger.warning("""Von Mises stress for torsion of non-circular cross section gives invalid results!""")

        values = values[stress_key[option]['index']]
        graph = Section_graph(element, values, line, length)

        fig, ax = graph.get_fig()
        ax.set_title(title)

        self._mpl_figs.append(fig)

        return fig

