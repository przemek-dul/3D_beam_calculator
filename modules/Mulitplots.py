import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State


class Multi_graph:
    def __init__(self, results):
        self.results = results
        
        self.app = dash.Dash("Results")
        self.app.layout = html.Div([
            html.Div([

                # Plot controls forms
                html.Div([
                    html.H5("Plot Options"),
                    html.Label("Graph type:"),
                    dcc.RadioItems(
                        id='fig-type',
                        options=[
                            {'label': 'Linear', 'value': 'linear'},
                            {'label': 'Bar', 'value': 'bar'},
                        ],
                        value='linear',
                        labelStyle={'display': 'block'},
                        style={'margin-bottom': '20px'}
                    ),
                    html.Label("Results to plot:"),
                    dcc.RadioItems(
                        id='vals_to_plot',
                        options=[
                            {'label': 'Ux displacement', 'value': 'ux.d'},
                            {'label': 'Uy displacement', 'value': 'uy.d'},
                            {'label': 'Uz displacement', 'value': 'uz.d'},
                            {'label': 'Rot_x angular displacement', 'value': 'rotx.d'},
                            {'label': 'Rot_y angular displacement', 'value': 'roty.d'},
                            {'label': 'Rot_z angular displacement', 'value': 'rotz.d'},
                            {'label': 'Total displacement', 'value': 'total_disp.d'},
                            {'label': 'Total angular displacement', 'value': 'total_rot.d'},
                            {'label': 'Normal stress due to stretch', 'value': 'nx.s'},
                            {'label': 'Maximum Shear stress due to bending in local y-direction', 'value': 'sy.s'},
                            {'label': 'Maximum Shear stress due to bending in local z-direction', 'value': 'sz.s'},
                            {'label': 'Maximum shear stress due to torsion', 'value': 'st.s'},
                            {'label': 'Maximum Normal stress due to bending in local y-direction', 'value': 'ny.s'},
                            {'label': 'Maximum Normal stress due to bending in local z-direction', 'value': 'nz.s'},
                            {'label': 'Maximum von Misses Stress', 'value': 'total.s'},
                            {'label': 'Strain force - local x direction', 'value': 'fx.f'},
                            {'label': 'Shear force - local y direction', 'value': 'fy.f'},
                            {'label': 'Shear force - local z direction', 'value': 'fz.f'},
                            {'label': 'Torsion moment - local x axis', 'value': 'mx.f'},
                            {'label': 'Bending moment - local y axis', 'value': 'my.f'},
                            {'label': 'Bending moment - local z axis', 'value': 'mz.f'},
                        ],
                        value='uy.d',
                        labelStyle={'display': 'block'},
                        style={'margin-bottom': '20px'}
                    ),

                    html.Br(),

                    html.Label("Display options:"),
                    dcc.Checklist(
                        id='disp_opt',
                        options=[
                            {'label': 'show undeformed', 'value': 'v1'},
                            {'label': 'show nodes', 'value': 'v2'},
                            {'label': 'show points', 'value': 'v3'}
                        ],
                        labelStyle={'display': 'block'},
                        value=[]
                    ),

                    html.Br(),

                    html.Label("Scale option:"),
                    dcc.RadioItems(
                        id='scale-option',
                        options=[
                            {'label': 'Auto Scale', 'value': 'auto'},
                            {'label': 'Custom Scale', 'value': 'custom'}
                        ],
                        value='auto',
                        labelStyle={'display': 'block'}
                    ),

                    # Input field for custom scale (disabled by default if Auto Scale is selected)
                    html.Label("Custom Scale:"),
                    dcc.Input(
                        id='custom-scale-input',
                        type='number',
                        value=1,
                        min=0.1,
                        step=0.1,
                        disabled=True,
                        style={'margin-bottom': '20px'}
                    ),

                    html.Br(),

                    # Resolution Options
                    html.Label("Resolution option:"),
                    dcc.RadioItems(
                        id='resolution-option',
                        options=[
                            {'label': 'Auto Resolution', 'value': 'auto'},
                            {'label': 'Custom Resolution', 'value': 'custom'}
                        ],
                        value='auto',  # Default value
                        labelStyle={'display': 'block'}
                    ),

                    # Input field for custom resolution (disabled by default if Auto Resolution is selected)
                    html.Label("Custom Resolution:"),
                    dcc.Input(
                        id='custom-resolution-input',
                        type='number',
                        value=3,
                        min=3,
                        step=1,
                        disabled=True,  # Initially disabled
                        style={'margin-bottom': '20px'}
                    ),

                    html.Br(),

                    # Replot button
                    html.Button('Replot', id='replot-button', n_clicks=0,
                                style={
                                    'background-color': '#4CAF50',
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '10px 24px',
                                    'text-align': 'center',
                                    'font-size': '16px',
                                    'cursor': 'pointer',
                                    'border-radius': '5px'
                                }),

                ], style={
                    'width': '30%',
                    'float': 'left',
                    'padding': '20px',
                    'background-color': '#f9f9f9',
                    'border-right': '2px solid #d3d3d3',
                    'height': '100vh',
                    'box-sizing': 'border-box'
                }),

                html.Div([
                    dcc.Loading(
                        id="loading-spinner",
                        type="circle",
                        children=[
                            dcc.Graph(
                                id='graph',
                                style={'height': '80vh'},
                                config={'displayModeBar': True, 'responsive': True}
                            )
                        ],
                        fullscreen=False
                    )
                ], style={
                    'width': '75%',
                    'float': 'right',
                    'padding': '20px',
                    'box-sizing': 'border-box'
                })

            ], style={'display': 'flex', 'height': '80vh'})
        ])

        @self.app.callback(
            [Output('custom-scale-input', 'disabled'),
             Output('custom-resolution-input', 'disabled'),
             Output('disp_opt', 'value'),
             Output('scale-option', 'value')],

            [Input('scale-option', 'value'),
             Input('resolution-option', 'value'),
             Input('fig-type', 'value'),
             Input('disp_opt', 'value')]
        )
        def toggle_inputs(scale_option, resolution_option, fig_type, vals):
            scale_disabled = scale_option != 'custom'
            resolution_disabled = resolution_option != 'custom'
            if fig_type != 'linear':
                vals = []
                scale_option = 'auto'
                scale_disabled = True
            return scale_disabled, resolution_disabled, vals, scale_option

        @self.app.callback(
            Output('graph', 'figure'),
            [Input('replot-button', 'n_clicks')],
            [State('fig-type', 'value'),
             State('vals_to_plot', 'value'),
             State('disp_opt', 'value'),
             State('scale-option', 'value'),
             State('custom-scale-input', 'value'),
             State('resolution-option', 'value'),
             State('custom-resolution-input', 'value')]
        )
        def update_graph(_, fig_type, vals_to_plot, disp_opt, scale_option, scale_val, res_option, res_val,):

            show_undeformed = 'v1' in disp_opt
            show_nodes = 'v2' in disp_opt
            show_points = 'v3' in disp_opt
            scale = 'auto'
            res = 'auto'
            if scale_option == 'custom':
                scale = float(scale_val)
            if res_option == 'custom':
                res = int(res_val)
            values, values_type = vals_to_plot.split('.')
            if values_type == 'd':
                if fig_type == 'linear':
                    fig = self.results.deformation_3d(values, show_undeformed=show_undeformed, show_nodes=show_nodes,
                                                      show_points=show_points, scale=scale, resolution=res)
                else:
                    fig = self.results.bar_deformation_3d(values, resolution=res)
            elif values_type == 's':
                if fig_type == 'linear':
                    fig = self.results.stress_3d(values, show_undeformed=show_undeformed, show_nodes=show_nodes,
                                                      show_points=show_points, scale=scale, resolution=res)
                else:
                    fig = self.results.bar_stress_3d(values, resolution=res)
            else:
                if fig_type == 'linear':
                    fig = self.results.force_3d(values, show_undeformed=show_undeformed, show_nodes=show_nodes,
                                                      show_points=show_points, scale=scale, resolution=res)
                else:
                    fig = self.results.bar_force_3d(values, resolution=res)

            return fig

    def get_app(self):
        return self.app