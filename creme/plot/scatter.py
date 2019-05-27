import itertools

try:
    from bokeh import io
    from bokeh import models
    from bokeh import palettes
    from bokeh import plotting
    BOKEH_INSTALLED = True
except ImportError:
    BOKEH_INSTALLED = False


__all__ = ['Scatter']


class Scatter:
    """Live scatter plot based on ``bokeh``.

    .. warning::
        The API of this class is more than likely to change in the next version.

    Example:

        ::

            >> import functools
            >> from creme import cluster
            >> from creme import plot
            >> import numpy as np

            >> scatter = plot.Scatter(
            ..     width=600,
            ..     height=500,
            ..     x_lim=(-1.2, 1.8),
            ..     y_lim=(-1.2, 1.2)
            .. )

            >> n_points_per_cycle = 200
            >> n_cycles = 3
            >> n_clusters = 5

            >> k_means = cluster.KMeans(n_clusters=n_clusters)

            >> noise = functools.partial(np.random.uniform, -0.1, 0.1)

            >> for rad in np.tile(np.linspace(0, 2 * np.pi, n_points_per_cycle), reps=n_cycles):
            ..
            ..     for i in range(n_clusters):
            ..
            ..         p = {
            ..             'x': np.cos(rad + i * 2 * np.pi / n_clusters) + noise(),
            ..             'y': np.sin(rad + i * 2 * np.pi / n_clusters) + noise()
            ..         }
            ..
            ..         cluster = k_means.fit_predict_one(p)
            ..
            ..         scatter.update(x=p['x'], y=p['y'], label=f'Cluster {cluster}')

    """

    def __init__(self, max_points=100, palette=None, alpha=0.7, size=8, width=400, height=400,
                 x_lim=None, y_lim=None, notebook=True):
        if not BOKEH_INSTALLED:
            raise RuntimeError('bokeh is not installed')

        if notebook:
            io.output_notebook()

        if palette is None:
            palette = palettes.Category10[10]

        self.max_points = max_points
        self.palette = itertools.cycle(palette)
        self.alpha = alpha
        self.size = size
        self.notebook = notebook

        # Add figure
        self.fig = plotting.figure(plot_width=width, plot_height=height)
        self.fig.xaxis.axis_label = 'x'
        self.fig.yaxis.axis_label = 'y'

        # Add limits
        if x_lim is not None:
            self.fig.x_range = models.Range1d(x_lim[0], x_lim[1])
        if y_lim is not None:
            self.fig.y_range = models.Range1d(y_lim[0], y_lim[1])

        self.source = models.ColumnDataSource(data={'x': [], 'y': [], 'label': []})

        self.views = {}
        self.scatters = {}

    def add_group(self, label):
        self.views[label] = models.CDSView(
            source=self.source,
            filters=[models.GroupFilter(column_name='label', group=label)]
        )
        self.scatters[label] = self.fig.scatter(
            x='x',
            y='y',
            alpha=self.alpha,
            size=self.size,
            source=self.source,
            view=self.views[label],
            color=next(self.palette),
            legend=label
        )
        if len(self.views) == 1:
            self.handle = io.show(self.fig, notebook_handle=self.notebook)

    def update(self, x, y, label):

        if label not in self.views:
            self.add_group(label=label)

        self.source.stream({'x': [x], 'y': [y], 'label': [label]}, rollover=self.max_points)

        io.push_notebook(handle=self.handle)

        return self
