from bokeh import io
from bokeh import models
from bokeh import plotting


__all__ = ['Scatter']


class Scatter:

    def __init__(self, width=400, height=400, notebook=True):
        if notebook:
            io.output_notebook()

        self.fig = plotting.figure(plot_width=width, plot_height=height)
        self.source = models.ColumnDataSource(data=dict(x=[], y=[]))
        self.handle = io.show(self.fig, notebook_handle=notebook)
        self.scatter = self.fig.scatter('x', 'y', source=self.source)

    def update(self, x, y, label):

        # Add the label if it's never been seen before
        # if label not in self.source.data:
        #     self.source.add([], name=label)

        self.source.stream({'x': [x], 'y': [y]})
