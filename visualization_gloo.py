import numpy as np
from vispy import app, gloo, visuals
from vispy.util import ptime


# Define a simple vertex shader. Template position and transform.
vertex_shader = """
void main() {
   gl_Position = $transform(vec4($position, 0, 1));
}
"""

# Very simple fragment shader. Template color.
fragment_shader = """
void main() {
  gl_FragColor = $color;
}
"""


class BoatCanvas(app.Canvas):
    """

    """
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(1600, 800))