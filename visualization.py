import numpy as np
from vispy import app, gloo, visuals, scene
from vispy.visuals.transforms import *

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

# Visual class for a boat arrow
class BoatVisual(visuals.Visual):
    """
    Parameters
    ----------
    x : float
        x coordinate of origin
    y : float
        y coordinate of origin
    w : float
        width of arrow root
    l : float
        length of arrow

    origin is in the center 1/2 width, 1/2 length
    basic orientation is to the right -->
    """
    def __init__(self, x, y, th, w, l, rgba):
        # Initialize the visual with a vertex shader and fragment shader
        visuals.Visual.__init__(self, vertex_shader, fragment_shader)
        self._x = x
        self._y = y
        self._th = th
        self._w = w
        self._l = l

        # vertices for a triangle
        self.vbo = gloo.VertexBuffer(np.array([
            [x-0.5*l, y+0.5*w], [x-0.5*l, y-0.5*w], [x+0.5*l, y],
        ], dtype=np.float32))

        # Assign values to the $position and $color template variables in the shaders.
        self.shared_program.vert['position'] = self.vbo
        self.shared_program.frag['color'] = tuple(rgba)
        self._draw_mode = 'triangles'

    def _prepare_transforms(self, view):
        # This method is called when the user or the scenegraph has assigned
        # new transforms to this visual (ignore the *view* argument for now;
        # we'll get to that later). This method is thus responsible for
        # connecting the proper transform functions to the shader program.

        # The most common approach here is to simply take the complete
        # transformation from visual coordinates to render coordinates. Later
        # tutorials detail more complex transform handling.
        view.view_program.vert['transform'] = view.get_transform()

    def new_pose(self, new_x, new_y, new_th):
        self._x = new_x
        self._y = new_y
        self._th = new_th
        homogeneous_vertices = np.array(
            [
                [-0.5 * self._l, 0.5 * self._w, 1],
                [-0.5 * self._l, -0.5 * self._w, 1],
                [0.5 * self._l, 0.0, 1]
            ], dtype=np.float32)
        # print homogeneous_vertices[:, :-1]
        homogeneous_transform = np.array([[np.cos(new_th), np.sin(new_th), new_x], [np.sin(new_th), -np.cos(new_th), new_y], [0, 0, 1]])
        # print homogeneous_transform
        vertices = np.transpose(np.dot(homogeneous_transform, np.transpose(homogeneous_vertices)))
        # print vertices[:, :-1]
        self.vbo = gloo.VertexBuffer(np.array(vertices[:, :-1], dtype=np.float32))
        self.shared_program.vert['position'] = self.vbo


# set up a Canvas and TransformSystem for drawing
# An easier approach is to make the visual usable in a scenegraph, in which
# case the canvas will take care of drawing the visual and setting up the
# TransformSystem for us.
# To be able to use our new Visual in a scenegraph, it needs to be
# a subclass of scene.Node. In vispy we achieve this by creating a parallel
# set of classes that inherit from both Node and each Visual subclass.
# This can be done automatically using scene.visuals.create_visual_node():
BoatNode = scene.visuals.create_visual_node(BoatVisual)

# Create a canvas to display our visual
pixel_width = 800
pixel_height = 800
real_width = 2*30
real_height = 2*30
# remember 0, 0 is upper left in pixel coordinates, (pixel_width, pixel_height) is the lower right in pixel coordinates
# In real coordinates 0, 0 is the center, negatives are to the left and down
canvas = scene.SceneCanvas(keys='interactive', show=True, size=(pixel_width, pixel_height))
view_1 = scene.widgets.ViewBox(parent=canvas.scene, name="view_1", margin=0, bgcolor=(1, 1, 1, 1), size=(pixel_width, pixel_height), pos=(0, 0))


# Create two instances of the visual, each using canvas.scene as their parent
boats = [BoatNode(10.*pixel_width/real_width + pixel_width/2.0, -1*0.*pixel_height/real_height + pixel_height/2.0, 0, 20, 40, (0, .6, .6, 1), parent=canvas.scene),
         BoatNode(-28.*pixel_width/real_width + pixel_width/2.0, -1*-29.*pixel_height/real_height + pixel_height/2.0, 0, 20, 40, (.6, 0, 0, 1), parent=canvas.scene)]

boats[1].new_pose(100, 100, 45*np.pi/180.)

# ..and optionally start the event loop
if __name__ == '__main__':
    import sys

    if sys.flags.interactive != 1:
        app.run()