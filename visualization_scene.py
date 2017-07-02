import numpy as np
from vispy import app, gloo, visuals, scene
# from vispy.visuals.transforms import *
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


def format_time_string(time, decimals):
    time = np.round(time, decimals)
    if decimals == 0:
        return "{}".format(int(time))
    fraction = time - np.floor(time)
    fraction_as_integer = "{}".format(int(np.power(10, decimals)*fraction))
    return "{}.{}".format(int(time), fraction_as_integer)


class BoatCanvas(app.Canvas):
    """

    """
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(800, 800))


class TimeTextVisual(visuals.TextVisual):
    """
    Display the string "t = X" where X is the time in seconds.
    """
    def __init__(self, text, color='black', bold=False, italic=False, face='OpenSans', font_size=12, pos=[0, 0, 0], rotation=0.0, anchor_x='center', anchor_y='center', font_manager=None):
        super(TimeTextVisual, self).__init__(text, color, bold, italic, face, font_size, pos, rotation, anchor_x, anchor_y, font_manager)
        self.unfreeze()  # super class froze things. Need to unfreeze.
        visuals.Visual.__init__(self, vertex_shader, fragment_shader)
        self.timer = app.Timer(interval='auto', connect=self.update_time, start=False)
        self._time = 0.0
        self._first_time = ptime.time()
        self._last_time = ptime.time()
        self.text = text
        self.freeze()
        self.shared_program['time'] = self._time
        self.shared_program['text_scale'] = 1
        self.timer.start()

    def update_time(self, ev):  # argument ev is required for scene, but doesn't have to be used
        t = ptime.time()
        self._time += t - self._last_time
        self._last_time = t
        self.shared_program['time'] = self._time
        x = t - self._first_time
        self.text = "t = {%.4f}".format(x)
        self.update()


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
        self._draw_mode = 'triangles'
        self.freeze()  # no more attributes
        self.shared_program.vert['position'] = self.vbo
        self.shared_program.frag['color'] = tuple(rgba)

    def _prepare_transforms(self, view):
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
        vertices = np.array(vertices[:, :-1], dtype=np.float32)
        # apparently the above transpose makes the array discontiguous. Need to fix to stop warnings from firing constantly.
        # We accomplish this by using reshape into a vector, then reshape back into the original shape
        vertices = np.reshape(vertices, vertices.size)
        vertices = np.reshape(vertices, (3, 2))
        # print vertices
        self.vbo = gloo.VertexBuffer(vertices)
        self.shared_program.vert['position'] = self.vbo


"""
# set up a Canvas and TransformSystem for drawing
# An easier approach is to make the visual usable in a scenegraph, in which
# case the canvas will take care of drawing the visual and setting up the
# TransformSystem for us.
# To be able to use our new Visual in a scenegraph, it needs to be
# a subclass of scene.Node. In vispy we achieve this by creating a parallel
# set of classes that inherit from both Node and each Visual subclass.
# This can be done automatically using scene.visuals.create_visual_node():
BoatNode = scene.visuals.create_visual_node(BoatVisual)
TextNode = scene.visuals.create_visual_node(visuals.TextVisual)

# Create a canvas to display our visual
window_width = 1600
window_height = 800
arena_width = 800
arena_height = window_height
data_width = window_width - arena_width
data_height = window_height
real_width = 2*30
real_height = 2*30
# remember 0, 0 is upper left in pixel coordinates, (pixel_width, pixel_height) is the lower right in pixel coordinates
# In real coordinates 0, 0 is the center, negatives are to the left and down
canvas = scene.SceneCanvas(keys='interactive', show=True, size=(window_width, window_height))
view_1 = scene.widgets.ViewBox(parent=canvas.scene, name="arena_view", margin=0, bgcolor=(1, 1, 1, 1), size=(arena_width, arena_height), pos=(0, 0))
view_2 = scene.widgets.ViewBox(parent=canvas.scene, name="data_view", margin=0, bgcolor=(0.8, 0.8, 0.8, 1), size=(data_width, data_height), pos=(arena_width, 0))


# Create two instances of the visual, each using canvas.scene as their parent
boats = [BoatNode(10.*arena_width/real_width + arena_width/2.0, -1*0.*arena_height/real_height + arena_height/2.0, 0, 20, 40, (0, .6, .6, 1), parent=canvas.scene),
         BoatNode(-28.*arena_width/real_width + arena_width/2.0, -1*-29.*arena_height/real_height + arena_height/2.0, 0, 20, 40, (.6, 0, 0, 1), parent=canvas.scene)]

textboxes = [TextNode("t = ", pos=(arena_width+100, 30), parent=canvas.scene, bold=True, font_size=30)]

boats[1].new_pose(100, 100, 45*np.pi/180.)


# a general timer and the function that is called each tick
first_time = ptime.time()
last_time = 0
def iterate(event):
    global first_time
    global last_time
    current_time = ptime.time() - first_time
    dt = current_time - last_time
    last_time = current_time
    print dt
    textboxes[0].text = "t = {}".format(format_time_string(current_time, 2))
    x = np.round(100.*current_time, 4)
    boats[0].new_pose(x, x, np.pi*current_time)
    boats[1].new_pose(10*np.random.randn() + x, 10*np.random.randn() + x, np.random.uniform(0, 2 * np.pi, (1,)))
    canvas.update()

# create the general timer, runs a callback
overall_timer = app.Timer('auto', connect=iterate, start=True)

"""
if __name__ == '__main__':
    app.run()
