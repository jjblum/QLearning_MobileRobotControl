import numpy as np
import scipy.integrate as spi
import Boat
import Strategies
import Designs
from visualization_scene import BoatVisual, format_time_string
from vispy import scene, visuals, app
from vispy.util import ptime


TIME_DILATION = 20.0  # the number of seconds that pass in the program for every real-time second
FAILED_WAYPOINT_TIMEOUT = 30.0  # number of seconds before abandoning a waypoint


BoatNode = scene.visuals.create_visual_node(BoatVisual)
TextNode = scene.visuals.create_visual_node(visuals.TextVisual)

# Create a canvas to display our visual
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 800
ARENA_WIDTH = 800
ARENA_HEIGHT = WINDOW_HEIGHT
DATA_WIDTH = WINDOW_WIDTH - ARENA_WIDTH
DATA_HEIGHT = WINDOW_HEIGHT
ARENA_CENTER = (ARENA_WIDTH/2., ARENA_HEIGHT/2.)
ARENA_EDGE_SIZE = 100.0

# remember 0, 0 is upper left in pixel coordinates, (pixel_width, pixel_height) is the lower right in pixel coordinates
# In real coordinates 0, 0 is the center, negatives are to the left and down
CANVAS = scene.SceneCanvas(keys='interactive', show=True, size=(WINDOW_WIDTH, WINDOW_HEIGHT))
ARENA_VIEW = scene.widgets.ViewBox(parent=CANVAS.scene, name="arena_view", margin=0, bgcolor=(1, 1, 1, 1), size=(ARENA_WIDTH, ARENA_HEIGHT), pos=(0, 0))
DATA_VIEW = scene.widgets.ViewBox(parent=CANVAS.scene, name="data_view", margin=0, bgcolor=(0.8, 0.8, 0.8, 1), size=(DATA_WIDTH, DATA_HEIGHT), pos=(ARENA_WIDTH, 0))

# Create two instances of the visual, each using canvas.scene as their parent
BOAT_VISUALS = [BoatNode(ARENA_CENTER[0], ARENA_CENTER[1], 0, 20, 40, (0, .6, .6, 1), parent=CANVAS.scene),
                BoatNode(ARENA_CENTER[0], ARENA_CENTER[1], 0, 20, 40, (.6, 0, 0, 1), parent=CANVAS.scene)]
TEXT_BOXES = {"time": TextNode("t = ", pos=(ARENA_WIDTH + 100, 30), parent=CANVAS.scene, bold=True, font_size=30),
              "waypoint_symbol": TextNode("+", pos=(0, 0), parent=CANVAS.scene, bold=True, font_size=40),
              "waypoint": TextNode("[]", pos=(ARENA_WIDTH + 100, 70), parent=CANVAS.scene, bold=True, font_size=30)}

BOATS = [Boat.Boat(), Boat.Boat()]


def iterate(event):  # event is unused
    global FIRST_TIME, LAST_TIME, BOATS, CANVAS, TIME_DILATION, LAST_COMPLETED_WP_TIME, FAILED_WAYPOINT_TIMEOUT
    current_time = TIME_DILATION*(ptime.time() - FIRST_TIME)
    TEXT_BOXES["time"].text = "t = {}".format(format_time_string(current_time, 2))
    # USE ODE TO PROPAGATE BOAT STATE
    times = np.linspace(LAST_TIME, current_time, 100)
    for i in range(len(BOATS)):
        boat = BOATS[i]
        boat.control()
        boat.time = current_time
        states = spi.odeint(Boat.ode, boat.state, times, (boat,))
        boat.state = states[-1]
        px, py = xy_location_to_pixel_location(states[-1][0], states[-1][1])
        heading = Boat.wrapTo2Pi(states[-1][4])
        BOAT_VISUALS[i].new_pose(px, py, heading)
        if boat.strategy.finished or current_time - LAST_COMPLETED_WP_TIME > FAILED_WAYPOINT_TIMEOUT:
            LAST_COMPLETED_WP_TIME = current_time
            waypoint = np.random.uniform(-ARENA_EDGE_SIZE/2., ARENA_EDGE_SIZE/2., size=[2, ])
            px, py = xy_location_to_pixel_location(waypoint[0], waypoint[1])
            TEXT_BOXES["waypoint_symbol"].pos = (px, py)
            TEXT_BOXES["waypoint"].text = "[{:.0f}, {:.0f}]".format(px, py)
            boat.strategy = Strategies.DestinationOnly(boat, waypoint)
    LAST_TIME = current_time
    CANVAS.update()


FIRST_TIME = ptime.time()
LAST_TIME = 0
LAST_COMPLETED_WP_TIME = 0
GLOBAL_TIMER = app.Timer('auto', connect=iterate, start=True)


def xy_location_to_pixel_location(x, y):
    global ARENA_WIDTH, ARENA_HEIGHT, ARENA_EDGE_SIZE
    px, py = x*ARENA_WIDTH/ARENA_EDGE_SIZE + ARENA_WIDTH/2., -1*y*ARENA_HEIGHT/ARENA_EDGE_SIZE + ARENA_HEIGHT/2.
    # print "{},{}  -->  {},{}".format(x, y, px, py)
    return px, py


def setup():
    global BOATS, ARENA_EDGE_SIZE
    pid_boat = BOATS[0]
    pid_boat.design = Designs.TankDriveDesign()
    pid_boat.time = 0
    pid_boat.state = np.zeros((6,))
    pid_boat.plotData = None

    q_boat = BOATS[1]
    q_boat.design = Designs.TankDriveDesign()
    q_boat.time = 0
    q_boat.state = np.zeros((6,))
    q_boat.plotData = None

    waypoint = np.random.uniform(-ARENA_EDGE_SIZE/2., ARENA_EDGE_SIZE/2., size=[2, ])
    px, py = xy_location_to_pixel_location(waypoint[0], waypoint[1])
    TEXT_BOXES["waypoint_symbol"].pos = (px, py)
    TEXT_BOXES["waypoint"].text = "[{:.0f}, {:.0f}]".format(px, py)
    pid_boat.strategy = Strategies.DestinationOnly(pid_boat, waypoint)
    q_boat.strategy = Strategies.DestinationOnly(q_boat, waypoint, controller_name="QLearnPointAndShoot")


if __name__ == "__main__":
    setup()
    app.run()
