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
WAYPOINTS_BEFORE_RESET = 10  # the number of waypoints attempted before the boats reset to the center. A "batch"

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
BOAT_VISUALS = {"pid": BoatNode(ARENA_CENTER[0], ARENA_CENTER[1], 0, 20, 40, (0, .6, .6, 1), parent=CANVAS.scene),
                "q": BoatNode(ARENA_CENTER[0], ARENA_CENTER[1], 0, 10, 30, (.6, 0, 0, 1), parent=CANVAS.scene)}
TEXT_BOXES = {"time": TextNode("t = ", pos=(ARENA_WIDTH + 100, 30), parent=CANVAS.scene, bold=True, font_size=30),
              "waypoint_symbol": {"pid": TextNode("+", pos=(0, 0), parent=CANVAS.scene, bold=True, font_size=40, color=(0, .6, .6, 1)),
                                  "q": TextNode("o", pos=(0, 0), parent=CANVAS.scene, bold=True, font_size=40, color=(.6, 0, 0, 1))},
              "waypoint_text": {"pid": TextNode("[]", pos=(ARENA_WIDTH + 100, 70), parent=CANVAS.scene, bold=True, font_size=30, color=(0, .6, .6)),
                                "q": TextNode("[]", pos=(ARENA_WIDTH + 300, 70), parent=CANVAS.scene, bold=True, font_size=30, color=(.6, 0, 0, 1))},
              "waypoint_count": {"pid": TextNode("#", pos=(ARENA_WIDTH + 100, 110), parent=CANVAS.scene, bold=True, font_size=30, color=(0, .6, .6)),
                                 "q": TextNode("#", pos=(ARENA_WIDTH + 300, 110), parent=CANVAS.scene, bold=True, font_size=30, color=(.6, 0, 0, 1))}}

BOATS = {"pid": Boat.Boat(),
         "q": Boat.Boat()}

WAYPOINTS_INDEX = {"pid": 0,
                   "q": 0}

CONTROLLERS = {"pid": "PointAndShoot",
               "q": "QLearnPointAndShoot"}

WAYPOINT_QUEUE = list()


def iterate(event):  # event is unused
    global FIRST_TIME, LAST_TIME, BOATS, CANVAS, TIME_DILATION, LAST_COMPLETED_WP_TIME, FAILED_WAYPOINT_TIMEOUT, WAYPOINTS_INDEX, CONTROLLERS, WAYPOINT_QUEUE
    global TEXT_BOXES
    current_time = TIME_DILATION*(ptime.time() - FIRST_TIME)
    TEXT_BOXES["time"].text = "t = {}".format(format_time_string(current_time, 2))
    # USE ODE TO PROPAGATE BOAT STATE
    times = np.linspace(LAST_TIME, current_time, 100)
    for k in BOATS:
        boat = BOATS[k]
        boat.control()
        boat.time = current_time
        states = spi.odeint(Boat.ode, boat.state, times, (boat,))
        boat.state = states[-1]
        px, py = xy_location_to_pixel_location(states[-1][0], states[-1][1])
        heading = Boat.wrapTo2Pi(states[-1][4])
        BOAT_VISUALS[k].new_pose(px, py, heading)
        if boat.strategy.finished or current_time - LAST_COMPLETED_WP_TIME[k] > FAILED_WAYPOINT_TIMEOUT:
            WAYPOINTS_INDEX[k] += 1
            LAST_COMPLETED_WP_TIME[k] = current_time
            if WAYPOINTS_INDEX[k] < len(WAYPOINT_QUEUE):
                waypoint = WAYPOINT_QUEUE[WAYPOINTS_INDEX[k]]
                px, py = xy_location_to_pixel_location(waypoint[0], waypoint[1])
                TEXT_BOXES["waypoint_symbol"][k].pos = (px, py)
                TEXT_BOXES["waypoint_text"][k].text = "[{:.0f}, {:.0f}]".format(px, py)
                TEXT_BOXES["waypoint_count"][k].text = "#{} of {}".format(WAYPOINTS_INDEX[k]+1, WAYPOINTS_BEFORE_RESET)
                boat.strategy = Strategies.DestinationOnly(boat, waypoint, controller_name=CONTROLLERS[k])
    if not WAYPOINTS_INDEX["pid"] < WAYPOINTS_BEFORE_RESET or not WAYPOINTS_INDEX["q"] < WAYPOINTS_BEFORE_RESET:
        reset_boats()
    else:
        LAST_TIME = current_time
    CANVAS.update()


FIRST_TIME = ptime.time()
LAST_TIME = 0
LAST_COMPLETED_WP_TIME = {"pid": 0,
                          "q": 0}
GLOBAL_TIMER = app.Timer('auto', connect=iterate, start=True)


def xy_location_to_pixel_location(x, y):
    global ARENA_WIDTH, ARENA_HEIGHT, ARENA_EDGE_SIZE
    px, py = x*ARENA_WIDTH/ARENA_EDGE_SIZE + ARENA_WIDTH/2., -1*y*ARENA_HEIGHT/ARENA_EDGE_SIZE + ARENA_HEIGHT/2.
    # print "{},{}  -->  {},{}".format(x, y, px, py)
    return px, py


# TODO: generate an actual random set each time. same seed produces exact same queue.
# TODO: need to reset last completed WP time too

def generate_random_waypoints_queue():
    global WAYPOINTS_BEFORE_RESET, WAYPOINT_QUEUE, ARENA_EDGE_SIZE
    WAYPOINT_QUEUE = list()
    for i in range(WAYPOINTS_BEFORE_RESET):
        waypoint = np.random.uniform(-ARENA_EDGE_SIZE/2., ARENA_EDGE_SIZE/2., size=[2, ])
        WAYPOINT_QUEUE.append(waypoint)


def reset_boats():
    global BOATS, CONTROLLERS, WAYPOINT_QUEUE, WAYPOINTS_INDEX, WAYPOINTS_BEFORE_RESET, LAST_COMPLETED_WP_TIME, LAST_TIME, FIRST_TIME
    BOATS = {"pid": Boat.Boat(),
             "q": Boat.Boat()}
    # generate all the random waypoints
    generate_random_waypoints_queue()
    waypoint = WAYPOINT_QUEUE[0]
    px, py = xy_location_to_pixel_location(waypoint[0], waypoint[1])
    LAST_TIME = 0
    FIRST_TIME = ptime.time()
    for k in BOATS:
        boat = BOATS[k]
        WAYPOINTS_INDEX[k] = 0
        LAST_COMPLETED_WP_TIME[k] = 0
        boat.state = np.zeros((6,))
        boat.time = 0
        TEXT_BOXES["waypoint_symbol"][k].pos = (px, py)
        TEXT_BOXES["waypoint_text"][k].text = "[{:.0f}, {:.0f}]".format(px, py)
        TEXT_BOXES["waypoint_count"][k].text = "#{} of {}".format(WAYPOINTS_INDEX[k] + 1, WAYPOINTS_BEFORE_RESET)
        boat.strategy = Strategies.DestinationOnly(boat, waypoint, controller_name=CONTROLLERS[k])


def setup():
    global BOATS
    pid_boat = BOATS["pid"]
    pid_boat.design = Designs.TankDriveDesign()
    pid_boat.time = 0

    q_boat = BOATS["q"]
    q_boat.design = Designs.TankDriveDesign()
    q_boat.time = 0

    reset_boats()


if __name__ == "__main__":
    setup()
    app.run()
