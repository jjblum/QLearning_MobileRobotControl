import numpy as np
import scipy.integrate as spi
import Boat
import Strategies
import Designs
from visualization_scene import BoatVisual, vertex_shader, fragment_shader
from vispy import scene, visuals, app
from vispy.util import ptime


BoatNode = scene.visuals.create_visual_node(BoatVisual)
TextNode = scene.visuals.create_visual_node(visuals.TextVisual)

TOTAL_TIME = 120.0  # [s]
ARENA_EDGE_SIZE = 30.0

# Create a canvas to display our visual
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 800
ARENA_WIDTH = 800
ARENA_HEIGHT = WINDOW_HEIGHT
DATA_WIDTH = WINDOW_WIDTH - ARENA_WIDTH
DATA_HEIGHT = WINDOW_HEIGHT
ARENA_CENTER = (ARENA_WIDTH/2., ARENA_HEIGHT/2.)

# remember 0, 0 is upper left in pixel coordinates, (pixel_width, pixel_height) is the lower right in pixel coordinates
# In real coordinates 0, 0 is the center, negatives are to the left and down
CANVAS = scene.SceneCanvas(keys='interactive', show=True, size=(WINDOW_WIDTH, WINDOW_HEIGHT))
ARENA_VIEW = scene.widgets.ViewBox(parent=CANVAS.scene, name="arena_view", margin=0, bgcolor=(1, 1, 1, 1), size=(ARENA_WIDTH, ARENA_HEIGHT), pos=(0, 0))
DATA_VIEW = scene.widgets.ViewBox(parent=CANVAS.scene, name="data_view", margin=0, bgcolor=(0.8, 0.8, 0.8, 1), size=(DATA_WIDTH, DATA_HEIGHT), pos=(ARENA_WIDTH, 0))

# Create two instances of the visual, each using canvas.scene as their parent
boat_visuals = [BoatNode(ARENA_CENTER[0], ARENA_CENTER[1], 0, 20, 40, (0, .6, .6, 1), parent=CANVAS.scene),
                BoatNode(ARENA_CENTER[0], ARENA_CENTER[1], 0, 20, 40, (.6, 0, 0, 1), parent=CANVAS.scene)]
textboxes = [TextNode("t = ", pos=(ARENA_WIDTH+100, 30), parent=CANVAS.scene, bold=True, font_size=30)]


def xy_location_to_pixel_location(x, y):
    global ARENA_WIDTH, ARENA_HEIGHT, ARENA_EDGE_SIZE
    return x*ARENA_WIDTH/ARENA_EDGE_SIZE + ARENA_WIDTH/2., -1*y*ARENA_HEIGHT/ARENA_EDGE_SIZE + ARENA_HEIGHT/2.


"""
if WITH_PLOTTING:
    figx = 10.0
    figy = 10.0
    fig = plt.figure(figsize=(figx, figy))
    ax_main = fig.add_axes([0.01, 0.01, 0.95*figy/figx, 0.95])  # main ax must come last for arrows to appear on it!!!
    ax_main.elev = 10
    ax_main.grid(b=False)  # no grid b/c it won't update correctly
    ax_main.set_xticks([])  # turn off axis labels b/c they wont update correctly
    ax_main.set_yticks([])  # turn off axis labels b/c they wont update correctly
    axes = [-PLOT_SIZE, PLOT_SIZE, -PLOT_SIZE, PLOT_SIZE]
    boat_arrows = None
    plt.ioff()
    fig.show()
    background_main = fig.canvas.copy_from_bbox(ax_main.bbox)  # must be below fig.show()!
    fig.canvas.draw()


def plotSystem(pid_boat, q_boat, plot_time, waypoints):
    fig.canvas.restore_region(background_main)
    # gather the X and Y location
    pid_boat_x = pid_boat.state[0]
    pid_boat_y = pid_boat.state[1]
    pid_boat_th = pid_boat.state[4]
    q_boat_x = q_boat.state[0]
    q_boat_y = q_boat.state[1]
    q_boat_th = q_boat.state[4]
    ax_main.plot([-PLOT_SIZE, PLOT_SIZE], [-PLOT_SIZE, PLOT_SIZE],
                 'x', markersize=1, markerfacecolor='white', markeredgecolor='white')
    ax_main.axis(axes)  # requires those invisible 4 corner white markers
    for wp in waypoints:
        ax_main.draw_artist(ax_main.plot(wp[0], wp[1], 'kx', markersize=20.0, markeredgewidth=5.0)[0])
    waypoints_array = np.array(waypoints)
    ax_main.draw_artist(ax_main.plot(waypoints_array[:, 0], waypoints_array[:, 1], 'k--', linewidth=2.0)[0])

    # rectangle coords
    left, width = 0.01, 0.95
    bottom, height = 0.0, 0.96
    right = left + width
    top = bottom + height
    time_text = ax_main.text(right, top, "time = {:.2f} s".format(plot_time),
                             horizontalalignment='right', verticalalignment='bottom',
                             transform=ax_main.transAxes, size=20)
    ax_main.draw_artist(time_text)

    pid_boat_arrow = pylab.arrow(pid_boat_x, pid_boat_y, 0.05*np.cos(pid_boat_th), 0.05*np.sin(pid_boat_th),
                             fc="g", ec="k", head_width=1.5, head_length=3.0)
    q_boat_arrow = pylab.arrow(q_boat_x, q_boat_y, 0.05*np.cos(q_boat_th), 0.05*np.sin(q_boat_th),
                             fc="r", ec="k", head_width=1.5, head_length=3.0)
    ax_main.draw_artist(pid_boat_arrow)
    ax_main.draw_artist(q_boat_arrow)
    fig.canvas.blit(ax_main.bbox)
"""


def main():
    pid_boat = Boat.Boat()
    pid_boat.design = Designs.TankDriveDesign()
    pid_boat.time = 0
    pid_boat.state = np.zeros((6,))
    pid_boat.plotData = None

    q_boat = Boat.Boat()
    q_boat.design = Designs.TankDriveDesign()
    q_boat.time = 0
    q_boat.state = np.zeros((6,))
    q_boat.plotData = None

    """
    waypoints = [
        [10.0, 0.0],
        [20.0, 2.0],
        [10.0, 20.0],
        [-10.0, 10.0],
        [-10.0, 0.0]
    ]
    boat.strategy = Strategies.StrategySequence(boat, [
        (Strategies.DestinationOnly, (boat, waypoints[0], THRUST_PID, HEADING_PID, HEADING_ERROR_SURGE_CUTOFF_ANGLE, 2.0, False)),
        (Strategies.DestinationOnly, (boat, waypoints[1], THRUST_PID, HEADING_PID, HEADING_ERROR_SURGE_CUTOFF_ANGLE, 2.0, False)),
        (Strategies.DestinationOnly, (boat, waypoints[2], THRUST_PID, HEADING_PID, HEADING_ERROR_SURGE_CUTOFF_ANGLE, 2.0, False)),
        (Strategies.DestinationOnly, (boat, waypoints[3], THRUST_PID, HEADING_PID, HEADING_ERROR_SURGE_CUTOFF_ANGLE, 2.0, False)),
        (Strategies.DestinationOnly, (boat, waypoints[4], THRUST_PID, HEADING_PID, HEADING_ERROR_SURGE_CUTOFF_ANGLE, 1.0, True))
    ])
    """
    """
    number_of_wp = np.random.randint(1, 10)
    waypoints = np.random.uniform(-PLOT_SIZE, PLOT_SIZE, size=[number_of_wp, 2])
    strategy_sequence = list()
    for wp in range(waypoints.shape[0]):
        strategy_sequence.append((Strategies.DestinationOnly, (boat, waypoints[wp], THRUST_PID, HEADING_PID, HEADING_ERROR_SURGE_CUTOFF_ANGLE, 3.0)))
    boat.strategy = Strategies.StrategySequence(boat, strategy_sequence)
    """

    waypoint = np.random.uniform(-AREA_EDGE_SIZE, AREA_EDGE_SIZE, size=[2, ])
    pid_boat.strategy = Strategies.DestinationOnly(pid_boat, waypoint)

    t = 0.0
    step = 1
    while t < TOTAL_TIME:
        times = np.linspace(t, t+dt, 2)
        pid_boat.time = t
        q_boat.time = t
        pid_boat.control()
        q_boat.control()
        pid_boat_states = spi.odeint(Boat.ode, pid_boat.state, times, (pid_boat,))
        pid_boat.state = pid_boat_states[1]
        q_boat_states = spi.odeint(Boat.ode, q_boat.state, times, (q_boat,))
        q_boat.state = q_boat_states[1]
        t += dt
        step += 1

        if pid_boat.strategy.finished:
            waypoint = np.random.uniform(-AREA_EDGE_SIZE, AREA_EDGE_SIZE, size=[2, ])
            pid_boat.strategy = Strategies.DestinationOnly(pid_boat, waypoint)

        if t > TOTAL_TIME:
            print "Your time exceeded the maximum time of {} seconds. Ending simulation".format(TOTAL_TIME)
            return

        #if WITH_PLOTTING:
        #    plotSystem(pid_boat, q_boat, t, [waypoint])

if __name__ == "__main__":
    app.run()
    main()
