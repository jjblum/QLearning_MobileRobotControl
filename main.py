import numpy as np
import scipy.integrate as spi
#import matplotlib.pyplot as plt
#import pylab
import Boat
import Strategies
import Designs


#WITH_PLOTTING = True
TOTAL_TIME = 120.0  # [s]
AREA_EDGE_SIZE = 30.0
PLOT_SIZE = AREA_EDGE_SIZE + 10.0  # edge size of plot area
dt = 0.25






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
    c = Canvas()
    app.run()
    main()
