from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy


def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*

        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motion_particles = []

    #no movement detected so particles stay same
    if odom[0] == 0 and odom[1] == 0 and odom[2] == 0:
        return particles

    for particle in particles:

        odom_with_noise = add_odometry_noise(odom, ODOM_HEAD_SIGMA, ODOM_TRANS_SIGMA)

        (dx, dy, dh) = odom_with_noise

        #dx and dy relative in terms of robot local frame, so need to have motion dependent on particle frame
        xr, yr = rotate_point(dx, dy, particle.h)

        new_x = particle.x + xr
        new_y = particle.y + yr
        new_h = particle.h + dh

        new_particle = Particle(new_x, new_y, new_h)
        motion_particles.append(new_particle)

    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """

    def closest_particle_marker(particle_markers, robot_marker):

        minDistance = grid_distance(particle_markers[0][0],particle_markers[0][1], robot_marker[0], robot_marker[1])
        minAngle = diff_heading_deg(particle_markers[0][2], robot_marker[2])

        for particle_marker in particle_markers:
            dist = grid_distance(particle_marker[0], particle_marker[1], robot_marker[0], robot_marker[1])
            if minDistance > dist:
                minDistance = dist
                minAngle = diff_heading_deg(particle_marker[2], robot_marker[2])

        return minDistance, minAngle

    weights = [1.0 for _ in range(len(particles))]

    for i, particle in enumerate(particles):

        if not grid.is_in(particle.x, particle.y) or not grid.is_free(particle.x, particle.y):
            weights[i] = 0.0
            continue

        particle_markers = particle.read_markers(grid)


        for robot_marker in measured_marker_list:
            new_robot_marker = add_marker_measurement_noise(robot_marker, MARKER_TRANS_SIGMA, MARKER_ROT_SIGMA)

            if len(particle_markers) > 0:
                min_distance, min_angle = closest_particle_marker(particle_markers, new_robot_marker)

                exponent = ((min_distance ** 2) / (2 * (MARKER_TRANS_SIGMA ** 2))) + \
                           ((min_angle ** 2) / (2 * (MARKER_ROT_SIGMA ** 2)))
                weights[i] *= numpy.exp(-exponent)
            else:
                weights[i] = 0.0

        if not len(measured_marker_list) == len(particle_markers):
            weights[i] *= DETECTION_FAILURE_RATE ** (abs(len(measured_marker_list) - len(particle_markers)))


    weight_sum = sum(weights)
    if weight_sum == 0:
        weights = [1.0 / len(particles) for _ in range(len(particles))]
    else:
        for i in range(len(weights)):
            weights[i] /= weight_sum

    random_particles = Particle.create_random(200, grid)

    chosen_particles = numpy.random.choice(particles, len(particles) - len(random_particles), True, weights)

    measured_particles = numpy.concatenate((chosen_particles, random_particles))

    return measured_particles


