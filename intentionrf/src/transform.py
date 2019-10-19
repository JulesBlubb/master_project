#!/usr/bin/env python

import rospy
import numpy as np
import tf
import os
import glob
import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/home/juliane/multiclass-people-tracking')

from std_msgs.msg import Int32, String
from geometry_msgs.msg import PointStamped, Vector3Stamped, WrenchStamped
from mobilityaids_detector.msg import Detections
import subprocess, signal
from multiclass_tracking.viz import Visualizer
from visualization_msgs.msg import Marker, MarkerArray


# extract velocity and position of message
def extract_data(dat):
    vel, pos, cov = [],[],[]
    detections = dat.detections
    for d in detections:
      vel.append(d.velocity.point)
      pos.append(d.position.point)
      #print(pos)
      cov.append(d.cov)
    return vel, pos, cov

def extract_force(dat):
    force = []
    force_msg = dat.wrench.force
    force.append(force_msg)
    return force

# get data of geometry msg
def prepare_data(dat, type):
    x,y,z = ([] for i in range(3))

    if type == 'pos':
        raw = dat.point
    else:
        raw = dat.vector

    x = str(raw.x)
    y = str(raw.y)
    z = str(raw.z)

    return x,y,z

def delete_all_markers(marker_publisher):
    delete_marker = Marker()
    delete_markers = MarkerArray()
    delete_marker.action = Marker.DELETEALL
    delete_markers.markers.append(delete_marker)

    marker_publisher.publish(delete_markers)

# transformed vs. original
# better call it publish marker
def publish_marker(cov, pub, x, y, z):
    viz = Visualizer(1)

    # remove all rviz markers before we publish the new ones
    delete_all_markers(pub)

    markers = MarkerArray()

    # setup marker
    marker = Marker()
    marker.header.frame_id = 'base_link'

    marker.ns = "mobility_aids"
    marker.type = Marker.SPHERE
    marker.action = Marker.MODIFY
    marker.lifetime = rospy.Duration()

    #maker position
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z

    #marker color
    marker.color.b = float(0)
    marker.color.g = float(0.5)
    marker.color.r = float(0.7)
    marker.color.a = 1.0

    #get error ellipse
    width, height, scale, angle = 0.5, 0.5, 0.5, 0.0

    #if a pose covariance is given, like for tracking, plot ellipse marker
    if cov is not None:
        width, height, angle = viz.get_error_ellipse(cov)
        angle = angle + np.pi/2
        scale = 0.1

        quat = tf.transformations.quaternion_from_euler(0, 0, angle)
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        marker.scale.x = height
        marker.scale.y = width
        marker.scale.z = scale

        markers.markers.append(marker)

    pub.publish(markers)

# Transform Position
def callback(msg, args):
    # cov = float64[25]
    pub = args[0]
    file = args[1]
    vel, pos, cov = extract_data(msg)
    rate = rospy.Rate(30)

    pointstamp = PointStamped()
    pointstamp.header.frame_id = '/odom'
    pointstamp.header.stamp = rospy.Time(0)

    vel_in_odom = Vector3Stamped()
    vel_in_odom.header.frame_id = '/odom'
    vel_in_odom.header.stamp = rospy.Time(0)

    # Loop over Position and Velocity Data and make Transformation from /odom to /base_link
    for v, p in zip(vel,pos):
        #print('pos:', pos)

        pointstamp.point.x = p.x
        pointstamp.point.y = p.y
        pointstamp.point.z = p.z

        vel_in_odom.vector.x = v.x
        vel_in_odom.vector.y = v.y
        vel_in_odom.vector.z = v.z

        try:
            # to prevent LookupException, target frame does not exist in first seconds
            listener.waitForTransform('/base_link', '/odom', rospy.Time(), rospy.Duration(4.0))
            (trans,rot) = listener.lookupTransform('/base_link', '/odom', rospy.Time(0))

            # Return homogeneous rotation matrix from quaternion.
            # 4x4
            rot_mat = tf.transformations.quaternion_matrix(rot)

            #print(rot_mat)
            #[[ 0.82836522  0.56018842  0.          0.        ]
            # [-0.56018842  0.82836522  0.          0.        ]
            # [ 0.          0.          1.          0.        ]
            # [ 0.          0.          0.          1.        ]]

            rot_3d = rot_mat[0:3, 0:3]

            # split cov matrix in 5 arrays and convert to
            # matrix for inner product
            temp = np.split(np.asarray(cov)[0], 5)
            temp = np.asmatrix(temp)

            cov_pos = temp[0:3, 0:3]
            cov_trans = rot_3d.dot(cov_pos).dot(np.transpose(rot_3d))

            # transform covariance in array for writing into file
            temp = np.array(cov_trans)
            temp = tuple(temp.flatten())

            result_pos = listener.transformPoint('/base_link', pointstamp)
            print(result_pos)
            result_vel = listener.transformVector3('/base_link', vel_in_odom)

            # Check that the transformed covariance is right. (The cicles should be overlapping)
            # for this you should run rviz and open the config file 'config_detection.rivz' of mobilityaids_detector
            # publish_marker(cov_trans, pub, result_pos.point.x, result_pos.point.y, result_pos.point.z)

            only_coord_pos = prepare_data(result_pos, 'pos')
            only_coord_vel = prepare_data(result_vel, 'vel')

            file.write('Position: ' + str(only_coord_pos) + '\n' + 'Velocity: ' + str(only_coord_vel) + '\n' + 'Cov_posi: ' + str(temp) + '\n')

        except Exception as e:
            print('Exception thrown: ', e)
            pass

        rate.sleep()

# Transform Position
def callback_force(msg, file):
    force = extract_force(msg)

    x,y,z = [],[],[]
    for f in force:
        x = str(f.x)
        y = str(f.y)
        z = str(f.z)

    file.write('Force: ' + x + y + z + "\n")
    rate.sleep()

if __name__ == '__main__':
    rospy.init_node('point_converter')
    rate = rospy.Rate(30)
    #while not rospy.is_shutdown():

    for file in glob.glob(os.path.join('/media/juliane/Robot/Recorded-data-intention/data/', '*.bag')):
            fileName = os.path.basename(file).split('.')[0]

            path = "/home/juliane/catkin_ws/src/intentionrf/src/transformed/"
            filePath = path + fileName + "_transformed.txt"
            curr_file = open(filePath, "a+")

            #if os.path.getsize(filePath):
            #    rospy.signal_shutdown('Finish!')

            # for plotting publish MarkerArray
            pub = rospy.Publisher('~cov', MarkerArray, queue_size=1)
            rospy.Subscriber("mobilityais_detector/tracks", Detections, callback, (pub, curr_file))

            # for recorded forces
            #rospy.Subscriber("clipped_filtered_force", WrenchStamped, callback_force, fileTest)

            listener = tf.TransformListener()

            subprocess.call(["rosbag", "play", file])
            #curr_file.close()

    rospy.spin()
