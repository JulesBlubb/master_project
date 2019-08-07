#!/usr/bin/env python

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification


import rospy
import numpy as np
import rosbag
import tf2_ros
import tf
import os

#import plotly.graph_objs as go
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#import plotly.io as pio

from std_msgs.msg import Int32, String
from geometry_msgs.msg import PointStamped, Vector3Stamped
from mobilityaids_detector.msg import Detections


# extract velocity and position of message
def extract_data(dat):
    vel, pos = [],[]
    detections = dat.detections
    for d in detections:
      vel.append(d.velocity.point)
      pos.append(d.position.point)
    return vel, pos

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

# Transform Position
def callback(msg):
    vel, pos = extract_data(msg)
    rate = rospy.Rate(30)

    pointstamp = PointStamped()
    pointstamp.header.frame_id = '/odom'
    pointstamp.header.stamp = rospy.Time(0)


    vel_in_odom = Vector3Stamped()
    vel_in_odom.header.frame_id = '/odom'
    vel_in_odom.header.stamp = rospy.Time(0)

    file = open("intentionrf/src/results.txt","a")

    # Loop over Position and Velocity Data and make Transformation from /odom to /base_link
    for v, p in zip(vel,pos):
        #print('X:', p.x)
        pointstamp.point.x = p.x
        pointstamp.point.y = p.y
        pointstamp.point.z = p.z

        vel_in_odom.vector.x = v.x
        vel_in_odom.vector.y = v.y
        vel_in_odom.vector.z = v.z

        try:
            # to prevent LookupException, target frame does not exist in first seconds
            listener.waitForTransform('/base_link', '/odom', rospy.Time(), rospy.Duration(4.0))

            result_pos = listener.transformPoint('/base_link', pointstamp)
            result_vel = listener.transformVector3('/base_link', vel_in_odom)

            only_coord_pos = prepare_data(result_pos, 'pos')
            only_coord_vel = prepare_data(result_vel, 'vel')

            print('Position: ', only_coord_pos)
            print('Veloctiy: ', only_coord_vel)

            file.write('Position: ' + str(only_coord_pos) + '\n' + 'Velocity: ' + str(only_coord_vel) + '\n')

            #print(result)
        except Exception as e:
            print('Exception thrown: ', e)
            pass

        rate.sleep()


        #rospy.spin()


if __name__ == '__main__':
    rospy.init_node('point_converter')

    if os.path.exists("intentionrf/src/results.txt"):
        os.remove("intentionrf/src/results.txt")


    rate = rospy.Rate(30)
    rospy.Subscriber("mobilityais_detector/tracks", Detections, callback)

    listener = tf.TransformListener()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    file.close




