#! /usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse
from nav_reaction.srv import StartInteraction, StartInteractionResponse
from geometry_msgs.msg import Twist


def my_callback(request):
    rospy.loginfo("The Service start_interaction has been called")
    if request.keypressed == 1:
    	move.linear.x = 0.0
    	# move.linear.y = 0.0
    	move.angular.z = 0.0
    elif request.keypressed == 2:
        move.linear.x = 0.3
        # move.linear.y = 0.0
        move.angular.z = 0.3

    my_pub.publish(move)
    rospy.loginfo("Finished start_interaction")
    response = StartInteractionResponse()
    response.success = True
    return response
    #return EmptyResponse

rospy.init_node('interact_stop_moving')
my_service = rospy.Service('/start_interaction', StartInteraction, my_callback) # create the Service start_interaction
my_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
move = Twist()
rospy.loginfo("Service /start_interaction")
rospy.spin() # maintain the service open.


