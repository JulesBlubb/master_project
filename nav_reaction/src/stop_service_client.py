#! /usr/bin/env python
import rospkg
import rospy
from std_srvs.srv import Empty, EmptyRequest 
from nav_reaction.srv import StartInteraction, StartInteractionRequest

rospy.init_node('interact_stop_moving_client') # Initialise a ROS node with the name service_client
rospy.wait_for_service('/start_interaction') # Wait for the service client /start_interaction to be running
interact_stop_moving_client = rospy.ServiceProxy('/start_interaction', StartInteraction) # Create the connection to the service
interact_stop_moving_request_object = StartInteractionRequest() 

"""
# StartInteraction.srv
int32 keypressed    # if 1 is pressed the robot stops because interaction is started 
---
bool success         # Did it achieve it?
"""

print "Please choose the reaction mode"
give_key = raw_input()
print "output: " + give_key
converted = int(give_key)
interact_stop_moving_request_object.keypressed = converted

rospy.loginfo("Doing Service Call...")
result = interact_stop_moving_client(interact_stop_moving_request_object) # Send through the connection the path to the trajectory file to be executed
rospy.loginfo(str(result)) # Print the result given by the service called

rospy.loginfo("END of Service call...")
