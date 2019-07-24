from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import rospy
import rosbag
import numpy as np
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.io as pio



from std_msgs.msg import Int32, String

passing = rosbag.Bag('/home/juliane/catkin_ws/src/intentionrf/passing.bag')

#<node pkg="tf" type="tf_remap" name="tf_remapper" output="screen">
#  <rosparam param="mappings">
#    [{old: world, new: base_link}]
#  </rosparam>
#</node>

#<node pkg="rosbag" type="play" name="player" args="--clock passing.bag">
#  <remap from="tf" to="tf_old" />
#</node>


# t = timestamp
def extract_topic(bag):
    for topic, msg, t in passing.read_messages(topics=['scan']):
        return msg.ranges
    bag.close()

ranges = extract_topic(passing)

def extract_val(bag):
    vel = []
    pos = []
    for topic, msg, t in passing.read_messages(topics=['mobilityais_detector/tracks']):
        detections = msg.detections
        for d in detections:
            vel.append(d.velocity.point)
            pos.append(d.position.point)
    return vel, pos
    bag.close()


dat = extract_val(passing)

# get data of geometry msg
def prepare_data(dat):
    for i in dat:
        x = list(map(lambda i: i.x, dat))
        y = list(map(lambda i: i.y, dat))
        z = list(map(lambda i: i.z, dat))
    return x,y,z

prepare_data(dat[0])


# #### PLOT DATA #####

# # Plot Laser Scanner Data
# plot_ranges = go.Scatter(
#     #x=range(len(ranges)),
#     y=ranges,
#     mode='markers',
# )
# layout = go.Layout(title='Laser Scanner Data')
# data = [plot_ranges]
# fig_ranges = go.Figure(data=data, layout=layout)
# #plot(fig_ranges)


# # Plot Velocity Data
plot_velocity = go.Scatter(
    x=prepare_data(dat[1])[0],
    y=prepare_data(dat[1])[1],
    mode='markers'
)
layout = go.Layout(title='Velocity')
data = [plot_velocity]
fig_veloctiy = go.Figure(data=data, layout=layout)
plot(fig_veloctiy)

# # Plot Position
# trace1 = go.Scatter3d(
#     x=prepare_data(dat[1])[0],
#     y=prepare_data(dat[1])[1],
#     z=prepare_data(dat[1])[2],
#     mode='markers',
#     marker=dict(
#         size=5,
#         colorscale='Viridis',   # choose a colorscale
#         opacity=0.8
#     )
# )

# data = [trace1]
# layout = go.Layout(
#     title = 'Human Position',
#     margin=dict(
#         l=65,
#         r=50,
#         b=65,
#         t=90
#     )
# )
# fig = go.Figure(data=data, layout=layout)
# #plot(fig)


# #### RANDOM FOREST ######

# X = [ranges, prepare_data(dat[0]), prepare_data(dat[1])]
# # passing = 0, interaction 1
# # y = [0, 1, 2]

# clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

# #z = clf.fit(X,y)
# #print(z)

# ## Tranform world to robot fram

# #bag = rosbag('passing.bag')
# #frames = passing.AvailableFrames;

# #tf = fet Tranform(bag, 'world', frames{1})


