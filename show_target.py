#!/usr/bin/env python3

import rospy
import time
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Point,Pose

class ShowTarget:
    def __init__(self):        
        rospy.init_node('show_target')
        self.set_point_x = 0
        self.set_point_y = 0
        self.last_pos_x = self.set_point_x
        self.last_pos_y = self.set_point_y
        self.model_sdf = open('/home/pedro/tcc_ws/src/tcc/scripts/target/model.sdf', 'r').read()

    def del_model(self):
        models = rospy.wait_for_message('gazebo/model_states', ModelStates)
        for i in range(len(models.name)):
            if models.name[i] == 'target':
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox('target')

    def get_target_pos(self,point_msg):
        #self.del_model()
        target_positon = Pose()
        target_positon.position.x = point_msg.x
        target_positon.position.y = point_msg.y
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_prox('target', self.model_sdf, 'robotos_name_space',target_positon, "world")

    def run(self):
        while True:
            self.get_target_pos(rospy.wait_for_message('/target_point', Point)) 
            self.show_target()


ShowTarget().run()