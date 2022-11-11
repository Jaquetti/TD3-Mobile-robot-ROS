#!/usr/bin/env python3

from hashlib import new
from os import system
from turtle import color
import torch
import rospy
import random
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from math import *
import numpy as np
import tf
import TD3
import utils
import torch
from geometry_msgs.msg import Twist, Point,Pose
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SpawnModel, DeleteModel
import time
import matplotlib.pyplot as plt


class Robot_ros:
	def __init__(self):
		rospy.init_node('robot')
		self.pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)
		self.pub_point = rospy.Publisher('/target_point', Point, queue_size = 1)
		self.rate = rospy.Rate(10)
		self.vel = Twist()
		self.rel_theta = 0
		self.yaw = 0
		self.diff_angle = 0
		self.posi_x = 0
		self.posi_y = 0
		self.rang = 0
		self.theta = 0
		self.distance = []
		self.angle = []
		self.angles = []
		self.actions_ = []
		self.lidar = []
		self.last_av = 0
		self.diagonal_dis = 0
		self.last_lv = 0
		self.prev_min_lid  = 0
		self.chegou = 0

		
		

		#Para o mapa stage 4 é complicado gerar aleatóriamente
		self.points_x = [2 , 2,  0.5, 1, 1, 0.7, 2, 1.7, 0, -2, -2, -2, -1.5, 
						0.8,   0.41,  0.3,   0.3, 0.3, 0.0, 1, 1, -0.3, -1.5, -1.7,  -1.2, -1.1]
		self.points_y = [0, -2, -2,   0, 1, 1.9, 1, 1.7, 2,  2,  0, -1, -2, 0.25, -1.3,  -1.0,  
						-0.8, 1.3, 1.2, 1, 1,  1.0,  1.0,  1.0,  -0.7, -1]
		
		self.train = False
		self.set_point_x = None
		self.set_point_y = None
		self.change_point()
		
		
		self.epochs = 1000000
		self.max_steps = 200

		self.end = False
		self.prev_dist_error = 0
		self.dist_error = 0
		self.prev_diff_angle = 0
		self.prev_ang_error = 0
		self.ang_error = 0
		self.theta_goal = 0
		self.prev_min_dist = 0
		self.state = self.sensor_read(rospy.wait_for_message('/scan', LaserScan,timeout=None))
		self.new_state = 0

		self.reward = 0
		self.steps = 0
		self.score = 0
		self.scores = []
		self.evaluations = []
		self.step_robot = 0
		self.score_avg = []

		self.prev_ang_error = 0
		
		self.state_dim = 81
		self.action_dim = 2
		self.batch_size = 256
		# self.max_action=torch.Tensor([0.4,0.7])
		self.max_action=torch.Tensor([0.35,0.6])
		self.policy_noise=0.2*self.max_action
		self.noise_clip=0.5*self.max_action
		self.policy_freq = 50
		#0.7 0.4 0.3
		self.expl_noise = 0.2
		self.noise_control = 0
		self.count_crash = 0
		self.eval_episodes =10
		self.policy = TD3.TD3(state_dim= self.state_dim,action_dim=self.action_dim,
			max_action = self.max_action,policy_noise = self.policy_noise,
				noise_clip = self.noise_clip, policy_freq = self.policy_freq)
		
		self.replay_buffer = utils.ReplayBuffer(self.state_dim, self.action_dim)
		self.load_weights()
	

	def plot_target(self):
		rospy.wait_for_service('/gazebo/pause_physics')
		models = rospy.wait_for_message('gazebo/model_states', ModelStates)
		for i in range(len(models.name)):
			if models.name[i] == 'target':
				del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
				del_model_prox('target')
		
		time.sleep(0.2)
		target_positon = Pose()
		target_positon.position.x = self.set_point_x
		target_positon.position.y = self.set_point_y
		spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
		self.model_sdf = open('/home/pedro/tcc_ws/src/tcc/scripts/target/model.sdf', 'r').read()
		spawn_model_prox('target', self.model_sdf, 'robotos_name_space',target_positon, "world")
		rospy.wait_for_service('/gazebo/unpause_physics')

	def change_point(self):
		#Para o mapa 1/3
		# self.set_point_x = round(random.uniform(-1.5, 1.5),1)
		# self.set_point_y = round(random.uniform(-1.5, 1.5),1)

		#Para o mapa 2
		# self.set_point_x = round(random.uniform(-1.5, 1.5),1)
		# self.set_point_y = round(random.uniform(-1.5, 1.5),1)

		# free = False
		# while not free:
		# 	if 0.3<abs(self.set_point_x)<0.9 and 0.3<abs(self.set_point_y)<0.9:
		# 		self.set_point_x = round(random.uniform(-1.5, 1.5),1)
		# 		self.set_point_y = round(random.uniform(-1.5, 1.5),1)
		# 	else:
		# 		free = True

		#Para o mapa 4
		self.index = np.random.randint(0,len(self.points_x )-1)
		self.set_point_x = self.points_x[self.index]
		self.set_point_y = self.points_y[self.index]

		if self.train==False:
			self.plot_target()
		
	def get_odom(self, data):
		return data
		
	def sensor_read(self,msgScan):
		self.distance = []

		k = 0
		for i in range(len(msgScan.ranges)):
			k+=1
			if k%5==0:
				self.distance.append(msgScan.ranges[i])
				k=0

		self.distances = np.array(self.distance,dtype=np.float32)
		self.distances[self.distances == np.inf] = msgScan.range_max
		
	
		return self.distances

	def check_norm_dist(self):
		if self.dist_error>self.diagonal_dis:
			self.diagonal_dis = self.dist_error

	def take_action(self, lv, av):
		self.vel.linear.x = lv
		self.vel.linear.y = 0
		self.vel.linear.z = 0	
		self.vel.angular.x = 0
		self.vel.angular.y = 0
		self.vel.angular.z = av

		self.pub_vel.publish(self.vel)

		self.last_lv = lv
		self.last_av = av
		
	def stop_robot(self):
		self.vel = Twist()
		self.vel.linear.x = 0
		self.vel.linear.y = 0
		self.vel.linear.z = 0	
		self.vel.angular.x = 0
		self.vel.angular.y = 0
		self.vel.angular.z = 0
		self.pub_vel.publish(self.vel)
		
	def reset_simulation(self):
		self.stop_robot()
		rospy.wait_for_service('/gazebo/reset_simulation')
		clear_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
		clear_sim()

	def load_weights(self):
		try:
			self.policy.load('/home/pedro/tcc_ws/src/tcc/scripts/pesos')
			print('Carregou os pesos')
		except:
			print('Não carregou os pesos')
			pass

	def generate_real_msg_lidar(self):
		msg = None
		while msg==None:
			try:
				msg = rospy.wait_for_message('/scan', LaserScan,timeout=None)
			except:
				pass
		
		return msg

	def generate_real_msg_odom(self):
		msg = None
		while msg==None:
			try:
				msg = rospy.wait_for_message('/odom', Odometry,timeout=None)
			except:
				pass
		
		return msg

	def step(self,action):
		done = False
		self.take_action(action[0],action[1])
		
		self.lidar = self.sensor_read(self.generate_real_msg_lidar())
		
		rospy.wait_for_service('/gazebo/pause_physics')
		min_obs_distance = min(self.lidar)
		obs_angle = np.argmin(self.lidar)
		self.odom = self.get_odom(self.generate_real_msg_odom())
		self.posi_x = self.odom.pose.pose.position.x
		self.posi_y = self.odom.pose.pose.position.y

		orientation_q = self.odom.pose.pose.orientation
		(_, _, yaw) = tf.transformations.euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
		
	
		rel_dis_y = round(self.set_point_y-self.posi_y, 1)
		rel_dis_x = round(self.set_point_x-self.posi_x, 1)

		self.dist_error = pow(rel_dis_x,2) + pow(rel_dis_y,2)
		rel_theta = round((atan2(rel_dis_y,rel_dis_x)), 2)
		
		ang_error = yaw-rel_theta

		if ang_error >pi:
			ang_error-=2*pi
		elif ang_error<-pi:
			ang_error+=2*pi

		self.ang_error = ang_error
		self.end = False

		if min(self.distance)<0.12 or self.step_robot>=200:
			done = True
			reward = -200
			self.count_crash+=1
			if self.count_crash==10:
				self.change_point()
				self.count_crash = 0
			self.reset_simulation()

		elif self.dist_error<=0.3:
			self.count_crash = 0
			self.step_robot = 0
			done = True
			self.end = True
			self.change_point()
			reward = 200
			self.chegou+=1
			
		else:
			if min(self.distance)<0.3:
				obs_reward = -0.2
			else:
				obs_reward = 0

			if min(self.distance)>=self.prev_min_dist:
				obs_far = 0.2
			else:
				obs_far = 0

			reward = 10*(self.prev_dist_error-self.dist_error) \
			        + 2*(self.prev_ang_error-self.ang_error)+obs_reward+obs_far

		self.prev_dist_error = self.dist_error
		self.prev_ang_error = self.ang_error
		self.prev_min_dist  = min(self.distance)

		new_state = np.append(self.lidar, np.array([self.posi_x,self.posi_y,yaw,self.last_lv, self.last_av, self.dist_error, self.ang_error, min_obs_distance,obs_angle]))
		rospy.wait_for_service('/gazebo/unpause_physics')
		
		return new_state, reward, done

	def run(self, train = False):
		
		self.train = train
		self.reset_simulation()
		if self.train is not True:
			self.evaluations.append(self.eval_policy())
		else:
			for _ in range(self.epochs):
				self.state, _, _ = self.step([0,0])
				self.score = 0
				self.step_robot = 0
				self.done = False
				self.prev_dist_error = 0
				self.prev_ang_error = 0
				
				
				while not self.done:
					action = self.policy.select_action(np.array(self.state))
					action = (action + np.random.normal(0, self.expl_noise, size=2)).clip(torch.Tensor([0,-0.6]), self.max_action)

					self.new_state, self.reward, self.done = self.step(action)

					self.score+=self.reward
					self.step_robot+=1

					self.replay_buffer.add(self.state, action, self.new_state, self.reward, self.done)


					self.state = self.new_state
					self.noise_control+=1


					if self.noise_control >= 1000:
						self.policy.train(self.replay_buffer, self.batch_size)

					if self.noise_control % 1000 == 0:
						self.policy.save('/home/pedro/tcc_ws/src/tcc/scripts/pesos')
				

				if self.noise_control%100 ==0 and self.noise_control>200000 and self.expl_noise>0.05:
						self.expl_noise*=0.97
						self.evaluations.append(self.eval_policy())
						np.save('/home/pedro/tcc_ws/src/tcc/scripts/eval.npy', self.evaluations)
						self.policy.save('/home/pedro/tcc_ws/src/tcc/scripts/pesos')
					
				if self.done:
					self.end = False	
					self.scores.append(self.score)
					self.avg_score = np.mean(self.scores[-100:])
					self.score_avg.append(self.avg_score)
					np.save('/home/pedro/tcc_ws/src/tcc/scripts/score_avg.npy', self.score_avg)
					print('exp',self.expl_noise,'score: ', self.score, 'avg_score: ',
					self.avg_score,'chegou: ', self.chegou, 
					'pontos: ', (self.set_point_x, self.set_point_y),' noise_control:', self.noise_control, 
					'last eval',self.evaluations[-1] if len(self.evaluations)>0 else 0)
		
	def eval_policy(self):
		avg_reward = 0.
		print('ENTRANDO NO EVAL')
		for _ in range(self.eval_episodes):
			self.done = False
			self.state, _, _ = self.step([0,0])
			self.step_robot = 0
			while not self.done:
				self.step_robot+=1
				action = self.policy.select_action(np.array(self.state))
				self.state, self.reward, self.done = self.step(action)
				avg_reward += self.reward

			if self.done:
				self.stop_robot()
				if self.end ==False:
					self.reset_simulation()
				else:
					print('Chegou no ponto')

		avg_reward /= self.eval_episodes
		print("Recompensa média de "+ str(avg_reward)+" em "+str(self.eval_episodes)+" episódios")
	
		return avg_reward



robot = Robot_ros()
robot.run()


