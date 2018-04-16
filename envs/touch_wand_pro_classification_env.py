import pybullet as pb
import numpy as np
import time,os,math,inspect,re,errno
import random,glob,math
from shutil import copyfile
import sensenet
from sensenet.envs.handroid.hand_env import HandEnv
from sensenet import spaces
from sklearn.model_selection import train_test_split
from os import environ
import os

class TouchWandProClassificationEnv(HandEnv):
    def __init__(self,options={}):
        self.options = options
        if environ.get('TOUCH_DATA') is not None:
            self.options['data_path'] = os.environ['TOUCH_DATA'] 
        self.steps = 0

        #TODO check if options is a string, so we know which environment to load
        if 'render' in self.options and self.options['render'] == True:
            pb.connect(pb.GUI)
        else:
            pb.connect(pb.DIRECT)
        pb.setGravity(0,0,-0.001)
        pb.setRealTimeSimulation(0)
        self.move = 0.01 # 0.01
        self.prev_distance = 10000000
        self.max_steps = 1000
        self.wandLength = 0.5
        self.wandSide = 0.005
        self.cameraImageHeight = int((2 * 0.851 * self.wandSide)*11800)
        self.class_count = len([x[0] for x in os.walk(self.options['data_path'])]) #path must be a full path,TODO convert to full path
        self.action_space = spaces.Discrete(6 + self.class_count)
        self.timestep_limit = 400
        #self.observation_space = spaces.Box(0, 1, [6,  self.cameraImageHeight, self.cameraImageHeight])
        self.observation_space = spaces.Box(-100, 100, self.cameraImageHeight**2+6)
        self.touch_history = []

    def load_object(self, file=None):
        obj_x = 0
        obj_y = -1
        obj_z = 0
        obj_type = 'obj'

        if file is not None:
            stlfile = file
        elif file is None:
            if 'data_path' in self.options and self.options['data_path'] is not None:
                path = self.options['data_path']
            else:
                raise KeyError('Expected data_path in command line')
            self.files = glob.glob(path+"/**/*."+obj_type,recursive=True)
            stlfile = self.files[random.randrange(0, self.files.__len__())]

        if os.name == 'nt':
            self.class_label = int(stlfile.split("\\")[-3].split("_")[0])
        elif os.name == 'posix' or os.name == 'mac':
            self.class_label = int(stlfile.split("/")[-2].split("_")[0])

        dir_path = os.path.dirname(os.path.realpath(__file__))
        copyfile(stlfile, dir_path+"/data/file."+obj_type)
        urdf_path = dir_path+"/data/loader."+obj_type+".urdf"
        self.obj_to_classify = pb.loadURDF(urdf_path,(obj_x,obj_y,obj_z),useFixedBase=1)
        pb.changeVisualShape(self.obj_to_classify,-1,rgbaColor=[1,0,0,1])

    def load_agent(self):
        xyz=[-1,-1,0]
        orientation = [0,1,0,1]
        self.agent = pb.createCollisionShape(pb.GEOM_BOX,
                                             halfExtents=[self.wandSide,
                                                          self.wandSide,
                                                          self.wandLength])

        self.agent_mb= pb.createMultiBody(1,self.agent,-1,basePosition=xyz,
                                          baseOrientation=orientation)

        pb.changeVisualShape(self.agent,-1,rgbaColor=[1,1,0,1])

        self.agent_cid = pb.createConstraint(self.agent,-1,-1,-1,pb.JOINT_FIXED,
                                             [0,0,0],[0,0,0],xyz,
                                             childFrameOrientation=orientation)

        self.fov = 45


    def ahead_view(self):

        link_state = pb.getBasePositionAndOrientation(self.agent)
        link_p = link_state[0]
        link_o = link_state[1]
        handmat = pb.getMatrixFromQuaternion(link_o)
        """
        axisX = [handmat[0],handmat[3],handmat[6]]
        axisY = [-handmat[1],-handmat[4],-handmat[7]] # Negative Y axis
        axisZ = [handmat[2],handmat[5],handmat[8]]
        """
        # we define the up axis in terms of the object to classify since it is
        # stationary. If you define up in terms of the orientation of the wand
        # then your up axis can rotate as the wand rotates due to the normal
        # contact forces with the object to classify. This causes the camera
        # image to exhibit some unwanted behavior, such as inverting the up
        # and down directions.
        # If your object to classify is not stationary, then you could switch to
        # the above code to use the wand orientation to define your up axis.

        state = pb.getBasePositionAndOrientation(self.obj_to_classify)
        state_o = state[1]
        mat = pb.getMatrixFromQuaternion(state_o)
        axisZ2 = [mat[2], mat[5], mat[8]]
        eyeOffset = self.wandLength
        focusOffset = eyeOffset + 0.0000001

        self.eye_pos = [link_p[0] + eyeOffset, link_p[1], link_p[2]]
        self.target_pos = [link_p[0] + focusOffset, link_p[1], link_p[2]]

        up = axisZ2 # Up is Z axis of obj_to_classify - change to axisZ to use
        # the wand orientation as up
        viewMatrix = pb.computeViewMatrix(self.eye_pos,self.target_pos,up)

        if 'render' in self.options and self.options['render'] == True:
            pass

        return viewMatrix

    def _step(self,action):
        done = False
        aspect = 1
        camTargetPos = [0,0,0]
        yaw = 40
        pitch = 10.0
        roll=0
        upAxisIndex = 2
        camDistance = 4
        pixelWidth = 320
        pixelHeight = 240
        nearPlane = 0.0001
        farPlane = 0.022
        lightDirection = [0,1,0]
        lightColor = [1,1,1]#optional
        agent_po = pb.getBasePositionAndOrientation(self.agent_mb)
        x = agent_po[0][0]
        y = agent_po[0][1]
        z = agent_po[0][2]
        
        if action - self.class_count == 0: #down
          x += self.move
        elif action - self.class_count == 1: #up
          x -= self.move
        elif action -self.class_count == 2: #left
          y += self.move
        elif action-self.class_count == 3: #right
          y -= self.move
        elif action -self.class_count== 4: #<
          z += self.move
        elif action -self.class_count== 5: #>
          z -= self.move
        elif action -self.class_count== 9: #TODO fix this one
          #print(self.current_observation[:100])
          #print(np.amax(self.current_observation))
          print(np.unique(self.current_observation))
          if self.is_touching():
              print('I am touching')
          else:
              print('I am not touching')
        pivot = [x,y,z]

        orn = pb.getQuaternionFromEuler([0,0,0])
        pb.changeConstraint(self.agent_cid,pivot,
                            jointChildFrameOrientation=[0,1,0,1],
                            maxForce=100)

        viewMatrix = self.ahead_view()
        projectionMatrix = pb.computeProjectionMatrixFOV(self.fov,aspect,
                                                    nearPlane,farPlane)
        w,h,img_arr,depths,mask = pb.getCameraImage(self.cameraImageHeight,
                                                    self.cameraImageHeight,
                                                    viewMatrix,
                                                    projectionMatrix,
                                                    lightDirection,
                                                    lightColor,
                                                    renderer=pb.ER_TINY_RENDERER)

        info = [42]
        pb.stepSimulation()

        self.steps += 1
        reward = 0
        touch_reward = 0

        if self.is_touching():
            if self.is_unique_touch():
                self.unique_touch = True
                touch_reward = 1                
            else:
                touch_reward = 0
                self.unique_touch = False
            depths = np.asarray(depths,dtype=np.float32)
            new_obs = np.absolute(depths - 1.0).flatten()
            proPrio = np.array(pivot)

            new_obs = np.concatenate((proPrio, self.object_contact_points, new_obs))
            # if you want binary representation of depth camera, uncomment
            # the line below
            #new_obs[new_obs > 0] = 1
            #self.current_observation = new_obs.flatten()
            self.current_observation = new_obs
        elif action < self.class_count :
            if self.class_label == action:
                reward += 5
                print("same")
            else:
                reward += -5
            done = True
        else:
            touch_reward = 0
            self.current_observation = np.zeros(self.cameraImageHeight*self.cameraImageHeight+6)
        agent_po = pb.getBasePositionAndOrientation(self.agent_mb)
        obj_po = pb.getBasePositionAndOrientation(self.obj_to_classify)
        reward += touch_reward
        #print("XXXXXXXXXXXXXcount",self.class_count,"class",self.class_label,"action",action,"reward",reward)

        if self.steps >= self.max_steps:
            done = True
        return self.current_observation,reward,done,info

    def is_touching(self):
        points = pb.getContactPoints(self.agent,self.obj_to_classify)
        if len(points) > 0:
            self.object_contact_points = np.round(points[0][6], decimals=2)
            #self.touch_history.append(self.object_contact_points)
        return len(points) > 0

    def is_unique_touch(self):
        if len(self.touch_history) > 0:
            for point in self.touch_history:       
                if point in self.object_contact_points:                
                    return False                              
            self.touch_history.append(self.object_contact_points)
            return True
        else:
            self.touch_history.append(self.object_contact_points)
            return True
        
        

    def _reset(self, file=None):
        # load a new object to classify
        # move hand to 0,0,0
        pb.resetSimulation()
        self.touch_history = []
        self.load_agent()
        self.load_object(file)
        default = np.zeros(self.cameraImageHeight*self.cameraImageHeight+6)
        self.steps = 0
        self.current_observation = default
        return default
