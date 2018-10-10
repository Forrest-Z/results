#!/usr/bin/env python

############PROGRAM STARTS HERE ######################
import numpy as np
import math as MT
from math import floor
import matplotlib.pyplot as plt
import time
import rospy
import tf
from geometry_msgs.msg import Twist,PoseStamped,Pose,PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid,MapMetaData,Path
from scipy.spatial import KDTree


cf=[-50.5,-50.5]
fv=[0,0]

###CONSTANTS
max_angle = 0.785398 #45Deg
min_angle = -0.785398 #-45Deg
free_space=0
locked_space=1
grid_offset_value=10

### HYPER PARAMETERS
NUMBERS_OF_STEERS=4
STEER_OFFSET=5.0*np.pi/180
LENGTH=2.67
NUM_THETA_CELLS =60



### MOTION MATRIX FOR ASTAR
motion_mat=np.array([[1,0],[-1,0],[0,-1],[0,1]])

### STATE CLASS
class state:
  def __init__(self,x,y,theta,g,f,h,steer):
    self.x=x
    self.y=y
    self.theta=theta
    self.g=g
    self.f=f
    self.h=h
    self.steer=steer

class values:
  def __init__(self,grid_on_x,grid_on_y,GRID_TEST,value_map):
    self.grid_on_x=grid_on_x
    self.grid_on_y=grid_on_y
    self.GRID_TEST=GRID_TEST
    self.value_map=value_map
    self.is_processing=True
    self.feed_map=True
    self.get_start_point=True
    # self.grid_x_m=grid_x_m
    # self.grid_y_m=grid_y_m

    


###WAYPOINTS
class waypoints():
    def __init__(self,input_co_ordinates,center):
        self.input_co_ordinates=input_co_ordinates
        self.center=center


    
 ## GOAL NODE        
class goal:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
### INPUT VEHICLE CO-ORDINATES
class vehicle_points():
  def __init__(self,input_co_ordinates,center):
    self.input_co_ordinates=input_co_ordinates
    self.center=center
    
### PATH CLASS FOR TRACKING   
class path():
  def __init__(self,closed,came_from,final):
    self.closed=closed
    self.came_from=came_from
    self.final=final
    

### INITIALIZE INPUTS
get_value= values(1,1,np.array([]),np.array([]))


def find_intersection_X(a,b,r,w,v,t,c):
    c6=w**2
    c5=2*w*v
    c4=v**2+2*w*t
    c3=2*(v*t+w*c-w*b)
    c2=t*t + 2*v*c - 2*v*b+1
    c1=2*t*c-2*t*b-2*a
    c0=(c-b)**2+a**2-r**2
    
    return np.roots([c6,c5,c4,c3,c2,c1,c0])


### EUCLIDEAN DISTANCE
def euclidean_distance(start_point,end_point):
  return np.round(np.sqrt((end_point[0]-start_point[0])**2 +(end_point[1]-start_point[1])**2),4)

### PADDER
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


### TRANSFORM FRAME CO-ORDINATES 
def transform_frame_co_ordinates(current_frame_waypoints_obj, frame_expected):
    displaced_matrix = np.array([frame_expected[0]-current_frame_waypoints_obj.center[0],frame_expected[1]-current_frame_waypoints_obj.center[1]])
    transformed_matrix=np.add(current_frame_waypoints_obj.input_co_ordinates,displaced_matrix)
    return waypoints(transformed_matrix,frame_expected)

### AUGMENT DELTA +/- GIVEN OFFSET
def delta_augmentation(delta, numbers, offset):
    delta_list = []
    delta_list.append(delta)
    delta_calc_add=delta_calc_sub = delta
    for i in range(0 ,numbers):
        delta_calc_add += offset
        delta_calc_sub -= offset
        if delta_calc_add < max_angle:
            delta_list.append(delta_calc_add)
        if delta_calc_sub > min_angle:
            delta_list.append(delta_calc_sub)
    return delta_list
  


### NEW STATE TRANSITIONS
def new_state_transition(current_state,goal,speed):
    next_states = []
    delta_angles = delta_augmentation( delta=current_state.steer, numbers=NUMBERS_OF_STEERS,offset=STEER_OFFSET)
    DT=1.0/speed
    for delta in delta_angles:
        omega  = (speed / LENGTH) * np.tan(delta)
        theta2 = normalize_theta(current_state.theta + (omega * DT))
        dX = speed * np.cos(theta2) * DT
        dY = speed * np.sin(theta2) * DT
        x2     = current_state.x + dX
        y2     = current_state.y + dY
        g2     = current_state.g + np.sqrt(dX*dX + dY*dY)
        arc_cost=arc_heuristic(goal.x-x2,goal.y-y2,theta2) 
        h2=get_value.value_map[idx(x2)][idx(y2)]+arc_cost
        f2     = g2 + h2
        new_state=state(x2,y2,theta2,g2,f2,h2,delta)
        next_states.append(new_state)
    return next_states

### TRANSFORM VEHICLE CO-ORDINATES 
def transform_vehicle_co_ordinates(vehicle_point_object, next_state, angle_of_rotation):
    displaced_matrix = np.array([next_state[0]-vehicle_point_object.center[0],next_state[1]-vehicle_point_object.center[1]])
    transformed_matrix=np.add(vehicle_point_object.input_co_ordinates,displaced_matrix)
    return vehicle_points(rotate_vehicle_co_ordinates(vehicle_points(transformed_matrix,next_state),angle_of_rotation),next_state)

### TRANSFORM CO-ORDINATES 
def transform_co_ordinates(current_frame, frame_to_be_viewed,point_to_be_transformed):
    displaced_matrix = np.array([frame_to_be_viewed[0]-current_frame[0],frame_to_be_viewed[1]-current_frame[1]])
    transformed_matrix=np.add(np.array(point_to_be_transformed),displaced_matrix)
    return transformed_matrix
  
### ROTATE VEHICLE CO-ORDINATES     
def rotate_vehicle_co_ordinates(vehicle_point_object,angle_of_rotation):
    rotation_matrix = np.array([[np.cos(angle_of_rotation), np.sin(angle_of_rotation)], 
                                [-np.sin(angle_of_rotation), np.cos(angle_of_rotation)]])
    return np.add(vehicle_point_object.center,np.matmul(np.subtract(vehicle_point_object.input_co_ordinates,vehicle_point_object.center), rotation_matrix))
  
  
### CHECK VEHICLE IN SAFE POSITION  
def is_vehicle_in_safe_position(vehicle_point_object,grid):
  for point in vehicle_point_object.input_co_ordinates:
    if(is_within_grid( idx(point[0]),idx(point[1])) and  
       (grid[idx(point[0])][idx(point[1])]==0)):
      continue
    else:
      return False
  return True

### CHK A STAR VEHICLE:
def A_vehicle_is_safe(vehicle_point_A,add_value,grid):
  vp=vehicle_point_A.input_co_ordinates+add_value
  for point in vp:
    if(is_within_grid( idx(point[0]),idx(point[1])) and  
       (grid[idx(point[0])][idx(point[1])]==0)):
      continue
    else:
      return False
  return True
    
    

### EUCLIDEAN DISTANCE
def euclidean_distance(start_point,end_point):
  return np.round(np.sqrt((end_point[0]-start_point[0])**2 +(end_point[1]-start_point[1])**2),4)

### ARC HEURISTIC
def arc_heuristic(x,y,theta_to_be_taken):
  ang_rad=normalize_theta(np.arctan2(y,x))
  diff=np.pi-abs(abs(theta_to_be_taken-ang_rad)-np.pi)
  return diff
  
### NORMALIZE THETA
def normalize_theta(theta):
  if( theta<0 ):
    theta +=( 2*np.pi )
  elif( theta>2*np.pi ):
    theta %=( 2*np.pi)
  return theta

### THETA TO STACK NUMBER
def theta_to_stack_number(theta):
  new = (theta+2*np.pi)%(2*np.pi)
  stack_number = round(new*NUM_THETA_CELLS/2*np.pi)%NUM_THETA_CELLS
  return int(stack_number)

### FLOOR VALUE
def idx(value):
  return int(MT.floor(value))

### CHECK WITHIN GRID  
def is_within_grid(x,y):
  return (x>=0 and x<get_value.grid_on_x and y>=0 and y<get_value.grid_on_y)

### IS_GOAL_REACHED
def is_goal_reached(start,goal):
  result=False
  if( idx(start[0]) == idx(goal[0]) and idx(start[1])==idx(goal[1])):
    result=True
  return result


### BUILDS THE COST MAP
def build_cost_map(current_state,goal,grid):
  print('f',current_state.x,current_state.y)
  expand_grid = [[' ' for x in range(get_value.grid_on_x)] for y in range(get_value.grid_on_y)]
  print('LENGTH',len(expand_grid))
  expand_grid[current_state.x][current_state.y]='*'
  
  open_list = []
  is_goal_attained=False
  open_list.append(current_state)
  #IMPORTANT
  get_value.value_map[current_state.x][current_state.y]=0
  while(len(open_list)>0):
    old_state=open_list.pop(0)
    node=np.array([old_state.x,old_state.y])
    if(goal.x==old_state.x and goal.y==old_state.y):
      is_goal_attained=True
      print("GOAL IS REACHABLE!")
      
    for move in motion_mat:
      nxt_node=node+move
      if( is_within_grid(nxt_node[0],nxt_node[1])):
        if(grid[nxt_node[0]][nxt_node[1]]==0 and expand_grid[nxt_node[0]][nxt_node[1]]!='*'):
          if(A_vehicle_is_safe(vehicle_point_A,np.array([nxt_node]),grid)):
            g2=old_state.g+1
            new_state=state(nxt_node[0],nxt_node[1],0,g2,0,0,0)
            open_list.append(new_state)
            expand_grid[nxt_node[0]][nxt_node[1]]='*'
            get_value.value_map[nxt_node[0]][nxt_node[1]]=g2
  return is_goal_attained
            

### SEARCH ALGORITHM
def Hybrid_A_Star(grid,current_state,goal,vehicle_point_object,speed):
  print("STARTED HYBRID A*")
  start_time = time.time()
  closed = np.array([[[free_space for x in range(get_value.grid_on_x)] for y in range(get_value.grid_on_y)] for cell in range(NUM_THETA_CELLS)])
  came_from = [[[free_space for x in range(get_value.grid_on_x)] for y in range(get_value.grid_on_y)] for cell in range(NUM_THETA_CELLS)]
  is_goal_attained=False
  stack_number=theta_to_stack_number(current_state.theta)
  closed[stack_number][idx(current_state.x)][idx(current_state.y)]=1
  came_from[stack_number][idx(current_state.x)][idx(current_state.y)]=current_state
  total_closed=1
  opened=[current_state]
  
  while (len(opened)>0):
    opened.sort(key=lambda state_srt : float(state_srt.f))
    state_now=opened.pop(0)
    #print([state_now.x,state_now.y,state_now.theta*np.pi/180])
    if(is_goal_reached([idx(state_now.x),idx(state_now.y)],[idx(goal.x),idx(goal.y)])):
      is_goal_attained=True
      print('GOAL REACHED BY HYBRID A*')
      ret_path=path(closed,came_from,state_now)
      end_time = time.time()
      print(end_time - start_time)
      return (is_goal_attained,ret_path)
    
    for evry_state in new_state_transition(state_now,goal,speed):
      #print('Before',[evry_state.x,evry_state.y,evry_state.theta*np.pi/180])
      if(not is_within_grid(idx(evry_state.x),idx(evry_state.y))):
        continue
      
      stack_num=theta_to_stack_number(evry_state.theta)
      #print([stack_num,idx(evry_state.x),idx(evry_state.y)])
      if closed[stack_num][idx(evry_state.x)][idx(evry_state.y)]==0 and grid[idx(evry_state.x)][idx(evry_state.y)]==0:
        new_vehicle_point_obj = transform_vehicle_co_ordinates(vehicle_point_object,[evry_state.x,evry_state.y],evry_state.theta)
        #print(new_vehicle_point_obj.input_co_ordinates)
        if(is_vehicle_in_safe_position(new_vehicle_point_obj,grid)):
            opened.append(evry_state)
            closed[stack_num][idx(evry_state.x)][idx(evry_state.y)]=1
            came_from[stack_num][idx(evry_state.x)][idx(evry_state.y)]=state_now
            total_closed+= 1
            #print('After',[evry_state.x,evry_state.y,evry_state.theta*np.pi/180])
            #plt.plot([state_now.x,evry_state.x],[state_now.y,evry_state.y])
      #closed[stack_num][idx(evry_state.x)][idx(evry_state.y)]=1
        #print('-------------')
  print('No Valid path')
  ret_path=path(closed,came_from,evry_state)
  return (is_goal_attained,ret_path)



### RECONSTRUCT PATH
def reconstruct_path(came_from, start, final):
    path                 = [(final)]
    stack                = theta_to_stack_number(final.theta)
    current              = came_from[stack][idx(final.x)][idx(final.y)]
    stack                = theta_to_stack_number(current.theta)
    while [idx(current.x), idx(current.y)] != [idx(start[0]), idx(start[1])] :
        path.append(current)
        current              = came_from[stack][idx(current.x)][idx(current.y)]
        stack                = theta_to_stack_number(current.theta)
    return path


###DISPLAY PATH
def show_path(path, start, goal,vehicle_pt_obj_act):
  X=[start[0]]
  Y=[start[1]]#ASTAR
  Theta=[]
  #path.reverse()
  X     += [p.x for p in path]
  Y     += [p.y for p in path]
  Theta+=[p.theta for p in path]
  for i in range(len(X)-1):
    Xj=[]
    Yj=[]
    vehicle_pt_obj_now=transform_vehicle_co_ordinates(vehicle_pt_obj_act,[X[i],Y[i]], Theta[i])
    rev=vehicle_pt_obj_now.input_co_ordinates
    revI=rev[:4]
    revL=rev[4:]
    revF=np.concatenate([revI,revL[::-1]])
    l=np.append(revF,[revF[0]],axis=0)
    for i in l:
      Xj.append(i[0])
      Yj.append(i[1])
    plt.plot(Xj,Yj)
  print([np.round(p.steer*180/np.pi,2) for p in path])
  plt.plot(X,Y, color='black')
  plt.scatter([start[0]], [start[1]], color='blue')
  plt.scatter([goal[0]], [goal[1]], color='red')
  plt.scatter([0],[0])
  plt.scatter([99],[99])
  ##############################
  s=[]
  t=[]
  for v in range(len(get_value.GRID_TEST)):
    for w in range(len(get_value.GRID_TEST[0])):
      if(get_value.GRID_TEST[v][w]>0):
        s.append(v)
        t.append(w)
  plt.scatter(v,w)
  #plt.show()
  
### PUT OBSTACLES:
def put_obstacles(X_list,Y_list,grid):
  if(len(X_list)>0):
    for i in  X_list:
      x_XO=[]
      x_YO=[]
      for k in range(i[1],i[2]):
        x_XO.append(i[0])
        x_YO.append(k)
        grid[i[0]][k]=1
      plt.scatter(x_XO,x_YO)
  if(len(Y_list)>0):
    for i in Y_list:
      y_XO=[]
      y_YO=[]
      for k in range(i[1],i[2]):
        y_XO.append(i[0])
        y_YO.append(k)
        grid[k][i[0]]=1
      plt.scatter(y_YO,y_XO)
      
      
      
def build_commands(path,speed):
  ret_list=[]
  #ret_list+=[twist_commands(p.x,p.y,p.theta,p.steer,speed) for p in path]
  #print('f',len(ret_list))
  return ret_list

def search(start,goal_node,present_heading,grid,speed):
  #print(get_value.grid_on_x)
  vehicle_pt_obj=transform_vehicle_co_ordinates(vehicle_pt_obj_actual,start,present_heading)
  current_state = state(vehicle_pt_obj.center[0], vehicle_pt_obj.center[1], present_heading, 0.0, 0.0, 0.0,0.0)
  if(build_cost_map(state(idx(goal_node.x),idx(goal_node.y),0,0,0,0,0),goal(idx(start[0]),idx(start[1])),grid)):
    process_further,ret_val=Hybrid_A_Star(get_value.GRID_TEST,current_state,goal_node,vehicle_pt_obj,speed)
    if(process_further):
      retrieved_path=reconstruct_path(ret_val.came_from,start,ret_val.final)
      retrieved_path.reverse()
      print('PATH',len(retrieved_path))
      # for i in retrieved_path:
      #   print(transform_co_ordinates(fv,cf,[i.x,i.y]))
      #ret_val=build_commands(retrieved_path,speed)

      show_path(retrieved_path,start,[goal_node.x,goal_node.y],vehicle_pt_obj_actual)
      return retrieved_path
    else:
      print("GOAL CANT BE REACHED!!")
  else:
    print("GOAL CANT BE REACHED!!")

#put_obstacles([[24,0,25],[26,0,25],[27,0,25],[60,15,35]],[],GRID_TEST)
### A STAR VEHICLE POINTS
#vehicle_point_A=vehicle_points(np.array([[0,1],[0,-1],[1,0],[-1,0]]),[0,0])
vehicle_point_A=vehicle_points(np.array([[0,2],[0,1],[0,-1],[0,-2],[1,0],[2,0],[-1,0],[-2,0]]),[0,0])
### HYBRID VEHICLE POINTS
#vehicle_pt_obj_actual = vehicle_points( np.array([[0.5,0.5],[0.5,1.5],[0.5,2.5],[0.5,3.5],[1.5,0.5],[1.5,1.5],[1.5,2.5],[1.5,3.5]]),[0,2] )
vehicle_pt_obj_actual = vehicle_points( np.array([[0.5,1.5],[0.5,2.5],[1.5,1.5],[1.5,2.5]]),[0,2] )

## CALL SEARCH

#def velocity_control(cmd_list):


start_state=state(0,0,0,0,0,0,0)



def feed_map(msg):
  if(get_value.feed_map==True):
    get_value.feed_map=False
    print('Im feed map')
    grid_in=msg.data
    grid_x_m=msg.info.width
    grid_y_m=msg.info.height
    coll_cell_side=msg.info.resolution
    grid_on_x = np.int( np.ceil(grid_x_m/coll_cell_side))
    #print('grid_on_x : ',get_value.grid_on_x)
    grid_on_y = np.int( np.ceil(grid_y_m/coll_cell_side) )
    grid_in=np.flipud(np.rot90(np.reshape(grid_in,(grid_x_m,grid_y_m))))
    grid_in[grid_in != 0]=1
    local_grid=grid_in
    #print(local_grid)
    get_value.GRID_TEST=np.pad(local_grid,grid_offset_value,pad_with)
    get_value.grid_on_x=get_value.grid_on_y=get_value.GRID_TEST.shape[0]
    print('shape',get_value.GRID_TEST.shape)
    get_value.value_map = np.array([[1000 for x in range(get_value.grid_on_x)] for y in range(get_value.grid_on_y)])


def get_start_point(msg):
  if(get_value.get_start_point==True):
    get_value.get_start_point=False
    print('Im get_start_point')
    quaternion = (
    msg.pose.pose.orientation.x,
    msg.pose.pose.orientation.y,
    msg.pose.pose.orientation.z,
    msg.pose.pose.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    #print(quaternion)
    #print(msg.pose.position.x)
    #origin=rospy.Subscriber('/map_metadata', MapMetaData, get_origin)
    #print(origin)
    #x=transform_co_ordinates([])
    #print([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z])
    #transformed_point=transform_co_ordinates(cf,fv,[msg.pose.position.x,msg.pose.position.y])
    start_state.x = msg.pose.pose.position.x
    print('Hey 2',msg.pose.pose.position.x)
    start_state.y = msg.pose.pose.position.y
    start_state.theta=yaw



def global_trajectory_call_back(msg):
  if(get_value.is_processing==True):
    get_value.is_processing=False
    path_array=[]
    for pose_loaded in msg.poses:
      path_array.append([pose_loaded.pose.position.x,pose_loaded.pose.position.y])
    print('PATH ARRAY:', path_array)
    print('HEY 1',start_state.x,start_state.y)
    waypoint_tree=KDTree(path_array)
    closest_way_point_id=waypoint_tree.query([start_state.x,start_state.y],1)[1]
    closest_coord=np.array(path_array[closest_way_point_id])
    previous_coord=np.array(path_array[closest_way_point_id-1])
    position_coord=np.array([start_state.x,start_state.y])
    det_val=np.dot(closest_coord-previous_coord,position_coord-closest_coord)
    if(det_val>0):
      closest_way_point_id=(closest_way_point_id+1)%len(path_array)

    way_point_obj=waypoints(np.array(path_array[closest_way_point_id:]),[start_state.x,start_state.y])
    # rospy.Subscriber('/local_map/local_map', OccupancyGrid, feed_map)
    ###CENTER
    #print('get_value.grid_on_x/2 :',get_value.grid_on_x/2)
    transformed_points= transform_frame_co_ordinates(way_point_obj,[idx(get_value.grid_on_x/2),idx(get_value.grid_on_y/2)])
    #waypoint_tree=KDTree(transformed_points.input_co_ordinates)
    #closest_way_point_id=waypoint_tree.query([idx(get_value.grid_on_x/2),idx(get_value.grid_on_y/2)],1)[1]
    #closest_coord=np.array(transformed_points.input_co_ordinates[closest_way_point_id])
    #previous_coord=np.array(transformed_points.input_co_ordinates[closest_way_point_id-1])
    #position_coord=np.array([idx(get_value.grid_on_x/2),idx(get_value.grid_on_y/2)])
    spit_X_Y=np.hsplit(transformed_points.input_co_ordinates,2)
    flatX=np.concatenate(spit_X_Y[0],axis=0)
    flatY=np.concatenate(spit_X_Y[1],axis=0)
    #print('closest_way_point=============',flatX.tolist()[closest_way_point_id],flatY.tolist()[closest_way_point_id])
    coeff=np.polyfit(flatX.tolist(),flatY.tolist(),3)
    print('Im coeff',coeff)
    center_of_circle=[idx(get_value.grid_on_x/2),idx(get_value.grid_on_y/2)]
    radius=11
    #center_of_circle.append(radius)
    print('center',center_of_circle)
    #final_coeff=center_of_circle+coeff.tolist()
    #coeff=np.polyfit(spit_X_Y[0][0],spit_X_Y[1][0],3)
    intersection_roots= find_intersection_X(center_of_circle[0],center_of_circle[1],radius,coeff[0],coeff[1],coeff[2],coeff[3])
    safe_list=[]
    samp_list=[]
    for i in intersection_roots:
      print('before',[i,np.polyval(coeff, i)])
      if(i.imag==0):
          op=np.polyval(coeff, i)
          print('helloo world',[i,op])
          samp_list.append([i.real,op.real])
    setting=0
    final_cord=[]
    waypoint_tree2=KDTree(transformed_points.input_co_ordinates)
    for i in samp_list:
      cl_id=waypoint_tree2.query(i,1)[1]
      if(cl_id>setting):
        setting=cl_id
        final_cord=i
    safe_list.append(final_cord)
    if(len(safe_list)==0):
      safe_list.append(transformed_points.input_co_ordinates[len(transformed_points.input_co_ordinates.tolist())-1])
    s=[]
    t=[]
    print(get_value.GRID_TEST)
    for v in range(len(get_value.GRID_TEST)):
      for w in range(len(get_value.GRID_TEST[0])):
        if(get_value.GRID_TEST[v][w]!=0):
          s.append(v)
          t.append(w)
    plt.scatter(s,t)
    plt.scatter([idx(get_value.grid_on_x/2)],[idx(get_value.grid_on_y/2)],color='red')
    plt.scatter([safe_list[0][0]],[safe_list[0][1]],color='green')
    plt.scatter([0],[0],color='black')
    #plt.show()
    print(start_state.theta)
    returned_path=search([idx(get_value.grid_on_x/2),idx(get_value.grid_on_y/2)],goal(safe_list[0][0],safe_list[0][1]),start_state.theta,get_value.GRID_TEST,3)
    a = np.array([[p.x] for p in returned_path])
    b = np.array([[p.y] for p in returned_path])
    thetas=[[p.theta] for p in returned_path]
    final_array=np.hstack((a,b))
    current_waypoint_obj=waypoints(final_array,[idx(get_value.grid_on_x/2),idx(get_value.grid_on_y/2)])
    #rint('final_array',final_array)
    #print(current_waypoint_obj.center)
    #print([start_state.x,start_state.y])
    final_co_ordinates=transform_frame_co_ordinates(current_waypoint_obj, [start_state.x,start_state.y])
    print('joooooooooooo',[start_state.x,start_state.y])
    path_arr=np.hstack((final_co_ordinates.input_co_ordinates,thetas))
    pub = rospy.Publisher('local_trajectory', Path, queue_size=50)
    path_pub=Path()
    path_pub.header.frame_id='map'
    print('shape',path_arr.shape[0])
    for i in range(path_arr.shape[0]):#
      new_pose=PoseStamped()
      path_now=path_arr[i]
      #pos_on_map=
      quaternion_map=tf.transformations.quaternion_from_euler(0.0, 0.0, path_now[2])
      new_pose.header.seq=i+1
      new_pose.header.frame_id='map'
      new_pose.pose.position.x=path_now[0]
      new_pose.pose.position.y=path_now[1]
      new_pose.pose.orientation.x=quaternion_map[0]
      new_pose.pose.orientation.y=quaternion_map[1]
      new_pose.pose.orientation.z=quaternion_map[2]
      new_pose.pose.orientation.w=quaternion_map[3]
      path_pub.poses.append(new_pose)
    rate = rospy.Rate(50)
    get_value.is_processing=True
    get_value.get_start_point=True
    get_value.feed_map=True
    #while not rospy.is_shutdown():
        #get_value.is_processing=True
    rospy.loginfo('Im publishing')
    pub.publish(path_pub)
    #rate.sleep()
    


if __name__== "__main__":
    rospy.init_node( 'local_planner')
    print('Main')  
    rospy.Subscriber('/trajectory_generator',Path,global_trajectory_call_back)
    rospy.Subscriber('/local_map/local_map', OccupancyGrid, feed_map)
    rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, get_start_point)
    # rate=rospy.Rate(1)
    # rate.sleep()
    # get_value.is_processing=True
    # get_value.get_start_point=True
    rospy.spin()
    

