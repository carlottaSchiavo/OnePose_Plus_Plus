import hydra
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm
from loguru import logger
import os
os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module, this line should before import torch
import os.path as osp
import glob
import numpy as np
import natsort
import torch
import argparse
import src.utils.metric_utils
from src.utils import data_utils
from src.utils import vis_utils
from src.utils.metric_utils import query_pose_error,projection_2d_error
import glob
from scipy.spatial import distance,KDTree,ConvexHull
import open3d as o3d
import plyfile


#-----------------------------------------------------------------------------------------#
# ---------------------------------Utilties functions-------------------------------------#
#-----------------------------------------------------------------------------------------#

def get_paths(data_test, data_gt, data_dir):

    pred_poses_dir = osp.join(data_test, 'pred_poses')
    gt_poses_dir=osp.join(data_gt, 'poses')

    paths = {
        'data_dir':data_dir,
        'pred_pose_dir': pred_poses_dir,
        'gt_pose_dir':gt_poses_dir
    }
    return  paths


def compute_metrics(pred_poses,gt_poses):
    mean_angular_distance=0
    mean_translation_distance=0


    for (id_pred,pred_pose),(id_gt,gt_pose) in zip(pred_poses.items(),gt_poses.items()):
        ang_dist, t_dist=query_pose_error(pred_pose,gt_pose)
        mean_angular_distance+=ang_dist
        mean_translation_distance+=t_dist

    return mean_angular_distance/len(pred_poses),mean_translation_distance/len(gt_poses)

def load_point_cloud(pcl_path):
    with open(pcl_path, "rb") as f:
        plydata = plyfile.PlyData.read(f)
        xyz = np.stack([np.array(plydata["vertex"][c]).astype(float) for c in ("x", "y", "z")], axis=1)
    return xyz

def get_diameter(pointcloud):

    pc=load_point_cloud(pointcloud)
     # Compute the convex hull of the point cloud
    hull = ConvexHull(pc)
    
    # Extract the points that are vertices of the convex hull
    hull_points = pc[hull.vertices]
    
    # Build a KD-tree with the convex hull points
    tree = KDTree(hull_points)
    
    # Initialize the maximum distance
    max_distance = 0
    
    # Check the farthest neighbor for each point on the convex hull
    for point in hull_points:
        # Query the farthest distance in the KD-tree
        dists, _ = tree.query(point, k=len(hull_points))
        max_distance = max(max_distance, np.max(dists))
    return max_distance


def project_points(M, R, T, K):
    """
    Projects 3D model points to the 2D image plane using the given pose and camera intrinsics.
    
    Args:
    - M (numpy array of shape (N, 3)): 3D model points.
    - R (numpy array of shape (3, 3)): Rotation matrix.
    - T (numpy array of shape (3,)): Translation vector.
    - K (numpy array of shape (3, 3)): Camera intrinsic matrix.
    
    Returns:
    - projections (numpy array of shape (N, 2)): 2D projections of the 3D points.
    """
    # Transform model points using the pose
    M_transformed = np.dot(M, R.T) + T
    
    # Project points using the camera intrinsic matrix
    M_projected = np.dot(M_transformed, K.T)
    
    # Normalize by the third coordinate to get the 2D points
    projections = M_projected[:, :2] / M_projected[:, 2, np.newaxis]
    
    return projections

#-----------------------------------------------------------------------------------------#
# ---------------------------------Metrics computation -----------------------------------#
#-----------------------------------------------------------------------------------------#

def ADD_metric(M, gt_pose, pred_pose):
    """
    Calculate the ADD metric for asymmetric objects.
    
    Args:
    - M (numpy array of shape (N, 3)): 3D model points.
    - R_gt (numpy array of shape (3, 3)): Ground truth rotation matrix.
    - T_gt (numpy array of shape (3,)): Ground truth translation vector.
    - R_pred (numpy array of shape (3, 3)): Predicted rotation matrix.
    - T_pred (numpy array of shape (3,)): Predicted translation vector.

    Returns:
    - float: The ADD metric value.
    """
    
    R_gt=gt_pose[0:3,0:3]
    T_gt=gt_pose[0:3,3]

    R_pred=pred_pose[0:3,0:3]
    T_pred=pred_pose[0:3,3]
    
    # Transform model points using ground truth pose
    M_gt = np.dot(M, R_gt.T) + T_gt
    # Transform model points using predicted pose
    M_pred = np.dot(M, R_pred.T) + T_pred
    # Compute the average distance
    add = np.mean(np.linalg.norm(M_gt - M_pred, axis=1))
    return add

def ADD_s_metric(M, R_gt, T_gt, R_pred, T_pred):
    """
    Calculate the ADD-S metric for symmetric objects.
    
    Args:
    - M (numpy array of shape (N, 3)): 3D model points.
    - R_gt (numpy array of shape (3, 3)): Ground truth rotation matrix.
    - T_gt (numpy array of shape (3,)): Ground truth translation vector.
    - R_pred (numpy array of shape (3, 3)): Predicted rotation matrix.
    - T_pred (numpy array of shape (3,)): Predicted translation vector.

    Returns:
    - float: The ADD-S metric value.
    """
    # Transform model points using ground truth pose
    M_gt = np.dot(M, R_gt.T) + T_gt
    # Transform model points using predicted pose
    M_pred = np.dot(M, R_pred.T) + T_pred
    
    # For each transformed model point in M_gt, find the minimum distance in M_pred
    add_s = np.mean([np.min(np.linalg.norm(M_gt[i] - M_pred, axis=1)) for i in range(len(M))])
    return add_s

def check_ADD_threshold(add_metric_value, diameter,perc_accurancy):
    """
    Check if the pose estimate is considered correct.
    
    Args:
    - add_metric_value (float): The ADD or ADD-S metric value.
    - diameter (float): The object diameter.
    
    Returns:
    - bool: True if the pose is correct, False otherwise.
    """
    return add_metric_value < perc_accurancy * diameter


def compute_all_ADD(M, GT_poses, pred_poses, diameter,perc_acc, symmetric=False):
    """
    Calculate the ADD or ADD-S metrics for a set of poses.
    
    Args:
    - M (numpy array of shape (N, 3)): 3D model points.
    - GT_poses (list of tuples): List of ground truth poses, each containing (R_gt, T_gt).
    - pred_poses (list of tuples): List of predicted poses, each containing (R_pred, T_pred).
    - diameter (float): The object diameter.
    - symmetric (bool): Whether the object is symmetric.
    
    Returns:
    - metrics (list of floats): List of ADD or ADD-S metric values.
    - correctness (list of bools): List indicating whether each pose is correct.
    """
    metrics = []
    correctness = []

    for (id_gt,GT_pose), (id_pred,pred_pose) in zip(GT_poses.items(), pred_poses.items()):
        if symmetric:
            add_value = ADD_s_metric(M, GT_pose, pred_pose)
        else:
            add_value = ADD_metric(M, GT_pose, pred_pose)
        
        metrics.append(add_value)
        correctness.append(check_ADD_threshold(add_value, diameter,perc_acc))

    return metrics, correctness





def proj2D_metric(M, gt_pose, pred_pose, K):
    """
    Calculate the 2D projection metric between ground truth and predicted poses.
    
    Args:
    - M (numpy array of shape (N, 3)): 3D model points.
    - R_gt (numpy array of shape (3, 3)): Ground truth rotation matrix.
    - T_gt (numpy array of shape (3,)): Ground truth translation vector.
    - R_pred (numpy array of shape (3, 3)): Predicted rotation matrix.
    - T_pred (numpy array of shape (3,)): Predicted translation vector.
    - K (numpy array of shape (3, 3)): Camera intrinsic matrix.
    
    Returns:
    - float: The mean 2D projection distance.
    """
    R_gt=gt_pose[0:3,0:3]
    T_gt=gt_pose[0:3,3]

    R_pred=pred_pose[0:3,0:3]
    T_pred=pred_pose[0:3,3]

    # Project points using ground truth pose
    projections_gt = project_points(M, R_gt, T_gt, K)
    
    # Project points using predicted pose
    projections_pred = project_points(M, R_pred, T_pred, K)
    
    # Compute the mean distance between the 2D projections
    projection_error = np.mean(np.linalg.norm(projections_gt - projections_pred, axis=1))
    
    return projection_error

def check_proj2D_threshold(projection_error, pixel_threshold):
    """
    Check if the pose estimate is considered correct based on 2D projection error.
    
    Args:
    - projection_error (float): The mean 2D projection distance.
    - pixel_threshold (float): The threshold in pixels to consider the pose correct.
    
    Returns:
    - bool: True if the pose is correct, False otherwise.
    """
    return projection_error < pixel_threshold

def compute_all_proj2D(M, GT_poses, pred_poses, K, pixel_threshold):
    """
    Calculate the proj2D metrics for a set of poses.
    
    Args:
    - M (numpy array of shape (N, 3)): 3D model points.
    - GT_poses (list of tuples): List of ground truth poses, each containing (R_gt, T_gt).
    - pred_poses (list of tuples): List of predicted poses, each containing (R_pred, T_pred).
    - K (numpy array of shape (3, 3)): Camera intrinsic matrix.
    - pixel_threshold (float): The threshold in pixels to consider the pose correct.

    Returns:
    - metrics (list of floats): List of proj2D metric values.
    - correctness (list of bools): List indicating whether each projection is correct.
    """
    metrics = []
    correctness = []

    for (id_gt,GT_pose), (if_pred,pred_pose) in zip(GT_poses.items(), pred_poses.items()):
        
        projErr=proj2D_metric(M,GT_pose,pred_pose,K)
        #projErr=projection_2d_error(M,GT_pose,pred_pose,K)
        print("projErr: ",projErr)
        metrics.append(projErr)
        correctness.append(check_proj2D_threshold(projErr, pixel_threshold))

    return metrics, correctness


def proj2D_bbox_metric(bbox3d,gt_pose,pred_pose,K,pixel_threshold):
   
    """
    Compute the projection error of the bounding box and check if it is within the threshold.
    
    Args:
    - bbox_corners (numpy array of shape (8, 3)): 3D corners of the bounding box.
    - R_gt (numpy array of shape (3, 3)): Ground truth rotation matrix.
    - T_gt (numpy array of shape (3,)): Ground truth translation vector.
    - R_pred (numpy array of shape (3, 3)): Predicted rotation matrix.
    - T_pred (numpy array of shape (3,)): Predicted translation vector.
    - K (numpy array of shape (3, 3)): Camera intrinsic matrix.
    - threshold (float): The pixel threshold for determining accuracy.
    
    Returns:
    - float: The mean projection error of the bounding box in pixels.
    - bool: Whether the error is within the specified threshold.
    """
    R_gt=gt_pose[0:3,0:3]
    T_gt=gt_pose[0:3,3]

    R_pred=pred_pose[0:3,0:3]
    T_pred=pred_pose[0:3,3]

    # Project bounding box corners using ground truth pose
    projections_gt = project_points(bbox3d, R_gt, T_gt, K)
    
    # Project bounding box corners using predicted pose
    projections_pred = project_points(bbox3d, R_pred, T_pred, K)
    
    # Compute the mean distance between the projected bounding box corners
    projection_error = np.mean(np.linalg.norm(projections_gt - projections_pred, axis=1))
    
    # Check if the error is within the threshold
    is_accurate = projection_error < pixel_threshold
    
    return projection_error, is_accurate

def compute_all_proj2D_bbox(bbox3d, GT_poses, pred_poses, K, pixel_threshold):
    """
    Calculate the proj2D bbox metrics for a set of poses.
    
    Args:
    - bbox3d (array of coordinates): list of bbox corners 
    - GT_poses (list of tuples): List of ground truth poses, each containing (R_gt, T_gt).
    - pred_poses (list of tuples): List of predicted poses, each containing (R_pred, T_pred).
    - K (numpy array of shape (3, 3)): Camera intrinsic matrix.
    - pixel_threshold (float): The threshold in pixels to consider the pose correct.

    Returns:
    - metrics (list of floats): List of proj2D metric values.
    - correctness (list of bools): List indicating whether each projection is correct.
    """
    metrics = []
    correctness = []

    for (id_gt,GT_pose), (if_pred,pred_pose) in zip(GT_poses.items(), pred_poses.items()):
        
        projErr,is_accurate=proj2D_bbox_metric(bbox3d,GT_pose,pred_pose,K,pixel_threshold)
        metrics.append(projErr)
        correctness.append(check_proj2D_threshold(projErr, pixel_threshold))

    return metrics, correctness



def aggregate_ADD_and_proj2D_bboxes(point_cloud_path,bbox3d,poses_pred,poses_gt,K,perc_ADD,pixel_threshold,model_unit='m'):

    diameter=get_diameter(point_cloud_path)
    model_3D_pts=load_point_cloud(point_cloud_path)

    #metrics computation...
    add,corr_add=compute_all_ADD(model_3D_pts,poses_gt,poses_pred,diameter,perc_ADD)
    proj,corr_proj=compute_all_proj2D_bbox(bbox3d,poses_gt,poses_pred,K,pixel_threshold)

    mean_add_metric = np.mean(add)
    mean_proj_metric = np.mean(proj)

    return mean_add_metric,mean_proj_metric
    



    

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_test", type=str, required=True)
    parser.add_argument("--data_gt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)

    args = parser.parse_args()
    return args


#def main(cfg: DictConfig):
def main():
    
    args=parse_args()
    data_test=args.data_test
    data_gt=args.data_gt
    data_dir=args.data_dir
    paths=get_paths(data_test,data_gt,data_dir)

    #put all the poses matrices into a map
    pred_poses=dict()
    gt_poses=dict()

    path_pred=paths['data_dir']+"/"+paths['pred_pose_dir']+"/"
    path_gt=paths['data_dir']+"/"+paths['gt_pose_dir']+"/"
    path_k=data_dir+"/"+data_gt+"/"
    path_bbox3d=data_dir+"/"+"box3d_corners.txt"
    #pred_poses
    for filename in glob.glob(os.path.join(path_pred,'*.txt')):
        with open(filename,'r') as f:
            tran_matrix=np.loadtxt(f,dtype=float)
            id=os.path.splitext(os.path.basename(filename))[0]
            pred_poses[id]=tran_matrix

    #gt_poses
    for filename in glob.glob(os.path.join(path_gt,"*.txt")):
        with open(filename,'r') as f:
            tran_matrix=np.loadtxt(f,dtype=float)
            id=os.path.splitext(os.path.basename(filename))[0]
            gt_poses[id]=tran_matrix


    #K generation
    coeff=[]
    with open(path_k+"intrinsics.txt",'r') as f:
        file_cont=f.read()
    for l in file_cont.splitlines():
        c=float(l.split(':')[1].strip())
        coeff.append(c)

    bbox3d=[]
    with open(path_bbox3d,'r') as f:
        bbox3d=np.loadtxt(f,dtype=float)
    
  

    K=np.asarray([[coeff[0], 0.0,coeff[2] ],[0.0, coeff[1], coeff[3]],[0.0, 0.0, 1.0]],np.float32)

    #read pointcloud
    model_3D_pts_file=data_dir+"/model.ply"
   

    mean_add_metric,mean_proj_metric=aggregate_ADD_and_proj2D_bboxes(model_3D_pts_file,bbox3d,pred_poses,gt_poses,K,perc_ADD=0.1,pixel_threshold=50,model_unit='m')

    print("ADD: ",mean_add_metric)
    print("proj2D: ",mean_proj_metric)
    

if __name__ == "__main__":
    main()
