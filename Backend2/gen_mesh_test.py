import open3d as o3d
from SLC_Library.Mesh_Generator import MeshGenerator
import cv2 as cv
import numpy as np

point_file = cv.FileStorage("3Dpoints.xml", cv.FILE_STORAGE_READ)

point_cloud = point_file.getNode('a3D_Points').mat()

point_file.release()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

downpcd = pcd.voxel_down_sample(voxel_size=0.05)
print(len(downpcd.points))
new_point, ind = downpcd.remove_statistical_outlier(nb_neighbors=100,std_ratio=2.0)
#print(ind)
print(new_point)
downpcd.estimate_normals()
# o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("out.ply", downpcd)
# mesh.DisplayMesh(True)

