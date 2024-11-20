import open3d
import numpy

# This takes in an array of points and generates a point cloud

class MeshGenerator:
    """This generates a Mesh using a point cloud, letting one display and save it.
    """
    
    def __init__(self, point_cloud: numpy.ndarray, normal_estimation_point_ref: int = 100, filter_point_cloud: bool = True,
                vox_size: float = .05, nb_neighbors:int = 1000, std_ratios= 1.5, rgb_colors: float = [1, 0, 0]) -> None:
        """Creates a mesh based off the input point cloud. 

        Args:
            point_cloud (numpy.ndarray): The point cloud to be converted into a mesh.
        """
        # Setup the point cloud in an open3d friendly format
        self.point_cloud = open3d.geometry.PointCloud()
        self.point_cloud.points = open3d.utility.Vector3dVector(point_cloud)
        
        if filter_point_cloud:
            # down samples, then removes outlier
            self.point_cloud = self.point_cloud.voxel_down_sample(voxel_size=vox_size)
            self.point_cloud, ind = self.point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratios)
        
        self.point_cloud.estimate_normals()
        # The number is the number of points to reference when creating normals
        self.point_cloud.orient_normals_consistent_tangent_plane(normal_estimation_point_ref)
        
        # Generate the mesh
        self.mesh, self.densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.point_cloud)
        
        # Paint and compute vertex normals for a better view
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color(rgb_colors)
        
        
    

    def SaveMesh(self, file_path: str) -> None:
        """_summary_

        Args:
            file_path (str): The file path to save to. this can be ('.ply', '.stl', '.obj') file extensions.

        """

        isWritten = open3d.io.write_triangle_mesh(file_path, self.mesh)
        if not isWritten:
            print(f"The file {file_path} failed to write.")

    def DisplayMesh(self, showPointCloudToo: bool = False) -> None:
        """Displays the Mesh and optionally the point cloud.

        Args:
            showPointCloudToo (bool, optional): Decide if the mesh should be shown. Defaults to False.
        """
        if showPointCloudToo:
            open3d.visualization.draw_geometries([self.point_cloud, self.mesh])
        else :
            open3d.visualization.draw_geometries([self.mesh])