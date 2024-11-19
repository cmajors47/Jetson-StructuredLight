import open3d
import numpy

# This takes in an array of points and generates a point cloud

class MeshGenerator:
    """This generates a Mesh using a point cloud, letting one display and save it.
    """
    
    def __init__(self, point_cloud: numpy.ndarray, normal_estimation_point_ref: int = 100) -> None:
        """Creates a mesh based off the input point cloud. 

        Args:
            point_cloud (numpy.ndarray): The point cloud to be converted into a mesh.
        """
        # Setup the point cloud in an open3d friendly format
        self.point_cloud = open3d.geometry.PointCloud()
        self.point_cloud.points = open3d.utility.Vector3dVector(point_cloud)
        self.point_cloud.estimate_normals()
        # The number is the number of points to reference when creating normals
        self.point_cloud.orient_normals_consistent_tangent_plane(normal_estimation_point_ref)
        
        # Generate the mesh
        self.mesh, self.densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.point_cloud)
        
    

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