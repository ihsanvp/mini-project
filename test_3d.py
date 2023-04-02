import open3d as o3d
import numpy as np
import os

FILE = "data/mesh/" + os.listdir("data/mesh")[1]
MAX_VERTICES = 3000
MAX_FACES = 3000

print("Using", FILE)

mesh = o3d.io.read_triangle_mesh(FILE)
pcd = o3d.io.read_point_cloud(FILE)

vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

print(mesh)
print(pcd)
print(vertices.shape)
print(triangles.shape)

mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

mesh_2 = o3d.geometry.TriangleMesh()

padded_vertices = np.zeros((len(vertices), MAX_VERTICES, 3))
for i, v in enumerate(vertices):
    padded_vertices[i, :vertices.shape[0], :] = v

print(vertices.shape, padded_vertices.shape)

mesh_2.vertices = o3d.utility.Vector3dVector(vertices)
mesh_2.triangles = o3d.utility.Vector3iVector(triangles)
mesh_2.triangle_uvs = o3d.utility.Vector2dVector(
    np.random.rand(len(triangles) * 3, 2))

mesh_2.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_2])

o3d.io.write_triangle_mesh("mesh.obj", mesh_2, write_triangle_uvs=True)
