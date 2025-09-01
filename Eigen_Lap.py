"""
This script contains functions to calculate the Laplacian matrix and its eigenvalues/eigenfunctions for 3D meshes.

Key variables:
- N_verts: number of vertices
- N_faces: number of faces
- X: coordinates of vertices, shape (N_verts, 3)
- Tri: indices of triangles, shape (N_faces, 3)
- num: number of smallest eigenpairs to compute
- L: Laplacian matrix, shape (N_verts, N_verts)
- M: mass matrix (diagonal), shape (N_verts, N_verts)
- Q: eigenvalues, shape (num,)
- V: eigenfunctions, shape (N_verts, num)
"""

import autograd.numpy as np
import trimesh
from scipy.sparse import csr_matrix, lil_matrix, diags
from scipy.special import cotdg
from scipy.sparse.linalg import eigsh
import numpy

def lap_mat(X, Tri, type):
    """
    Calculate Laplacian matrix for a 3D mesh using the cotangent formula:
    L_ij = 1/2(cot α_ij + cot β_ij) for j ∈ N(i)
    """
    RAD_TO_DEG = 180.0 / np.pi

    # Normalize the mesh
    if type == 'normalized':
      av_edge = X[Tri[:, 0:2]]
      av_edge_length = np.linalg.norm(av_edge[:, 1, :] - av_edge[:, 0, :], axis=1).mean() # scale the mesh to make the average edge length is 1.
      X = (X - X.mean(axis=0)) / av_edge_length

    # Create a trimesh object
    # mesh = trimesh.Trimesh(vertices=X, faces=Tri, process=False)
    # Build trimesh object and try to clean it
    mesh = trimesh.Trimesh(vertices=X, faces=Tri, process=False)
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()

    if mesh.faces.shape[0] == 0:
        raise ValueError("Mesh has no valid faces after cleaning. Check input Tri.")

    # Use the cleaned mesh data to ensure consistency
    # After cleaning, mesh.vertices and mesh.faces are the authoritative data
    X_clean = mesh.vertices
    Tri_clean = mesh.faces

    # Calculate face areas and angles
    areas = mesh.area_faces
    angles = mesh.face_angles # angles within each face, ordered same way as vertices are listed in face

    # Calculate vertex areas (|Ω_i|) using barycentric method
    vertex_faces = mesh.vertex_faces
    MA = numpy.ma.masked_array(areas[vertex_faces], vertex_faces < 0)  # Masked array for vertex faces, vertex_faces is padded with -1s
    M = MA.sum(axis=1) / 3.0  # Barycentric area approximation
    M = np.array(M)  # Convert to autograd.numpy array

    # Initialize Laplacian matrix
    L = lil_matrix((X_clean.shape[0], X_clean.shape[0]))

    # Fill Laplacian matrix using cotangent weights
    for ff, face in enumerate(Tri_clean):
        cot_ang = cotdg(RAD_TO_DEG * angles[ff, 2])
        L[face[0], face[1]] += cot_ang
        L[face[1], face[0]] += cot_ang

        cot_ang = cotdg(RAD_TO_DEG * angles[ff, 0])
        L[face[1], face[2]] += cot_ang
        L[face[2], face[1]] += cot_ang

        cot_ang = cotdg(RAD_TO_DEG * angles[ff, 1])
        L[face[2], face[0]] += cot_ang
        L[face[0], face[2]] += cot_ang

    # Set diagonal elements
    L.setdiag(-L.sum(axis=1), k=0) # set each diagonal element to the negative sum of its row

    L = L.tocsr() # Convert to CSR format
    L = L.multiply(-0.5)  # multiply by (-0.5) to get the final Laplacian matrix

    # Create diagonal mass matrix
    M = diags(M)

    return L, M

def lap_eigen(X, Tri, num):
    """Solve Laplacian eigenproblem."""
    print(f"Computing {num} eigenfunctions with smallest eigenvalues")

    LS, M = lap_mat(X, Tri, type='normalized')
    # LS: Laplacian matrix, shape(N_verts, N_verts);
    # M: Mass matrix, shape(N_verts, N_verts).


    # Solve the generalized eigenvalue problem: LS * x = λ * M * x, using eigsh, because LS is symmetric.
    Q, V = eigsh(LS, k=num, sigma=0, which="LM", M=M)
    # Q: eigenvalues, V: eigenvectors (eigenfunctions)
    # eigsh is used for symmetric matrices
    # k: number of eigenvalues/vectors to compute
    # sigma=0, which="LM": find eigenvalues closest to 0
    # M: mass matrix for generalized eigenvalue problem

    # Remove any tiny imaginary parts
    Q, V = np.real(Q), np.real(V)
    Q[Q < 0] = 0.0  # Ensure non-negative eigenvalues

    return Q, V

def eigen_sol(X, Tri, num):
    """Main function"""
    Q, V = lap_eigen(X, Tri, num)

    return Q, V