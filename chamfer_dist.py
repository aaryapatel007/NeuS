import os
import numpy as np
import json, argparse
import trimesh
import trimesh.proximity
import trimesh.sample


# Arguments
parser = argparse.ArgumentParser(
    description='Evaluation'
)
parser.add_argument('--mesh_gt', type=str, required=True,)
parser.add_argument('--mesh_pred', type=str, required=True,)
parser.add_argument('--num_samples', type=int, default=10000,)
parser.add_argument('--apply_transform', default=False, action = 'store_true')
args = parser.parse_args()

def get_chamfer_dist(src_mesh, tgt_mesh, case_name, num_samples=10000, apply_transformation = False):
    
    with open ('/home/ojaswa/aarya/research_papers_implementation/NeuS/public_data/objects/trans.json') as f:
        transformations = json.load(f)
    
    if apply_transformation:
        vertices = src_mesh.vertices
        
        center = transformations[case_name]['center']
        scale = transformations[case_name]['scale']

        # Apply the transformation to the mesh
        vertices_new = (vertices - center) / scale

        src_mesh.vertices = vertices_new
        
    # Chamfer
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgt_mesh, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(src_mesh, tgt_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    src_tgt_dist_mean = src_tgt_dist.mean()
    tgt_src_dist_mean = tgt_src_dist.mean()

    chamfer_dist = (src_tgt_dist_mean + tgt_src_dist_mean) / 2
    
    src_mesh_vertices = src_mesh.vertices
    tgt_mesh_vertices = tgt_mesh.vertices
    
    print('src_mesh_vertices:', src_mesh_vertices.shape)
    print('tgt_mesh_vertices:', tgt_mesh_vertices.shape)

    return chamfer_dist

case_name = args.mesh_gt.split('/')[-1].split('.')[0]
apply_transformation = args.apply_transform
mesh_gt = trimesh.load(args.mesh_gt)
mesh_pred = trimesh.load(args.mesh_pred)
chamfer = get_chamfer_dist(mesh_pred, mesh_gt, case_name, num_samples=args.num_samples, apply_transformation = apply_transformation)
print(f'Chamfer Distance (mm):  {chamfer*1000:.3f}')