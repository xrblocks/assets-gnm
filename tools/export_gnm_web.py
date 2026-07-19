"""Exports the GNM head model + semantic samplers to compact web binaries.

Reads the GNM npz model, the 68-landmark definition, and the two Keras CVAE
decoder .h5 files, and writes two container files consumed by the XR Blocks
GNM demo:

  assets/gnm_head_web.bin      quantized model (bases as int8 + f32 scales)
  assets/gnm_samplers_web.bin  identity/expression decoder MLP weights (f32)

Container layout (little endian):
  bytes 0..3   magic 'GNMW'
  uint32       format version (1)
  uint32       JSON header byte length
  ...          JSON header (utf-8)
  ...          binary sections, each 4-byte aligned, described by the header

Usage:
  python export_gnm_web.py --gnm_root=path/to/gnm_repo --out_dir=../assets

Requires only numpy + h5py (no TensorFlow).
"""

import argparse
import json
import os
import re

import h5py
import numpy as np

# Class label names mirrored from gnm/shape/semantic_sampler.py.
GENDERS = ['female', 'male']
ETHNICITIES = ['middle_eastern', 'asian', 'white', 'black']
EXPRESSION_CLASSES = [
    'surprise', 'disgust', 'suck', 'compress_face', 'stretch_face', 'happy',
    'squint', 'platysma', 'blow', 'funneler', 'smile_wide', 'corners_down',
    'pucker', 'wink_left', 'wink_right', 'mouth_left', 'mouth_right',
    'lips_roll_in', 'snarl', 'tongue_center'
]

# Per-vertex material ids (priority ordered, later wins).
MATERIALS = ['skin', 'teeth', 'gums', 'tongue', 'scleras', 'irises', 'pupils']

DTYPE_NAMES = {
    np.dtype(np.float32): 'float32',
    np.dtype(np.int8): 'int8',
    np.dtype(np.uint8): 'uint8',
    np.dtype(np.uint16): 'uint16',
    np.dtype(np.int32): 'int32',
}


class ContainerWriter:
  """Accumulates named binary sections and writes a GNMW container file."""

  def __init__(self):
    self.sections = []
    self.blobs = []
    self.offset = 0

  def add(self, name, array):
    array = np.ascontiguousarray(array)
    dtype = DTYPE_NAMES[array.dtype]
    data = array.tobytes()
    self.sections.append({
        'name': name,
        'dtype': dtype,
        'shape': list(array.shape),
        'offset': self.offset,
        'byteLength': len(data),
    })
    pad = (-len(data)) % 4
    self.blobs.append(data + b'\x00' * pad)
    self.offset += len(data) + pad

  def write(self, path, meta):
    # Section offsets are relative to the end of the header; the reader adds
    # the base (12 + header length) itself.
    header = {'meta': meta, 'sections': self.sections}
    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    pad = (-len(header_bytes)) % 4
    header_bytes += b' ' * pad
    preamble = b'GNMW' + np.array([1, len(header_bytes)],
                                  dtype='<u4').tobytes()
    with open(path, 'wb') as f:
      f.write(preamble)
      f.write(header_bytes)
      for blob in self.blobs:
        f.write(blob)
    total = len(preamble) + len(header_bytes) + self.offset
    print(f'Wrote {path} ({total / 1e6:.2f} MB)')


def quantize_int8(matrix):
  """Symmetric per-row int8 quantization. matrix: (rows, cols) float32."""
  scales = np.abs(matrix).max(axis=1) / 127.0
  scales = np.maximum(scales, 1e-12).astype(np.float32)
  q = np.clip(np.round(matrix / scales[:, None]), -127, 127).astype(np.int8)
  return q, scales


def group_weight_matrix(groups, names, lookup):
  """Stacks the vertex-group weight rows for the requested group names."""
  index = {n: i for i, n in enumerate(names)}
  return np.stack([groups[index[n]] for n in lookup], axis=0)


def load_keras_mlp(path):
  """Extracts ordered Dense (kernel, bias, activation) triples from a h5."""
  with h5py.File(path, 'r') as f:
    config = json.loads(f.attrs['model_config'])
    layers = []
    for layer in config['config']['layers']:
      if layer['class_name'] != 'Dense':
        continue
      name = layer['config']['name']
      activation = layer['config']['activation']
      group = f['model_weights'][name]
      # Keras nests weights under a scope with the layer name.
      while 'kernel:0' not in group:
        group = group[list(group.keys())[0]]
      kernel = np.array(group['kernel:0'], dtype=np.float32)
      bias = np.array(group['bias:0'], dtype=np.float32)
      layers.append({'name': name, 'activation': activation,
                     'kernel': kernel, 'bias': bias})
    return layers


def run_mlp(layers, x):
  for i, layer in enumerate(layers):
    x = x @ layer['kernel'] + layer['bias']
    if layer['activation'] == 'relu':
      x = np.maximum(x, 0.0)
  return x


def main():
  parser = argparse.ArgumentParser()
  script_dir = os.path.dirname(os.path.abspath(__file__))
  default_root = os.path.normpath(
      os.path.join(script_dir, '..', '..', '..', 'GNM'))
  parser.add_argument('--gnm_root', default=default_root,
                      help='Path to the GNM repository checkout.')
  parser.add_argument('--out_dir',
                      default=os.path.join(script_dir, '..', 'assets'))
  args = parser.parse_args()

  shape_dir = os.path.join(args.gnm_root, 'gnm', 'shape')
  npz_path = os.path.join(shape_dir, 'data', 'versions', 'v3_0',
                          'gnm_head.npz')
  landmarks_path = os.path.join(shape_dir, 'data', 'landmarks',
                                'head_sparse_68.txt')
  sampler_dir = os.path.join(shape_dir, 'data', 'semantic_sampler')
  os.makedirs(args.out_dir, exist_ok=True)

  d = np.load(npz_path)
  num_vertices = d['template_vertex_positions'].shape[0]
  num_joints = d['template_joint_positions'].shape[0]
  identity_dim = d['vertex_identity_basis'].shape[0]
  expression_dim = d['expression_basis'].shape[0]
  print(f'GNM {d["variant"]} v{d["version"]}: {num_vertices} vertices, '
        f'{identity_dim} identity, {expression_dim} expression, '
        f'{num_joints} joints')

  # ---- Quantize the large bases (per-component int8). -----------------------
  identity_basis = d['vertex_identity_basis'].reshape(identity_dim, -1)
  expression_basis = d['expression_basis'].reshape(expression_dim, -1)
  correctives = d['pose_correctives_regressor']  # (9J, 3V)

  identity_q, identity_scales = quantize_int8(identity_basis)
  expression_q, expression_scales = quantize_int8(expression_basis)
  # The v3.0 open-source release ships an all-zero correctives regressor; only
  # export it when it carries signal.
  has_pose_correctives = bool(np.abs(correctives).max() > 0)
  if has_pose_correctives:
    correctives_q, correctives_scales = quantize_int8(correctives)

  # ---- Validate quantization error. ----------------------------------------
  rng = np.random.default_rng(0)
  coeff_id = rng.normal(size=(16, identity_dim)).astype(np.float32)
  coeff_ex = rng.normal(size=(16, expression_dim)).astype(np.float32)
  exact = coeff_id @ identity_basis + coeff_ex @ expression_basis
  approx = ((coeff_id * identity_scales) @ identity_q.astype(np.float32) +
            (coeff_ex * expression_scales) @ expression_q.astype(np.float32))
  err = np.abs(exact - approx)
  print(f'Bind-pose int8 error: max {err.max() * 1000:.4f} mm, '
        f'mean {err.mean() * 1000:.5f} mm')

  if has_pose_correctives:
    pose_features = rng.uniform(-0.6, 0.6, size=(16, 9 * num_joints)).astype(
        np.float32)
    exact_pc = pose_features @ correctives
    approx_pc = (pose_features * correctives_scales) @ correctives_q.astype(
        np.float32)
    err_pc = np.abs(exact_pc - approx_pc)
    print(f'Pose-correctives int8 error: max {err_pc.max() * 1000:.4f} mm, '
          f'mean {err_pc.mean() * 1000:.5f} mm')
  else:
    print('Pose-correctives regressor is all zeros; skipping export.')

  # ---- Per-vertex classification maps. -------------------------------------
  groups = d['vertex_groups']
  group_names = [str(n) for n in d['vertex_group_names']]

  component_names = [str(n) for n in d['mesh_component_names']]
  component_weights = group_weight_matrix(groups, group_names, component_names)
  component_id = component_weights.argmax(axis=0).astype(np.uint8)

  material_id = np.zeros(num_vertices, dtype=np.uint8)
  for material_index, material in enumerate(MATERIALS):
    if material == 'skin':
      continue
    weights = group_weight_matrix(groups, group_names, [material])[0]
    material_id[weights > 1e-4] = material_index

  # The sclera/iris/pupil labels live on the interior eye layer; the visible
  # surface is the transparent cornea shell (eye_exteriors). Project the
  # interior colors onto the shell so eyes render with sclera + iris.
  template = d['template_vertex_positions']
  exterior = np.where(
      group_weight_matrix(groups, group_names, ['eye_exteriors'])[0] > 1e-4
  )[0]
  interior = np.where(
      group_weight_matrix(groups, group_names, ['eye_interiors'])[0] > 1e-4
  )[0]
  if len(exterior) and len(interior):
    d2 = ((template[exterior][:, None, :] -
           template[interior][None, :, :]) ** 2).sum(-1)
    material_id[exterior] = material_id[interior[d2.argmin(axis=1)]]

  region_names = [n for n in group_names if n.endswith('_region')]
  region_weights = group_weight_matrix(groups, group_names, region_names)
  region_max = region_weights.max(axis=0)
  region_id = region_weights.argmax(axis=0).astype(np.uint8)
  region_id[region_max <= 1e-4] = 255

  # ---- Landmarks. -----------------------------------------------------------
  landmark_rows = []
  with open(landmarks_path, 'r', encoding='utf-8') as f:
    for line in f:
      parts = line.split()
      if len(parts) >= 6:
        landmark_rows.append([float(x) for x in parts[:6]])
  landmarks = np.array(landmark_rows, dtype=np.float64)
  landmark_indices = landmarks[:, 0::2].astype(np.uint16)
  landmark_weights = landmarks[:, 1::2].astype(np.float32)
  print(f'Landmarks: {landmark_indices.shape[0]}')

  # ---- Write the model container. ------------------------------------------
  writer = ContainerWriter()
  writer.add('template', d['template_vertex_positions'].astype(np.float32))
  writer.add('triangles', d['triangles'].astype(np.uint16))
  writer.add('quads', d['quads'].astype(np.uint16))
  writer.add('template_joints',
             d['template_joint_positions'].astype(np.float32))
  writer.add('joint_parents', d['joint_parent_indices'].astype(np.int32))
  writer.add('skinning_weights', d['skinning_weights'].astype(np.float32))
  writer.add('joint_identity_basis',
             d['joint_identity_basis'].astype(np.float32))
  writer.add('identity_basis', identity_q)
  writer.add('identity_scales', identity_scales)
  writer.add('expression_basis', expression_q)
  writer.add('expression_scales', expression_scales)
  if has_pose_correctives:
    writer.add('pose_correctives', correctives_q)
    writer.add('pose_correctives_scales', correctives_scales)
  writer.add('component_id', component_id)
  writer.add('material_id', material_id)
  writer.add('region_id', region_id)
  writer.add('landmark_indices', landmark_indices)
  writer.add('landmark_weights', landmark_weights)

  meta = {
      'model': 'GNM Head',
      'gnmVersion': str(d['version']),
      'variant': str(d['variant']),
      'numVertices': int(num_vertices),
      'numJoints': int(num_joints),
      'identityDim': int(identity_dim),
      'expressionDim': int(expression_dim),
      'identityNames': [str(n) for n in d['identity_names']],
      'expressionNames': [str(n) for n in d['expression_names']],
      'jointNames': [str(n) for n in d['joint_names']],
      'componentNames': component_names,
      'materialNames': MATERIALS,
      'regionNames': [re.sub('_region$', '', n) for n in region_names],
      'bboxMin': [float(x) for x in template.min(axis=0)],
      'bboxMax': [float(x) for x in template.max(axis=0)],
      'hasPoseCorrectives': has_pose_correctives,
  }
  writer.write(os.path.join(args.out_dir, 'gnm_head_web.bin'), meta)

  # ---- Semantic sampler decoders. ------------------------------------------
  sampler_writer = ContainerWriter()
  sampler_meta = {}
  specs = [
      ('identity', 'identity_decoder_model.h5', GENDERS + ETHNICITIES),
      ('expression', 'expression_decoder_model.h5', EXPRESSION_CLASSES),
  ]
  for key, filename, labels in specs:
    layers = load_keras_mlp(os.path.join(sampler_dir, filename))
    layer_meta = []
    for i, layer in enumerate(layers):
      writer_key_w = f'{key}_w{i}'
      writer_key_b = f'{key}_b{i}'
      sampler_writer.add(writer_key_w, layer['kernel'])
      sampler_writer.add(writer_key_b, layer['bias'])
      layer_meta.append({
          'weights': writer_key_w,
          'bias': writer_key_b,
          'activation': layer['activation'],
          'inputDim': int(layer['kernel'].shape[0]),
          'outputDim': int(layer['kernel'].shape[1]),
      })
    condition_dim = len(labels)
    latent_dim = layer_meta[0]['inputDim'] - condition_dim
    sampler_meta[key] = {
        'latentDim': latent_dim,
        'conditionDim': condition_dim,
        'labels': labels,
        'layers': layer_meta,
    }
    # Smoke test: decode a zero latent with the first class.
    x = np.zeros((1, layer_meta[0]['inputDim']), dtype=np.float32)
    x[0, latent_dim] = 1.0
    out = run_mlp(layers, x)
    print(f'{key} decoder: latent {latent_dim} + {condition_dim} classes -> '
          f'{out.shape[1]} params (zero-latent output range '
          f'[{out.min():.3f}, {out.max():.3f}])')

  sampler_meta['identity']['genders'] = GENDERS
  sampler_meta['identity']['ethnicities'] = ETHNICITIES
  sampler_writer.write(
      os.path.join(args.out_dir, 'gnm_samplers_web.bin'), sampler_meta)


if __name__ == '__main__':
  main()
