"""Paints a GNM head skin texture from a reference photo with Nano Banana 2.

The GNM head unwraps into a single 0..1 UV square per mesh component, so the
'skin' component owns one square that holds the whole face, both ears, the
scalp and the neck. This tool bakes two conditioning guides out of the GNM npz
so an image model knows exactly where every feature lands in that square:

  <stem>_guide_relief.png   shaded relief of the neutral head, baked into UV
  <stem>_guide_regions.png  flat semantic color map (nose, lips, brows, ears)

Both guides, the reference and a crop of its head go to Gemini 3.1 Flash Image
("Nano Banana 2"), which returns a flat albedo laid out in the same UV square.

Hair is a second pass. Told where the hairline is, the model reliably ignores it
and redraws a portrait's forehead-to-hair proportions instead, which renders as
a bald head. So the skin pass paints a bald head, a separate pass paints a
square of hair, and the two are composited through a scalp mask taken straight
off the mesh - placement the model cannot drift away from. --hairline and
--temple_drop set where that mask starts, which is what controls how much
forehead the head ends up with.

Complexion is corrected the same way, for the same reason: the model drifts
toward a neutral, pinker skin than the reference no matter how the prompt
describes the colour. So the reference's lit skin is measured, quoted into the
prompt as hex targets, and then the painted result is measured too and pulled
onto it per channel (--tone).

The result is edge-padded into the unused corners of the atlas so mipmapping
never pulls background into a silhouette, and rendered onto the neutral head for
a look-at-it check. Alongside <out>.png the run leaves its guides, the head
crop, the exact prompt, and both raw passes, so a disappointing result can be
diagnosed without another call.

Usage:
  python gnm_texture_from_photo.py
  python gnm_texture_from_photo.py --photo=selfie.jpg --out=../assets/skin.png
  python gnm_texture_from_photo.py --guides_only

The default --photo is the public-domain C2RMF scan of the Mona Lisa on
Wikimedia Commons and --style=auto then appends a likeness block written for
that painting. Pass a local path to condition on anything else; --face_crop
takes left,top,right,bottom in 0..1 to point out the head in it.

The API key comes from keys.json - {"gemini": {"apiKey": "..."}} - looked up in
the working directory and then up from this script to the repository root, or
from --keys=PATH, or from GEMINI_API_KEY.

Requires numpy + Pillow, and google-genai for the generation step
(`pip install google-genai`).
"""

import argparse
import io
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
from PIL import Image, ImageFilter

MONA_LISA_URL = (
    'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/'
    'Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/1280px-'
    'Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg'
)
# Her head is about a tenth of the panel, so a crop goes in alongside the whole
# painting. Normalized left, top, right, bottom of the 1280x1908 scan.
MONA_LISA_FACE_CROP = (0.26, 0.055, 0.76, 0.425)
USER_AGENT = 'xrblocks-gnm-texture-tool/1.0'
KEYS_FILENAME = 'keys.json'
# Fraction of the way from the brows to the top of the skull where the hairline
# sits at the front of the face. The Mona Lisa's hairline is plucked right back,
# so this sits high; lower it for an ordinary head of hair.
DEFAULT_HAIRLINE = 0.92
# How far the hair cap dips at the temples, as a multiple of brow-to-nape. Low
# values keep the hair off the temples and widen the forehead.
DEFAULT_TEMPLE_DROP = 0.40
# Rate limiting and capacity, as opposed to anything wrong with the request.
RETRYABLE_STATUSES = (429, 500, 502, 503, 504)
RETRY_BACKOFF_SECONDS = 8

# Nano Banana 2. --model=gemini-3-pro-image selects Nano Banana Pro instead,
# which is slower but holds a layout more faithfully at 4K.
DEFAULT_MODEL = 'gemini-3.1-flash-image'

# Vertex groups painted into the semantic guide, in priority order: a later
# entry overwrites an earlier one where the groups overlap.
REGION_COLORS = [
    ('skin', (198, 164, 142)),
    # Straight after skin so every facial feature below overrides it.
    ('scalp', (58, 66, 150)),
    ('forehead_region', (212, 178, 154)),
    ('left_temple_region', (202, 170, 148)),
    ('right_temple_region', (202, 170, 148)),
    ('left_parotid_region', (204, 168, 146)),
    ('right_parotid_region', (204, 168, 146)),
    ('left_zygomatic_region', (212, 158, 140)),
    ('right_zygomatic_region', (212, 158, 140)),
    ('left_cheek_region', (216, 160, 142)),
    ('right_cheek_region', (216, 160, 142)),
    ('left_infraorbital_region', (214, 168, 150)),
    ('right_infraorbital_region', (214, 168, 150)),
    ('chin_region', (200, 164, 144)),
    ('ears', (90, 205, 140)),
    ('nose_region', (250, 155, 60)),
    ('left_orbital_region', (178, 152, 208)),
    ('right_orbital_region', (178, 152, 208)),
    ('left_brow_region', (112, 76, 50)),
    ('right_brow_region', (112, 76, 50)),
    ('middle_brow_region', (134, 98, 68)),
    ('eye_sockets', (240, 240, 248)),
    ('upper_lip_region', (196, 60, 70)),
    ('lower_lip_region', (214, 78, 88)),
    ('mouth_sock', (70, 16, 26)),
]

# Flat colors for the eyeball components in the preview render; the eyeballs
# carry their own UV square and are not part of the generated skin texture.
EYE_COLORS = [
    ('eyes', (198, 190, 176)),
    # Dull ivory rather than white: a bright sclera reads as a modern render
    # and pulls the preview away from a painted eye.
    ('scleras', (214, 206, 190)),
    ('irises', (104, 78, 52)),
    ('pupils', (12, 12, 14)),
]

LAYOUT_PROMPT = """\
Repaint image 1. This is an edit of image 1, not a new picture: image 1 is a UV
texture atlas for a 3D head mesh, and the output is the same atlas with its
surface repainted. Placement is a hard geometric contract, not a stylistic
choice - the mesh samples these exact pixels.

{references}

Image 1 is a grey clay render of the head flattened into its UV square: the face
in the middle, the two ears splayed out to the left and right, the scalp fanning
across the top, and the neck and the back of the head filling the bottom. Image
2 labels the same square by region. The small island in its top-right corner is
the inner lining of the mouth, which stays a dark desaturated red.

The blue band in image 2 is the cranium: the crown, the sides above the ears and
the back of the head. Paint it as bare scalp skin, the same complexion as the
forehead, slightly cooler and paler. This head is bald and stays bald - hair is
added separately afterwards - so paint no hair anywhere: no strands, no
hairline, no fringe, no sideburns, no shadow of a hairstyle.

Geometry lock. Do not redraw the head. Every landmark stays on the exact pixel
it occupies in image 1, at the same size: the eye openings, the nostrils, the
lip line, the philtrum, the ear folds, the jaw edge, the hairline. Do not shrink
the face, do not recentre it, do not turn this into a framed portrait. Overlaid
on image 1 at half opacity, the result must line up landmark for landmark. You
change colour only; you add nothing.

Image 2's colours are labels, not paint. Never copy them into the output. The
ears are skin, not green. The nose and nostrils are skin, not orange. The brow
ridge is skin, not brown. The lips are lip-coloured, not scarlet. The cranium is
bare scalp, not blue. The white ovals are eye openings: paint them as skin in
gentle shadow, never white - but only a shade or two darker than the cheek. No
dark rings, no hollow sunken sockets, no smoky eyeshadow, no bruised look. The
eyelids are the same complexion as the rest of the face.

This is a texture, not a painting on a panel. Its four edges run off the atlas
and get sampled as-is, so nothing may sit along them: no wooden panel edge, no
gilt frame, no painted border, no darkened margin, no vignette, no craquelure
crust at the rim. Skin runs all the way into all four corners.

Hard requirements:
- Flat unlit albedo. Take placement from image 1, then erase its lighting. No
  cast shadows, no specular highlights, no rim light, no ambient occlusion.
- Even brightness edge to edge. The temples, the cheeks out by the ears, the
  jaw, the sides and back of the neck and everything behind the ears must be
  the same brightness as the middle of the face. Do not darken the sides, do
  not vignette the edges of the head, do not shade anything to suggest that it
  curves away - the renderer adds all of that later, and painting it in as well
  makes the sides of the head go muddy and black.
- Skin only: no hair, no clothing, no jewellery, no background, no landscape.
- Do not paint eyeballs, irises or pupils - those are separate geometry.
- Keep the lips closed and do not paint teeth.
- Fill the whole square edge to edge, including all four corners. No black, no
  empty margins, no vignette, no frame, no border, no text.
- Keep the left and right halves near mirror images so the UV seam running down
  the back of the head does not show, and keep the texture free of any hard
  horizontal or vertical banding.
"""

# Appended for the default reference. Naming the painting lets the model reach
# for what it already knows about it; the bullets pin down the parts that
# actually survive into a texture.
MONA_LISA_STYLE = """\
Likeness. The source is Leonardo da Vinci's Mona Lisa (La Gioconda), the C2RMF
retouched scan of the Louvre panel. The finished head has to read as her, so
match this painting rather than a generic face:

- Skin: deep warm golden-olive, the colour of aged varnish over flesh - closer
  to old gold than to pink. It must read distinctly yellow, never neutral,
  never rosy, never beige. Modelled in sfumato: no hard edge anywhere, every
  transition smoked out.
{palette}
- Brows and lashes: she has none. Leave the brow ridge bare skin. Paint no
  eyebrow hairs and no eyelashes at all.
- Forehead: enormous. Hers is the most striking thing about the head - a very
  high, broad, rounded expanse of bare skin, because the hairline is plucked
  right back over the crown in the fashion of the period. The distance from her
  brow up to her hairline is nearly half the distance from her brow down to her
  chin. Give the forehead that much room and keep it smooth and unshaded.
- Eye shape: hers are narrow almonds, not round openings. Paint the aperture as
  a long almond whose outer corner sits a little lower than the inner one, so
  the eye tilts very slightly down and out. A heavy, smooth upper lid comes down
  over the top of it, hooding it and shortening it - the eye should look calm
  and half-lidded, never wide. Below, a soft rounded fullness rather than a
  crease. Keep both lids the same warm complexion as the cheek, only slightly
  deeper: Leonardo painted them luminous, so no dark ring, no liner, no lashes.
  Both eyes must match exactly.
- Nose: narrow straight bridge, soft rounded tip, shading only down her left
  side.
- Mouth: small and closed, muted rose-brown barely darker than the skin, the
  corners lifted a fraction and dissolved into shadow - the ambiguous smile. No
  lip liner, no gloss, no saturated red.
- Cheeks, jaw and chin: full and softly rounded, no visible cheekbone edge, a
  soft double curve under the chin.
- Hair: none here. Her hair is painted in a separate pass and composited on top,
  so leave the cranium bare.
- Surface: the aged yellow-brown varnish cast over everything, plus the fine
  oil-on-poplar craquelure of the original, kept low contrast so the result
  still reads as skin.
"""

HAIR_PROMPT = """\
Paint a square texture of hair and nothing else. It gets wrapped onto the
cranium of a 3D head, so it must be hair edge to edge - no face, no skin, no
ears, no scalp showing, no background, no border, no framing, no text, and no
head silhouette anywhere in it.

The vertical centre line of the square is the top of the head, so run a soft
centre parting straight down the middle, with the hair sweeping away to the left
on the left half and to the right on the right half, mirrored. Away from the
parting the strands should flow smoothly and fill every corner.

The left and right edges of the square meet each other around the back of the
head, so they have to line up: the strands running off the left edge must
continue exactly where the right edge begins, at the same brightness. Keep both
edges mid-toned and even - no dark band, no seam, no border down either side.

Keep the whole square evenly lit. No parting highlight, no sheen band, no dark
roots, nothing that would read as a stripe once it is wrapped onto a head.

Flat unlit albedo: no cast shadows, no specular sheen, no highlights, no
lighting of any kind, just the colour of the hair itself.

{hair}
"""

MONA_LISA_HAIR = """\
The hair is Mona Lisa's in Leonardo's painting, taken from the reference. Match
how dark it is: the mass of it is near black, a very deep brown, and the auburn
only shows as thin warm reddish glints along a few strands. Long loose soft
waves, matte and dry - no sheen, no gloss band, no lit highlight running across
it. Keep individual strands legible at full resolution and carry over the aged
yellow-brown varnish cast and the fine craquelure of the original."""

PHOTO_HAIR = """\
The hair is the reference subject's: match its colour, tone variation and the
scale and direction of its waves or curls."""

PHOTO_STYLE = """\
Likeness. The finished head has to read as the same person as the reference.
Carry over their exact skin tone and its variation across the face, complexion,
freckles and blemishes where they fall, beard or stubble density and edge, brow
shape and thickness, lip color and shape, and hairline. Photoreal skin
micro-detail: pores, faint mottling, subtle redness at the nose and ears. The
cranium stays bare - hair is painted separately and composited on top.
{palette}
"""


def scalp_mask(positions, group_mask, hairline,
               temple_drop=DEFAULT_TEMPLE_DROP):
  """Marks the cranium the hair has to cover.

  GNM has no scalp group, and the vertex groups that come closest put the
  hairline at the very crown, which renders as a bald head from the front. So
  the hairline is built instead: a height that starts at `hairline` of the way
  from the brows to the top of the skull at the face, and falls to the bottom
  of the ears as the surface turns backwards or outwards. That drop at the
  temples is what makes painted hair frame the face.
  """
  brows = (group_mask('left_brow_region') | group_mask('right_brow_region')
           | group_mask('middle_brow_region'))
  brow_y = positions[brows, 1].max()
  front_y = brow_y + hairline * (positions[:, 1].max() - brow_y)
  nape_y = positions[group_mask('ears'), 1].min()

  face_z = positions[group_mask('forehead_region'), 2].max()
  back_z = positions[:, 2].min()
  depth = np.clip((face_z - positions[:, 2]) / (face_z - back_z), 0.0, 1.0)
  lateral = (np.abs(positions[:, 0]) / np.abs(positions[:, 0]).max()) ** 2
  fall = np.clip(1.6 * depth + temple_drop * lateral, 0.0, 1.0)

  above = positions[:, 1] > front_y - (front_y - nape_y) * fall
  return group_mask('skin') & ~group_mask('ears') & above


def load_skin_mesh(npz_path, hairline=DEFAULT_HAIRLINE,
                   temple_drop=DEFAULT_TEMPLE_DROP):
  """Loads the GNM head and splits out the skin component and the eyeballs."""
  data = np.load(npz_path, allow_pickle=True)
  positions = data['template_vertex_positions'].astype(np.float64)
  triangles = data['triangles'].astype(np.int64)
  triangle_uvs = data['triangle_uvs'].astype(np.float64)
  group_names = [str(name) for name in data['vertex_group_names']]
  groups = data['vertex_groups']

  def group_mask(name):
    if name not in group_names:
      return np.zeros(len(positions), dtype=bool)
    return groups[group_names.index(name)] > 0.5

  skin = group_mask('skin')
  eyes = group_mask('eyes')
  # eye_exteriors is the clear cornea shell; drawn opaque it would hide the
  # iris and pupil sitting behind it.
  cornea = group_mask('eye_exteriors')
  skin_faces = skin[triangles].all(axis=1)
  eye_faces = eyes[triangles].all(axis=1) & ~cornea[triangles].all(axis=1)

  scalp = scalp_mask(positions, group_mask, hairline, temple_drop)
  # Plain lit skin, used to measure and correct the overall complexion: no
  # hair, no lips, no mouth lining, no brow shadow, no ears.
  tone = skin & ~scalp
  for name in ('ears', 'mouth_sock', 'upper_lip_region', 'lower_lip_region',
               'eye_sockets', 'left_brow_region', 'right_brow_region',
               'middle_brow_region'):
    tone = tone & ~group_mask(name)

  region_colors = np.full((len(positions), 3), (190, 158, 138),
                          dtype=np.float64)
  for name, color in REGION_COLORS:
    region_colors[scalp if name == 'scalp' else group_mask(name)] = color
  eye_colors = np.full((len(positions), 3), (200, 200, 200), dtype=np.float64)
  for name, color in EYE_COLORS:
    eye_colors[group_mask(name)] = color

  return {
      'positions': positions,
      'normals': vertex_normals(positions, triangles),
      'skin_triangles': triangles[skin_faces],
      'skin_uvs': triangle_uvs[skin_faces],
      'eye_triangles': triangles[eye_faces],
      'scalp': scalp,
      'tone': tone,
      'region_colors': region_colors,
      'eye_colors': eye_colors,
      'version': str(data['version']),
      'variant': str(data['variant']),
  }


def vertex_normals(positions, triangles):
  """Area-weighted vertex normals."""
  normals = np.zeros_like(positions)
  a = positions[triangles[:, 0]]
  b = positions[triangles[:, 1]]
  c = positions[triangles[:, 2]]
  face_normals = np.cross(b - a, c - a)
  for corner in range(3):
    np.add.at(normals, triangles[:, corner], face_normals)
  lengths = np.linalg.norm(normals, axis=1, keepdims=True)
  lengths[lengths < 1e-12] = 1.0
  return normals / lengths


def rasterize_uv(size, triangle_uvs, triangles, vertex_attributes, background):
  """Bakes a per-vertex attribute into the UV square.

  Args:
    size: Side length of the square output in pixels.
    triangle_uvs: Per-corner texture coordinates, (T, 3, 2).
    triangles: Per-corner vertex indices, (T, 3).
    vertex_attributes: The attribute to interpolate, (V, C).
    background: The value written where no triangle covers a pixel, (C,).

  Returns:
    (image, coverage) where image is (size, size, C) float64 and coverage is a
    (size, size) bool mask of the pixels a triangle actually landed on.
  """
  channels = vertex_attributes.shape[1]
  image = np.empty((size, size, channels), dtype=np.float64)
  image[:] = background
  coverage = np.zeros((size, size), dtype=bool)

  # v runs bottom-up in UV space and rows run top-down in image space.
  px = triangle_uvs[:, :, 0] * (size - 1)
  py = (1.0 - triangle_uvs[:, :, 1]) * (size - 1)
  x0 = np.clip(np.floor(px.min(axis=1)).astype(int), 0, size - 1)
  x1 = np.clip(np.ceil(px.max(axis=1)).astype(int), 0, size - 1)
  y0 = np.clip(np.floor(py.min(axis=1)).astype(int), 0, size - 1)
  y1 = np.clip(np.ceil(py.max(axis=1)).astype(int), 0, size - 1)

  for face in range(len(triangles)):
    ax, ay = px[face, 0], py[face, 0]
    bx, by = px[face, 1], py[face, 1]
    cx, cy = px[face, 2], py[face, 2]
    area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    if abs(area) < 1e-12:
      continue
    xs = np.arange(x0[face], x1[face] + 1)
    ys = np.arange(y0[face], y1[face] + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    w0 = (bx - grid_x) * (cy - grid_y) - (cx - grid_x) * (by - grid_y)
    w1 = (cx - grid_x) * (ay - grid_y) - (ax - grid_x) * (cy - grid_y)
    w2 = area - w0 - w1
    # Compare against the winding of this triangle so both orientations fill.
    sign = 1.0 if area > 0 else -1.0
    epsilon = -1e-9 * abs(area)
    inside = ((w0 * sign >= epsilon) & (w1 * sign >= epsilon)
              & (w2 * sign >= epsilon))
    if not inside.any():
      continue
    corners = vertex_attributes[triangles[face]]
    value = ((w0 / area)[..., None] * corners[0]
             + (w1 / area)[..., None] * corners[1]
             + (w2 / area)[..., None] * corners[2])
    tile = image[y0[face]:y1[face] + 1, x0[face]:x1[face] + 1]
    tile[inside] = value[inside]
    coverage[y0[face]:y1[face] + 1, x0[face]:x1[face] + 1][inside] = True
  return image, coverage


def build_guides(mesh, size):
  """Renders the relief and semantic guides plus the atlas coverage mask."""
  light = np.array([0.30, 0.35, 1.0])
  light /= np.linalg.norm(light)
  lambert = 0.28 + 0.72 * np.clip(mesh['normals'] @ light, 0.0, 1.0)
  # A little facing-ratio term keeps the ears and the jaw from going black.
  shade = 0.88 * lambert + 0.12 * (0.5 + 0.5 * mesh['normals'][:, 2])
  gray = np.clip(shade, 0.0, 1.0)[:, None] * np.array([[240.0, 232.0, 222.0]])

  relief, coverage = rasterize_uv(size, mesh['skin_uvs'],
                                  mesh['skin_triangles'], gray,
                                  (14.0, 14.0, 18.0))
  regions, _ = rasterize_uv(size, mesh['skin_uvs'], mesh['skin_triangles'],
                            mesh['region_colors'], (18.0, 18.0, 22.0))
  return to_pil(relief), to_pil(regions), coverage


def build_hair_alpha(mesh, size, feather):
  """Rasterizes the scalp mask into UV space as a soft alpha for compositing.

  Asking the image model to put the hairline in the right place does not work -
  it keeps redrawing a portrait's forehead-to-hair proportions over the layout.
  So hair is generated as its own texture and blended in against this mask,
  which comes straight off the mesh and cannot drift.
  """
  return build_mask_alpha(mesh, mesh['scalp'], size, feather)


def build_mask_alpha(mesh, vertex_mask, size, feather):
  """Rasterizes a per-vertex boolean mask into UV space as a soft alpha."""
  weight = vertex_mask.astype(np.float64)[:, None]
  alpha, _ = rasterize_uv(size, mesh['skin_uvs'], mesh['skin_triangles'],
                          weight, (0.0,))
  alpha = alpha[:, :, 0]
  # The mask is a hard per-vertex threshold, so soften the edge into a hairline
  # instead of a staircase.
  radius = max(int(round(feather * size)), 0)
  if radius > 0:
    blurred = Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8))
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius))
    alpha = np.asarray(blurred, dtype=np.float64) / 255.0
  return np.clip(alpha, 0.0, 1.0)


def solve_hooded_eyes(npz_path, narrow):
  """Finds the expression coefficients that hood the eyes into almonds.

  The Mona Lisa's eyes are calm and half-lidded, while GNM's neutral head has
  round, wide-open ones - and no amount of painting fixes that, because the
  eyeball is separate geometry that the skin texture cannot cover. GNM's
  expression basis is per-region PCA rather than named blendshapes, so there is
  no lid slider either. But the eye aperture is linear in the coefficients, so
  its gradient over the eye-region components is the cheapest direction that
  closes the lid, and a short step along it narrows the eye without disturbing
  the rest of the face.

  Args:
    npz_path: The GNM model.
    narrow: Fraction to reduce the aperture by; about 0.08 reads as hooded,
      0.2 already looks sleepy.

  Returns:
    (coefficients, names) or (None, None) if the groups are missing.
  """
  data = np.load(npz_path, allow_pickle=True)
  positions = data['template_vertex_positions'].astype(np.float64)
  basis = data['expression_basis'].astype(np.float64)
  names = [str(n) for n in data['expression_names']]
  group_names = [str(n) for n in data['vertex_group_names']]
  groups = data['vertex_groups']

  def group_mask(name):
    if name not in group_names:
      return np.zeros(len(positions), dtype=bool)
    return groups[group_names.index(name)] > 0.5

  gradient = np.zeros(len(names))
  aperture = 0.0
  eyes = ('left_eye', 'right_eye')
  for eye in eyes:
    ball = group_mask(eye)
    if not ball.any():
      return None, None
    centre = positions[ball].mean(axis=0)
    radius = 1.45 * np.linalg.norm(positions[ball] - centre, axis=1).max()
    lid = (group_mask('skin') & group_mask('eye_sockets')
           & (np.linalg.norm(positions - centre, axis=1) < radius))
    upper = lid & (positions[:, 1] > centre[1])
    lower = lid & (positions[:, 1] < centre[1])
    if not (upper.any() and lower.any()):
      return None, None
    aperture += positions[upper, 1].mean() - positions[lower, 1].mean()
    gradient += (basis[:, upper, 1].mean(axis=1)
                 - basis[:, lower, 1].mean(axis=1))
  aperture /= len(eyes)
  gradient /= len(eyes)

  # Confine the change to the eyes; the rest of the face is the texture's job.
  gradient *= np.array([('eye_region' in name) for name in names])
  norm = float(np.linalg.norm(gradient))
  if norm < 1e-9:
    return None, None
  return (-narrow * aperture / norm ** 2) * gradient, names


def lit_skin_color(pixels):
  """Mean color of the lit skin among `pixels`, an (N, 3) array.

  Skin in a portrait is neither the darkest nor the brightest thing in frame,
  so hair, shadow and specular highlights are trimmed off by luminance before
  averaging. Without that the Mona Lisa's dark hair and background drag the
  measured complexion far away from the face.
  """
  if len(pixels) == 0:
    return None
  luma = pixels @ np.array([0.2126, 0.7152, 0.0722])
  low, high = np.percentile(luma, [55, 92])
  lit = pixels[(luma >= low) & (luma <= high)]
  return (lit if len(lit) else pixels).mean(axis=0)


def reference_skin_color(photo, box):
  """Measures the reference's complexion from the middle of the head crop.

  A head crop still has background in its corners and hair down its sides. The
  Mona Lisa's landscape is green enough that averaging the whole crop reports a
  duller, greener skin than her face actually is - and the correction then
  pulls the wrong way. So this keeps the centre of the crop, and only pixels
  ordered like skin (red above green above blue).
  """
  image = crop_face(photo, box) if box else photo
  pixels = np.asarray(image, dtype=np.float64)
  height, width = pixels.shape[:2]
  centre = pixels[int(0.18 * height):int(0.82 * height),
                  int(0.20 * width):int(0.80 * width)].reshape(-1, 3)
  warm = centre[(centre[:, 0] > centre[:, 1]) & (centre[:, 1] > centre[:, 2])]
  return lit_skin_color(warm if len(warm) > 64 else centre)


def match_skin_tone(texture, tone_alpha, target, strength):
  """Pulls the texture's complexion toward `target`, per channel.

  The image model reliably drifts toward a neutral, pinker skin than the
  reference, so the correction is measured off the render rather than asked
  for. The gain is applied to the whole atlas, which is how an aged varnish
  actually behaves, and clamped so a bad measurement cannot wreck the map.
  """
  if strength <= 0 or target is None:
    return texture, None
  pixels = np.asarray(texture, dtype=np.float64)
  mask = tone_alpha
  if mask.shape != pixels.shape[:2]:
    mask = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).resize(
            texture.size, Image.BILINEAR), dtype=np.float64) / 255.0
  current = lit_skin_color(pixels[mask > 0.5].reshape(-1, 3))
  if current is None or (current < 1e-3).any():
    return texture, None
  gain = np.clip(target / current, 0.55, 1.8)
  gain = 1.0 + (gain - 1.0) * strength
  return to_pil(pixels * gain), (current, gain)


def flatten_shading(texture, tone_alpha, strength, radius_fraction=0.06):
  """Divides out baked lighting so the sides of the head stop reading dark.

  However firmly the prompt asks for flat albedo, the model paints a lit face:
  bright down the middle, falling away at the temples and jaw. On a mesh that
  gets shaded again at render time, that doubles up and the sides go muddy. So
  the low-frequency luminance is measured and normalised away, leaving pores
  and craquelure - which live in the high frequencies - untouched.
  """
  if strength <= 0:
    return texture, None
  pixels = np.asarray(texture, dtype=np.float64)
  size = pixels.shape[0]
  mask = tone_alpha
  if mask.shape != pixels.shape[:2]:
    mask = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).resize(
            texture.size, Image.BILINEAR), dtype=np.float64) / 255.0

  luma = pixels @ np.array([0.2126, 0.7152, 0.0722])
  radius = max(int(round(radius_fraction * size)), 2)
  # Blur luminance and the mask together, so unlit regions outside the face do
  # not bleed into the estimate near its edges.
  blur = ImageFilter.GaussianBlur(radius)
  lit = np.asarray(
      Image.fromarray(np.clip(luma * mask, 0, 255).astype(np.uint8)).filter(
          blur), dtype=np.float64)
  weight = np.asarray(
      Image.fromarray(np.clip(mask * 255, 0, 255).astype(np.uint8)).filter(
          blur), dtype=np.float64) / 255.0
  low = lit / np.maximum(weight, 1e-3)

  inside = mask > 0.5
  if not inside.any():
    return texture, None
  target = np.median(low[inside])
  gain = np.clip(target / np.maximum(low, 1.0), 0.7, 1.45)
  gain = 1.0 + (gain - 1.0) * strength * mask
  spread_before = low[inside].std()
  flattened = np.clip(pixels * gain[..., None], 0, 255)
  return to_pil(flattened), (spread_before, target)


def match_seam_columns(texture, blend=None):
  """Makes the two vertical edges of the atlas agree.

  The back of the head is split down the middle at u=0 and u=1, so those two
  columns are the same line of skin sampled twice. The two halves of a
  generated texture rarely match in brightness there, which shows up as a
  stripe down the back of the head, so they are eased toward their shared
  average. The band has to be wide - matching only the outermost texels leaves
  the mismatch intact a few pixels in - and it is smoothstepped so the
  correction itself does not read as an edge.
  """
  pixels = np.asarray(texture, dtype=np.float64)
  width = pixels.shape[1]
  blend = max(8, width // 24) if blend is None else max(1, blend)
  blend = min(blend, width // 4)
  average = 0.5 * (pixels[:, 0] + pixels[:, -1])
  for offset in range(blend):
    x = offset / blend
    # Smoothstep from a half-and-half blend at the edge to nothing at `blend`.
    t = 0.5 * (1.0 - (x * x * (3.0 - 2.0 * x)))
    far = width - 1 - offset
    pixels[:, offset] += (average - pixels[:, offset]) * t
    pixels[:, far] += (average - pixels[:, far]) * t
  return to_pil(pixels)


def composite_hair(skin, hair, alpha):
  """Blends the hair texture over the skin texture through the scalp alpha."""
  size = skin.size[0]
  if hair.size != skin.size:
    hair = hair.resize(skin.size, Image.LANCZOS)
  if alpha.shape[0] != size:
    alpha = np.asarray(
        Image.fromarray((alpha * 255).astype(np.uint8)).resize(
            skin.size, Image.BILINEAR), dtype=np.float64) / 255.0
  skin_pixels = np.asarray(skin, dtype=np.float64)
  hair_pixels = np.asarray(hair, dtype=np.float64)
  blend = alpha[..., None]
  return to_pil(hair_pixels * blend + skin_pixels * (1.0 - blend))


def to_pil(array):
  return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), 'RGB')


def pad_atlas(image, coverage, iterations):
  """Bleeds texel colors outward into the parts of the square no face covers.

  Without this the black corners of the atlas bleed into the silhouette once
  the texture is mipmapped. Islands get `iterations` texels of real bleed;
  anything still empty past that is far enough from a UV border that a flat
  fill with the average skin color is invisible.
  """
  if iterations <= 0:
    return image
  pixels = np.asarray(image, dtype=np.float64)
  if coverage.shape != pixels.shape[:2]:
    ratio = pixels.shape[0] / coverage.shape[0]
    coverage = np.asarray(
        Image.fromarray(coverage.astype(np.uint8) * 255).resize(
            (pixels.shape[1], pixels.shape[0]), Image.NEAREST)) > 127
    # An upscaled mask straddles the island border by a texel or two, which
    # would seed the bleed from the dark pixels just outside it. Pull it in.
    for _ in range(max(2, int(math.ceil(ratio)) * 2)):
      shrunk = coverage.copy()
      for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shrunk &= np.roll(np.roll(coverage, dy, axis=0), dx, axis=1)
      coverage = shrunk
  if not coverage.any():
    return image
  filled = coverage.copy()
  for _ in range(iterations):
    if filled.all():
      break
    total = np.zeros_like(pixels)
    count = np.zeros(pixels.shape[:2], dtype=np.float64)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
      shifted_pixels = np.roll(np.roll(pixels, dy, axis=0), dx, axis=1)
      shifted_filled = np.roll(np.roll(filled, dy, axis=0), dx, axis=1)
      total += shifted_pixels * shifted_filled[..., None]
      count += shifted_filled
    grow = (~filled) & (count > 0)
    if not grow.any():
      break
    pixels[grow] = total[grow] / count[grow][:, None]
    filled |= grow
  if not filled.all():
    pixels[~filled] = pixels[coverage].mean(axis=0)
  return to_pil(pixels)


def load_image(source):
  """Opens a local path or an http(s) URL as an RGB image."""
  if source.startswith('http://') or source.startswith('https://'):
    print(f'Fetching reference photo: {source}')
    request = urllib.request.Request(source, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
      payload = response.read()
    return Image.open(io.BytesIO(payload)).convert('RGB')
  return Image.open(source).convert('RGB')


def find_keys_file(explicit):
  """Locates keys.json: the flag, then the cwd, then up from this script."""
  if explicit:
    return explicit if os.path.exists(explicit) else None
  directory = os.path.dirname(os.path.abspath(__file__))
  candidates = [os.path.join(os.getcwd(), KEYS_FILENAME)]
  for _ in range(6):
    candidates.append(os.path.join(directory, KEYS_FILENAME))
    parent = os.path.dirname(directory)
    if parent == directory:
      break
    directory = parent
  for path in candidates:
    if os.path.exists(path):
      return path
  return None


def load_api_key(explicit):
  """Reads the Gemini key from keys.json, falling back to the environment.

  keys.json is {"gemini": {"apiKey": "..."}}; a bare string under "gemini" and
  a top-level "apiKey" or "GEMINI_API_KEY" are accepted too.
  """
  path = find_keys_file(explicit)
  if path:
    with open(path, 'r', encoding='utf-8') as handle:
      keys = json.load(handle)
    section = keys.get('gemini', keys.get('google', {}))
    if isinstance(section, str):
      key = section
    else:
      key = (section.get('apiKey') or section.get('api_key')
             or keys.get('apiKey') or keys.get('GEMINI_API_KEY'))
    if key:
      print(f'Using the Gemini key from {path}')
      return key
    print(f'{path} has no gemini apiKey; falling back to the environment.')
  key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
  if key:
    print('Using the Gemini key from the environment')
    return key
  raise SystemExit(
      'No Gemini API key. Write one to keys.json as\n'
      '  {"gemini": {"apiKey": "..."}}\n'
      'next to this script or at the repository root, pass --keys=PATH, or set '
      'GEMINI_API_KEY.')


def parse_crop(spec, photo_source):
  """Turns a 'left,top,right,bottom' spec in 0..1 into a pixel-crop callable."""
  if spec == 'none':
    return None
  if spec == 'auto':
    if photo_source != MONA_LISA_URL:
      return None
    box = MONA_LISA_FACE_CROP
  else:
    parts = [float(value) for value in spec.replace(' ', '').split(',')]
    if len(parts) != 4:
      raise SystemExit('--face_crop wants four numbers: left,top,right,bottom')
    box = tuple(parts)
  if not (0.0 <= box[0] < box[2] <= 1.0 and 0.0 <= box[1] < box[3] <= 1.0):
    raise SystemExit(f'--face_crop out of range or inverted: {box}')
  return box


def crop_face(photo, box):
  """Crops a normalized box out of the reference and upsizes it a little."""
  width, height = photo.size
  crop = photo.crop((int(box[0] * width), int(box[1] * height),
                     int(box[2] * width), int(box[3] * height)))
  # The head is a small slice of a full painting; give the model more of it.
  if max(crop.size) < 768:
    scale = 768 / max(crop.size)
    crop = crop.resize((round(crop.width * scale), round(crop.height * scale)),
                       Image.LANCZOS)
  return crop


def palette_block(reference_color):
  """Turns the measured reference complexion into concrete prompt targets.

  Naming a colour beats describing one: the model matches hex values far more
  closely than it matches adjectives like "golden-olive".
  """
  if reference_color is None:
    return ''
  base = np.asarray(reference_color, dtype=np.float64)
  stops = (('deepest shadow', 0.62), ('mid tone', 0.88),
           ('lit skin', 1.0), ('highlight', 1.18))
  lines = ['- Skin palette, measured off the reference. Hit these:']
  for name, scale in stops:
    r, g, b = np.clip(base * scale, 0, 255).astype(int)
    lines.append(f'    {name}: #{r:02x}{g:02x}{b:02x}')
  ratio = base / max(base.max(), 1e-6)
  lines.append(
      f'  Keep the channel balance near R:G:B = 1.00 : {ratio[1]:.2f} : '
      f'{ratio[2]:.2f} - blue must stay far below red, that is what makes it '
      'read as varnished gold rather than plain skin.')
  return '\n'.join(lines)


def build_prompt(style, has_crop, extra, reference_color=None):
  """Assembles the layout contract, the likeness block and any extra text."""
  if has_crop:
    references = (
        'Image 1 is the atlas to repaint, image 2 its region map, image 3 the '
        'source to take the likeness from, and image 4 a tight crop of the '
        'head in image 3.')
  else:
    references = (
        'Image 1 is the atlas to repaint, image 2 its region map, and image 3 '
        'the source to take the likeness from.')
  blocks = [LAYOUT_PROMPT.format(references=references)]
  palette = palette_block(reference_color)
  if style == 'mona_lisa':
    blocks.append(MONA_LISA_STYLE.format(palette=palette))
  elif style == 'photo':
    blocks.append(PHOTO_STYLE.format(palette=palette))
  if extra.strip():
    blocks.append(extra.strip() + '\n')
  return '\n'.join(blocks)


def response_parts(response):
  """Reads the parts of a generate_content response across SDK versions."""
  parts = getattr(response, 'parts', None)
  if parts:
    return parts
  for candidate in getattr(response, 'candidates', None) or []:
    content = getattr(candidate, 'content', None)
    if content is not None and getattr(content, 'parts', None):
      return content.parts
  return []


def part_image(part):
  """Decodes an inline image part into a PIL image, or returns None."""
  inline = getattr(part, 'inline_data', None)
  if inline is None or not getattr(inline, 'data', None):
    return None
  # part.as_image() hands back a PIL image on some SDK versions and a
  # types.Image wrapper on others; only the former can be used directly.
  as_image = getattr(part, 'as_image', None)
  if callable(as_image):
    image = as_image()
    if image is not None and hasattr(image, 'convert'):
      return image.convert('RGB')
    payload = getattr(image, 'image_bytes', None)
    if payload:
      return Image.open(io.BytesIO(payload)).convert('RGB')
  return Image.open(io.BytesIO(inline.data)).convert('RGB')


def generate_texture(model, prompt, images, api_size, attempts, api_key):
  """Calls Nano Banana and returns the first image part it produces."""
  try:
    from google import genai
    from google.genai import types
  except ImportError as error:
    raise SystemExit(
        'google-genai is required for generation. Install it with\n'
        '  pip install google-genai\n'
        f'(import failed: {error})') from error

  client = genai.Client(api_key=api_key)
  contents = [prompt] + list(images)

  def make_config(with_image_config):
    kwargs = {'response_modalities': ['TEXT', 'IMAGE']}
    if with_image_config:
      kwargs['image_config'] = types.ImageConfig(
          aspect_ratio='1:1', image_size=api_size)
    return types.GenerateContentConfig(**kwargs)

  # image_config landed after response_modalities did; fall back to a plain
  # square request on SDKs that predate it.
  sized = True
  try:
    make_config(True)
  except (TypeError, ValueError, AttributeError) as error:
    print(f'ImageConfig unsupported by this SDK ({error}); using defaults.')
    sized = False

  last_text = ''
  attempt = 0
  while attempt < attempts:
    attempt += 1
    print(f'Generating with {model} ({api_size}), '
          f'attempt {attempt}/{attempts}...')
    try:
      response = client.models.generate_content(
          model=model, contents=contents, config=make_config(sized))
    except TypeError as error:
      if not sized:
        raise
      print(f'Retrying without image_config ({error}).')
      sized = False
      attempt -= 1
      continue
    except Exception as error:  # pylint: disable=broad-except
      status = getattr(error, 'code', None) or getattr(error, 'status_code', 0)
      # 429/500/503 are load, not a bad request; back off and try again.
      if status not in RETRYABLE_STATUSES or attempt == attempts:
        raise
      delay = RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1)
      print(f'{status} from the API; retrying in {delay}s.')
      time.sleep(delay)
      continue
    texts = []
    for part in response_parts(response):
      image = part_image(part)
      if image is not None:
        return image
      if getattr(part, 'text', None):
        texts.append(part.text)
    last_text = ' '.join(texts).strip()
    said = last_text[:400] or '(none)'
    print(f'No image in the response. Model said: {said}')
  raise SystemExit(
      f'{model} returned no image after {attempts} attempts. '
      f'Last response text: {last_text[:400] or "(none)"}')


def render_preview(mesh, texture, size, distance=0.62, fov_degrees=26.0):
  """Renders the neutral head from the front with the texture applied.

  A small perspective-correct z-buffer rasterizer, so the only dependency for
  checking the result is numpy.
  """
  positions = mesh['positions']
  center = np.array([0.0, positions[:, 1].mean(), positions[:, 2].mean()])
  eye = center + np.array([0.0, 0.0, distance])
  focal = 1.0 / math.tan(math.radians(fov_degrees) * 0.5)

  view = positions - eye
  depth = -view[:, 2]
  safe_depth = np.where(depth > 1e-6, depth, 1e-6)
  screen_x = (focal * view[:, 0] / safe_depth * 0.5 + 0.5) * (size - 1)
  screen_y = (0.5 - focal * view[:, 1] / safe_depth * 0.5) * (size - 1)
  inv_depth = 1.0 / safe_depth

  light = np.array([0.35, 0.45, 1.0])
  light /= np.linalg.norm(light)
  # Soft and ambient-heavy: the point of the preview is to read the albedo, not
  # to relight it, so keep the lighting from swamping the painted texture.
  shade = 0.55 + 0.50 * np.clip(mesh['normals'] @ light, 0.0, 1.0)

  texels = np.asarray(texture.convert('RGB'), dtype=np.float64)
  tex_h, tex_w = texels.shape[:2]

  image = np.zeros((size, size, 3), dtype=np.float64)
  image[:] = (24.0, 26.0, 32.0)
  z_buffer = np.full((size, size), -np.inf)

  def draw(triangles, uvs, flat_colors):
    for face in range(len(triangles)):
      idx = triangles[face]
      if (depth[idx] <= 1e-6).any():
        continue
      ax, ay = screen_x[idx[0]], screen_y[idx[0]]
      bx, by = screen_x[idx[1]], screen_y[idx[1]]
      cx, cy = screen_x[idx[2]], screen_y[idx[2]]
      area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
      if abs(area) < 1e-12:
        continue
      lo_x = max(int(math.floor(min(ax, bx, cx))), 0)
      hi_x = min(int(math.ceil(max(ax, bx, cx))), size - 1)
      lo_y = max(int(math.floor(min(ay, by, cy))), 0)
      hi_y = min(int(math.ceil(max(ay, by, cy))), size - 1)
      if lo_x > hi_x or lo_y > hi_y:
        continue
      grid_x, grid_y = np.meshgrid(np.arange(lo_x, hi_x + 1),
                                   np.arange(lo_y, hi_y + 1))
      l0 = ((bx - grid_x) * (cy - grid_y)
            - (cx - grid_x) * (by - grid_y)) / area
      l1 = ((cx - grid_x) * (ay - grid_y)
            - (ax - grid_x) * (cy - grid_y)) / area
      l2 = 1.0 - l0 - l1
      inside = (l0 >= 0.0) & (l1 >= 0.0) & (l2 >= 0.0)
      if not inside.any():
        continue
      w = (l0 * inv_depth[idx[0]] + l1 * inv_depth[idx[1]]
           + l2 * inv_depth[idx[2]])
      tile_z = z_buffer[lo_y:hi_y + 1, lo_x:hi_x + 1]
      visible = inside & (w > tile_z)
      if not visible.any():
        continue
      if uvs is not None:
        corner_uv = uvs[face]
        # Perspective-correct: interpolate uv/z and 1/z, then divide.
        u = (l0 * corner_uv[0, 0] * inv_depth[idx[0]]
             + l1 * corner_uv[1, 0] * inv_depth[idx[1]]
             + l2 * corner_uv[2, 0] * inv_depth[idx[2]]) / w
        v = (l0 * corner_uv[0, 1] * inv_depth[idx[0]]
             + l1 * corner_uv[1, 1] * inv_depth[idx[1]]
             + l2 * corner_uv[2, 1] * inv_depth[idx[2]]) / w
        tex_x = np.clip((u * (tex_w - 1)).astype(int), 0, tex_w - 1)
        tex_y = np.clip(((1.0 - v) * (tex_h - 1)).astype(int), 0, tex_h - 1)
        albedo = texels[tex_y, tex_x]
      else:
        corner_colors = flat_colors[idx]
        albedo = (l0[..., None] * corner_colors[0]
                  + l1[..., None] * corner_colors[1]
                  + l2[..., None] * corner_colors[2])
      lit = (l0 * shade[idx[0]] + l1 * shade[idx[1]] + l2 * shade[idx[2]])
      color = albedo * lit[..., None]
      tile = image[lo_y:hi_y + 1, lo_x:hi_x + 1]
      tile[visible] = color[visible]
      tile_z[visible] = w[visible]

  draw(mesh['skin_triangles'], mesh['skin_uvs'], None)
  draw(mesh['eye_triangles'], None, mesh['eye_colors'])
  return to_pil(image)


def main():
  parser = argparse.ArgumentParser(
      description='Paint a GNM skin texture from a photo with Nano Banana 2.')
  script_dir = os.path.dirname(os.path.abspath(__file__))
  default_root = os.path.normpath(
      os.path.join(script_dir, '..', '..', '..', 'GNM'))
  parser.add_argument('--gnm_root', default=default_root,
                      help='Path to the GNM repository checkout.')
  parser.add_argument('--photo', default=MONA_LISA_URL,
                      help='Reference image: a local path or an http(s) URL.')
  parser.add_argument('--out',
                      default=os.path.join(script_dir, '..', 'assets',
                                           'gnm_skin_mona_lisa.png'),
                      help='Where to write the generated skin texture.')
  parser.add_argument('--model', default=DEFAULT_MODEL,
                      help='Image model id (Nano Banana 2 by default).')
  parser.add_argument('--api_size', default='2K',
                      choices=['512', '1K', '2K', '4K'],
                      help='Resolution requested from the image model.')
  parser.add_argument('--size', type=int, default=2048,
                      help='Side length of the texture that gets written.')
  parser.add_argument('--guide_size', type=int, default=1024,
                      help='Side length of the baked conditioning guides.')
  parser.add_argument('--pad', type=int, default=24,
                      help='Texels of edge padding bled into the unused atlas.')
  parser.add_argument('--hairline', type=float, default=DEFAULT_HAIRLINE,
                      help='Where the hair cap starts, 0 at the brows and 1 at '
                           'the top of the skull. Higher means more forehead.')
  parser.add_argument('--temple_drop', type=float,
                      default=DEFAULT_TEMPLE_DROP,
                      help='How far the hair dips at the temples. Lower keeps '
                           'it off the temples and widens the forehead.')
  parser.add_argument('--tone', type=float, default=0.8,
                      help='How hard to pull the painted complexion onto the '
                           "reference's measured skin colour, 0 to disable.")
  parser.add_argument('--deshade', type=float, default=0.7,
                      help='How hard to divide out the lighting the model '
                           'painted in, which otherwise darkens the sides.')
  parser.add_argument('--eye_narrow', type=float, default=0.08,
                      help='Fraction to narrow the eye aperture by in the '
                           'companion pose file. 0 writes no pose.')
  parser.add_argument('--attempts', type=int, default=3,
                      help='Retries when the model answers with text only.')
  parser.add_argument('--keys', default='',
                      help='keys.json holding {"gemini": {"apiKey": ...}}.')
  parser.add_argument('--style', default='auto',
                      choices=['auto', 'mona_lisa', 'photo', 'none'],
                      help='Likeness block appended to the layout prompt; '
                           'auto picks mona_lisa for the default reference.')
  parser.add_argument('--face_crop', default='auto',
                      help='"auto", "none", or left,top,right,bottom in 0..1 '
                           'marking the head in the reference image.')
  parser.add_argument('--extra', default='',
                      help='Extra instructions appended to the prompt.')
  parser.add_argument('--no_hair_pass', action='store_true',
                      help='Leave the head bald instead of generating hair and '
                           'compositing it into the scalp mask.')
  parser.add_argument('--feather', type=float, default=0.006,
                      help='Hairline softness as a fraction of texture width.')
  parser.add_argument('--guides_only', action='store_true',
                      help='Bake the guides and stop, without calling the API.')
  parser.add_argument('--no_preview', action='store_true',
                      help='Skip the textured head render.')
  parser.add_argument('--preview_size', type=int, default=768,
                      help='Side length of the preview render.')
  args = parser.parse_args()

  npz_path = os.path.join(args.gnm_root, 'gnm', 'shape', 'data', 'versions',
                          'v3_0', 'gnm_head.npz')
  if not os.path.exists(npz_path):
    raise SystemExit(f'GNM model not found at {npz_path}. Pass --gnm_root.')

  out_path = os.path.abspath(args.out)
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  stem = os.path.splitext(out_path)[0]

  print(f'Loading {npz_path}')
  mesh = load_skin_mesh(npz_path, args.hairline, args.temple_drop)
  print(f'GNM {mesh["variant"]} v{mesh["version"]}: '
        f'{len(mesh["skin_triangles"])} skin triangles, '
        f'{len(mesh["eye_triangles"])} eyeball triangles, '
        f'{int(mesh["scalp"].sum())} scalp vertices '
        f'(hairline {args.hairline}, temple drop {args.temple_drop})')

  print(f'Baking {args.guide_size}px UV guides...')
  relief, regions, coverage = build_guides(mesh, args.guide_size)
  relief_path = f'{stem}_guide_relief.png'
  regions_path = f'{stem}_guide_regions.png'
  relief.save(relief_path)
  regions.save(regions_path)
  print(f'Wrote {relief_path}')
  print(f'Wrote {regions_path}')
  if args.guides_only:
    return

  api_key = load_api_key(args.keys)
  photo = load_image(args.photo)
  # The relief leads: image models hold a layout far better when the first
  # image is the thing being edited rather than one reference among several.
  references = [relief, regions, photo]
  box = parse_crop(args.face_crop, args.photo)
  if box:
    face = crop_face(photo, box)
    face_path = f'{stem}_reference_face.png'
    face.save(face_path)
    references.append(face)
    print(f'Wrote {face_path} ({face.width}x{face.height} head crop)')

  reference_color = reference_skin_color(photo, box)
  if reference_color is not None:
    r, g, b = reference_color.astype(int)
    print(f'Reference complexion: rgb({r},{g},{b}) #{r:02x}{g:02x}{b:02x}')

  style = args.style
  if style == 'auto':
    style = 'mona_lisa' if args.photo == MONA_LISA_URL else 'photo'
  prompt = build_prompt(style, box is not None, args.extra, reference_color)
  prompt_path = f'{stem}_prompt.txt'
  with open(prompt_path, 'w', encoding='utf-8') as handle:
    handle.write(prompt)
  print(f'Wrote {prompt_path} (style={style}, {len(references)} references)')

  texture = generate_texture(args.model, prompt, references, args.api_size,
                             args.attempts, api_key)
  print(f'Received {texture.width}x{texture.height} skin pass')
  texture.save(f'{stem}_raw_skin.png')
  if texture.size != (args.size, args.size):
    texture = texture.resize((args.size, args.size), Image.LANCZOS)

  # Correct the complexion before the hair goes on, so the gain is measured
  # against skin only and never tints the hair.
  tone_alpha = build_mask_alpha(mesh, mesh['tone'], args.guide_size, 0.02)
  texture, shading = flatten_shading(texture, tone_alpha, args.deshade)
  if shading:
    spread, target = shading
    print(f'De-shaded the skin: luminance spread {spread:.1f} '
          f'about {target:.0f}')
  texture, report = match_skin_tone(texture, tone_alpha, reference_color,
                                    args.tone)
  if report:
    current, gain = report
    print('Skin tone: painted rgb(%d,%d,%d) -> gain %.2f/%.2f/%.2f'
          % (*current.astype(int), *gain))

  if not args.no_hair_pass:
    hair_prompt = HAIR_PROMPT.format(
        hair=MONA_LISA_HAIR if style == 'mona_lisa' else PHOTO_HAIR)
    hair_references = references[2:] or [photo]
    hair = generate_texture(args.model, hair_prompt, hair_references,
                            args.api_size, args.attempts, api_key)
    print(f'Received {hair.width}x{hair.height} hair pass')
    hair.save(f'{stem}_raw_hair.png')
    alpha = build_hair_alpha(mesh, args.guide_size, args.feather)
    texture = composite_hair(texture, hair, alpha)
    print(f'Composited hair over {100 * alpha.mean():.1f}% of the atlas')

  texture = pad_atlas(texture, coverage, args.pad)
  texture = match_seam_columns(texture)
  texture.save(out_path)

  if args.eye_narrow > 0:
    coefficients, names = solve_hooded_eyes(npz_path, args.eye_narrow)
    if coefficients is None:
      print('Could not solve the eye pose; skipping it.')
    else:
      pose_path = f'{stem}_pose.json'
      with open(pose_path, 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'model': 'GNM Head',
                'gnmVersion': mesh['version'],
                'expressionDim': len(names),
                'apertureChange': -args.eye_narrow,
                'expression': [round(float(c), 5) for c in coefficients],
            }, handle)
      strongest = int(np.argmax(np.abs(coefficients)))
      print(f'Wrote {pose_path} (eyes {100 * args.eye_narrow:.0f}% narrower, '
            f'strongest {names[strongest]} {coefficients[strongest]:+.2f})')
  print(f'Wrote {out_path}')

  if not args.no_preview:
    print('Rendering preview...')
    preview_path = f'{stem}_preview.png'
    render_preview(mesh, texture, args.preview_size).save(preview_path)
    print(f'Wrote {preview_path}')


if __name__ == '__main__':
  sys.exit(main())
