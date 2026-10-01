"""Atmosphere shell: an outer sphere with a sun-lit limb glow.

Additive material (Transparent + Emission) that works out, per pixel, how much air the
view ray crosses:

    P = shading point in Earth space, D = view ray, R = Earth radius, |P| = shell radius
    closest approach b^2 = |P|^2 - (P.D)^2
    path length L = ray hits Earth ? -(P.D) - sqrt(R^2 - b^2)   (stops at the ground)
                                   : -2 (P.D)                     (full chord through shell)
    density at closest approach = exp(-(b - R) / H),  H = scale height (fraction of shell)
    air column (Chapman): grazing  sqrt(2 pi R H) * density, capped by L
                          to ground  H / cos(landing zenith angle)
    alpha = 1 - exp(-density_k * column / H)

so the glow is thickest just above the horizon and fades to nothing at the shell's outer
edge (a plain Fresnel/Layer Weight rim does the opposite: brightest at the edge).

Light comes from the shared "Sun Direction" node group, which follows whatever light
Earth["atmosphere_sun"] points at (direction from its rotation, strength from its energy).
The same group is meant for day/night switching on other materials.
  - Sky Colour ramp on sun angle: night (none) -> twilight orange -> day blue
  - forward scattering: brighter rim when the sun is behind the Earth
"""

import bpy

from . import clouds, config, drivers, earth

SUN_DIR_GROUP = "Sun Direction"
SUN_PROP = "atmosphere_sun"

DEFAULTS = {
    # name: (default, soft_max, description)
    "atmosphere_density": (0.08, 1.0, "Optical depth looking straight down; the horizon is ~25x thicker"),
    "atmosphere_brightness": (0.6, 10.0, "Glow strength, multiplied by the sun's strength"),
    "atmosphere_thickness_km": (100.0, 500.0, "Shell height above the cloud shell, km"),
    "atmosphere_scale_height": (0.25, 1.0, "Density falloff as a fraction of the air layer; "
                                           "smaller = glow hugs the horizon"),
    "atmosphere_forward_scatter": (2.0, 10.0, "Extra glow when looking towards the sun (sun behind the Earth)"),
}


# ---------------------------------------------------------------- node helpers

def _drive(target, prop, earth_obj, path, expression="v", index=-1):
    return drivers.drive(target, prop, earth_obj, {"v": path}, expression, index)


def _value(nodes, earth_obj, prop, loc, expression="v"):
    n = nodes.new("ShaderNodeValue")
    n.name = n.label = prop
    n.location = loc
    _drive(n.outputs[0], "default_value", earth_obj, drivers.prop_path(prop), expression)
    return n.outputs[0]


class _G:
    """Tiny builder so the shader maths reads like the formula in the docstring."""

    def __init__(self, tree):
        self.nodes, self.links = tree.nodes, tree.links
        self.x = 0

    def _in(self, sock, value):
        if hasattr(value, "is_linked"):        # a socket
            self.links.new(value, sock)
        elif value is not None:
            sock.default_value = value

    def _place(self, n, y):
        n.location = (self.x, y)
        self.x += 40
        return n

    def math(self, op, a, b=None, clamp=False, y=0):
        n = self._place(self.nodes.new("ShaderNodeMath"), y)
        n.operation = op
        n.use_clamp = clamp
        self._in(n.inputs[0], a)
        self._in(n.inputs[1], b)
        return n.outputs[0]

    def vmath(self, op, a, b=None, y=0):
        n = self._place(self.nodes.new("ShaderNodeVectorMath"), y)
        n.operation = op
        self._in(n.inputs[0], a)
        self._in(n.inputs[1], b)
        return n.outputs["Value"] if op in ('DOT_PRODUCT', 'LENGTH') else n.outputs["Vector"]


# ---------------------------------------------------------------- shared sun group

def build_sun_group(earth_obj):
    """Node group with outputs Direction (world unit vector towards the sun) and Strength."""
    ng = bpy.data.node_groups.get(SUN_DIR_GROUP)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(SUN_DIR_GROUP, "ShaderNodeTree")
    ng.interface.new_socket("Direction", in_out='OUTPUT', socket_type='NodeSocketVector')
    ng.interface.new_socket("Strength", in_out='OUTPUT', socket_type='NodeSocketFloat')
    out = ng.nodes.new("NodeGroupOutput")
    out.location = (300, 0)

    xyz = ng.nodes.new("ShaderNodeCombineXYZ")
    xyz.name = xyz.label = "Towards Sun"
    for i in range(3):
        # A light shines down its local -Z, so its +Z axis (matrix column 2) points at the sun.
        # RNA paths index matrices [column][row].
        _drive(xyz.inputs[i], "default_value", earth_obj, f'["{SUN_PROP}"].matrix_world[2][{i}]')
    norm = ng.nodes.new("ShaderNodeVectorMath")
    norm.operation = 'NORMALIZE'
    norm.location = (150, 0)
    ng.links.new(xyz.outputs[0], norm.inputs[0])
    ng.links.new(norm.outputs["Vector"], out.inputs["Direction"])

    strength = ng.nodes.new("ShaderNodeValue")
    strength.name = strength.label = "Sun Strength"
    strength.location = (0, -200)
    _drive(strength.outputs[0], "default_value", earth_obj, f'["{SUN_PROP}"].data.energy')
    ng.links.new(strength.outputs[0], out.inputs["Strength"])
    return ng


# ---------------------------------------------------------------- material

def build_material(earth_obj, radius_km=config.RADIUS_KM):
    mat = bpy.data.materials.new(config.ATMOSPHERE_NAME)
    mat.surface_render_method = 'BLENDED'
    mat.use_backface_culling = True
    tree = mat.node_tree
    nodes, links = tree.nodes, tree.links
    nodes.remove(nodes["Principled BSDF"])
    out = nodes["Material Output"]
    g = _G(tree)

    # --- inputs
    tc = nodes.new("ShaderNodeTexCoord")
    tc.object = earth_obj
    geo = nodes.new("ShaderNodeNewGeometry")
    sun = nodes.new("ShaderNodeGroup")
    sun.node_tree = build_sun_group(earth_obj)
    for i, n in enumerate((tc, geo, sun)):
        n.location = (-600, 300 - i * 250)

    density_k = _value(nodes, earth_obj, "atmosphere_density", (-600, -500))
    brightness = _value(nodes, earth_obj, "atmosphere_brightness", (-600, -600))
    height_frac = _value(nodes, earth_obj, "atmosphere_scale_height", (-600, -700), "max(v, 0.001)")
    forward = _value(nodes, earth_obj, "atmosphere_forward_scatter", (-600, -800))

    # --- geometry of the view ray in Earth space
    P = tc.outputs["Object"]                                    # km from Earth centre
    d_world = g.vmath('SCALE', geo.outputs["Incoming"], None)   # Incoming points at the camera
    d_world.node.inputs["Scale"].default_value = -1.0
    to_obj = nodes.new("ShaderNodeVectorTransform")
    to_obj.vector_type = 'VECTOR'
    to_obj.convert_from, to_obj.convert_to = 'WORLD', 'OBJECT'
    links.new(d_world, to_obj.inputs[0])
    D = g.vmath('NORMALIZE', to_obj.outputs[0])

    R2 = radius_km * radius_km
    PD = g.vmath('DOT_PRODUCT', P, D)
    PP = g.vmath('DOT_PRODUCT', P, P)
    b2 = g.math('SUBTRACT', PP, g.math('MULTIPLY', PD, PD))
    b = g.math('SQRT', g.math('MAXIMUM', b2, 0.0))
    shell_r = g.math('SQRT', PP)
    thickness = g.math('MAXIMUM', g.math('SUBTRACT', shell_r, radius_km), 0.001)

    chord = g.math('MULTIPLY', PD, -2.0)
    to_ground = g.math('SUBTRACT', g.math('MULTIPLY', PD, -1.0),
                       g.math('SQRT', g.math('MAXIMUM', g.math('SUBTRACT', R2, b2), 0.0)))
    hits_earth = g.math('LESS_THAN', b, radius_km)
    path = g.math('ADD', chord, g.math('MULTIPLY', hits_earth, g.math('SUBTRACT', to_ground, chord)))

    alt = g.math('MAXIMUM', g.math('SUBTRACT', b, radius_km), 0.0)
    H = g.math('MULTIPLY', thickness, height_frac)
    dens = g.math('EXPONENT', g.math('MULTIPLY', g.math('DIVIDE', alt, H), -1.0))

    # Air column along the ray, in units of km of sea-level air (Chapman approximation):
    #   grazing ray:    sqrt(2 pi R H) * density at closest approach, capped by the chord
    #                   so the shell's outer edge fades to nothing
    #   ray to ground:  H / cos(zenith angle where it lands), capped at half a grazing path
    horizon = g.math('SQRT', g.math('MULTIPLY', H, 2.0 * 3.141592653589793 * radius_km))
    graze = g.math('MINIMUM', g.math('MULTIPLY', horizon, dens),
                   g.math('MULTIPLY', path, dens))
    scale_d = nodes.new("ShaderNodeVectorMath")
    scale_d.operation = 'SCALE'
    links.new(D, scale_d.inputs[0])
    links.new(to_ground, scale_d.inputs["Scale"])
    ground_pt = g.vmath('NORMALIZE', g.vmath('ADD', P, scale_d.outputs["Vector"]))
    cos_z = g.math('MAXIMUM', g.math('MULTIPLY', g.vmath('DOT_PRODUCT', ground_pt, D), -1.0), 0.001)
    ground = g.math('MINIMUM', g.math('DIVIDE', H, cos_z), g.math('MULTIPLY', horizon, 0.5))
    column = g.math('ADD', graze, g.math('MULTIPLY', hits_earth, g.math('SUBTRACT', ground, graze)))

    # density_k is the straight-down optical depth, so tau = k * column / H
    tau = g.math('MULTIPLY', g.math('DIVIDE', column, H), density_k)
    alpha = g.math('SUBTRACT', 1.0, g.math('EXPONENT', g.math('MULTIPLY', tau, -1.0)))
    alpha = g.math('MULTIPLY', alpha, g.math('SUBTRACT', 1.0, geo.outputs["Backfacing"]), clamp=True)

    # --- lighting from the shared sun group
    L = sun.outputs["Direction"]
    mu = g.vmath('DOT_PRODUCT', geo.outputs["Normal"], L, y=-300)
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = ramp.label = "Sky Colour"
    ramp.location = (g.x, -300)
    fac = g.math('MULTIPLY_ADD', mu, 0.5, y=-300)    # -1..1 -> 0..1
    fac.node.inputs[2].default_value = 0.5
    links.new(fac, ramp.inputs["Fac"])
    els = ramp.color_ramp.elements
    # mu = cos(sun angle at the shell) mapped to 0..1: 0.5 is the terminator. Real twilight
    # spans only a few degrees, so the orange band is kept narrow and dim.
    els[0].position, els[0].color = 0.465, (0.0, 0.0, 0.0, 1.0)         # night
    els[1].position, els[1].color = 1.00, (0.45, 0.70, 1.00, 1.0)       # noon
    for pos, col in ((0.487, (0.50, 0.15, 0.04, 1.0)),                  # twilight
                     (0.510, (0.30, 0.55, 1.00, 1.0))):                 # day
        e = els.new(pos)
        e.color = col

    cos_sun = g.vmath('DOT_PRODUCT', d_world, L, y=-500)
    phase = g.math('ADD', 1.0, g.math('MULTIPLY', forward,
                   g.math('POWER', g.math('MAXIMUM', cos_sun, 0.0, y=-500), 8.0, y=-500), y=-500), y=-500)
    strength = g.math('MULTIPLY', g.math('MULTIPLY', sun.outputs["Strength"], brightness, y=-500), phase, y=-500)
    strength = g.math('MULTIPLY', strength, alpha, y=-500)

    emit = nodes.new("ShaderNodeEmission")
    emit.location = (g.x + 200, -200)
    links.new(ramp.outputs["Color"], emit.inputs["Color"])
    links.new(strength, emit.inputs["Strength"])

    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (g.x + 200, 0)
    add = nodes.new("ShaderNodeAddShader")
    add.location = (g.x + 400, 0)
    links.new(transparent.outputs[0], add.inputs[0])
    links.new(emit.outputs[0], add.inputs[1])
    links.new(add.outputs[0], out.inputs["Surface"])
    out.location = (g.x + 600, 0)
    return mat


# ---------------------------------------------------------------- object

def build(earth_obj, sun, settings=None):
    """Atmosphere shell above the cloud shell, lit by `sun`. `settings` overrides DEFAULTS."""
    settings = settings or {}
    earth_obj[SUN_PROP] = sun
    for name, (default, soft_max, desc) in DEFAULTS.items():
        drivers.set_prop(earth_obj, name, settings.get(name, default), desc, soft_max=soft_max)

    mesh = earth_obj.data.copy()
    mesh.name = config.ATMOSPHERE_NAME
    mesh.materials.clear()
    mesh.materials.append(build_material(earth_obj))
    for poly in mesh.polygons:
        poly.material_index = 0

    atmo = bpy.data.objects.new(config.ATMOSPHERE_NAME, mesh)
    for coll in earth_obj.users_collection:
        coll.objects.link(atmo)
    atmo.parent = earth_obj
    atmo.hide_select = True
    # Camera-only: no shadows, no light bounced onto the Earth.
    atmo.visible_shadow = False
    atmo.visible_diffuse = atmo.visible_glossy = atmo.visible_transmission = False
    atmo.visible_volume_scatter = False
    earth.add_sphere_modifiers(atmo, 3, 4)

    r = config.RADIUS_KM
    variables = {"exag": drivers.prop_path(config.EXAGGERATION_PROP),
                 "thick": drivers.prop_path("atmosphere_thickness_km")}
    for axis in range(3):
        drivers.drive(atmo, "scale", earth_obj, variables,
                      f"({r} + {clouds.shell_altitude_expr()} + thick) / {r}", axis)
    return atmo
