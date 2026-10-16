"""Atmosphere shell, and the sunlight that crosses the air to reach the ground and clouds.

The atmosphere is an outer sphere whose material ray-marches the view ray through an
ideal (analytic) shell, so nothing depends on the faceted mesh:

    P = shading point, D = view ray (Earth space, km), R = Earth radius, Rs = shell radius
    closest approach b^2 = |P|^2 - (P.D)^2           (the same anywhere along the ray)
    enter at t0 = -(P.D) - sqrt(Rs^2 - b^2)
    leave at t1 = -(P.D) + sqrt(Rs^2 - b^2), or at the ground -(P.D) - sqrt(R^2 - b^2)
    SAMPLES midpoints between t0 and t1; at each one:
        density   rho = exp(-altitude / H),  H = scale height, tapered to 0 at the shell top
        sunlight  T_sun = exp(-k_sun * beta * Chapman column towards the sun) * Earth shadow
        view      T_view = exp(-k * beta * air column back to the camera)
        light    += k * beta * T_sun * T_view * rho * ds / H

beta is Rayleigh's 1/lambda^4 at 680 / 550 / 440 nm, so blue is scattered more and is lost
first along a long sun path: the limb is blue by day and turns orange, then red, then dark
as its air passes into the Earth's shadow, on its own, with no colour ramp. The shadow edge is
softened by the sun's 0.53 deg disc, seen across the grazing path (the penumbra).

The Chapman column (air crossed towards the sun, in units of a vertical column) is
    sun above the local horizon (mu >= 0):  rho / sqrt(mu^2 + 2H / (pi r))
    sun below it (mu < 0): the ray dips to h0 = r sqrt(1 - mu^2) - R and climbs out again:
                           2 rho(h0) sqrt(pi r / 2H) - (the sun-above column at |mu|)
which agree at mu = 0. Below h0 = 0 the Earth itself blocks the sun.

The Earth and the clouds are lit through the "Surface Light" group:
  Direct  Air Sample at the surface: sunlight transmittance times the Earth's shadow with its
          penumbra. The low sun dims and reddens before the terminator, and points above the
          ground (clouds, peaks) keep grazing, reddened sun past it until the Earth's limb
          blocks it. It multiplies the albedo, so it tints the sun lamp's light.
  Sky     light from the air above that is still in sunlight (closed form, see the group):
          wraps light past the terminator into the twilight band, fading and reddening over
          several degrees (emission = albedo * Sky * sun strength / pi)
With --no-atmosphere there is no air, so the surface keeps the sun lamp's plain light.

Light comes from the shared "Sun Direction" node group, which follows whatever light
Earth["atmosphere_sun"] points at (direction from its rotation, strength from its energy).
"""

import math

import bpy

from . import clouds, config, drivers, earth

SUN_DIR_GROUP = "Sun Direction"
AIR_PARAMS_GROUP = "Air Parameters"
AIR_SAMPLE_GROUP = "Air Sample"
ATMO_STEP_GROUP = "Atmosphere Step"
SURFACE_GROUP = "Surface Light"
SUN_PROP = "atmosphere_sun"

SAMPLES = 16                         # ray-march steps through the shell
SUN_ANGULAR_RADIUS = math.radians(0.533 / 2)
# Rayleigh extinction ~ 1/lambda^4 at 680 / 550 / 440 nm, relative to 550 nm (green).
BETA = tuple((550.0 / nm) ** 4 for nm in (680.0, 550.0, 440.0))
SKY_COLOUR_DEFAULT = (1.0, 1.0, 1.0)
# Skylight on the ground is whiter than single-scattered Rayleigh blue (multiple scattering,
# ground bounce): roughly a 10,000 K sky, linear RGB.
SKYLIGHT_RGB = (0.62, 0.78, 1.0)

DEFAULTS = {
    # name: (default, soft_max, description)
    "atmosphere_density": (0.08, 1.0, "Optical depth (green) looking straight down; Rayleigh "
                                      "sets red / blue from it"),
    "atmosphere_brightness": (0.6, 10.0, "Glow strength, multiplied by the sun's strength"),
    "atmosphere_thickness_km": (100.0, 500.0, "Shell height above the cloud shell, km"),
    "atmosphere_scale_height": (0.25, 1.0, "Density falloff as a fraction of the air layer; "
                                           "smaller = glow hugs the horizon"),
    "atmosphere_forward_scatter": (2.0, 10.0, "Extra glow when looking towards the sun (sun behind the Earth)"),
    "atmosphere_sun_extinction": (1.0, 10.0, "Reddening of sunlight crossing the air on its way in: "
                                             "scales the density along the sun's path"),
    "atmosphere_twilight": (0.1, 1.0, "Skylight on the ground and clouds, relative to the sun: "
                                       "light that wraps past the day/night line"),
}

V, F, C = 'NodeSocketVector', 'NodeSocketFloat', 'NodeSocketColor'


# ---------------------------------------------------------------- node helpers

def _drive(target, prop, earth_obj, path, expression="v", index=-1):
    return drivers.drive(target, prop, earth_obj, {"v": path}, expression, index)


class _G:
    """Tiny builder so the shader maths reads like the formulas in the docstring."""

    def __init__(self, tree):
        self.nodes, self.links = tree.nodes, tree.links
        self.x = 0

    def _in(self, sock, value):
        if hasattr(value, "is_linked"):        # a socket
            self.links.new(value, sock)
        elif value is not None:
            sock.default_value = value

    def _place(self, n, y=0):
        n.location = (self.x, y)
        self.x += 40
        return n

    def math(self, op, a, b=None, c=None, clamp=False, y=0):
        n = self._place(self.nodes.new("ShaderNodeMath"), y)
        n.operation = op
        n.use_clamp = clamp
        self._in(n.inputs[0], a)
        self._in(n.inputs[1], b)
        self._in(n.inputs[2], c)
        return n.outputs[0]

    def vmath(self, op, a, b=None, y=0):
        n = self._place(self.nodes.new("ShaderNodeVectorMath"), y)
        n.operation = op
        self._in(n.inputs[0], a)
        self._in(n.inputs[1], b)
        return n.outputs["Value"] if op in ('DOT_PRODUCT', 'LENGTH') else n.outputs["Vector"]

    def vscale(self, v, s, y=0):
        n = self._place(self.nodes.new("ShaderNodeVectorMath"), y)
        n.operation = 'SCALE'
        self._in(n.inputs[0], v)
        self._in(n.inputs["Scale"], s)
        return n.outputs["Vector"]

    def exp_neg(self, v, y=0):
        """exp(-v) per component (Vector Math has no exponent in Blender 4.2)."""
        sep = self._place(self.nodes.new("ShaderNodeSeparateXYZ"), y)
        self._in(sep.inputs[0], v)
        comb = self.nodes.new("ShaderNodeCombineXYZ")
        for i in range(3):
            self.links.new(self.math('EXPONENT', self.math('MULTIPLY', sep.outputs[i], -1.0, y=y), y=y),
                           comb.inputs[i])
        self._place(comb, y)
        return comb.outputs[0]

    def group(self, tree, inputs=None, y=0):
        n = self._place(self.nodes.new("ShaderNodeGroup"), y)
        n.node_tree = tree
        n.label = tree.name
        for name, value in (inputs or {}).items():
            self._in(n.inputs[name], value)
        return n

    def to_object(self, world_vec, y=0):
        """World direction -> the shaded object's space, normalised (shells are scaled)."""
        n = self._place(self.nodes.new("ShaderNodeVectorTransform"), y)
        n.vector_type = 'VECTOR'
        n.convert_from, n.convert_to = 'WORLD', 'OBJECT'
        self._in(n.inputs[0], world_vec)
        return self.vmath('NORMALIZE', n.outputs[0], y=y)


def _new_group(name, inputs=(), outputs=()):
    ng = bpy.data.node_groups.new(name, "ShaderNodeTree")
    for n, t in inputs:
        ng.interface.new_socket(n, in_out='INPUT', socket_type=t)
    for n, t in outputs:
        ng.interface.new_socket(n, in_out='OUTPUT', socket_type=t)
    gin = ng.nodes.new("NodeGroupInput")
    gin.location = (-800, 0)
    gout = ng.nodes.new("NodeGroupOutput")
    return ng, gin, gout


def _finish(g, gout):
    gout.location = (g.x + 300, 0)


# ---------------------------------------------------------------- shared groups

# The Sun Direction drivers read the lamp object itself (not a path through
# Earth["atmosphere_sun"]): Blender only re-evaluates a driver when its target ID changes,
# so a path through the Earth's pointer property went stale when the Sun was rotated,
# replaced or deleted. bind_sun() retargets them.
_SUN_PATHS = {**{f'nodes["Towards Sun"].inputs[{i}].default_value': f"matrix_world[2][{i}]" for i in range(3)},
              'nodes["Sun Strength"].outputs[0].default_value': "data.energy"}


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
    xyz.inputs[2].default_value = 1.0
    for i in range(3):
        # A light shines down its local -Z, so its +Z axis (matrix column 2) points at the sun.
        xyz.inputs[i].driver_add("default_value")
    norm = ng.nodes.new("ShaderNodeVectorMath")
    norm.operation = 'NORMALIZE'
    norm.location = (150, 0)
    ng.links.new(xyz.outputs[0], norm.inputs[0])
    ng.links.new(norm.outputs["Vector"], out.inputs["Direction"])

    strength = ng.nodes.new("ShaderNodeValue")
    strength.name = strength.label = "Sun Strength"
    strength.location = (0, -200)
    strength.outputs[0].driver_add("default_value")
    ng.links.new(strength.outputs[0], out.inputs["Strength"])
    bind_sun(earth_obj, earth_obj.get(SUN_PROP))
    return ng


def bind_sun(earth_obj, sun_obj):
    """Make `sun_obj` the light the atmosphere, ground and clouds follow (None: no sun)."""
    if earth_obj.get(SUN_PROP) is not sun_obj:
        earth_obj[SUN_PROP] = sun_obj
    ng = bpy.data.node_groups.get(SUN_DIR_GROUP)
    if ng is None or ng.animation_data is None:
        return
    for fc in ng.animation_data.drivers:
        path = _SUN_PATHS.get(fc.data_path)
        if path is None:
            continue
        drv = fc.driver
        drv.type = 'SCRIPTED'
        var = drv.variables[0] if drv.variables else drv.variables.new()
        var.name, var.type = "v", 'SINGLE_PROP'
        t = var.targets[0]
        t.id_type = 'OBJECT'
        if sun_obj is None:
            drv.expression = "0"          # no Sun in the scene: no sunlight in the shaders
        else:
            t.id, t.data_path = sun_obj, path
            drv.expression = "v"


def bound_sun():
    """The lamp the Sun Direction drivers currently read (None if unbound)."""
    ng = bpy.data.node_groups.get(SUN_DIR_GROUP)
    if ng is None or ng.animation_data is None:
        return None
    for fc in ng.animation_data.drivers:
        if fc.driver.expression == "v" and fc.driver.variables:
            return fc.driver.variables[0].targets[0].id
    return None


def build_params_group(earth_obj):
    """Every driven atmosphere constant in one place (radii, scale height, extinction, ...)."""
    ng = bpy.data.node_groups.get(AIR_PARAMS_GROUP)
    if ng is not None:
        return ng
    ng, _, gout = _new_group(AIR_PARAMS_GROUP, outputs=(
        ("Ground Radius", F), ("Shell Radius", F), ("Scale Height", F),
        ("View Extinction", V), ("Sun Extinction", V), ("Penumbra", F),
        ("Twilight", F), ("Sky Colour", C), ("Night Lights", F), ("Night Floor", F)))
    nodes, g = ng.nodes, _G(ng)
    r = config.RADIUS_KM
    p = drivers.prop_path
    shell_vars = {"exag": p(config.EXAGGERATION_PROP), "thick": p("atmosphere_thickness_km"),
                  "relief": p(clouds.RELIEF_KM_PROP)}
    air_km = f"({clouds.shell_altitude_expr()} + relief + thick)"   # ground to top of the shell

    def driven(label, variables, expr, y):
        n = nodes.new("ShaderNodeValue")
        n.name = n.label = label
        n.location = (-400, y)
        drivers.drive(n.outputs[0], "default_value", earth_obj, variables, expr)
        return n.outputs[0]

    def driven_rgb(label, variables, expr, y):
        n = nodes.new("ShaderNodeCombineXYZ")
        n.name = n.label = label
        n.location = (-400, y)
        for i, beta in enumerate(BETA):
            drivers.drive(n.inputs[i], "default_value", earth_obj, variables, f"{expr} * {beta:.4f}")
        return n.outputs[0]

    ground = nodes.new("ShaderNodeValue")
    ground.name = ground.label = "Ground Radius"
    ground.location = (-400, 300)
    ground.outputs[0].default_value = r
    shell = driven("Shell Radius", shell_vars, f"{r} + {air_km}", 200)
    H = driven("Scale Height", {**shell_vars, "f": p("atmosphere_scale_height")},
               f"{air_km} * max(f, 0.001)", 100)
    view_ext = driven_rgb("View Extinction", {"k": p("atmosphere_density")}, "k", 0)
    sun_ext = driven_rgb("Sun Extinction", {"k": p("atmosphere_density"), "e": p("atmosphere_sun_extinction")},
                         "k * e", -150)
    twilight = driven("Twilight", {"t": p("atmosphere_twilight")}, "t", -300)
    night = driven("Night Lights", {"n": p(earth.NIGHT_PROP)}, "n", -550)
    night_floor = driven("Night Floor", {"f": p(earth.NIGHT_FLOOR_PROP)}, "f", -650)
    sky = nodes.new("ShaderNodeRGB")
    sky.name = sky.label = "Sky Colour"
    sky.location = (-400, -400)
    sky.outputs[0].default_value = (*SKY_COLOUR_DEFAULT, 1.0)
    # The sun's disc seen across a grazing path ~sqrt(2 R H) long blurs the shadow edge this much.
    penumbra = g.math('MULTIPLY', g.math('SQRT', g.math('MULTIPLY', H, 2.0 * r)), SUN_ANGULAR_RADIUS)

    for name, sock in (("Ground Radius", ground.outputs[0]), ("Shell Radius", shell), ("Scale Height", H),
                       ("View Extinction", view_ext), ("Sun Extinction", sun_ext), ("Penumbra", penumbra),
                       ("Twilight", twilight), ("Sky Colour", sky.outputs[0]),
                       ("Night Lights", night), ("Night Floor", night_floor)):
        ng.links.new(sock, gout.inputs[name])
    _finish(g, gout)
    return ng


def build_air_sample_group(earth_obj):
    """Position (Earth space, km) + Sun (Earth-space unit) -> air Density, Sun Transmittance."""
    ng = bpy.data.node_groups.get(AIR_SAMPLE_GROUP)
    if ng is not None:
        return ng
    ng, gin, gout = _new_group(AIR_SAMPLE_GROUP, inputs=(("Position", V), ("Sun", V)),
                               outputs=(("Density", F), ("Sun Transmittance", C)))
    g = _G(ng)
    prm = g.group(build_params_group(earth_obj), y=-400).outputs
    R, H = prm["Ground Radius"], prm["Scale Height"]
    X, L = gin.outputs["Position"], gin.outputs["Sun"]

    r = g.math('MAXIMUM', g.vmath('LENGTH', X), 1.0)
    h = g.math('MAXIMUM', g.math('SUBTRACT', r, R), 0.0)
    rho = g.math('EXPONENT', g.math('MULTIPLY', g.math('DIVIDE', h, H), -1.0))
    mu = g.math('DIVIDE', g.vmath('DOT_PRODUCT', X, L), r)
    mu2 = g.math('MULTIPLY', mu, mu)
    inv_x = g.math('DIVIDE', g.math('MULTIPLY', H, 2.0 / math.pi), r)        # 2H / (pi r)
    ch = g.math('DIVIDE', 1.0, g.math('SQRT', g.math('ADD', mu2, inv_x)))     # Chapman, sun up
    ch0 = g.math('DIVIDE', 1.0, g.math('SQRT', inv_x))                        # ... at the horizon
    col_up = g.math('MULTIPLY', rho, ch)

    # Sun below the local horizon: the ray dips to h0, then climbs out the far side.
    h0 = g.math('SUBTRACT', g.math('MULTIPLY', r, g.math('SQRT', g.math('MAXIMUM',
                g.math('SUBTRACT', 1.0, mu2), 0.0))), R)
    rho0 = g.math('EXPONENT', g.math('MINIMUM', g.math('MULTIPLY', g.math('DIVIDE', h0, H), -1.0), 60.0))
    col_down = g.math('SUBTRACT', g.math('MULTIPLY', g.math('MULTIPLY', rho0, ch0), 2.0), col_up)
    up = g.math('GREATER_THAN', mu, 0.0)
    # Weighted sum, not a + up * (b - a): col_down can reach 1e27, which would swamp col_up.
    col = g.math('ADD', g.math('MULTIPLY', up, col_up),
                 g.math('MULTIPLY', g.math('SUBTRACT', 1.0, up), col_down))

    # Earth's shadow: the sun ray passes below the ground (h0 < 0). Smoothstep over the penumbra.
    w = prm["Penumbra"]
    h0_eff = g.math('MULTIPLY_ADD', up, 1.0e5, h0)                  # sun up: never shadowed
    s = g.math('DIVIDE', g.math('ADD', h0_eff, w), g.math('MULTIPLY', w, 2.0), clamp=True)
    lit = g.math('MULTIPLY', g.math('MULTIPLY', s, s), g.math('MULTIPLY_ADD', s, -2.0, 3.0))

    T = g.vscale(g.exp_neg(g.vscale(prm["Sun Extinction"], col)), lit)
    # Density for the glow tapers to exactly 0 at the top of the shell, (rho - rho_top) /
    # (1 - rho_top), instead of stopping at rho_top (~2 %): the limb fades out smoothly and still
    # ends at the shell's height. (The sun-path columns above keep the plain exponential; the
    # difference is that same ~2 % of the thinnest air.)
    rho_top = g.math('EXPONENT', g.math('MULTIPLY', g.math('DIVIDE',
                     g.math('SUBTRACT', prm["Shell Radius"], R), H), -1.0))
    rho_glow = g.math('DIVIDE', g.math('MAXIMUM', g.math('SUBTRACT', rho, rho_top), 0.0),
                      g.math('MAXIMUM', g.math('SUBTRACT', 1.0, rho_top), 1e-4))
    ng.links.new(rho_glow, gout.inputs["Density"])
    ng.links.new(T, gout.inputs["Sun Transmittance"])
    _finish(g, gout)
    return ng


def build_step_group(earth_obj):
    """One ray-march sample: light scattered towards the camera at P + t D over length ds."""
    ng = bpy.data.node_groups.get(ATMO_STEP_GROUP)
    if ng is not None:
        return ng
    ng, gin, gout = _new_group(ATMO_STEP_GROUP,
                               inputs=(("P", V), ("D", V), ("Sun", V), ("t", F), ("ds", F), ("Column", F)),
                               outputs=(("Column", F), ("Light", C)))
    g = _G(ng)
    prm = g.group(build_params_group(earth_obj), y=-400).outputs
    i = gin.outputs
    X = g.vmath('ADD', i["P"], g.vscale(i["D"], i["t"]))
    air = g.group(build_air_sample_group(earth_obj), {"Position": X, "Sun": i["Sun"]}).outputs
    dcol = g.math('DIVIDE', g.math('MULTIPLY', air["Density"], i["ds"]), prm["Scale Height"])
    mid = g.math('MULTIPLY_ADD', dcol, 0.5, i["Column"])           # column back to the camera
    t_view = g.exp_neg(g.vscale(prm["View Extinction"], mid))
    light = g.vmath('MULTIPLY', g.vmath('MULTIPLY', prm["View Extinction"], air["Sun Transmittance"]), t_view)
    ng.links.new(g.vscale(light, dcol), gout.inputs["Light"])
    ng.links.new(g.math('ADD', i["Column"], dcol), gout.inputs["Column"])
    _finish(g, gout)
    return ng


def build_surface_group(earth_obj):
    """Albedo in -> Base Color / Specular Tint / Emission for a sunlit surface under the air."""
    ng = bpy.data.node_groups.get(SURFACE_GROUP)
    if ng is not None:
        return ng
    ng, gin, gout = _new_group(SURFACE_GROUP, inputs=(("Albedo", C), ("Position", V), ("Night", C), ("Height", F)),
                               outputs=(("Base Color", C), ("Specular Tint", C), ("Emission Color", C),
                                        ("Emission Strength", F), ("Sun Cosine", F)))
    g = _G(ng)
    prm = g.group(build_params_group(earth_obj), y=-400).outputs
    sun = g.group(build_sun_group(earth_obj), y=-600).outputs
    L = g.to_object(sun["Direction"])
    R, H, ext = prm["Ground Radius"], prm["Scale Height"], prm["Sun Extinction"]
    # Only the direction comes from the mesh (Earth space, linked per material: EEVEE ignores a
    # Texture Coordinate node's object inside a group). The radius is the ideal sphere's, R +
    # Height (the true, unexaggerated altitude of the surface), so the lighting -- above all the
    # Earth-shadow edge -- doesn't step from face to face at low subdivision levels.
    pos = g.vscale(g.vmath('NORMALIZE', gin.outputs["Position"]), g.math('ADD', R, gin.outputs["Height"]))

    # r, mu = radius and sun cosine at the point. The skylight below uses the same Chapman
    # physics as Air Sample in closed form (one lookup, not a column of samples), since this
    # group runs in all 1,296 tile materials.
    r = g.math('MAXIMUM', g.vmath('LENGTH', pos), 1.0)
    mu = g.math('DIVIDE', g.vmath('DOT_PRODUCT', pos, L), r)
    mu2 = g.math('MULTIPLY', mu, mu)
    inv_x = g.math('DIVIDE', g.math('MULTIPLY', H, 2.0 / math.pi), r)          # 2H / (pi r)
    ch = g.math('DIVIDE', 1.0, g.math('SQRT', g.math('ADD', mu2, inv_x)))       # Chapman(|mu|)
    ch0 = g.math('DIVIDE', 1.0, g.math('SQRT', inv_x))                          # ... horizon
    rho = g.math('EXPONENT', g.math('DIVIDE', g.math('MINIMUM', g.math('SUBTRACT', R, r), 0.0), H))
    up = g.math('GREATER_THAN', mu, 0.0)

    # Direct: sunlight left after its path through the air, times the Earth's shadow with its
    # penumbra. The shadow matters: bump-mapped cloud tops tilt towards the sun past the
    # terminator and cast no shadow of their own, so without it the sun lamp lights them deep
    # into the night. Higher points (clouds, peaks) see the sun for longer, in reddened light.
    direct = g.group(build_air_sample_group(earth_obj), {"Position": pos, "Sun": L}).outputs["Sun Transmittance"]

    # Sky: past the terminator, sunlight grazing the ground (radius R) reaches radius
    # R / cos(depression) above the point, so the air below a_min = R / cos(dep) - r is in the
    # Earth's shadow (clouds sit higher, so they stay lit longer). The lit part of the column is
    # exp(-a_min / H); its light comes in along a grazing path, Chapman 2 ch0 - ch(|mu|)
    # (= ch at mu = 0), taken one scale height up.
    a_min = g.math('MULTIPLY', g.math('SUBTRACT', 1.0, up), g.math('MAXIMUM', g.math('SUBTRACT', g.math('DIVIDE', R,
                   g.math('SQRT', g.math('MAXIMUM', g.math('SUBTRACT', 1.0, mu2), 1e-4))), r), 0.0))
    lit = g.math('EXPONENT', g.math('MULTIPLY', g.math('DIVIDE', a_min, H), -1.0))
    ch_sky = g.math('ADD', g.math('MULTIPLY', up, ch), g.math('MULTIPLY', g.math('SUBTRACT', 1.0, up),
                    g.math('SUBTRACT', g.math('MULTIPLY', ch0, 2.0), ch)))
    rho_sky = g.math('MULTIPLY', g.math('MULTIPLY', rho, lit), math.exp(-1.0))
    sky = g.vscale(g.exp_neg(g.vscale(ext, g.math('MULTIPLY', rho_sky, ch_sky))), lit)
    sky = g.vmath('MULTIPLY', g.vmath('MULTIPLY', sky, SKYLIGHT_RGB), prm["Sky Colour"])
    sky = g.vscale(sky, prm["Twilight"])

    albedo = gin.outputs["Albedo"]
    ng.links.new(g.vmath('MULTIPLY', albedo, direct), gout.inputs["Base Color"])
    ng.links.new(direct, gout.inputs["Specular Tint"])
    skylight = g.vscale(g.vmath('MULTIPLY', albedo, sky), g.math('MULTIPLY', sun["Strength"], 1.0 / math.pi))

    # Night lights (Black Marble): the map above its moonlit-ground floor, switched on as the
    # sun sets -- full below 6 deg depression (end of civil twilight), off above 3 deg elevation.
    # The floor is on luminance, and scales the colour, so dim background of any hue (the
    # map's blue-tinted moonlit desert and sea) goes while the lights keep their colour.
    night_rgb = gin.outputs["Night"]
    lum = g.vmath('DOT_PRODUCT', night_rgb, (0.2126, 0.7152, 0.0722))
    keep = g.math('DIVIDE', g.math('MAXIMUM', g.math('SUBTRACT', lum, prm["Night Floor"]), 0.0),
                  g.math('MAXIMUM', lum, 1e-4))
    lights = g.vscale(night_rgb, keep)
    lo, hi = math.sin(math.radians(-6.0)), math.sin(math.radians(3.0))
    s = g.math('DIVIDE', g.math('SUBTRACT', mu, lo), hi - lo, clamp=True)
    day = g.math('MULTIPLY', g.math('MULTIPLY', s, s), g.math('MULTIPLY_ADD', s, -2.0, 3.0))
    dark = g.math('MULTIPLY', g.math('SUBTRACT', 1.0, day), prm["Night Lights"])
    ng.links.new(g.vmath('ADD', skylight, g.vscale(lights, dark)), gout.inputs["Emission Color"])
    one = g._place(ng.nodes.new("ShaderNodeValue"))
    one.outputs[0].default_value = 1.0            # the colour above already carries the strengths
    ng.links.new(one.outputs[0], gout.inputs["Emission Strength"])
    ng.links.new(mu, gout.inputs["Sun Cosine"])
    _finish(g, gout)
    return ng


# Cloud shadows: the shell sits at 8 km x terrain exaggeration (160 km by default), so with the
# sun low its shadows land 1000+ km from their clouds, as tile-shaped dark patches. They fade
# out between these sun elevations (shadow rays only; the clouds themselves are unchanged).
CLOUD_SHADOW_FADE_DEG = (5.0, 25.0)
_SURFACE_OUTPUTS = ("Base Color", "Specular Tint", "Emission Color", "Emission Strength")
CLOUD_SHADOW_NODE = "Low Sun Shadow"


def _cloud_shadow_fade(mat, surface_node):
    """Alpha for shadow rays = alpha * smoothstep(sun elevation). Idempotent."""
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    if CLOUD_SHADOW_NODE in nodes:
        return
    bsdf = nodes["Principled BSDF"]
    alpha = bsdf.inputs["Alpha"]
    if not alpha.is_linked:
        return
    src = alpha.links[0].from_socket
    x, y = bsdf.location.x - 220, bsdf.location.y - 400
    lo, hi = (math.sin(math.radians(d)) for d in CLOUD_SHADOW_FADE_DEG)
    ramp = nodes.new("ShaderNodeMapRange")
    ramp.name = ramp.label = CLOUD_SHADOW_NODE
    ramp.interpolation_type = 'SMOOTHSTEP'
    ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = lo, hi
    ramp.location = (x - 200, y)
    links.new(surface_node.outputs["Sun Cosine"], ramp.inputs["Value"])
    path = nodes.new("ShaderNodeLightPath")
    path.location = (x - 400, y - 150)
    # factor = 1 for camera rays; ramp for shadow rays
    fac = nodes.new("ShaderNodeMix")
    fac.data_type = 'FLOAT'
    fac.location = (x - 50, y)
    links.new(path.outputs["Is Shadow Ray"], fac.inputs["Factor"])
    fac.inputs["A"].default_value = 1.0
    links.new(ramp.outputs["Result"], fac.inputs["B"])
    mul = nodes.new("ShaderNodeMath")
    mul.operation = 'MULTIPLY'
    mul.location = (x + 120, y)
    links.new(src, mul.inputs[0])
    links.new(fac.outputs["Result"], mul.inputs[1])
    links.new(mul.outputs[0], alpha)


def light_surfaces(earth_obj, materials):
    """Route each Principled material's albedo through Surface Light. Idempotent: also
    re-applies the per-material settings to tiles that already have it."""
    group = build_surface_group(earth_obj)
    for mat in materials:
        nodes = mat.node_tree.nodes
        n = nodes.get(SURFACE_GROUP) or _add_surface_node(earth_obj, mat, group)
        bsdf, links = nodes["Principled BSDF"], mat.node_tree.links
        for name in _SURFACE_OUTPUTS:                   # always by name (see _swap_group)
            if not any(l.from_node is n and l.from_socket.name == name for l in bsdf.inputs[name].links):
                for l in list(bsdf.inputs[name].links):
                    links.remove(l)
                links.new(n.outputs[name], bsdf.inputs[name])
        is_cloud = mat.name.startswith(config.CLOUDS_NAME)
        # Lighting uses the true altitude, not the exaggerated shell: clouds see the sun ~3 deg
        # past the terminator, as real 8 km cloud tops do.
        n.inputs["Height"].default_value = clouds.CLOUD_ALT_KM if is_cloud else 0.0
        if is_cloud:
            _cloud_shadow_fade(mat, n)


def _add_surface_node(earth_obj, mat, group):
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    base = bsdf.inputs["Base Color"]
    n = nodes.new("ShaderNodeGroup")
    n.name = n.label = SURFACE_GROUP
    n.node_tree = group
    n.location = (bsdf.location.x - 220, bsdf.location.y + 250)
    tc = nodes.new("ShaderNodeTexCoord")
    tc.name = tc.label = "Earth Space"
    tc.object = earth_obj
    tc.location = (n.location.x - 200, n.location.y)
    links.new(tc.outputs["Object"], n.inputs["Position"])
    if base.is_linked:
        src = base.links[0].from_socket
        links.remove(base.links[0])
        links.new(src, n.inputs["Albedo"])
    else:
        n.inputs["Albedo"].default_value = base.default_value
    night = nodes.get(earth.NIGHT_NODE)
    if night is not None:
        links.new(night.outputs["Color"], n.inputs["Night"])
    # The skylight emission is ambient fill, not a light source: without this Cycles turns
    # every subdivided Earth and cloud triangle into a light (gigabytes of light tree).
    mat.cycles.emission_sampling = 'NONE'
    return n


# The shell mesh is drawn slightly larger than the sphere the shader integrates (Shell Radius):
# its flat faces sit inside their vertices' sphere (by up to 1 - cos(7.1 deg) at subdivision
# level 0), and any glow outside the mesh would be clipped into a faceted edge. The shader works
# from the ideal radius, so the size of the glow doesn't change.
SHELL_MESH_MARGIN = 1.0 / math.cos(math.radians(7.1))


def drive_shell_scale(atmo, earth_obj):
    r = config.RADIUS_KM
    variables = {"exag": drivers.prop_path(config.EXAGGERATION_PROP),
                 "thick": drivers.prop_path("atmosphere_thickness_km"),
                 "relief": drivers.prop_path(clouds.RELIEF_KM_PROP)}
    for axis in range(3):
        atmo.driver_remove("scale", axis)
        drivers.drive(atmo, "scale", earth_obj, variables,
                      f"({r} + {clouds.shell_altitude_expr()} + relief + thick) / {r} * {SHELL_MESH_MARGIN:.5f}", axis)


def _swap_group(tree, node, new_tree):
    """Point a group node at another tree, keeping its links by socket *name* (Blender keeps
    them by socket identifier, which shifts when the new group has a different socket list)."""
    ins = [(l.from_socket, sock.name) for sock in node.inputs for l in sock.links]
    outs = [(sock.name, l.to_socket) for sock in node.outputs for l in sock.links]
    vals = {sock.name: sock.default_value for sock in node.inputs
            if not sock.is_linked and hasattr(sock, "default_value")}
    node.node_tree = new_tree
    for sock in list(node.inputs) + list(node.outputs):
        for l in list(sock.links):
            tree.links.remove(l)
    for src, name in ins:
        if name in node.inputs:
            tree.links.new(src, node.inputs[name])
    for name, dst in outs:
        if name in node.outputs:
            tree.links.new(node.outputs[name], dst)
    for name, v in vals.items():
        if name in node.inputs and hasattr(node.inputs[name], "default_value"):
            try:
                node.inputs[name].default_value = v
            except (TypeError, ValueError):
                pass


def update_shaders(earth_obj):
    """Rebuild the shared atmosphere / lighting node groups in an existing scene, in place.

    For a scene built by an older PhoneHome: objects, modifiers, images and settings are left
    alone; only the node groups' maths is replaced, the tile materials re-wired and the
    atmosphere mesh's size driver refreshed. The Sky
    tint is kept; the Sun Direction group (with its binding to the Sun lamp) is kept as is.
    """
    old = {}
    keep_sky = None
    for name in (SURFACE_GROUP, ATMO_STEP_GROUP, AIR_SAMPLE_GROUP, AIR_PARAMS_GROUP):
        ng = bpy.data.node_groups.get(name)
        if ng is not None:
            if name == AIR_PARAMS_GROUP and "Sky Colour" in ng.nodes:
                keep_sky = tuple(ng.nodes["Sky Colour"].outputs[0].default_value)
            ng.name = name + " (old)"
            old[name] = ng
    new = {AIR_PARAMS_GROUP: build_params_group(earth_obj), AIR_SAMPLE_GROUP: build_air_sample_group(earth_obj),
           ATMO_STEP_GROUP: build_step_group(earth_obj), SURFACE_GROUP: build_surface_group(earth_obj)}
    if keep_sky is not None:
        new[AIR_PARAMS_GROUP].nodes["Sky Colour"].outputs[0].default_value = keep_sky
    trees = [m.node_tree for m in bpy.data.materials if m.node_tree] + list(bpy.data.node_groups)
    for tree in trees:
        for node in tree.nodes:
            if node.type == 'GROUP' and node.node_tree is not None:
                for name, ng in old.items():
                    if node.node_tree is ng:
                        _swap_group(tree, node, new[name])
    for ng in old.values():
        bpy.data.node_groups.remove(ng)
    surfaces = list(earth_obj.data.materials)
    clouds_obj = bpy.data.objects.get(config.CLOUDS_NAME)
    if clouds_obj is not None:
        surfaces += list(clouds_obj.data.materials)
    light_surfaces(earth_obj, surfaces)
    atmo = bpy.data.objects.get(config.ATMOSPHERE_NAME)
    if atmo is not None:
        drive_shell_scale(atmo, earth_obj)
    bind_sun(earth_obj, earth_obj.get(SUN_PROP))


# ---------------------------------------------------------------- atmosphere material

def build_material(earth_obj):
    mat = bpy.data.materials.new(config.ATMOSPHERE_NAME)
    mat.surface_render_method = 'BLENDED'
    mat.use_backface_culling = True
    tree = mat.node_tree
    nodes, links = tree.nodes, tree.links
    nodes.remove(nodes["Principled BSDF"])
    out = nodes["Material Output"]
    g = _G(tree)

    tc = nodes.new("ShaderNodeTexCoord")
    tc.object = earth_obj
    geo = nodes.new("ShaderNodeNewGeometry")
    sun = nodes.new("ShaderNodeGroup")
    sun.node_tree = build_sun_group(earth_obj)
    prm = nodes.new("ShaderNodeGroup")
    prm.node_tree = build_params_group(earth_obj)
    for i, n in enumerate((tc, geo, sun, prm)):
        n.location = (-600, 300 - i * 250)
    brightness = nodes.new("ShaderNodeValue")
    brightness.label = "atmosphere_brightness"
    brightness.location = (-600, -750)
    _drive(brightness.outputs[0], "default_value", earth_obj, drivers.prop_path("atmosphere_brightness"))
    forward = nodes.new("ShaderNodeValue")
    forward.label = "atmosphere_forward_scatter"
    forward.location = (-600, -850)
    _drive(forward.outputs[0], "default_value", earth_obj, drivers.prop_path("atmosphere_forward_scatter"))

    # --- the view ray in Earth space
    P = tc.outputs["Object"]                                    # km from Earth centre
    d_world = g.vscale(geo.outputs["Incoming"], -1.0)           # Incoming points at the camera
    D = g.to_object(d_world)
    L_world = sun.outputs["Direction"]
    L = g.to_object(L_world)
    R, Rs = prm.outputs["Ground Radius"], prm.outputs["Shell Radius"]

    pd = g.vmath('DOT_PRODUCT', P, D)
    b2 = g.math('SUBTRACT', g.vmath('DOT_PRODUCT', P, P), g.math('MULTIPLY', pd, pd))
    t_mid = g.math('MULTIPLY', pd, -1.0)
    half = g.math('SQRT', g.math('MAXIMUM', g.math('SUBTRACT', g.math('MULTIPLY', Rs, Rs), b2), 0.0))
    t0 = g.math('SUBTRACT', t_mid, half)
    t_out = g.math('ADD', t_mid, half)
    t_ground = g.math('SUBTRACT', t_mid, g.math('SQRT', g.math('MAXIMUM',
                      g.math('SUBTRACT', g.math('MULTIPLY', R, R), b2), 0.0)))
    hits = g.math('LESS_THAN', b2, g.math('MULTIPLY', R, R))
    t1 = g.math('ADD', g.math('MULTIPLY', hits, t_ground), g.math('MULTIPLY', g.math('SUBTRACT', 1.0, hits), t_out))
    ds = g.math('DIVIDE', g.math('MAXIMUM', g.math('SUBTRACT', t1, t0), 0.0), float(SAMPLES))

    # --- march
    step = build_step_group(earth_obj)
    column, light = 0.0, None
    for i in range(SAMPLES):
        t = g.math('MULTIPLY_ADD', ds, i + 0.5, t0, y=-150)
        s = g.group(step, {"P": P, "D": D, "Sun": L, "t": t, "ds": ds, "Column": column}, y=-150)
        column = s.outputs["Column"]
        light = s.outputs["Light"] if light is None else g.vmath('ADD', light, s.outputs["Light"], y=-150)

    # --- phase: Rayleigh, plus forward scattering towards the sun
    cos_sun = g.vmath('DOT_PRODUCT', d_world, L_world, y=-500)
    rayleigh = g.math('MULTIPLY_ADD', g.math('MULTIPLY', cos_sun, cos_sun, y=-500), 0.75, 0.75, y=-500)
    fwd = g.math('MULTIPLY_ADD', forward.outputs[0],
                 g.math('POWER', g.math('MAXIMUM', cos_sun, 0.0, y=-500), 8.0, y=-500), 1.0, y=-500)
    front = g.math('SUBTRACT', 1.0, geo.outputs["Backfacing"], y=-500)
    strength = g.math('MULTIPLY', sun.outputs["Strength"], brightness.outputs[0], y=-500)
    strength = g.math('MULTIPLY', g.math('MULTIPLY', strength, g.math('MULTIPLY', rayleigh, fwd, y=-500),
                                         y=-500), front, y=-500)

    emit = nodes.new("ShaderNodeEmission")
    emit.location = (g.x + 200, -200)
    # Light is k * beta weighted; / beta(blue) puts the bluest channel at the old glow's scale.
    links.new(g.vmath('MULTIPLY', g.vscale(light, 1.0 / BETA[2]), prm.outputs["Sky Colour"]),
              emit.inputs["Color"])
    links.new(strength, emit.inputs["Strength"])

    # What's behind (ground, clouds) is dimmed and reddened by the air in front of it.
    t_total = g.exp_neg(g.vscale(prm.outputs["View Extinction"], column))
    behind = nodes.new("ShaderNodeMix")
    behind.data_type = 'RGBA'
    behind.location = (g.x + 200, 150)
    links.new(geo.outputs["Backfacing"], behind.inputs["Factor"])  # far side of the shell: pass through
    links.new(t_total, behind.inputs["A"])
    behind.inputs["B"].default_value = (1.0, 1.0, 1.0, 1.0)
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (g.x + 400, 100)
    links.new(behind.outputs["Result"], transparent.inputs["Color"])
    add = nodes.new("ShaderNodeAddShader")
    add.location = (g.x + 600, 0)
    links.new(transparent.outputs[0], add.inputs[0])
    links.new(emit.outputs[0], add.inputs[1])
    links.new(add.outputs[0], out.inputs["Surface"])
    out.location = (g.x + 800, 0)
    return mat


# ---------------------------------------------------------------- object

def build(earth_obj, sun, settings=None):
    """Atmosphere shell above the cloud shell, lit by `sun`. `settings` overrides DEFAULTS.

    Also routes the Earth's (and clouds') materials through Surface Light, so the ground
    and clouds get the air's reddened sunlight and twilight skylight.
    """
    settings = settings or {}
    earth_obj[SUN_PROP] = sun
    for name, (default, soft_max, desc) in DEFAULTS.items():
        drivers.set_prop(earth_obj, name, settings.get(name, default), desc, soft_max=soft_max)
    # The atmosphere sits above the tallest displaced cloud too.
    if clouds.RELIEF_KM_PROP not in earth_obj:
        earth_obj[clouds.RELIEF_KM_PROP] = 0.0

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
    # Low levels are enough: the glow comes from the ideal sphere, and the mesh only has to
    # cover it (SHELL_MESH_MARGIN).
    earth.add_sphere_modifiers(atmo, 1, 2)

    drive_shell_scale(atmo, earth_obj)

    surfaces = list(earth_obj.data.materials)
    clouds_obj = bpy.data.objects.get(config.CLOUDS_NAME)
    if clouds_obj is not None:
        surfaces += list(clouds_obj.data.materials)
    light_surfaces(earth_obj, surfaces)
    bind_sun(earth_obj, sun)
    return atmo
