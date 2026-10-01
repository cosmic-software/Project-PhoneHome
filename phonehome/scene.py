"""Scene setup around the Earth: sun, camera, world, render and (debug) save."""

import math

import bpy
from mathutils import Vector

from . import config


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def unit(lat_deg, lon_deg):
    """Earth-fixed unit vector for a latitude/longitude."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return Vector((math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)))


def _look(obj, direction):
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def add_sun(lat_deg, lon_deg, strength=4.0):
    """Sun lamp shining from the given sub-solar lat/lon."""
    data = bpy.data.lights.new(config.SUN_NAME, 'SUN')
    data.energy = strength
    # EEVEE's default is 0.001 units = 1 m at our km scale: it overflows the shadow pool
    # ("Shadow buffer full") and drops shadows. 200 m is finer than a render pixel here.
    data.shadow_maximum_resolution = 0.2
    sun = bpy.data.objects.new(config.SUN_NAME, data)
    bpy.context.scene.collection.objects.link(sun)
    _look(sun, -unit(lat_deg, lon_deg))
    return sun


def add_camera(lat_deg, lon_deg, distance_radii=3.5, lens_mm=50.0):
    """Camera above the given lat/lon, looking at the Earth's centre."""
    dist = config.RADIUS_KM * distance_radii
    data = bpy.data.cameras.new("Camera")
    data.lens = lens_mm
    data.clip_start, data.clip_end = 10.0, dist * 3
    cam = bpy.data.objects.new("Camera", data)
    bpy.context.scene.collection.objects.link(cam)
    eye = unit(lat_deg, lon_deg) * dist
    cam.location = eye
    _look(cam, -eye)
    bpy.context.scene.camera = cam
    return cam


def add_world(colour=(0.02, 0.02, 0.02)):
    world = bpy.data.worlds.new("World")
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (*colour, 1.0)
    bpy.context.scene.world = world
    return world


def render(path, engine='BLENDER_EEVEE', resolution=1024, samples=None):
    scene = bpy.context.scene
    scene.render.engine = engine
    if engine == 'BLENDER_EEVEE':
        scene.eevee.shadow_pool_size = '1024'
    if samples is not None:
        if engine == 'CYCLES':
            scene.cycles.samples = samples
        else:
            scene.eevee.taa_render_samples = samples
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def save_blend(path):
    """Save the generated scene as a .blend with the PhoneHome control panel.

    Image paths are made relative, and the panel loader is embedded, so the file works
    from anywhere inside the project folder (data/ next to it).
    """
    from . import ui
    bpy.context.scene.eevee.shadow_pool_size = '1024'   # same as render(); see add_sun()
    r = config.RADIUS_KM
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    # Default clip_end is 1000 units; at 1 unit = 1 km the Earth would be clipped away.
                    space.clip_start = 1.0
                    space.clip_end = r * 100
                    space.shading.type = 'MATERIAL'
                    space.region_3d.view_location = (0.0, 0.0, 0.0)
                    space.region_3d.view_distance = r * 4
    ui.embed_loader()
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.file.make_paths_relative()
    for img in bpy.data.images:
        img.filepath = img.filepath.replace("\\", "/")   # Blender reads "/" on every OS; "\" only on Windows
    bpy.ops.wm.save_mainfile()
