"""The PhoneHome control panel (Object Properties, with the Earth selected).

Registered by a small loader text block embedded in the project .blend (see LOADER),
which finds this project folder next to the .blend. All the code lives here in the
package; the .blend only carries the loader.

Panels:
  PhoneHome Sun         UTC date + hour (moves the Sun live), Now, Place Sun (re-creates a
                        deleted Sun), sub-solar point / distance read-out, strength at 1 AU
  PhoneHome Earth       terrain exaggeration; oceans: bathymetric map <-> water shader slider,
                        water colour, water roughness; night lights strength + floor
  PhoneHome Clouds      date boxes + Load Clouds, cached (offline) dates, offline switch,
                        cloud brightness, relief mode dropdown (Bump / Displacement)
                        + strength/billow/height,
                        thin-cloud boost + Cloud Ramp, show/hide
  PhoneHome Atmosphere  sun picker, density/brightness/thickness/scale height/forward
                        scatter/sun extinction/twilight, Sky Colour tint, show/hide,
                        Update Shaders (new shader code into an existing scene, in place)
"""

import datetime
import os
import threading
import time

import bpy

from . import atmosphere, clouds, config, earth, net, sun, tileset

LOADER_NAME = "phonehome_ui.py"
LOADER = '''"""PhoneHome control panel loader -- runs when this .blend opens (click Allow Execution).

The panel code lives in the PhoneHome project (phonehome/ui.py). This finds the project
folder that contains this .blend and registers it.
"""
import os
import sys

import bpy


def _find_project():
    d = os.path.dirname(bpy.data.filepath)
    for _ in range(4):
        if os.path.isfile(os.path.join(d, "phonehome", "ui.py")):
            return d
        d = os.path.dirname(d)
    return None


root = _find_project()
if root is None:
    print("PhoneHome: project folder not found around this .blend -- control panel not loaded")
else:
    if root not in sys.path:
        sys.path.insert(0, root)
    import phonehome.ui
    phonehome.ui.register()
'''


def embed_loader():
    txt = bpy.data.texts.get(LOADER_NAME) or bpy.data.texts.new(LOADER_NAME)
    txt.from_string(LOADER)
    txt.use_module = True


def _earth(context=None):
    obj = (context.object if context else None) or bpy.data.objects.get(config.EARTH_NAME)
    return obj if obj and config.EXAGGERATION_PROP in obj else None


# Listing cached cloud dates touches the disk; panels redraw constantly, so cache briefly.
_dates_cache = (0.0, [])


def _cached_dates():
    global _dates_cache
    if time.time() - _dates_cache[0] > 5:
        _dates_cache = (time.time(), clouds.cached_dates())
    return _dates_cache[1]


def _offline_update(self, context):
    net.OFFLINE = self.phonehome_offline


# ---------------------------------------------------------------- keep the shaders on the Sun

def _follow_sun(scene, depsgraph=None):
    """After any change: if the Sun lamp was replaced, deleted or re-picked in the panel,
    point the Sun Direction drivers at the scene's current sun lamp (or at none)."""
    earth_obj = bpy.data.objects.get(config.EARTH_NAME)
    if earth_obj is None or bpy.data.node_groups.get(atmosphere.SUN_DIR_GROUP) is None:
        return
    lamp = sun.find(earth_obj)
    if lamp is not atmosphere.bound_sun() or earth_obj.get(atmosphere.SUN_PROP) is not lamp:
        atmosphere.bind_sun(earth_obj, lamp)


# ---------------------------------------------------------------- operators

class PHONEHOME_OT_place_sun(bpy.types.Operator):
    """Put the Sun where it is at the date and UTC hour above (creates the Sun lamp if it's gone)"""
    bl_idname = "phonehome.place_sun"
    bl_label = "Place Sun"

    def execute(self, context):
        earth_obj = bpy.data.objects.get(config.EARTH_NAME)
        try:
            sun.place(earth_obj)
        except ValueError as e:
            self.report({'ERROR'}, f"Invalid date: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


class PHONEHOME_OT_update_shaders(bpy.types.Operator):
    """Replace this scene's atmosphere and lighting shader maths with the project's current code.
Objects, modifiers, images and settings are kept"""
    bl_idname = "phonehome.update_shaders"
    bl_label = "Update Shaders"

    def execute(self, context):
        earth_obj = bpy.data.objects.get(config.EARTH_NAME)
        if earth_obj is None or bpy.data.objects.get(config.ATMOSPHERE_NAME) is None:
            self.report({'ERROR'}, "No PhoneHome atmosphere in this scene")
            return {'CANCELLED'}
        atmosphere.update_shaders(earth_obj)
        self.report({'INFO'}, "Shaders updated")
        return {'FINISHED'}


class PHONEHOME_OT_sun_now(bpy.types.Operator):
    """Set the date and hour to now (UTC) and move the Sun there"""
    bl_idname = "phonehome.sun_now"
    bl_label = "Now"

    def execute(self, context):
        sun.set_time(bpy.data.objects.get(config.EARTH_NAME), datetime.datetime.now(datetime.timezone.utc))
        return {'FINISHED'}


class PHONEHOME_OT_load_clouds(bpy.types.Operator):
    """Fetch that day's VIIRS clouds (or load them from the tileset) onto the cloud shell"""
    bl_idname = "phonehome.load_clouds"
    bl_label = "Load Clouds"

    def execute(self, context):
        earth_obj = bpy.data.objects.get(config.EARTH_NAME)
        self._clouds = bpy.data.objects.get(config.CLOUDS_NAME)
        if earth_obj is None or self._clouds is None:
            self.report({'ERROR'}, "Earth / Clouds objects not found")
            return {'CANCELLED'}
        try:
            self._date = clouds.picked_date(earth_obj)
        except ValueError as e:
            self.report({'ERROR'}, f"Invalid date: {e}")
            return {'CANCELLED'}
        if self._date >= datetime.datetime.now(datetime.timezone.utc).date():
            self.report({'ERROR'}, "Pick a date before today (UTC) -- today's imagery isn't complete yet")
            return {'CANCELLED'}
        net.OFFLINE = context.scene.phonehome_offline
        if net.OFFLINE and self._date not in _cached_dates():
            self.report({'ERROR'}, f"Offline: no clouds for {self._date} in the tileset")
            return {'CANCELLED'}

        self._state = {"done": 0, "total": 0, "paths": None, "error": None}

        def work():
            try:
                self._state["paths"] = clouds.prepare(
                    self._date, lambda d, t: self._state.update(done=d, total=t))
            except Exception as e:  # reported back on the main thread
                self._state["error"] = str(e)

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        s = self._state
        if s["total"]:
            context.workspace.status_text_set(f"Clouds {self._date}: {s['done']}/{s['total']} tiles")
        if self._thread.is_alive():
            return {'PASS_THROUGH'}

        context.window_manager.event_timer_remove(self._timer)
        context.workspace.status_text_set(None)
        if s["error"]:
            self.report({'ERROR'}, s["error"])
            return {'CANCELLED'}
        clouds.apply(bpy.data.objects[config.EARTH_NAME], self._clouds, self._date, s["paths"])
        global _dates_cache
        _dates_cache = (0.0, [])
        self.report({'INFO'}, f"Clouds loaded for {self._date}")
        return {'FINISHED'}


# ---------------------------------------------------------------- panels

class _EarthPanel:
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return _earth(context) is not None


def _visibility(layout, name, label):
    obj = bpy.data.objects.get(name)
    if obj:
        row = layout.row(align=True)
        row.prop(obj, "hide_viewport", text=f"Hide {label} (viewport)")
        row.prop(obj, "hide_render", text="(render)")


class PHONEHOME_PT_sun(_EarthPanel, bpy.types.Panel):
    bl_label = "PhoneHome Sun"

    def draw(self, context):
        obj, layout = _earth(context), self.layout
        if not hasattr(obj, sun.TIME_PROPS[0]):
            layout.label(text="Reload the panel (run phonehome_ui.py)")
            return
        row = layout.row(align=True)
        for p in sun.TIME_PROPS[:3]:
            row.prop(obj, p)
        row = layout.row(align=True)
        row.prop(obj, sun.TIME_PROPS[3], slider=True)
        row.operator(PHONEHOME_OT_sun_now.bl_idname, text="Now")
        lamp = sun.find(obj)
        if lamp is None:
            layout.label(text="No Sun lamp in the scene", icon='ERROR')
        layout.operator(PHONEHOME_OT_place_sun.bl_idname, icon='LIGHT_SUN')
        if sun.SUBSOLAR_PROPS[0] in obj:
            lat, lon = obj[sun.SUBSOLAR_PROPS[0]], obj[sun.SUBSOLAR_PROPS[1]]
            layout.label(text=f"Overhead at {abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, "
                              f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'};  "
                              f"{obj.get(sun.DISTANCE_PROP, 1.0):.4f} AU")
        if sun.STRENGTH_PROP in obj:
            layout.prop(obj, f'["{sun.STRENGTH_PROP}"]', text="Strength at 1 AU")


class PHONEHOME_PT_terrain(_EarthPanel, bpy.types.Panel):
    bl_label = "PhoneHome Earth"

    def draw(self, context):
        obj, layout = _earth(context), self.layout
        layout.prop(obj, f'["{config.EXAGGERATION_PROP}"]', text="Terrain Exaggeration")
        if earth.OCEAN_WATER_PROP not in obj:
            return
        box = layout.box()
        box.label(text="Oceans: 0 = bathymetric map, 1 = water shader")
        box.prop(obj, f'["{earth.OCEAN_WATER_PROP}"]', text="Water Shader", slider=True)
        ng = bpy.data.node_groups.get(earth.OCEAN_GROUP)
        if ng and "Ocean Colour" in ng.nodes:
            box.prop(ng.nodes["Ocean Colour"].outputs[0], "default_value", text="Water Colour")
        box.prop(obj, f'["{earth.OCEAN_ROUGHNESS_PROP}"]', text="Water Roughness (glint)")
        if earth.NIGHT_PROP in obj:
            box = layout.box()
            box.label(text="Night lights (NASA Black Marble 2016)")
            row = box.row(align=True)
            row.prop(obj, f'["{earth.NIGHT_PROP}"]', text="Strength")
            row.prop(obj, f'["{earth.NIGHT_FLOOR_PROP}"]', text="Floor")


class PHONEHOME_PT_clouds(_EarthPanel, bpy.types.Panel):
    bl_label = "PhoneHome Clouds"

    def draw(self, context):
        obj, layout = _earth(context), self.layout
        if clouds.DATE_PICK_PROPS[0] not in obj:
            layout.label(text="No cloud shell in this scene")
            return
        col = layout.column(align=True)
        for prop, label in zip(clouds.DATE_PICK_PROPS, ("Year", "Month", "Day")):
            col.prop(obj, f'["{prop}"]', text=label)
        layout.operator(PHONEHOME_OT_load_clouds.bl_idname, icon='IMPORT')
        layout.label(text=f"Showing: {obj.get(clouds.DATE_PROP, 'none')}")

        dates = _cached_dates()
        layout.prop(context.scene, "phonehome_offline")
        layout.label(text=f"Available offline: {len(dates)} date(s)", icon='DISK_DRIVE')
        for d in dates[-6:]:
            layout.label(text=f"    {d.isoformat()}")

        if clouds.BRIGHTNESS_PROP in obj:
            layout.prop(obj, f'["{clouds.BRIGHTNESS_PROP}"]', text="Cloud Brightness", slider=True)

        relief = layout.box()
        relief.label(text="Cloud relief (shading + shadows)")
        relief.prop(obj, clouds.RELIEF_PROP, text="Mode")
        mode = getattr(obj, clouds.RELIEF_PROP)
        if clouds.BUMP_PROP in obj:
            relief.prop(obj, f'["{clouds.BUMP_PROP}"]', text="Bump Strength", slider=True)
        if clouds.BILLOW_PROP in obj:
            relief.prop(obj, f'["{clouds.BILLOW_PROP}"]', text="Billow (texture)", slider=True)
        if mode == "DISPLACEMENT":
            relief.prop(obj, f'["{clouds.RELIEF_KM_PROP}"]', text="Cloud Height (km)")
            relief.label(text="~16x cloud geometry: slower renders, more memory", icon='INFO')

        box = layout.box()
        box.label(text="Cloud density (all tiles)")
        if clouds.HAZE_OCEAN_PROP in obj:
            row = box.row(align=True)
            row.prop(obj, f'["{clouds.HAZE_OCEAN_PROP}"]', text="Haze Clamp Ocean", slider=True)
            row.prop(obj, f'["{clouds.HAZE_LAND_PROP}"]', text="Land", slider=True)
        box.prop(obj, f'["{clouds.BOOST_PROP}"]', text="Thin Cloud Boost (log)")
        ng = bpy.data.node_groups.get(clouds.DENSITY_GROUP)
        if ng and "Cloud Ramp" in ng.nodes:
            box.template_color_ramp(ng.nodes["Cloud Ramp"], "color_ramp", expand=True)
        _visibility(layout, config.CLOUDS_NAME, "Clouds")


class PHONEHOME_PT_atmosphere(_EarthPanel, bpy.types.Panel):
    bl_label = "PhoneHome Atmosphere"

    def draw(self, context):
        obj, layout = _earth(context), self.layout
        if atmosphere.SUN_PROP not in obj:
            layout.label(text="No atmosphere in this scene")
            return
        col = layout.column()
        col.prop(obj, f'["{atmosphere.SUN_PROP}"]', text="Sun")
        for name in atmosphere.DEFAULTS:
            col.prop(obj, f'["{name}"]', text=name.replace("atmosphere_", "").replace("_", " ").title())
        ng = bpy.data.node_groups.get(atmosphere.AIR_PARAMS_GROUP)
        if ng and "Sky Colour" in ng.nodes:
            box = layout.box()
            box.label(text="Sky tint (sunset colours come from the sunlight's path through the air)")
            box.prop(ng.nodes["Sky Colour"].outputs[0], "default_value", text="")
        _visibility(layout, config.ATMOSPHERE_NAME, "Atmosphere")
        layout.operator(PHONEHOME_OT_update_shaders.bl_idname, icon='FILE_REFRESH')


_classes = (PHONEHOME_OT_load_clouds, PHONEHOME_OT_place_sun, PHONEHOME_OT_sun_now, PHONEHOME_OT_update_shaders,
            PHONEHOME_PT_sun, PHONEHOME_PT_terrain, PHONEHOME_PT_clouds, PHONEHOME_PT_atmosphere)


def register():
    unregister()   # safe to call again (e.g. the loader re-run by hand)
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.phonehome_offline = bpy.props.BoolProperty(
        name="Offline (tileset only)",
        description="Never download: use only tiles already on disk (installed tileset / cache)",
        update=_offline_update)
    clouds.register_props()
    sun.register_props()
    bpy.app.handlers.depsgraph_update_post.append(_follow_sun)


def unregister():
    for c in reversed(_classes):
        if hasattr(bpy.types, c.__name__):
            bpy.utils.unregister_class(getattr(bpy.types, c.__name__))
    if hasattr(bpy.types.Scene, "phonehome_offline"):
        del bpy.types.Scene.phonehome_offline
    if hasattr(bpy.types.Object, clouds.RELIEF_PROP):
        delattr(bpy.types.Object, clouds.RELIEF_PROP)
    sun.unregister_props()
    handlers = bpy.app.handlers.depsgraph_update_post
    for h in [h for h in handlers if getattr(h, "__name__", "") == "_follow_sun"]:
        handlers.remove(h)
