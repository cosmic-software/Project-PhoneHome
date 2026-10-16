"""The Sun: a sun lamp placed from a UTC date and time.

subsolar(t) gives the point where the Sun is overhead and the Earth-Sun distance, from
the NOAA solar position algorithm (Meeus, Astronomical Algorithms, ch. 25; ~0.01 deg):

    T       = Julian centuries since J2000.0
    L0, M   = Sun's geometric mean longitude and mean anomaly
    C       = equation of centre;  true longitude = L0 + C,  true anomaly = M + C
    R       = 1.000001018 (1 - e^2) / (1 + e cos(true anomaly))     (AU)
    lambda  = apparent longitude (nutation + aberration)
    epsilon = obliquity of the ecliptic (corrected)
    declination  = asin(sin epsilon sin lambda)                       -> sub-solar latitude
    right ascension, minus Greenwich mean sidereal time               -> sub-solar longitude

The Earth is fixed in PhoneHome (+X through lat 0 / lon 0, +Z north), so the Sun moves:
the lamp points from the sub-solar direction towards the Earth, with
  - energy  = sun_strength / R^2 (the 3.4 % perihelion/aphelion swing; a driver, so
              changing Earth["sun_strength"] applies at once)
  - angle   = 0.533 deg, the Sun's disc seen from Earth (soft shadow edges)
  - colour  = 5900 K, sunlight above the atmosphere (the air's reddening is in the shaders)

The time lives in four registered properties on the Earth (year, month, day, UTC hour)
whose update moves the Sun, so scrubbing the hour in the panel sweeps the terminator.
The atmosphere, ground and clouds read the lamp through the shared Sun Direction group,
which atmosphere.bind_sun() points at whichever lamp is the scene's Sun.
"""

import datetime
import math

import bpy
from mathutils import Vector

from . import config, drivers

STRENGTH_PROP = "sun_strength"          # lamp energy at 1 AU
DISTANCE_PROP = "sun_distance_au"       # written by place(): Earth-Sun distance
SUBSOLAR_PROPS = ("sun_subsolar_lat", "sun_subsolar_lon")
TIME_PROPS = ("phonehome_sun_year", "phonehome_sun_month", "phonehome_sun_day", "phonehome_sun_hour")
ANGLE_DEG = 0.533
TEMPERATURE_K = 5900.0
SHADOW_RES_KM = 0.2


# ---------------------------------------------------------------- ephemeris

def julian_day(t):
    """Julian day of an aware or naive-UTC datetime."""
    if t.tzinfo is not None:
        t = t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return 2440587.5 + (t - datetime.datetime(1970, 1, 1)).total_seconds() / 86400.0


def subsolar(t):
    """(latitude deg, longitude deg east, distance AU) of the Sun at UTC datetime t."""
    jd = julian_day(t)
    T = (jd - 2451545.0) / 36525.0
    rad = math.radians
    L0 = (280.46646 + T * (36000.76983 + 0.0003032 * T)) % 360.0
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    C = (math.sin(rad(M)) * (1.914602 - T * (0.004817 + 0.000014 * T))
         + math.sin(rad(2 * M)) * (0.019993 - 0.000101 * T)
         + math.sin(rad(3 * M)) * 0.000289)
    true_long = L0 + C
    nu = M + C
    R = 1.000001018 * (1 - e * e) / (1 + e * math.cos(rad(nu)))
    omega = 125.04 - 1934.136 * T
    lam = true_long - 0.00569 - 0.00478 * math.sin(rad(omega))
    eps0 = 23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * math.cos(rad(omega))
    dec = math.degrees(math.asin(math.sin(rad(eps)) * math.sin(rad(lam))))
    ra = math.degrees(math.atan2(math.cos(rad(eps)) * math.sin(rad(lam)), math.cos(rad(lam))))
    d = jd - 2451545.0
    gmst = (280.46061837 + 360.98564736629 * d + T * T * (0.000387933 - T / 38710000.0)) % 360.0
    lon = (ra - gmst + 180.0) % 360.0 - 180.0
    return dec, lon, R


def unit(lat_deg, lon_deg):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return Vector((math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)))


# ---------------------------------------------------------------- lamp

def find(earth_obj):
    """The scene's Sun: Earth["atmosphere_sun"] if it's a sun lamp in the scene, else any sun lamp."""
    from . import atmosphere
    objs = bpy.context.scene.objects
    s = earth_obj.get(atmosphere.SUN_PROP)
    if s is not None and s.name in objs and s.type == 'LIGHT' and s.data.type == 'SUN':
        return s
    for o in objs:
        if o.type == 'LIGHT' and o.data.type == 'SUN':
            return o
    return None


def ensure(earth_obj):
    """Find the scene's sun lamp or create one, set up as the real Sun."""
    s = find(earth_obj)
    if s is None:
        data = bpy.data.lights.new(config.SUN_NAME, 'SUN')
        s = bpy.data.objects.new(config.SUN_NAME, data)
        bpy.context.scene.collection.objects.link(s)
    data = s.data
    # EEVEE's default is 0.001 units = 1 m at our km scale: it overflows the shadow pool
    # ("Shadow buffer full") and drops shadows. 200 m is finer than a render pixel here.
    data.shadow_maximum_resolution = SHADOW_RES_KM
    data.angle = math.radians(ANGLE_DEG)
    if hasattr(data, "use_temperature"):        # Blender 4.5+
        data.use_temperature = True
        data.temperature = TEMPERATURE_K
    if STRENGTH_PROP not in earth_obj:
        drivers.set_prop(earth_obj, STRENGTH_PROP, 4.0, "Sun lamp strength at 1 AU (the lamp gets "
                         "this / distance^2)", soft_max=20.0)
    if DISTANCE_PROP not in earth_obj:
        earth_obj[DISTANCE_PROP] = 1.0
    if data.animation_data is None or not any(d.data_path == "energy" for d in data.animation_data.drivers):
        drv = data.driver_add("energy").driver
        drv.type = 'SCRIPTED'
        for name, prop in (("s", STRENGTH_PROP), ("d", DISTANCE_PROP)):
            v = drv.variables.new()
            v.name = name
            v.targets[0].id_type = 'OBJECT'
            v.targets[0].id = earth_obj
            v.targets[0].data_path = drivers.prop_path(prop)
        drv.expression = "s / max(d * d, 0.5)"
    return s


def point(sun_obj, lat_deg, lon_deg):
    """Aim the lamp so it shines from the sub-solar direction onto the Earth."""
    sun_obj.rotation_mode = 'XYZ'
    sun_obj.rotation_euler = (-unit(lat_deg, lon_deg)).to_track_quat('-Z', 'Y').to_euler()


def picked_time(earth_obj):
    y, mo, d, h = (getattr(earth_obj, p) for p in TIME_PROPS)
    h = min(max(h, 0.0), 24.0)
    return datetime.datetime(y, mo, d, tzinfo=datetime.timezone.utc) + datetime.timedelta(hours=h)


def place(earth_obj, when=None, lat_lon=None):
    """Put the Sun where it is at `when` (UTC; default: the Earth's time properties), or at
    an explicit sub-solar (lat, lon). Binds the atmosphere to it. Returns the lamp."""
    from . import atmosphere
    register_props()
    s = ensure(earth_obj)
    if lat_lon is not None:
        lat, lon = lat_lon
        dist = 1.0
    else:
        if when is None:
            when = picked_time(earth_obj)
        else:
            set_time(earth_obj, when, move=False)
        lat, lon, dist = subsolar(when)
    point(s, lat, lon)
    earth_obj[SUBSOLAR_PROPS[0]], earth_obj[SUBSOLAR_PROPS[1]] = lat, lon
    earth_obj[DISTANCE_PROP] = dist
    atmosphere.bind_sun(earth_obj, s)
    return s


def set_time(earth_obj, when, move=True):
    """Write a UTC datetime into the Earth's time properties (moves the Sun if move)."""
    register_props()
    when = when.astimezone(datetime.timezone.utc) if when.tzinfo else when
    global _suspend
    _suspend = True
    try:
        earth_obj.phonehome_sun_year, earth_obj.phonehome_sun_month = when.year, when.month
        earth_obj.phonehome_sun_day = when.day
        earth_obj.phonehome_sun_hour = when.hour + when.minute / 60 + when.second / 3600
    finally:
        _suspend = False
    if move:
        place(earth_obj)


# ---------------------------------------------------------------- registered time properties

_suspend = False


def _time_changed(self, context):
    if _suspend or self.name != config.EARTH_NAME:
        return
    try:
        place(self)
    except ValueError:       # e.g. 31 February while the boxes are being edited
        pass


def register_props():
    """Year / month / day / UTC hour on the Earth. Idempotent; used by the build and the panel."""
    if hasattr(bpy.types.Object, TIME_PROPS[0]):
        return
    O, P = bpy.types.Object, bpy.props
    O.phonehome_sun_year = P.IntProperty(name="Year", default=2026, min=1950, max=2100, update=_time_changed)
    O.phonehome_sun_month = P.IntProperty(name="Month", default=1, min=1, max=12, update=_time_changed)
    O.phonehome_sun_day = P.IntProperty(name="Day", default=1, min=1, max=31, update=_time_changed)
    O.phonehome_sun_hour = P.FloatProperty(name="Hour (UTC)", default=12.0, min=0.0, max=24.0,
                                           step=25, precision=2, update=_time_changed,
                                           description="Time of day, UTC hours (12.5 = 12:30)")


def unregister_props():
    for p in TIME_PROPS:
        if hasattr(bpy.types.Object, p):
            delattr(bpy.types.Object, p)
