"""PhoneHome: a headless Blender pipeline that builds and renders a data-driven Earth.

Three spheres, all generated from code -- no .blend file is part of the project:

  Earth       36 x 18 sphere, one NASA tile per 10-degree face (GIBS Blue Marble colour
              + Visible Earth topography displacement)
  Clouds      a slightly larger shell with that day's VIIRS cloud cover
  Atmosphere  an outer shell with a sun-lit limb glow (Chapman air-column model)

Run inside Blender, headless:

    blender --background --factory-startup --python run_phonehome.py -- [options]

See README.md for the options.
"""

__version__ = "0.1.0"
