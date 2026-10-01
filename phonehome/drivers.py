"""Blender driver helpers. Every control lives as a custom property on the Earth object."""


def drive(target, prop, earth, variables, expression, index=-1):
    """Add a scripted driver on target.prop reading Earth paths.

    `variables` maps driver variable name -> RNA path on the Earth object, e.g.
    {"exag": '["terrain_exaggeration"]'}. Expressions are plain arithmetic, so they run
    without Python auto-exec.
    """
    drv = target.driver_add(prop, index).driver
    drv.type = 'SCRIPTED'
    for name, path in variables.items():
        var = drv.variables.new()
        var.name = name
        var.type = 'SINGLE_PROP'
        var.targets[0].id_type = 'OBJECT'
        var.targets[0].id = earth
        var.targets[0].data_path = path
    drv.expression = expression
    return drv


def prop_path(name):
    return f'["{name}"]'


def set_prop(obj, name, value, description, lo=0.0, soft_max=None):
    obj[name] = value
    ui = {"min": lo, "description": description}
    if soft_max is not None:
        ui["soft_max"] = soft_max
    obj.id_properties_ui(name).update(**ui)
