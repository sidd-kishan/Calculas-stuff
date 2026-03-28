bl_info = {
    "name": "Whip Around Pole Simulator",
    "author": "M365 Copilot",
    "version": (1, 1, 1),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Whip Sim",
    "description": "Chain-link approximation of a whip wrapping around a pole with real-time HUD and normalized energy graph",
    "category": "Physics",
}

import bpy
import blf
import gpu
import math
from mathutils import Vector
from gpu_extras.batch import batch_for_shader
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty

SIM_COLLECTION_NAME = "WhipSim"
HUD_HANDLE = None

SIM_STATE = {
    "prev_positions": {},
    "prev_energy_total": 0.0,
    "initial_ke": 0.0,
    "energy_history": [],
    "energy_norm_history": [],
    "contacted_links": set(),
    "contact_events": [],
    "last_contact": None,
    "last_frame": -1,
    "current_ke": 0.0,
    "cumulative_loss": 0.0,
    "estimated_remaining_ke": 0.0,
    "analytic_theta": 0.0,
    "analytic_energy": 0.0,
    "analytic_tip_speed": 0.0,
    "first_contact_index": 0,
    "link_names": [],
    "constraint_names": [],
    "pole_name": "",
}


def get_props(context=None):
    if context is None:
        context = bpy.context
    return context.scene.whip_sim_props


def ensure_collection(name=SIM_COLLECTION_NAME):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def clear_collection_objects(coll):
    objs = list(coll.objects)
    for obj in objs:
        bpy.data.objects.remove(obj, do_unlink=True)


def safe_select_only(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def link_object_to_collection(obj, coll):
    if obj.name not in coll.objects:
        coll.objects.link(obj)


def ensure_rigidbody_world():
    scene = bpy.context.scene
    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()

    rbw = scene.rigidbody_world
    rbw.enabled = True

    if hasattr(rbw, "substeps_per_frame"):
        rbw.substeps_per_frame = 10
    elif hasattr(rbw, "steps_per_second"):
        rbw.steps_per_second = 240

    if hasattr(rbw, "solver_iterations"):
        rbw.solver_iterations = 40
    elif hasattr(rbw, "num_solver_iterations"):
        rbw.num_solver_iterations = 40

    if hasattr(rbw, "time_scale"):
        rbw.time_scale = 1.0

    if hasattr(rbw, "point_cache") and rbw.point_cache:
        rbw.point_cache.frame_start = scene.frame_start
        rbw.point_cache.frame_end = scene.frame_end

    return rbw


def get_whip_links():
    out = []
    for name in SIM_STATE["link_names"]:
        obj = bpy.data.objects.get(name)
        if obj:
            out.append(obj)
    return out


def get_pole():
    return bpy.data.objects.get(SIM_STATE["pole_name"])


def object_world_bbox(obj):
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def world_bbox_radius_xy(obj):
    pts = object_world_bbox(obj)
    if not pts:
        return 1.0
    return max(math.hypot(p.x, p.y) for p in pts)


def estimated_effective_pole_radius(obj, props):
    if obj is None:
        return max(props.pole_radius, 0.01)
    if props.use_selected_pole and obj.type == 'MESH':
        return max(world_bbox_radius_xy(obj), 0.01)
    return max(props.pole_radius * max(obj.scale.x, obj.scale.y), 0.01)


def link_mass(obj, props):
    if obj and obj.rigid_body:
        return max(obj.rigid_body.mass, 1e-6)
    return max(props.link_mass, 1e-6)


def link_velocity(obj, scene):
    try:
        if obj.rigid_body:
            lv = obj.rigid_body.linear_velocity
            if lv is not None:
                return Vector(lv)
    except Exception:
        pass

    fps = scene.render.fps / scene.render.fps_base
    prev = SIM_STATE["prev_positions"].get(obj.name)
    cur = obj.matrix_world.translation.copy()
    if prev is None:
        SIM_STATE["prev_positions"][obj.name] = cur
        return Vector((0.0, 0.0, 0.0))
    return (cur - prev) * fps


def total_kinetic_energy(scene, props):
    total = 0.0
    for obj in get_whip_links():
        v = link_velocity(obj, scene)
        m = link_mass(obj, props)
        total += 0.5 * m * v.length_squared
    return total


def update_prev_positions():
    for obj in get_whip_links():
        SIM_STATE["prev_positions"][obj.name] = obj.matrix_world.translation.copy()


def collision_margin(props):
    return max(props.link_radius * 0.2, 0.005)


def reset_state():
    SIM_STATE["prev_positions"].clear()
    SIM_STATE["contacted_links"].clear()
    SIM_STATE["contact_events"].clear()
    SIM_STATE["last_contact"] = None
    SIM_STATE["prev_energy_total"] = 0.0
    SIM_STATE["initial_ke"] = 0.0
    SIM_STATE["energy_history"] = []
    SIM_STATE["energy_norm_history"] = []
    SIM_STATE["last_frame"] = -1
    SIM_STATE["current_ke"] = 0.0
    SIM_STATE["cumulative_loss"] = 0.0
    SIM_STATE["estimated_remaining_ke"] = 0.0
    SIM_STATE["analytic_theta"] = 0.0
    SIM_STATE["analytic_energy"] = 0.0
    SIM_STATE["analytic_tip_speed"] = 0.0
    SIM_STATE["first_contact_index"] = 0
    SIM_STATE["link_names"] = []
    SIM_STATE["constraint_names"] = []
    SIM_STATE["pole_name"] = ""


def reset_runtime_metrics_only():
    SIM_STATE["prev_positions"].clear()
    SIM_STATE["contacted_links"].clear()
    SIM_STATE["contact_events"].clear()
    SIM_STATE["last_contact"] = None
    SIM_STATE["prev_energy_total"] = 0.0
    SIM_STATE["initial_ke"] = 0.0
    SIM_STATE["energy_history"] = []
    SIM_STATE["energy_norm_history"] = []
    SIM_STATE["last_frame"] = -1
    SIM_STATE["current_ke"] = 0.0
    SIM_STATE["cumulative_loss"] = 0.0
    SIM_STATE["estimated_remaining_ke"] = 0.0
    SIM_STATE["analytic_theta"] = 0.0
    SIM_STATE["analytic_energy"] = 0.0
    SIM_STATE["analytic_tip_speed"] = 0.0


def analytic_energy_model(props, effective_radius):
    L = props.link_count * props.link_length
    x = props.first_contact_ratio * L
    R = max(effective_radius, 1e-6)
    theta = max((L - x) / R, 0.0)
    m_eff = max(props.link_mass, 1e-6)
    E0 = 0.5 * m_eff * props.launch_speed ** 2
    E = E0 * math.exp(-props.energy_loss_beta * theta)
    v_tip = props.launch_speed * math.exp(-props.energy_loss_beta * theta * 0.5)
    return theta, E, v_tip


def link_contacts_pole(link_obj, pole_obj, props, depsgraph=None):
    if pole_obj is None or link_obj is None:
        return False

    link_center = link_obj.matrix_world.translation
    eff_link_r = max(props.link_radius, 0.001)

    if not props.use_selected_pole:
        local = pole_obj.matrix_world.inverted() @ link_center
        if props.generated_pole_shape == 'CYLINDER':
            radial = math.hypot(local.x, local.y)
            h = props.pole_height * 0.5
            return (radial <= props.pole_radius + eff_link_r + props.contact_epsilon) and (-h <= local.z <= h)
        if props.generated_pole_shape == 'BOX':
            sx = props.pole_scale_x * 0.5 + eff_link_r + props.contact_epsilon
            sy = props.pole_scale_y * 0.5 + eff_link_r + props.contact_epsilon
            sz = props.pole_height * 0.5 + eff_link_r + props.contact_epsilon
            return (abs(local.x) <= sx and abs(local.y) <= sy and abs(local.z) <= sz)
        if props.generated_pole_shape == 'SPHERE':
            return local.length <= props.pole_radius + eff_link_r + props.contact_epsilon

    try:
        eval_obj = pole_obj.evaluated_get(depsgraph) if depsgraph else pole_obj
        inv = eval_obj.matrix_world.inverted()
        local = inv @ link_center
        ok, closest, normal, face_idx = eval_obj.closest_point_on_mesh(local)
        if ok:
            world_closest = eval_obj.matrix_world @ closest
            dist = (link_center - world_closest).length
            return dist <= eff_link_r + props.contact_epsilon
    except Exception:
        pass

    bbox = object_world_bbox(pole_obj)
    if not bbox:
        return False

    minx = min(p.x for p in bbox) - eff_link_r - props.contact_epsilon
    maxx = max(p.x for p in bbox) + eff_link_r + props.contact_epsilon
    miny = min(p.y for p in bbox) - eff_link_r - props.contact_epsilon
    maxy = max(p.y for p in bbox) + eff_link_r + props.contact_epsilon
    minz = min(p.z for p in bbox) - eff_link_r - props.contact_epsilon
    maxz = max(p.z for p in bbox) + eff_link_r + props.contact_epsilon
    p = link_center
    return minx <= p.x <= maxx and miny <= p.y <= maxy and minz <= p.z <= maxz


def add_rigidbody(obj, body_type='ACTIVE'):
    safe_select_only(obj)
    if obj.rigid_body is None:
        bpy.ops.rigidbody.object_add(type=body_type)
    else:
        obj.rigid_body.type = body_type
    return obj


def configure_passive_collision(obj, props):
    rb = obj.rigid_body
    rb.type = 'PASSIVE'
    rb.friction = props.friction
    rb.restitution = props.restitution
    rb.use_margin = True
    rb.collision_margin = props.contact_epsilon

    if props.use_selected_pole:
        shape = props.selected_pole_collision_shape
        rb.collision_shape = shape if shape in {'MESH', 'CONVEX_HULL', 'BOX', 'SPHERE', 'CYLINDER'} else 'MESH'
    else:
        if props.generated_pole_shape == 'CYLINDER':
            rb.collision_shape = 'CYLINDER'
        elif props.generated_pole_shape == 'BOX':
            rb.collision_shape = 'BOX'
        elif props.generated_pole_shape == 'SPHERE':
            rb.collision_shape = 'SPHERE'
        else:
            rb.collision_shape = 'MESH'


def make_generated_pole(coll, props):
    bpy.ops.object.select_all(action='DESELECT')

    if props.generated_pole_shape == 'CYLINDER':
        bpy.ops.mesh.primitive_cylinder_add(radius=props.pole_radius, depth=props.pole_height, location=(0, 0, props.pole_height * 0.5))
    elif props.generated_pole_shape == 'BOX':
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, props.pole_height * 0.5))
        obj = bpy.context.active_object
        obj.scale = Vector((props.pole_scale_x * 0.5, props.pole_scale_y * 0.5, props.pole_height * 0.5))
    elif props.generated_pole_shape == 'SPHERE':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=props.pole_radius, location=(0, 0, props.pole_radius))

    pole = bpy.context.active_object
    pole.name = "WS_Pole"
    link_object_to_collection(pole, coll)
    add_rigidbody(pole, body_type='PASSIVE')
    configure_passive_collision(pole, props)
    SIM_STATE["pole_name"] = pole.name
    return pole


def use_selected_pole_object(coll, props):
    src = bpy.context.active_object
    if src is None:
        raise RuntimeError("No active object selected for pole.")
    if src.type != 'MESH':
        raise RuntimeError("Selected pole object must be a mesh.")

    obj = src.copy()
    if src.data:
        obj.data = src.data.copy()
    obj.name = "WS_Pole_Selected"
    obj.matrix_world = src.matrix_world.copy()
    coll.objects.link(obj)
    add_rigidbody(obj, body_type='PASSIVE')
    configure_passive_collision(obj, props)
    SIM_STATE["pole_name"] = obj.name
    return obj


def create_link_mesh(name, shape, length, radius):
    bpy.ops.object.select_all(action='DESELECT')

    if shape == 'BOX':
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.scale = (length * 0.5, radius, radius)
    elif shape == 'CYLINDER':
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length, location=(0, 0, 0), rotation=(0, math.radians(90), 0))
        obj = bpy.context.active_object
    elif shape == 'ELLIPSOID':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.scale = (length * 0.5, radius, radius)
    elif shape == 'HEX':
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=radius, depth=length, location=(0, 0, 0), rotation=(0, math.radians(90), 0))
        obj = bpy.context.active_object
    elif shape == 'TORUS':
        bpy.ops.mesh.primitive_torus_add(major_radius=max(length * 0.28, radius * 1.5), minor_radius=radius * 0.55, location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.rotation_euler = (math.radians(90), 0, 0)
    else:
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.scale = (length * 0.5, radius, radius)

    obj.name = name
    return obj


def configure_active_link(obj, props):
    add_rigidbody(obj, body_type='ACTIVE')
    rb = obj.rigid_body
    rb.mass = props.link_mass
    rb.friction = props.friction
    rb.restitution = props.restitution
    rb.linear_damping = props.linear_damping
    rb.angular_damping = props.angular_damping
    rb.use_margin = True
    rb.collision_margin = collision_margin(props)

    if props.link_shape == 'BOX':
        rb.collision_shape = 'BOX'
    elif props.link_shape in {'CYLINDER', 'HEX'}:
        rb.collision_shape = 'CYLINDER'
    elif props.link_shape == 'ELLIPSOID':
        rb.collision_shape = 'SPHERE'
    elif props.link_shape == 'TORUS':
        rb.collision_shape = 'CONVEX_HULL'
    else:
        rb.collision_shape = 'BOX'


def create_constraint_between(a, b, idx, props, coll):
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(a.location + b.location) * 0.5)
    c = bpy.context.active_object
    c.name = f"WS_Constraint_{idx:03d}"
    link_object_to_collection(c, coll)

    safe_select_only(c)
    bpy.ops.rigidbody.constraint_add(type='GENERIC_SPRING')
    rbc = c.rigid_body_constraint
    rbc.object1 = a
    rbc.object2 = b
    rbc.type = 'GENERIC_SPRING'

    link_gap = props.link_gap
    rbc.use_limit_lin_x = True
    rbc.use_limit_lin_y = True
    rbc.use_limit_lin_z = True
    rbc.limit_lin_x_lower = -link_gap
    rbc.limit_lin_x_upper = link_gap
    rbc.limit_lin_y_lower = -link_gap
    rbc.limit_lin_y_upper = link_gap
    rbc.limit_lin_z_lower = -link_gap
    rbc.limit_lin_z_upper = link_gap

    rbc.use_limit_ang_x = True
    rbc.use_limit_ang_y = True
    rbc.use_limit_ang_z = True
    ang = math.radians(props.angular_freedom_deg)
    rbc.limit_ang_x_lower = -ang
    rbc.limit_ang_x_upper = ang
    rbc.limit_ang_y_lower = -ang
    rbc.limit_ang_y_upper = ang
    rbc.limit_ang_z_lower = -ang
    rbc.limit_ang_z_upper = ang

    if hasattr(rbc, 'use_spring_ang_x'):
        rbc.use_spring_ang_x = True
        rbc.use_spring_ang_y = True
        rbc.use_spring_ang_z = True
        if hasattr(rbc, 'spring_stiffness_ang_x'):
            rbc.spring_stiffness_ang_x = props.spring_stiffness
            rbc.spring_stiffness_ang_y = props.spring_stiffness
            rbc.spring_stiffness_ang_z = props.spring_stiffness
        if hasattr(rbc, 'spring_damping_ang_x'):
            rbc.spring_damping_ang_x = props.spring_damping
            rbc.spring_damping_ang_y = props.spring_damping
            rbc.spring_damping_ang_z = props.spring_damping

    if hasattr(rbc, 'use_breaking'):
        rbc.use_breaking = props.enable_fragmentation
    if hasattr(rbc, 'breaking_threshold'):
        rbc.breaking_threshold = props.breaking_threshold

    return c


def place_links_for_first_contact(links, pole_radius_eff, props):
    count = len(links)
    if count == 0:
        return 0

    first_idx = max(0, min(count - 1, round(props.first_contact_ratio * (count - 1))))
    SIM_STATE["first_contact_index"] = first_idx

    spacing = props.link_length + props.link_gap
    z0 = props.start_height
    rc = pole_radius_eff + props.link_radius + props.contact_offset
    tangent_angle = math.pi
    tangent_point = Vector((rc * math.cos(tangent_angle), rc * math.sin(tangent_angle), z0))

    for i in range(first_idx, -1, -1):
        k = first_idx - i
        pos = tangent_point + Vector((-spacing * (k + 0.2), 0.0, 0.0))
        links[i].location = pos
        links[i].rotation_euler = (0.0, 0.0, 0.0)

    dtheta = spacing / max(rc, 1e-6)
    pitch_per_turn = max(props.coil_pitch_per_turn, 0.001)

    for i in range(first_idx + 1, count):
        k = i - first_idx
        ang = tangent_angle + k * dtheta
        x = rc * math.cos(ang)
        y = rc * math.sin(ang)
        z = z0 + (pitch_per_turn / (2.0 * math.pi)) * (ang - tangent_angle)
        links[i].location = Vector((x, y, z))

        dx = -rc * math.sin(ang)
        dy = rc * math.cos(ang)
        dz = pitch_per_turn / (2.0 * math.pi)
        tangent_dir = Vector((dx, dy, dz)).normalized()
        yaw = math.atan2(tangent_dir.y, tangent_dir.x)
        links[i].rotation_euler = (0.0, 0.0, yaw)

    return first_idx


def apply_initial_velocities(links, first_idx, pole_obj, props):
    pole_center = pole_obj.matrix_world.translation.copy()
    for i, obj in enumerate(links):
        pos = obj.matrix_world.translation
        radial = pos - pole_center
        radial.z = 0.0
        if radial.length < 1e-6:
            radial = Vector((1, 0, 0))
        radial.normalize()

        tang = Vector((-radial.y, radial.x, 0.0))
        inward = -radial

        if i < first_idx:
            vel = inward * props.launch_speed * 0.6 + tang * props.launch_speed * 0.4
        else:
            vel = tang * props.launch_speed + inward * props.launch_speed * props.inward_bias

        try:
            obj.rigid_body.linear_velocity = vel
        except Exception:
            pass

        try:
            obj.rigid_body.angular_velocity = Vector((0.0, 0.0, props.angular_launch_speed))
        except Exception:
            pass


def draw_text(x, y, text, size=14, color=(1, 1, 1, 1)):
    font_id = 0
    blf.position(font_id, x, y, 0)
    blf.size(font_id, size)
    blf.color(font_id, *color)
    blf.draw(font_id, text)


def draw_polyline(points, color=(0.2, 0.9, 0.3, 1.0), width=2.0):
    if len(points) < 2:
        return
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": points})
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(width)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def draw_rect_lines(x, y, w, h, color=(1, 1, 1, 0.4)):
    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
    draw_polyline(pts, color=color, width=1.0)


def draw_hud():
    props = get_props()
    if not props.show_hud:
        return

    scene = bpy.context.scene
    pole = get_pole()
    pole_radius_eff = estimated_effective_pole_radius(pole, props)

    x0, y0, dy = 30, 740, 20
    draw_text(x0, y0, 'Whip Around Pole Simulator', 18, (1.0, 0.95, 0.6, 1.0))
    y = y0 - 28

    draw_text(x0, y, f'Frame: {scene.frame_current}', 13); y -= dy
    draw_text(x0, y, f'Links: {props.link_count} | Shape: {props.link_shape}', 13); y -= dy
    draw_text(x0, y, f'Pole radius eff R: {pole_radius_eff:.4f}', 13); y -= dy
    draw_text(x0, y, f'First contact ratio x/L: {props.first_contact_ratio:.3f}', 13); y -= dy
    draw_text(x0, y, f'First contact index: {SIM_STATE["first_contact_index"]}', 13); y -= dy

    draw_text(x0, y, 'Analytic model:', 14, (0.8, 1.0, 1.0, 1.0)); y -= dy
    draw_text(x0, y, 'theta = (L - x) / R', 13); y -= dy
    draw_text(x0, y, 'E(theta) = E0 exp(-beta theta)', 13); y -= dy
    draw_text(x0, y, 'v_tip = v0 exp(-beta theta / 2)', 13); y -= dy

    y -= 8
    draw_text(x0, y, f'theta_f: {SIM_STATE["analytic_theta"]:.4f}', 13, (0.75, 1.0, 0.75, 1.0)); y -= dy
    draw_text(x0, y, f'Analytic E_tip: {SIM_STATE["analytic_energy"]:.6f}', 13, (0.75, 1.0, 0.75, 1.0)); y -= dy
    draw_text(x0, y, f'Analytic v_tip: {SIM_STATE["analytic_tip_speed"]:.6f}', 13, (0.75, 1.0, 0.75, 1.0)); y -= dy

    y -= 8
    draw_text(x0, y, f'Current total KE: {SIM_STATE["current_ke"]:.6f}', 13, (1.0, 0.85, 0.85, 1.0)); y -= dy
    ke_norm = SIM_STATE['current_ke'] / max(SIM_STATE['initial_ke'], 1e-9)
    draw_text(x0, y, f'Energy remaining (norm): {ke_norm:.6f}', 13, (0.75, 1.0, 0.85, 1.0)); y -= dy
    draw_text(x0, y, f'Energy lost total: {SIM_STATE["cumulative_loss"]:.6f}', 13, (1.0, 0.75, 0.75, 1.0)); y -= dy
    draw_text(x0, y, f'Contact count: {len(SIM_STATE["contact_events"])}', 13, (1.0, 0.75, 0.75, 1.0)); y -= dy

    if SIM_STATE['last_contact']:
        lc = SIM_STATE['last_contact']
        draw_text(x0, y, f'Last contact: {lc["link"]} | frame={lc["frame"]} | approx dKE={lc["loss"]:.6f}', 13, (1.0, 0.85, 0.5, 1.0))
        y -= dy

    gx, gy = 430, 60
    gw, gh = 520, 180
    draw_text(gx, gy + gh + 10, 'Normalized Energy Remaining in Rope', 14, (0.9, 0.95, 1.0, 1.0))
    draw_rect_lines(gx, gy, gw, gh, color=(1, 1, 1, 0.35))

    hist = SIM_STATE['energy_norm_history'][-props.graph_max_points:]
    if hist:
        vals = [max(min(v, 1.2), 0.0) for _, v in hist]

        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            yy = gy + gh * frac
            draw_polyline([(gx, yy), (gx + gw, yy)], color=(1, 1, 1, 0.12), width=1.0)

        if len(vals) == 1:
            xs = [gx + gw * 0.5]
        else:
            xs = [gx + (gw * i / (len(vals) - 1)) for i in range(len(vals))]

        pts = []
        for i, val in enumerate(vals):
            px = xs[i]
            py = gy + gh * val
            pts.append((px, py))
            d = 3.0
            draw_polyline([(px - d, py), (px + d, py)], color=(1.0, 0.95, 0.3, 1.0), width=2.0)
            draw_polyline([(px, py - d), (px, py + d)], color=(1.0, 0.95, 0.3, 1.0), width=2.0)

        if len(pts) >= 2:
            draw_polyline(pts, color=(0.25, 0.95, 0.45, 1.0), width=2.5)

        draw_text(gx - 28, gy - 4, '0', 10, (0.9, 0.9, 0.9, 1.0))
        draw_text(gx - 36, gy + gh * 0.25 - 4, '0.25', 10, (0.9, 0.9, 0.9, 1.0))
        draw_text(gx - 28, gy + gh * 0.5 - 4, '0.5', 10, (0.9, 0.9, 0.9, 1.0))
        draw_text(gx - 36, gy + gh * 0.75 - 4, '0.75', 10, (0.9, 0.9, 0.9, 1.0))
        draw_text(gx - 28, gy + gh - 4, '1', 10, (0.9, 0.9, 0.9, 1.0))

        if props.graph_show_labels:
            for i, (frame_no, _val) in enumerate(hist[:20]):
                px = xs[i]
                draw_text(px - 8, gy - 18, str(frame_no), 10, (0.9, 0.9, 0.9, 1.0))


def install_hud():
    global HUD_HANDLE
    if HUD_HANDLE is None:
        HUD_HANDLE = bpy.types.SpaceView3D.draw_handler_add(draw_hud, (), 'WINDOW', 'POST_PIXEL')


def remove_hud():
    global HUD_HANDLE
    if HUD_HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(HUD_HANDLE, 'WINDOW')
        HUD_HANDLE = None


def update_contact_graph_and_metrics(scene, depsgraph=None):
    props = scene.whip_sim_props
    if not props.enable_live_monitor:
        return

    if SIM_STATE['last_frame'] == scene.frame_current:
        return
    SIM_STATE['last_frame'] = scene.frame_current

    pole = get_pole()
    if pole is None:
        return

    pole_radius_eff = estimated_effective_pole_radius(pole, props)
    theta, E_analytic, v_tip = analytic_energy_model(props, pole_radius_eff)
    SIM_STATE['analytic_theta'] = theta
    SIM_STATE['analytic_energy'] = E_analytic
    SIM_STATE['analytic_tip_speed'] = v_tip

    current_ke = total_kinetic_energy(scene, props)
    prev_ke = SIM_STATE['prev_energy_total']

    if scene.frame_current <= scene.frame_start or SIM_STATE['initial_ke'] <= 0.0:
        SIM_STATE['initial_ke'] = max(current_ke, 1e-9)

    SIM_STATE['current_ke'] = current_ke
    SIM_STATE['estimated_remaining_ke'] = current_ke
    SIM_STATE['cumulative_loss'] = max(SIM_STATE['initial_ke'] - current_ke, 0.0)

    ke_norm = current_ke / max(SIM_STATE['initial_ke'], 1e-9)
    SIM_STATE['energy_history'].append((scene.frame_current, current_ke))
    SIM_STATE['energy_norm_history'].append((scene.frame_current, ke_norm))

    max_hist = max(props.graph_max_points, 10)
    if len(SIM_STATE['energy_history']) > max_hist:
        SIM_STATE['energy_history'] = SIM_STATE['energy_history'][-max_hist:]
    if len(SIM_STATE['energy_norm_history']) > max_hist:
        SIM_STATE['energy_norm_history'] = SIM_STATE['energy_norm_history'][-max_hist:]

    for obj in get_whip_links():
        touched = link_contacts_pole(obj, pole, props, depsgraph)
        if touched and obj.name not in SIM_STATE['contacted_links']:
            SIM_STATE['contacted_links'].add(obj.name)
            event = {
                'frame': scene.frame_current,
                'link': obj.name,
                'loss': max(prev_ke - current_ke, 0.0),
            }
            SIM_STATE['contact_events'].append(event)
            SIM_STATE['last_contact'] = event

    SIM_STATE['prev_energy_total'] = current_ke
    update_prev_positions()


class WhipSimProps(PropertyGroup):
    use_selected_pole: BoolProperty(name='Use Selected Pole', description='Use a duplicate of the active mesh object as the pole', default=False)

    generated_pole_shape: EnumProperty(
        name='Generated Pole Shape',
        items=[('CYLINDER', 'Cylinder', ''), ('BOX', 'Box', ''), ('SPHERE', 'Sphere', '')],
        default='CYLINDER'
    )

    selected_pole_collision_shape: EnumProperty(
        name='Selected Pole Collision',
        items=[('MESH', 'Mesh', ''), ('CONVEX_HULL', 'Convex Hull', ''), ('BOX', 'Box', ''), ('SPHERE', 'Sphere', ''), ('CYLINDER', 'Cylinder', '')],
        default='MESH'
    )

    pole_radius: FloatProperty(name='Pole Radius', default=0.35, min=0.01)
    pole_height: FloatProperty(name='Pole Height', default=4.0, min=0.05)
    pole_scale_x: FloatProperty(name='Pole Scale X', default=0.7, min=0.01)
    pole_scale_y: FloatProperty(name='Pole Scale Y', default=0.7, min=0.01)

    link_count: IntProperty(name='Link Count', default=28, min=3, max=300)
    link_length: FloatProperty(name='Link Length', default=0.18, min=0.01)
    link_radius: FloatProperty(name='Link Radius', default=0.04, min=0.005)
    link_mass: FloatProperty(name='Link Mass', default=0.05, min=0.0001)
    link_gap: FloatProperty(name='Link Gap', default=0.01, min=0.0)

    link_shape: EnumProperty(
        name='Link Shape',
        items=[('BOX', 'Box', ''), ('CYLINDER', 'Cylinder', ''), ('ELLIPSOID', 'Ellipsoid', ''), ('HEX', 'Hex Prism', ''), ('TORUS', 'Torus', '')],
        default='CYLINDER'
    )

    launch_speed: FloatProperty(name='Launch Speed', default=6.5, min=0.0)
    angular_launch_speed: FloatProperty(name='Angular Launch Speed', default=12.0, min=0.0)
    inward_bias: FloatProperty(name='Inward Bias', default=0.25, min=0.0, max=2.0)

    friction: FloatProperty(name='Friction', default=0.55, min=0.0, max=1.0)
    restitution: FloatProperty(name='Restitution', default=0.05, min=0.0, max=1.0)
    linear_damping: FloatProperty(name='Linear Damping', default=0.04, min=0.0, max=1.0)
    angular_damping: FloatProperty(name='Angular Damping', default=0.1, min=0.0, max=1.0)

    first_contact_ratio: FloatProperty(name='First Contact Ratio x/L', description='0 = base, 1 = tip', default=0.15, min=0.0, max=1.0)
    energy_loss_beta: FloatProperty(name='Energy Loss Beta', description='Analytic loss coefficient in E = E0 exp(-beta theta)', default=0.12, min=0.0, max=10.0)

    coil_pitch_per_turn: FloatProperty(name='Coil Pitch / Turn', description='Vertical rise per full wrap around the pole', default=0.09, min=0.001, max=10.0)
    start_height: FloatProperty(name='Start Height', default=1.2, min=0.0, max=100.0)
    contact_offset: FloatProperty(name='Contact Offset', default=0.02, min=0.0, max=1.0)
    contact_epsilon: FloatProperty(name='Contact Epsilon', default=0.03, min=0.0001, max=1.0)

    spring_stiffness: FloatProperty(name='Spring Stiffness', default=12.0, min=0.0, max=1e6)
    spring_damping: FloatProperty(name='Spring Damping', default=0.7, min=0.0, max=1e6)
    angular_freedom_deg: FloatProperty(name='Angular Freedom (deg)', default=35.0, min=1.0, max=180.0)

    enable_fragmentation: BoolProperty(name='Enable Fragmentation', description='Allow rigid body constraints to break', default=False)
    breaking_threshold: FloatProperty(name='Breaking Threshold', description='Rigid body constraint breaking threshold', default=20.0, min=0.0)

    enable_live_monitor: BoolProperty(name='Enable Live Monitor', default=True)
    show_hud: BoolProperty(name='Show HUD', default=True)
    graph_max_points: IntProperty(name='Graph Max Samples', default=100, min=10, max=1000)
    graph_show_labels: BoolProperty(name='Graph Labels', default=False)
    auto_reset_frame: BoolProperty(name='Reset Timeline on Build', default=True)


class WSIM_OT_build(Operator):
    bl_idname = 'wsim.build'
    bl_label = 'Build / Rebuild Simulator'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = get_props(context)
        scene = context.scene

        remove_handlers()
        remove_hud()
        reset_state()

        coll = ensure_collection()
        clear_collection_objects(coll)
        ensure_rigidbody_world()

        try:
            if props.use_selected_pole:
                pole = use_selected_pole_object(coll, props)
            else:
                pole = make_generated_pole(coll, props)
        except Exception as e:
            self.report({'ERROR'}, f'Pole creation failed: {e}')
            install_hud()
            return {'CANCELLED'}

        pole_radius_eff = estimated_effective_pole_radius(pole, props)

        links = []
        for i in range(props.link_count):
            obj = create_link_mesh(f'WS_Link_{i:03d}', props.link_shape, props.link_length, props.link_radius)
            link_object_to_collection(obj, coll)
            configure_active_link(obj, props)
            links.append(obj)
            SIM_STATE['link_names'].append(obj.name)

        first_idx = place_links_for_first_contact(links, pole_radius_eff, props)

        for i in range(len(links) - 1):
            c = create_constraint_between(links[i], links[i + 1], i, props, coll)
            SIM_STATE['constraint_names'].append(c.name)

        bpy.context.view_layer.update()
        apply_initial_velocities(links, first_idx, pole, props)

        if props.auto_reset_frame:
            scene.frame_set(scene.frame_start)

        update_prev_positions()
        ke0 = total_kinetic_energy(scene, props)
        SIM_STATE['prev_energy_total'] = ke0
        SIM_STATE['initial_ke'] = max(ke0, 1e-9)
        SIM_STATE['current_ke'] = ke0
        SIM_STATE['energy_history'] = [(scene.frame_current, ke0)]
        SIM_STATE['energy_norm_history'] = [(scene.frame_current, 1.0)]

        install_hud()
        install_handlers()
        self.report({'INFO'}, 'Whip simulator built.')
        return {'FINISHED'}


class WSIM_OT_reset(Operator):
    bl_idname = 'wsim.reset'
    bl_label = 'Reset Simulation State'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = get_props(context)
        scene = context.scene
        scene.frame_set(scene.frame_start)
        reset_runtime_metrics_only()
        update_prev_positions()

        ke0 = total_kinetic_energy(scene, props)
        SIM_STATE['prev_energy_total'] = ke0
        SIM_STATE['initial_ke'] = max(ke0, 1e-9)
        SIM_STATE['current_ke'] = ke0
        SIM_STATE['energy_history'] = [(scene.frame_current, ke0)]
        SIM_STATE['energy_norm_history'] = [(scene.frame_current, 1.0)]
        self.report({'INFO'}, 'Simulation metrics reset.')
        return {'FINISHED'}


class WSIM_OT_toggle_hud(Operator):
    bl_idname = 'wsim.toggle_hud'
    bl_label = 'Toggle HUD'

    def execute(self, context):
        props = get_props(context)
        props.show_hud = not props.show_hud
        if props.show_hud:
            install_hud()
        else:
            remove_hud()
        return {'FINISHED'}


class WSIM_OT_clear_all(Operator):
    bl_idname = 'wsim.clear_all'
    bl_label = 'Delete Simulator'

    def execute(self, context):
        remove_handlers()
        remove_hud()
        reset_state()
        coll = bpy.data.collections.get(SIM_COLLECTION_NAME)
        if coll:
            clear_collection_objects(coll)
        self.report({'INFO'}, 'Simulator deleted.')
        return {'FINISHED'}


class WSIM_PT_panel(Panel):
    bl_label = 'Whip Sim'
    bl_idname = 'WSIM_PT_panel'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Whip Sim'

    def draw(self, context):
        layout = self.layout
        props = get_props(context)

        col = layout.column(align=True)
        col.label(text='Pole')
        col.prop(props, 'use_selected_pole')
        if props.use_selected_pole:
            col.prop(props, 'selected_pole_collision_shape')
            col.label(text='Select a mesh and click Build')
        else:
            col.prop(props, 'generated_pole_shape')
            if props.generated_pole_shape in {'CYLINDER', 'SPHERE'}:
                col.prop(props, 'pole_radius')
            if props.generated_pole_shape == 'CYLINDER':
                col.prop(props, 'pole_height')
            elif props.generated_pole_shape == 'BOX':
                col.prop(props, 'pole_scale_x')
                col.prop(props, 'pole_scale_y')
                col.prop(props, 'pole_height')

        box = layout.box()
        box.label(text='Whip / Chain')
        box.prop(props, 'link_count')
        box.prop(props, 'link_shape')
        box.prop(props, 'link_length')
        box.prop(props, 'link_radius')
        box.prop(props, 'link_mass')
        box.prop(props, 'link_gap')

        box = layout.box()
        box.label(text='Launch / Contact')
        box.prop(props, 'first_contact_ratio', slider=True)
        box.prop(props, 'launch_speed')
        box.prop(props, 'angular_launch_speed')
        box.prop(props, 'inward_bias')
        box.prop(props, 'start_height')
        box.prop(props, 'contact_offset')
        box.prop(props, 'contact_epsilon')

        box = layout.box()
        box.label(text='Coil Geometry')
        box.prop(props, 'coil_pitch_per_turn')

        box = layout.box()
        box.label(text='Material / Dynamics')
        box.prop(props, 'friction')
        box.prop(props, 'restitution')
        box.prop(props, 'linear_damping')
        box.prop(props, 'angular_damping')
        box.prop(props, 'spring_stiffness')
        box.prop(props, 'spring_damping')
        box.prop(props, 'angular_freedom_deg')

        box = layout.box()
        box.label(text='Energy Model')
        box.prop(props, 'energy_loss_beta')

        box = layout.box()
        box.label(text='Fragmentation')
        box.prop(props, 'enable_fragmentation')
        sub = box.column()
        sub.enabled = props.enable_fragmentation
        sub.prop(props, 'breaking_threshold')

        box = layout.box()
        box.label(text='HUD / Graph')
        box.prop(props, 'enable_live_monitor')
        box.prop(props, 'show_hud')
        box.prop(props, 'graph_max_points')
        box.prop(props, 'graph_show_labels')
        box.prop(props, 'auto_reset_frame')

        row = layout.row(align=True)
        row.operator('wsim.build', icon='PHYSICS')
        row.operator('wsim.reset', icon='LOOP_BACK')

        row = layout.row(align=True)
        row.operator('wsim.toggle_hud', icon='HIDE_OFF')
        row.operator('wsim.clear_all', icon='TRASH')

        layout.separator()
        layout.label(text='Live Metrics')
        layout.label(text=f'Contacts: {len(SIM_STATE["contact_events"])}')
        layout.label(text=f'KE: {SIM_STATE["current_ke"]:.6f}')
        ke_norm = SIM_STATE['current_ke'] / max(SIM_STATE['initial_ke'], 1e-9)
        layout.label(text=f'Energy Remaining: {ke_norm:.6f}')
        layout.label(text=f'Cumulative loss: {SIM_STATE["cumulative_loss"]:.6f}')
        if SIM_STATE['last_contact']:
            lc = SIM_STATE['last_contact']
            layout.label(text=f'Last: {lc["link"]} approx dKE={lc["loss"]:.6f}')


def install_handlers():
    remove_handlers()
    bpy.app.handlers.frame_change_post.append(update_contact_graph_and_metrics)


def remove_handlers():
    if update_contact_graph_and_metrics in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(update_contact_graph_and_metrics)


classes = (
    WhipSimProps,
    WSIM_OT_build,
    WSIM_OT_reset,
    WSIM_OT_toggle_hud,
    WSIM_OT_clear_all,
    WSIM_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.whip_sim_props = PointerProperty(type=WhipSimProps)
    install_hud()


def unregister():
    remove_handlers()
    remove_hud()
    if hasattr(bpy.types.Scene, 'whip_sim_props'):
        del bpy.types.Scene.whip_sim_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()