# Whip Around Pole Simulator for Blender

A Blender add-on / script that approximates a **whip wrapping around a pole** using a **chain of rigid links and rigid-body constraints**.

This version of the project includes the following important fixes and improvements:

- **Blender 4.x-safe rigid body world setup**
- **Safe selected-pole duplication** (the original selected mesh is no longer mutated or deleted)
- **Constant-radius helical wrapping** so the coil hugs the pole instead of growing outward like a funnel
- **Normalized energy remaining graph** instead of unreliable per-contact loss spikes
- **Reset logic fixed** so the graph and energy baseline restart correctly
- **`coil_pitch_per_turn` control** for tuning the tightness / spacing of the wrap

> **Important:** This is a **chain-link approximation** of a whip, not a full continuum whip solver. It is intended as a controllable mechanics visualization and research prototype.

---

## 1. What this simulator does

This project explores the mechanics idea:

> If a whip-like body wraps around a pole and loses energy through repeated contact, how does the remaining energy evolve as the wrap develops?

The Blender implementation represents the whip as:

- a sequence of **rigid links**
- joined by **rigid body constraints**
- interacting with a pole-shaped collider
- monitored in real time with a **HUD** and **energy graph**

The simulator is useful for studying:

- how the **first contact position** changes wrap potential
- how much **kinetic energy remains** as wrapping continues
- how **pole geometry** affects the motion
- how **fragmentation** changes the outcome if the whip is modeled as chain links

---

## 2. Key updates in this fixed version

### Safe selected-pole behavior

In earlier versions, using a selected object as the pole could risk modifying or deleting the original user object during rebuild / cleanup.

This version now:

- creates a **duplicate** of the selected mesh
- uses that duplicate as the simulation pole
- preserves the user’s original object

### Surface-hugging coil geometry

The wrap is now initialized as a **constant-radius helix** rather than a radius-growing spiral.

That means the links are placed around the pole using:

- a constant centerline radius
- an angle step based on link spacing
- a vertical pitch per full turn

This prevents the old **funnel-shaped outward growth**.

### Better graph metric

The graph now shows:
