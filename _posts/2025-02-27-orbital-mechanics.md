---
layout: post
title: "Reference Frames for Orbital Mechanics"
subtitle: "Cartesian and Keplerian"
date: 2025-02-17
categories: [space]
tags: [orbital-mechanics]
---

In a simple two-body problem where two bodies have finite mass, their trajectories in space will follow an elliptical orbit. When the mass of one of the bodies is much larger than the other, $$m_1>>m_2$$, we can approximate the larger body to be stationary while the smaller body will orbit around it. The trajectory of the smaller body can be modelled with either the Keplerian or Cartesian reference frames, each with their own pros and cons. The Cartesian reference frame is useful for numerical calculations, such as deriving future trajectories of an object given the current trajectory. The Keplerian frame is useful for gaining a more intuitive understanding of the object's orbit as it describes mathematically the phase space of the object's orbit.

#### Cartesian Reference Frame
<p align="center">
<img src="/images/orbit1.jpg" alt="Orbit Image" width="500">
</p>

This is the reference frame that is familiar to most people, with position $$\vec{r} = [x, y, z]$$ and velocity $$\vec{v} = [v_x, v_y, v_z]$$. It is also an inertial reference frame, with point of reference being the larger body which is typically Earth. The cartesian state vector is described as a six dimensional vector formed from the concatenation of the position and velocity vectors:

$$ \vec{c} = [x, y, z, v_x, v_y, v_z]$$
 
This state vector along with the gravitational parameter $$\mu$$ is sufficient to calculate future trajectories of the object.

#### Keplerian Reference Frame

In the Keplerian reference frame, an orbit is described not by instantaneous position and velocity components but by six **orbital elements** that capture the shape, size, orientation, and current position along the orbit. These six elements are:

- **Semimajor Axis** $$\textbf{a}$$:  
  Defines the size of the orbit—the average distance between the orbiting object and the central body.

- **Eccentricity,** $$\textbf{e}$$:  
  Describes the shape of the orbit. An eccentricity of 0 corresponds to a circular orbit, while values between 0 and 1 indicate an ellipse.

- **Inclination,** \(i\):  
  The tilt of the orbital plane relative to a chosen reference plane (often the equatorial or ecliptic plane).

- **Right Ascension of the Ascending Node** ($$\Omega$$):  
  The angle in the reference plane from a fixed direction (typically the vernal equinox) to the point where the orbiting body crosses the reference plane going upward (the ascending node).

- **Argument of Periapsis,** \($$\omega\$$):  
  Measured in the orbital plane, this angle specifies the direction of the closest approach (periapsis) relative to the ascending node.

- **True Anomaly,** \(\nu\):  
  The angle (also measured in the orbital plane) that indicates the object's current position along its orbit relative to the periapsis.

These elements combine to form the **Keplerian state vector**:

$$\vec{k} = [\,a,\; e,\; i,\; \Omega,\; \omega,\; \nu\,].$$

Together with the gravitational parameter \(\mu\), this 6‑element vector fully defines the orbit in the two‑body problem. Even though it does not directly include position and velocity components, these orbital elements contain all the necessary information—through Kepler’s laws and the laws of motion—to compute the object’s future trajectory.

By knowing \(\vec{k}\) and \(\mu\), one can derive the instantaneous Cartesian state vector (and vice versa) through established transformation formulas. This makes the Keplerian state vector not only a compact representation of the orbit’s geometry but also sufficient for predicting the motion of the orbiting object over time.
