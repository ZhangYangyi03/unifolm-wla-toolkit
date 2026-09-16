# -*- coding: utf-8 -*-
"""urdf_kin.py - zero-dependency URDF forward kinematics.

Why this is here: the unified action space needs end-effector and base poses
(spec sec. 3.1, slots [0:6], [13:19], [35:41]), but the public Unitree
datasets -- the ones WLA-1.0 was trained on -- ship joint angles only.
So the missing step between "downloaded a Unitree dataset" and "trainable in
the unified space" is forward kinematics, and it must be the FK of the exact
URDF the data was recorded with, not an approximation.

Implemented: the URDF kinematic tree, revolute / continuous / prismatic / fixed
joints, origin xyz+rpy (URDF rpy is the same fixed-axis convention the spec uses
in sec. 5.1), FK from the root to any link, and a closed-form self-check.

    python urdf_kin.py                                  closed-form check
    python urdf_kin.py robot.urdf                       list links and joints
    python urdf_kin.py robot.urdf --link left_wrist_yaw_link --joints "1.0,0,0"
"""
import argparse, math, os, tempfile, xml.etree.ElementTree as ET
import numpy as np


def rpy_to_R(rpy):
    r, p, y = [float(v) for v in rpy]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                     # fixed-axis xyz, same as spec sec. 5.1


def axis_angle_R(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    u = axis / n
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


class Joint:
    def __init__(self, name, jtype, parent, child, origin, axis, limit=None):
        self.name, self.jtype = name, jtype
        self.parent, self.child = parent, child
        self.origin, self.axis, self.limit = origin, axis, limit

    def motion(self, q):
        T = np.eye(4)
        if self.jtype in ('revolute', 'continuous'):
            T[:3, :3] = axis_angle_R(self.axis, q)
        elif self.jtype == 'prismatic':
            n = np.linalg.norm(self.axis) or 1.0
            T[:3, 3] = self.axis / n * q
        elif self.jtype != 'fixed':
            raise ValueError('joint type %r not supported' % self.jtype)
        return T

    def __repr__(self):
        return '<Joint %s %s %s->%s>' % (self.name, self.jtype, self.parent, self.child)


class Urdf:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.name = root.get('name')
        self.links = [l.get('name') for l in root.findall('link')]
        self.joints = []
        for j in root.findall('joint'):
            origin = np.eye(4)
            o = j.find('origin')
            if o is not None:
                xyz = [float(v) for v in (o.get('xyz') or '0 0 0').split()]
                rpy = [float(v) for v in (o.get('rpy') or '0 0 0').split()]
                origin[:3, :3] = rpy_to_R(rpy)
                origin[:3, 3] = xyz
            ax = j.find('axis')
            axis = np.array([float(v) for v in (ax.get('xyz') or '1 0 0').split()], dtype=float) \
                if ax is not None else np.array([1.0, 0.0, 0.0])
            lim = j.find('limit')
            limit = (float(lim.get('lower', -1e9)), float(lim.get('upper', 1e9))) if lim is not None else None
            self.joints.append(Joint(j.get('name'), j.get('type'),
                                     j.find('parent').get('link'), j.find('child').get('link'),
                                     origin, axis, limit))
        self.by_child = {j.child: j for j in self.joints}
        self.by_name = {j.name: j for j in self.joints}
        children = set(self.by_child)
        self.root = next((l for l in self.links if l not in children), None)
        if self.root is None:
            raise ValueError('no root link found (cycle in the kinematic tree?)')

    def chain(self, link):
        out, cur = [], link
        while cur != self.root:
            if cur not in self.by_child:
                raise KeyError('%s is not connected to the root %s' % (link, self.root))
            j = self.by_child[cur]
            out.append(j)
            cur = j.parent
        return list(reversed(out))

    def movable_joints(self):
        return [j for j in self.joints if j.jtype != 'fixed']

    def fk(self, q, link=None):
        T = np.eye(4)
        for j in self.chain(link) if link else []:
            T = T @ j.origin @ j.motion(float((q or {}).get(j.name, 0.0)))
        return T

    def fk_all(self, q):
        T = {self.root: np.eye(4)}
        for j in self.joints:
            if j.parent in T:
                T[j.child] = T[j.parent] @ j.origin @ j.motion(float((q or {}).get(j.name, 0.0)))
        return T

    def fk_series(self, qseries, links):
        """qseries: (T, njoints) array in movable-joints order -> {link: (T,4,4)}."""
        qseries = np.atleast_2d(np.asarray(qseries, dtype=np.float64))
        names = [j.name for j in self.movable_joints()]
        out = {l: np.zeros((len(qseries), 4, 4)) for l in links}
        for i, row in enumerate(qseries):
            q = {n: float(v) for n, v in zip(names, row)}
            T = self.fk_all(q)
            for l in links:
                if l not in T:
                    raise KeyError('%s is not reachable from the root' % l)
                out[l][i] = T[l]
        return out


TWO_LINK = '\n'.join([
    '<robot name="two_link">',
    '  <link name="base"/><link name="l1"/><link name="l2"/><link name="tip"/>',
    '  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>',
    '    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>',
    '    <limit lower="-3.14" upper="3.14"/></joint>',
    '  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>',
    '    <origin xyz="0.3 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>',
    '    <limit lower="-3.14" upper="3.14"/></joint>',
    '  <joint name="jt" type="fixed"><parent link="l2"/><child link="tip"/>',
    '    <origin xyz="0.2 0 0" rpy="0 0 0"/></joint>',
    '</robot>'])


def demo_check():
    """Two-link planar arm with a closed-form answer: the FK must match it."""
    p = os.path.join(tempfile.mkdtemp(), 'two_link.urdf')
    open(p, 'w', encoding='utf-8').write(TWO_LINK)
    u = Urdf(p)
    q = {'j1': 0.5, 'j2': -0.3}
    T = u.fk(q, 'tip')
    a, b = 0.5, 0.2                      # absolute orientation of each link
    ref = np.array([0.3 * math.cos(a) + 0.2 * math.cos(b),
                    0.3 * math.sin(a) + 0.2 * math.sin(b), 0.1])
    return float(np.max(np.abs(T[:3, 3] - ref))), T


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('urdf', nargs='?')
    ap.add_argument('--link')
    ap.add_argument('--joints')
    a = ap.parse_args()
    if not a.urdf:
        err, T = demo_check()
        print('two-link closed-form FK check: max error %.3e' % err)
        print(np.round(T, 6))
        raise SystemExit(0 if err < 1e-12 else 1)
    u = Urdf(a.urdf)
    if a.link:
        q = {}
        if a.joints:
            q = {j.name: float(v) for j, v in zip(u.movable_joints(), a.joints.split(','))}
        print('robot=%s root=%s link=%s chain=%s'
              % (u.name, u.root, a.link, [j.name for j in u.chain(a.link)]))
        print(np.round(u.fk(q, a.link), 6))
    else:
        print('robot=%s  root=%s  links=%d  joints=%d (%d movable)'
              % (u.name, u.root, len(u.links), len(u.joints), len(u.movable_joints())))
        for j in u.joints:
            print('  %-28s %-11s %s -> %s' % (j.name, j.jtype, j.parent, j.child))
