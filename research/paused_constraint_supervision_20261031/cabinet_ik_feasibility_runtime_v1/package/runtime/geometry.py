"""Pure XML/NumPy geometry; no solver or GPU imports."""
import numpy as np
import xml.etree.ElementTree as ET
def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def quaternion(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def pose(p, R):
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = p
    return out


def origin(joint):
    o = joint.find("origin")
    p = np.fromstring(o.get("xyz", "0 0 0"), sep=" ")
    roll, pitch, yaw = np.fromstring(o.get("rpy", "0 0 0"), sep=" ")
    R = rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)
    return pose(p, R)


def angular_error(a, b):
    relative = a.T @ b
    v = np.array([relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0], relative[1, 0]-relative[0, 1]])
    return float(np.arctan2(np.linalg.norm(v) / 2, (np.trace(relative)-1) / 2))



def fk(urdf, vectors):
    joints={j.get('name'):j for j in ET.parse(urdf).getroot().findall('joint')}
    output=[]
    for vector in vectors:
        T=np.eye(4); parent='fr_base_link'
        for name,angle in zip(['fr_joint'+str(i) for i in range(1,7)],vector,strict=True):
            j=joints[name]
            if j.get('type')!='revolute' or j.find('parent').get('link')!=parent:raise ValueError('Unexpected URDF chain')
            parent=j.find('child').get('link')
            axis=np.fromstring(j.find('axis').get('xyz'),sep=' ')
            T=T@origin(j)@pose([0,0,0],rotation(axis,float(angle)))
        if parent!='fr_link6':raise ValueError('Unexpected end link')
        output.append(T)
    return np.array(output)


def matrix_quaternion(R):
    # Fixed largest-diagonal algorithm, WXYZ, canonical w>=0. No solver-dependent choice.
    R=np.asarray(R,dtype=float); tr=float(np.trace(R))
    if tr>0:
        s=np.sqrt(tr+1.0)*2; q=np.array([s/4,(R[2,1]-R[1,2])/s,(R[0,2]-R[2,0])/s,(R[1,0]-R[0,1])/s])
    else:
        i=int(np.argmax(np.diag(R))); j=(i+1)%3;k=(i+2)%3
        s=np.sqrt(1+R[i,i]-R[j,j]-R[k,k])*2;q=np.zeros(4)
        q[0]=(R[k,j]-R[j,k])/s;q[i+1]=s/4;q[j+1]=(R[j,i]+R[i,j])/s;q[k+1]=(R[k,i]+R[i,k])/s
    q/=np.linalg.norm(q)
    return q if q[0]>=0 else -q
