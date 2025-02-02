import os;
import numpy as np
import latexify

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

@latexify.algorithmic
def generate_pose(file):
        
        bottom = [0,0,0,1]
        for line in file:
            elems = line
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            R = qvec2rotmat(-qvec)
            t = tvec.reshape([3,1])
            m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            c2w = np.linalg.inv(m)
            c2w[0:3,2] *= -1 # flip the y and z axis
            c2w[0:3,1] *= -1
            c2w = c2w[[1,0,2,3],:]
            c2w[2,:] *= -1 # flip whole world upside down

print(generate_pose)