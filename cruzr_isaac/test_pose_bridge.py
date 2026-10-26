import math
import unittest
from pose_bridge import matrix, multiply, transform_gaussian, validate_rigid


class CoordinateTests(unittest.TestCase):
    def test_millimetres_and_binding_order(self):
        world = matrix([1,2,3], [math.sqrt(.5),0,0,math.sqrt(.5)])
        bind = matrix([.2,0,0], [1,0,0,0])
        mean,cov = transform_gaussian([1000,0,0], [[4,0,0],[0,1,0],[0,0,9]], multiply(world,bind), .001)
        for got,want in zip(mean,[1,3.2,3]): self.assertAlmostEqual(got,want)
        for i,want in enumerate([1e-6,4e-6,9e-6]): self.assertAlmostEqual(cov[i][i],want)

    def test_reject_reflection_and_scale(self):
        for value in [-1,2]:
            t=matrix([0,0,0],[1,0,0,0]);t[0][0]=value
            with self.assertRaises(ValueError):validate_rigid(t)

    def test_reject_bad_quaternion(self):
        for q in [[0,0,0,0],[float('nan'),0,0,1]]:
            with self.assertRaises(ValueError):matrix([0,0,0],q)

    def test_inverse_rigid_round_trip(self):
        t=matrix([1,-2,3],[.5,.5,.5,.5]);validate_rigid(t)
        r=[[t[j][i] for j in range(3)] for i in range(3)]
        inv=[r[i]+[-sum(r[i][j]*t[j][3] for j in range(3))] for i in range(3)]+[[0,0,0,1]]
        out=multiply(t,inv)
        for i in range(4):
            for j in range(4):self.assertAlmostEqual(out[i][j],float(i==j))


if __name__=='__main__':unittest.main()
