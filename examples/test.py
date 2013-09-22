#!/usr/bin/env python2

import pyxflow.px as px

M = px.CreateMesh("../examples/uniform_tri_q1_2.gri")
print M
print px.DestroyMesh(M)

