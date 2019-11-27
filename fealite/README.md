![](https://i.imgur.com/Fb4SmAp.png)

*A Simple Workflow for Non Linear Poisson Problems*
#### Example 1: *Dielectric Cylindrical Shell in a Uniform Electric Field*
Create the Mesh with Mathematica
```wolfram
Needs["NDSolve`FEM`"]
Needs["FEMAddOns`"]

bmesh = BoundaryElementMeshJoin[
   ToBoundaryMesh[Annulus[{0, 0}, {1, 1.5}], 
    "BoundaryMarkerFunction" -> Function[{coords, points}, If[
         Norm[# // Mean] > 1, 5,
         6 ] & /@ coords]], 
   ToBoundaryMesh[Rectangle[{-3, -3}, {3, 3}]]];
air = {{0, 0}, {0, 2}};
dielectric = {0, 1.25};
mesh = ToElementMesh[bmesh, 
   "RegionMarker" -> Append[{#, 1} & /@ air, {dielectric, 2}], 
   "MeshOrder" -> 1, "NodeReordering" -> True];
mesh["Wireframe"]
```
![](https://i.imgur.com/e9ZjHOM.png)

###### tags: `Non-Linear` `FEA` `Simulation`
