![](https://i.imgur.com/Fb4SmAp.png)

*A Simple Workflow for Linear & Non Linear Poisson Problems*
### Linear Finite Elements
#### Example 1: *Dielectric Cylinder in a Uniform Electric Field*
Create the Mesh with Mathematica
```Mathematica
(* If needed *)
(* ResourceFunction["FEMAddOnsInstall"][] *)
<< MeshTools.wl

bmesh = BoundaryElementMeshJoin[
   ToBoundaryMesh[Annulus[{0, 0}, {1, 1.5}], 
    "BoundaryMarkerFunction" -> Function[{coords, points}, If[
         Norm[# // Mean] > 1, 5,
         6 ] & /@ coords]], 
   ToBoundaryMesh[Rectangle[{-3, -3}, {3, 3}]]];
air = {{0, 0}, {0, 2}};
dielectric = {0, 1.25};
mesh = ToElementMeshDefault[bmesh, (* Specify Markers *) Append[{#, 1} & /@ air, {dielectric, 2}]];
```
View the mesh
```Mathematica
MeshInspect[mesh]
```
![](https://i.imgur.com/wlg9WmG.png)

Export to .tmh format
```Mathematica
ExportMesh[MeshDirectory<>"annulus.tmh", mesh]
```
The exported file looks like the following
```
# Coordinates-467
-0.8743	-1.2190
. . .
# Triangle Elements-467
36	37	38	1
. . .
# Boundary Elements-164
317	318	6	3
. . .
```
Implement or choose a premade `PoissonProblemDefinition`
```python
from fealite.poisson import DielectricObjectInUniformField, Poisson

problem_definition =
     DielectricObjectInUniformField(mesh='meshes/annulus.tmh',
     source_marker=2, sink_marker=4,
     dielectric_marker=2)
     
solver = Poisson(problem_definition)
solver.export_solution()
```
Visualize with Mathematica
```Mathematica
InterpAndShow[
  Import[
    SolutionDirectory<>"annulus_dielectric.txt", "Data"
  ]
]
```
![](https://i.imgur.com/MGJ96Kb.png)
*Equipotential Contours and Field Arrows*

###### tags: `Non-Linear` `FEA` `Simulation`
