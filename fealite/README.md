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
Export to .tmh format
```wolfram
FormatNumbers[{x_, 
   y_}] := (ToString[
     NumberForm[#1, {8, 8}, ExponentFunction -> (Null &)]] &) /@ {x, y}
     
ExportMesh[name_, mesh_] := 
 Export[name, 
  Join[{"# Coordinates-" <> ToString[Length[mesh["Coordinates"]]]}, 
   FormatNumbers /@ 
    mesh["Coordinates"], {"# Triangle Elements-" <> 
     ToString[Length[mesh["Coordinates"]]]}, 
   Flatten /@ 
    Transpose[{mesh["MeshElements"][[1, 1]], 
      mesh["MeshElements"][[1, 2]]}], {"# Boundary Elements-" <> 
     ToString[Length[mesh["BoundaryElements"][[1, 2]]]]}, 
   Flatten /@ 
    Transpose[{mesh["BoundaryElements"][[1, 1]], 
      mesh["BoundaryElements"][[1, 2]]}]], "Table"]
      
ExportMesh[MeshDirectory<>"annulus.tmh", mesh]
```
```
# Coordinates-467
-0.8743	-1.2190
. . .
# Triangle Elements-467
36	37	38	1
. . .
# Boundary Elements-164
317	318	6
. . .
```
Implement or choose a premade `PoissonProblemDefinition`
```python
from fealite.poisson import DielectricObjectInUniformField, Poisson

problem_definition =
     DielectricObjectInUniformField(mesh='meshes/annulus.tmh',
     source_marker=2, sink_marker=6,
     dielectric_marker=2)
     
solver = Poisson(problem_definition)
solver.export_solution()
```
Visualize with Mathematica
```wolfram
InterpAndShow[data_, object_] := 
 Module[{f = Interpolation[data, InterpolationOrder -> 1]}, 
  g = -Grad[f[xx, yy],{xx, yy}]; 
  Show[ContourPlot[
    f[xx, yy], {xx, yy} \[Element] Rectangle[{-3, -3}, {3, 3}], 
    ColorFunction -> "Pastel"], 
   Graphics[{Opacity[0], EdgeForm[Thickness[.01/5]], object}], 
   StreamPlot[g, {xx, yy} \[Element] Rectangle[{-3, -3}, {3, 3}]]]]
   
InterpAndShow[
  Import[
    SolutionDirectory<>"annulus_dielectric.txt", "Data"
  ]
]
```
![](https://i.imgur.com/QcYc7fF.png)

###### tags: `Non-Linear` `FEA` `Simulation`
