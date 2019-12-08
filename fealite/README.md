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
from fealite.poisson import LinearPoisson
from fealite.problem_definitions import PoissonProblemDefinition

problem_definition =
     DielectricObjectInUniformField(mesh='meshes/annulus.tmh',
     source_marker=2, sink_marker=4,
     dielectric_marker=2)
     
solver = LinearPoisson(problem_definition)
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

### Non Linear Finite Elements
#### Example 2: *BLDC Motor*
Setup
```Mathematica
SetDirectory[NotebookDirectory[]]
MeshPath = "meshes/"
<< MeshTools.wl
```
Specify the motor dimensions

```Mathematica
outerDiameter = 1;
outerShellThickness = .1;
rotorInnerDiameter = .15;
rotorOuterDiameter = .425;
numCoils = 8;
rotorStatorDistance = .05;
coilInnerDiameter = .25;
coilOuterDiameter = coilInnerDiameter*1.3;
coilInnerOffset = .03;
coilOuterOffset = 0.05;
offsetCoils = 1/2;
coilColor = Darker[Darker[Gray]];
rotorColor = Darker[Red // Darker];
statorColor = Black;
```
Create a boundary mesh

```Mathematica
bMotorMesh = 
  BoundaryElementMeshJoin[
   RegionUnion[
      Annulus[{0, 0}, {outerDiameter - outerShellThickness, 
        outerDiameter}], 
      Annulus[{0, 0}, {rotorInnerDiameter, rotorOuterDiameter}], 
      Sequence @@ 
       Table[RotationTransform[x, {0, 0}][
         Rectangle[{rotorOuterDiameter + rotorStatorDistance, -(
            coilInnerDiameter/2)}, {outerDiameter - 
            outerShellThickness, coilInnerDiameter/2}]], {x, (
          2 \[Pi])/numCoils*offsetCoils, 
         2 \[Pi] - (2 \[Pi])/numCoils (-offsetCoils + 1), (2 \[Pi])/
         numCoils}]] // BoundaryDiscretizeRegion // ToBoundaryMesh, 
   ToBoundaryMesh[
    BoundaryDiscretizeRegion[
     RegionUnion @@ 
      Table[RotationTransform[x, {0, 0}][
        Rectangle[{rotorOuterDiameter + rotorStatorDistance + 
           coilInnerOffset, -(coilOuterDiameter/2)}, {outerDiameter - 
           outerShellThickness - coilOuterOffset, coilOuterDiameter/
          2}]], {x, (2 \[Pi])/numCoils*offsetCoils, 
        2 \[Pi] - (2 \[Pi])/numCoils (-offsetCoils + 1), (2 \[Pi])/
        numCoils}]]]];
bMotorMesh["Wireframe"]
```
<src="https://i.imgur.com/Rf6sAZt.png", width=300>

Specify the region coordinates

```Mathematica
markList[m_][{x_, y_}] := {{x, y}, m}
{air, rotor, stator} = Range[0, 2];

statorSpec = 
  markList[stator] /@ 
   Flatten[{Table[
      RotationTransform[
        x, {0, 0}][{rotorOuterDiameter + rotorStatorDistance + 
          outerDiameter - outerShellThickness, 0}/2], {x, (2 \[Pi])/
        numCoils*offsetCoils, 
       2 \[Pi] - (2 \[Pi])/numCoils*(-offsetCoils + 1), (2 \[Pi])/
       numCoils}], 
     Table[RotationTransform[
        x, {0, 0}][{rotorOuterDiameter + 
         rotorStatorDistance + $MachineEpsilon, 0}], {x, (2 \[Pi])/
        numCoils*offsetCoils, 
       2 \[Pi] - (2 \[Pi])/numCoils (-offsetCoils + 1), (2 \[Pi])/
       numCoils}], {{outerDiameter - outerShellThickness/2, 0}}}, 1];
rotorSpec = {{{rotorOuterDiameter/2 + rotorInnerDiameter/2, 0}, 
    rotor}};
airSpec = {{{0, 0}, 
    air}, {{rotorOuterDiameter + rotorStatorDistance/2, 0}, air}};
coilSpec = 
  Join @@ Table[{{RotationTransform[
        x, {0, 0}]@{((rotorOuterDiameter + rotorStatorDistance + 
             coilInnerOffset) + (outerDiameter - outerShellThickness -
              coilOuterOffset))/
         2, (-coilInnerDiameter - coilOuterDiameter)/4}, 
      Piecewise[{{2 + n, n <= 8}}, (2 + n) - numCoils + 
         If[Mod[n, 2] == 1, 1, -1]] /. 
       n -> (2 (x - (2 \[Pi])/numCoils*offsetCoils)*numCoils/(2*Pi) + 
          1)}, {RotationTransform[
        x, {0, 0}]@{((rotorOuterDiameter + rotorStatorDistance + 
             coilInnerOffset) + (outerDiameter - outerShellThickness -
              coilOuterOffset))/
         2, (coilInnerDiameter + coilOuterDiameter)/4}, 
      Piecewise[{{2 + n, n <= 8}}, (2 + n) - numCoils + 
         If[Mod[n, 2] == 1, 1, -1]] /. 
       n -> (2 (x - (2 \[Pi])/numCoils*offsetCoils)*numCoils/(2*Pi) + 
          2)}}, {x, (2 \[Pi])/numCoils*offsetCoils, 
     2 \[Pi] - (2 \[Pi])/numCoils (-offsetCoils + 1), (2 \[Pi])/
     numCoils}];
spec = Join[statorSpec, airSpec, rotorSpec, coilSpec];

```
Create the element mesh

```Mathematica
MotorMesh = ToElementMeshDefault[bMotorMesh, spec];
MeshInspect[MotorMesh]
```

![](https://i.imgur.com/JTipxTM.png)



###### tags: `Non-Linear` `FEA` `Simulation`
