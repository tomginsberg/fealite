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
from src.poisson import LinearPoisson
from src import PoissonProblemDefinition

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
outerDiameter = 10/100;
outerShellThickness = 1/100;
rotorInnerDiameter = 1.5/100;
rotorOuterDiameter = 4.5/100;
numCoils = 6;
rotorStatorDistance = .25/100;
coilInnerDiameter = 3.5/100;
coilOuterDiameter = coilInnerDiameter*1.4;
coilInnerOffset = 0.3/100;
coilOuterOffset = 0.4/100;
offsetCoils = 1/2;
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
          2 π)/numCoils*offsetCoils, 
         2 π - (2 π)/numCoils (-offsetCoils + 1), (2 π)/
         numCoils}]] // BoundaryDiscretizeRegion // ToBoundaryMesh, 
   ToBoundaryMesh[
    BoundaryDiscretizeRegion[
     RegionUnion @@ 
      Table[RotationTransform[x, {0, 0}][
        Rectangle[{rotorOuterDiameter + rotorStatorDistance + 
           coilInnerOffset, -(coilOuterDiameter/2)}, {outerDiameter - 
           outerShellThickness - coilOuterOffset, coilOuterDiameter/
          2}]], {x, (2 π)/numCoils*offsetCoils, 
        2 π - (2 π)/numCoils (-offsetCoils + 1), (2 π)/
        numCoils}]]]];
bMotorMesh["Wireframe"]
```
![](https://i.imgur.com/YEW7HGl.png)


Specify the region coordinates

```Mathematica
markList[m_][{x_, y_}] := {{x, y}, m}
{air, rotor, stator} = Range[0, 2];

Clear[x, n]
statorSpec = 
  markList[stator] /@ 
   Flatten[{Table[
      RotationTransform[
        x, {0, 0}][{rotorOuterDiameter + rotorStatorDistance + 
          outerDiameter - outerShellThickness, 0}/2], {x, (2 π)/
        numCoils*offsetCoils, 
       2 π - (2 π)/numCoils*(-offsetCoils + 1), (2 π)/
       numCoils}], 
     Table[RotationTransform[
        x, {0, 0}][{rotorOuterDiameter + rotorStatorDistance + 
         coilInnerOffset/2, 0}], {x, (2 π)/numCoils*offsetCoils, 
       2 π - (2 π)/numCoils (-offsetCoils + 1), (2 π)/
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
      Piecewise[{{2 + n, n <= numCoils}}, (2 + n) - numCoils + 
         If[Mod[n, 2] == 1, 1, -1]] /. 
       n -> (2 (x - (2 π)/numCoils*offsetCoils)*numCoils/(2*Pi) + 
          1)}, {RotationTransform[
        x, {0, 0}]@{((rotorOuterDiameter + rotorStatorDistance + 
             coilInnerOffset) + (outerDiameter - outerShellThickness -
              coilOuterOffset))/
         2, (coilInnerDiameter + coilOuterDiameter)/4}, 
      Piecewise[{{2 + n, n <= numCoils}}, (2 + n) - numCoils + 
         If[Mod[n, 2] == 1, 1, -1]] /. 
       n -> (2 (x - (2 π)/numCoils*offsetCoils)*numCoils/(2*Pi) + 
          2)}}, {x, (2 π)/numCoils*offsetCoils, 
     2 π - (2 π)/numCoils (-offsetCoils + 1), (2 π)/
     numCoils}];
spec = Join[statorSpec, airSpec, rotorSpec, coilSpec];
```
Create and export the element mesh

```Mathematica
MotorMesh = 
  ToElementMesh[bMotorMesh, "RegionMarker" -> spec, MeshQualityGoal -> .3, 
   MaxCellMeasure -> .00001, "NodeReordering" -> True, 
   "MeshOrder" -> 1];
MeshInspect[MotorMesh]
ExportMesh["../meshes/bldc-6.tmh", MotorMesh]
```

![](https://i.imgur.com/zfGwmWP.png)

Define the boundary value problem by implementing a `NonLinearPoissonProblemDefinition`
```python
from typing import Union, Optional
import numpy as np
from mesh import TriangleMesh, Meshes
from poisson import NonLinearPoissonProblemDefinition, NonLinearPoisson
from math import e, pi

MU0 = 4e-7 * pi

# A helper function to produce a square wave
def square_wave(x: float) -> int:
    if 1 / 2 <= x % 1 < 1:
        return -1
    return 1


# A polynomial interpolation of (B, B/H) from a BH curve for steel
def nu(x: float, div: bool) -> float:
    if div:
        return -((0.0000657508 - 0.0000385014 * x + 0.000830739 * x ** 2 - 0.00163689 * x ** 3 + 0.00104806 * x ** 4
                  - 0.000228563 * x ** 5) / (0.0000379551 + 0.0000657508 * x - 0.0000192507 * x ** 2
                                             + 0.000276913 * x ** 3 - 0.000409223 * x ** 4 + 0.000209611 * x ** 5
                                             - 0.0000380938 * x ** 6) ** 2)

    return 1 / (
            0.000037955129560483474 + 0.00006575078254916768 * x - 0.000019250684240476186 * x ** 2 +
            0.0002769129642720887 * x ** 3 - 0.00040922344805632805 * x ** 4 + 0.0002096110387525365 * x ** 5
            - 0.00003809378487585398 * x ** 6)


class BLDC(NonLinearPoissonProblemDefinition):
    def __init__(self, mesh: Union[str, TriangleMesh] = '../meshes/bldc-6.tmh'):
        super().__init__(mesh)
        # Specify a coil current density by assuming a 1mm wire diameter and 20A
        wire_diameter = 1e-3
        self.coil_current = 20 / (pi / 4 * (wire_diameter ** 2))
        # Specify the relative permeability of a Neodymium 
        self.magnet_mur = 1.05
        # Specify a realistic surface current density for the permanent magnet
        # See the presentation slides for a proper derivation
        current_sheet_thickness = .8/100
        self.magnet_current = 500000/current_sheet_thickness
    
    # Specify source current locations
    def source(self, element_marker: int, coordinate: np.ndarray) -> float:
        if element_marker == 3 or element_marker == 8:
            return -self.coil_current
        if element_marker == 4 or element_marker == 7:
            return self.coil_current
        if element_marker == 1:
            x, y = coordinate
            norm = np.sqrt(x ** 2 + y ** 2)
            if 0.015 <= norm <= 0.023:
                return self.magnet_current * square_wave(np.arctan2(y, x) / (2 * pi) + 1 / 4)
            if 0.037 <= norm <= 0.045:
                return -self.magnet_current * square_wave(np.arctan2(y, x) / (2 * pi) + 1 / 4)
        return 0
    
    # Specify zero boundary conditions on the border of the motor
    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        if coordinate[0] ** 2 + coordinate[1] ** 2 > 0.009025:
            return 0
        return None

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return None
    
    # Specify linear and non linear permeabilities 
    def material(self, element_marker: int, norm_grad_phi: Optional[float] = None, div: bool = False) -> float:
        # Stator
        if element_marker == 2:
            if norm_grad_phi is None:
                norm_grad_phi = 0
            return nu(norm_grad_phi, div)

        if div:
            return 0

        # Magnet
        if element_marker == 1:
            return 1 / (self.magnet_mur * MU0)

        # Air
        return 1 / MU0

# Solve the non linear instance
if __name__ == '__main__':
    problem = NonLinearPoisson(BLDC())
    problem.solve_and_export()
```

Visualize the results
```Mathematica
current = 20
A = Interpolation[
  Import["../solutions/bldc-6_linearized.txt", "Data"], 
  InterpolationOrder -> 1]
{min, max} = MinMax[A["ValuesOnGrid"]]

potential = 
 Show[ContourPlot[A[x, y], {x, y} \[Element] MotorMesh, 
   PlotRange -> Full, Contours -> Range[min, max, (max - min)/15], 
   PlotLabel -> 
    Style[StringForm[
      "Potential and Flux Inside a `` Coil `` Pole BLDC Motor: ``", 
      numCoils, poles, 
      If[current == 0, "Unloaded", 
       StringForm["Load Current `` A", current]]], FontSize -> 16], 
   PlotTheme -> "Scientific", ColorFunction -> "TemperatureMap", 
   LabelStyle -> {FontFamily -> "CMU Serif", FontColor -> Black, 
     FontSize -> 14}, 
   PlotLegends -> 
    BarLegend[Automatic, 
     LegendLabel -> 
      Placed[Rotate["Magnetic Potential [T\[CenterDot]m]", Pi/2], 
       Right], LegendMarkerSize -> 500]], 
  StreamPlot[-Curl[A[x, y], {x, y}] // Evaluate, {x, y} \[Element] 
    MotorMesh, StreamPoints -> Fine, 
   StreamColorFunction -> "Rainbow"], 
  Graphics@Table[{FaceForm[{Opacity[.3], {Red // Lighter, 
         Blue // Lighter}[[Mod[x + 1, 2] + 1]]}], 
     Annulus[{0, 0}, {rotorInnerDiameter, 
       rotorOuterDiameter}, {2*Pi/poles (x + offset), 
       2*Pi/poles (x + offset) + 2*Pi/poles}]}, {x, 0, poles}], 
  bMotorMesh["Wireframe"], ImageSize -> 600, 
  FrameLabel -> {"[m]", "[m]"}, 
  LabelStyle -> {FontFamily -> "CMU Serif", FontColor -> Black, 
    FontSize -> 14}]
```
![](https://i.imgur.com/BEMEcJA.jpg)
```Mathematica
fluxDensity = 
 Show[StreamDensityPlot[-Curl[A[x, y], {x, y}] // 
    Evaluate, {x, y} \[Element] MotorMesh, StreamPoints -> Fine, 
   ColorFunction -> "TemperatureMap", PlotTheme -> "Scientific", 
   PlotRange -> Full, 
   PlotLabel -> 
    Style[StringForm[
      "Flux Lines and Density Inside a `` Coil `` Pole BLDC Motor: \
``", numCoils, poles, 
      If[current == 0, "Unloaded", 
       StringForm["Load Current `` A", current]]], FontSize -> 16], 
   ColorFunctionScaling -> True, 
   LabelStyle -> {FontFamily -> "CMU Serif", FontColor -> Black, 
     FontSize -> 14}, StreamMarkers -> "PinDart", 
   StreamColorFunction -> Function[x, Black], 
   PlotLegends -> 
    BarLegend[Automatic, 
     LegendLabel -> Placed[Rotate["Flux Density [T]", Pi/2], Right], 
     LegendMarkerSize -> 500]], bMotorMesh["Wireframe"], 
  Graphics[{White, 
    DiscretizeRegion[
     RegionDifference[Rectangle[{-.11, -.11}, {.11, .11}], 
      Disk[{0, 0}, outerDiameter]]]}], ImageSize -> 600, 
  FrameLabel -> {"[m]", "[m]"}, 
  LabelStyle -> {FontFamily -> "CMU Serif", FontColor -> Black, 
    FontSize -> 14}]
```
![](https://i.imgur.com/UNX9WNk.jpg)

Calculate the torque on ther magnet by averaging several contour integrals inside the airgap
```Mathematica
torque = ListLinePlot[#, 
    PlotLabel -> 
     Style[StringForm[
       "\!\(\*OverscriptBox[\(\[Tau]\), \(^\)]\)=`` [N]", 
       Mean[#[[;; , 2]]]], FontSize -> 16], 
    LabelStyle -> {FontFamily -> "CMU Serif", FontColor -> Black, 
      FontSize -> 14}, AxesLabel -> {"Integration Radius", "Torque"}, 
    ImageSize -> Large, 
    Filling -> 
     Axis] &@(Table[{s, -Quiet@
        NIntegrate[
         r^2*Evaluate[
            Fold[Times, 
             TransformedField[
              "Cartesian" -> 
               "Polar", -Curl[A[x, y], {x, y}], {x, 
                y} -> {r, \[Theta]}]]] /. r -> s, {\[Theta], 0, 2*Pi},
          Method -> "LocalAdaptive"]/Subscript[µ, 0]}, {s, 
     Subdivide[rotorOuterDiameter, 
      rotorOuterDiameter + rotorStatorDistance, 10]}])
```
![](https://i.imgur.com/73joVEu.png)



###### tags: `Non-Linear` `FEA` `Simulation`
