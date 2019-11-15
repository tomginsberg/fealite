Needs["NDSolve`FEM`"]
input = $ScriptCommandLine[[2]]
output = $ScriptCommandLine[[3]]
data = StringSplit[Import[input], "#"];

coords = (ToExpression@StringSplit[#, "\t"]) & /@
   StringSplit[data[[1]], "\n"][[2 ;; All]];
meshElements =
  TriangleElement @@
   Transpose[(({{#1, #2, #3}, #4} &) @@
        ToExpression[StringSplit[#1, "\t"]] &) /@
     StringSplit[data[[2]], "\n"][[2 ;; All]]];
boundaryElements =
  LineElement @@
   Transpose[(({{#1, #2}, #3} &) @@
        ToExpression[StringSplit[#1, "\t"]] &) /@
     StringSplit[data[[3]], "\n"][[2 ;; All]]];
meshMarkers = CountDistinct[meshElements[[2]]];
colors = {LightBlue, LightRed, LightGray, LightOrange, LightPurple,
   LightGreen, LightBrown}~Join~
  If[meshMarkers > 7, RandomColor[meshMarkers - 7], {}]
mesh = ToElementMesh["Coordinates" -> coords,
   "MeshElements" -> {meshElements},
   "BoundaryElements" -> {boundaryElements}][
  "Wireframe"["MeshElementStyle" -> FaceForm /@ colors]]
Export[output, mesh]