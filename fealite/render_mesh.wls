Needs["NDSolve`FEM`"]
input = $ScriptCommandLine[[2]]
output = $ScriptCommandLine[[3]]
data = StringSplit[Import[input, "String"], "#"]

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

colors = ColorData["DarkRainbow"]/@Subdivide[meshMarkers]

mesh = ToElementMesh["Coordinates" -> coords,
   "MeshElements" -> {meshElements},
   "BoundaryElements" -> {boundaryElements}][
  "Wireframe"["MeshElementStyle" -> FaceForm /@ colors]]

Export[output, mesh]