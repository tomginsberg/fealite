(* ::Package:: *)

Needs["NDSolve`FEM`"]
Needs["FEMAddOns`"]


FormatNumbers[{x_,y_}]:=(ToString[NumberForm[#1,{8,8},ExponentFunction->(Null&)]]&)/@{x,y}
ExportMesh[name_,mesh_]:=Export[name,Join[{"# Coordinates-"<>ToString[Length[mesh["Coordinates"]]]},FormatNumbers/@mesh["Coordinates"],{"# Triangle Elements-"<>ToString[Length[mesh["Coordinates"]]]},Flatten/@Transpose[{mesh["MeshElements"][[1,1]],mesh["MeshElements"][[1,2]]}],{"# Boundary Elements-"<>ToString[Length[mesh["BoundaryElements"][[1,2]]]]},Flatten/@Transpose[{mesh["BoundaryElements"][[1,1]],mesh["BoundaryElements"][[1,2]]}]],"Table"]
PlotIncidence[mesh_]:=MatrixPlot[SparseArray[Flatten[Map[Outer[List,#,#]&,Join@@ElementIncidents[mesh["MeshElements"]]],2]->_]]
ReadMesh[file_]:=Module[{data=StringSplit[Import[file,"String"],"#"]},
	coords=(ToExpression@StringSplit[#,"\t"])&/@StringSplit[data[[1]],"\n"][[2;;All]];
	meshElements=TriangleElement@@Transpose[(({{#1,#2,#3},#4}&)@@ToExpression[StringSplit[#1,"\t"]]&)/@StringSplit[data[[2]],"\n"][[2;;All]]];
	boundaryElements=LineElement@@Transpose[(({{#1,#2},#3}&)@@ToExpression[StringSplit[#1,"\t"]]&)/@StringSplit[data[[3]],"\n"][[2;;All]]];
	meshMarkers=CountDistinct[meshElements[[2]]];
	ToElementMesh["Coordinates"->coords,"MeshElements"->{meshElements},"BoundaryElements"->{boundaryElements}]
]
ToElementMeshDefault[bmesh_,markers_]:=ToElementMesh[bmesh,"RegionMarker"->markers,"NodeReordering"->True,"MeshOrder"->1]
MeshInspect[mesh_]:=GraphicsRow[{mesh["Wireframe"["MeshElement"->"BoundaryElements","MeshElementMarkerStyle"->Red]],mesh["Wireframe"["MeshElementStyle"->FaceForm/@ColorData["DarkRainbow"]/@Subdivide[CountDistinct[mesh["MeshElements"][[1,2]]]]]]}]//Legended[#,SwatchLegend[ColorData["DarkRainbow"]/@Subdivide[CountDistinct[mesh["MeshElements"][[1,2]]]],KeySort[mesh["MeshElements"][[1,2]]//Counts]//Keys]]&
InterpAndShow[data_,object_]:=Module[{f=Interpolation[data,InterpolationOrder->1]},
g=-Grad[f[xx,yy],{xx,yy}];domain=Rectangle@@(f["Domain"]//Transpose);Show[ContourPlot[f[xx,yy],{xx,yy}\[Element]domain,ColorFunction->"Pastel",PlotLegends->Automatic],Graphics[{Opacity[0],EdgeForm[Thickness[.01/5]],object}],StreamPlot[g,{xx,yy}\[Element]domain]]]
RescaleRegion[r_]:=RegionResize[r,{#*{-1,1},{-1,1}}&@(#[[1]]/#[[2]]&@(Total[{-1,1}.#]&/@Transpose@(BoundingRegion[r][[i]]~Table~{i,1,2})))]
