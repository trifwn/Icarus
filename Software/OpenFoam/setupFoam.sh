#!/bin/bash
source /usr/lib/openfoam/openfoam/etc/bashrc
echo "Making Mesh"
struct='/opt/structAirfoilMesher/x64/Release/structAirfoilMesher'

while getopts n:p:b: flag
do
    case "${flag}" in
        n) airfoilName=${OPTARG};;
        p) airfoilPATH=${OPTARG};;
        b) baseDIR=${OPTARG};;
    esac
done

echo $airfoilPATH
cp $airfoilPATH .

$struct $airfoilName < struct.input > struct.out
plot3dToFoam -noBlank  $airfoilName.p3d > plot3dToFoam.out
autoPatch 45 > outPatch.out
rm -rf 1/
autoPatch -overwrite 45 > outPatch.out
cp boundaryTemplate constant/polyMesh/boundary
