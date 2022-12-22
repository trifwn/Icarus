#!/bin/bash
source /usr/lib/openfoam/openfoam2206/etc/bashrc

struct='/home/tryfonas/Applications/structAirfoilMesher/x64/Release/structAirfoilMesher'

while getopts f:a:r: flag
do
    case "${flag}" in
        f) airfoilFile=${OPTARG};;
        a) airfoil=${OPTARG};;
        r) fullname=${OPTARG};;
    esac
done
pwd
echo $airfoilFile
cp -r $airfoilFile Base/
cd Base/
pwd

# $struct $airfoilFile < struct.input > struct.out
# plot3dToFoam -noBlank  $airfoilFile.p3d > plot3dToFoam.out
# autoPatch 45 > outPatch.out
# rm -rf 1/
# autoPatch -overwrite 45 > outPatch.out
cp boundaryTemplate constant/polyMesh/boundary
