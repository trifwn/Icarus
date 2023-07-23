#!/bin/bash
source /home/tryfonas/Applications/openfoam/openfoam/etc/bashrc
struct='/home/tryfonas/Applications/structAirfoilMesher/x64/Release/structAirfoilMesher'

while getopts n:p:b: flag
do
    case "${flag}" in
        n) airfoilName=${OPTARG};;
    esac
done

$struct $airfoilName < struct.input > struct.out
plot3dToFoam -noBlank  $airfoilName.p3d > plot3dToFoam.out
autoPatch 45 > outPatch.out
rm -rf 1/
autoPatch -overwrite 45 > outPatch.out
