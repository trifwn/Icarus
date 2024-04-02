#!/bin/bash
source /home/tryfonas/Applications/openfoam/openfoam/etc/bashrc
struct='/home/tryfonas/Applications/structAirfoilMesher/x64/Release/structAirfoilMesher'

while getopts n:p:b: flag
do
    case "${flag}" in
        n) AirfoilName=${OPTARG};;
    esac
done

$struct $AirfoilName < struct.input > struct.out
plot3dToFoam -noBlank  $AirfoilName.p3d > plot3dToFoam.out
autoPatch 45 > outPatch.out
rm -rf 1/
autoPatch -overwrite 45 > outPatch.out
