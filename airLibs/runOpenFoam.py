from subprocess import call
import os
import shutil
import numpy as np

def makeMesh(airfoilFile):
    call(['/bin/bash', '-i', '-c', '../airLibs/setupFoam.sh -f ' +airfoilFile])
    
def setupOpenFoam(Reynolds,Mach,anglesALL,silent = False):
    folders = next(os.walk('.'))[1]
    parentdir = os.getcwd()
    # makeMesh()
    for ang in anglesALL:
        if ang >= 0:
            folder = str(ang)[::-1].zfill(7)[::-1] +'/'
        else:
            folder = 'm'+str(ang)[::-1].strip('-').zfill(6)[::-1] +'/'      
        if folder[:-1]  in folders:
            os.chdir(f'{parentdir}/{folder}')
            ang = ang*np.pi/180
            cwd = os.getcwd()
            
            shutil.copytree('../Base/0/',cwd+'/0/',dirs_exist_ok=True)
            filen = '0/U'
            with open(filen, 'r',newline='\n') as file:
                data = file.readlines()
            data[26]=f'internalField uniform ( {np.cos(ang)} {np.sin(ang)} 0. );\n'
            with open(filen, 'w') as file:
                file.writelines( data )
                
            shutil.copytree('../Base/constant/',cwd+'/constant/',dirs_exist_ok=True)
            filen = 'constant/transportProperties'
            with open(filen, 'r',newline='\n') as file:
                data = file.readlines()
            data[20] = f'nu              [0 2 -1 0 0 0 0] {np.format_float_scientific(1/Reynolds,sign=False,precision=3)};\n'
            with open(filen, 'w') as file:
                file.writelines( data )
                                  
            shutil.copytree('../Base/system/',cwd+'/system/',dirs_exist_ok=True)
            filen = 'system/controlDict'
            with open(filen, 'r',newline='\n') as file:
                data = file.readlines()
            data[95] = f'\t\tliftDir ({-np.sin(ang)} {np.cos(ang)} {0.});\n'
            data[96] = f'\t\tdragDir ({np.cos(ang)} {np.sin(ang)} {0.});\n'
            data[110] = f'\t\tUInf ({np.cos(ang)} {np.sin(ang)} {0});\n'
            with open(filen, 'w') as file:
                file.writelines( data )
            if silent == False:
                print(f'{cwd} Ready to Run')
    os.chdir(f'{parentdir}')

def runFoamAngle(angle):
    if angle >= 0:
        folder = str(angle)[::-1].zfill(7)[::-1] +'/'
    else:
        folder = 'm'+str(angle)[::-1].strip('-').zfill(6)[::-1] +'/'    
    parentDir  = os.getcwd()
    os.chdir(folder)
    os.system('../../airLibs/runFoam.sh')
    os.chdir(parentDir)

def runFoam(anglesAll):
    for angle in anglesAll:
        runFoamAngle(angle)
        
def makeCLCD(anglesAll):
    cd = []
    cl = []
    cm = []
    folders = next(os.walk('.'))[1]
    angleSucc = []
    for angle in anglesAll:
        if angle >= 0:
            folder = str(angle)[::-1].zfill(7)[::-1] 
        else:
            folder = 'm'+str(angle)[::-1].strip('-').zfill(6)[::-1] 
        if folder in folders:
            angleSucc.append(angle)   
            Time,Cd,Cdf,Cdr,Cl,Clf,Clr,CmPitch,CmRoll,CmYaw,Cs,Csf,Csr=[float(i) for i in getCoeffs(angle).split('\t')]
            cd.append(Cd)
            cl.append(Cl)
            cm.append(CmPitch)   
    return np.vstack([angleSucc,cl,cd,cm]).T

def getCoeffs(angle):
    if angle >= 0:
        folder = str(angle)[::-1].zfill(7)[::-1] +'/'
    else:
        folder = 'm'+str(angle)[::-1].strip('-').zfill(6)[::-1] +'/'    
    parentDir  = os.getcwd()
    os.chdir(folder)
    os.chdir('postProcessing/force_coefs/0')
    filen = 'coefficient.dat'
    with open(filen, 'r',newline='\n') as file:
        data = file.readlines()
    os.chdir(parentDir)
    return data[-1]
