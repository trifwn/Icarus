import numpy as np
from . import DB3D, APPHOME
import os
import pandas as pd

from ICARUS.Software.GenuVP3.postProcess.convergence import getLoadsConvergence, addErrorConvergence2df

from ICARUS.Core.struct import Struct

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
jsonpickle_pd.register_handlers()


class Database_3D():
    def __init__(self):
        self.HOMEDIR = APPHOME
        self.DATADIR = DB3D
        self.rawData = Struct()
        self.Data = Struct()
        self.Planes = Struct()
        self.dynPlanes = Struct()
        self.Convergence = Struct()
    
    def loadData(self):
        self.scan()
        self.makeData()

    def scan(self):
        planenames = next(os.walk(DB3D))[1]
        for plane in planenames: # For each plane planename == folder 
            # if plane == 'bmark':
            #     continue           
            foundPlane = False
            # Load Plane object
            file = os.path.join(DB3D, plane, f'{plane}.json')
            foundPlane = self.loadPlaneFromFile(plane,file)
            
            # Load DynPlane object
            file = os.path.join(DB3D, plane, f'dyn_{plane}.json')
            temp = self.loadDynPlaneFromFile(plane,file)
            
            foundPlane = foundPlane or temp                   

            # Loading Forces from forces.gnvp3 file
            file = os.path.join(DB3D, plane, 'forces.gnvp3')
            self.loadGNVPForces(plane,file)
            
            if foundPlane:
                self.Convergence[plane] = Struct()
                cases = next(os.walk(os.path.join(DB3D, plane)))[1]
                for case in cases:
                    if case.startswith("Dyn") or case.startswith("Sens"):
                        continue
                    self.loadGNVPcaseConvergence(plane, case)

    def loadPlaneFromFile(self,name,file):
        try:
            with open(file, 'r') as f:
                json_obj = f.read()
                try:
                    self.Planes[name] = jsonpickle.decode(json_obj)
                except Exception as e:
                    print(f'Error decoding Plane object {name}! Got error {e}')
            foundPlane = True
        except FileNotFoundError:
            print(f'No Plane object found in {name} folder at {file}!')
            foundPlane = False
        return foundPlane
    
    def loadDynPlaneFromFile(self,name,file):
        try:
            with open(file, 'r') as f:
                json_obj = f.read()
                try:
                    self.dynPlanes[name] = jsonpickle.decode(json_obj)
                    if name not in self.Planes.keys():
                        print('Plane object doesnt exist! Creating it...')
                        self.Planes[name] = self.dynPlanes[name]
                except Exception as e:
                    print(f'Error decoding Dyn Plane object {name} ! Got error {e}')
        except FileNotFoundError:
            print(f"No Plane object found in {name} folder at {file}!")

    def loadGNVPForces(self, planename, file):
        # Should get deprecated in favor of analysis logic in the future
        try:
            self.rawData[planename] = pd.read_csv(file)
            return
        except FileNotFoundError:
            print(f'No forces.gnvp3 file found in {planename} folder at {DB3D}!')
            if planename in self.Planes.keys():
                print('Since plane object exists with that name trying to create polars...')
                pln = self.Planes[planename]
                try:
                    from ICARUS.Software.GenuVP3.filesInterface import makePolar
                    CASEDIR = os.path.join(DB3D, pln.CASEDIR)
                    makePolar(CASEDIR, self.HOMEDIR)
                    file = os.path.join(DB3D, planename, 'forces.gnvp3')
                    self.rawData[planename] = pd.read_csv(file)
                except Exception as e:
                    print(f'Failed to create Polars! Got Error:\n{e}')

    def loadGNVPcaseConvergence(self, planename, case):
        # Get Load Convergence Data from LOADS_aer.dat
        file = os.path.join(DB3D, planename, case, 'LOADS_aer.dat')
        
        loads = getLoadsConvergence(file) 
        if loads is not None:
            # Get Error Convergence Data from gnvp.out
            file = os.path.join(DB3D, planename, case, 'gnvp.out')
            # self.Convergence[planename][case] = addErrorConvergence2df(file, loads) # IT OUTPUTS LOTS OF WARNINGS
            with open(file, 'r') as f:
                lines = f.readlines() 
            time, error, errorm = [] , [] , []
            for line in lines: 
                if not line.startswith(" STEP="):
                    continue

                a = line[6:].split()
                time.append(int(a[0]))
                error.append(float(a[2]))
                errorm.append(float(a[6]))
            try:
                foo = len(loads['TTIME'])
                if foo > len(time):
                    loads = loads.tail(len(time))
                else:
                    error = error[-foo:]
                    errorm = errorm[-foo:]
                loads["ERROR"] = error
                loads["ERRORM"] = errorm
                self.Convergence[planename][case] = loads
            except ValueError as e:
                print(f"Some Run Had Problems!\n{e}")

    def getPlanes(self):
        return list(self.Planes.keys())

    def getPolar(self, plane, mode):
        try:
            cols = ["AoA", f"CL_{mode}", f"CD_{mode}", f"Cm_{mode}"]
            return self.Data[plane][cols].rename(columns={f"CL_{mode}": "CL",
                                                          f"CD_{mode}": "CD",
                                                          f"Cm_{mode}": "Cm"})
        except KeyError:
            print("Polar Doesn't exist! You should compute it first!")

    def makeData(self):
        for plane in list(self.Planes.keys()):
            self.Data[plane] = pd.DataFrame()
            pln = self.Planes[plane]
            self.Data[plane]["AoA"] = self.rawData[plane]["AoA"]
            AoA = self.rawData[plane]["AoA"] * np.pi/180
            for enc, name in zip(["", "2D", "DS2D"], ["Potential", "2D", "ONERA"]):
                Fx = self.rawData[plane][f"TFORC{enc}(1)"]
                Fy = self.rawData[plane][f"TFORC{enc}(2)"]
                Fz = self.rawData[plane][f"TFORC{enc}(3)"]

                Mx = self.rawData[plane][f"TAMOM{enc}(1)"]
                My = self.rawData[plane][f"TAMOM{enc}(2)"]
                Mz = self.rawData[plane][f"TAMOM{enc}(3)"]

                Fx_new = Fx * np.cos(AoA) + Fz * np.sin(AoA)
                Fy_new = Fy
                Fz_new = -Fx * np.sin(AoA) + Fz * np.cos(AoA)

                My_new = My
                try:
                    Q = pln.Q
                    S = pln.S
                    MAC = pln.MAC
                    self.Data[plane][f"CL_{name}"] = Fz_new / (Q*S)
                    self.Data[plane][f"CD_{name}"] = Fx_new / (Q*S)
                    self.Data[plane][f"Cm_{name}"] = My_new / (Q*S*MAC)
                except AttributeError:
                    print("Plane doesn't have Q, S or MAC!")
    
    def __str__(self):
        return f"Vehicle Database"
    
    def __enter__(self,obj):
        pass
    
    def __exit__(self):
        pass

