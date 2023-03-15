%% Longitudal
function [sysfull,sysapprox] = longitudal()
%%% FULL Longitudal Matrices
%matrices are for delta control = 0 around trim position
A_longitudal=    [ -0.0602993            0.430874                   0               -9.81
                     -1.32785           -4.29191              14.369                   0
                  -0.00102582           -1.00827            -1.58981                   0
                            0                  0                   1                   0];
B_longitudal=     [-0.1335799
                   -4.614464
                   -42.52232
                    0];
C_longitudal= [0 0 1 0];
D_longitudal = 0;

%%% System Definition
System_longitudal = ss(A_longitudal,B_longitudal,C_longitudal,D_longitudal);


%% Short Period APPROXIMATION
A_sp = A_longitudal(2:3,2:3);
B_sp = B_longitudal(2:3);
C_sp = [0 1 ];
D_sp = 0;

%%% System Decomposition 
System_sp = ss(A_sp,B_sp,C_sp,D_sp);

sysapprox = System_sp;
sysfull = System_longitudal;
end

