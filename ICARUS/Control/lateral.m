%% LATERAL
function sysfull = lateral()
%%% FULL Longitudal Matrices
%matrices are for delta control = 0 around trim position
A_lateral=      [-0.115565          -0.0210722            -14.6598                9.81
                  0.334433            -29.0929             10.1297                   0
                   0.62446            -2.09237           -0.470998                   0
                         0                   1                   0                   0];
B_lateral=     [6.184988e-14
                   1.579904e-09
                  -3.218844e-08
                              0];
C_lateral= [0 0 1 0];
D_lateral = 0;

%%% System Definition
System_lateral = ss(A_lateral,B_lateral,C_lateral,D_lateral);

%%% EIGEN VALUE DECOMPOSITION
Lamda_La = eig(System_lateral);
sysfull = System_lateral;
end
