%% PITCH AUTO PILOT - Mc Donnell F-4

% Define the system
[sys_full , sys_sp] = longitudal();
% EIGEN VALUE DECOMPOSITION
[V_sp,Lamda_sp] = eig(sys_sp.A);

%% PLANT
% Pitch Rate to Elevator Transfer Function - Short Period Approximation
[N_q_sp,Den_sp] = ss2tf(sys_sp.A, sys_sp.B, sys_sp.C, sys_sp.D);

N_q_sp = N_q_sp(2:3);   % simplify eqs
Den_sp = Den_sp(1:3);  % simplify eqs
Gplant = tf(N_q_sp,Den_sp);            % Plant Transfer Function

% Short period dynamic properties
om_s = sqrt(Den_sp(3));
zeta_s = Den_sp(2)/(2*om_s);

Kplant = N_q_sp(2);    % Plant Gain
z1 = roots(N_q_sp);    % Plant zeros
poloi = roots(Den_sp); % Plant poles
p1 = poloi(1);   p2 = poloi(2);

%% FEEDBACK
Kq = 1;         % Rate Gyroscope Gain
Grg = tf(Kq,1); % Rate Gyroscope Transfer Function

K8eta = 1;      % Gyroscope Gain
Gg = tf(K8eta,1);  % Gyroscope Transfer Function

%% ACTUATOR
% Elevator hydraulic actuator
Kact = 1;                % Actuator gain
lambda = 5;              % Actuator inverse time constant
N_act = Kact*lambda; % Numerator
D_act = [1 lambda];      % Denominator
Gact= tf(N_act,D_act);   % Actuator Transfer Function

%% CONTROLLER - Pitch Damper
% Desired plant poles
p1des = -1.8+1i*2.4;
p2des = -1.8-1i*2.4;

% Controller Gain
om = 3; % [rad/sec] Frequency for gain calculation
s = 1i*om; % Complex variable

Kol = abs( ((s+lambda)*(s-p1des)*(s-p2des))/(s-z1) );
Kcl = (Kol/(1+Kq*Kol));
Kcontq = -abs( Kol / (Kq*Kplant*Kact*lambda) );
Kcontq =  -0.063; %Kcontq/2;

Gcontq = tf(Kcontq,1);
%% CONTROLLER - Auto-Pilot
Kcont8 = 1.32; %Kerdos tou controller
Kd = 0.5;
Td = Kd/Kcont8;
Gcont8 = Kcont8*tf([Td 1],[0 1]);


[Num_gcont8 , Denum_gcont8] =  tfdata(Gcont8);
Num_gcont8 = cell2mat(Num_gcont8);
Denum_gcont8 = cell2mat(Denum_gcont8);
Denum_gcont8 = Denum_gcont8(2)
%% LOOP TRANSFER FUNCTIONS
integrator = tf([0 1],[1 0]);

G = series(Gcontq,Gact);
G = series(G,Gplant);
Gpd = feedback(G,Grg);
Gap = series(Gcont8,Gpd);
Gap = series(Gap,integrator);
PAP_OL = series(Gap,Gg);
PAP_CL = feedback(Gap,Gg);

%% FLYING & HANDLING QUALITIES
 g = 9.81;
 VTe = 20; ae = 2*pi/180;
 Ue = VTe*cos(ae);
 T82 = -1/z1;
 [polPAP,zerPAP] = pzmap(PAP_CL);
 om_s_new = abs(polPAP(1));

 na = Ue/g/T82;
 CAP = om_s^2/na;
 CAP_new = om_s_new^2/na;

 sim('AutoPilot1',30) 

%% PLOT FIGURES
% Auto-pilot Root Locus
figure()
[polPD,zerPD] = pzmap(Gpd);
area=100;
hold on
pl11 = scatter(real(polPD),imag(polPD),area,'X','LineWidth',3);
hold on
pl12 = scatter(real(zerPD),imag(zerPD),area,'O','LineWidth',3);
hold on
pl13 = scatter(0,0,area,'X','LineWidth',3);
hold on
pl14 = scatter(-1/Td,0,area,'O','LineWidth',3);
hold on
rlocusplot(PAP_CL,0:0.1:50) 
% hold on 
% rlocusplot(PAP_OL,0:0.1:5) 

title('Pitch Auto-pilot - Root Locus','interpreter','latex')

hleg = legend([pl11 pl12 pl13 pl14],'location','best');
hleg.String = {'Pitch Damper Poles','Pitch Damper Zeros','Integrator Pole','Controller Zero'};

set(gca,'FontSize',12);hold on;
    ax = gca;
    ax.TickLabelInterpreter = 'latex';
    grid on
    grid minor

% Pitch attitude response
figure()
plot(xronos,theta_response*180/pi,'LineWidth',2)
hold on
plot(xronos,theta_inp*180/pi,'r--','LineWidth',2)
title('Pitch attitude response','interpreter','latex')
xlabel('t [sec]','interpreter','latex');
ylabel('$\theta$ [deg]','interpreter','latex');

set(gca,'FontSize',12);hold on;
    ax = gca;
    ax.TickLabelInterpreter = 'latex';
%     axis tight
    grid on
    grid minor
    
% Pitch rate response
figure()
plot(xronos,q,'LineWidth',2)
title('Pitch rate response','interpreter','latex')
xlabel('t [sec]','interpreter','latex');
ylabel('q [rad/sec]','interpreter','latex');

set(gca,'FontSize',12);hold on;
    ax = gca;
    ax.TickLabelInterpreter = 'latex';
%     axis tight
    grid on
    grid minor