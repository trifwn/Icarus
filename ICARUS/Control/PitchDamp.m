%% PITCH AUTO PILOT - Hermes V model airplane
%%% The present script is executed before the associated Simulink model ...
%%% ... (Pitch_Damper.mdl) %%%

% Define the system
[sys_full , sys_sp] = longitudal();


% EIGEN VALUE DECOMPOSITION
[V_sp,Lamda_sp] = eig(sys_sp.A);

%% PLANT
% Pitch Rate to Elevator Transfer Function - Short Period Approximation
[N_q_sp,Den_sp] = ss2tf(sys_sp.A, sys_sp.B, sys_sp.C, sys_sp.D);
Gplant = tf(N_q_sp,Den_sp);            % Plant Transfer Function

% Short period dynamic properties
om_s = sqrt(Den_sp(3)); zeta_s = Den_sp(2)/(2*om_s);

Kplant = N_q_sp(1);    % Plant Gain
z1 = roots(N_q_sp);    % Plant zeros
poloi = roots(Den_sp); % Plant poles
p1 = poloi(1);   p2 = poloi(2);

%% FEEDBACK
Kq = 1;            % Rate Gyroscope Gain
Grg = tf(Kq,1); % Rate Gyroscope Transfer Function
%% ACTUATOR
% Elevator hydraulic actuator
Kact = 1;                % Actuator gain
lambda = 5;              % Actuator inverse time constant
N_act = Kact*lambda;     % Numerator
D_act = [1 lambda];      % Denominator
Gact= tf(N_act,D_act);   % Actuator Transfer Function

%% CONTROLLER
% Desired plant poles
p1des = -1.8+1i*2.4;
p2des = -1.8-1i*2.4;

% Controller Gain
om = 3; % [rad/sec] Frequency for gain calculation
s = 1i*om; % Complex variable

% Kol = abs( ((s+lambda)*(s-p1des)*(s-p2des))/(s-z1) );
Kol = abs( ((s+lambda)*(s-p1des)*(s-p2des))/(s-z1) );
Kcl = (Kol/(1+Kq*Kol));
Kcont = -abs( Kol / (Kq*Kplant*Kact*lambda) );
Kcont = -0.063;

Gcont=tf(Kcont,1);

% Loop Transfer Functions
G = series(Gcont,Gact);
G = series(G,Gplant);  % Forward Loop Transfer Function
Gol = series(G,Grg);   % Open Loop Transfer Function
Gcl = feedback(G,Grg); % Closed Loop Transfer Function
sim('Pitch_Damper',40)

%% PLOT FIGURES
%% Root locus for positive/negative gain
figure()

subplot(1,2,1)
rlocus(Gol)

title('Root locus for $K_{cont}>0$','interpreter','latex')
%axis([-5 1 -3 3])
set(gca,'FontSize',12);hold on;
    ax = gca;
    ax.TickLabelInterpreter = 'latex';
    grid on
    grid minor

subplot(1,2,2)
rlocus(-Gol)

title('Root locus for $K_{cont}<0$','interpreter','latex')
%axis([-5 1 -3 3])
set(gca,'FontSize',12);hold on;
    ax = gca;
    ax.TickLabelInterpreter = 'latex';
    grid on
    grid minor

%% Pitch Damper root locus
figure()
%pzmap(Gplant);
area=100;
hold on
pl11 = scatter(real(poloi),imag(poloi),area,'X','LineWidth',3);
pl12 = scatter(real(z1),imag(z1),area,'O','LineWidth',3);
pl13 = scatter(-lambda,0,area,'X','LineWidth',3);

 hold on
rlocus(Gcl)

title('Pitch Damper Root Locus','interpreter','latex')

hleg = legend([pl11 pl12 pl13],'location','best');
hleg.String = {'Plant Poles','Plant Zeros','Actuator Pole'};

set(gca,'FontSize',12);hold on;
    ax = gca;
    ax.TickLabelInterpreter = 'latex';
    grid on
    grid minor

%% Pitch rate response
 figure()
 plot(xronos,q_response,'LineWidth',2)
 hold on
 plot(xronos,qcomm,'r--','LineWidth',2)

 title('\textbf{Pitch rate response}','interpreter','latex')
 xlabel('t [sec]','interpreter','latex');
 ylabel('q $[\frac{rad}{sec}]$','interpreter','latex');
%  legend({'$q_{response}$','$q_{comm}$'},'interpreter','latex');

 set(gca,'FontSize',12);hold on;
     ax = gca;
     ax.TickLabelInterpreter = 'latex';
     axis tight
     grid on
     grid minor

% STEP INPUT
R = tf([0,1],[1,0]);
Q = Gcl * R;

[ num , den ] = tfdata(Q);                                   % Transfer Function Object
syms s t                                                     % Invoke Symbolic Math Toolbox
snum = poly2sym(num, s);                                     % Symbolic Numerator Polynomial
sden = poly2sym(den, s);                                     % Symbolic Denominator Polynomial
FT_time_domain = ilaplace(snum/sden);                        % Inverse Laplace Transform

FT_time_domain = collect(FT_time_domain, exp(-t));
line = 1;

figure(3)
fplot(line,[0,2])
hold on
fplot(FT_time_domain,[0,2])
