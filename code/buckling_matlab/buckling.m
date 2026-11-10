clc
clear all
close all

%% =============== Theoretical aspects =================

% Il raggio giratore (o raggio d'inerzia), indicato solitamente con $\rho$, 
% $k$ o $r$, è un parametro geometrico che esprime quanto l'area di una sezione
%  è distribuita lontano dal suo asse  baricentrico.

%% ===============Buckling data==================

E = 7.5e6; % Pa
R = 0.003 ; % m
L =0.05 ; % m
u = 0.01; % m
A = pi*R^2; % m^2
I = pi*R^4/4; % m^4
rho = sqrt(I/A); % m
Leff = 2*L; % effective length
lmb = Leff/rho;

Papp = E*A/L*u;

Pcr = pi^2/Leff^2*(E*I);

%% ===============Printing results==================

fprintf('--- Risultati del Calcolo ---\n');
fprintf('Modulo di Young (E):    %.2e Pa\n', E);
fprintf('Area della sezione (A): %.4e m^2\n', A);
fprintf('Momento d''inerzia (I):  %.4e m^4\n', I);
fprintf('Carico Applicato (Papp): %.2f N\n', Papp);
fprintf('Carico Critico (Pcr):    %.2f N\n', Pcr);

if Papp > Pcr
    fprintf('ATTENZIONE: Il carico applicato supera il carico critico (Buckling)!\n');
else
    fprintf('Stato: La struttura è stabile.\n');
end

%% ============ WHAT CAN WE CHANGE ========================

% -   decrease the displacement u
% -   decrease the legnth of the beam L
% -   increase the radius of the beam R

%% =========== TRYING SMALL DISPLACEMENT =================

% we set umax = 1 mm = 0.001 m


E = 7.5e6; % Pa
R = 0.003 ; % m
L =0.05 ; % m
umax = 0.001; % m
A = pi*R^2; % m^2
I = pi*R^4/4; % m^4
rho = sqrt(I/A); % m
Leff = 2*L; % effective length
lmb = Leff/rho;

Papp_u = E*A/L*umax; % N

Pcr_u = pi^2/Leff^2*(E*I); % N

%% ===============Printing results==================

fprintf('\n--- Risultati del Calcolo con u piccolo ---\n');
fprintf('Modulo di Young (E):    %.2e Pa\n', E);
fprintf('Area della sezione (A): %.4e m^2\n', A);
fprintf('Momento d''inerzia (I):  %.4e m^4\n', I);
fprintf('Carico Applicato (Papp): %.2f N\n', Papp_u);
fprintf('Carico Critico (Pcr):    %.2f N\n', Pcr_u);

if Papp_u > Pcr_u
    fprintf('ATTENZIONE: Il carico applicato supera il carico critico (Buckling)!\n');
else
    fprintf('Stato: La struttura è stabile.\n');
end

% we observe a decrease of the max load applied clearly.


%% ============= MODIFY D ======================

% it's better to fix the slenderness

Dvec = 0.006:0.001:L;
Ivec = pi*(Dvec/2).^4/4;

Pcr_vec = pi^2/Leff^2*E*Ivec;


figure()

plot(Dvec, Pcr_vec,'*r')
grid on
xlabel('Diametro (D) [m]');
ylabel('Carico Critico (Pcr) [N]');
title('Carico Critico in funzione del Diametro, L_{eff} = const');
legend('Pcr')

%% ============= MODIFY L ======================

D_try = 2*R;
Lvec = L:-0.001:D_try; % Length decrease from L to diameter
Leff_vec = 2*Lvec;
Pcr_Lvec = pi^2./(Leff_vec).^2 * E * (pi*R^4/4); % Critical load for varying lengths

figure()

plot(Lvec, Pcr_Lvec,'*r')
grid on
xlabel('Lunghezza (L) [m]');
ylabel('Carico Critico (Pcr) [N]');
title('Carico Critico in funzione della lunghezza, D = const');
legend('Pcr')

%% ============== TRY SOME VALUES ==========



% we set umax = 1 mm = 0.001 m


E_new = 7.5e6; % Pa
R_new = 0.005 ; % m
L_new =0.05 ; % m
umax_new = 0.001; % m
A_new = pi*R_new^2; % m^2
I_new = pi*R_new^4/4; % m^4
rho_new = sqrt(I_new/A_new); % m
Leff_new = 2*L_new; % effective length
lmb_new = Leff_new/rho_new;

Papp_u_new = E_new*A_new/L_new*umax_new; % N

Pcr_u_new = pi^2/Leff_new^2*(E_new*I_new); % N

%% ===============Printing results==================

fprintf('\n--- Risultati del Calcolo di tentativo ---\n');
fprintf('Modulo di Young (E):    %.2e Pa\n', E_new);
fprintf('Area della sezione (A): %.4e m^2\n', A_new);
fprintf('Momento d''inerzia (I):  %.4e m^4\n', I_new);
fprintf('Carico Applicato (Papp): %.2f N\n', Papp_u_new);
fprintf('Carico Critico (Pcr):    %.2f N\n', Pcr_u_new);

if Papp_u_new > Pcr_u_new
    fprintf('ATTENZIONE: Il carico applicato supera il carico critico (Buckling)!\n');
else
    fprintf('Stato: La struttura è stabile.\n');
end

%% ============ Max admissable displacement to avoid buckling for a given L ================

% make use of the formula u = L*pi^2/lmb^2

% set L= 0.05 m = 50 mm
lmb_vec = 0:1:100;
u_adm = L*pi^2./lmb_vec.^2;

figure()
semilogy(lmb_vec, u_adm,'ob')
xlabel('Slenderness [1]')
ylabel('Max displacement admissable [m]')
title('maximum displacement to avoid buckling')
legend('max disp')
grid on

%% ============ Max admissable displacement to avoid buckling as function of L ================

lmb_vec = 0:1:100; % vector containing the slenderness
L_TRY_VEC = 0.003:0.01:L;
maxiter = length(L_TRY_VEC);
U_ADM_MAT = [];

% Definiamo una mappa di colori (es. 'jet', 'hsv', 'cool')
colori = jet(maxiter);

figure()
hold on

for i  = 1:maxiter
u_adm = L*pi^2./lmb_vec.^2;
U_ADM_MAT(i, :) = u_adm; % Store the admissible displacements for each length

% plot the results
semilogy(lmb_vec, u_adm,'Color', colori(i,:), 'LineWidth', 1.5, ...
             'DisplayName', sprintf('L = %.3f m', L_TRY_VEC(i)));
end

xlabel('Slenderness \lambda [1]')
ylabel('Max displacement admissible [m]')
title('Maximum displacement to avoid buckling')
grid on
legend('show', 'Location', 'northeastoutside'); % Mostra la legenda fuori dal grafico


%% ========== Useful formulae ==========


% Pcr = pi^2*E*A/lmb^2
%
% lmb = 4*L/R
%
% rho = R/2
%
% u = L*pi^2/lmb^2
