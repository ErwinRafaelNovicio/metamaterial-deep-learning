clear; 
clc;

% --- PARAMETRI ---
y = 0.1;    % Rapporto Ucr/L desiderato (10%)
K = 0.5;    % Fattore di vincolo (0.5 = incastro-scorrevole)
E = 7.5e6;  % Modulo di Young (Pa) 

% --- 1. SCEGLI IL RAGGIO O LA LUNGHEZZA ---
% Poiché ci sono infinite soluzioni, devi fissare un parametro. 
% Scegliamo un Raggio di progetto (es. 3 mm):
R = 0.004;  % [m]

% --- 2. CALCOLO ESATTO DI L ---
% Usiamo la formula matematica invertita
L = R * (pi / (2 * K * sqrt(y)));

% --- 3. VERIFICA ---
% Ora ricalcoliamo tutto in avanti per provare che la teoria funziona
A = pi * R^2; 
I = pi * R^4 / 4; 

Pcr = (pi^2 * E * I) / (K * L)^2;
Ucr = (Pcr * L) / (E * A);
Ucr_percentuale = (Ucr / L) * 100;

% --- STAMPA RISULTATI ---
fprintf('\n--- DIMENSIONAMENTO ESATTO ---\n');
fprintf('Se fissiamo un Raggio (R) di: %.5f m (%.2f mm)\n', R, R*1000);
fprintf('La Lunghezza esatta (L) deve essere: %.5f m (%.2f mm)\n', L, L*1000);
fprintf('Rapporto di snellezza (L/R): %.4f\n\n', L/R);

fprintf('--- VERIFICA DEL BUCKLING ---\n');
fprintf('Carico Critico calcolato (Pcr): %.2f N\n', Pcr);
fprintf('Spostamento Critico calcolato (Ucr): %.5f m\n', Ucr);
fprintf('Deformazione al Buckling (Ucr / L): %.2f %%\n', Ucr_percentuale);

if abs(Ucr_percentuale - (y*100)) < 1e-6
    fprintf('\nOK: La trave va in instabilità esattamente al %.1f%% della sua lunghezza!\n', y*100);
end
