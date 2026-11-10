%% 1. INIZIALIZZAZIONE E CARICAMENTO MODELLO
clear; close all; clc;
tic;

import com.comsol.model.util.*
model = mphload('TPU_MR_beam_MF_Magnet_E_eq_lin_BH_axial.mph');
model.study('std1').feature('param').active(false);

%% 2. DEFINIZIONE PARAMETRI
id_edge_asse_MRF  = 7;   % <--- SOSTITUISCI CON IL NUMERO DELL'EDGE DELL'ASSE MRF
indice_coord_asse = 2;   % 1 se asse orizzontale (r), 2 se verticale (z)

n_step_B = 20; %20
B_vec    = linspace(0, 1.2, n_step_B);
L_vec    = 20:1:40; %1
n_step_L = length(L_vec);

%% 3. PRE-ALLOCAZIONE DELLA MEMORIA
% Cell array [n_step_B x n_step_L]: ogni cella ha dimensione diversa
% perché il numero di punti mesh cambia al variare di TPU_L
x_coords = cell(n_step_B, n_step_L);  % coordinate spaziali [m], offset a zero
t_vecs   = cell(n_step_B, n_step_L);  % vettore tempi [s]
H_data   = cell(n_step_B, n_step_L);  % [N_punti x N_tempi] in A/m
B_data   = cell(n_step_B, n_step_L);  % [N_punti x N_tempi] in T

% Numero di punti e passi temporali per ogni simulazione (utile dopo)
Npts_mat = NaN(n_step_B, n_step_L);
Nt_mat   = NaN(n_step_B, n_step_L);

% Buffer long format
% Colonne: B_magn | TPU_L | t_index | t_val | x_coord | H_val | B_val
stima_righe  = n_step_B * n_step_L * 101 * 60;
dataset_long = NaN(stima_righe, 7);
indice_riga  = 1;

try
    model.result.dataset.remove('dset_matlab');
catch
end
model.result.dataset.create('dset_matlab', 'Solution');
model.result.dataset('dset_matlab').set('solution', 'sol1');

%% 4. CICLI DI SWEEP
for i = 1:n_step_B
    val_B = B_vec(i);
    model.param.set('B_magn', [num2str(val_B), ' [T]']);

    for j = 1:n_step_L
        val_L = L_vec(j);
        model.param.set('TPU_L', [num2str(val_L), ' [mm]']);

        fprintf('Simulazione -> B: %d/%d (%.2f T) | L: %d/%d (%d mm)\n', ...
            i, n_step_B, val_B, j, n_step_L, val_L);

        model.component('comp1').geom('geom1').run();
        model.component('comp1').mesh('mesh1').run();

        model.study('std1').run();

        %% 5. ESTRAZIONE DEI DATI

        % Vettore tempi
        tempo_vec = mphglobal(model, 't', 'dataset', 'dset_matlab');
        tempo_vec = tempo_vec(:);
        step_effettivi = length(tempo_vec);

        % Distribuzione H lungo il boundary
        dati_H = mpheval(model, 'mfnc.normH', ...
                    'edim', 'boundary', 'selection', id_edge_asse_MRF, ...
                    'dataset', 'dset_matlab');

        % Distribuzione B lungo lo stesso boundary
        dati_B = mpheval(model, 'mfnc.normB', ...
                    'edim', 'boundary', 'selection', id_edge_asse_MRF, ...
                    'dataset', 'dset_matlab');

        % Coordinate spaziali (stessa mesh per H e B)
        x_dati_raw = dati_H.p(indice_coord_asse, :)';
        x_dati     = x_dati_raw - min(x_dati_raw);  % offset a zero

        % Ordina per coordinata crescente (la mesh non e' ordinata)
        [x_sorted, sort_idx] = sort(x_dati);

        % Matrici [N_punti x N_tempi] ordinate
        H_mat = dati_H.d1(:, sort_idx)';
        B_mat = dati_B.d1(:, sort_idx)';
        N_pts = length(x_sorted);

        fprintf('   N_punti mesh: %d | N_tempi: %d\n', N_pts, step_effettivi);

        %% 6. SALVATAGGIO CELL ARRAY
        x_coords{i,j} = x_sorted;
        t_vecs{i,j}   = tempo_vec;
        H_data{i,j}   = H_mat;
        B_data{i,j}   = B_mat;
        Npts_mat(i,j) = N_pts;
        Nt_mat(i,j)   = step_effettivi;

        %% 7. SALVATAGGIO LONG FORMAT
        n_righe_sim = N_pts * step_effettivi;

        if (indice_riga + n_righe_sim - 1) > size(dataset_long, 1)
            dataset_long = [dataset_long; NaN(stima_righe, 7)];
        end

        % Costruzione vettori replicati senza loop sul tempo
        x_rep     = repmat(x_sorted,    step_effettivi, 1);  % [N_pts*N_t x 1]
        t_rep     = repelem(tempo_vec,  N_pts);               % [N_pts*N_t x 1]
        t_idx_rep = repelem((1:step_effettivi)', N_pts);      % [N_pts*N_t x 1]
        H_vec     = H_mat(:);    % column-major: tutti i punti a t=1, poi t=2...
        B_vec_sim = B_mat(:);

        indici_da_riempire = indice_riga : (indice_riga + n_righe_sim - 1);

        dataset_long(indici_da_riempire, 1) = val_B;
        dataset_long(indici_da_riempire, 2) = val_L;
        dataset_long(indici_da_riempire, 3) = t_idx_rep;
        dataset_long(indici_da_riempire, 4) = t_rep;
        dataset_long(indici_da_riempire, 5) = x_rep;
        dataset_long(indici_da_riempire, 6) = H_vec;
        dataset_long(indici_da_riempire, 7) = B_vec_sim;

        indice_riga = indice_riga + n_righe_sim;

        %% 8. PULIZIA RAM E DISCO
        model.sol('sol1').clearSolutionData();
        ModelUtil.clearCache();
    end
end

%% 9. CHIUSURA E SALVATAGGIO
dataset_long = dataset_long(1:indice_riga-1, :);

fprintf('\nRighe totali long format: %d\n', size(dataset_long, 1));

% .mat con cell array (per MATLAB)
save('Risultati_BH_distribution.mat', ...
    'B_vec', 'L_vec', ...
    'x_coords', 't_vecs', 'H_data', 'B_data', ...
    'Npts_mat', 'Nt_mat', '-v7.3');

% .txt long format (per Python/ML)
fid = fopen('Risultati_BH_distribution.txt', 'w');
fprintf(fid, 'B_magn_T\tTPU_L_mm\tt_index\tt_val_s\tx_coord_m\tH_val_Am\tB_val_T\n');
fclose(fid);
writematrix(dataset_long, 'Risultati_BH_distribution.txt', ...
    'Delimiter', '\t', 'WriteMode', 'append');

time = toc;
disp('=== FINISH ===');
fprintf('Total time = %f sec\n', time);