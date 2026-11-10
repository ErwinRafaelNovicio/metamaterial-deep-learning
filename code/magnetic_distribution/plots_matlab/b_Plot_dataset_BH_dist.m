%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT DATASET - DISTRIBUZIONI B(x) e H(x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;

%% 1. CARICAMENTO DEI DATI
load('Risultati_BH_distribution.mat');

n_step_B = length(B_vec);
n_step_L = length(L_vec);

%% 2. PARAMETRO TEMPORALE DA PLOTTARE
t_plot = 1;  % indice del passo temporale da visualizzare

%% 3. COLORI
mappa_rgb = turbo(n_step_L);
hex_colors = cell(n_step_L, 1);
for j = 1:n_step_L
    r = round(mappa_rgb(j, 1) * 255);
    g = round(mappa_rgb(j, 2) * 255);
    b = round(mappa_rgb(j, 3) * 255);
    hex_colors{j} = sprintf('#%02X%02X%02X', r, g, b);
end

%% 4. FIGURE PER H(x)
for i = 1:n_step_B
    val_B = B_vec(i);

    if val_B == 0
        continue;
    end

    figure('Color', 'w', 'Name', sprintf('H distribution - B = %.2f T', val_B));
    hold on; grid on; box on;
    h_plots = gobjects(n_step_L, 1);
    labels  = strings(n_step_L, 1);

    for j = 1:n_step_L
        val_L = L_vec(j);

        x_plot = x_coords{i, j};              % [N_punti x 1] in metri
        y_plot = H_data{i, j}(:, t_plot);     % [N_punti x 1] ad istante t_plot

        h_plots(j) = plot(x_plot, y_plot, ...
            'Color',     hex_colors{j}, ...
            'LineWidth', 1.5);
        labels(j) = sprintf('L = %.1f mm', val_L);
    end

    title(sprintf('Magnetic Field Norm H(x)  (B_{magn} = %.2f T,  t_{idx} = %d)', val_B, t_plot), ...
        'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Distance along axis [m]', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('normH [A/m]',             'FontSize', 12, 'FontWeight', 'bold');
    legend(h_plots, labels, 'Location', 'eastoutside', 'FontSize', 10);
    hold off;
end

%% 5. FIGURE PER B(x)
for i = 1:n_step_B
    val_B = B_vec(i);

    if val_B == 0
        continue;
    end

    figure('Color', 'w', 'Name', sprintf('B distribution - B = %.2f T', val_B));
    hold on; grid on; box on;
    h_plots = gobjects(n_step_L, 1);
    labels  = strings(n_step_L, 1);

    for j = 1:n_step_L
        val_L = L_vec(j);

        x_plot = x_coords{i, j};              % [N_punti x 1] in metri
        y_plot = B_data{i, j}(:, t_plot);     % [N_punti x 1] ad istante t_plot

        h_plots(j) = plot(x_plot, y_plot, ...
            'Color',     hex_colors{j}, ...
            'LineWidth', 1.5);
        labels(j) = sprintf('L = %.1f mm', val_L);
    end

    title(sprintf('Magnetic Flux Density B(x)  (B_{magn} = %.2f T,  t_{idx} = %d)', val_B, t_plot), ...
        'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Distance along axis [m]', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('normB [T]',               'FontSize', 12, 'FontWeight', 'bold');
    legend(h_plots, labels, 'Location', 'eastoutside', 'FontSize', 10);
    hold off;
end