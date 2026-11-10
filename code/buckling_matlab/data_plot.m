clc
clear all
close all

load("MIO_dataset_PROVA.txt")

data = MIO_dataset;

B_values = [0 , 0.5, 1, 1.5, 2];

B_matrix = cell(length(B_values), 1);

for i = 1:length(B_values)
    % Seleziona il valore corrente (es. al primo giro sarà 0, poi 1, ecc.)
    val = B_values(i);
    
    % Estrai il blocco e salvalo nella cella i-esima usando le parentesi graffe {}
    B_matrix{i} = data(data(:, 1) == val, :);
end

figure();
for i= 1 : length(B_values)
    
    mat = B_matrix{i};
    plot(mat(:,4), mat(:,3));
    xlabel('Strain')
    ylabel('Stress')
    grid on
    hold on
end





